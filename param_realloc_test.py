from typing import *
import os
import shutil

import torch
import torch.distributed as dist
import transformers
import dataclasses

from reallm.api.core.config import ModelFamily, ModelName, ModelShardID
from reallm.api.core.model_api import HF_MODEL_FAMILY_REGISTRY, ReaLModelConfig
from reallm.base import constants, logging
from tests.utils import clear_name_resolve, init_global_constants, LocalMultiProcessTest

if TYPE_CHECKING:
    from reallm.impl.model.nn.real_llm_api import ReaLModel

logger = logging.getLogger("tests.test_saveload")


def _shrink_mconfig(mconfig: ReaLModelConfig):
    mconfig.hidden_dim = 128
    mconfig.head_dim = 8
    mconfig.n_kv_heads = 1
    mconfig.intermediate_dim = 256
    mconfig.n_layers = 2
    mconfig.use_attention_bias = False
    return mconfig


def create_model(model_family_name, model_name, hf_path, is_critic, instantiate=True) -> "ReaLModel":
    # NOTE: import here to avoid initializing CUDA context in the main process
    from reallm.impl.model.nn.real_llm_api import ReaLModel

    with constants.model_scope(model_name):
        tokenizer = transformers.AutoTokenizer.from_pretrained(hf_path)
        hf_config = transformers.AutoConfig.from_pretrained(hf_path)
        mconfig: ReaLModelConfig = getattr(ReaLModel, f"config_from_{model_family_name}")(hf_config)
        mconfig = _shrink_mconfig(mconfig)
        mconfig.is_critic = is_critic
        mconfig.use_contiguous_param = True

        # initialize model
        model = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
        model.eval()
        # model._instantiation_hooks.append(
        #     lambda: getattr(model, f"from_{model_family_name}")(hf_path, False)
        # )
        if instantiate:
            model.instantiate()
            for p in model.parameters():
                dist.all_reduce(p.data, group=constants.data_parallel_group())
                p.data /= dist.get_world_size(group=constants.data_parallel_group())
    return model


def get_topo(model_name):
    with constants.model_scope(model_name):
        return constants.grid().topology()


def build_engine(module, model_name):

    from reallm.api.core import model_api
    from reallm.impl.model.backend.megatron import MegatronTrainBackend

    with constants.model_scope(model_name):
        backend = MegatronTrainBackend()
        _model = backend.initialize(
            model_api.Model(
                None,
                module,
                None,
                module.device,
                module.dtype,
            ),
            None,
        )

    return _model.module


def setup_constants_and_param_realloc(
    from_model_name,
    to_model_name,
    from_pp_dp_mp,
    to_pp_dp_mp,
):
    from reallm.impl.model.comm.param_realloc import setup_param_realloc, set_trainable

    num_pp, num_dp, num_mp = from_pp_dp_mp
    assert num_pp * num_dp * num_mp == 8
    init_global_constants(
        num_dp=num_dp,
        num_mp=num_mp,
        num_pp=num_pp,
        model_name=from_model_name,
        sequence_parallel=False,
    )

    num_pp, num_dp, num_mp = to_pp_dp_mp
    assert num_pp * num_dp * num_mp == 8
    init_global_constants(
        num_dp=num_dp,
        num_mp=num_mp,
        num_pp=num_pp,
        model_name=to_model_name,
        sequence_parallel=False,
    )

    assert dist.get_world_size() == 8, dist.get_world_size()

    from_topo = get_topo(from_model_name)
    to_topo = get_topo(to_model_name)
    model_topos = {from_model_name: from_topo, to_model_name: to_topo}

    msid2mwid = {}
    for i in range(8):
        for _model_name in [from_model_name, to_model_name]:
            coord = model_topos[_model_name].get_coord(i)
            k = ModelShardID(
                _model_name,
                dp_rank=coord.data,
                mp_rank=coord.model,
                pp_rank=coord.pipe,
                topo=model_topos[_model_name],
            )
            msid2mwid[k] = i

    pg_info = setup_param_realloc(
        model_topos=model_topos,
        msid2mwid=msid2mwid,
        param_realloc_pairs=[
            (from_model_name, to_model_name),
            (to_model_name, from_model_name),
        ],
    )
    set_trainable(from_model_name, True)
    set_trainable(to_model_name, False)
    return pg_info


def setup_save_path():
    save_path = "/tmp/ReaL-param_realloc_test"
    if dist.get_rank() == 0:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
    dist.barrier()
    return save_path


@dataclasses.dataclass
class ParamRedistributer:
    from_model_name: ModelName
    to_model_name: ModelName
    from_model: Any
    to_model: Any
    pg_info: Any

    def _redist(self, m1, m2, n1, n2):
        with constants.model_scope(n1):
            t1 = constants.grid().topology()
        with constants.model_scope(n2):
            t2 = constants.grid().topology()
        a, b, c = m1.build_reparallelized_layers_async(
            from_model_name=n1,
            to_model_name=n2,
            from_topo=t1,
            to_topo=t2,
            to_model_config=m2.config,
            pg_info=self.pg_info,
        )
        m2.patch_reparallelization((a, b))

        assert m1.layers is None
        assert m1.contiguous_param is None
        assert m2.layers is not None
        assert m2.contiguous_param is not None

    def forward(self):
        self._redist(self.from_model, self.to_model, self.from_model_name, self.to_model_name)

    def backward(self):
        self._redist(self.to_model, self.from_model, self.to_model_name, self.from_model_name)


def _test_para_realloc(
    model_family_name: str,
    is_critic: bool,
    init_critic_from_actor: bool,
    from_pp_dp_mp: Tuple,
    to_pp_dp_mp: Tuple,
    tokenizer_path: str,
):

    from reallm.impl.model.nn.real_llm_api import add_helper_functions

    # os.environ["REAL_SAVE_MAX_SHARD_SIZE_BYTE"] = str(int(1e6))

    from_model_name = ModelName("param_realloc_test", 0)
    to_model_name = ModelName("param_realloc_test", 1)

    pg_info = setup_constants_and_param_realloc(from_model_name, to_model_name, from_pp_dp_mp, to_pp_dp_mp)
    save_path = setup_save_path()

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    # Create model 1
    from_model = create_model(
        model_family_name,
        from_model_name,
        tokenizer_path,
        is_critic,
        instantiate=True,
    )
    add_helper_functions(from_model)

    # save parameters of from_model
    # with constants.model_scope(from_model_name):
    #     getattr(from_model, f"to_{model_family_name}")(tokenizer, save_path)
    # dist.barrier()

    # Creat model 2
    to_model = create_model(
        model_family_name,
        to_model_name,
        tokenizer_path,
        is_critic,
        instantiate=True,
    )
    to_model.eval()
    add_helper_functions(to_model)

    # Create redistributer.
    redist = ParamRedistributer(
        from_model_name,
        to_model_name,
        from_model,
        to_model,
        pg_info,
    )

    engine = build_engine(from_model, from_model_name)
    redist.forward()

    genp1 = to_model.contiguous_param.clone()
    test_input_ids = torch.randint(0, to_model.config.vocab_size, (10, 128), dtype=torch.long, device="cuda")
    with constants.model_scope(to_model_name):
        to_model.eval()
        with torch.no_grad():
            test_out = to_model(test_input_ids).logits
            test_out_ = to_model(test_input_ids).logits
        assert torch.allclose(test_out, test_out_)

    redist.backward()

    with constants.model_scope(from_model_name):
        max_trials = 10

        for i in range(max_trials):
            input_ids = torch.randint(
                0,
                from_model.config.vocab_size,
                (10, 128),
                dtype=torch.long,
                device="cuda",
            )
            engine.eval()
            engine.zero_grad()

            x: torch.Tensor = engine(input_ids).logits
            loss = -torch.nn.functional.log_softmax(x, -1)[..., 0].sum()

            engine.backward(loss)

            p1 = engine.megatron_module.buffers[0].param_data.clone().detach()

            p2 = engine.megatron_module.buffers[0].param_data.clone().detach()
            update_successful, grad_norm, num_zeros_in_grad = engine.step()
            p3 = engine.megatron_module.buffers[0].param_data.clone().detach()

            try:
                assert torch.allclose(p1, p2)
                assert not torch.allclose(p3, p2)
                delta = (p3 - p2).float()
                delta_reduce = delta.clone()
                dist.all_reduce(delta_reduce, group=constants.data_parallel_group())
                delta_reduce /= dist.get_world_size(group=constants.data_parallel_group())
                assert torch.allclose(delta, delta_reduce, atol=1e-4), (delta - delta_reduce).abs().max()

            except AssertionError as e:
                if i < max_trials - 1:
                    continue
                else:
                    raise e

    redist.forward()
    with constants.model_scope(to_model_name):
        to_model.eval()
        with torch.no_grad():
            test_out2 = to_model(test_input_ids).logits
    assert not torch.allclose(test_out, test_out2)
    assert not torch.allclose(genp1, to_model.contiguous_param)


def test_param_realloc(
    model_family_name: str,
    is_critic: bool,
    init_critic_from_actor: bool,
    from_pp_dp_mp: Tuple,
    to_pp_dp_mp: Tuple,
    tokenizer_path: str,
):
    expr_name = "param_realloc_test"
    trial_name = "test"
    clear_name_resolve(expr_name=expr_name, trial_name=trial_name)
    test_impl = LocalMultiProcessTest(
        world_size=8,
        func=_test_para_realloc,
        expr_name=expr_name,
        trial_name=trial_name,
        model_family_name=model_family_name,
        is_critic=is_critic,
        init_critic_from_actor=init_critic_from_actor,
        from_pp_dp_mp=from_pp_dp_mp,
        to_pp_dp_mp=to_pp_dp_mp,
        tokenizer_path=tokenizer_path,
    )
    test_impl.launch()
    print("success")


if __name__ == "__main__":
    for i, (from_pp_dp_mp, to_pp_dp_mp) in enumerate([((1, 4, 2), (1, 2, 4))]):
        print(
            ">" * 10 + f" testing with from_pp_dp_mp={from_pp_dp_mp}, to_pp_dp_mp={to_pp_dp_mp} " + "<" * 10
        )
        for model_family_name, path in [("llama", "/lustre/public/pretrained_model_weights/Llama-2-7b-hf/")]:
            test_param_realloc(model_family_name, False, False, from_pp_dp_mp, to_pp_dp_mp, path)
