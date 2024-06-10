from typing import *
import dataclasses
import itertools
import json
import os
import shutil

import torch
import torch.distributed as dist
import transformers

from reallm.api.core.config import ModelFamily, ModelName, ModelShardID
from reallm.api.core.model_api import HF_MODEL_FAMILY_REGISTRY, ReaLModelConfig
from reallm.base import constants, logging
from reallm.base.testing import clear_name_resolve, init_global_constants, LocalMultiProcessTest

if TYPE_CHECKING:
    from reallm.impl.model.backend.inference import PipelinableInferenceEngine, PipelineInferenceBackend
    from reallm.impl.model.backend.megatron import MegatronTrainBackend, ReaLMegatronEngine
    from reallm.impl.model.nn.real_llm_api import ReaLModel

logger = logging.getLogger("tests.test_saveload")


def _shrink_mconfig(mconfig: ReaLModelConfig):
    mconfig.hidden_dim = 128
    mconfig.head_dim = 8
    mconfig.n_kv_heads = 1
    mconfig.intermediate_dim = 256
    mconfig.use_attention_bias = False
    return mconfig


def create_model(model_family_name, model_name, hf_path, is_critic, max_pp, instantiate=True) -> "ReaLModel":
    # NOTE: import here to avoid initializing CUDA context in the main process
    from reallm.impl.model.nn.real_llm_api import add_helper_functions, ReaLModel

    with constants.model_scope(model_name):
        tokenizer = transformers.AutoTokenizer.from_pretrained(hf_path)
        hf_config = transformers.AutoConfig.from_pretrained(hf_path)
        mconfig: ReaLModelConfig = getattr(ReaLModel, f"config_from_{model_family_name}")(hf_config)
        mconfig = _shrink_mconfig(mconfig)
        mconfig.n_layers = max_pp
        mconfig.is_critic = is_critic

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
        if constants.pipe_parallel_world_size() == 1:
            add_helper_functions(model)
    return model


def get_topo(model_name):
    with constants.model_scope(model_name):
        return constants.grid().topology()


def build_engine(module, model_name) -> "ReaLMegatronEngine":

    from reallm.api.core import model_api
    from reallm.impl.model.backend.megatron import MegatronTrainBackend, ReaLMegatronEngine

    with constants.model_scope(model_name):
        backend = MegatronTrainBackend(initial_loss_scale=8.0)
        _model = backend.initialize(
            model_api.Model(
                None,
                module,
                None,
                module.device,
                module.dtype,
            ),
            model_api.FinetuneSpec(
                total_train_epochs=1,
                total_train_steps=20,
                steps_per_epoch=20,
            ),
        )
    return _model.module


def build_inference_engine(module, model_name) -> "PipelinableInferenceEngine":
    from reallm.impl.model.backend.inference import PipelinableInferenceEngine

    with constants.model_scope(model_name):
        return PipelinableInferenceEngine(module)


def setup_constants_and_param_realloc(
    from_model_name,
    to_model_name,
    from_pp_dp_mp,
    to_pp_dp_mp,
):
    from reallm.impl.model.comm.param_realloc import set_trainable, setup_param_realloc

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


def make_random_input(vocab_size):
    seqlens_cpu = [int(torch.randint(20, 128, (1,)).item()) for _ in range(16)]
    packed_input_ids = torch.randint(0, vocab_size, (sum(seqlens_cpu),), dtype=torch.long, device="cuda")
    input_lens = torch.tensor(seqlens_cpu, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0))
    prompt_mask = [[1] * 10 + [0] * (seqlen - 10) for seqlen in seqlens_cpu]
    prompt_mask = torch.tensor(sum(prompt_mask, []), dtype=torch.bool, device="cuda")
    return dict(
        seqlens_cpu=seqlens_cpu,
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens,
        prompt_mask=prompt_mask,
    )


def _test_para_realloc(
    model_family_name: str,
    is_critic: bool,
    init_critic_from_actor: bool,
    from_pp_dp_mp: Tuple,
    to_pp_dp_mp: Tuple,
    tokenizer_path: str,
):
    # os.environ["REAL_SAVE_MAX_SHARD_SIZE_BYTE"] = str(int(1e6))

    from_model_name = ModelName("param_realloc_test", 0)
    to_model_name = ModelName("param_realloc_test", 1)

    pg_info = setup_constants_and_param_realloc(from_model_name, to_model_name, from_pp_dp_mp, to_pp_dp_mp)
    save_path = setup_save_path()

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    max_pp = max(from_pp_dp_mp[0], to_pp_dp_mp[0])
    # Create model 1
    from_model = create_model(
        model_family_name,
        from_model_name,
        tokenizer_path,
        is_critic,
        max_pp,
        instantiate=True,
    )

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
        max_pp,
        instantiate=True,
    )
    to_model.eval()

    # Create redistributer.
    redist = ParamRedistributer(
        from_model_name,
        to_model_name,
        from_model,
        to_model,
        pg_info,
    )

    engine = build_engine(from_model, from_model_name)
    to_engine = build_inference_engine(to_model, to_model_name)
    redist.forward()

    genp1 = to_model.contiguous_param.clone()

    test_x = make_random_input(to_model.config.vocab_size)
    with constants.model_scope(to_model_name):
        to_model.eval()
        with torch.no_grad():
            test_out = to_engine.forward(
                seqlens_cpu=test_x["seqlens_cpu"],
                packed_input_ids=test_x["packed_input_ids"],
                cu_seqlens=test_x["cu_seqlens"],
            )

    redist.backward()

    from reallm.impl.model.backend.megatron import ReaLMegatronEngine
    from reallm.impl.model.interface.sft_interface import compute_packed_sft_loss

    with constants.model_scope(from_model_name):
        max_trials = 20

        for i in range(max_trials):
            x = make_random_input(from_model.config.vocab_size)
            seqlens_cpu = x["seqlens_cpu"]
            packed_input_ids = x["packed_input_ids"]
            cu_seqlens = x["cu_seqlens"]
            prompt_mask = x["prompt_mask"]
            input_lens = cu_seqlens[1:] - cu_seqlens[:-1]

            loss_fn_kwargs = dict(
                prompt_mask=prompt_mask,
                input_lens=input_lens,
            )
            engine.eval()

            p2 = engine.engine.ddp.module.contiguous_param.clone().detach()

            engine: ReaLMegatronEngine
            stats = engine.train_batch(
                seqlens_cpu=seqlens_cpu,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                loss_fn=compute_packed_sft_loss,
                num_micro_batches=constants.pipe_parallel_world_size() * 2,
                version_steps=i,
                **loss_fn_kwargs,
            )

            p3 = engine.engine.ddp.module.contiguous_param.clone().detach()

            all_p3 = [torch.zeros_like(p3) for _ in range(constants.data_parallel_world_size())]
            dist.all_gather(all_p3, p3, group=constants.data_parallel_group())

            try:
                # assert torch.allclose(p1, p2)
                assert not torch.allclose(p3, p2)
                delta = (p3 - p2).float()
                delta_reduce = delta.clone()
                dist.all_reduce(delta_reduce, group=constants.data_parallel_group())
                delta_reduce /= dist.get_world_size(group=constants.data_parallel_group())
                assert torch.allclose(delta, delta_reduce, atol=2e-4), ((delta - delta_reduce).abs().max())
                for i, p3i in enumerate(all_p3):
                    assert torch.allclose(p3i, p3), (
                        i,
                        constants.data_parallel_rank(),
                        p3i.flatten()[:100],
                        p3.flatten()[:100],
                        (p3i - p3).abs().max(),
                    )

            except AssertionError as e:
                if i < max_trials - 1:
                    continue
                else:
                    raise e

    redist.forward()
    with constants.model_scope(to_model_name):
        to_model.eval()
        with torch.no_grad():
            test_out2 = to_engine.forward(
                seqlens_cpu=test_x["seqlens_cpu"],
                packed_input_ids=test_x["packed_input_ids"],
                cu_seqlens=test_x["cu_seqlens"],
            )
    if test_out is not None:
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


def decompose_to_three_factors(n: int):
    factors = []
    for i in range(1, int(n**(1 / 2)) + 1):
        if n % i == 0:
            for j in range(i, int((n // i)**(1 / 2)) + 1):
                if (n // i) % j == 0:
                    k = (n // i) // j
                    factors += list(set(itertools.permutations([i, j, k])))
    return factors


if __name__ == "__main__":
    # factors = decompose_to_three_factors(8)
    # logfile = "param_realloc_test.log"
    # if os.path.exists(logfile):
    #     with open(logfile, "r") as f:
    #         tested = f.readlines()
    # else:
    #     tested = []
    # for i, (from_pp_dp_mp, to_pp_dp_mp) in enumerate(itertools.product(factors, factors)):
    #     if from_pp_dp_mp == (1, 1, 8) and to_pp_dp_mp == (1, 8, 1):
    #         # This case will always overflow
    #         continue
    #     if to_pp_dp_mp == (8, 1, 1):
    #         # This case will always overflow
    #         continue
    #     print(">" * 10 + f" testing with from_pp_dp_mp={from_pp_dp_mp}, to_pp_dp_mp={to_pp_dp_mp} " +
    #           "<" * 10)
    #     for model_family_name, path in [("llama", "/lustre/public/pretrained_model_weights/Llama-2-7b-hf/")]:
    #         test_param_realloc(model_family_name, False, False, from_pp_dp_mp, to_pp_dp_mp, path)
    #     with open(logfile, "a") as f:
    #         f.write(f"{from_pp_dp_mp}, {to_pp_dp_mp}\n")
    for i, (from_pp_dp_mp, to_pp_dp_mp) in enumerate([[(2, 4, 1), (4, 1, 2)]]):
        print(">" * 10 + f" testing with from_pp_dp_mp={from_pp_dp_mp}, to_pp_dp_mp={to_pp_dp_mp} " +
              "<" * 10)
        for model_family_name, path in [("llama", "/lustre/public/pretrained_model_weights/Llama-2-7b-hf/")]:
            test_param_realloc(model_family_name, False, False, from_pp_dp_mp, to_pp_dp_mp, path)
