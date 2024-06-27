import dataclasses
import itertools
import json
import os
import pathlib
import shutil
import uuid
from typing import *

import pytest
import torch
import torch.distributed as dist

from realhf.api.core.config import ModelFamily, ModelName, ModelShardID
from realhf.api.core.model_api import HF_MODEL_FAMILY_REGISTRY, ReaLModelConfig
from realhf.base import constants, logging, topology
from realhf.base.testing import (
    LocalMultiProcessTest,
    clear_name_resolve,
    init_global_constants,
    random_sample,
)
from tests.hf_utils import hf_config_factory

if TYPE_CHECKING:
    from realhf.impl.model.backend.inference import (
        PipelinableInferenceEngine,
        PipelineInferenceBackend,
    )
    from realhf.impl.model.backend.megatron import (
        MegatronTrainBackend,
        ReaLMegatronEngine,
    )
    from realhf.impl.model.nn.real_llm_api import ReaLModel


def compute_critic_loss(
    logits: torch.Tensor,
    packed_input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prompt_mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    from realhf.impl.model.utils.functional import (
        build_shift_one_indices,
        gather_packed_shifted_log_probs,
    )

    shift_one_indices = build_shift_one_indices(logits, cu_seqlens)
    prompt_mask = prompt_mask[shift_one_indices]
    scores = logits.squeeze().float()[shift_one_indices]
    scores = torch.where(prompt_mask, 0, scores)

    loss = ((scores - torch.zeros_like(scores)) ** 2).sum() / (
        prompt_mask.numel() - prompt_mask.count_nonzero()
    )
    return loss, {"loss": loss.clone().detach()}


def create_model(
    tmp_dir: pathlib.Path,
    model_family_name: str,
    model_name,
    is_critic: int,
    instantiate=True,
) -> "ReaLModel":
    # NOTE: import here to avoid initializing CUDA context in the main process
    from realhf.impl.model.nn.real_llm_api import ReaLModel

    with constants.model_scope(model_name):
        hf_config = hf_config_factory(model_family_name)
        mconfig: ReaLModelConfig = getattr(
            ReaLModel, f"config_from_{model_family_name}"
        )(hf_config)
        mconfig.is_critic = is_critic

        # initialize model
        model = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
        model.eval()
        if instantiate:
            model.instantiate()
            init_save_path = tmp_dir / "init"
            # sync initialized parameters
            getattr(model, f"to_{model_family_name}")(None, init_save_path)
            dist.barrier(group=constants.parallelism_group())
            model = getattr(model, f"from_{model_family_name}")(
                init_save_path, init_critic_from_actor=False
            )
    return model


def get_topo(model_name):
    with constants.model_scope(model_name):
        return constants.grid().topology()


def build_engine(module, model_name, trainable) -> "ReaLMegatronEngine":
    from realhf.api.core import model_api
    from realhf.impl.model.backend.inference import PipelineInferenceBackend
    from realhf.impl.model.backend.megatron import MegatronTrainBackend
    from realhf.impl.model.nn.real_llm_api import add_helper_functions

    with constants.model_scope(model_name):
        if constants.pipe_parallel_world_size() == 1:
            add_helper_functions(module)
        if trainable:
            backend = MegatronTrainBackend(initial_loss_scale=8.0)
        else:
            backend = PipelineInferenceBackend()
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


def setup_constants_and_param_realloc(
    from_model_name,
    to_model_name,
    from_pp_dp_mp,
    to_pp_dp_mp,
    from_sequence_parallel,
    to_sequence_parallel,
):
    from realhf.impl.model.comm.param_realloc import set_trainable, setup_param_realloc

    from_num_pp, from_num_dp, from_num_mp = from_pp_dp_mp
    to_num_pp, to_num_dp, to_num_mp = to_pp_dp_mp

    from_world_size = from_num_dp * from_num_mp * from_num_pp
    to_world_size = to_num_dp * to_num_mp * to_num_pp

    from_topo = topology.PipeModelDataParallelTopology(
        num_dp=from_num_dp,
        num_mp=from_num_mp,
        num_pp=from_num_pp,
        sequence_parallel=from_sequence_parallel,
        gradient_checkpointing=False,
        max_prompt_len=None,
    )
    to_topo = topology.PipeModelDataParallelTopology(
        num_dp=to_num_dp,
        num_mp=to_num_mp,
        num_pp=to_num_pp,
        sequence_parallel=to_sequence_parallel,
        gradient_checkpointing=False,
        max_prompt_len=None,
    )

    model_topos = {from_model_name: from_topo, to_model_name: to_topo}

    msid2mwid = {}
    for i in range(dist.get_world_size()):
        # We assume the `from_model` occupies the first serveral GPUs,
        # while the `to_model` occupies GPUs from the last one.
        # For example, when the world size of `from_model` is 6 and
        # the world size of `to_model` is 4, the GPU layout is:
        # GPU 0-3: from_model (shard 0-3)
        # GPU 4-5: from_model (shard 4-5) + to_model (shard 0-1)
        # GPU 6-7: to_model (shard 2-3)
        _model_names = []
        if i < from_world_size:
            _model_names.append(from_model_name)
        if i >= dist.get_world_size() - to_world_size:
            _model_names.append(to_model_name)
        for _model_name in _model_names:
            if _model_name == from_model_name:
                coord = model_topos[_model_name].get_coord(i)
            else:
                coord = model_topos[_model_name].get_coord(
                    i + to_world_size - dist.get_world_size()
                )
            k = ModelShardID(
                _model_name,
                dp_rank=coord.data,
                mp_rank=coord.model,
                pp_rank=coord.pipe,
                topo=model_topos[_model_name],
            )
            msid2mwid[k] = i

    init_global_constants(
        num_dp=from_num_dp,
        num_mp=from_num_mp,
        num_pp=from_num_pp,
        topo=from_topo,
        model_name=from_model_name,
        sequence_parallel=from_sequence_parallel,
        msid2mwid=msid2mwid,
    )

    init_global_constants(
        num_dp=to_num_dp,
        num_mp=to_num_mp,
        num_pp=to_num_pp,
        model_name=to_model_name,
        sequence_parallel=to_sequence_parallel,
        msid2mwid=msid2mwid,
    )

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


@dataclasses.dataclass
class ParamRedistributer:
    from_model_name: ModelName
    to_model_name: ModelName
    from_model: Any
    to_model: Any
    pg_info: Any

    def _redist(self, m1, m2, n1, n2):
        if m1 is None and m2 is None:
            return
        with constants.model_scope(n1):
            t1 = constants.grid().topology()
        with constants.model_scope(n2):
            t2 = constants.grid().topology()
        m = m1 if m1 is not None else m2
        a, b, c = m.build_reparallelized_layers_async(
            from_model_name=n1,
            to_model_name=n2,
            from_topo=t1,
            to_topo=t2,
            to_model_config=m.config,
            pg_info=self.pg_info,
        )
        if m2 is not None:
            m2.patch_reparallelization((a, b))

        if m1 is not None:
            assert m1.layers is None
            assert m1.contiguous_param is None
        if m2 is not None:
            assert m2.layers is not None
            assert m2.contiguous_param is not None

    def forward(self):
        self._redist(
            self.from_model,
            self.to_model,
            self.from_model_name,
            self.to_model_name,
        )

    def backward(self):
        self._redist(
            self.to_model,
            self.from_model,
            self.to_model_name,
            self.from_model_name,
        )


def _load_all_pytorch_bin(path: pathlib.Path):
    with open(path / "pytorch_model.bin.index.json", "r") as f:
        hf_sd_mapping = json.load(f)["weight_map"]
    sd = {}
    for fn in hf_sd_mapping.values():
        sd.update(torch.load(path / fn, map_location="cpu"))
    return sd


def _test_para_realloc(
    tmp_path: pathlib.Path,
    model_family_name: str,
    is_critic: bool,
    from_pp_dp_mp: Tuple,
    to_pp_dp_mp: Tuple,
    from_sequence_parallel: bool,
    to_sequence_parallel: bool,
):
    # os.environ["REAL_SAVE_MAX_SHARD_SIZE_BYTE"] = str(int(1e6))
    from realhf.impl.model.backend.megatron import ReaLMegatronEngine
    from realhf.impl.model.interface.sft_interface import compute_packed_sft_loss

    from_model_name = ModelName("param_realloc_test", 0)
    to_model_name = ModelName("param_realloc_test", 1)

    pg_info = setup_constants_and_param_realloc(
        from_model_name,
        to_model_name,
        from_pp_dp_mp,
        to_pp_dp_mp,
        from_sequence_parallel,
        to_sequence_parallel,
    )

    # Create model 1
    if dist.get_rank() < from_pp_dp_mp[0] * from_pp_dp_mp[1] * from_pp_dp_mp[2]:
        from_model = create_model(
            tmp_dir=tmp_path,
            model_family_name=model_family_name,
            model_name=from_model_name,
            is_critic=is_critic,
            instantiate=True,
        )
    else:
        from_model = None
    # Creat model 2
    if (
        dist.get_rank()
        >= dist.get_world_size() - to_pp_dp_mp[0] * to_pp_dp_mp[1] * to_pp_dp_mp[2]
    ):
        to_model = create_model(
            tmp_dir=tmp_path,
            model_family_name=model_family_name,
            model_name=to_model_name,
            is_critic=is_critic,
            instantiate=False,
        )
    else:
        to_model = None

    # Create redistributer.
    redist = ParamRedistributer(
        from_model_name,
        to_model_name,
        from_model,
        to_model,
        pg_info,
    )

    if from_model is not None:
        train_engine = build_engine(from_model, from_model_name, trainable=True)
    if to_model is not None:
        inf_engine = build_engine(to_model, to_model_name, trainable=False)

    prev_to_model_param = None
    prev_to_model_sd = None

    n_iterations = 4
    for i in range(n_iterations):
        if from_model is not None:
            vocab_size = from_model.config.vocab_size
        elif to_model is not None:
            vocab_size = to_model.config.vocab_size
        else:
            vocab_size = None

        if vocab_size is not None:
            x = random_sample(bs=32, seq_len=32, vocab_size=vocab_size)
            packed_input_ids = x["packed_input_ids"].cuda()
            cu_seqlens = x["cu_seqlens"].cuda()
            prompt_mask = x["prompt_mask"].cuda()
            input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            seqlens_cpu = input_lens.cpu().numpy().tolist()

        # Synchronize the initial parameters at the start of this iteration.
        if from_model is not None:
            with constants.model_scope(from_model_name):
                getattr(from_model, f"to_{model_family_name}")(
                    None, tmp_path / f"save_from_{i}"
                )
        dist.barrier()
        sd1 = _load_all_pytorch_bin(tmp_path / f"save_from_{i}")

        # Run redistribution.
        redist.forward()
        dist.barrier()

        # Synchronize the redistributed parameters. They should be identical to the initial parameters.
        # Also, they should be different from the parameters of the previous iteration
        # because we have updated the parameters.
        if to_model is not None:
            if prev_to_model_param is not None:
                assert not torch.allclose(
                    prev_to_model_param, to_model.contiguous_param
                )
            prev_to_model_param = to_model.contiguous_param.detach().clone()
            with constants.model_scope(to_model_name):
                getattr(to_model, f"to_{model_family_name}")(
                    None, tmp_path / f"save_to_{i}"
                )
        dist.barrier()
        sd2 = _load_all_pytorch_bin(tmp_path / f"save_to_{i}")
        if prev_to_model_sd is not None:
            assert not all(
                torch.allclose(v, sd2[k]) for k, v in prev_to_model_sd.items()
            ), (k, v, sd2[k])
        prev_to_model_sd = sd2
        for k, v in sd1.items():
            assert torch.allclose(v, sd2[k]), (k, v, sd2[k])

        # Run a forward with the redistributed model.
        if to_model is not None:
            with constants.model_scope(to_model_name):
                inf_engine.eval()
                logits1 = inf_engine.forward(
                    seqlens_cpu=seqlens_cpu,
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                )

        # Run redistribution backwards.
        redist.backward()
        dist.barrier()

        # Re-run redistribution to examine whether inference results are identical.
        redist.forward()
        dist.barrier()
        if to_model is not None:
            with constants.model_scope(to_model_name):
                inf_engine.eval()
                logits2 = inf_engine.forward(
                    seqlens_cpu=seqlens_cpu,
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                )
            if logits1 is not None:
                assert torch.allclose(logits1, logits2)
        redist.backward()
        dist.barrier()

        # Synchronize the redistributed parameters. They should be identical to the initial parameters.
        if from_model is not None:
            with constants.model_scope(from_model_name):
                getattr(from_model, f"to_{model_family_name}")(
                    None, tmp_path / f"save_back_{i}"
                )
        dist.barrier()
        sd3 = _load_all_pytorch_bin(tmp_path / f"save_back_{i}")
        for k, v in sd1.items():
            assert torch.allclose(v, sd3[k]), (k, v, sd3[k])

        # Train the model.
        if from_model is not None:
            for _ in range(2):
                loss_fn_kwargs = dict(
                    prompt_mask=prompt_mask,
                    input_lens=input_lens,
                )
                train_engine.eval()

                p = train_engine.engine.ddp.module.contiguous_param.clone().detach()

                with constants.model_scope(from_model_name):
                    train_engine: ReaLMegatronEngine
                    stats = train_engine.train_batch(
                        seqlens_cpu=seqlens_cpu,
                        packed_input_ids=packed_input_ids,
                        cu_seqlens=cu_seqlens,
                        loss_fn=(
                            compute_packed_sft_loss
                            if not is_critic
                            else compute_critic_loss
                        ),
                        num_micro_batches=constants.pipe_parallel_world_size() * 2,
                        version_steps=i,
                        **loss_fn_kwargs,
                    )

                p_ = train_engine.engine.ddp.module.contiguous_param.clone().detach()
                # After training, the parameters should have changed.
                assert not torch.allclose(p, p_)

        # Re-run redistribution to ensure that inference results changed.
        redist.forward()
        dist.barrier()
        if to_model is not None:
            with constants.model_scope(to_model_name):
                inf_engine.eval()
                logits3 = inf_engine.forward(
                    seqlens_cpu=seqlens_cpu,
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                )
            if logits1 is not None:
                assert not torch.allclose(logits1, logits3)
        redist.backward()
        dist.barrier()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 8,
    reason="This test requires at least 8 GPUs to run.",
)
@pytest.mark.slow
@pytest.mark.distributed
@pytest.mark.parametrize("model_family_name", ["gpt2", "llama"])
@pytest.mark.parametrize("is_critic", [False, True])
@pytest.mark.parametrize("from_pp_dp_mp", [(4, 1, 1)])
@pytest.mark.parametrize("to_pp_dp_mp", [(8, 1, 1)])
@pytest.mark.parametrize("from_sequence_parallel", [False])
@pytest.mark.parametrize("to_sequence_parallel", [False])
def test_param_realloc(
    model_family_name: str,
    is_critic: bool,
    from_pp_dp_mp: Tuple,
    to_pp_dp_mp: Tuple,
    from_sequence_parallel: bool,
    to_sequence_parallel: bool,
):
    expr_name = uuid.uuid4()
    trial_name = uuid.uuid4()
    clear_name_resolve(expr_name=expr_name, trial_name=trial_name)
    tmp_dir = pathlib.Path(f"/tmp/{expr_name}/{trial_name}")
    os.makedirs(tmp_dir, exist_ok=True)
    test_impl = LocalMultiProcessTest(
        world_size=8,
        func=_test_para_realloc,
        expr_name=expr_name,
        trial_name=trial_name,
        tmp_path=tmp_dir,
        model_family_name=model_family_name,
        is_critic=is_critic,
        from_pp_dp_mp=from_pp_dp_mp,
        to_pp_dp_mp=to_pp_dp_mp,
        from_sequence_parallel=from_sequence_parallel,
        to_sequence_parallel=to_sequence_parallel,
    )
    test_impl.launch()
    shutil.rmtree(tmp_dir)


def decompose_to_three_factors(n: int):
    factors = []
    for i in range(1, int(n ** (1 / 2)) + 1):
        if n % i == 0:
            for j in range(i, int((n // i) ** (1 / 2)) + 1):
                if (n // i) % j == 0:
                    k = (n // i) // j
                    factors += list(set(itertools.permutations([i, j, k])))
    return factors
