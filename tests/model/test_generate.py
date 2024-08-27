import dataclasses
import os
import pickle
import time
from typing import *

import pytest
import torch
import torch.distributed as dist
import torch.utils.data
import transformers

import realhf.api.core.model_api as model_api
from realhf.api.quickstart.model import ParallelismConfig
from realhf.base import constants, logging, testing

logger = logging.getLogger("tests.test_generate")

# NOTE: To run test for a new model class, add a new entry to MODEL_CLASS_TO_PATH
# with the model class name as the key and the path to the model weights as the value.
MODEL_CLASS_TO_PATH = {
    "llama": "/lustre/public/pretrained_model_weights/Llama-2-7b-hf",
    "gpt2": "/lustre/public/pretrained_model_weights/gpt2",
}
_available_model_classes = []
for k, v in MODEL_CLASS_TO_PATH.items():
    if os.path.exists(v):
        _available_model_classes.append(k)


@pytest.fixture(params=_available_model_classes)
def model_class(request):
    return request.param


@dataclasses.dataclass
class GenerateTestParams:
    parallel: ParallelismConfig = dataclasses.field(default_factory=ParallelismConfig)
    huggingface: bool = False
    prompt_len: int = 64
    gen_len: int = 64
    batch_size: int = 32
    n_batches: int = 2
    use_cuda_graph: bool = False

    def identifier(self):
        s = "real" if not self.huggingface else "huggingface"
        s += f"dp{self.parallel.data_parallel_size}"
        s += f"mp{self.parallel.model_parallel_size}"
        s += f"pp{self.parallel.pipeline_parallel_size}"
        s += f"G" if self.use_cuda_graph else ""
        return s


def check_sequences_consistency(
    batched_seq1: torch.LongTensor, batched_seq2: torch.LongTensor
):
    matched_tokens = 0
    matched_seqs = 0
    total_tokens = 0
    assert len(batched_seq1) == len(batched_seq2)
    for i in range(len(batched_seq1)):
        a = batched_seq1[i]
        b = batched_seq2[i]
        assert torch.is_tensor(a) and torch.is_tensor(b)
        assert a.dim() == 1 and b.dim() == 1, (a.shape, b.shape)
        gen_len = a.shape[0] if a.shape[0] < b.shape[0] else b.shape[0]
        b = b[:gen_len]
        a = a[:gen_len]
        for j in range(gen_len):
            if a[j] != b[j]:
                logger.info(f"Mismatch at sequence {i} position {j}")
                break
            matched_tokens += 1
        else:
            matched_seqs += 1
        total_tokens += gen_len
    logger.info(
        f"Matched {matched_seqs}/{len(batched_seq1)} "
        f"sequences and {matched_tokens}/{total_tokens} tokens"
    )
    return (
        matched_seqs,
        matched_tokens,
        float(matched_tokens) / total_tokens,
        float(matched_seqs) / len(batched_seq1),
    )


def check_generation_consistency(
    model_class: str,
    save_result_path: str,
    test_params_list: List[GenerateTestParams],
    seq_acc_threshold: float = 1.0,
    token_acc_threshold: float = 1.0,
):
    identifiers = [test_params.identifier() for test_params in test_params_list]
    dp_rank_counter = {identifier: 0 for identifier in identifiers}
    for f in os.listdir(save_result_path):
        if not f.endswith(".pkl"):
            continue
        identifier, _, _ = f.split("-")
        if identifier in identifiers:
            dp_rank_counter[identifier] += 1

    results = {}
    for identifier in identifiers:
        tmp = []
        for i in range(dp_rank_counter[identifier]):
            fn = f"{identifier}-{model_class}-{i}.pkl"
            logger.info(f"loading file {fn}")
            load_path = os.path.join(save_result_path, fn)
            t = pickle.load(open(load_path, "rb"))
            tmp.append(t)
        res = torch.cat(tmp, dim=0)
        logger.info(f"test {identifier} loaded result shape {res.shape}")
        results[identifier] = res

    baseline_result = results[identifiers[0]]
    for identifier, result in results.items():
        if identifier == identifiers[0]:
            continue
        _, _, seq_acc, token_acc = check_sequences_consistency(baseline_result, result)
        assert seq_acc >= seq_acc_threshold and token_acc >= token_acc_threshold, (
            identifier,
            identifiers[0],
            seq_acc,
            token_acc,
        )


def save_result_filename(
    model_class: str,
    save_result_path: str,
    dp_rank: int,
    test_params: GenerateTestParams,
):
    fn = test_params.identifier()
    fn += f"-{model_class}"
    fn += f"-{dp_rank}.pkl"
    return os.path.join(save_result_path, fn)


@pytest.fixture
def model_path(model_class: str):
    return MODEL_CLASS_TO_PATH[model_class]


@pytest.fixture
def save_result_path(tmpdir_factory: pytest.TempdirFactory):
    return tmpdir_factory.mktemp("save_result")


@pytest.fixture
def tokenizer(model_path: str):
    return model_api.load_hf_tokenizer(model_path)


def make_generate_model(model_class, model_path):
    with constants.model_scope(testing.MODEL_NAME):
        from realhf.impl.model.backend.inference import PipelinableInferenceEngine
        from realhf.impl.model.nn.real_llm_api import ReaLModel, add_helper_functions

        mconfig = getattr(ReaLModel, f"config_from_{model_class}")(
            model_path=model_path,
            is_critic=False,
        )
        model = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
        if constants.pipe_parallel_world_size() == 1:
            model = add_helper_functions(model)
        model.instantiate()
        getattr(model, f"from_{model_class}")(
            load_dir=model_path, init_critic_from_actor=False
        )
        model = PipelinableInferenceEngine(model)
        model.eval()
    return model


def make_huggingface_inference_model(model_path):
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_path).to(
        dtype=torch.float16, device="cuda"
    )
    hf_model.eval()
    return hf_model


@torch.no_grad()
def real_model_generate(
    model_class: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str,
    save_result_path: str,
    test_params: GenerateTestParams,
):
    from realhf.api.core.model_api import GenerationHyperparameters

    assert not test_params.huggingface
    parallel = test_params.parallel
    generation_config = GenerationHyperparameters(
        min_new_tokens=test_params.gen_len,
        max_new_tokens=test_params.gen_len,
        use_cuda_graph=test_params.use_cuda_graph,
        greedy=True,
    )
    testing.init_global_constants(
        num_dp=parallel.data_parallel_size,
        num_mp=parallel.model_parallel_size,
        num_pp=parallel.pipeline_parallel_size,
        sequence_parallel=parallel.use_sequence_parallel,
        max_prompt_len=test_params.prompt_len + test_params.gen_len,
    )
    model = make_generate_model(model_class, model_path)
    with constants.model_scope(testing.MODEL_NAME):
        data_batches = testing.make_random_packed_batches(
            test_params.n_batches,
            test_params.batch_size,
            test_params.prompt_len,
            tokenizer.vocab_size,
            seed=1,
        )
        if_save_result = (
            constants.pipe_parallel_rank() == constants.pipe_parallel_world_size() - 1
            and constants.model_parallel_rank() == 0
        )

        results = []
        for i, data_batch in enumerate(data_batches):
            data_batch = data_batch.cuda()
            logger.info(
                f"Batch {i} rank {constants.parallelism_rank()} "
                f"prompt shape {data_batch.data['packed_input_ids'].shape}"
            )
            st = time.monotonic()
            res = model.generate(
                input_=data_batch,
                tokenizer=tokenizer,
                gconfig=generation_config,
            )
            t = time.monotonic() - st

            if res is None:
                logger.info(f"Batch {i} rank {dist.get_rank()} return None")
            else:
                gen_tokens, logprobs, logits_mask, *_ = res
                logger.info(
                    f"Batch {i} rank {dist.get_rank()} generated shape = {gen_tokens.shape}, "
                    f"logprobs shape = {logprobs.shape}"
                )
                if if_save_result and res is not None:
                    results.append(gen_tokens)
            logger.info(f"Batch {i} rank {dist.get_rank()} time = {t:.2f}s")

        if if_save_result and len(results) > 0:
            results = torch.cat(results, dim=0)
            path = save_result_filename(
                model_class,
                save_result_path,
                constants.data_parallel_rank(),
                test_params,
            )
            pickle.dump(results, open(path, "wb"))


def huggingface_model_generate(
    model_class: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str,
    save_result_path: str,
    test_params: GenerateTestParams,
):
    assert test_params.huggingface
    parallel = test_params.parallel
    assert (
        parallel.data_parallel_size
        == parallel.model_parallel_size
        == parallel.pipeline_parallel_size
        == 1
    )
    data_batches = testing.make_random_unpacked_batches(
        test_params.n_batches,
        test_params.batch_size,
        test_params.prompt_len,
        tokenizer.vocab_size,
        seed=1,
        dp_rank=0,
        dp_size=1,
    )
    model = make_huggingface_inference_model(model_path)
    results = []

    for i, batch in enumerate(data_batches):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        prompt_len = input_ids.shape[1]
        res = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=test_params.gen_len,  # match generate output length
            min_new_tokens=test_params.gen_len,
            do_sample=False,
        )
        res = res[:, prompt_len:]
        results.append(res)

    if len(results) > 0:
        results = torch.cat(results, dim=0)
        path = save_result_filename(model_class, save_result_path, 0, test_params)
        pickle.dump(results, open(path, "wb"))


real_simple = GenerateTestParams()
huggingface = GenerateTestParams(huggingface=True)
real_cudagraph = GenerateTestParams(use_cuda_graph=True)
real_3d_parallel = GenerateTestParams(parallel=ParallelismConfig(2, 2, 2, False))
real_3d_parallel_cudagraph = GenerateTestParams(
    parallel=ParallelismConfig(2, 2, 2, False), use_cuda_graph=True
)
real_mp = GenerateTestParams(parallel=ParallelismConfig(8, 1, 1, False))
real_mp_cudagraph = GenerateTestParams(
    parallel=ParallelismConfig(8, 1, 1, False), use_cuda_graph=True
)
real_dp = GenerateTestParams(parallel=ParallelismConfig(1, 1, 8, False))
real_dp_cudagraph = GenerateTestParams(
    parallel=ParallelismConfig(1, 1, 8, False), use_cuda_graph=True
)
real_pp = GenerateTestParams(parallel=ParallelismConfig(1, 4, 1, False))
real_pp_cudagraph = GenerateTestParams(
    parallel=ParallelismConfig(1, 4, 1, False), use_cuda_graph=True
)


@pytest.mark.parametrize(
    "test_params1,test_params2,seq_acc_threshold,token_acc_threshold",
    [
        (real_simple, huggingface, 0.8, 0.8),
        (real_simple, real_cudagraph, 1.0, 1.0),
        (real_simple, real_3d_parallel, 0.8, 0.8),
        (real_mp, real_mp_cudagraph, 0.8, 0.8),
        (real_dp, real_dp_cudagraph, 1.0, 1.0),
        (real_pp, real_pp_cudagraph, 1.0, 1.0),
        (real_3d_parallel, real_3d_parallel_cudagraph, 0.8, 0.8),
    ],
)
@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.skipif(
    not torch.cuda.is_available() or len(_available_model_classes) == 0,
    reason="either no GPUs or no model weights found",
)
def test_generate_consistency(
    model_class: str,
    tokenizer: Any,
    model_path: str,
    save_result_path: str,
    test_params1: GenerateTestParams,
    test_params2: GenerateTestParams,
    seq_acc_threshold: float,
    token_acc_threshold: float,
):
    from realhf.impl.model.nn.real_llm_api import ReaLModel

    mconfig = getattr(ReaLModel, f"config_from_{model_class}")(
        model_path=model_path,
        is_critic=False,
    )
    if (
        mconfig.vocab_size % test_params1.parallel.model_parallel_size != 0
        or mconfig.vocab_size % test_params2.parallel.model_parallel_size != 0
    ):
        return

    def launch_test(test_params: GenerateTestParams):
        func = (
            huggingface_model_generate
            if test_params.huggingface
            else real_model_generate
        )
        ngpus = (
            test_params.parallel.data_parallel_size
            * test_params.parallel.model_parallel_size
            * test_params.parallel.pipeline_parallel_size
        )
        assert ngpus <= torch.cuda.device_count()

        test = testing.LocalMultiProcessTest(
            ngpus,
            func,
            model_class,
            tokenizer,
            model_path,
            save_result_path,
            test_params,
        )
        test.launch()

    launch_test(test_params1)
    launch_test(test_params2)

    check_generation_consistency(
        model_class,
        save_result_path,
        [test_params1, test_params2],
        seq_acc_threshold,
        token_acc_threshold,
    )
