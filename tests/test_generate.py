from typing import *
import os
import pickle
import time

import torch
import torch.distributed as dist
import torch.utils.data

from realhf.api.core import data_api
from realhf.api.core.config import ModelFamily
from realhf.api.quickstart.model import ParallelismConfig
from realhf.base import constants, logging, testing
import realhf.api.core.model_api as model_api
import realhf.api.core.system_api as system_api

logger = logging.getLogger("tests.test_generate")

# MODEL_FAMILY_TO_PATH = {
#     ModelFamily("llama", 7, is_critic=False): "/path/to/Llama-2-7b-hf",
#     ModelFamily("llama", 13, is_critic=False): "/path/to/Llama-2-13b-hf",
# }
# PROMPT_DATASET_PATH = "/path/to/prompt_dataset.jsonl"

MODEL_FAMILY_TO_PATH = {
    ModelFamily(
        "llama", 7, is_critic=False
    ): "/lustre/public/pretrained_model_weights/Llama-2-7b-hf",
    ModelFamily(
        "llama", 13, is_critic=False
    ): "/lustre/public/pretrained_model_weights/Llama-2-13b-hf",
}
PROMPT_DATASET_PATH = (
    "/lustre/meizy/data/antropic-hh/ppo_prompt_only_short.jsonl"
)
TMP_SAVE_DIR = "profile_result/generate/"


@torch.no_grad()
def real_model_parallel_generate(
    model_family: ModelFamily,
    parallel: ParallelismConfig,
    max_prompt_len: int = 256,
    batch_size: int = 32,
    min_new_tokens: int = 128,
    max_new_tokens: int = 128,
    save_result: bool = False,
    max_num_batches: int = 10,
    use_cuda_graph: bool = True,
):
    import os

    os.environ["USE_CUDA_GRAPH"] = "1" if use_cuda_graph else "0"
    from realhf.base.namedarray import NamedArray
    from realhf.impl.model.nn.real_llm_generate import GenerationConfig

    model_path = MODEL_FAMILY_TO_PATH[model_family]
    testing.init_global_constants(
        num_dp=parallel.data_parallel_size,
        num_mp=parallel.model_parallel_size,
        num_pp=parallel.pipeline_parallel_size,
        sequence_parallel=parallel.use_sequence_parallel,
        max_prompt_len=max_prompt_len + max_new_tokens,
    )
    dataset_config = system_api.Dataset(
        "prompt",
        args=dict(
            dataset_path=PROMPT_DATASET_PATH,
            max_length=max_prompt_len,
            pad_to_max_length=False,
        ),
    )
    backend_config = system_api.ModelBackend("inference")
    model, dataset, backend = testing.prepare(
        model_family, model_path, backend_config, dataset_config
    )

    data_batches = testing.make_packed_input_batches(
        dataset, batch_size // parallel.data_parallel_size
    )

    to_save = []
    with constants.model_scope(testing.MODEL_NAME):
        gconfig_dict = dict(
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            greedy=True,
        )
        if_save = save_result and constants.data_parallel_rank() == 0
        # from realhf.impl.model.parallelism.model_parallel.custom_all_reduce import init_custom_ar
        # init_custom_ar()

        for i, data_batch in enumerate(data_batches[:max_num_batches]):
            data_batch: NamedArray
            packed_prompts = data_batch["packed_prompts"].cuda()
            prompt_lengths = torch.tensor(
                data_batch.metadata["seqlens"],
                dtype=torch.int32,
                device=packed_prompts.device,
            )
            prompt_cu_seqlens = torch.nn.functional.pad(
                prompt_lengths.cumsum(0), (1, 0)
            )

            model.module.eval()
            st = time.monotonic()
            res = model.module.generate(
                seqlens_cpu=data_batch.metadata["seqlens"],
                tokenizer=model.tokenizer,
                packed_input_ids=packed_prompts,
                cu_seqlens=prompt_cu_seqlens,
                gconfig=GenerationConfig(**gconfig_dict),
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
                if if_save and res is not None:
                    to_save.append(gen_tokens)
            logger.info(f"Batch {i} rank {dist.get_rank()} time = {t:.2f}s")

        if (
            if_save
            and constants.model_parallel_rank() == 0
            and len(to_save) > 0
        ):
            to_save = torch.cat(to_save, dim=0)
            print("saving", to_save.shape)
            identifier = "realcuda"
            identifier += f"dp{parallel.data_parallel_size}"
            identifier += f"mp{parallel.model_parallel_size}"
            identifier += f"pp{parallel.pipeline_parallel_size}"
            identifier += f"G" if use_cuda_graph else ""
            testing.save_test_result(
                to_save,
                TMP_SAVE_DIR,
                model_family,
                constants.data_parallel_rank(),
                identifier,
            )


@torch.no_grad()
def huggingface_model_generate_simple(
    model_family: ModelFamily,
    max_prompt_len: int = 256,
    batch_size: int = 32,
    min_new_tokens: int = 128,
    max_new_tokens: int = 128,
    save_result: bool = False,
    device: str = "cuda",
    max_num_batches: int = 10,
    shrink_model: bool = False,
):
    import realhf.impl.dataset

    model_path = MODEL_FAMILY_TO_PATH[model_family]
    tokenizer = model_api.load_hf_tokenizer(MODEL_FAMILY_TO_PATH[model_family])
    dataset = data_api.load_shuffle_split_dataset(
        data_api.DatasetUtility(1, 0, 1, tokenizer), PROMPT_DATASET_PATH
    )
    data_batches = testing.make_huggingface_generate_input_batches(
        dataset, tokenizer, max_prompt_len, batch_size
    )

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="flash_attention_2"
    )
    model.to("cuda", dtype=torch.float16)
    # print(model.state_dict()["lm_head.weight"])

    to_save = []
    for i, batch in enumerate(data_batches[:max_num_batches]):
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        prompt_len = input_ids.shape[1]
        res = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,  # match generate output length
            min_new_tokens=min_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1,
            top_k=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        logger.info(
            f"Batch {i} tokens shape "
            f"{res['sequences'].shape} scores shape {res['scores'][-1].shape}"
        )
        to_save.append(res["sequences"][:, prompt_len:])

    if save_result:
        to_save = torch.cat(to_save, dim=0)
        testing.save_test_result(
            to_save, TMP_SAVE_DIR, model_family, 0, f"huggingface{device}"
        )


if __name__ == "__main__":
    model_family = ModelFamily("llama", 7, is_critic=False)
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    num_mp, num_pp, num_dp = 1, 1, 1
    n_gpus = num_mp * num_pp * num_dp
    testing.remove_file_cache(TMP_SAVE_DIR)

    test1 = testing.LocalMultiProcessTest(
        n_gpus,
        real_model_parallel_generate,
        model_family,
        ParallelismConfig(num_mp, num_pp, num_dp, False),
        256,
        64,
        128,
        128,
        True,
        3,
        True,
    )
    test1.launch()

    test2 = testing.LocalMultiProcessTest(
        n_gpus,
        real_model_parallel_generate,
        model_family,
        ParallelismConfig(num_mp, num_pp, num_dp, False),
        256,
        64,
        128,
        128,
        True,
        3,
        False,
    )
    test2.launch()

    # hf_test = testing.LocalMultiProcessTest(
    #     1, huggingface_model_generate_simple,
    #     model_family,
    #     256, 32, 128, 128, True, "cuda", 16
    # )
    # hf_test.launch()
    non_cudagraph = "realcuda" + f"dp{num_dp}" + f"mp{num_mp}" + f"pp{num_pp}"
    cuda_graph = non_cudagraph + "G"
    testing.check_generation_consistency(
        TMP_SAVE_DIR, model_family, [cuda_graph, non_cudagraph]
    )
