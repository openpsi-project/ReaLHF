import json
import os

import transformers

from reallm.impl.model.nn.real_llm_api import ReaLModel
from reallm.impl.model.nn.real_llm_base import FlashMQATConfig
from reallm.impl.model.utils.save_load import load_from_disk


def main():
    llama_path = "/lustre/public/pretrained_model_weights/CodeLlama-34b-hf"
    model_dir = "/lustre/aigc/llm/checkpoints/xss/codellama34b-ccs-ppo-gtrw-py3-fromapp-critic34b_bs512v3-sft-epoch48step0-1e-5/"
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = FlashMQATConfig(**json.load(f))
    state_dict = load_from_disk(model_dir)
    print("loaded state dict")
    output_dir = "/lustre/aigc/llm/checkpoints/fw/pipeinf-test2/"
    ReaLModel.dump_to_llama(config, state_dict, output_dir, hf_base_model_path=llama_path)

    # test availability in huggingface
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(output_dir)

    # llama_path = "/lustre/public/pretrained_model_weights/Llama-2-7b-hf"
    # model_dir = "/lustre/public/pretrained_model_weights/sharded_new/Llama-2-7b-hf_2pp_1mp/"
    # with open(os.path.join(model_dir, "flash_mqat_config.json"), "r") as f:
    #     config = FlashMQATConfig(**json.load(f))
    # state_dict = load_from_disk(model_dir)
    # print("loaded state dict")
    # output_dir = "/lustre/aigc/llm/checkpoints/fw/_pipeinf-test/"
    # ReaLModel.dump_to_llama(config, state_dict, output_dir, hf_base_model_path=llama_path)
    # hf_model = transformers.AutoModelForCausalLM.from_pretrained(output_dir)

    print("success")


if __name__ == "__main__":
    main()
