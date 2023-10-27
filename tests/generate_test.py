import logging
import os
import random
import time

import deepspeed
import torch

from base.constants import DATE_FORMAT, LOG_FORMAT
from base.namedarray import NamedArray, recursive_apply
import api.config
import api.data
import api.huggingface
import api.model
import impl.dataset
import impl.dataset.chat_dataset
import impl.model


def get_input(tokenizer, device, s):
    prompts = tokenizer(s, return_tensors="pt")
    # print(tokenizer.eos_token_id)
    # print(tokenizer.pad_token_id)
    input_ids, attention_mask = prompts["input_ids"], prompts["attention_mask"]

    return input_ids, attention_mask


def random_sentence(min_len=1, max_len=200):
    words = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
    sentence_length = random.randint(min_len, max_len)
    return " ".join(random.choices(words, k=sentence_length))


def main():
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level="INFO")

    ppo_kwargs = dict(
        ppo_epochs=1,
        mini_batch_size=2,
        kl_ctl=0.1,
        discount=1.0,
        gae_lambda=1.0,
        eps_clip=0.2,
        value_eps_clip=0.2,
        max_reward_clip=20.0,
    )
    interface_config = api.config.ModelInterface('chat_actor', args=ppo_kwargs)
    interface = api.model.make_interface(interface_config)

    backend_config = api.config.ModelBackend(
        'ds_train',
        args=dict(
            optimizer_name='adam',
            optimizer_config=dict(
                lr=2.5e-4,
                weight_decay=0.0,
                eps=1e-5,
                betas=(0.9, 0.95),
            ),
            lr_scheduler_type='cosine',
            warmup_steps_proportion=0.0,
            min_lr_ratio=0.0,
            zero_stage=2,
            enable_fp16=True,
            enable_hybrid_engine=True,
        ),
    )
    backend = api.model.make_backend(backend_config)

    actor_path = "/data/meizy/models/cfgonly/opt-1024-24"
    generation_kwargs = dict(
        max_new_tokens=256,
        min_new_tokens=256,
        # do_sample=False,
        # top_p=1.0,
        # top_k=1,
        # temperature=1.0,
        # num_beams=1,
        # num_beam_groups=1,
        # num_return_sequences=1,
    )
    model_config = api.config.Model(
        "causal_lm",
        args=dict(
            model_name_or_path=actor_path,
            init_from_scratch=False,
            from_pretrained_kwargs=dict(  # torch_dtype=torch.float16, 
                #use_cache=True,
            ),
            generation_kwargs=generation_kwargs,
            # quantization_kwargs=dict(load_in_8bit=True),
        ),
    )
    model = api.model.make_model(model_config, 'test_generate', torch.device('cuda:0'))

    ft_spec = api.model.FinetuneSpec(total_train_epochs=1,
                                     total_train_steps=1,
                                     steps_per_epoch=1,
                                     batch_size_per_device=2)
    model = backend.initialize(model, ft_spec)
    # model = deepspeed.init_inference(
    #     model=model.module,
    #     config={
    #         "replace_with_kernel_inject": True
    #     }
    # )
    # print(model.module.device)
    tokenizer = api.huggingface.load_hf_tokenizer(actor_path, padding_side="left")
    import pickle

    for i in range(10):
        sentences = [random_sentence() for _ in range(2)]
        prompts = tokenizer(sentences, return_tensors="pt", padding='max_length', max_length=256)
        input_ids, attention_mask = prompts["input_ids"], prompts["attention_mask"]
        # with open(f"/datafiles/prompts_{i}.pkl", "rb") as f:
        #     input_ids = pickle.load(f)
        # with open(f"/datafiles/mask_{i}.pkl", "rb") as f:
        #     attention_mask = pickle.load(f)

        data = NamedArray(
            prompts=input_ids,
            prompt_att_mask=attention_mask,
        )

        st = time.perf_counter()
        res = interface.generate(model, data)
        # res = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)
        et = time.perf_counter()
        print(f"generate res seq {res.seq.shape} time cost {et-st}")


if __name__ == "__main__":
    main()
