import itertools
import json
import logging

import numpy as np
import torch
import torch.utils.data

import api.data
import api.utils
import base.namedarray

logger = logging.getLogger("Chat Dataset")


class ChatPromptDataset(torch.utils.data.Dataset):

    def __init__(self, util: api.data.DatasetUtility, dataset_path, max_seq_len):
        self.util = util
        seed = self.util.seed
        world_size = self.util.world_size
        tokenizer = self.util.tokenizer
        ddp_rank = self.util.ddp_rank

        if not dataset_path.endswith(".jsonl"):
            raise NotImplementedError("Only support .jsonal dataset format.")

        with open(dataset_path, 'r') as f:
            _data_bytes = [ff for ff in f]
            datasize_per_rank = len(_data_bytes) // world_size
            shuffle_indices = api.data.get_shuffle_indices(seed, datasize_per_rank * world_size)
            subset_indices = shuffle_indices[ddp_rank * datasize_per_rank:(ddp_rank + 1) * datasize_per_rank]
            data = [json.loads(_data_bytes[i]) for i in subset_indices]

        prompt_dataset = []
        for i, tmp_data in enumerate(data):
            # prompt token should not add an eos token
            # prompt is just used for masking out the loss of task tokens
            try:
                prompt = tmp_data["prompt"]
            except Exception as e:
                print(tmp_data)
                raise e
            prompt_token = tokenizer(prompt, return_tensors='pt')
            for key_word in ["input_ids", "attention_mask"]:
                length = prompt_token[key_word].size()[-1]
                if length > max_seq_len:
                    y = prompt_token[key_word].squeeze(0)[length - (max_seq_len - 1):].flip(0)
                else:
                    y = prompt_token[key_word].squeeze(0).flip(0)
                prompt_token[key_word] = y
            # print(tokenizer.batch_decode(new_prompt_token['input_ids']))
            prompt_dataset.append(prompt_token)

        self.prompt_dataset = prompt_dataset

    def __len__(self):
        return len(self.prompt_dataset)

    def __getitem__(self, idx):
        return (self.prompt_dataset[idx]["input_ids"], self.prompt_dataset[idx]["attention_mask"],
                self.util.tokenizer.pad_token_id, self.util.tokenizer.eos_token_id)


class DataCollatorRLHF:

    def __init__(self, max_token_len):
        self.max_token_len = max_token_len

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-2]
        eos_token_id = data[-1][-1]

        prompt = torch.nn.utils.rnn.pad_sequence([f[0] for f in data],
                                                 padding_value=pad_token_id,
                                                 batch_first=True)
        prompt_mask = torch.nn.utils.rnn.pad_sequence([f[1] for f in data], padding_value=0, batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompts"] = torch.nn.functional.pad(prompt,
                                                       pad=(0, pad_length),
                                                       mode='constant',
                                                       value=pad_token_id)
            batch["prompt_att_mask"] = torch.nn.functional.pad(prompt_mask,
                                                               pad=(0, pad_length),
                                                               mode='constant',
                                                               value=0)
        else:
            batch["prompts"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompts"] = batch["prompts"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        batch['pad_token_id'] = pad_token_id
        batch['eos_token_id'] = eos_token_id
        return batch


def RLHFDataLoader(dataset, max_token_len, *args, **kwargs):
    collator = DataCollatorRLHF(max_token_len)
    return torch.utils.data.DataLoader(dataset, *args, collate_fn=collator, **kwargs)


api.data.register_dataloader("chat_rlhf", RLHFDataLoader)
api.data.register_dataset("chat_prompt", ChatPromptDataset)

if __name__ == "__main__":
    tokenizer = api.utils.load_hf_tokenizer("/lustre/meizy/base_models/opt-125m")
    utils = [
        api.data.DatasetUtility(seed=1, ddp_rank=rank, world_size=4, tokenizer=tokenizer) for rank in range(4)
    ]

    datasets = [
        ChatPromptDataset(utils[rank], "/lustre/meizy/datasets/Dahoas/rm-static/data/data.jsonl", 512)
        for rank in range(4)
    ]

    dataset = datasets[0]
    print(len(dataset))
