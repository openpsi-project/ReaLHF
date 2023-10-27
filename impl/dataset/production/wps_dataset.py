import json
import logging

import numpy as np
import torch
import torch.utils.data

import api.data
import api.huggingface

logger = logging.getLogger("WPS Excel Dataset")


def get_prompt(head, task):
    # This is specially for StarCoder (the "Answer: " postfix)
    return f"我的表格格式是：{head}，请写一段JavaScript代码完成下述任务：{task}。代码如下：\nAnswer: "


def get_prompt_and_chosen(head, task, code):
    return get_prompt(head, task) + code


from scripts.data.utils import RUBBISH_CODE_COLLECTIONS


class ExcelPlackettLuceRewardDataset(torch.utils.data.Dataset):

    def __init__(self, util: api.data.DatasetUtility, dataset_path, max_seq_len, contrastive_dim):
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

            if world_size > 1:
                other_indices = np.concatenate([
                    shuffle_indices[rank * datasize_per_rank:(rank + 1) * datasize_per_rank]
                    for rank in range(world_size) if rank != ddp_rank
                ])
                self.global_data = data + [json.loads(_data_bytes[j]) for j in other_indices]
            else:
                self.global_data = data

        self.raw_str_data = data
        self.labeled_codes = [d['labeled_codes'] for d in data]
        self.contrastive_dim = contrastive_dim
        self.max_seq_len = max_seq_len

        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.raw_str_data)

    def __getitem__(self, idx):
        head = self.raw_str_data[idx]['head']
        task = self.raw_str_data[idx]['task']

        existing_neg_codes = [x['code'] for x in self.labeled_codes[idx] if not x['correctness_label']]
        is_all_neg = all((not x['correctness_label']) for x in self.labeled_codes[idx])
        n_required_neg_codes = self.contrastive_dim - 1 if not is_all_neg else self.contrastive_dim

        sampled_other_code_indices = []
        sampled_rubbish_code_indices = []
        while len(existing_neg_codes) < n_required_neg_codes:
            if self.rng.random() < 0.1:
                rubbish_code_idx = self.rng.choice(len(RUBBISH_CODE_COLLECTIONS))
                if rubbish_code_idx in sampled_rubbish_code_indices:
                    continue
                existing_neg_codes.append(RUBBISH_CODE_COLLECTIONS[rubbish_code_idx])
                sampled_rubbish_code_indices.append(rubbish_code_idx)
                continue

            other_data_idx = self.rng.choice(len(self.global_data))
            if other_data_idx == idx:
                continue
            other_code_idx = self.rng.choice(len(self.global_data[other_data_idx]['labeled_codes']))
            if (other_data_idx, other_code_idx) in sampled_other_code_indices:
                continue
            other_code = self.global_data[other_data_idx]['labeled_codes'][other_code_idx]['code']
            existing_neg_codes.append(other_code)
            sampled_other_code_indices.append((other_data_idx, other_code_idx))

        if not is_all_neg:
            codes = [self.rng.choice([x['code'] for x in self.labeled_codes[idx] if x['correctness_label']])
                     ] + existing_neg_codes
        else:
            codes = existing_neg_codes
        assert len(codes) == self.contrastive_dim, len(codes)

        chosen_codes = [
            get_prompt_and_chosen(head, task, code) + self.util.tokenizer.eos_token for code in codes
        ]

        chosen_tokens = self.util.tokenizer(chosen_codes,
                                            max_length=self.max_seq_len,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt")

        return {
            "input_ids": chosen_tokens['input_ids'],
            "attention_mask": chosen_tokens['attention_mask'],
            "labels": torch.tensor(int(1 - is_all_neg), dtype=torch.long),
        }


api.data.register_dataset("wps_reward_plackett_luce", ExcelPlackettLuceRewardDataset)
