import itertools
import json
import logging

import numpy as np
import torch
import torch.utils.data

import api.data
import api.utils
import base.namedarray

logger = logging.getLogger("WPS Excel Dataset")


def get_prompt(head, task):
    return f"我的表格格式是：{head}，请写一段JavaScript代码完成下述任务：{task}。代码如下：\n"


def get_prompt_and_chosen(sample):
    return get_prompt(sample['head'], sample['task']) + sample['code']


class ExcelPromptDataset(api.data.Dataset):

    def __init__(self, seed, ddp_rank, world_size, dataset_path, tokenizer_name_or_path, max_seq_len):
        tokenizer = api.utils.load_hf_tokenizer(tokenizer_name_or_path)

        super().__init__(seed, ddp_rank, world_size, tokenizer)

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
            prompt = get_prompt(tmp_data['head'], tmp_data['task'])
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
                self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)


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


api.data.register_dataloader("excel_rlhf", RLHFDataLoader)
api.data.register_dataset("excel_prompt", ExcelPromptDataset)


class ExcelRewardModelingPairDataset(api.data.Dataset):

    def __init__(self, seed, ddp_rank, world_size, dataset_path, tokenizer_name_or_path, max_seq_len):
        tokenizer = api.utils.load_hf_tokenizer(tokenizer_name_or_path)

        super().__init__(seed, ddp_rank, world_size, tokenizer)

        if not dataset_path.endswith(".jsonl"):
            raise NotImplementedError("Only support .jsonal dataset format.")

        with open(dataset_path, 'r') as f:
            _data_bytes = [ff for ff in f]
            datasize_per_rank = len(_data_bytes) // world_size
            shuffle_indices = api.data.get_shuffle_indices(seed, datasize_per_rank * world_size)
            subset_indices = shuffle_indices[ddp_rank * datasize_per_rank:(ddp_rank + 1) * datasize_per_rank]
            data = [json.loads(_data_bytes[i]) for i in subset_indices]

        end_of_conversation_token = tokenizer.eos_token

        prompts_str = []
        grouped_chosen_sentences_str = []
        grouped_rejected_sentences_str = []
        for i, tmp_data in enumerate(data):
            # tokenize the text
            chosen_sentences = [x + end_of_conversation_token for x in self.get_prompt_and_chosen(tmp_data)]
            rejected_sentences = [
                x + end_of_conversation_token for x in self.get_prompt_and_rejected(tmp_data)
            ]

            prompts_str.append(self.get_prompt(tmp_data))
            grouped_chosen_sentences_str.append(chosen_sentences)
            grouped_rejected_sentences_str.append(rejected_sentences)

        group_sizes = [len(x) for x in grouped_chosen_sentences_str]

        self.starts = np.cumsum([0] + group_sizes[:-1])
        self.ends = np.cumsum(group_sizes)
        self.chosen_tokens = tokenizer(
            list(itertools.chain.from_iterable(grouped_chosen_sentences_str)),
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.rejected_tokens = tokenizer(
            list(itertools.chain.from_iterable(grouped_rejected_sentences_str)),
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.prompt_tokens = tokenizer(
            prompts_str,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self):
        return self.prompt_tokens['input_ids'].shape[0]

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = self.ends[idx]
        return (
            self.prompt_tokens['input_ids'][idx],
            self.prompt_tokens['attention_mask'][idx],
            self.chosen_tokens['input_ids'][start:end],
            self.chosen_tokens['attention_mask'][start:end],
            self.rejected_tokens['input_ids'][start:end],
            self.rejected_tokens['attention_mask'][start:end],
        )

    @staticmethod
    def get_prompt(sample):
        return get_prompt(sample['head'], sample['task'])

    @staticmethod
    def get_chosen(sample):
        return sample['codes_chosen']

    @staticmethod
    def get_rejected(sample):
        return sample['codes_reject']

    @staticmethod
    def get_prompt_and_chosen(sample):
        return [
            ExcelRewardModelingPairDataset.get_prompt(sample) + x
            for x in ExcelRewardModelingPairDataset.get_chosen(sample)
        ]

    @staticmethod
    def get_prompt_and_rejected(sample):
        return [
            ExcelRewardModelingPairDataset.get_prompt(sample) + x
            for x in ExcelRewardModelingPairDataset.get_rejected(sample)
        ]


class _DataCollatorReward:

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, data):
        batch = {}
        group_prompt_input_ids = [f[0] for f in data]
        group_prompt_attention_mask = [f[1] for f in data]

        chosen_input_ids = [f[2] for f in data]
        chosen_attention_mask = [f[3] for f in data]
        reject_input_ids = [f[4] for f in data]
        reject_atttention_mask = [f[5] for f in data]

        prompt_input_ids = []
        prompt_group_sizes = []
        for i, b in enumerate(chosen_input_ids):
            prompt_group_sizes += [b.shape[0] for _ in range(b.shape[0])]
            prompt_input_ids += [group_prompt_input_ids[i] for _ in range(b.shape[0])]

        batch["input_ids"] = torch.cat(chosen_input_ids + reject_input_ids, dim=0)
        batch["attention_mask"] = torch.cat(chosen_attention_mask + reject_atttention_mask, dim=0)
        batch["group_factor"] = 1 / torch.tensor(prompt_group_sizes, dtype=torch.float32)

        prompt_input_ids = torch.stack(prompt_input_ids * 2)
        labels = batch["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100
        labels[prompt_input_ids != self.pad_token_id] = -100
        batch['labels'] = labels
        # print(batch["group_factor"], [len(f[0]) for f in data])
        return batch


def RewardModelingDataLoader(dataset, *args, **kwargs):
    collator = _DataCollatorReward(dataset.tokenizer.pad_token_id)
    return torch.utils.data.DataLoader(dataset, *args, collate_fn=collator, **kwargs)


api.data.register_dataloader("excel_reward_pair", RewardModelingDataLoader)
api.data.register_dataset("excel_reward_pair", ExcelRewardModelingPairDataset)


class ExcelRewardModelingUnpairedDataset(api.data.Dataset):

    def __init__(self, seed, ddp_rank, world_size, dataset_path, tokenizer_name_or_path, max_seq_len):
        tokenizer = api.utils.load_hf_tokenizer(tokenizer_name_or_path)

        super().__init__(seed, ddp_rank, world_size, tokenizer)

        if not dataset_path.endswith(".jsonl"):
            raise NotImplementedError("Only support .jsonal dataset format.")

        with open(dataset_path, 'r') as f:
            _data_bytes = [ff for ff in f]
            datasize_per_rank = len(_data_bytes) // world_size
            shuffle_indices = api.data.get_shuffle_indices(seed, datasize_per_rank * world_size)
            subset_indices = shuffle_indices[ddp_rank * datasize_per_rank:(ddp_rank + 1) * datasize_per_rank]
            data = [json.loads(_data_bytes[i]) for i in subset_indices]

        end_of_conversation_token = tokenizer.eos_token

        chosen_sentences_str = []
        for i, tmp_data in enumerate(data):
            # tokenize the text
            chosen_sentence = get_prompt_and_chosen(tmp_data)
            chosen_sentence += end_of_conversation_token
            chosen_sentences_str.append(chosen_sentence)
        self.chosen_token = tokenizer(chosen_sentences_str,
                                      max_length=max_seq_len,
                                      padding="max_length",
                                      truncation=True,
                                      return_tensors="pt")
        self.chosen_token['correctness_labels'] = torch.tensor([d['correctness_label'] for d in data],
                                                               dtype=torch.long)

        input_ids = self.chosen_token['input_ids']
        eos_mask = (input_ids == self.tokenizer.eos_token_id).float()
        seq_no_eos_mask = (eos_mask.sum(1) == 0).float()
        eos_indices = eos_mask.argmax(1)
        eos_indices = (eos_indices * (1 - seq_no_eos_mask) + seq_no_eos_mask * (max_seq_len - 1)).long()
        self.chosen_token['eos_indices'] = eos_indices

    def __len__(self):
        return self.chosen_token['input_ids'].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.chosen_token["input_ids"][idx],
            "attention_mask": self.chosen_token["attention_mask"][idx],
            "correctness_labels": self.chosen_token["correctness_labels"][idx],
            "eos_indices": self.chosen_token['eos_indices'][idx],
        }


api.data.register_dataset("excel_reward_modeling_unpaired", ExcelRewardModelingUnpairedDataset)
