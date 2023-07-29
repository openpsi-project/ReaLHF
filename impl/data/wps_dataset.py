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
    # This is specially for StarCoder (the "Answer: " postfix)
    return f"我的表格格式是：{head}，请写一段JavaScript代码完成下述任务：{task}。代码如下：\nAnswer: "


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

# TODO: use GPT to generate more rubbish codes
RUBBISH_CODE_COLLECTIONS = [
    "const sheet = Application.ActiveSheet;\nconst usedRange = sheet.UsedRange;\nconst rowCount = usedRange.Rows.Count;",
    "const sheet = Application.ActiveSheet\nconst usedRange = sheet.UsedRange\nconst rowCount = usedRange.Rows.Count",
    "const sheet = Application.ActiveSheet\nconst usedRange = sheet.UsedRange\nconst rowCount = usedRange.Rows.Count\nfor(let i = usedRange.Row + 1; i <= rowCount; i++) {\n}",
]


class ExcelContrastiveRewardDataset(api.data.Dataset):

    def __init__(self, seed, ddp_rank, world_size, dataset_path, tokenizer_name_or_path, max_prompt_len,
                 max_code_len, contrastive_dim):
        tokenizer = api.utils.load_hf_tokenizer(tokenizer_name_or_path)

        # TODO: refactor this sh*t
        super().__init__(seed, ddp_rank, world_size, tokenizer)

        if not dataset_path.endswith(".jsonl"):
            raise NotImplementedError("Only support .jsonal dataset format.")

        with open(dataset_path, 'r') as f:
            _data_bytes = [ff for ff in f]
            datasize_per_rank = len(_data_bytes) // world_size
            shuffle_indices = api.data.get_shuffle_indices(seed, datasize_per_rank * world_size)
            subset_indices = shuffle_indices[ddp_rank * datasize_per_rank:(ddp_rank + 1) * datasize_per_rank]
            data = [json.loads(_data_bytes[i]) for i in subset_indices]

            other_indices = np.concatenate([
                shuffle_indices[rank * datasize_per_rank:(rank + 1) * datasize_per_rank]
                for rank in range(world_size) if rank != ddp_rank
            ])
            self.global_data = data + [json.loads(_data_bytes[j]) for j in other_indices]

        prompts_str = [get_prompt(d['head'], d['task']) for d in data]
        tokenizer.padding_side = 'left'
        self.prompt_token = tokenizer(prompts_str,
                                      max_length=max_prompt_len,
                                      padding="max_length",
                                      truncation=True,
                                      return_tensors="pt")
        tokenizer.padding_side = "right"

        self.labeled_codes = [d['labeled_codes'] for d in data]
        self.contrastive_dim = contrastive_dim
        self.max_code_len = max_code_len

    def __len__(self):
        return self.prompt_token['input_ids'].shape[0]

    def __getitem__(self, idx):
        existing_neg_codes = [x['code'] for x in self.labeled_codes[idx] if not x['correctness_label']]
        is_all_neg = all((not x['correctness_label']) for x in self.labeled_codes[idx])
        n_required_neg_codes = self.contrastive_dim - 1 if not is_all_neg else self.contrastive_dim

        sampled_other_code_indices = []
        sampled_rubbish_code_indices = []
        while len(existing_neg_codes) < n_required_neg_codes:
            if np.random.random() < 0.1:
                rubbish_code_idx = np.random.choice(len(RUBBISH_CODE_COLLECTIONS))
                if rubbish_code_idx in sampled_rubbish_code_indices:
                    continue
                existing_neg_codes.append(RUBBISH_CODE_COLLECTIONS[rubbish_code_idx])
                sampled_rubbish_code_indices.append(rubbish_code_idx)
                continue

            other_data_idx = np.random.choice(len(self.global_data))
            if other_data_idx == idx:
                continue
            other_code_idx = np.random.choice(len(self.global_data[other_data_idx]['labeled_codes']))
            if (other_data_idx, other_code_idx) in sampled_other_code_indices:
                continue
            other_code = self.global_data[other_data_idx]['labeled_codes'][other_code_idx]['code']
            existing_neg_codes.append(other_code)
            sampled_other_code_indices.append((other_data_idx, other_code_idx))

        if not is_all_neg:
            codes = [
                np.random.choice([x['code'] for x in self.labeled_codes[idx] if x['correctness_label']])
            ] + existing_neg_codes
        else:
            codes = existing_neg_codes
        assert len(codes) == self.contrastive_dim, len(codes)

        codes = [code + self.tokenizer.eos_token for code in codes]

        assert self.tokenizer.padding_side == 'right'
        code_tokens = self.tokenizer(codes,
                                     max_length=self.max_code_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")

        return {
            "prompts": self.prompt_token["input_ids"][idx],
            "prompt_attention_mask": self.prompt_token["attention_mask"][idx],
            "responses": code_tokens['input_ids'],
            "response_attention_mask": code_tokens['attention_mask'],
            "labels": torch.tensor(int(1 - is_all_neg), dtype=torch.long),
        }


api.data.register_dataset("wps_reward_contrastive", ExcelContrastiveRewardDataset)