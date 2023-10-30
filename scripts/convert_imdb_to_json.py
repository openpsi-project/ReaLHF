import json
import os

import numpy as np
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("/lustre/fw/pretrained/gpt2-large")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
max_seq_len = 4096
prompt_len = 10

# train set postive samples: tokens min 16 max 3097 avg 303
# test set positive samples: tokens min 15 max 2972 avg 293
# => set prompt length to 10

seed = 42
rng = np.random.RandomState(seed=seed)

sft_ratio, rw_ratio, ppo_ratio = 0.4, 0.4, 0.2
assert sft_ratio + rw_ratio + ppo_ratio == 1
train_ratio, valid_ratio = 0.8, 0.2

n_pos = len(os.listdir("/lustre/fw/datasets/imdb/aclImdb/train/pos"))
pos_indices = np.arange(n_pos)
rng.shuffle(pos_indices)
sft_pos_indices = pos_indices[:int(n_pos * sft_ratio)]
rw_pos_indices = pos_indices[int(n_pos * sft_ratio):int(n_pos * (sft_ratio + rw_ratio))]
ppo_pos_indices = pos_indices[int(n_pos * (sft_ratio + rw_ratio)):]

n_neg = len(os.listdir("/lustre/fw/datasets/imdb/aclImdb/train/neg"))
neg_indices = np.arange(n_neg)
rng.shuffle(neg_indices)
sft_neg_indices = neg_indices[:int(n_neg * sft_ratio)]
rw_neg_indices = neg_indices[int(n_neg * sft_ratio):int(n_neg * (sft_ratio + rw_ratio))]
ppo_neg_indices = neg_indices[int(n_neg * (sft_ratio + rw_ratio)):]

print("Loading texts...")
texts = []
for split in ['train']:
    for label in ['pos', 'neg']:
        root = f"/lustre/fw/datasets/imdb/aclImdb/{split}/{label}"
        text_files = os.listdir(root)
        for f in text_files:
            idx = int(f.split('_')[0])
            assert 0 <= idx < len(text_files)
            with open(os.path.join(root, f), 'r') as fin:
                texts.append(fin.read())

print("Start tokenizing...")
tokenizer.padding_side = 'right'
encodings_ = tokenizer(
    texts,
    max_length=max_seq_len,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
    return_length=True,
)

prompts_ = tokenizer.batch_decode(encodings_['input_ids'][:, :prompt_len], skip_special_tokens=True)
answers_ = tokenizer.batch_decode(encodings_['input_ids'][:, prompt_len:], skip_special_tokens=True)

pos_prompts, pos_answers = prompts_[:n_pos], answers_[:n_pos]
neg_prompts, neg_answers = prompts_[n_pos:], answers_[n_pos:]
assert len(neg_prompts) == n_neg == len(neg_answers)

print("Creating pos neg sft data...")
prompts = [pos_prompts[idx] for idx in sft_pos_indices] + [neg_prompts[idx] for idx in sft_neg_indices]
answers = [pos_answers[idx] for idx in sft_pos_indices] + [neg_answers[idx] for idx in sft_neg_indices]
assert len(prompts) == len(answers)
prompt_indices = np.arange(len(prompts))
rng.shuffle(prompt_indices)
data = [dict(prompt=prompts[idx], answer=answers[idx]) for idx in prompt_indices]
with open(f"/lustre/fw/datasets/imdb/rl/sft_pos_neg-all.jsonl", 'w') as f:
    for x in data:
        json.dump(x, f, ensure_ascii=False)
        f.write('\n')
with open(f"/lustre/fw/datasets/imdb/rl/sft_pos_neg-train.jsonl", 'w') as f:
    for x in data[:int(len(data) * train_ratio)]:
        json.dump(x, f, ensure_ascii=False)
        f.write('\n')
with open(f"/lustre/fw/datasets/imdb/rl/sft_pos_neg-valid.jsonl", 'w') as f:
    for x in data[int(len(data) * train_ratio):]:
        json.dump(x, f, ensure_ascii=False)
        f.write('\n')

print("Creating pos sft data...")
prompts = [pos_prompts[idx] for idx in sft_pos_indices]
answers = [pos_answers[idx] for idx in sft_pos_indices]
assert len(prompts) == len(answers)
prompt_indices = np.arange(len(prompts))
rng.shuffle(prompt_indices)
data = [dict(prompt=prompts[idx], answer=answers[idx]) for idx in prompt_indices]
with open(f"/lustre/fw/datasets/imdb/rl/sft_pos-all.jsonl", 'w') as f:
    for x in data:
        json.dump(x, f, ensure_ascii=False)
        f.write('\n')
with open(f"/lustre/fw/datasets/imdb/rl/sft_pos-train.jsonl", 'w') as f:
    for x in data[:int(len(data) * train_ratio)]:
        json.dump(x, f, ensure_ascii=False)
        f.write('\n')
with open(f"/lustre/fw/datasets/imdb/rl/sft_pos-valid.jsonl", 'w') as f:
    for x in data[int(len(data) * train_ratio):]:
        json.dump(x, f, ensure_ascii=False)
        f.write('\n')

print("Creating rw prompts...")
prompts = [pos_prompts[idx] for idx in rw_pos_indices] + [neg_prompts[idx] for idx in rw_neg_indices]
prompt_indices = np.arange(len(prompts))
rng.shuffle(prompt_indices)
data = [dict(prompt=prompts[idx]) for idx in prompt_indices]
with open(f"/lustre/fw/datasets/imdb/rl/rw_prompt-all.jsonl", 'w') as f:
    for x in data:
        json.dump(x, f, ensure_ascii=False)
        f.write('\n')
with open(f"/lustre/fw/datasets/imdb/rl/rw_prompt-train.jsonl", 'w') as f:
    for x in data[:int(len(data) * train_ratio)]:
        json.dump(x, f, ensure_ascii=False)
        f.write('\n')
with open(f"/lustre/fw/datasets/imdb/rl/rw_prompt-valid.jsonl", 'w') as f:
    for x in data[int(len(data) * train_ratio):]:
        json.dump(x, f, ensure_ascii=False)
        f.write('\n')

print("Creating rl prompts...")
prompts = [pos_prompts[idx] for idx in ppo_pos_indices] + [neg_prompts[idx] for idx in ppo_neg_indices]
prompt_indices = np.arange(len(prompts))
rng.shuffle(prompt_indices)
data = [dict(prompt=prompts[idx]) for idx in prompt_indices]
with open(f"/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl", 'w') as f:
    for x in data:
        json.dump(x, f, ensure_ascii=False)
        f.write('\n')
