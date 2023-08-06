import json
import os
import pickle
import subprocess

import numpy as np
import torch
import tqdm
import transformers

from scripts.data.utils import IMPOSSIBLE_TASKS, longest_common_substring, jaro_Winkler


@torch.inference_mode()
def get_prompt_embedding(prompts, model_name_or_path: str):
    import api.utils
    print("Loading model...")
    tokenizer = api.utils.load_hf_tokenizer(model_name_or_path)
    model = api.utils.create_hf_nn(transformers.AutoModelForCausalLM,
                                   model_name_or_path)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    all_embeds = []
    for prompt in tqdm.tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        embed = model(**inputs, output_hidden_states=True).hidden_states[0][0, -1]
        assert len(embed.shape) == 1
        all_embeds.append(embed.cpu().numpy().tolist())
    return all_embeds


def filter_prompts(data):
    import api.utils
    tokenizer = api.utils.load_hf_tokenizer("/home/aigc/llm/checkpoints/starcoder-wps-best/")

    prompts = [d['starcoder']['prompt'] for d in data]
    tasks = [d['task_cn'] for d in data]
    heads = [d['head_cn'] for d in data]
    all_codes = [[x['code'] for x in d['starcoder']['inference_result']] for d in data]

    filtered_data = set()
    duplicate_tasks = set()
    for prompt, head, task, codes in tqdm.tqdm(list(zip(prompts, heads, tasks, all_codes))):
        # if the reference code is too long, throw it away
        tokens = tokenizer([code + tokenizer.eos_token for code in codes])
        lengths = [len(x) for x in tokens['input_ids']]
        if sum(lengths) / len(lengths) > 256:
            continue

        key = (head, task)

        # filter out impossible tasks like sending emails
        if any(t in task for t in IMPOSSIBLE_TASKS):
            continue

        # if the longest common substring of task is larger than 12, skip
        if key not in filtered_data and key[1] not in duplicate_tasks:
            flag = False
            for other_key in filtered_data:
                jw_score = jaro_Winkler(key[1], other_key[1])
                if jw_score > 0.8:
                    flag = True
                sub_s = longest_common_substring(key[1], other_key[1])
                if len(sub_s) >= 15:
                    flag = True

                if flag:
                    print(f"Task {key[1]} and {other_key[1]} are considered duplicate. "
                          f"The longest common sub-string (length {len(sub_s)}, score {jw_score}): {sub_s}.")
                    duplicate_tasks.add(key[1])
                    break
            if flag:
                continue

        filtered_data.add(key)
    return [dict(task=key[1], head=key[0]) for key in filtered_data]


if __name__ == "__main__":
    with open("/home/aigc/llm/raw/starcoder-inference-300k-4.json", 'r') as f:
        raw_data = json.load(f)[:30000]
    np.random.shuffle(raw_data)
    # prompts = [d['starcoder']['prompt'] for d in data]
    # TODO: filter prompts with clustering
    # embeds = get_prompt_embedding(prompts, "/data/marl/checkpoints/fw/starcoder-wps-best/")
    # with open("/data/aigc/public/wps-excel/starcoder-inference-300k-4-prompt-embeds.pkl", 'wb') as f:
    #     pickle.dump(embeds, f)
    # with open("/data/aigc/public/wps-excel/starcoder-inference-300k-4-prompt-embeds.pkl", 'rb') as f:
    #     embeds = pickle.load(f)
    data = filter_prompts(raw_data)

    fn = "tmp.json"
    with open(fn, "w") as f:
        json.dump(data, f)
    subprocess.check_output(
        ["python3", "-m", "scripts.data.wash_head", "--input", fn, '--output', 'tmp_.json'])
    with open('tmp_.json', "r") as fin:
        data = json.load(fin)
    os.system("rm tmp.json tmp_.json")

    train_proportion = 0.9
    output_root_dir = "/home/aigc/llm/datasets/prompts/"
    os.makedirs(output_root_dir, exist_ok=True)
    np.random.shuffle(data)

    n_train = int(len(data) * train_proportion)
    train_data = data[:n_train]
    valid_data = data[n_train:]

    with open(os.path.join(output_root_dir, 'train.jsonl'), "w") as fout:
        for d in train_data:
            json.dump(d, fout, ensure_ascii=False)
            fout.write("\n")

    with open(os.path.join(output_root_dir, 'valid.jsonl'), "w") as fout:
        for d in valid_data:
            json.dump(d, fout, ensure_ascii=False)
            fout.write("\n")
    print(f"Raw data size: {len(raw_data)}")
    print(f"Number of train data: {len(train_data)}, number of validation data: {len(valid_data)}.")
