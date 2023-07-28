import json
import os
import pickle
import subprocess

import numpy as np
import torch
import tqdm
import transformers


@torch.inference_mode()
def get_prompt_embedding(prompts, model_name_or_path: str):
    import api.utils
    print("Loading model...")
    tokenizer = api.utils.load_hf_tokenizer(model_name_or_path)
    model = api.utils.create_hf_nn(transformers.AutoModelForCausalLM,
                                   model_name_or_path,
                                   disable_dropout=True)
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


if __name__ == "__main__":
    with open("/home/aigc/llm/raw/starcoder-inference-300k-4.json", 'r') as f:
        data = json.load(f)
    prompts = [d['starcoder']['prompt'] for d in data]
    # TODO: filter prompts with clustering
    # embeds = get_prompt_embedding(prompts, "/data/marl/checkpoints/fw/starcoder-wps-best/")
    # with open("/data/aigc/public/wps-excel/starcoder-inference-300k-4-prompt-embeds.pkl", 'wb') as f:
    #     pickle.dump(embeds, f)
    # with open("/data/aigc/public/wps-excel/starcoder-inference-300k-4-prompt-embeds.pkl", 'rb') as f:
    #     embeds = pickle.load(f)
    data = [dict(task=d['task_cn'], head=d['head_cn']) for d in data]

    fn = "tmp.json"
    with open(fn, "w") as f:
        json.dump(data, f)
    subprocess.check_output(["python3", "-m", "scripts.wash_head", "--input", fn, '--output', 'tmp_.json'])
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

    with open(os.path.join(output_root_dir, 'train50000.jsonl'), "w") as fout:
        for d in train_data[:50000]:
            json.dump(d, fout, ensure_ascii=False)
            fout.write("\n")

    with open(os.path.join(output_root_dir, 'valid.jsonl'), "w") as fout:
        for d in valid_data:
            json.dump(d, fout, ensure_ascii=False)
            fout.write("\n")
    print(f"Number of train data: {len(train_data)}, number of validation data: {len(valid_data)}.")