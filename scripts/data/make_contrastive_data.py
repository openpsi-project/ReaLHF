from typing import Literal
import collections
import json
import os
import re
import subprocess

import numpy as np
import tqdm

from scripts.data.utils import IMPOSSIBLE_TASKS, jaro_Winkler, longest_common_substring


def labeled2code_and_result(labeled_data):

    def get_label_fn(d):
        return d['label']

    labeled_data = [d for d in labeled_data if 'label' in d and get_label_fn(d) in {0, 1}]
    assert set([get_label_fn(d) for d in labeled_data]) == {0,
                                                            1}, set([get_label_fn(d) for d in labeled_data])
    prompt2infresult = collections.defaultdict(list)
    duplicate_tasks = set()
    invalid_cnt = 0
    for d in tqdm.tqdm(labeled_data):
        key = (d['head_cn'], d['task_cn'])

        if any(t in key[1] for t in IMPOSSIBLE_TASKS):
            continue

        # if the longest common substring of task is larger than 5, skip
        if key not in prompt2infresult and key[1] not in duplicate_tasks:
            flag = False
            for other_key in prompt2infresult:
                sub_s = longest_common_substring(key[1], other_key[1])
                if len(sub_s) >= 12:
                    print(f"Task {key[1]} and {other_key[1]} are considered duplicate. "
                          f"The longest common sub-string (length {len(sub_s)}): {sub_s}.")
                    duplicate_tasks.add(key[1])
                    flag = True
                    break
            if flag:
                continue

        if get_label_fn(d) == 0:
            try:
                assert d['exec_result_name'] == 'RunSuccess', d['exec_result_name']
            except AssertionError:
                invalid_cnt += 1
                continue
        existing_codes = [x[0] for x in prompt2infresult[key]]
        if len(d['code'].strip()) == 0:
            continue
        if any((d['code'].strip() in c.strip() or c.strip() in d['code']) for c in existing_codes):
            continue
        assert len(d['code'].strip()) > 0, d['code'].strip()

        prompt2infresult[key].append(
            (d['code'].strip(), bool(d['exec_result_name'] == 'RunSuccess'), bool(get_label_fn(d) == 0)))
    print(f"The number of unique prompts (head+task):", len(prompt2infresult))
    print(f"number of invalid data: {invalid_cnt}")
    prompt2infresult = dict(prompt2infresult)
    atleast_one_pos_cnt = sum([any(v[-1] for v in vv) for vv in prompt2infresult.values()])
    print(f"Number of prompts with at least one positive code: {atleast_one_pos_cnt}")
    return dict(prompt2infresult)


def make_data(
    prompt2infresult,
    max_n_labels: int = 100,
    criterion: Literal['compile', 'label'] = 'label',
):
    assert criterion in {'compile', 'label'}
    data = []
    n_pos = 0
    criterion_idx = 2 if criterion == 'label' else 1
    for (head, task), inf_results in prompt2infresult.items():
        inf_results = inf_results[:max_n_labels]
        data += [
            dict(head=head, task=task, code=x[0].strip(), correctness_label=int(x[criterion_idx]))
            for x in inf_results
        ]
        n_pos += sum([x[criterion_idx] for x in inf_results])
    print(f"Number of data: {len(data)}")
    return data


if __name__ == "__main__":
    train_proportion = 0.9

    raw_file_name = "/home/aigc/llm/raw/starcoder_compile_5000_flatten_labels0625.json"
    with open(raw_file_name, "r") as f:
        raw_data = json.load(f)
        np.random.shuffle(raw_data)
    print(f"Raw dataset size: {len(raw_data)}")

    prompt2infresult = labeled2code_and_result(raw_data)
    prompts = list(prompt2infresult.keys())
    np.random.shuffle(prompts)
    n_train = int(len(prompts) * train_proportion)
    train_prompts = prompts[:n_train]
    valid_prompts = prompts[n_train:]

    train_data = []
    for head, task in train_prompts:
        labeled_codes = []
        for (code, compilable, correct) in prompt2infresult[(head, task)]:
            labeled_codes.append(dict(code=code, correctness_label=correct))
        train_data.append(dict(head=head, task=task, labeled_codes=labeled_codes))
    valid_data = []
    for head, task in valid_prompts:
        labeled_codes = []
        for (code, compilable, correct) in prompt2infresult[(head, task)]:
            labeled_codes.append(dict(code=code, correctness_label=correct))
        valid_data.append(dict(head=head, task=task, labeled_codes=labeled_codes))

    labeled_codes_len = [len(x['labeled_codes']) for x in train_data + valid_data]
    print(f"Average number of labeld codes: {np.mean(labeled_codes_len)}, "
          f"min {np.min(labeled_codes_len)}, max {np.max(labeled_codes_len)}")
    n_pos_only = sum(
        [all(y['correctness_label'] for y in x['labeled_codes']) for x in train_data + valid_data])
    n_neg_only = sum(
        [all((not y['correctness_label']) for y in x['labeled_codes']) for x in train_data + valid_data])
    print(f"Number of positive only data: {n_pos_only}, "
          f"number of negative only data: {n_neg_only}, "
          f"others: {len(labeled_codes_len) - n_pos_only - n_neg_only}")

    fn = "tmp.json"
    with open(fn, "w") as f:
        json.dump(train_data, f)
    subprocess.check_output(
        ["python3", "-m", "scripts.data.wash_head", "--input", fn, '--output', 'tmp_.json'])
    with open('tmp_.json', "r") as fin:
        train_data = json.load(fin)
    os.system("rm tmp.json tmp_.json")

    fn = "tmp.json"
    with open(fn, "w") as f:
        json.dump(valid_data, f)
    subprocess.check_output(
        ["python3", "-m", "scripts.data.wash_head", "--input", fn, '--output', 'tmp_.json'])
    with open('tmp_.json', "r") as fin:
        valid_data = json.load(fin)
    os.system("rm tmp.json tmp_.json")

    output_root_dir = "/home/aigc/llm/datasets/rw-contrastive/"
    os.makedirs(output_root_dir, exist_ok=True)

    with open(os.path.join(output_root_dir, 'train.jsonl'), "w") as fout:
        for d in train_data:
            json.dump(d, fout, ensure_ascii=False)
            fout.write("\n")

    with open(os.path.join(output_root_dir, 'train-small.jsonl'), "w") as fout:
        for d in train_data[:32]:
            json.dump(d, fout, ensure_ascii=False)
            fout.write("\n")

    with open(os.path.join(output_root_dir, 'valid.jsonl'), "w") as fout:
        for d in valid_data:
            json.dump(d, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Number of train data: {len(train_data)}, number of validation data: {len(valid_data)}.")
