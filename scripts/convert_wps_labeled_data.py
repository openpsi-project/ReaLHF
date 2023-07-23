from typing import Literal
import collections
import json
import os
import re
import subprocess

import numpy as np


def labeled2code_and_result(labeled_data):

    def get_label_fn(d):
        return d['label']

    labeled_data = [d for d in labeled_data if 'label' in d and get_label_fn(d) in {0, 1}]
    assert set([get_label_fn(d) for d in labeled_data]) == {0,
                                                            1}, set([get_label_fn(d) for d in labeled_data])
    prompt2infresult = collections.defaultdict(list)
    invalid_cnt = 0
    for d in labeled_data:
        key = (d['head_cn'], d['task_cn'])
        if get_label_fn(d) == 0:
            try:
                assert d['exec_result_name'] == 'RunSuccess', d['exec_result_name']
            except AssertionError:
                invalid_cnt += 1
                continue
        prompt2infresult[key].append(
            (d['code'], bool(d['exec_result_name'] == 'RunSuccess'), bool(get_label_fn(d) == 0)))
    print(f"The number of unique prompts (head+task):", len(prompt2infresult))
    print(f"number of invalid data: {invalid_cnt}")
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
            dict(head=head, task=task, code=x[0], correctness_label=int(x[criterion_idx]))
            for x in inf_results
        ]
        n_pos += sum([x[criterion_idx] for x in inf_results])
    print(f"Number of data: {len(data)}")
    return data


if __name__ == "__main__":
    train_proportion = 0.9

    raw_file_name = "/data/aigc/public/wps-excel/json/starcoder_compile_5000_flatten_labels0625.json"
    with open(raw_file_name, "r") as f:
        data = json.load(f)
    print(f"Raw dataset size: {len(data)}")

    prompt2infresult = labeled2code_and_result(data)
    data = make_data(prompt2infresult, criterion='label', max_n_labels=10)

    fn = "tmp.json"
    with open(fn, "w") as f:
        json.dump(data, f)
    subprocess.check_output(["python3", "-m", "scripts.wash_head", "--input", fn, '--output', 'tmp_.json'])

    with open('tmp_.json', "r") as fin:
        data = json.load(fin)

    os.system("rm tmp.json tmp_.json")

    output_root_dir = "/data/aigc/llm/fw/datasets/rw-unpaired/"
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

    print(f"Number of train data: {len(train_data)}, number of validation data: {len(valid_data)}.")
    n_pos = sum([x['correctness_label'] for x in train_data])
    print(f"Number of positive samples: {n_pos}/{len(train_data)}")