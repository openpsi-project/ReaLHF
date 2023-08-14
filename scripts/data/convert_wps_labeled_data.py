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
        existing_codes = [x[0] for x in prompt2infresult[key]]
        if any(d['code'].strip() in c.strip() or c.strip() in d['code'] for c in existing_codes):
            continue
        prompt2infresult[key].append(
            (d['code'], bool(d['exec_result_name'] == 'RunSuccess'), bool(get_label_fn(d) == 0)))
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


def cross_task_code_augmentation(data):
    augment_size = len(data) // 2
    augment_record = set()
    augment_data = []
    while len(augment_data) < augment_size:
        idx1, idx2 = np.random.randint(0, len(data), (2,)).tolist()
        if idx1 > idx2:
            idx2, idx1 = idx1, idx2
        if (idx1, idx2) in augment_record:
            continue
        if data[idx1]['head'] == data[idx2]['head'] or data[idx1]['task'] == data[idx2]['task']:
            continue
        augment_record.add((idx1, idx2))
        augment_data += [
            dict(head=data[idx1]['head'],
                 task=data[idx1]['task'],
                 code=data[idx2]['code'],
                 correctness_label=0),
            dict(head=data[idx2]['head'],
                 task=data[idx2]['task'],
                 code=data[idx1]['code'],
                 correctness_label=0)
        ]
    return data + augment_data


RUBBISH_CODE_COLLECTIONS = [
    "const sheet = Application.ActiveSheet;\nconst usedRange = sheet.UsedRange;\nconst rowCount = usedRange.Rows.Count;",
    "const sheet = Application.ActiveSheet\nconst usedRange = sheet.UsedRange\nconst rowCount = usedRange.Rows.Count",
    "const sheet = Application.ActiveSheet\nconst usedRange = sheet.UsedRange\nconst rowCount = usedRange.Rows.Count\nfor(let i = usedRange.Row + 1; i <= rowCount; i++) {\n}",
]


def rubbish_compilable_code_augmentation(head_tasks, data):
    augment_size = 1000
    augment_data = []
    indices = np.random.choice(len(head_tasks), augment_size, replace=False)
    for idx in indices:
        head, task = head_tasks[idx]
        augment_data.append(
            dict(
                head=head,
                task=task,
                code=np.random.choice(RUBBISH_CODE_COLLECTIONS),
                correctness_label=0,
            ),)
    return data


if __name__ == "__main__":
    train_proportion = 0.9

    raw_file_name = "/data/aigc/llm/raw/starcoder_compile_5000_flatten_labels0625.json"
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

    train_data = make_data({k: prompt2infresult[k]
                            for k in train_prompts},
                           criterion='label',
                           max_n_labels=10)
    valid_data = make_data({k: prompt2infresult[k]
                            for k in valid_prompts},
                           criterion='label',
                           max_n_labels=10)

    # data = cross_task_code_augmentation(data)
    # data = rubbish_compilable_code_augmentation(list(prompt2infresult.keys()), data)

    fn = "tmp.json"
    with open(fn, "w") as f:
        json.dump(train_data, f)
    subprocess.check_output(["python3", "-m", "scripts.wash_head", "--input", fn, '--output', 'tmp_.json'])
    with open('tmp_.json', "r") as fin:
        train_data = json.load(fin)
    os.system("rm tmp.json tmp_.json")

    fn = "tmp.json"
    with open(fn, "w") as f:
        json.dump(valid_data, f)
    subprocess.check_output(["python3", "-m", "scripts.wash_head", "--input", fn, '--output', 'tmp_.json'])
    with open('tmp_.json', "r") as fin:
        valid_data = json.load(fin)
    os.system("rm tmp.json tmp_.json")

    output_root_dir = "/data/aigc/llm/datasets/rw-unpaired/"
    os.makedirs(output_root_dir, exist_ok=True)

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
    n_neg = len(train_data) - n_pos
    pos_weight = n_neg / n_pos
    print(f"Number of positive samples: {n_pos}/{len(train_data)}, pos weight {pos_weight}")