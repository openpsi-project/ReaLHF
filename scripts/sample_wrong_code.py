import collections
import json
import pickle
import random

import zmq


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


def get_prompt_for_wrong_code(head, task):
    return f"我的表格格式是：{head}，请写一段JavaScript代码完成下述任务：{task}。不正确的代码如下：\nIncorrect Answer: "


if __name__ == "__main__":

    raw_file_name = "/data/aigc/public/wps-excel/json/starcoder_compile_5000_flatten_labels0625.json"
    with open(raw_file_name, "r") as f:
        data = json.load(f)
    print(f"Raw dataset size: {len(data)}")

    prompt2infresult = labeled2code_and_result(data)
    prompts = list([k for k in prompt2infresult.keys() if any(v[-1] for v in prompt2infresult[k])])
    head, task = random.choice(prompts)

    prompt = get_prompt_for_wrong_code(head, task)

    socket = zmq.Context().socket(zmq.REQ)
    socket.connect("tcp://localhost:7777")

    socket.send(pickle.dumps(prompt))
    result = pickle.loads(socket.recv())

    print(result)
    print(">>>>>>>>>>>>>>>>")
    for code, _, correct in prompt2infresult[(head, task)]:
        if correct:
            print(code)
