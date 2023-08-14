import argparse
import json
import random


def iter_json_data(args):
    with open(args.file, "r") as f:
        data = json.load(f)
        print(f"------********-------\nWe have {len(data)} training data\n------********-------\n")
    while True:
        # print(f"------Example -------")
        d = random.choice(data)
        print(f"Keys: {d.keys()}")
        for k, v in d.items():
            print(k + ":")
            print(v)
            print("")
        input()


def iter_jsonl_data(args):
    with open(args.file, "r") as f:
        _data_bytes = [ff for ff in f]
        print(f"------********-------\nWe have {len(_data_bytes)} training data\n------********-------\n")
        while True:
            d = json.loads(random.choice(_data_bytes))
            print(f"Keys: {d.keys()}")
            for k, v in d.items():
                print(k + ":")
                print(v)
                print("")
            input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",
                        "-f",
                        type=str,
                        required=False,
                        default="/data/aigc/llm/datasets/prompts/train.jsonl")
    args = parser.parse_args()
    if args.file.endswith(".json"):
        iter_json_data(args=args)
    elif args.file.endswith(".jsonl"):
        iter_jsonl_data(args)
