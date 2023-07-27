import argparse
import json
import random


def iter_data(args):
    with open(args.file, "r") as f:
        data = json.load(f)
        print(f"------********-------\nWe have {len(data)} training data\n------********-------\n")
        # print(f"------Example -------")
        d = random.choice(data)
        print(f"Keys: {d.keys()}")
        for k, v in d.items():
            print(k + ":")
            print(v)
            print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",
                        "-f",
                        type=str,
                        required=False,
                        default="/home/aigc/llm/raw/starcoder_compile_5000_flatten_labels0625.json")
    args = parser.parse_args()
    iter_data(args=args)
