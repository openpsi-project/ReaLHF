import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action='store_true')
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    assert args.gpu or args.cpu
    assert args.gpu != args.cpu
    while True:
        if args.gpu:
            os.system("docker build -t 10.122.2.14:5000/llm/llm-gpu "
                      "-f docker/Dockerfile.gpu /local/fw/packages/docker")
        elif args.cpu:
            os.system("docker build -t 10.122.2.14:5000/llm/llm-cpu "
                      "-f docker/Dockerfile.cpu /local/fw/packages/docker")
        else:
            raise NotImplementedError()
        time.sleep(5)
