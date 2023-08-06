import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action='store_true')
args = parser.parse_args()

while True:
    if args.gpu:
        os.system("docker build -t llm-gpu -f Dockerfile.gpu .")
    else:
        os.system("docker build -t llm-cpu -f Dockerfile.cpu .")
    time.sleep(10)
