import argparse
import os
import time
import socket

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action='store_true')
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    assert args.gpu or args.cpu
    assert args.gpu != args.cpu
    diip = "10.122.2.14" if 'ctrl' not in socket.gethostname() else "10.210.14.10"
    identi = "cpu" if args.cpu else "gpu"
    for _ in range(100):
        cmd = (f"docker build -t {diip}:5000/llm/llm-{identi} "
               f"-f docker/Dockerfile.{identi}")
        if args.gpu and 'ctrl' in socket.gethostname():
            cmd += " --build-arg base_image=nvcr.io/nvidia/pytorch:22.04-py3"
        cmd += " /local/fw/packages/docker"
        print(cmd)
        os.system(cmd)
        time.sleep(1)
    os.system(f"docker push {diip}:5000/llm/llm-{identi}")
