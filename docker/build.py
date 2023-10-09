import argparse
import os
import socket
import time

LATEST_TV = "23.08"

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action='store_true')
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--retry", type=int, default=100)
parser.add_argument("--tv", type=str, default=LATEST_TV)
args = parser.parse_args()

if __name__ == "__main__":
    assert args.gpu or args.cpu
    assert args.gpu != args.cpu
    diip = "10.122.2.14" if 'ctrl' not in socket.gethostname() else "10.210.14.10"
    identi = "cpu" if args.cpu else "gpu"
    if args.tv != LATEST_TV:
        identi += f":{args.tv}"
    target_img = f"{diip}:5000/llm/llm-{identi}"
    for _ in range(args.retry):
        cmd = (f"docker build -t {target_img} -f docker/Dockerfile")
        if args.gpu:
            cmd += f" --build-arg base_image=nvcr.io/nvidia/pytorch:{args.tv}-py3"
        else:
            cmd += f" --build-arg base_image=ubuntu:22.04"
        cmd += " /local/fw/packages/docker"
        print(cmd)
        os.system(cmd)
        time.sleep(1)
    os.system(f"docker push {target_img}")
