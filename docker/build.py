import argparse
import os
import socket
import time

from base.cluster import spec as cluster_spec

LATEST_TV = "23.10"

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--retry", type=int, default=100)
parser.add_argument("--tv", type=str, default=LATEST_TV)
parser.add_argument("--rebuild", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    assert args.gpu or args.cpu
    assert args.gpu != args.cpu
    if cluster_spec.name == "qizhi":
        diip = "10.122.2.14"
    elif cluster_spec.name == "qh":
        diip = "10.119.12.14"
    elif cluster_spec.name == "yl":
        diip = "10.122.2.14"
    else:
        raise NotImplementedError()
    identi = "cpu" if args.cpu else "gpu"
    if args.tv != LATEST_TV:
        identi += f":{args.tv}"
    target_img = f"{diip}:5000/llm/llm-{identi}"
    for _ in range(args.retry):
        cmd = f"docker build -t {target_img} -f docker/Dockerfile"
        if args.gpu:
            base_image = f"nvcr.io/nvidia/pytorch:{args.tv}-py3"
            cmd += (f" --build-arg base_image={base_image} "
                    f"--build-arg INCUBATOR_VER={time.time()}")
        else:
            cmd += f".cpu --build-arg base_image=ubuntu:22.04"
        if args.rebuild:
            cmd += " --no-cache "
        cmd += " ."
        print(cmd)
        os.system(cmd)
        time.sleep(1)
    os.system(f"docker push {target_img}")
