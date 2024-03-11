from collections import defaultdict
from statistics import mean, stdev
import getpass
import json
import os
import time

import torch
import torch.distributed as dist

from base.network import gethostname
import base.cluster
import base.constants
import base.logging as logging

logger = logging.getLogger("Profile", "benchmark")

N_GPUS_PER_NODE = 8


def make_stats_key(host_name, op_name, global_gpu_id, target_gpu_id):
    return f"{host_name}|{op_name}|{global_gpu_id}|{target_gpu_id}"


def parse_stats_key(key):
    host_name, op_name, global_gpu_id, target_gpu_id = key.split("|")
    return host_name, op_name, global_gpu_id, target_gpu_id


class ProfileCommunication:

    def __init__(self, device_mesh_name, device, local_gpu_id, global_gpu_id, world_size):
        self.device_mesh_name = device_mesh_name
        self.device = device
        self.local_gpu_id = local_gpu_id
        self.global_gpu_id = global_gpu_id
        self.world_size = world_size
        self.stats = defaultdict(list)

        self.local_groups = []
        self.local_rank_groups = []
        self.local_group = None
        self.local_rank_group = None

        self.init_groups()

    def init_groups(self):
        assert self.world_size % N_GPUS_PER_NODE == 0
        # within host
        for i in range(self.world_size // N_GPUS_PER_NODE):
            group_start = i * N_GPUS_PER_NODE
            group_end = (i + 1) * N_GPUS_PER_NODE
            self.local_groups.append(dist.new_group(ranks=range(group_start, group_end)))

        self.local_group = self.local_groups[self.local_gpu_id // N_GPUS_PER_NODE]

        for i in range(N_GPUS_PER_NODE):
            self.local_rank_groups.append(dist.new_group(ranks=range(i, self.world_size, N_GPUS_PER_NODE)))

        self.local_rank_group = self.local_rank_groups[self.local_gpu_id]

        logger.info(f"gpu id {self.global_gpu_id} "
                    f"local rank group ranks = {dist.get_process_group_ranks(self.local_rank_group)}")

        logger.info(f"local_rank_group rank {dist.get_rank(self.local_rank_group)}")

    def profile_comm(self):
        self.profile_localhost_p2p("2GB", (1, 1024, 1024, 1024))
        self.profile_crosshost_p2p("2GB", (1, 1024, 1024, 1024))
        self.profile_offload("20MB", (20, 1024, 1024))

    def profile_localhost_p2p(self, name, tensor_shape, dtype=torch.float16):
        logger.info(f"Profiling {name} localhost p2p communication")
        send_buf = torch.rand(tensor_shape, dtype=dtype, device=self.device)
        recv_buf = torch.zeros_like(send_buf, device=self.device)

        # within host
        # 1-2, 3-4, 5-6, 7-8
        logger.info(f"Profiling {name} 1-2, 3-4, 5-6, 7-8")
        torch.cuda.synchronize()
        dist.barrier()
        st = time.monotonic_ns()
        if self.local_gpu_id % 2 == 0:
            dist.send(send_buf, self.local_gpu_id + 1, self.local_group)
            target_gpu_id = self.global_gpu_id + 1
            op_name = name + "_send"
        else:
            dist.recv(recv_buf, self.local_gpu_id - 1, self.local_group)
            target_gpu_id = self.global_gpu_id - 1
            op_name = name + "_recv"

        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id, target_gpu_id)].append(cost)

        logger.info(f"Profiling {name} reverse 1-2, 3-4, 5-6, 7-8")
        # reverse
        dist.barrier()
        st = time.monotonic_ns()
        if self.local_gpu_id % 2 == 0:
            dist.recv(recv_buf, self.local_gpu_id + 1, self.local_group)
            target_gpu_id = self.global_gpu_id + 1
            op_name = name + "_recv"
        else:
            dist.send(send_buf, self.local_gpu_id - 1, self.local_group)
            target_gpu_id = self.global_gpu_id - 1
            op_name = name + "_send"

        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id, target_gpu_id)].append(cost)

        logger.info(f"Profiling {name} 1-5, 2-6, 3-7, 4-8")
        # 1-5, 2-6, 3-7, 4-8
        dist.barrier()
        st = time.monotonic_ns()
        if self.local_gpu_id < 4:
            dist.send(send_buf, self.local_gpu_id + 4, self.local_group)
            target_gpu_id = self.global_gpu_id + 4
            op_name = name + "_send"
        else:
            dist.recv(recv_buf, self.local_gpu_id - 4, self.local_group)
            target_gpu_id = self.global_gpu_id - 4
            op_name = name + "_recv"

        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id, target_gpu_id)].append(cost)

        logger.info(f"Profiling {name} reverse 1-5, 2-6, 3-7, 4-8")
        # reverse
        dist.barrier()
        st = time.monotonic_ns()
        if self.local_gpu_id < 4:
            dist.recv(recv_buf, self.local_gpu_id + 4, self.local_group)
            target_gpu_id = self.global_gpu_id + 4
            op_name = name + "_recv"
        else:
            dist.send(send_buf, self.local_gpu_id - 4, self.local_group)
            target_gpu_id = self.global_gpu_id - 4
            op_name = name + "_send"

        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id, target_gpu_id)].append(cost)

    def profile_crosshost_p2p(self, name, tensor_shape, dtype=torch.float16):
        logger.info(f"Profiling {name} crosshost p2p communication")
        send_buf = torch.rand(tensor_shape, dtype=dtype, device=self.device)
        recv_buf = torch.zeros_like(send_buf, device=self.device)
        num_hosts = self.world_size // N_GPUS_PER_NODE
        node_rank = self.global_gpu_id // N_GPUS_PER_NODE

        if num_hosts <= 1:
            return

        logger.info(f"Profiling {name} 0-1, 2-3, ..")
        # 0-1, 2-3 ...
        torch.cuda.synchronize()
        dist.barrier()

        st = time.monotonic_ns()
        if node_rank % 2 == 0 and node_rank < num_hosts - 1:
            dist.send(send_buf, (node_rank + 1) * N_GPUS_PER_NODE + self.local_gpu_id)
            target_gpu_id = self.global_gpu_id + N_GPUS_PER_NODE
            op_name = name + "_send"
        else:
            dist.recv(recv_buf, (node_rank - 1) * N_GPUS_PER_NODE + self.local_gpu_id)
            target_gpu_id = self.global_gpu_id - N_GPUS_PER_NODE
            op_name = name + "_recv"

        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id, target_gpu_id)].append(cost)

        logger.info(f"Profiling {name} reverse 0-1, 2-3, ..")
        # reverse
        dist.barrier()
        st = time.monotonic_ns()
        if node_rank % 2 == 0 and node_rank < num_hosts - 1:
            dist.recv(recv_buf, (node_rank + 1) * N_GPUS_PER_NODE + self.local_gpu_id)
            target_gpu_id = self.global_gpu_id + N_GPUS_PER_NODE
            op_name = name + "_recv"
        else:
            dist.send(send_buf, (node_rank - 1) * N_GPUS_PER_NODE + self.local_gpu_id)
            target_gpu_id = self.global_gpu_id - N_GPUS_PER_NODE
            op_name = name + "_send"

        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id, target_gpu_id)].append(cost)

        # logger.info(f"Profiling {name} 1-2, 3-4, ...")
        # # 1-2, 3-4 ...
        # dist.barrier()
        # st = time.monotonic_ns()
        # if self.local_gpu_id % 2 == 1 and node_rank < num_hosts - 1:
        #     dist.send(send_buf, (node_rank + 1) * N_GPUS_PER_NODE + self.local_gpu_id)
        #     target_gpu_id = self.global_gpu_id + N_GPUS_PER_NODE
        #     op_name = name + "_send"
        # elif self.local_gpu_id % 2 == 0 and node_rank > 0:
        #     dist.recv(recv_buf, (node_rank - 1) * N_GPUS_PER_NODE + self.local_gpu_id)
        #     target_gpu_id = self.global_gpu_id - N_GPUS_PER_NODE
        #     op_name = name + "_recv"

        # torch.cuda.synchronize()
        # cost = time.monotonic_ns() - st
        # self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id,
        #                 target_gpu_id)].append(cost)

        # logger.info(f"Profiling {name} reverse 1-2, 3-4, ...")
        # # reverse
        # dist.barrier()
        # st = time.monotonic_ns()
        # if self.local_gpu_id % 2 == 1 and node_rank < num_hosts - 1:
        #     dist.recv(recv_buf, (node_rank + 1) * N_GPUS_PER_NODE + self.local_gpu_id)
        #     target_gpu_id = self.global_gpu_id + N_GPUS_PER_NODE
        #     op_name = name + "_recv"
        # elif self.local_gpu_id % 2 == 0 and node_rank > 0:
        #     dist.send(send_buf, (node_rank - 1) * N_GPUS_PER_NODE + self.local_gpu_id)
        #     target_gpu_id = self.global_gpu_id - N_GPUS_PER_NODE
        #     op_name = name + "_send"

        # torch.cuda.synchronize()
        # cost = time.monotonic_ns() - st
        # self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id,
        #                 target_gpu_id)].append(cost)

        # logger.info(f"Profiling {name} 0-N, N-0")
        # # 0-N N-0
        # dist.barrier()
        # st = time.monotonic_ns()
        # if node_rank == 0:
        #     dist.send(send_buf, (num_hosts - 1) * N_GPUS_PER_NODE + self.local_gpu_id)
        #     target_gpu_id = self.global_gpu_id + (num_hosts - 1) * N_GPUS_PER_NODE
        #     op_name = name + "_send"
        # elif node_rank == num_hosts - 1:
        #     dist.recv(recv_buf, self.local_gpu_id)
        #     target_gpu_id = self.global_gpu_id - (num_hosts - 1) * N_GPUS_PER_NODE
        #     op_name = name + "_recv"

        # torch.cuda.synchronize()
        # cost = time.monotonic_ns() - st
        # self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id,
        #                 target_gpu_id)].append(cost)

        # logger.info(f"Profiling {name} reverse 0-N, N-0")
        # # reverse
        # dist.barrier()
        # st = time.monotonic_ns()
        # if node_rank == 0:
        #     dist.recv(recv_buf, (num_hosts - 1) * N_GPUS_PER_NODE + self.local_gpu_id)
        #     target_gpu_id = self.global_gpu_id + (num_hosts - 1) * N_GPUS_PER_NODE
        #     op_name = name + "_recv"
        # elif node_rank == num_hosts - 1:
        #     dist.send(send_buf, self.local_gpu_id)
        #     target_gpu_id = self.global_gpu_id - (num_hosts - 1) * N_GPUS_PER_NODE
        #     op_name = name + "_send"

        # torch.cuda.synchronize()
        # cost = time.monotonic_ns() - st
        # self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id,
        #                 target_gpu_id)].append(cost)

    def profile_offload(self, name, tensor_shape, dtype=torch.float16):
        logger.info(f"Profiling {name} offload")
        dist.barrier()
        buf = torch.rand(tensor_shape, dtype=dtype, device=self.device)

        st = time.monotonic_ns()
        # buf.to("cpu")
        buf = buf.cpu()
        cost = time.monotonic_ns() - st
        op_name = name + "_store"
        self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id, "MEM")].append(cost)

        dist.barrier()
        st = time.monotonic_ns()
        # buf.to(self.device)
        buf = buf.cuda()
        cost = time.monotonic_ns() - st
        op_name = name + "_load"
        self.stats[make_stats_key(gethostname(), op_name, self.global_gpu_id, "MEM")].append(cost)

    def dump_stats(self):
        # full stats
        # dump_path = f"./profile_result/{self.device_mesh_name}/full-{gethostname()}-{self.local_gpu_id}.json"
        DUMP_DIR = os.path.join(
            base.cluster.spec.fileroot,
            "logs",
            getpass.getuser(),
            base.constants.experiment_name(),
            base.constants.trial_name(),
            "profile_result",
        )
        dump_path = os.path.join(DUMP_DIR, self.device_mesh_name,
                                 f"full-{gethostname()}-{self.local_gpu_id}.json")
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "w") as f:
            logger.info(f"dumping to {os.path.abspath(dump_path)}")
            json.dump(self.stats, f, indent=4)

        # summary
        summary = {}
        for key, value in self.stats.items():
            summary[key] = {
                "len": len(value),
                "mean": mean(value) / 1e3,
                "stdev": stdev(value) / 1e3,
                "min": min(value) / 1e3,
                "max": max(value) / 1e3
            }
        dump_path = os.path.join(DUMP_DIR, self.device_mesh_name,
                                 f"summary-{gethostname()}-{self.local_gpu_id}.json")
        with open(dump_path, "w") as f:
            json.dump(summary, f, indent=4)

    def reset_stats(self):
        self.stats = defaultdict(list)

    def print_stats(self):
        for key, value in self.stats.items():
            host_name, op_name, global_gpu_id, target_gpu_id = parse_stats_key(key)
            logger.info(
                f"host {host_name} gpu {global_gpu_id} {op_name} to {target_gpu_id} cost {mean(value):.2f} ns"
            )
