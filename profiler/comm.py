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
        self.adhoc_network_test()
        # self.profile_localhost_p2p("2GB", (1, 1024, 1024, 1024), dtype=torch.float16)
        # self.profile_crosshost_p2p("2GB", (1, 1024, 1024, 1024), dtype=torch.float16)
        self.profile_offload("20MB", (10, 1024, 1024), dtype=torch.float16)

        # self.profile_all_gather("200MB", (100, 1024, 1024), dtype=torch.float16)
        # self.profile_all_reduce("200MB", (100, 1024, 1024), dtype=torch.float16)
        # self.profile_broadcast("200MB", (100, 1024, 1024), dtype=torch.float16)

    def profile_offload(self, name, tensor_shape, dtype=torch.float16):
        logger.info(f"Profiling {name} offload")
        dist.barrier()
        buf = torch.rand(tensor_shape, dtype=dtype, device=self.device)

        st = time.monotonic_ns()
        # buf.to("cpu")
        buf = buf.cpu()
        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        op_name = "offload_store"
        self.stats[op_name].append(cost)

        dist.barrier()
        st = time.monotonic_ns()
        # buf.to(self.device)
        buf = buf.cuda()
        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        op_name = "offload_load"
        self.stats[op_name].append(cost)

    def adhoc_network_test(self):
        # assert self.world_size >= 16
        tensor_shape = (1024, 1024, 1024)
        dtype = torch.float16

        send_buf = torch.rand(tensor_shape, dtype=dtype, device=self.device)
        recv_buf = torch.zeros_like(send_buf, device=self.device)

        for _ in range(10):
            st = time.monotonic_ns()
            if self.global_gpu_id % N_GPUS_PER_NODE == 0:
                dist.send(send_buf, self.global_gpu_id + 1)
            elif self.global_gpu_id % N_GPUS_PER_NODE == 1:
                dist.recv(recv_buf, self.global_gpu_id - 1)

            if self.global_gpu_id in [0, 1]:
                torch.cuda.synchronize()
                cost = time.monotonic_ns() - st
                # speed = int(2 * 1024 * 1024 * 1024 * 1e3/cost) # bytes/micro seconds
                op_name = "local_send" if self.global_gpu_id == 0 else "local_recv"
                self.stats[op_name].append(cost)
            dist.barrier()

        if self.world_size >= 16:
            for _ in range(10):
                st = time.monotonic_ns()
                if self.global_gpu_id % N_GPUS_PER_NODE == 0:
                    is_even = (self.global_gpu_id // N_GPUS_PER_NODE) % 2 == 0
                    if is_even:
                        dist.send(send_buf, self.global_gpu_id + 8)
                    else:
                        dist.recv(recv_buf, self.global_gpu_id - 8)

                    torch.cuda.synchronize()
                    cost = time.monotonic_ns() - st
                    op_name = "remote_send" if is_even else "remote_recv"
                    self.stats[op_name].append(cost)
                dist.barrier()

        print("adhoc network test done")

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
            logger.info(f"{key} cost {mean(value)/1e3:.2f} micro seconds.")
