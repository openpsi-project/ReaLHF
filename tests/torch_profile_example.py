from torch.profiler import profile, ProfilerActivity, record_function
import torch


def main():
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')
    print(a.data_ptr())
    idx = [i for i in range(500, 1000)]
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True) as prof:
        c = a.mm(b)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


if __name__ == "__main__":
    main()
