import getpass
import os

print(os.environ['CUDA_VISIBLE_DEVICES'])
PYTORCH_KERNEL_CACHE_PATH = f"/data/aigc/llm/.cache/{getpass.getuser()}/torch/kernels"
TRITON_CACHE_PATH = f"/data/aigc/llm/.cache/{getpass.getuser()}/triton"
DATASET_CACHE_PATH = f'/data/aigc/llm/.cache/{getpass.getuser()}/datasets'
TORCH_EXTENSIONS_DIR = f"/data/aigc/llm/.cache/{getpass.getuser()}/torch/extensions"
_LLM_ENVVARS = {
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTORCH_KERNEL_CACHE_PATH": PYTORCH_KERNEL_CACHE_PATH,
    "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
    "TOKENIZERS_PARALLELISM": "true",
    "TORCH_EXTENSIONS_DIR": TORCH_EXTENSIONS_DIR,
    "CUDA_LAUNCH_BLOCKING": "1",
    "TORCH_USE_CUDA_DSA": "1",
}
for k, v in _LLM_ENVVARS.items():
    os.environ[k] = v

import deepspeed
import torch
import transformers

model_name = "test"
world_size = 16
local_gpu_id = int(os.environ["SLURM_PROCID"]) % 8
ddp_rank = int(os.environ["SLURM_PROCID"])
ddp_init_address = "tcp://10.122.2.11:7777"
torch_dist_kwargs = dict(world_size=world_size, rank=ddp_rank, init_method=ddp_init_address, backend='nccl')

# Set local rank and make all devices visible. These variables are used by DeepSpeed.
os.environ['CUDA_VISIBLE_DEVICES'] = str(local_gpu_id)
torch.cuda.set_device(0)  # initialize CUDA here with only a single visible device
os.environ['LOCAL_RANK'] = "0"

torch.distributed.init_process_group(**torch_dist_kwargs, group_name=model_name)
# deepspeed.init_distributed(auto_mpi_discovery=False)

from api.model import FinetuneSpec, Model, ModelVersion
from api.huggingface import create_hf_nn, load_hf_tokenizer

device = torch.device("cuda:0")
model_path = f"/data/aigc/llm/checkpoints/codegen2b-wps/"
tokenizer = load_hf_tokenizer(model_path)
module = create_hf_nn(transformers.AutoModelForCausalLM, model_path)
model = Model('test', module, tokenizer, device, ModelVersion())

from impl.model.backend.deepspeed import DeepspeedTrainBackend

backend = DeepspeedTrainBackend('adam', dict(lr=1e-5), warmup_steps_proportion=0.1, min_lr_ratio=0.0)
model = backend.initialize(model, FinetuneSpec(1, 100, 100, 4))

print("success!")
