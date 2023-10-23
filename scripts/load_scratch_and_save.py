import os
import sys
sys.path.append("../")

from api.huggingface import create_hf_nn
from transformers import AutoModelForCausalLM

import logging 

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level="INFO")

LOAD_PATH = "/lustre/meizy/models/cfg/starcoder"
SAVE_PATH = "/lustre/meizy/models/starcoder_scratch"

if __name__ == "__main__":
    model = create_hf_nn(
        AutoModelForCausalLM,
        model_name_or_path=LOAD_PATH,
        init_from_scratch=True,
        low_cpu_mem_usage=False,
    )
    model.save_pretrained(SAVE_PATH)
