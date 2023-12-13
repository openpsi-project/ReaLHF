from statistics import mean
import logging
import os
import random
import time
import unittest

# import transformers
import torch
import torch.multiprocessing as mp

from base.namedarray import NamedArray
from tests.parallel.utils import *
import api.config as config_package

# import base.consistency

NUM_PP = 2
NUM_DP = 8 // NUM_PP
WORLD_SIZE = NUM_PP * NUM_DP
MODEL_TYPE = "llama"
if MODEL_TYPE == "llama":
    BASELINE_MODEL_PATH = "/lustre/public/pretrained_model_weights/Llama-2-13b-hf"
    PIPELINE_MODEL_PATH = f"/lustre/public/pretrained_model_weights/sharded/Llama-2-13b-hf_{NUM_PP}pp_3s"
    # BASELINE_MODEL_PATH = "/home/meizy/models/test/Llama-2-4l"
    # PIPELINE_MODEL_PATH = "/home/meizy/models/test/llama-2-4l_4pp_3s"
elif MODEL_TYPE == "starcoder":
    BASELINE_MODEL_PATH = "/lustre/meizy/models/starcoder_4l"
    PIPELINE_MODEL_PATH = F"/lustre/meizy/models/pipe_starcoder_4l_4pp_1s"
elif MODEL_TYPE == "gpt2":
    BASELINE_MODEL_PATH = "/lustre/fw/pretrained/gpt2/"
    PIPELINE_MODEL_PATH = F"/lustre/public/pretrained_model_weights/testOnly/gpt2_4pp_1s"
BATCH_SIZE = 32
MIN_NEW_TOKENS = 1
MAX_NEW_TOKENS = 2048

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level="DEBUG")
logger = logging.getLogger("pipe_mqat_test")


def make_pipe_interface():
    import api.model
    return api.model.make_interface(config_package.ModelInterface(type_="pipe_flash_sft", args=dict()))


def make_pipe_backend():
    import api.model
    return api.model.make_backend(
        config_package.ModelBackend(type_='ds_train',
                                    args=dict(optimizer_name='adam',
                                              optimizer_config=dict(lr=1e-5,
                                                                    weight_decay=0.0,
                                                                    betas=(0.9, 0.95)),
                                              warmup_steps_proportion=0.0,
                                              min_lr_ratio=0.0,
                                              zero_stage=1,
                                              engine_type="pipe",
                                              num_pipeline_stages=NUM_PP)))


def make_pipe_model(model_path, device):
    # from impl.model.backend.ds_pipe_engine import PipeDataParallelTopology
    from base.topology import PipeDataParallelTopology
    import api.model
    import impl.model
    topology = PipeDataParallelTopology(num_pp=NUM_PP, num_dp=NUM_DP)
    model_config = config_package.Model(type_="flash_mqat_pipe",
                                        args=dict(
                                            model_path=model_path,
                                            num_pp=NUM_PP,
                                            num_dp=NUM_DP,
                                            from_type=MODEL_TYPE,
                                            dtype=torch.float16,
                                        ))
    model = api.model.make_model(model_config, name=MODEL_NAME, device=device)
    return topology, model


def init_handles(rank):
    device = setup_gpu(rank)
    init_global_constants(NUM_DP, 1, NUM_PP)

    cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
    logger.info(f"PROCESS RANK: {rank}; \n"
                f"TORCH DIST RANK: {torch.distributed.get_rank()}; \n"
                f"CUDA VISIBLE: {cuda_visible}")
    topo, model = make_pipe_model(PIPELINE_MODEL_PATH, device)
    backend = make_pipe_backend()
    ft_spec = make_finetune_spec(BATCH_SIZE)
    interface = make_pipe_interface()

    backend.initialize(model, ft_spec)

    return device, model, backend, interface


def pipe_load_save(rank):
    device, model, backend, interface = init_handles(rank)
    os.makedirs("/tmp/pipe_mqat_test", exist_ok=True)
    model.module.save("/tmp/pipe_mqat_test")
    print("pipeline module save successful")
    model.module.load("/tmp/pipe_mqat_test")
    print("pipeline module load successful")


def pipe_generate(rank, res_queue: mp.Queue, seed: int):
    device, model, backend, interface = init_handles(rank)
    data = init_data(model.tokenizer, device, BATCH_SIZE, seed=seed)

    random.seed(seed + rank)

    from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
    gconfig = GenerationConfig(min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=MAX_NEW_TOKENS)

    st = time.monotonic()
    outputs = interface.generate(model, data, gconfig=gconfig)
    t = time.monotonic() - st
    # logger.info(input_ids)

    if len(outputs) > 0 and res_queue is not None:
        # logger.info(input_ids)
        logger.info(outputs["gen_tokens"].shape)
        # logger.info(outputs["log_probs"])
        res_queue.put(outputs["gen_tokens"])
        res_queue.put(outputs["log_probs"])
        res_queue.put(t)
        time.sleep(2)  # wait for queue get, otherwise invalid tensor handle


def pipe_train_batch(rank, res_queue: mp.Queue, seed: int):
    device, model, backend, interface = init_handles(rank)
    ts = []
    for seed in range(20):
        data = init_data(rank, model, device, seed)
        st = time.monotonic()
        outputs = interface.train_step(model, data)
        t = time.monotonic() - st
        ts.append(t)
    t = mean(ts[1:])
    print(f"{rank} {outputs} timecost {t} {ts}")


def pipe_train_batch_accordance(rank: int, res_queue: mp.Queue, seed: int):
    device, model, backend, interface = init_handles(rank)
    # data = init_data(rank, model, device, seed)
    module = model.module.module  # pipeline module

    before = {name: param.detach().clone() for name, param in module.state_dict().items()}

    for _ in range(20):
        data = init_data(rank, model, device, seed)
        outputs = interface.train_step(model, data)
        seed = random.randint(0, 7777)

    after = module.state_dict()
    # for name, param in after.items():
    #     # grads
    #     print(f"grads {name}: {param.grad}")

    print(f"rank {rank} PipeModule parameters")
    for name, param in before.items():
        print(f"train diff {name}: {torch.abs(param-after[name]).max()}")
        # print(f"rank {rank} {name}: {param.size()}")

    [sampled] = random.sample(list(module.state_dict().items()), 1)
    res_queue.put(sampled)

    time.sleep(2)


class PipeFlashMQATTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # name_resolve.clear_subtree(names.trainer_ddp_peer(EXPR_NAME, TRIAL_NAME, MODEL_NAME))
        clear_name_resolve()
        cls.baseline_model = None

    def init_tokenizer(self):
        import api.huggingface

        self.tokenizer = api.huggingface.load_hf_tokenizer(BASELINE_MODEL_PATH)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def init_baseline_model(self):
        import transformers

        from impl.model.nn.flash_mqat.flash_generate import generate, GenerationConfig
        from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATModel
        import impl.model.nn.flash_mqat.flash_from_hf_impl

        self.device = device = 'cuda'
        self.dtype = dtype = torch.float16

        self.init_tokenizer()

        # pretrained: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        #     pretrained_model_name_or_path=BASELINE_MODEL_PATH).to(dtype=dtype, device=device)
        # pretrained.eval()

        self.baseline_model = None
        if MODEL_TYPE == "llama":
            self.baseline_model = FlashMQATModel.from_llama(model_path=BASELINE_MODEL_PATH)
        elif MODEL_TYPE == "starcoder":
            self.baseline_model = FlashMQATModel.from_starcoder(model_path=BASELINE_MODEL_PATH)
        else:
            raise NotImplementedError()

        self.baseline_model.to(dtype=dtype, device=device)
        self.baseline_model.eval()

    @torch.no_grad()
    def testGenerateAccordance(self):
        clear_name_resolve()
        self.seed = 212
        self.res_queue = mp.Queue(maxsize=128)
        self.pipe_model_processes = [
            mp.Process(target=pipe_generate, args=(i, self.res_queue, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()

        g = self.res_queue.get()
        logprob = self.res_queue.get()
        t = self.res_queue.get()

        for p in self.pipe_model_processes:
            p.join()

        from impl.model.nn.flash_mqat.flash_generate import generate, GenerationConfig
        self.gconfig = GenerationConfig(min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=MAX_NEW_TOKENS)
        # baseline model calculate
        self.init_baseline_model()
        prompt, prompt_att_mask = make_batch(self.tokenizer,
                                             self.device,
                                             BATCH_SIZE,
                                             0,
                                             NUM_DP,
                                             seed=self.seed)

        s2 = time.monotonic()

        # base.consistency.set_step_id(0)
        # base.consistency.set_parallel_mode(False)
        vg, vlogprob, vmask, *_ = generate(model=self.baseline_model,
                                           tokenizer=self.tokenizer,
                                           input_ids=prompt,
                                           attention_mask=prompt_att_mask,
                                           gconfig=self.gconfig)
        t2 = time.monotonic() - s2
        # base.consistency.check_all()
        print("pipe time:", t)
        print("vanilla time:", t2)

        print(f"at seed {self.seed} diff {torch.abs(g-vg)}, {torch.abs(g-vg).max()}")

        # try:
        #     assert torch.allclose(logprob, vlogprob, atol=0, rtol=0.01)
        # except AssertionError as e:
        #     print(
        #         f"at seed {self.seed} diff {torch.abs(logprob-vlogprob)}, {torch.abs(logprob-vlogprob).abs()}"
        #     )
        #     raise e

    def testSaveLoad(self):
        clear_name_resolve()
        self.pipe_model_processes = [mp.Process(target=pipe_load_save, args=(i,)) for i in range(WORLD_SIZE)]
        for p in self.pipe_model_processes:
            p.start()
        for p in self.pipe_model_processes:
            p.join()

    @torch.no_grad()
    def testGenerate(self):
        clear_name_resolve()
        self.seed = random.randint(0, 1000)
        self.res_queue = mp.Queue(maxsize=128)
        self.pipe_model_processes = [
            mp.Process(target=pipe_generate, args=(i, self.res_queue, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()

        g = self.res_queue.get()
        logprob = self.res_queue.get()
        t = self.res_queue.get()

        for p in self.pipe_model_processes:
            p.join()

        print(f"Generated shape {g.shape}")
        # print(logprob)
        # print(t)

    def testTrainBatch(self):
        clear_name_resolve()
        self.seed = random.randint(0, 1000)
        self.res_queue = mp.Queue(maxsize=128)
        self.pipe_model_processes = [
            mp.Process(target=pipe_train_batch, args=(i, self.res_queue, self.seed))
            for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()
        for p in self.pipe_model_processes:
            p.join()

    def testTrainBatchAccordance(self):
        self.seed = random.randint(0, 1000)
        self.res_queue = mp.Queue(maxsize=128)
        self.pipe_model_processes = [
            mp.Process(target=pipe_train_batch_accordance, args=(i, self.res_queue, self.seed))
            for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()

        sampled_weights = {}
        for _ in range(WORLD_SIZE):
            param_name, param = self.res_queue.get()
            sampled_weights[param_name] = param
            print(f"sampled weight {param_name} {param.size()}")

        for p in self.pipe_model_processes:
            p.join()


if __name__ == "__main__":
    unittest.main(defaultTest="PipeFlashMQATTest.testTrainBatch")
