import functools
import math
import random

from api.config import *
from api.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology

gen_score = ModelRPC(
    "default",
    ModelInterfaceType.GENERATE,
    input_data=['prompts', 'prompt_att_mask'],
)


class PackedGenerateScoringExperiment(Experiment):

    def __init__(
        self,
        dp_size=8,
        seed=1,
        base_model='gpt2',
        prompt_dataset_path="/lustre/fw/datasets/imdb/rl/rw_prompt-valid.jsonl",
        batch_size: int = 256,
    ):
        self.dp_size = dp_size
        self.n_data_workers = 1
        self.seed = seed

        self.base_model = base_model
        self.prompt_dataset_path = prompt_dataset_path

        self.batch_size = batch_size

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            data_worker=TasksGroup(
                count=self.n_data_workers,
                scheduling=Scheduling.data_worker_default(
                    cpu=2,
                    mem=10000,
                ),
            ),
            master_worker=TasksGroup(
                count=1,
                scheduling=Scheduling.master_worker_default(cpu=4, mem=20000),
            ),
            model_worker=TasksGroup(
                count=self.dp_size,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type='tesla',
                    mem=60000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        if self.base_model == 'starcoder':
            base_model_path = "/data/aigc/public/starcoder-16bit"
        elif self.base_model == 'gpt2':
            base_model_path = "/lustre/fw/pretrained/gpt2-large/"
        else:
            raise NotImplementedError()

        assert self.batch_size % self.n_data_workers == 0
        dataset = Dataset(
            'prompt',
            args=dict(
                max_prompt_len=256,
                pad_to_max_length=True,
                dataset_path=self.prompt_dataset_path,
            ),
        )
        dataloader = DataLoader('default_eval', args=dict(batch_size=self.batch_size // self.n_data_workers))
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=base_model_path,
                datasets=[dataset],
                dataloader=dataloader,
                seed=self.seed,
            ) for i in range(self.n_data_workers)
        ]

        backend = ModelBackend('ds_inference', args=dict(enable_fp16=True))

        model = Model(
            "flash_mqat_clm_hf",
            args=dict(
                model_path=
                "/data/aigc/llm/checkpoints/fw/senti-sft-pos-neg-s42/run20231031/default@pp_00-mp_00-dp_00/epoch8step0/",
                from_type="self",
                tokenizer_path=base_model_path,
            ),
        )

        gconfig = dict(
            min_new_tokens=10,
            max_new_tokens=512,
            temperature=0.7,
            greedy=False,
            top_p=1.0,
            top_k=50,
            num_samples=10,
        )
        interface = ModelInterface('flash_gen_score', args=dict(generation_config=gconfig))

        model_worker = [
            ModelWorker(
                seed=self.seed,
                model=model,
                backend=backend,
                interface=interface,
                model_name='default',
                dp_rank=i,
                topo=PipeModelDataParallelTopology(1, 1, self.dp_size),
                cuda_cache_clear_freq=60,
            ) for i in range(self.dp_size)
        ]

        cfg = ExperimentConfig(
            total_train_epochs=1,
            save_frequency_steps=None,
            save_frequency_epochs=1,
            save_frequency_seconds=None,
            eval_frequency_epochs=None,
            model_rpcs=[gen_score],
            data_worker=data_worker,
            model_worker=model_worker,
        )
        return cfg


seeds = list(range(1, 6)) + [42]
for s in seeds:
    exp_name = f"senti-genscore-s{s}"
    register_experiment(exp_name, functools.partial(PackedGenerateScoringExperiment, seed=s))
