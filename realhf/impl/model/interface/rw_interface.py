from typing import Dict, Optional
import dataclasses
import os

import colorama
import deepspeed
import torch
import torch.distributed as dist
import tqdm

from realhf.base import constants
from realhf.base.namedarray import from_dict, NamedArray, recursive_apply
from realhf.impl.model.nn.real_llm_api import ReaLModel
import realhf.api.core.model_api as model_api
import realhf.base.logging as logging

logger = logging.getLogger("Packed Reward Modeling Interface", "benchmark")


def _paired_rw_loss_from_model_outputs(
    scores: torch.FloatTensor,
    packed_input_ids: torch.LongTensor,
    cu_seqlens: torch.IntTensor,
    group_factor: torch.FloatTensor,
    **kwargs,
):
    scores = scores[cu_seqlens[1:] - 1].view(-1, 2).float()
    loss = -(
        torch.nn.functional.logsigmoid(scores[:, 0] - scores[:, 1])
        * group_factor
    ).sum()

    # Logging.
    correct_predictions = (
        (scores[:, 0] > scores[:, 1]).count_nonzero().detach().float()
    )
    total_predictions = torch.tensor(
        scores.shape[0], dtype=torch.float32, device=scores.device
    )
    dist.all_reduce(
        correct_predictions,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        total_predictions,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    pos_score_sum = scores[:, 0].sum().detach()
    max_pos_score = scores[:, 0].max(dim=0).values
    neg_score_sum = scores[:, 1].sum().detach()
    min_neg_score = scores[:, 1].min(dim=0).values
    dist.all_reduce(
        pos_score_sum,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        neg_score_sum,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    loss_logging = loss.detach()
    dist.all_reduce(
        loss_logging,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        max_pos_score,
        op=dist.ReduceOp.MAX,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        min_neg_score,
        op=dist.ReduceOp.MIN,
        group=constants.data_parallel_group(),
    )

    return loss, dict(
        loss=loss_logging,
        correct_predictions=correct_predictions,
        total_predictions=total_predictions,
        pos_score=pos_score_sum,
        neg_score=neg_score_sum,
        max_pos_score=max_pos_score,
        min_neg_score=min_neg_score,
    )


@dataclasses.dataclass
class PairedRewardInterface(model_api.ModelInterface):
    enable_save: bool = True

    output_scaling: float = 1.0
    output_bias: float = 0.0

    # training log
    train_total_predictions: int = 0
    train_total_correct_predictions: int = 0

    @torch.no_grad()
    def inference(self, model: model_api.Model, data: NamedArray) -> NamedArray:
        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_input_ids: torch.Tensor = data["packed_input_ids"]
        seqlens_cpu = data.metadata["seqlens"]
        max_seqlen = max(seqlens_cpu)
        cu_seqlens = torch.nn.functional.pad(
            torch.tensor(
                seqlens_cpu, dtype=torch.int32, device=model.device
            ).cumsum(0),
            (1, 0),
        )

        module: deepspeed.DeepSpeedEngine = model.module

        module.eval()

        r = module.forward(
            seqlens_cpu=data.metadata["seqlens"],
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
        )
        if r is None:
            return
        scores = r.float()

        scores = scores.squeeze(-1)[cu_seqlens[1:] - 1].float()  # [bs]
        scores = (scores - self.output_bias) * self.output_scaling

        ###################### logging ######################
        # input_ids = [packed_input_ids[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
        # seq_strs = model.tokenizer.batch_decode(input_ids,
        #                                         clean_up_tokenization_spaces=False,
        #                                         skip_special_tokens=True)
        # for seq_str, score in zip(seq_strs, scores):
        #     logger.info(
        #         f"reward is {colorama.Fore.RED}{score.item()}{colorama.Style.RESET_ALL}, "
        #         f"sequence is: {colorama.Fore.YELLOW + colorama.Style.DIM}{seq_str}{colorama.Style.RESET_ALL}"
        #     )
        #####################################################

        res = from_dict(dict(scores=scores))
        res.register_metadata(**data.metadata)
        return res

    def train_step(
        self, model: model_api.Model, data: NamedArray
    ) -> NamedArray:
        data = recursive_apply(data, lambda x: x.to(model.device))

        packed_input_ids: torch.Tensor = data["packed_input_ids"]
        pair_lens = torch.tensor(
            data.metadata["seqlens"], dtype=torch.int32, device=model.device
        )
        neg_input_lens = pair_lens - data["pos_input_lens"]
        input_lens: torch.Tensor = torch.stack(
            [data["pos_input_lens"], neg_input_lens], 1
        ).view(-1)
        group_factor: torch.Tensor = data["group_factor"]
        cu_seqlens = torch.cat(
            [input_lens.new_zeros(1), input_lens.cumsum(0)], 0
        ).int()
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module = model.module
        module.train()

        loss_fn_kwargs = dict(
            input_lens=pair_lens,
            group_factor=data["group_factor"],
        )
        stats = module.train_batch(
            seqlens_cpu=data.metadata["seqlens"],
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            loss_fn=_paired_rw_loss_from_model_outputs,
            input_lens_for_partition=pair_lens,
            version_steps=model.version.global_step,
            **loss_fn_kwargs,
        )

        res = {}
        if stats:
            if constants.pipe_parallel_world_size() > 1:
                stats["max_pos_score"] /= (
                    constants.pipe_parallel_world_size() * 2
                )
                stats["min_neg_score"] /= (
                    constants.pipe_parallel_world_size() * 2
                )
            self.train_total_predictions += int(stats["total_predictions"])
            self.train_total_correct_predictions += int(
                stats["correct_predictions"]
            )
            res = dict(
                loss=float(stats["loss"] / stats["total_predictions"]),
                epoch_acc=self.train_total_correct_predictions
                / self.train_total_predictions,
                batch_acc=float(
                    stats["correct_predictions"] / stats["total_predictions"]
                ),
                avg_pos_score=float(
                    stats["pos_score"] / stats["total_predictions"]
                ),
                avg_neg_score=float(
                    stats["neg_score"] / stats["total_predictions"]
                ),
                total_predictions=int(stats["total_predictions"]),
                correct_predictions=int(stats["correct_predictions"]),
                max_pos_score=float(stats["max_pos_score"]),
                min_neg_score=float(stats["min_neg_score"]),
            )

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            self.train_total_predictions = (
                self.train_total_correct_predictions
            ) = 0

        return res

    def save(self, model: model_api.Model, save_dir: str):
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model_: model_api.Model,
        eval_dataloader: torch.utils.data.DataLoader,
    ) -> Dict:
        device = model_.device
        model = model_.module

        model.eval()
        total_predictions = correct_predictions = 0
        losses = 0
        pos_score = neg_score = 0
        max_pos_score = -float("inf")
        min_neg_score = float("inf")

        for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
            pair_lens = torch.tensor(
                data.metadata["seqlens"], dtype=torch.int32, device=model.device
            )
            data = recursive_apply(data, lambda x: x.to(device))

            packed_input_ids: torch.Tensor = data["packed_input_ids"]
            neg_input_lens = pair_lens - data["pos_input_lens"]
            assert (neg_input_lens > 0).all()
            input_lens = torch.stack(
                [data["pos_input_lens"], neg_input_lens], 1
            ).view(-1)
            group_factor: torch.Tensor = data["group_factor"]
            cu_seqlens = torch.cat(
                [input_lens.new_zeros(1), input_lens.cumsum(0)], 0
            ).int()
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

            loss_fn_kwargs = dict(
                input_lens=pair_lens,
                group_factor=data["group_factor"],
            )
            stats = model.eval_batch(
                seqlens_cpu=data.metadata["seqlens"],
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                loss_fn=_paired_rw_loss_from_model_outputs,
                input_lens_for_partition=pair_lens,
                **loss_fn_kwargs,
            )

            if stats:
                assert input_lens.shape[0] % 2 == 0
                losses += stats["loss"].item()
                correct_predictions += stats["correct_predictions"].item()
                total_predictions += stats["total_predictions"].item()
                pos_score += stats["pos_score"].item()
                neg_score += stats["neg_score"].item()
                max_pos_score = max(
                    max_pos_score, stats["max_pos_score"].item()
                )
                min_neg_score = min(
                    min_neg_score, stats["min_neg_score"].item()
                )

        if total_predictions > 0:
            return dict(
                loss=float(losses / total_predictions),
                acc=correct_predictions / total_predictions,
                pos_score=float(pos_score / total_predictions),
                neg_score=float(neg_score / total_predictions),
                correct_predictions=int(correct_predictions),
                total_predictions=int(total_predictions),
                max_pos_score=float(max_pos_score),
                min_neg_score=float(min_neg_score),
            )
        return dict()


model_api.register_interface("paired_rw", PairedRewardInterface)
