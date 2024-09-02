import dataclasses
import itertools
import os
from typing import Dict, Optional

import colorama
import deepspeed
import torch
import torch.distributed as dist
import tqdm

import realhf.api.core.model_api as model_api
import realhf.base.logging as logging
from realhf.api.core.data_api import SequenceSample
from realhf.base import constants
from realhf.base.datapack import flat2d
from realhf.impl.model.nn.real_llm_api import ReaLModel

logger = logging.getLogger("Packed Reward Modeling Interface", "benchmark")


def flatten_list(l):
    return list(itertools.chain(*l))


def _paired_rw_loss_from_model_outputs(
    scores: torch.FloatTensor,
    input_: SequenceSample,
):
    # Normalize pairs of each prompt with the group factor,
    # which is the reciprocal of the number of pairs in the group.
    group_sizes = [len(x) // 2 for x in input_.seqlens["packed_input_ids"]]
    assert all([x >= 1 for x in group_sizes])
    group_factor = torch.tensor(
        flatten_list([[1 / g for _ in range(g)] for g in group_sizes]),
        device=scores.device,
    )

    input_lens = torch.tensor(flat2d(input_.seqlens["packed_input_ids"]))

    assert scores.shape[0] == input_lens.sum(), (scores.shape, input_lens.sum())
    scores = scores[input_lens.cumsum(0) - 1].view(-1, 2).float()
    loss = -(
        torch.nn.functional.logsigmoid(scores[:, 0] - scores[:, 1]) * group_factor
    ).sum()

    # Logging.
    correct_predictions = (scores[:, 0] > scores[:, 1]).count_nonzero().detach().float()
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
    def inference(
        self, model: model_api.Model, data: SequenceSample, n_mbs=None
    ) -> SequenceSample:

        module = model.module

        module.eval()

        r = module.forward(input_=data, num_micro_batches=n_mbs)
        if r is None:
            return
        scores = r.float()

        input_lens = torch.tensor(flat2d(data.seqlens["packed_input_ids"]))
        scores = scores.view(-1)[input_lens.cumsum(0) - 1].float()  # [bs]
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

        res = SequenceSample(
            keys=["rewards"],
            trailing_shapes=dict(rewards=()),
            dtypes=dict(rewards=torch.float32),
            ids=data.ids,
            seqlens=dict(
                rewards=[
                    [1 for _ in range(len(x))] for x in data.seqlens["packed_input_ids"]
                ]
            ),
            data=dict(rewards=scores),
        )
        return res

    def train_step(
        self, model: model_api.Model, data: SequenceSample, n_mbs=None
    ) -> SequenceSample:
        module = model.module
        module.train()

        stats = module.train_batch(
            input_=data,
            loss_fn=_paired_rw_loss_from_model_outputs,
            version_steps=model.version.global_step,
            num_micro_batches=n_mbs,
        )

        res = {}
        global_stats = constants.log_global_stats_tracker(
            return_dict=True, clear_stats_after_logging=True
        )
        if stats:
            if constants.pipe_parallel_world_size() > 1:
                stats["max_pos_score"] /= constants.pipe_parallel_world_size() * 2
                stats["min_neg_score"] /= constants.pipe_parallel_world_size() * 2
            self.train_total_predictions += int(stats["total_predictions"])
            self.train_total_correct_predictions += int(stats["correct_predictions"])
            res = dict(
                loss=float(stats["loss"] / stats["total_predictions"]),
                epoch_acc=self.train_total_correct_predictions
                / self.train_total_predictions,
                batch_acc=float(
                    stats["correct_predictions"] / stats["total_predictions"]
                ),
                avg_pos_score=float(stats["pos_score"] / stats["total_predictions"]),
                avg_neg_score=float(stats["neg_score"] / stats["total_predictions"]),
                total_predictions=int(stats["total_predictions"]),
                correct_predictions=int(stats["correct_predictions"]),
                max_pos_score=float(stats["max_pos_score"]),
                min_neg_score=float(stats["min_neg_score"]),
                **global_stats,
            )

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            self.train_total_predictions = self.train_total_correct_predictions = 0

        return res

    def save(self, model: model_api.Model, save_dir: str):
        if not self.enable_save:
            return
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
        model = model_.module

        model.eval()
        total_predictions = correct_predictions = 0
        losses = 0
        pos_score = neg_score = 0
        max_pos_score = -float("inf")
        min_neg_score = float("inf")

        for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
            data: SequenceSample
            res = model.eval_batch(
                input_=data.cuda(),
                loss_fn=_paired_rw_loss_from_model_outputs,
            )

            if res is not None:
                _, stats = res
                losses += stats["loss"].item()
                correct_predictions += stats["correct_predictions"].item()
                total_predictions += stats["total_predictions"].item()
                pos_score += stats["pos_score"].item()
                neg_score += stats["neg_score"].item()
                max_pos_score = max(max_pos_score, stats["max_pos_score"].item())
                min_neg_score = min(min_neg_score, stats["min_neg_score"].item())

        global_stats = constants.log_global_stats_tracker(
            return_dict=True, clear_stats_after_logging=True
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
                **global_stats,
            )
        return dict()


model_api.register_interface("paired_rw", PairedRewardInterface)
