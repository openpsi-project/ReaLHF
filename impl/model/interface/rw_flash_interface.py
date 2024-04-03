from typing import Dict, Optional
import dataclasses
import os

from deepspeed import DeepSpeedEngine
import colorama
import deepspeed
import torch
import tqdm

from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.backend.pipe_engine.ds_pipe_engine import DeepSpeedPipelineEngine
from impl.model.backend.pipe_inf import InferencePipelineEngine
from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel
import api.model
import base.constants
import base.logging as logging

logger = logging.getLogger("Packed Reward Modeling Interface", "benchmark")


def _paired_rw_loss_from_model_outputs(
    scores: torch.FloatTensor,
    packed_input_ids: torch.LongTensor,
    cu_seqlens: torch.IntTensor,
    group_factor: torch.FloatTensor,
    **kwargs,
):
    scores = scores[cu_seqlens[1:] - 1].view(-1, 2).float()
    loss = -(torch.nn.functional.logsigmoid(scores[:, 0] - scores[:, 1]) * group_factor).sum()
    correct_predictions = (scores[:, 0] > scores[:, 1]).count_nonzero().detach().float()
    return loss, dict(
        loss=loss.cpu(),
        correct_predictions=correct_predictions.cpu(),
        avg_pos_score=scores[:, 0].mean().detach().cpu(),
        avg_neg_score=scores[:, 1].mean().detach().cpu(),
    )


@dataclasses.dataclass
class PackedPairedRewardInterface(api.model.ModelInterface):
    enable_save: bool = True

    output_scaling: float = 1.0
    output_bias: float = 0.0

    pipe_inf_n_mbs: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.train_total_predictions = self.train_total_correct_predictions = 0
        if self.pipe_inf_n_mbs is None:
            self.pipe_inf_n_mbs = base.constants.pipe_parallel_world_size()

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_input_ids: torch.Tensor = data["packed_input_ids"]
        cu_seqlens: torch.Tensor = data["cu_seqlens"].int()

        module: deepspeed.DeepSpeedEngine = model.module
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module.eval()

        if isinstance(module, (InferencePipelineEngine, DeepSpeedPipelineEngine)):
            r = module.forward(seqlens_cpu=data.metadata['seqlens'],
                               packed_input_ids=packed_input_ids,
                               cu_seqlens=cu_seqlens,
                               num_micro_batches=self.pipe_inf_n_mbs)
            if r is None:
                return
            scores = r.float()
        else:
            scores: torch.FloatTensor = module(packed_input_ids=packed_input_ids,
                                               cu_seqlens=cu_seqlens,
                                               max_seqlen=max_seqlen).float()

        scores = (scores.squeeze(-1) - self.output_bias) * self.output_scaling
        chosen_end_scores = scores[cu_seqlens[1:] - 1]  # [bs]

        ###################### logging ######################
        # input_ids = [packed_input_ids[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
        # seq_strs = model.tokenizer.batch_decode(input_ids,
        #                                         clean_up_tokenization_spaces=False,
        #                                         skip_special_tokens=True)
        # for seq_str, score in zip(seq_strs, chosen_end_scores):
        #     logger.info(
        #         f"reward is {colorama.Fore.RED}{score.item()}{colorama.Style.RESET_ALL}, sequence is: {colorama.Fore.YELLOW + colorama.Style.DIM}{seq_str}{colorama.Style.RESET_ALL}"
        #     )
        #####################################################

        return from_dict(dict(scores=chosen_end_scores))

    def train_step(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        data = recursive_apply(data, lambda x: x.to(model.device))

        packed_input_ids: torch.Tensor = data["packed_input_ids"]
        input_lens: torch.Tensor = data["pair_input_lens"]
        group_factor: torch.Tensor = data["group_factor"]
        cu_seqlens = torch.cat([input_lens.new_zeros(1), input_lens.cumsum(0)], 0).int()
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module = model.module
        module.train()

        if isinstance(module, DeepSpeedPipelineEngine):
            loss_fn_kwargs = dict(
                input_lens=data["input_lens"],
                group_factor=data["group_factor"],
            )
            loss, stats = module.train_batch(
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                loss_fn=_paired_rw_loss_from_model_outputs,
                input_lens_for_partition=data["input_lens"],
                **loss_fn_kwargs,
            )
        else:
            scores: torch.FloatTensor = module(packed_input_ids=packed_input_ids,
                                               cu_seqlens=cu_seqlens,
                                               max_seqlen=max_seqlen)
            loss, stats = _paired_rw_loss_from_model_outputs(scores, packed_input_ids, cu_seqlens,
                                                             group_factor)

            module.backward(loss)
            module.step()

        if stats is not None:
            self.train_total_correct_predictions += stats["correct_predictions"].item()
            assert input_lens.shape[0] % 2 == 0
            self.train_total_predictions += input_lens.shape[0] // 2
            acc = self.train_total_correct_predictions / self.train_total_predictions
            stats["acc"] = acc

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()
            self.train_total_predictions = self.train_total_correct_predictions = 0

        if stats is None:
            stats = {}
        return {k: float(v) for k, v in stats.items()}

    def save(self, model: api.model.Model, output_dir):
        if not self.enable_save:
            return
        model.module.save(output_dir,
                          epoch=model.version.epoch,
                          epoch_step=model.version.epoch_step,
                          global_step=model.version.global_step)

    @torch.no_grad()
    def evaluate(self, model_: api.model.Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
        device = model_.device
        model = model_.module

        model.eval()
        total_predictions = correct_predictions = 0
        losses = 0
        pos_score = neg_score = 0

        for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
            data = recursive_apply(from_dict(data), lambda x: x.to(device))

            packed_input_ids: torch.Tensor = data["packed_input_ids"]
            input_lens: torch.Tensor = data["pair_input_lens"]
            group_factor: torch.Tensor = data["group_factor"]
            cu_seqlens = torch.cat([input_lens.new_zeros(1), input_lens.cumsum(0)], 0).int()
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

            if isinstance(model, DeepSpeedPipelineEngine):
                loss_fn_kwargs = dict(
                    input_lens=data["input_lens"],
                    group_factor=data["group_factor"],
                )
                loss, stats = model.eval_batch(
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    loss_fn=_paired_rw_loss_from_model_outputs,
                    input_lens_for_partition=data["input_lens"],
                    **loss_fn_kwargs,
                )
            else:
                scores: torch.FloatTensor = model(packed_input_ids=packed_input_ids,
                                                  cu_seqlens=cu_seqlens,
                                                  max_seqlen=max_seqlen)
                loss, stats = _paired_rw_loss_from_model_outputs(scores, packed_input_ids, cu_seqlens,
                                                                 group_factor)

            if stats is not None:
                assert input_lens.shape[0] % 2 == 0
                losses += loss.item() * (input_lens.shape[0] // 2)
                correct_predictions += stats["correct_predictions"].item()
                total_predictions += input_lens.shape[0] // 2
                pos_score += stats["avg_pos_score"].item()
                neg_score += stats["avg_neg_score"].item()

        if total_predictions > 0:
            return dict(
                loss=float(losses / total_predictions),
                acc=correct_predictions / total_predictions,
                pos_score=float(pos_score / total_predictions),
                neg_score=float(neg_score / total_predictions),
            )
        return dict()


api.model.register_interface("flash_paired_rw", PackedPairedRewardInterface)

# @dataclasses.dataclass
# class PackedPlackettLuceRewardInterface(api.model.ModelInterface):
#     enable_save: bool = True

#     def __post_init__(self):
#         self.train_total_predictions = self.train_total_correct_predictions = 0

#     @torch.no_grad()
#     def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
#         data = recursive_apply(data, lambda x: x.to(model.device))
#         packed_input_ids: torch.Tensor = data["packed_input_ids"].squeeze()
#         cu_seqlens: torch.Tensor = data["cu_seqlens"].squeeze()

#         module: deepspeed.DeepSpeedEngine = model.module
#         max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

#         module.eval()

#         scores: torch.FloatTensor = module(
#             packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
#         ).float()
#         chosen_end_scores = scores[cu_seqlens[1:] - 1]  # [bs]

#         ###################### logging ######################
#         input_ids = [packed_input_ids[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
#         seq_strs = model.tokenizer.batch_decode(
#             input_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
#         )
#         for seq_str, score in zip(seq_strs, chosen_end_scores):
#             logger.debug(f"reward is {score.item()}, sequence is: {seq_str}")
#         #####################################################

#         return from_dict(dict(scores=chosen_end_scores.cpu()))

#     def train_step(self, model: api.model.Model, data: NamedArray) -> NamedArray:
#         contrastive_dim = data["contrastive_dim"].item()
#         n_contrastive_batches = data["n_contrastive_batches"].item()
#         n_valid_seqs = contrastive_dim * n_contrastive_batches

#         data = recursive_apply(data, lambda x: x.to(model.device))

#         packed_input_ids: torch.Tensor = data[
#             "packed_input_ids"
#         ].squeeze()  # remove batch dim of 1, shape [tot_seqlen]
#         cu_seqlens: torch.Tensor = data["cu_seqlens"].squeeze()

#         cu_seqlens = cu_seqlens[: n_valid_seqs + 1]  # shape [bs * c_dim + 1]
#         packed_input_ids = packed_input_ids[: cu_seqlens[-1]]

#         module: deepspeed.DeepSpeedEngine = model.module
#         max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

#         module.train()

#         scores: torch.FloatTensor = module(
#             packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
#         ).float()
#         scores = scores[cu_seqlens[1:] - 1].view(n_contrastive_batches, contrastive_dim)

#         bs = scores.shape[0]
#         labels: torch.FloatTensor = (
#             data["labels"]
#             .squeeze()[: (contrastive_dim + 1) * n_contrastive_batches]
#             .view(n_contrastive_batches, contrastive_dim + 1)
#         )

#         scores = torch.cat([torch.zeros((bs, 1), dtype=scores.dtype, device=scores.device), scores], dim=1)
#         loss = torch.nn.functional.cross_entropy(scores, labels, reduction="mean")
#         # logger.info(f"scores: {scores}, labels: {labels}, loss: {loss}.")

#         module.backward(loss)
#         module.step()

#         correct_predictions = (scores.argmax(-1) == labels.argmax(-1)).float().sum().detach().item()
#         self.train_total_correct_predictions += correct_predictions
#         self.train_total_predictions += bs
#         acc = self.train_total_correct_predictions / self.train_total_predictions

#         cur_epoch = model.version.epoch
#         model.inc_version()
#         if model.version.epoch > cur_epoch:
#             module.tput_timer.update_epoch_count()
#             self.train_total_predictions = self.train_total_correct_predictions = 0

#         return dict(loss=loss.detach().item(), acc=acc)

#     def save(self, model: api.model.Model, output_dir):
#         if not self.enable_save:
#             return
#         from impl.model.nn.lora import is_lora_model

#         save_hf_or_lora_model(model, output_dir)
#         if is_lora_model(model.module):
#             save_path = os.path.abspath(
#                 os.path.join(
#                     output_dir,
#                     f"epoch{model.version.epoch}step{model.version.epoch_step}",
#                 )
#             )
#             os.makedirs(save_path, exist_ok=True)
#             torch.save(model.module.module.head.state_dict(), os.path.join(save_path, "rw_v_head.bin"))

#     @torch.no_grad()
#     def evaluate(self, model_: api.model.Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
#         device = model_.device
#         model = model_.module

#         model.eval()
#         total_predictions = correct_predictions = 0
#         loss = 0

#         for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
#             contrastive_dim = data["contrastive_dim"].item()
#             n_contrastive_batches = data["n_contrastive_batches"].item()
#             n_valid_seqs = contrastive_dim * n_contrastive_batches

#             data = recursive_apply(from_dict(data), lambda x: x.to(device))

#             packed_input_ids: torch.Tensor = data[
#                 "packed_input_ids"
#             ].squeeze()  # remove batch dim of 1, shape [tot_seqlen]
#             cu_seqlens: torch.Tensor = data["cu_seqlens"].squeeze()

#             cu_seqlens = cu_seqlens[: n_valid_seqs + 1]  # shape [bs * c_dim + 1]
#             packed_input_ids = packed_input_ids[: cu_seqlens[-1]]
#             max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

#             scores: torch.FloatTensor = model(
#                 packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
#             ).float()
#             scores = scores[cu_seqlens[1:] - 1].view(-1, contrastive_dim)  # [bs, c_dim]

#             bs = scores.shape[0]
#             labels: torch.FloatTensor = (
#                 data["labels"]
#                 .squeeze()[: (contrastive_dim + 1) * n_contrastive_batches]
#                 .view(n_contrastive_batches, contrastive_dim + 1)
#             )

#             scores = torch.cat(
#                 [torch.zeros((bs, 1), dtype=scores.dtype, device=scores.device), scores], dim=1
#             )
#             loss += torch.nn.functional.cross_entropy(scores, labels, reduction="sum")
#             correct_predictions += (scores.argmax(-1) == labels.argmax(-1)).float().sum().detach().item()
#             total_predictions += bs

#         return dict(loss=float(loss / total_predictions), acc=correct_predictions / total_predictions)

# api.model.register_interface("flash_plrw", PackedPlackettLuceRewardInterface)
