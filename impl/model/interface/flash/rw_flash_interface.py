from typing import Dict
import dataclasses
import logging
import os

import deepspeed
import torch
import tqdm

from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.utils.save import save_hf_or_lora_model
import api.model

logger = logging.getLogger("Packed Plackett Luce Reward Interface")


@dataclasses.dataclass
class PackedPlackettLuceRewardInterface(api.model.ModelInterface):

    def __post_init__(self):
        self.train_total_predictions = self.train_total_correct_predictions = 0

    @torch.inference_mode()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_input_ids: torch.Tensor = data['packed_input_ids'].squeeze()
        cu_seqlens: torch.Tensor = data['cu_seqlens'].squeeze()

        bs = data['n_seqs'].item()

        cu_seqlens = cu_seqlens[:bs + 1]
        packed_input_ids = packed_input_ids[:cu_seqlens[-1]]

        module: deepspeed.DeepSpeedEngine = model.module
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module.eval()

        scores: torch.FloatTensor = module(packed_input_ids=packed_input_ids,
                                           cu_seqlens=cu_seqlens,
                                           max_seqlen=max_seqlen).float()
        chosen_end_scores = scores[cu_seqlens[1:] - 1]  # [bs]

        ###################### logging ######################
        input_ids = [packed_input_ids[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
        seq_strs = model.tokenizer.batch_decode(input_ids,
                                                clean_up_tokenization_spaces=False,
                                                skip_special_tokens=True)
        for seq_str, score in zip(seq_strs, chosen_end_scores):
            logger.info(f"reward is {score.item()}, sequence is: {seq_str}")
        #####################################################

        return from_dict(dict(scores=chosen_end_scores.cpu()))

    def train_step(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        contrastive_dim = data['contrastive_dim'].item()
        n_contrastive_batches = data['n_contrastive_batches'].item()
        n_valid_seqs = contrastive_dim * n_contrastive_batches

        data = recursive_apply(data, lambda x: x.to(model.device))

        packed_input_ids: torch.Tensor = data['packed_input_ids'].squeeze(
        )  # remove batch dim of 1, shape [tot_seqlen]
        cu_seqlens: torch.Tensor = data['cu_seqlens'].squeeze()

        cu_seqlens = cu_seqlens[:n_valid_seqs + 1]  # shape [bs * c_dim + 1]
        packed_input_ids = packed_input_ids[:cu_seqlens[-1]]

        module: deepspeed.DeepSpeedEngine = model.module
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module.train()

        scores: torch.FloatTensor = module(packed_input_ids=packed_input_ids,
                                           cu_seqlens=cu_seqlens,
                                           max_seqlen=max_seqlen).float()
        scores = scores[cu_seqlens[1:] - 1].view(n_contrastive_batches, contrastive_dim)

        bs = scores.shape[0]
        labels: torch.FloatTensor = data['labels'].squeeze()[:(contrastive_dim + 1) *
                                                             n_contrastive_batches].view(
                                                                 n_contrastive_batches, contrastive_dim + 1)

        scores = torch.cat([torch.zeros((bs, 1), dtype=scores.dtype, device=scores.device), scores], dim=1)
        loss = torch.nn.functional.cross_entropy(scores, labels, reduction='mean')
        # logger.info(f"scores: {scores}, labels: {labels}, loss: {loss}.")

        module.backward(loss)
        module.step()

        correct_predictions = (scores.argmax(-1) == labels.argmax(-1)).float().sum().detach().item()
        self.train_total_correct_predictions += correct_predictions
        self.train_total_predictions += bs
        acc = self.train_total_correct_predictions / self.train_total_predictions

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()
            self.train_total_predictions = self.train_total_correct_predictions = 0

        return dict(loss=loss.detach().item(), acc=acc)

    def save(self, model: api.model.Model, output_dir):
        from impl.model.nn.lora import is_lora_model
        save_hf_or_lora_model(model, output_dir)
        if is_lora_model(model.module):
            save_path = os.path.abspath(
                os.path.join(
                    output_dir,
                    f"epoch{model.version.epoch}step{model.version.epoch_step}",
                ))
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.module.module.head.state_dict(), os.path.join(save_path, "rw_v_head.bin"))

    @torch.inference_mode()
    def evaluate(self, model_: api.model.Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
        device = model_.device
        model = model_.module

        model.eval()
        total_predictions = correct_predictions = 0
        loss = 0

        for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
            contrastive_dim = data['contrastive_dim'].item()
            n_contrastive_batches = data['n_contrastive_batches'].item()
            n_valid_seqs = contrastive_dim * n_contrastive_batches

            data = recursive_apply(from_dict(data), lambda x: x.to(device))

            packed_input_ids: torch.Tensor = data['packed_input_ids'].squeeze(
            )  # remove batch dim of 1, shape [tot_seqlen]
            cu_seqlens: torch.Tensor = data['cu_seqlens'].squeeze()

            cu_seqlens = cu_seqlens[:n_valid_seqs + 1]  # shape [bs * c_dim + 1]
            packed_input_ids = packed_input_ids[:cu_seqlens[-1]]
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

            scores: torch.FloatTensor = model(packed_input_ids=packed_input_ids,
                                              cu_seqlens=cu_seqlens,
                                              max_seqlen=max_seqlen).float()
            scores = scores[cu_seqlens[1:] - 1].view(-1, contrastive_dim)  # [bs, c_dim]

            bs = scores.shape[0]
            labels: torch.FloatTensor = data['labels'].squeeze()[:(contrastive_dim + 1) *
                                                                 n_contrastive_batches].view(
                                                                     n_contrastive_batches,
                                                                     contrastive_dim + 1)

            scores = torch.cat([torch.zeros((bs, 1), dtype=scores.dtype, device=scores.device), scores],
                               dim=1)
            loss += torch.nn.functional.cross_entropy(scores, labels, reduction='sum')
            correct_predictions += (scores.argmax(-1) == labels.argmax(-1)).float().sum().detach().item()
            total_predictions += bs

        return dict(loss=float(loss / total_predictions), acc=correct_predictions / total_predictions)


api.model.register_interface("flash_plrw", PackedPlackettLuceRewardInterface)