from typing import List, Union
import os

import torch
import torch.nn as nn
import transformers

import api.model
import api.utils


class RewardModel(nn.Module):

    def __init__(
        self,
        base_model_name_or_path: str,
        load_state_dict: bool,
        disable_dropout: bool,
    ):
        super().__init__()
        base_model = api.utils.create_hf_nn(
            transformers.AutoModel,
            model_name_or_path=base_model_name_or_path,
            disable_dropout=disable_dropout,
            init_from_scratch=load_state_dict,
        )
        self.config = base_model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(self.config,
                                                                "hidden_size") else self.config.n_embd
        # base_model is created by `AutoModel` instead of `AutoModelForCausalLM`
        # its forward function outputs the last hidden state instead of logits
        self.rwtranrsformer = base_model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False, dtype=torch.float16)
        if load_state_dict:
            model_ckpt_path = os.path.join(base_model_name_or_path, 'pytorch_model.bin')
            assert os.path.exists(model_ckpt_path), f"Cannot find model checkpoint at {model_ckpt_path}"
            model_ckpt = torch.load(model_ckpt_path, map_location='cpu')
            self.load_state_dict(model_ckpt)

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=False,
    ):
        transformer_outputs = self.rwtranrsformer(input_ids,
                                                  past_key_values=past_key_values,
                                                  attention_mask=attention_mask,
                                                  head_mask=head_mask,
                                                  inputs_embeds=inputs_embeds,
                                                  use_cache=use_cache)
        hidden_states = transformer_outputs[0]
        scores = self.v_head(hidden_states).squeeze(-1)
        return scores


class ZippedRewardModel(nn.Module):

    def __init__(
        self,
        reward_models: List[RewardModel],
    ):
        super().__init__()
        self.reward_models = nn.ModuleList(reward_models)

    def gradient_checkpointing_enable(self):
        for r in self.reward_models:
            r.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        for r in self.reward_models:
            r.gradient_checkpointing_disable()

    def forward(self, *args, **kwargs):
        all_scores = [r(*args, **kwargs) for r in self.reward_models]
        return sum(all_scores) / len(all_scores)


def create_wps_reward_model(
    name: str,
    model_name_or_path: str,
    load_state_dict: bool,
    disable_dropout: bool,
    device: Union[str, torch.device],
):
    module = RewardModel(
        base_model_name_or_path=model_name_or_path,
        load_state_dict=load_state_dict,
        disable_dropout=disable_dropout,
    )
    tokenizer = api.utils.load_hf_tokenizer(model_name_or_path)
    return api.model.Model(name, module, tokenizer, device)


def create_zipped_wps_reward_model(
    name: str,
    model_name_or_paths: List[str],
    disable_dropout: bool,
    device: Union[str, torch.device],
):
    module = ZippedRewardModel([
        RewardModel(
            base_model_name_or_path=model_name_or_path,
            load_state_dict=True,
            disable_dropout=disable_dropout,
        ) for model_name_or_path in model_name_or_paths
    ])
    tokenizers = [api.utils.load_hf_tokenizer(p) for p in model_name_or_paths]
    if not all(tokenizers[0].__class__.__name__ == t.__class__.__name__ for t in tokenizers):
        raise RuntimeError("Tokenizers must be the same!")
    tokenizer = tokenizers[0]
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("wps_reward", create_wps_reward_model)
api.model.register_model("zipped_wps_reward", create_zipped_wps_reward_model)