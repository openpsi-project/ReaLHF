from typing import Any, Dict, List, Optional, Union
import logging
import os

import torch
import torch.nn as nn
import transformers

import api.model
import api.utils

logger = logging.getLogger("Reward Model")


class RewardModel(nn.Module):

    def __init__(self,
                 base_model_name_or_path: str,
                 from_pretrained_kwargs: Dict[str, Any],
                 quantization_kwargs: Dict[str, Any],
                 output_scaling: float,
                 output_bias: float,
                 init_from_scratch: bool = False):
        super().__init__()
        base_model = api.utils.create_hf_nn(
            transformers.AutoModel,
            model_name_or_path=base_model_name_or_path,
            init_from_scratch=init_from_scratch,
            from_pretrained_kwargs=from_pretrained_kwargs,
            quantization_kwargs=quantization_kwargs,
        )

        self.output_scaling = output_scaling
        self.output_bias = output_bias

        self.config = base_model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(self.config,
                                                                "hidden_size") else self.config.n_embd
        # base_model is created by `AutoModel` instead of `AutoModelForCausalLM`
        # its forward function outputs the last hidden state instead of logits
        self.rwtranrsformer = base_model

        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False, dtype=torch.float16).to(base_model.device)

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def to(self, *args, **kwargs):
        # Checks if the model has been loaded in 8-bit
        if getattr(self.rwtranrsformer, "is_quantized", False):
            raise ValueError(
                "`.to` is not supported for `4-bit` or `8-bit` models. Please use the model as it is, since the"
                " model has already been set to the correct devices and casted to the correct `dtype`.")
        else:
            return super().to(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=False,
    ) -> torch.Tensor:
        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0]
        scores = self.v_head(hidden_states.to(self.v_head.weight.dtype)).squeeze(-1)
        return (scores - self.output_bias) * self.output_scaling


def create_wps_reward_model(
    name: str,
    device: Union[str, torch.device],
    model_name_or_path: str,
    from_pretrained_kwargs: Optional[Dict[str, Any]] = None,
    quantization_kwargs: Optional[Dict[str, Any]] = None,
    output_scaling: float = 1.0,
    output_bias: float = 0.0,
    load_v_head_path: Optional[str] = None,
    init_from_scratch: bool = False,
) -> api.model.Model:
    module = RewardModel(
        base_model_name_or_path=model_name_or_path,
        from_pretrained_kwargs=from_pretrained_kwargs,
        quantization_kwargs=quantization_kwargs,
        output_bias=output_bias,
        output_scaling=output_scaling,
        init_from_scratch=init_from_scratch,
    )

    if load_v_head_path is not None:
        v_head_sd: Dict = torch.load(load_v_head_path, map_location='cpu')
        module.v_head.load_state_dict(v_head_sd)

    tokenizer = api.utils.load_hf_tokenizer(model_name_or_path)
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("wps_reward", create_wps_reward_model)
