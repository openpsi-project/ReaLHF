from typing import Any, Dict, List, Optional, Union

import torch
import transformers

import api.model
import api.huggingface


def hf_model_factory(model_cls):

    def create_huggingface_model(
        name: str,
        device: Union[str, torch.device],
        model_name_or_path: str,
        init_from_scratch: bool = False,
        from_pretrained_kwargs: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        quantization_kwargs: Optional[Dict[str, Any]] = None,
    ) -> api.model.Model:
        module = api.huggingface.create_hf_nn(
            model_cls,
            model_name_or_path,
            init_from_scratch,
            from_pretrained_kwargs,
            generation_kwargs,
            quantization_kwargs,
        )
        tokenizer = api.huggingface.load_hf_tokenizer(model_name_or_path)
        return api.model.Model(name, module, tokenizer, device)

    return create_huggingface_model


api.model.register_model("causal_lm", hf_model_factory(transformers.AutoModelForCausalLM))