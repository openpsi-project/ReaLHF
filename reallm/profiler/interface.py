from collections import defaultdict
from typing import Dict, List, Optional
import time

from deepspeed.runtime.engine import DeepSpeedEngine
import deepspeed
import torch
import torch.utils.data
import tqdm

from reallm.impl.model.utils.functional import (build_leave_one_indices, build_shift_one_indices,
                                                gather_packed_shifted_log_probs)
from reallm.profiler.engine import ProfileEngine
import reallm.base.constants as constants


def compute_packed_sft_loss(
    logits: torch.Tensor,
    packed_input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prompt_mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # **kwargs is used to ensure the correctness of invoking this function
    shift_one_indices = build_shift_one_indices(logits, cu_seqlens)
    logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, packed_input_ids).float()
    prompt_mask = prompt_mask[shift_one_indices]
    # float16 will overflow here
    loss = -torch.where(prompt_mask, 0, logprobs).sum() / (prompt_mask.numel() - prompt_mask.count_nonzero())
    return loss, {"loss": loss.detach()}
