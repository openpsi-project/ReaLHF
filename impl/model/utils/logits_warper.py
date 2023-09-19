from typing import List, Optional, Tuple
import dataclasses

import torch


class LogitsWarper:
    """Abstract base class for all logit processors that can be applied during generation.
    
    Cloned from huggingface transformers/src/transformers/generation/logits_process.py,
    except that we can optionally change the logits or a logits mask inplace.
    """

    def __call__(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        mask: Optional[torch.FloatTensor] = None,
        change_logits: bool = True,
        change_mask: bool = True,
        change_logits_inplace: bool = False,
        change_mask_inplace: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called.")


def _process_logits_mask(
    logits: torch.FloatTensor,
    mask: torch.BoolTensor,
    indices_to_remove: torch.LongTensor,
    change_logits: bool,
    change_mask: bool,
    change_logits_inplace: bool,
    change_mask_inplace: bool,
    filter_value: float,
) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
    if change_logits:
        if change_logits_inplace:
            logits.masked_fill_(indices_to_remove, filter_value)
        else:
            logits = logits.masked_fill(indices_to_remove, filter_value)
    if mask is not None and change_mask:
        if change_mask_inplace:
            mask.masked_fill_(indices_to_remove, 0.0)
        else:
            mask = mask.masked_fill(indices_to_remove, 0.0)
    return logits, mask


@dataclasses.dataclass
class TemperatureLogitsWarper(LogitsWarper):
    temperature: float

    def __post_init__(self):
        if self.temperature > 1.0 or self.temperature < 0.0:
            raise ValueError("temperature has to be between 0 and 1")

    def __call__(
        self,
        _,
        logits: torch.FloatTensor,
        mask: Optional[torch.FloatTensor],
        change_logits: bool = True,
        change_mask: bool = True,
        change_logits_inplace: bool = False,
        change_mask_inplace: bool = True,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        if not change_logits:
            raise ValueError("Temperature logits processor can only be called for changing logits.")
        if change_logits_inplace:
            logits.div_(self.temperature)
        else:
            logits = logits / self.temperature
        return logits / self.temperature, mask


@dataclasses.dataclass
class TopPLogitsWarper(LogitsWarper):
    top_p: float
    filter_value: float = -float("Inf")
    min_tokens_to_keep: int = 1

    def __post_init__(self):
        self.top_p = top_p = float(self.top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(self.min_tokens_to_keep, int) or (self.min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, "
                             f"but is {self.min_tokens_to_keep}")

    def __call__(
        self,
        _,
        logits: torch.FloatTensor,
        mask: Optional[torch.FloatTensor],
        change_logits: bool = True,
        change_mask: bool = True,
        change_logits_inplace: bool = False,
        change_mask_inplace: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        return _process_logits_mask(logits, mask, indices_to_remove, change_logits, change_mask,
                                    change_logits_inplace, change_mask_inplace, self.filter_value)


@dataclasses.dataclass
class TopKLogitsWarper(LogitsWarper):
    top_k: int
    filter_value: float = -float("Inf")
    min_tokens_to_keep: int = 1

    def __post_init__(self):
        top_k = self.top_k
        min_tokens_to_keep = self.min_tokens_to_keep
        self.top_k = max(top_k, min_tokens_to_keep)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

    def __call__(
        self,
        _,
        logits: torch.FloatTensor,
        mask: Optional[torch.FloatTensor],
        change_logits: bool = True,
        change_mask: bool = True,
        change_logits_inplace: bool = False,
        change_mask_inplace: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        top_k = min(self.top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        return _process_logits_mask(logits, mask, indices_to_remove, change_logits, change_mask,
                                    change_logits_inplace, change_mask_inplace, self.filter_value)


@dataclasses.dataclass
class EtaLogitsWarper(LogitsWarper):
    epsilon: float
    filter_value: float = -float("Inf")
    min_tokens_to_keep: int = 1

    def __post_init__(self):
        self.epsilon = epsilon = float(self.epsilon)
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`eta_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        self.min_tokens_to_keep = min_tokens_to_keep = int(self.min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}")

    def __call__(
        self,
        _,
        logits: torch.FloatTensor,
        mask: Optional[torch.FloatTensor],
        change_logits: bool = True,
        change_mask: bool = True,
        change_logits_inplace: bool = False,
        change_mask_inplace: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        # Calculate the adaptive cutoff
        probabilities = logits.softmax(dim=-1)
        entropy = torch.distributions.Categorical(logits).entropy()
        eta = torch.min(self.epsilon, torch.sqrt(self.epsilon) * torch.exp(-entropy))[..., None]
        indices_to_remove = probabilities < eta

        # Keep the words with the 'min_tokens_to_keep'-highest probabilities
        top_k = min(self.min_tokens_to_keep, logits.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (logits < torch.topk(logits, top_k)[0][..., -1, None])

        return _process_logits_mask(logits, mask, indices_to_remove, change_logits, change_mask,
                                    change_logits_inplace, change_mask_inplace, self.filter_value)


def chained_logits_wraper(xs: List[LogitsWarper]):

    def foo(
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        mask: Optional[torch.FloatTensor] = None,
        change_logits: bool = True,
        change_mask: bool = True,
        change_logits_inplace: bool = False,
        change_mask_inplace: bool = False,
    ):
        for x in xs:
            logits, mask = x(input_ids, logits, mask, change_logits, change_mask, change_logits_inplace,
                             change_mask_inplace)
        return logits, mask

    return foo


def unioned_logits_wraper(xs: List[LogitsWarper], filter_value: float = -float("Inf")):

    def foo(
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        mask: Optional[torch.FloatTensor] = None,
        change_logits: bool = True,
        change_mask: bool = True,
        change_logits_inplace: bool = False,
        change_mask_inplace: bool = False,
    ):
        if mask is None:
            mask = torch.ones_like(logits, dtype=torch.bool)
        all_masks = []
        for x in xs:
            _, m = x(input_ids,
                     logits,
                     mask,
                     change_logits=False,
                     change_mask=True,
                     change_logits_inplace=False,
                     change_mask_inplace=False)
            all_masks.append(m)
        mask = torch.stack(all_masks, dim=0).all(dim=0)
        return _process_logits_mask(logits, mask, mask.logical_not(), change_logits, change_mask,
                                    change_logits_inplace, change_mask_inplace, filter_value)

    return foo