from typing import Optional
import random
import unittest

from transformers.generation.utils import top_k_top_p_filtering
import torch

from impl.model.utils.logits_warper import (chained_logits_wraper, TopKLogitsWarper, TopPLogitsWarper,
                                            unioned_logits_wraper)

vocab_size = 5000


def chained_top_k_top_p(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    return chained_logits_wraper([
        TopKLogitsWarper(top_k=top_k, filter_value=filter_value),
        TopPLogitsWarper(top_p=top_p, filter_value=filter_value)
    ],)(None, logits, change_mask=True, change_mask_inplace=True, mask=torch.ones_like(logits))


def unioned_top_k_top_p(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    return unioned_logits_wraper([
        TopKLogitsWarper(top_k=top_k, filter_value=filter_value),
        TopPLogitsWarper(top_p=top_p, filter_value=filter_value)
    ],
                                 filter_value=filter_value)(None,
                                                            logits,
                                                            change_mask=True,
                                                            change_mask_inplace=True,
                                                            mask=torch.ones_like(logits))


def generate_logits_ignoring_mask(logits: torch.FloatTensor,
                                  top_p: Optional[float] = 1.0,
                                  top_k: Optional[int] = -1) -> torch.BoolTensor:
    if top_p is None:
        top_p = 1.0
    if top_k is None:
        top_k = -1
    assert 0 < top_p <= 1.0
    if top_k < 0 or top_k > logits.size(-1):
        top_k = logits.size(-1)
    if top_p == 1.0 and top_k == logits.size(-1):
        return None

    sorted_logits, sorted_indices = torch.sort(logits, descending=False, dim=-1)
    sorted_logits: torch.FloatTensor
    sorted_indices: torch.LongTensor
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    top_p_indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)

    # Remove all tokens with a probability less than the last token of the top-k
    top_k_indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]

    return top_p_indices_to_remove.logical_or(top_k_indices_to_remove).bool()


class LogitsWarperTest(unittest.TestCase):

    def testChained(self):
        p = random.random()
        k = random.randint(100, vocab_size)
        logits = torch.randn(2, 100, vocab_size)

        x2, m = chained_top_k_top_p(logits, top_k=k, top_p=p, filter_value=-100)
        x3 = top_k_top_p_filtering(logits.flatten(end_dim=-2), top_k=k, top_p=p,
                                   filter_value=-100).view(2, 100, vocab_size)
        assert torch.allclose(x3, x2), (x3 - x2).abs().max()
        x4 = logits.masked_fill((1 - m).bool(), -100)
        assert torch.allclose(x2, x4), (x2 - x4).abs().max()

    def testUnioned(self):
        p = random.random()
        k = random.randint(100, vocab_size)
        logits = torch.randn(2, 100, vocab_size)

        m1 = generate_logits_ignoring_mask(logits, top_k=k, top_p=p)
        _, m2 = unioned_top_k_top_p(logits, top_k=k, top_p=p, filter_value=-100)
        assert torch.allclose(m1, m2.logical_not())


if __name__ == "__main__":
    unittest.main()