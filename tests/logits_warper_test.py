from typing import Optional
import random
import unittest

from transformers.generation.utils import top_k_top_p_filtering
import torch

from impl.model.utils.logits_warper import top_k_top_p_logits

vocab_size = 100


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

        filter_value = torch.finfo(logits.dtype).min
        x2 = top_k_top_p_logits(logits, top_k=k, top_p=p, inplace=False, ordered=True)
        x3 = top_k_top_p_filtering(logits.flatten(end_dim=-2), top_k=k, top_p=p,
                                   filter_value=filter_value).view(2, 100, vocab_size)
        assert torch.allclose(x3, x2), (x3 - x2).abs().max()

    def testUnioned(self):
        p = random.random()
        k = random.randint(10, vocab_size)
        logits = torch.randn(2, 10, vocab_size)

        m1 = generate_logits_ignoring_mask(logits, top_k=k, top_p=p)
        top_k_top_p_logits(logits, top_k=k, top_p=p, inplace=True, ordered=False)
        assert torch.allclose(m1, logits == torch.finfo(logits.dtype).min), (m1, logits == torch.finfo(
            logits.dtype).min)


if __name__ == "__main__":
    unittest.main()
