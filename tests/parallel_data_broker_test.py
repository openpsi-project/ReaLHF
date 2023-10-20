import unittest

import torch

import base.dataparallel as dp
import base.namedarray as namedarray


def get_packed_namedarray(nseqs):
    input_lens = torch.randint(10, 100, (nseqs,))
    vocab_size = 6000
    x = dict(
        seq_no_eos_mask=torch.randint(0, 2, (nseqs,)),
        packed_seq=torch.cat([torch.randint(0, vocab_size, (l,)) for l in input_lens]),
        packed_logprobs=torch.cat([torch.randn(l - 1) for l in input_lens]),
        packed_logits_mask=torch.cat([torch.randint(0, 2, (l, vocab_size)) for l in input_lens]),
        prompt_mask=torch.cat([torch.randint(0, 2, (l,)) for l in input_lens]),
        input_lens=input_lens,
    )
    return namedarray.from_dict(x)


class PackedTest(unittest.TestCase):

    def test_gather_from(self):
        nseqs_batch = torch.randint(1, 10, (5,))
        src = [get_packed_namedarray(i) for i in nseqs_batch]
        res = dp.PackedParallelDataRouter.gather_from(src)
        gathered_input_lens = torch.cat([x['input_lens'] for x in src])
        batch_input_lens = [sum(x['input_lens']) for x in src]
        offset = short1offset = 0
        batch_idx = 0
        batch_offset = batch_short1offset = 0
        batch_inner_idx = 0
        for i, l in enumerate(gathered_input_lens):
            assert (res['seq_no_eos_mask'][i] == src[batch_idx]['seq_no_eos_mask'][batch_inner_idx]).all(), (
                res['seq_no_eos_mask'],)
            assert (res['input_lens'][i] == src[batch_idx]['input_lens'][batch_inner_idx]).all(), (
                res['input_lens'], src[batch_idx]['input_lens'])

            assert (res['packed_seq'][offset:offset +
                                      l] == src[batch_idx]['packed_seq'][batch_offset:batch_offset +
                                                                         l]).all()
            assert (res['packed_logits_mask'][offset:offset + l] == src[batch_idx]['packed_logits_mask']
                    [batch_offset:batch_offset + l]).all()
            assert (res['prompt_mask'][offset:offset +
                                       l] == src[batch_idx]['prompt_mask'][batch_offset:batch_offset +
                                                                           l]).all()

            assert (res['packed_logprobs'][short1offset:short1offset + l - 1] == src[batch_idx]
                    ['packed_logprobs'][batch_short1offset:batch_short1offset + l - 1]).all()

            offset += l
            short1offset += l - 1
            batch_offset += l
            batch_short1offset += l - 1
            batch_inner_idx += 1
            if batch_offset >= batch_input_lens[batch_idx]:
                batch_idx += 1
                batch_offset = batch_short1offset = batch_inner_idx = 0

    def test_scatter_to(self):
        nseqs_batch = torch.randint(1, 10, (5,))
        src = [get_packed_namedarray(i) for i in nseqs_batch]
        res = dp.PackedParallelDataRouter.gather_from(src)
        splitted = dp.PackedParallelDataRouter.scatter_to(res, n_dp=int(torch.randint(1, 5, (1,))))
        res2 = dp.PackedParallelDataRouter.gather_from(splitted)
        for k, v in res.items():
            assert (v == res2[k]).all()


if __name__ == "__main__":
    unittest.main()