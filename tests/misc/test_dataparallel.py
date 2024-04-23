import unittest

import torch

import reallm.base.dataparallel as dp
import reallm.base.namedarray as namedarray


def get_packed_namedarray(nseqs):
    input_lens = torch.randint(10, 100, (nseqs,))
    vocab_size = 6000
    x = dict(seq_no_eos_mask=torch.randint(0, 2, (nseqs,)),
             packed_seq=torch.cat([torch.randint(0, vocab_size, (l,)) for l in input_lens]),
             packed_logprobs=torch.cat([torch.randn(l - 1) for l in input_lens]),
             packed_logits_mask=torch.cat([torch.randint(0, 2, (l, vocab_size)) for l in input_lens]),
             prompt_mask=torch.cat([torch.randint(0, 2, (l,)) for l in input_lens]),
             input_lens=input_lens,
             cu_seqlens=torch.cat([input_lens.new_zeros(1), input_lens.cumsum(0)]))
    return namedarray.from_dict(x)


class PackedTest(unittest.TestCase):

    def test_gather_from(self):
        nseqs_batch = torch.randint(1, 10, (5,))
        src = [get_packed_namedarray(i) for i in nseqs_batch]
        res = dp.PackedParallelDataBroker.gather_from(src)
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

    def test_gather_from_dict(self):
        src = [dict(a=1, b=2), dict(a=2, b=1, c=3), dict()]
        res = dp.ParallelDataBroker.gather_from(src)
        self.assertEqual(res, dict(a=1.5, b=1.5, c=3))

    def test_scatter_to(self):
        nseqs_batch = torch.randint(1, 10, (5,))
        src = [get_packed_namedarray(i) for i in nseqs_batch]
        res = dp.PackedParallelDataBroker.gather_from(src)
        splitted = dp.PackedParallelDataBroker.scatter_to(res, n_dp=int(torch.randint(2, 5, (1,))))
        res2 = dp.PackedParallelDataBroker.gather_from(splitted)
        for k, v in res.items():
            assert (v == res2[k]).all()

        # ensure that splitting is deterministic such that different processes can get the same result
        n_dp = int(torch.randint(2, 5, (1,)))
        splitted1 = dp.PackedParallelDataBroker.scatter_to(res, n_dp=n_dp)
        splitted2 = dp.PackedParallelDataBroker.scatter_to(res, n_dp=n_dp)
        for x1, x2 in zip(splitted1, splitted2):
            for k, v1, v2 in zip(x1.keys(), x1.values(), x2.values()):
                assert (v1 == v2).all()


if __name__ == "__main__":
    unittest.main()
