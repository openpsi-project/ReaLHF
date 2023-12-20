import itertools
import multiprocessing as mp
import unittest

import torch

from base.namedarray import *
from base.topology import *
from system.request_reply_stream import *
import api.config

EXP_NAME = "test_exp"
TRIAL_NAME = "test_trial"
SERIALIZATION_METHOD = "raw_bytes"
DP_SIZE = 2
PP_SIZE = 3
MP_SIZE = 4
TOPO = PipeModelDataParallelTopology(num_pp=PP_SIZE, num_dp=DP_SIZE, num_mp=MP_SIZE)
MODEL_NAME = "default"


class TestSocketStream(unittest.TestCase):

    def buildIPStreams(self):
        self.master_streams = [IpRequestClient(SERIALIZATION_METHOD) for _ in range(DP_SIZE)]
        self.model_streams = {}
        for i, j, k in itertools.product(range(DP_SIZE), range(PP_SIZE), range(MP_SIZE)):
            sid = api.config.ModelStreamID(MODEL_NAME, dp_rank=i, pp_rank=j, mp_rank=k)
            s = IpReplyServer(SERIALIZATION_METHOD)
            is_dp_head = j == PP_SIZE - 1 and k == 0
            s.accept(self.master_streams[i].address, is_dp_head=is_dp_head)
            self.model_streams[sid] = s
            if is_dp_head:
                self.master_streams[i].accept(s.address)

    def _test_post_poll(self):
        payloads1 = [
            Payload(
                handle_name="generate",
                data=namedarray.from_dict({
                    "a": torch.randn(1000),
                    "b": torch.randn(10000),
                    "logits_mask": torch.zeros(4096, 32000, dtype=torch.bool),
                }),
            ) for _ in range(DP_SIZE)
        ]
        payloads2 = [
            Payload(
                handle_name="generate",
                data=namedarray.from_dict({
                    "a": torch.randn(1000),
                    "b": torch.randn(10000),
                    "logits_mask": torch.zeros(4096, 32000, dtype=torch.bool),
                }),
            ) for _ in range(DP_SIZE)
        ]

        for m in self.model_streams.values():
            with self.assertRaises(NoMessage):
                m.poll(block=False)

        for mas, p in zip(self.master_streams, payloads1):
            mas.post(p)
        for mas, p in zip(self.master_streams, payloads2):
            mas.post(p)

        for payloads in [payloads1, payloads2]:
            for k, m in self.model_streams.items():
                is_dp_head = k.pp_rank == PP_SIZE - 1 and k.mp_rank == 0
                r = m.poll(block=True)
                if not is_dp_head:
                    assert r.data is None
                else:
                    self._assertPayloadEqual(r, payloads[k.dp_rank])

        for m in self.model_streams.values():
            with self.assertRaises(NoMessage):
                m.poll(block=False)

        payloads1 = [
            Payload(
                handle_name="generate",
                data=namedarray.from_dict({
                    "a": torch.randn(1000),
                    "b": torch.randn(10000)
                }),
            ) for _ in range(DP_SIZE)
        ]
        payloads2 = [
            Payload(
                handle_name="generate",
                data=namedarray.from_dict({
                    "a": torch.randn(1000),
                    "b": torch.randn(10000)
                }),
            ) for _ in range(DP_SIZE)
        ]
        for mas in self.master_streams:
            with self.assertRaises(NoMessage):
                mas.poll(block=False)
        for payloads in [payloads1, payloads2]:
            for k, m in self.model_streams.items():
                is_dp_head = k.pp_rank == PP_SIZE - 1 and k.mp_rank == 0
                if is_dp_head:
                    m.post(payloads[k.dp_rank])
        for payloads in [payloads1, payloads2]:
            for mas, p in zip(self.master_streams, payloads):
                r = mas.poll(block=True)
                self._assertPayloadEqual(r, p)

    def _assertPayloadEqual(self, p1: Payload, p2: Payload):
        self.assertEqual(p1.handle_name, p2.handle_name)
        self.assertEqual(p1.request_id, p2.request_id)
        self.assertEqual(list(p1.data.keys()), list(p2.data.keys()))
        for k in p1.data.keys():
            self.assertTrue(torch.allclose(p1.data[k], p2.data[k]))

    def testNameResolvingStream(self):
        name_resolve.clear_subtree(names.trial_root(EXP_NAME, TRIAL_NAME))
        worker_info = api.config.WorkerInformation(experiment_name=EXP_NAME, trial_name=TRIAL_NAME)
        mas_configs = [
            api.config.RequestReplyStream(
                push_stream_name=f"dp_{i}",
                pull_stream_name=str(
                    api.config.ModelStreamID(MODEL_NAME, dp_rank=i, mp_rank=0, pp_rank=PP_SIZE - 1)),
            ) for i in range(DP_SIZE)
        ]
        w_cfgs = {
            (i, j, k): api.config.RequestReplyStream(
                push_stream_name=str(api.config.ModelStreamID(MODEL_NAME, dp_rank=i, mp_rank=k, pp_rank=j)),
                pull_stream_name=f"dp_{i}",
            )
            for i, j, k in itertools.product(range(DP_SIZE), range(PP_SIZE), range(MP_SIZE))
        }

        def master_proc(dp_rank):
            stream = make_master_stream(worker_info, mas_configs[dp_rank], n_subscribers=PP_SIZE * MP_SIZE)
            p1 = Payload(
                handle_name="generate",
                data=namedarray.from_dict({
                    "a": torch.randn(1000),
                    "b": torch.randn(10000)
                }),
            )
            p2 = Payload(
                handle_name="train",
                data=namedarray.from_dict({
                    "a": torch.randn(1000),
                    "b": torch.randn(10000)
                }),
            )

            stream.post(p1)
            stream.post(p2)

            while True:
                try:
                    r1 = stream.poll()
                    break
                except NoMessage:
                    pass
            self._assertPayloadEqual(r1, p2)
            r2 = stream.poll(block=True)
            self._assertPayloadEqual(r2, p1)

        def model_proc(dp_rank, pp_rank, mp_rank):
            stream = make_worker_stream(
                worker_info,
                w_cfgs[(dp_rank, pp_rank, mp_rank)],
                is_dp_head=(pp_rank == PP_SIZE - 1 and mp_rank == 0),
            )
            p1 = stream.poll(block=True)
            p2 = stream.poll(block=True)

            time.sleep(1)
            if pp_rank == PP_SIZE - 1 and mp_rank == 0:
                assert p2.data is not None
                assert p1.data is not None
                stream.post(p2)
                stream.post(p1)
            else:
                assert p1.shapes is not None and p2.shapes is not None
                assert p1.dtypes is not None and p2.dtypes is not None
                assert p1.data is None and p2.data is None

        p1s = [mp.Process(target=master_proc, args=(i,)) for i in range(DP_SIZE)]
        p2s = [
            mp.Process(target=model_proc, args=(i, j, k))
            for i, j, k in itertools.product(range(DP_SIZE), range(PP_SIZE), range(MP_SIZE))
        ]
        for p1 in p1s:
            p1.start()
        for p2 in p2s:
            p2.start()
        for p1 in p1s:
            p1.join()
        for p2 in p2s:
            p2.join()

    def test_post_poll(self):
        self.buildIPStreams()
        self._test_post_poll()
        for m in self.model_streams.values():
            m.close()
        for mas in self.master_streams:
            mas.close()


if __name__ == "__main__":
    unittest.main()
