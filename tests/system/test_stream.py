import unittest
import torch
from system.request_reply_stream import *
from base.namedarray import *
import api.config
import multiprocessing as mp

EXP_NAME = "test_exp"
TRIAL_NAME = "test_trial"
PUSH_STREAM_NAME = "push_stream"
PULL_STREAM_NAMES = ["pull_stream1", "pull_stream2"]


class TestSocketStream(unittest.TestCase):
    def buildIPStreams(self):
        self.master_stream = IpRequestReplyMasterStream("pickle_compress")
        self.model_streams = {
            "a": IpRequestReplyWorkerStream("pickle_compress"),
            "b": IpRequestReplyWorkerStream("pickle_compress"),
        }
        for model_id, stream in self.model_streams.items():
            stream.accept("master", self.master_stream.address)
            self.master_stream.accept(model_id, stream.address)

    def _test_post_poll(self):
        k1, k2 = "a", "b"
        w1, w2 = self.model_streams["a"], self.model_streams["b"]
        mas = self.master_stream

        p1 = Payload(
            handle_name="generate",
            data=namedarray.from_dict({"a": torch.randn(1000), "b": torch.randn(10000)}),
        )
        p2 = Payload(
            handle_name="generate",
            data=namedarray.from_dict({"a": torch.randn(1000), "b": torch.randn(10000)}),
        )

        for w in [w1, w2]:
            with self.assertRaises(NoMessage):
                w.poll(block=False)
        mas.post(p1)
        mas.post(p2)
        for w in [w1, w2]:
            r = w.poll(block=True)
            self._assertPayloadEqual(p1, r)
            r = w.poll(block=True)
            self._assertPayloadEqual(p2, r)
            with self.assertRaises(NoMessage):
                w.poll(block=False)

        p1 = Payload(
            handle_name="generate",
            data=namedarray.from_dict({"a": torch.randn(1000), "b": torch.randn(10000)}),
        )
        p2 = Payload(
            handle_name="generate",
            data=namedarray.from_dict({"a": torch.randn(1000), "b": torch.randn(10000)}),
        )
        for k in k1, k2:
            with self.assertRaises(NoMessage):
                mas.poll(socket_name=k, block=False)
        w1.post(p1)
        w2.post(p2)
        r1, r2 = list(mas.poll_all_blocked().values())
        self._assertPayloadEqual(p1, r1)
        self._assertPayloadEqual(p2, r2)
        for k in k1, k2:
            with self.assertRaises(NoMessage):
                mas.poll(socket_name=k, block=False)

    def _assertPayloadEqual(self, p1: Payload, p2: Payload):
        self.assertEqual(p1.handle_name, p2.handle_name)
        self.assertEqual(p1.request_id, p2.request_id)
        self.assertEqual(list(p1.data.keys()), list(p2.data.keys()))
        for k in p1.data.keys():
            self.assertTrue(torch.allclose(p1.data[k], p2.data[k]))

    def testNameResolvingStream(self):
        name_resolve.clear_subtree(names.trial_root(EXP_NAME, TRIAL_NAME))
        worker_info = api.config.WorkerInformation(experiment_name=EXP_NAME, trial_name=TRIAL_NAME)
        mas_config = api.config.RequestReplyStream(
            push_stream_name=PUSH_STREAM_NAME,
            pull_stream_names=PULL_STREAM_NAMES,
        )
        w_cfgs = [
            api.config.RequestReplyStream(
                push_stream_name=x,
                pull_stream_names=[PUSH_STREAM_NAME],
            )
            for x in PULL_STREAM_NAMES
        ]

        def master_proc():
            stream = make_master_stream(worker_info, mas_config)
            p1 = Payload(
                handle_name="generate",
                data=namedarray.from_dict({"a": torch.randn(1000), "b": torch.randn(10000)}),
            )
            p2 = Payload(
                handle_name="train",
                data=namedarray.from_dict({"a": torch.randn(1000), "b": torch.randn(10000)}),
            )

            stream.post(p1)
            stream.post(p2)

            r1s = stream.poll_all_blocked()
            for r1 in r1s.values():
                self._assertPayloadEqual(r1, p2)
            r2s = stream.poll_all_blocked()
            for r2 in r2s.values():
                self._assertPayloadEqual(r2, p1)

        def model_proc(idx):
            time.sleep(2)
            stream = make_worker_stream(worker_info, w_cfgs[idx])
            p1 = stream.poll(block=True)
            p2 = stream.poll(block=True)
            time.sleep(1)
            stream.post(p2)
            stream.post(p1)

        p1 = mp.Process(target=master_proc)
        p2s = [mp.Process(target=model_proc, args=(i,)) for i in range(len(w_cfgs))]
        p1.start()
        for p2 in p2s:
            p2.start()
        p1.join()
        for p2 in p2s:
            p2.join()

    def test_post_poll(self):
        self.buildIPStreams()
        self._test_post_poll()
        for m in self.model_streams.values():
            m.close()
        self.master_stream.close()


if __name__ == "__main__":
    unittest.main()
