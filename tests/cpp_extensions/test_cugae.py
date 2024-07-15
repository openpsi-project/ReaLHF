import time

import pytest
import torch

from realhf.impl.model.utils.ppo_functional import (
    cugae1d_nolp_misalign_func,
    cugae2d_nolp_func,
    cugae2d_olp_func,
    pygae1d_nolp_misalign,
    pygae2d_nolp,
    pygae2d_olp,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires a GPU.")
@pytest.mark.parametrize("max_seqlen", [32, 128, 512])
@pytest.mark.parametrize("bs", [2, 4])
@pytest.mark.parametrize("gamma", [0.9, 1.0])
@pytest.mark.parametrize("lam", [0.5, 1.0])
@pytest.mark.gpu
def test_gae1d_nolp_misalign(max_seqlen: int, bs: int, gamma: float, lam: float):

    seqlens = torch.randint(1, max_seqlen, (bs,), dtype=torch.int32, device="cuda")
    rewards = torch.randn(seqlens.sum(), dtype=torch.float32, device="cuda")
    values = torch.randn(seqlens.sum() + bs, dtype=torch.float32, device="cuda")
    bootstrap = torch.ones(bs, dtype=torch.bool, device="cuda")
    cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).int()

    adv, ret = cugae1d_nolp_misalign_func(
        rewards, values, cu_seqlens, bootstrap, gamma, lam
    )
    py_adv, py_ret = pygae1d_nolp_misalign(
        rewards, values, cu_seqlens, bootstrap, gamma, lam
    )

    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    py_adv, py_ret = pygae1d_nolp_misalign(
        rewards, values, cu_seqlens, bootstrap, gamma, lam
    )
    torch.cuda.synchronize()
    t2 = time.perf_counter_ns()
    adv, ret = cugae1d_nolp_misalign_func(
        rewards, values, cu_seqlens, bootstrap, gamma, lam
    )
    torch.cuda.synchronize()
    t3 = time.perf_counter_ns()

    assert torch.allclose(adv, py_adv, atol=1e-5), (adv - py_adv).abs().max()
    assert torch.allclose(ret, py_ret, atol=1e-5), (ret - py_ret).abs().max()

    print(
        f"max_seqlen={max_seqlen},bs={bs}, CUDA acceleration ratio",
        (t2 - t1) / (t3 - t2),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires a GPU.")
@pytest.mark.parametrize("seqlen", [32, 128, 512, 1024])
@pytest.mark.parametrize("bs", [8, 16, 32, 100])
@pytest.mark.parametrize("gamma", [0.9, 1.0])
@pytest.mark.parametrize("lam", [0.5, 1.0])
def test_gae2d_olp(bs: int, seqlen: int, gamma: float, lam: float):

    torch.random.manual_seed(0)
    rewards = torch.randn(bs, seqlen).cuda()
    values = torch.randn(bs, seqlen + 1).cuda()
    dones = torch.randint(0, 2, (bs, seqlen + 1)).bool().cuda()
    truncates = dones.logical_and(torch.randint(0, 2, (bs, seqlen + 1)).bool().cuda())

    py_adv, py_ret = pygae2d_olp(rewards, values, dones, truncates, gamma, lam)
    adv, ret = cugae2d_olp_func(rewards, values, dones, truncates, gamma, lam)

    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    py_adv, py_ret = pygae2d_olp(rewards, values, dones, truncates, gamma, lam)
    torch.cuda.synchronize()
    t2 = time.perf_counter_ns()
    adv, ret = cugae2d_olp_func(rewards, values, dones, truncates, gamma, lam)
    torch.cuda.synchronize()
    t3 = time.perf_counter_ns()

    assert torch.allclose(adv, py_adv, atol=1e-5), (adv - py_adv).abs().max()
    assert torch.allclose(ret, py_ret, atol=1e-5), (ret - py_ret).abs().max()
    print(
        f"seqlen={seqlen},bs={bs}, CUDA acceleration ratio",
        (t2 - t1) / (t3 - t2),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires a GPU.")
@pytest.mark.parametrize("seqlen", [32, 128, 512, 1024])
@pytest.mark.parametrize("bs", [8, 16, 32, 100])
@pytest.mark.parametrize("gamma", [0.9, 1.0])
@pytest.mark.parametrize("lam", [0.5, 1.0])
def test_gae2d_nolp(bs: int, seqlen: int, gamma: float, lam: float):

    torch.random.manual_seed(0)
    rewards = torch.randn(bs, seqlen).cuda()
    values = torch.randn(bs, seqlen + 1).cuda()
    on_reset_ = torch.randint(0, 2, (bs, seqlen + 2)).bool().cuda()
    on_reset = on_reset_[:, :-1].contiguous()
    truncates = (
        on_reset_[:, 1:]
        .logical_and(torch.randint(0, 2, (bs, seqlen + 1)).bool().cuda())
        .contiguous()
    )

    py_adv, py_ret = pygae2d_nolp(rewards, values, on_reset, truncates, gamma, lam)
    adv, ret = cugae2d_nolp_func(rewards, values, on_reset, truncates, gamma, lam)

    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    py_adv, py_ret = pygae2d_nolp(rewards, values, on_reset, truncates, gamma, lam)
    torch.cuda.synchronize()
    t2 = time.perf_counter_ns()
    adv, ret = cugae2d_nolp_func(rewards, values, on_reset, truncates, gamma, lam)
    torch.cuda.synchronize()
    t3 = time.perf_counter_ns()

    adv = adv * (1 - truncates[:, :-1].float())
    py_adv = py_adv * (1 - truncates[:, :-1].float())
    ret = ret * (1 - truncates[:, :-1].float())
    py_ret = py_ret * (1 - truncates[:, :-1].float())
    assert torch.allclose(adv, py_adv, atol=1e-5), (adv - py_adv).abs().max()
    assert torch.allclose(ret, py_ret, atol=1e-5), (ret - py_ret).abs().max()
    print(
        f"seqlen={seqlen},bs={bs}, CUDA acceleration ratio",
        (t2 - t1) / (t3 - t2),
    )
