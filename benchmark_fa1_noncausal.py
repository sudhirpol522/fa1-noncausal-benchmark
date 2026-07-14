"""
benchmark_fa1_noncausal.py
Benchmark the custom non-causal FA1 CUDA kernel
(fa1-non-causal-alternate-implemetation.cu) against PyTorch's
F.scaled_dot_product_attention (non-causal), and report the speedup.

Run the correctness check first and confirm PASS before trusting the timings.

Usage:
    python benchmark_fa1_noncausal.py
    python benchmark_fa1_noncausal.py --batch 8 --heads 12 --head_dim 64 \
        --seq_lens 128 256 512 1024 --iters 50 --warmup 10
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))

ext = load(
    name="fa1_noncausal_ext",
    sources=[os.path.join(_HERE, "fa1_noncausal_ext.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)


def cuda_time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def make_qkv(B, H, N, d, device):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, d, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N, d, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N, d, device=device, dtype=torch.float32)
    return q, k, v


@torch.no_grad()
def check_correctness(B, H, N, d, device):
    q, k, v = make_qkv(B, H, N, d, device)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = ext.fa1_noncausal_forward(q, k, v)
    err = (ref - out).abs().max().item()
    ok = err < 5e-2  # fp32 flash vs SDPA tolerance
    print(f"[N={N:>5}] max|Δ| = {err:.3e}  ->  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA device required"
    device = "cuda"
    B, H, d = args.batch, args.heads, args.head_dim

    print("== Correctness: custom FA1 (non-causal) vs torch SDPA (non-causal) ==")
    all_ok = True
    for N in args.seq_lens:
        all_ok = check_correctness(B, H, N, d, device) and all_ok
    print("RESULT:", "PASS" if all_ok else "FAIL (inspect kernel before trusting timings)")

    print("\n== Latency: torch SDPA vs custom FA1 (non-causal) ==")
    print(f"{'N':>6} | {'torch (ms)':>12} | {'cuda (ms)':>12} | {'speedup':>8}")
    print("-" * 48)
    for N in args.seq_lens:
        q, k, v = make_qkv(B, H, N, d, device)

        with torch.no_grad():
            t_torch = cuda_time(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False),
                                iters=args.iters, warmup=args.warmup)
            t_cuda = cuda_time(lambda: ext.fa1_noncausal_forward(q, k, v),
                               iters=args.iters, warmup=args.warmup)

        speedup = t_torch / t_cuda
        print(f"{N:>6} | {t_torch:12.4f} | {t_cuda:12.4f} | {speedup:7.2f}x")

    print("\n(speedup = torch_ms / cuda_ms; > 1x means the custom kernel is faster)")


if __name__ == "__main__":
    main()
