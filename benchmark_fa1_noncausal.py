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


KERNELS = {
    "v1": lambda q, k, v: ext.fa1_noncausal_forward(q, k, v),
    "v2": lambda q, k, v: ext.fa1_noncausal_v2_forward(q, k, v),
}


@torch.no_grad()
def check_correctness(B, H, N, d, device):
    q, k, v = make_qkv(B, H, N, d, device)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    ok = True
    for name, fn in KERNELS.items():
        err = (ref - fn(q, k, v)).abs().max().item()
        this_ok = err < 5e-2  # fp32 flash vs SDPA tolerance
        ok = ok and this_ok
        print(f"[N={N:>5}] {name}: max|Δ| = {err:.3e}  ->  {'PASS' if this_ok else 'FAIL'}")
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

    print("\n== Latency: torch SDPA vs custom kernels (non-causal) ==")
    print(f"{'N':>6} | {'torch (ms)':>10} | {'v1 (ms)':>10} | {'v2 (ms)':>10} | "
          f"{'v1/torch':>8} | {'v2/torch':>8} | {'v2 vs v1':>8}")
    print("-" * 78)
    for N in args.seq_lens:
        q, k, v = make_qkv(B, H, N, d, device)

        with torch.no_grad():
            t_torch = cuda_time(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False),
                                iters=args.iters, warmup=args.warmup)
            t_v1 = cuda_time(lambda: KERNELS["v1"](q, k, v),
                             iters=args.iters, warmup=args.warmup)
            t_v2 = cuda_time(lambda: KERNELS["v2"](q, k, v),
                             iters=args.iters, warmup=args.warmup)

        print(f"{N:>6} | {t_torch:10.4f} | {t_v1:10.4f} | {t_v2:10.4f} | "
              f"{t_torch / t_v1:7.2f}x | {t_torch / t_v2:7.2f}x | {t_v1 / t_v2:7.2f}x")

    print("\n(x/torch = torch_ms / kernel_ms; > 1x means the kernel is faster than SDPA)")


if __name__ == "__main__":
    main()
