"""Check that PyTorch sees the GPU and measure the two numbers the time estimates in SETUP.md depend on: bf16 matmul
throughput (sets how fast a big batch of tokens goes through the model) and memory bandwidth (sets how fast one
sequence generates). Prints them next to the published figures so the estimates can be redone with measured values.

    python gpu/check_gpu.py
"""
import time

import torch


def timed(fn, reps):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    print(f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}, available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot see a CUDA GPU: see SETUP.md, step 3")
    p = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info()
    print(f"device: {p.name}, compute capability {cap[0]}.{cap[1]}, {p.multi_processor_count} SMs")
    print(f"memory CUDA can see: {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free")
    print(f"architectures this torch build has kernels for: {torch.cuda.get_arch_list()}")
    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list() and f"sm_{cap[0]}0" not in torch.cuda.get_arch_list():
        print("WARNING: no kernels built for this GPU's architecture; use the cu130 (or newer) wheels, see SETUP.md")

    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    dt = timed(lambda: a @ b, 20)
    tflops = 2 * n**3 / dt / 1e12
    print(f"bf16 matmul {n}x{n}: {tflops:.0f} TFLOPS")

    x = torch.empty(2 * 2**30 // 2, device="cuda", dtype=torch.bfloat16)  # 2 GiB
    y = torch.empty_like(x)
    dt = timed(lambda: y.copy_(x), 10)
    gbs = 2 * x.numel() * 2 / dt / 1e9  # read + write
    print(f"device copy: {gbs:.0f} GB/s (read + write)")

    print("\nFor comparison: DGX Spark is specified at 273 GB/s; RTX Spark reviews measured ~309 GB/s reads.")
    print("Rough forward-pass rate for a dense model of P billion parameters with big batches:")
    for name, params in (("Qwen3.5-4B", 4.2), ("Qwen3-8B", 8.2), ("Qwen3-32B", 32.8)):
        rate = 0.6 * tflops * 1e12 / (2 * params * 1e9 * 1.2)  # 60% of matmul peak, 20% for the recording hooks
        print(f"  {name:11s} ~{rate:,.0f} tok/s -> 1B tokens in ~{1e9 / rate / 86400:.1f} days "
              f"(single-stream generation <= {gbs / (2 * params):.1f} tok/s)")


if __name__ == "__main__":
    main()
