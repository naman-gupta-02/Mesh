"""Per-node throughput profiling.

Phase 1 runs every simulated node as a process on one machine, so a real
matmul benchmark alone can't reproduce cross-device heterogeneity (different
GPUs, thermal throttling, memory bandwidth). `simulated_scale` stands in for
that gap until Phase 2 puts shards on real, distinct hardware.
"""

import time

import torch


def benchmark_matmul(size: int = 512, iters: int = 20) -> float:
    """Real matmul micro-benchmark for this process. Returns estimated GFLOPS."""
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    for _ in range(3):
        a @ b

    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    elapsed = time.perf_counter() - start

    flops = 2 * size**3 * iters
    return flops / elapsed / 1e9


def measure_throughput(simulated_scale: float = 1.0) -> float:
    """Throughput estimate for this node, in (fake) GFLOPS.

    simulated_scale models the hardware a real volunteer device would report
    (e.g. a ThinkPad CPU vs. an M1 GPU) that a single dev machine can't
    otherwise produce.
    """
    return benchmark_matmul() * simulated_scale
