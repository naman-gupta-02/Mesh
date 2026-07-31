"""Tensor <-> bytes for gRPC message fields.

torch.save/load round-trips shape, dtype, and data in one call, which is
plenty for Phase 2 (correctness over wire format efficiency). A tighter
raw-buffer encoding is a later optimization once bandwidth, not shape/dtype
bookkeeping, is the bottleneck.
"""

import io

import torch


def encode(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def decode(data: bytes) -> torch.Tensor:
    buffer = io.BytesIO(data)
    return torch.load(buffer, weights_only=True)
