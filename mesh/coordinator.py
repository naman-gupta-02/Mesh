"""Scheduler: turns node throughput profiles into a layer-shard assignment.

Naive equal-layer-count partitioning is wrong for heterogeneous hardware — a
slow node holding as many layers as a fast one becomes the pipeline's
bottleneck stage. Instead we give each node a layer count proportional to
its share of total measured throughput, so every pipeline stage takes
roughly the same wall-clock time (a greedy heuristic, not an ILP solve).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeProfile:
    node_id: str
    throughput: float  # relative units (fake GFLOPS); higher = faster


@dataclass(frozen=True)
class ShardAssignment:
    node_id: str
    layer_start: int
    layer_end: int

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start


def model_layer_count(model_name: str) -> int:
    """How many transformer blocks `model_name` has, without downloading its
    weights -- just the (tiny) config file. Lets the coordinator support any
    GPT-2-family checkpoint (gpt2, gpt2-medium, distilgpt2, ...) instead of
    a hardcoded layer count.

    This is also the fast up-front validation gate before a model switch
    tears down a working cluster (see dashboard_server.switch_model()), so
    it needs to actually fail on a bad identifier. transformers'
    GPT2Config.from_pretrained() doesn't: given a local directory that
    exists but has no config.json (e.g. an incomplete upload), it silently
    falls back to a *default* GPT2Config instead of raising -- which would
    make every switch to an incomplete local model "succeed" with the
    wrong architecture. Guard that case explicitly.
    """
    from pathlib import Path

    from transformers import GPT2Config

    local_path = Path(model_name)
    if local_path.is_dir() and not (local_path / "config.json").is_file():
        raise ValueError(f"{model_name!r} is a local directory but has no config.json in it")

    return GPT2Config.from_pretrained(model_name).n_layer


def plan_partition(num_layers: int, profiles: list[NodeProfile]) -> list[ShardAssignment]:
    if not profiles:
        raise ValueError("need at least one node profile")
    if num_layers < len(profiles):
        raise ValueError(
            f"can't split {num_layers} layers across {len(profiles)} nodes "
            "(each node needs at least one layer)"
        )
    if any(p.throughput <= 0 for p in profiles):
        raise ValueError("throughput must be positive for every node")

    total_throughput = sum(p.throughput for p in profiles)
    # Fastest node first: pipeline order follows assignment order below, and
    # rounding remainder (see drift loop) should land on capable nodes first.
    ordered = sorted(profiles, key=lambda p: p.throughput, reverse=True)

    raw_shares = [num_layers * p.throughput / total_throughput for p in ordered]
    layer_counts = [max(1, round(share)) for share in raw_shares]

    # Rounding can drift the total off num_layers; nudge counts back into
    # balance rather than leaving a shard over- or under-assigned.
    drift = num_layers - sum(layer_counts)
    i = 0
    while drift != 0:
        idx = i % len(layer_counts)
        if drift > 0:
            layer_counts[idx] += 1
            drift -= 1
        elif layer_counts[idx] > 1:
            layer_counts[idx] -= 1
            drift += 1
        i += 1

    assignments = []
    cursor = 0
    for profile, count in zip(ordered, layer_counts):
        assignments.append(ShardAssignment(profile.node_id, cursor, cursor + count))
        cursor += count

    return assignments
