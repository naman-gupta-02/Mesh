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
