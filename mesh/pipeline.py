"""Orchestrates a full distributed forward pass: spawns one process per
shard, wires them into a pipeline via queues in assignment order, feeds
input_ids into the first stage, and collects the final logits plus
per-stage timing from the last stage.
"""

import multiprocessing as mp
from dataclasses import dataclass

import torch
from transformers import GPT2LMHeadModel

from mesh.coordinator import ShardAssignment
from mesh.model_shard import ModelShard
from mesh.node import NodeProcess


@dataclass(frozen=True)
class StageTiming:
    node_id: str
    elapsed_seconds: float


@dataclass(frozen=True)
class PipelineResult:
    logits: torch.Tensor
    stage_timings: list[StageTiming]


def build_shards(model: GPT2LMHeadModel, assignments: list[ShardAssignment]) -> list[ModelShard]:
    return [
        ModelShard(
            model,
            layer_start=a.layer_start,
            layer_end=a.layer_end,
            include_embed=(i == 0),
            include_head=(i == len(assignments) - 1),
        )
        for i, a in enumerate(assignments)
    ]


def run_pipeline(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    assignments: list[ShardAssignment],
    artificial_delays: dict[str, float] | None = None,
) -> PipelineResult:
    artificial_delays = artificial_delays or {}
    shards = build_shards(model, assignments)

    # queues[i] feeds stage i; queues[-1] yields the final output tensor.
    data_queues = [mp.Queue() for _ in range(len(shards) + 1)]
    telemetry_queue: mp.Queue = mp.Queue()

    processes = [
        NodeProcess(
            node_id=assignment.node_id,
            shard=shard,
            in_queue=data_queues[i],
            out_queue=data_queues[i + 1],
            telemetry_queue=telemetry_queue,
            artificial_delay=artificial_delays.get(assignment.node_id, 0.0),
        )
        for i, (assignment, shard) in enumerate(zip(assignments, shards))
    ]

    for p in processes:
        p.start()

    data_queues[0].put(input_ids)

    logits = data_queues[-1].get()

    stage_timings = [
        StageTiming(node_id, elapsed) for node_id, elapsed in (telemetry_queue.get() for _ in processes)
    ]

    for q in data_queues[:-1]:  # last queue has no reader (it's the pipeline's output)
        q.put(None)  # sentinel: shut each stage down
    for p in processes:
        p.join(timeout=5)

    return PipelineResult(logits=logits, stage_timings=stage_timings)
