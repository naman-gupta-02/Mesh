"""Coordinator client for Phase 2: drives real node daemons over gRPC
instead of spawning local processes wired by multiprocessing queues.
"""

from dataclasses import dataclass

import grpc
import torch

from mesh.coordinator import NodeProfile, ShardAssignment
from mesh.proto import mesh_pb2, mesh_pb2_grpc
from mesh.tensor_codec import decode, encode


@dataclass(frozen=True)
class NodeHandle:
    node_id: str
    address: str


@dataclass(frozen=True)
class StageTiming:
    node_id: str
    elapsed_seconds: float


@dataclass(frozen=True)
class JobResult:
    logits: torch.Tensor
    stage_timings: list[StageTiming]


def _stub(address: str) -> mesh_pb2_grpc.NodeDaemonStub:
    channel = grpc.insecure_channel(address)
    return mesh_pb2_grpc.NodeDaemonStub(channel)


def benchmark_nodes(nodes: list[NodeHandle]) -> list[NodeProfile]:
    """Real RPC round-trip to each node: ask it to run its own micro-benchmark
    and report back, rather than trusting a self-reported spec sheet.
    """
    profiles = []
    for node in nodes:
        response = _stub(node.address).Benchmark(mesh_pb2.BenchmarkRequest())
        profiles.append(NodeProfile(node_id=response.node_id, throughput=response.throughput))
    return profiles


def load_shards(
    nodes: list[NodeHandle],
    assignments: list[ShardAssignment],
    artificial_delays: dict[str, float] | None = None,
) -> None:
    """Tell each node which layer range to hold and who its next hop is,
    wiring the pipeline in assignment order (assignments[0] is the entry
    shard, holding layer_start == 0). artificial_delays is a fault-injection
    testing knob (see mesh/daemon.py); leave it empty in normal operation.
    """
    artificial_delays = artificial_delays or {}
    address_by_id = {n.node_id: n.address for n in nodes}
    for i, assignment in enumerate(assignments):
        next_hop_address = ""
        if i + 1 < len(assignments):
            next_hop_address = address_by_id[assignments[i + 1].node_id]

        request = mesh_pb2.LoadShardRequest(
            shard=mesh_pb2.ShardSpec(
                layer_start=assignment.layer_start,
                layer_end=assignment.layer_end,
                include_embed=(i == 0),
                include_head=(i == len(assignments) - 1),
            ),
            next_hop_address=next_hop_address,
            artificial_delay_seconds=artificial_delays.get(assignment.node_id, 0.0),
        )
        response = _stub(address_by_id[assignment.node_id]).LoadShard(request)
        if not response.ok:
            raise RuntimeError(f"{assignment.node_id} failed to load shard: {response.error}")


def submit_job(
    entry_node: NodeHandle, job_id: str, input_ids: torch.Tensor, timeout: float = 15.0
) -> JobResult:
    """Kicks off the pipeline with a single RPC to the entry node; its
    response only arrives once every downstream node has run and relayed
    its result back up the chain (see mesh/daemon.py's Forward handler).

    Raises grpc.RpcError if any node in the chain is unreachable or the
    call doesn't complete within `timeout` -- the caller (see recover_job)
    is expected to catch that and attempt recovery.
    """
    request = mesh_pb2.ForwardRequest(job_id=job_id, tensor=encode(input_ids))
    response = _stub(entry_node.address).Forward(request, timeout=timeout)
    logits = decode(response.logits)
    timings = [StageTiming(t.node_id, t.elapsed_seconds) for t in response.timings]
    return JobResult(logits=logits, stage_timings=timings)


def get_checkpoint(node: NodeHandle, job_id: str, timeout: float = 5.0) -> torch.Tensor | None:
    """This node's own cached output for job_id, or None if it never
    processed that job (dead before it got there, or a different job).
    """
    try:
        response = _stub(node.address).GetCheckpoint(mesh_pb2.GetCheckpointRequest(job_id=job_id), timeout=timeout)
    except grpc.RpcError:
        return None
    if not response.available:
        return None
    return decode(response.tensor)


def find_last_checkpoint(
    nodes: list[NodeHandle], assignments: list[ShardAssignment], job_id: str
) -> tuple[int, torch.Tensor | None]:
    """Walks the pipeline in assignment order and returns (index, tensor) for
    the deepest shard that has a checkpoint for job_id. Index -1 means not
    even the entry shard finished -- recovery must restart from input_ids.
    """
    address_by_id = {n.node_id: n.address for n in nodes}
    last_index = -1
    last_tensor = None
    for i, assignment in enumerate(assignments):
        node = NodeHandle(node_id=assignment.node_id, address=address_by_id[assignment.node_id])
        tensor = get_checkpoint(node, job_id)
        if tensor is not None:
            last_index, last_tensor = i, tensor
    return last_index, last_tensor


def recover_job(
    nodes: list[NodeHandle],
    assignments: list[ShardAssignment],
    standby: NodeHandle,
    job_id: str,
    input_ids: torch.Tensor,
    timeout: float = 15.0,
) -> JobResult:
    """Called after submit_job raises: finds the last shard boundary that
    completed, reassigns the (presumed dead) next shard's layer range to
    `standby`, and resumes the chain from there -- rather than restarting
    the whole job from scratch. Surviving downstream nodes keep the shard
    and next-hop wiring they already had from the original load_shards call.
    """
    last_index, last_tensor = find_last_checkpoint(nodes, assignments, job_id)
    failed_index = last_index + 1
    if failed_index >= len(assignments):
        raise RuntimeError(f"job {job_id}: no failed shard found (chain already completed)")

    failed_assignment = assignments[failed_index]
    address_by_id = {n.node_id: n.address for n in nodes}
    next_hop_address = ""
    if failed_index + 1 < len(assignments):
        next_hop_address = address_by_id[assignments[failed_index + 1].node_id]

    load_response = _stub(standby.address).LoadShard(
        mesh_pb2.LoadShardRequest(
            shard=mesh_pb2.ShardSpec(
                layer_start=failed_assignment.layer_start,
                layer_end=failed_assignment.layer_end,
                include_embed=(failed_index == 0),
                include_head=(failed_index == len(assignments) - 1),
            ),
            next_hop_address=next_hop_address,
        )
    )
    if not load_response.ok:
        raise RuntimeError(f"standby {standby.node_id} failed to load shard: {load_response.error}")

    resume_tensor = input_ids if last_index == -1 else last_tensor
    request = mesh_pb2.ForwardRequest(job_id=job_id, tensor=encode(resume_tensor))
    response = _stub(standby.address).Forward(request, timeout=timeout)
    logits = decode(response.logits)
    timings = [StageTiming(t.node_id, t.elapsed_seconds) for t in response.timings]
    return JobResult(logits=logits, stage_timings=timings)
