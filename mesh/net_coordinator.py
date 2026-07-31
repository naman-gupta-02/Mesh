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


def load_shards(nodes: list[NodeHandle], assignments: list[ShardAssignment]) -> None:
    """Tell each node which layer range to hold and who its next hop is,
    wiring the pipeline in assignment order (assignments[0] is the entry
    shard, holding layer_start == 0).
    """
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
        )
        response = _stub(address_by_id[assignment.node_id]).LoadShard(request)
        if not response.ok:
            raise RuntimeError(f"{assignment.node_id} failed to load shard: {response.error}")


def submit_job(entry_node: NodeHandle, job_id: str, input_ids: torch.Tensor) -> JobResult:
    """Kicks off the pipeline with a single RPC to the entry node; its
    response only arrives once every downstream node has run and relayed
    its result back up the chain (see mesh/daemon.py's Forward handler).
    """
    request = mesh_pb2.ForwardRequest(job_id=job_id, tensor=encode(input_ids))
    response = _stub(entry_node.address).Forward(request)
    logits = decode(response.logits)
    timings = [StageTiming(t.node_id, t.elapsed_seconds) for t in response.timings]
    return JobResult(logits=logits, stage_timings=timings)
