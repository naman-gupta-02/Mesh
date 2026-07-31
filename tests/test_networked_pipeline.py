"""Integration test: real node daemons over gRPC on localhost, not
multiprocessing queues. Spins up actual OS processes, drives them through
the coordinator client, and checks the result against a monolithic forward
pass -- the networked equivalent of tests/test_pipeline_correctness.py.
"""

import subprocess
import sys

import grpc
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from mesh.coordinator import plan_partition
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards, submit_job
from mesh.proto import mesh_pb2, mesh_pb2_grpc

BASE_PORT = 53000
DEVICES = [
    ("node-a", 0.6),
    ("node-b", 1.3),
    ("node-c", 2.0),
]


def _wait_ready(address: str, timeout: float = 60.0) -> None:
    channel = grpc.insecure_channel(address)
    grpc.channel_ready_future(channel).result(timeout=timeout)
    mesh_pb2_grpc.NodeDaemonStub(channel).Heartbeat(mesh_pb2.HeartbeatRequest(), timeout=timeout)


@pytest.fixture(scope="module")
def running_nodes():
    nodes = [
        NodeHandle(node_id=node_id, address=f"127.0.0.1:{BASE_PORT + i}")
        for i, (node_id, _scale) in enumerate(DEVICES)
    ]
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mesh.daemon",
                "--node-id",
                node.node_id,
                "--address",
                node.address,
                "--simulated-scale",
                str(scale),
            ]
        )
        for node, (_node_id, scale) in zip(nodes, DEVICES)
    ]
    try:
        for node in nodes:
            _wait_ready(node.address)
        yield nodes
    finally:
        for proc in processes:
            proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_networked_pipeline_matches_monolithic(running_nodes):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer("Mesh nodes are talking over the network now", return_tensors="pt").input_ids

    profiles = benchmark_nodes(running_nodes)
    assert {p.node_id for p in profiles} == {n.node_id for n in running_nodes}

    num_layers = len(model.transformer.h)
    assignments = plan_partition(num_layers, profiles)
    load_shards(running_nodes, assignments)

    entry_node = next(n for n in running_nodes if n.node_id == assignments[0].node_id)
    result = submit_job(entry_node, job_id="test-job", input_ids=input_ids)

    with torch.no_grad():
        expected = model(input_ids).logits

    assert torch.allclose(result.logits, expected, atol=1e-3)
    assert {t.node_id for t in result.stage_timings} == {n.node_id for n in running_nodes}


def test_heartbeat_reports_alive(running_nodes):
    for node in running_nodes:
        channel = grpc.insecure_channel(node.address)
        response = mesh_pb2_grpc.NodeDaemonStub(channel).Heartbeat(mesh_pb2.HeartbeatRequest())
        assert response.alive
        assert response.node_id == node.node_id
