"""Cluster.generate()/generate_stream() -- the "ask the model something"
feature -- and the model-mismatch guard on add_device().
"""

import subprocess
import sys

import grpc
import pytest
from transformers import GPT2LMHeadModel

from mesh.cluster import Cluster
from mesh.coordinator import plan_partition
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards
from mesh.proto import mesh_pb2, mesh_pb2_grpc

BASE_PORT = 60300
PRIMARY_DEVICES = [
    ("node-a", 0.6),
    ("node-b", 1.0),
]


def _wait_ready(address: str, timeout: float = 60.0) -> None:
    channel = grpc.insecure_channel(address)
    grpc.channel_ready_future(channel).result(timeout=timeout)
    mesh_pb2_grpc.NodeDaemonStub(channel).Heartbeat(mesh_pb2.HeartbeatRequest(), timeout=timeout)


def _spawn(node_id: str, address: str, scale: float, model_name: str = "gpt2") -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable, "-m", "mesh.daemon",
            "--node-id", node_id, "--address", address,
            "--simulated-scale", str(scale), "--model", model_name,
        ]
    )


@pytest.fixture()
def cluster():
    handles = {
        node_id: NodeHandle(node_id=node_id, address=f"127.0.0.1:{BASE_PORT + i}")
        for i, (node_id, _scale) in enumerate(PRIMARY_DEVICES)
    }
    processes = {node_id: _spawn(node_id, handles[node_id].address, scale) for node_id, scale in PRIMARY_DEVICES}
    try:
        for node in handles.values():
            _wait_ready(node.address)

        primary_nodes = list(handles.values())
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        profiles = benchmark_nodes(primary_nodes)
        assignments = plan_partition(len(model.transformer.h), profiles)
        load_shards(primary_nodes, assignments)
        c = Cluster(primary_nodes, assignments, [], heartbeat_interval=1.0, miss_threshold=2, model_name="gpt2")
        try:
            yield c
        finally:
            c.stop()
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_generate_produces_text(cluster):
    text = cluster.generate("The capital of France is", max_new_tokens=5)
    assert isinstance(text, str)
    assert text.strip() != ""


def test_generate_greedy_is_deterministic(cluster):
    first = cluster.generate("Hello, my name is", max_new_tokens=5, temperature=0.0)
    second = cluster.generate("Hello, my name is", max_new_tokens=5, temperature=0.0)
    assert first == second


def test_generate_stream_yields_pieces_matching_generate(cluster):
    pieces = list(cluster.generate_stream("The weather today is", max_new_tokens=4, temperature=0.0))
    assert len(pieces) >= 1
    assert "".join(pieces) == cluster.generate("The weather today is", max_new_tokens=4, temperature=0.0)


def test_generate_respects_max_new_tokens(cluster):
    # Greedy decoding rarely hits EOS in a handful of tokens for a plain
    # sentence prompt, so this is a reasonable (not airtight) upper-bound
    # check: generate() must never produce more pieces than requested.
    pieces = list(cluster.generate_stream("Once upon a time", max_new_tokens=3, temperature=0.0))
    assert len(pieces) <= 3


def test_add_device_rejects_model_mismatch(cluster):
    # The daemon reports its --model flag via Heartbeat without ever
    # loading it (that only happens lazily on LoadShard), so a bogus model
    # name here exercises the mismatch guard without downloading anything.
    mismatched_address = f"127.0.0.1:{BASE_PORT + 90}"
    proc = _spawn("mismatched-node", mismatched_address, 1.0, model_name="not-a-real-model-xyz")
    try:
        _wait_ready(mismatched_address)
        with pytest.raises(ValueError, match="not-a-real-model-xyz"):
            cluster.add_device(NodeHandle(node_id="mismatched-node", address=mismatched_address), join_as="active")

        status = cluster.status()
        assert "mismatched-node" not in {n["node_id"] for n in status["nodes"]}
        assert any("rejected" in e["message"] for e in status["events"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
