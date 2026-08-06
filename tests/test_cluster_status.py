"""Cluster.status() (the dashboard's data source) and add_device() (the
"new volunteer joins mid-session" feature).
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

BASE_PORT = 60100
PRIMARY_DEVICES = [
    ("node-a", 0.6),
    ("node-b", 1.0),
    ("node-c", 1.5),
]
NEW_DEVICE = ("node-newcomer", 1.0)


def _wait_ready(address: str, timeout: float = 60.0) -> None:
    channel = grpc.insecure_channel(address)
    grpc.channel_ready_future(channel).result(timeout=timeout)
    mesh_pb2_grpc.NodeDaemonStub(channel).Heartbeat(mesh_pb2.HeartbeatRequest(), timeout=timeout)


def _spawn(node_id: str, address: str, scale: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "mesh.daemon", "--node-id", node_id, "--address", address,
         "--simulated-scale", str(scale)]
    )


@pytest.fixture()
def rig():
    all_devices = PRIMARY_DEVICES + [NEW_DEVICE]
    handles = {
        node_id: NodeHandle(node_id=node_id, address=f"127.0.0.1:{BASE_PORT + i}")
        for i, (node_id, _scale) in enumerate(all_devices)
    }
    processes = {node_id: _spawn(node_id, handles[node_id].address, scale) for node_id, scale in all_devices}
    try:
        for node in handles.values():
            _wait_ready(node.address)
        yield handles
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def _make_cluster(handles) -> Cluster:
    primary_nodes = [handles[node_id] for node_id, _ in PRIMARY_DEVICES]
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    profiles = benchmark_nodes(primary_nodes)
    assignments = plan_partition(len(model.transformer.h), profiles)
    load_shards(primary_nodes, assignments)
    return Cluster(primary_nodes, assignments, [], heartbeat_interval=1.0, miss_threshold=2, model_name="gpt2")


def test_status_shape(rig):
    cluster = _make_cluster(rig)
    try:
        status = cluster.status()
        assert status["model_name"] == "gpt2"
        assert status["num_layers"] == 12
        assert len(status["nodes"]) == len(PRIMARY_DEVICES)
        assert {n["node_id"] for n in status["nodes"]} == {node_id for node_id, _ in PRIMARY_DEVICES}
        assert all(n["alive"] for n in status["nodes"])
        roles = {n["role"] for n in status["nodes"]}
        assert "entry" in roles and "exit" in roles
        assert sum(n["num_layers"] for n in status["nodes"]) == 12
        assert status["jobs_submitted"] == 0
        assert status["events"][-1]["message"].startswith("cluster started")
    finally:
        cluster.stop()


def test_add_device_active_flags_rebalance(rig):
    cluster = _make_cluster(rig)
    try:
        assert not cluster.needs_rebalance
        cluster.add_device(rig[NEW_DEVICE[0]], join_as="active")
        assert cluster.needs_rebalance

        status = cluster.status()
        node_ids = {n["node_id"] for n in status["nodes"]}
        assert NEW_DEVICE[0] in node_ids
        newcomer = next(n for n in status["nodes"] if n["node_id"] == NEW_DEVICE[0])
        assert newcomer["alive"]
        assert newcomer["num_layers"] == 0  # not yet folded into the partition -- that's rebalance's job
        assert any("device joined" in e["message"] for e in status["events"])
    finally:
        cluster.stop()


def test_add_device_standby_does_not_flag_rebalance(rig):
    cluster = _make_cluster(rig)
    try:
        cluster.add_device(rig[NEW_DEVICE[0]], join_as="standby")
        assert not cluster.needs_rebalance
        status = cluster.status()
        assert NEW_DEVICE[0] in status["standby_ids"]
        assert NEW_DEVICE[0] not in {n["node_id"] for n in status["nodes"]}
    finally:
        cluster.stop()


def test_add_device_rejects_bad_join_as(rig):
    cluster = _make_cluster(rig)
    try:
        with pytest.raises(ValueError):
            cluster.add_device(rig[NEW_DEVICE[0]], join_as="nonsense")
    finally:
        cluster.stop()
