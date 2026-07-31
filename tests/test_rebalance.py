"""After a recovery, the next submit() should rebalance -- re-profile every
active node and recompute the partition -- rather than leaving the standby
stuck with the dead node's old layer range regardless of its own speed.
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from mesh.cluster import Cluster
from mesh.coordinator import plan_partition
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards
from mesh.proto import mesh_pb2, mesh_pb2_grpc

BASE_PORT = 58000
PRIMARY_DEVICES = [
    ("node-a", 0.6),
    ("node-b", 1.0),  # will be killed mid-request
    ("node-c", 1.5),
]
# Much faster than what it replaces, so a real rebalance is visibly
# different from just inheriting node-b's old layer range.
STANDBY_DEVICE = ("node-spare-fast", 5.0)
DOOMED_NODE_ID = "node-b"
DOOMED_ARTIFICIAL_DELAY = 2.0
KILL_AFTER = 0.3


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
    all_devices = PRIMARY_DEVICES + [STANDBY_DEVICE]
    handles = {
        node_id: NodeHandle(node_id=node_id, address=f"127.0.0.1:{BASE_PORT + i}")
        for i, (node_id, _scale) in enumerate(all_devices)
    }
    processes = {node_id: _spawn(node_id, handles[node_id].address, scale) for node_id, scale in all_devices}
    try:
        for node in handles.values():
            _wait_ready(node.address)

        primary_nodes = [handles[node_id] for node_id, _ in PRIMARY_DEVICES]
        standby = handles[STANDBY_DEVICE[0]]

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        profiles = benchmark_nodes(primary_nodes)
        assignments = plan_partition(len(model.transformer.h), profiles)
        load_shards(primary_nodes, assignments)

        cluster = Cluster(primary_nodes, assignments, [standby], heartbeat_interval=1.0, miss_threshold=2)
        try:
            yield cluster, standby, model, tokenizer, processes
        finally:
            cluster.stop()
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_rebalance_gives_standby_its_proportional_share(rig):
    cluster, standby, model, tokenizer, processes = rig
    input_ids = tokenizer("Testing rebalancing after recovery", return_tensors="pt").input_ids
    with torch.no_grad():
        expected = model(input_ids).logits
    num_layers = len(model.transformer.h)

    result1 = cluster.submit(input_ids, job_id="job-1")
    assert torch.allclose(result1.logits, expected, atol=1e-3)
    assert not cluster.needs_rebalance

    cluster.inject_delay(DOOMED_NODE_ID, DOOMED_ARTIFICIAL_DELAY)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cluster.submit, input_ids, "job-2")
        time.sleep(KILL_AFTER)
        processes[DOOMED_NODE_ID].kill()
        processes[DOOMED_NODE_ID].wait(timeout=10)
        result2 = future.result()

    assert torch.allclose(result2.logits, expected, atol=1e-3)
    assert cluster.needs_rebalance, "a topology change (recovery) should flag a pending rebalance"

    healed_standby_layers = next(a.num_layers for a in cluster.assignments if a.node_id == standby.node_id)

    result3 = cluster.submit(input_ids, job_id="job-3")
    assert torch.allclose(result3.logits, expected, atol=1e-3)
    assert not cluster.needs_rebalance, "submit() should have rebalanced and cleared the flag"

    # partition invariants still hold after a rebalance
    assert sum(a.num_layers for a in cluster.assignments) == num_layers
    assert cluster.assignments[0].layer_start == 0
    assert cluster.assignments[-1].layer_end == num_layers

    rebalanced_standby_layers = next(a.num_layers for a in cluster.assignments if a.node_id == standby.node_id)
    # node-spare-fast (scale 5.0) is far faster than anything it could have
    # inherited from node-b (scale 1.0) -- a real rebalance must give it
    # more layers than the naive healed assignment did.
    assert rebalanced_standby_layers > healed_standby_layers
