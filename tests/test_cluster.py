"""Cluster heals routing across jobs, not just the one that failed.

Runs a normal job, kills a node mid-request during a second job (Cluster
should recover internally, without the caller seeing an exception), then
submits a third job and checks it routes through the standby directly --
proving the pipeline's wiring was actually healed, not just patched for
one job.
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

BASE_PORT = 56000
PRIMARY_DEVICES = [
    ("node-a", 0.6),
    ("node-b", 1.0),  # will be killed mid-request
    ("node-c", 1.5),
]
# plan_partition orders the pipeline fastest-first, so node-c ends up as the
# entry shard and node-a as the exit shard -- pipeline position is derived
# from cluster.assignments in the test below, never assumed from this list.
STANDBY_DEVICE = ("node-spare", 1.0)
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


def test_cluster_heals_routing_across_jobs(rig):
    cluster, standby, model, tokenizer, processes = rig
    input_ids = tokenizer("Testing cluster-level healing", return_tensors="pt").input_ids
    with torch.no_grad():
        expected = model(input_ids).logits

    result1 = cluster.submit(input_ids, job_id="job-1")
    assert torch.allclose(result1.logits, expected, atol=1e-3)
    assert DOOMED_NODE_ID in {t.node_id for t in result1.stage_timings}

    cluster.inject_delay(DOOMED_NODE_ID, DOOMED_ARTIFICIAL_DELAY)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cluster.submit, input_ids, "job-2")
        time.sleep(KILL_AFTER)
        processes[DOOMED_NODE_ID].kill()
        processes[DOOMED_NODE_ID].wait(timeout=10)
        result2 = future.result()  # Cluster.submit() must recover internally, not raise

    assert torch.allclose(result2.logits, expected, atol=1e-3)
    assert standby.node_id in {t.node_id for t in result2.stage_timings}

    # The key assertion: a fresh job submitted *after* recovery must route
    # through the standby on the first try, with no dead-node hop at all --
    # proving the assignment table and upstream wiring were actually healed,
    # not just patched over for job 2.
    result3 = cluster.submit(input_ids, job_id="job-3")
    assert torch.allclose(result3.logits, expected, atol=1e-3)
    stage_ids = {t.node_id for t in result3.stage_timings}
    assert standby.node_id in stage_ids
    assert DOOMED_NODE_ID not in stage_ids
    assignment_ids = {a.node_id for a in cluster.assignments}
    assert standby.node_id in assignment_ids
    assert DOOMED_NODE_ID not in assignment_ids
