"""Kill a node mid-request, verify the coordinator recovers.

This is the automated version of scripts/run_fault_injection_demo.py: an
intermediate node is configured with an artificial delay so the test can
SIGKILL it deterministically while a job is in flight, then checks that
recover_job() produces output matching a monolithic forward pass.
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from mesh.coordinator import plan_partition
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards, recover_job, submit_job
from mesh.proto import mesh_pb2, mesh_pb2_grpc

BASE_PORT = 54000
PRIMARY_DEVICES = [
    ("node-a", 0.6),  # entry
    ("node-b", 1.0),  # will be killed
    ("node-c", 1.5),  # exit
]
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
        [
            sys.executable,
            "-m",
            "mesh.daemon",
            "--node-id",
            node_id,
            "--address",
            address,
            "--simulated-scale",
            str(scale),
        ]
    )


@pytest.fixture()
def cluster():
    all_devices = PRIMARY_DEVICES + [STANDBY_DEVICE]
    handles = {
        node_id: NodeHandle(node_id=node_id, address=f"127.0.0.1:{BASE_PORT + i}")
        for i, (node_id, _scale) in enumerate(all_devices)
    }
    processes = {node_id: _spawn(node_id, handles[node_id].address, scale) for node_id, scale in all_devices}
    try:
        for node in handles.values():
            _wait_ready(node.address)
        yield handles, processes
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_recovers_after_node_killed_mid_request(cluster):
    handles, processes = cluster
    primary_nodes = [handles[node_id] for node_id, _ in PRIMARY_DEVICES]
    standby = handles[STANDBY_DEVICE[0]]

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer("Testing fault tolerance across the mesh", return_tensors="pt").input_ids

    profiles = benchmark_nodes(primary_nodes)
    num_layers = len(model.transformer.h)
    assignments = plan_partition(num_layers, profiles)
    load_shards(primary_nodes, assignments, artificial_delays={DOOMED_NODE_ID: DOOMED_ARTIFICIAL_DELAY})

    entry_node = next(n for n in primary_nodes if n.node_id == assignments[0].node_id)
    job_id = "test-fault-job"

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(submit_job, entry_node, job_id, input_ids)
        time.sleep(KILL_AFTER)
        processes[DOOMED_NODE_ID].kill()
        processes[DOOMED_NODE_ID].wait(timeout=10)

        with pytest.raises(grpc.RpcError):
            future.result()

    result = recover_job(primary_nodes, assignments, standby, job_id, input_ids)

    with torch.no_grad():
        expected = model(input_ids).logits

    assert torch.allclose(result.logits, expected, atol=1e-3)
    assert standby.node_id in {t.node_id for t in result.stage_timings}
    assert DOOMED_NODE_ID not in {t.node_id for t in result.stage_timings}


def test_get_checkpoint_unavailable_for_unknown_job(cluster):
    handles, _processes = cluster
    from mesh.net_coordinator import get_checkpoint

    node = handles[PRIMARY_DEVICES[0][0]]
    assert get_checkpoint(node, "no-such-job") is None
