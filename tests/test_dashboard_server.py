"""mesh/dashboard_server.py's HTTP endpoints: GET /, GET /api/status,
POST /api/devices (add a device, both "spawn" and "connect" modes),
POST /api/devices/<id>/kill, GET /api/generate/stream (SSE playground feed).
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

import grpc
import pytest
from transformers import GPT2LMHeadModel

from mesh.cluster import Cluster
from mesh.coordinator import plan_partition
from mesh.dashboard_server import ClusterHolder, serve
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards
from mesh.proto import mesh_pb2, mesh_pb2_grpc
from mesh.rig import DeviceSpec, Rig

BASE_PORT = 60200
DASHBOARD_PORT = 60299
PRIMARY_DEVICES = [
    ("node-a", 0.6),
    ("node-b", 1.0),
    ("node-c", 1.5),
]


def _wait_ready(address: str, timeout: float = 60.0) -> None:
    channel = grpc.insecure_channel(address)
    grpc.channel_ready_future(channel).result(timeout=timeout)
    mesh_pb2_grpc.NodeDaemonStub(channel).Heartbeat(mesh_pb2.HeartbeatRequest(), timeout=timeout)


def _get(path: str) -> tuple[int, dict]:
    req = urllib.request.Request(f"http://127.0.0.1:{DASHBOARD_PORT}{path}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(path: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{DASHBOARD_PORT}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture()
def running_server():
    rig = Rig(base_port=BASE_PORT)
    handles = {}
    for node_id, scale in PRIMARY_DEVICES:
        address = f"127.0.0.1:{rig.allocate_port()}"
        handles[node_id] = NodeHandle(node_id=node_id, address=address)
        rig.spawn(node_id, address, scale)

    try:
        for node in handles.values():
            _wait_ready(node.address)

        primary_nodes = list(handles.values())
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        profiles = benchmark_nodes(primary_nodes)
        assignments = plan_partition(len(model.transformer.h), profiles)
        load_shards(primary_nodes, assignments)
        cluster = Cluster(
            primary_nodes, assignments, [], heartbeat_interval=1.0, miss_threshold=2, model_name="gpt2"
        )
        primary_specs = [DeviceSpec(node_id, scale) for node_id, scale in PRIMARY_DEVICES]
        holder = ClusterHolder(cluster, primary_specs, [], heartbeat_interval=1.0, miss_threshold=2)
        server = serve(holder, rig, port=DASHBOARD_PORT)
        try:
            yield cluster, rig
        finally:
            server.shutdown()
            cluster.stop()
    finally:
        rig.shutdown_all()


def test_get_root_serves_html(running_server):
    req = urllib.request.Request(f"http://127.0.0.1:{DASHBOARD_PORT}/")
    with urllib.request.urlopen(req, timeout=10) as resp:
        status = resp.status
        body = resp.read().decode()
    assert status == 200
    assert "<title>Mesh" in body


def test_get_status(running_server):
    status, data = _get("/api/status")
    assert status == 200
    assert data["model_name"] == "gpt2"
    assert len(data["nodes"]) == len(PRIMARY_DEVICES)


def test_add_device_endpoint_onboards_node(running_server):
    cluster, _rig = running_server
    status, data = _post("/api/devices", {"node_id": "dashboard-newcomer", "scale": 1.0, "join_as": "active"})
    assert status == 202
    assert data["node_id"] == "dashboard-newcomer"

    deadline = time.time() + 15
    while time.time() < deadline:
        node_ids = {n["node_id"] for n in cluster.status()["nodes"]}
        if "dashboard-newcomer" in node_ids:
            break
        time.sleep(0.5)
    else:
        pytest.fail("newcomer never appeared in cluster status")


def test_add_device_rejects_missing_node_id(running_server):
    status, data = _post("/api/devices", {"scale": 1.0})
    assert status == 400
    assert "node_id" in data["error"]


def test_add_device_rejects_duplicate(running_server):
    _cluster, _rig = running_server
    status, _ = _post("/api/devices", {"node_id": "dupe-device", "scale": 1.0})
    assert status == 202
    status2, data2 = _post("/api/devices", {"node_id": "dupe-device", "scale": 1.0})
    assert status2 == 409
    assert "already exists" in data2["error"]


def test_kill_endpoint(running_server):
    cluster, _rig = running_server
    node_id = PRIMARY_DEVICES[0][0]
    status, data = _post(f"/api/devices/{node_id}/kill", {})
    assert status == 202
    assert data["node_id"] == node_id

    deadline = time.time() + 15
    while time.time() < deadline:
        node = next(n for n in cluster.status()["nodes"] if n["node_id"] == node_id)
        if not node["alive"]:
            break
        time.sleep(0.5)
    else:
        pytest.fail("killed node was never marked dead")


def test_kill_endpoint_unknown_node(running_server):
    status, data = _post("/api/devices/does-not-exist/kill", {})
    assert status == 404


def test_add_device_connect_mode_onboards_external_daemon(running_server):
    """mode="connect" is the "add my friend's laptop" path: the daemon is
    already running somewhere the dashboard's own Rig never spawned, and
    the endpoint must onboard it by address alone, not try to spawn it.
    """
    cluster, rig = running_server
    external_address = f"127.0.0.1:{BASE_PORT + 50}"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "mesh.daemon",
            "--node-id", "external-friend-laptop", "--address", external_address, "--model", "gpt2",
        ]
    )
    try:
        _wait_ready(external_address)
        status, data = _post(
            "/api/devices",
            {"node_id": "external-friend-laptop", "mode": "connect", "address": external_address, "join_as": "active"},
        )
        assert status == 202
        assert data["address"] == external_address
        assert "external-friend-laptop" not in rig.processes  # never spawned locally

        deadline = time.time() + 15
        while time.time() < deadline:
            node_ids = {n["node_id"] for n in cluster.status()["nodes"]}
            if "external-friend-laptop" in node_ids:
                break
            time.sleep(0.5)
        else:
            pytest.fail("externally-connected device never appeared in cluster status")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_add_device_connect_mode_requires_address(running_server):
    status, data = _post("/api/devices", {"node_id": "no-address-device", "mode": "connect"})
    assert status == 400
    assert "address" in data["error"]


def test_generate_stream_endpoint_streams_sse_events(running_server):
    url = f"http://127.0.0.1:{DASHBOARD_PORT}/api/generate/stream?prompt=Hello+there&max_new_tokens=3&temperature=0"
    with urllib.request.urlopen(url, timeout=30) as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "text/event-stream"
        body = resp.read().decode()

    data_lines = [line[len("data: "):] for line in body.splitlines() if line.startswith("data: ")]
    events = [json.loads(line) for line in data_lines]
    assert events, "expected at least one SSE event"
    assert events[-1] == {"done": True}
    assert any("piece" in e for e in events)


def test_generate_stream_endpoint_requires_prompt(running_server):
    status, data = _get("/api/generate/stream?max_new_tokens=3")
    assert status == 400
    assert "prompt" in data["error"]
