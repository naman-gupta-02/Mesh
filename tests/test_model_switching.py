"""Model switching: rebuilding the cluster to run a different model
(mesh/dashboard_server.py's switch_model()), the config.json validation
gap that closes in model_layer_count(), the hand-rolled multipart parser,
and the model-upload endpoint.
"""

import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from transformers import GPT2LMHeadModel

from mesh.cluster import Cluster
from mesh.coordinator import model_layer_count, plan_partition
from mesh.dashboard_server import UPLOAD_DIR, ClusterHolder, _parse_multipart, serve
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards
from mesh.rig import DeviceSpec, Rig, wait_until_ready

BASE_PORT = 60400
DASHBOARD_PORT = 60499
PRIMARY_DEVICES = [DeviceSpec("node-a", 0.6), DeviceSpec("node-b", 1.0)]


def _get(path: str) -> tuple[int, dict]:
    req = urllib.request.Request(f"http://127.0.0.1:{DASHBOARD_PORT}{path}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post_json(path: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{DASHBOARD_PORT}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post_multipart(path: str, fields: dict, files: list[tuple[str, str, bytes]]) -> tuple[int, dict]:
    boundary = "----testboundary123"
    body = bytearray()
    for name, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    for name, filename, content in files:
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n\r\n'
        ).encode()
        body += content
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{DASHBOARD_PORT}{path}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture()
def running_server():
    rig = Rig(base_port=BASE_PORT)
    handles = {}
    for spec in PRIMARY_DEVICES:
        address = f"127.0.0.1:{rig.allocate_port()}"
        handles[spec.node_id] = NodeHandle(node_id=spec.node_id, address=address)
        rig.spawn(spec.node_id, address, spec.scale, model_name="gpt2")
    try:
        for node in handles.values():
            wait_until_ready(node.address)

        primary_nodes = list(handles.values())
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        profiles = benchmark_nodes(primary_nodes)
        assignments = plan_partition(len(model.transformer.h), profiles)
        load_shards(primary_nodes, assignments)
        cluster = Cluster(
            primary_nodes, assignments, [], heartbeat_interval=1.0, miss_threshold=2, model_name="gpt2"
        )
        holder = ClusterHolder(cluster, PRIMARY_DEVICES, [], heartbeat_interval=1.0, miss_threshold=2)
        server = serve(holder, rig, port=DASHBOARD_PORT)
        try:
            yield holder
        finally:
            server.shutdown()
            if holder.cluster is not None:
                holder.cluster.stop()
    finally:
        rig.shutdown_all()


def test_model_layer_count_rejects_incomplete_local_directory(tmp_path):
    empty_dir = tmp_path / "incomplete-model"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="config.json"):
        model_layer_count(str(empty_dir))


def test_model_layer_count_works_for_real_model():
    assert model_layer_count("gpt2") == 12


def test_parse_multipart_handles_binary_content_with_crlf_bytes():
    boundary = b"----xyz"
    # deliberately contains raw \r\n bytes -- a naive parser that blindly
    # strips leading/trailing \r\n from a segment (rather than removing an
    # exact 2-byte prefix/suffix) would mangle this.
    binary_content = bytes([0x0D, 0x0A, 0x00, 0xFF, 0x0D, 0x0A, 42])
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="model_name"\r\n\r\n'
        b"my-model\r\n"
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="files"; filename="weights.bin"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n" + binary_content + b"\r\n"
        b"--" + boundary + b"--\r\n"
    )
    fields, files = _parse_multipart(body, f"multipart/form-data; boundary={boundary.decode()}")
    assert fields == {"model_name": "my-model"}
    assert len(files) == 1
    filename, content = files[0]
    assert filename == "weights.bin"
    assert content == binary_content


def test_switch_endpoint_rejects_bad_model_without_destroying_cluster(running_server):
    holder = running_server
    status, _ = _post_json("/api/models/switch", {"model_name": "totally-not-a-real-model-xyz"})
    assert status == 202

    deadline = time.time() + 20
    while time.time() < deadline:
        _, models = _get("/api/models")
        if models["model_switch"]["status"] in ("idle", "error"):
            break
        time.sleep(0.5)
    else:
        pytest.fail("switch never settled")

    assert models["model_switch"]["status"] == "error"
    assert "totally-not-a-real-model-xyz" in models["model_switch"]["error"]
    # validation failed before teardown -- the working cluster must be untouched
    assert holder.cluster is not None
    assert holder.cluster.model_name == "gpt2"


def test_switch_endpoint_rejects_concurrent_switch(running_server):
    status1, _ = _post_json("/api/models/switch", {"model_name": "distilgpt2"})
    assert status1 == 202
    status2, _data2 = _post_json("/api/models/switch", {"model_name": "gpt2-medium"})
    assert status2 == 409

    # let the first switch finish so fixture teardown isn't racing it
    deadline = time.time() + 90
    while time.time() < deadline:
        _, models = _get("/api/models")
        if models["model_switch"]["status"] != "switching":
            break
        time.sleep(1)


def test_switch_endpoint_rebuilds_cluster_with_new_model(running_server):
    status, _ = _post_json("/api/models/switch", {"model_name": "distilgpt2"})
    assert status == 202

    deadline = time.time() + 90
    while time.time() < deadline:
        _, models = _get("/api/models")
        if models["model_switch"]["status"] != "switching":
            break
        time.sleep(1)
    else:
        pytest.fail("switch never completed")

    assert models["model_switch"]["status"] == "idle"
    assert models["current_model"] == "distilgpt2"
    _, status_data = _get("/api/status")
    assert status_data["model_name"] == "distilgpt2"
    assert status_data["num_layers"] == 6
    assert all(n["alive"] for n in status_data["nodes"])


def test_upload_endpoint_accepts_valid_config(running_server):
    config_bytes = json.dumps(
        {
            "architectures": ["GPT2LMHeadModel"], "model_type": "gpt2",
            "n_layer": 2, "n_head": 2, "n_embd": 32, "n_positions": 64, "vocab_size": 100,
        }
    ).encode()
    status, data = _post_multipart(
        "/api/models/upload", {"model_name": "test-upload-valid"}, [("files", "config.json", config_bytes)]
    )
    assert status == 200
    assert data["name"] == "test-upload-valid"
    try:
        _, models = _get("/api/models")
        assert any(m["name"] == "test-upload-valid" for m in models["uploaded"])
    finally:
        shutil.rmtree(Path(data["path"]), ignore_errors=True)


def test_upload_endpoint_rejects_missing_config(running_server):
    status, data = _post_multipart(
        "/api/models/upload", {"model_name": "test-upload-invalid"}, [("files", "readme.txt", b"not a model")]
    )
    assert status == 400
    assert "valid GPT-2 model" in data["error"]
    assert not (UPLOAD_DIR / "test-upload-invalid").exists()


def test_upload_endpoint_requires_model_name(running_server):
    status, data = _post_multipart("/api/models/upload", {}, [("files", "config.json", b"{}")])
    assert status == 400
    assert "model_name" in data["error"]
