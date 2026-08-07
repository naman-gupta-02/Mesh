"""Lightweight HTTP server exposing Cluster.status() as JSON and serving a
self-contained animated dashboard page. Stdlib only (http.server) -- no new
dependency for a small local dev tool.

Interactive endpoints:
- POST /api/devices spawns a brand new local daemon and onboards it (mode
  "spawn", the default -- a simulated device for the demo), or onboards an
  already-running daemon at a caller-supplied address without spawning
  anything (mode "connect" -- a real device, e.g. a friend's laptop on the
  same network already running `python -m mesh.daemon`).
- POST /api/devices/<id>/kill kills a *locally spawned* node's process to
  trigger the fault-tolerance path live (only works for "spawn" mode
  devices -- see mesh/rig.py's docstring for why the dashboard can't kill
  someone else's laptop).
- GET /api/generate/stream is a Server-Sent Events endpoint driving
  Cluster.generate_stream() -- the "ask the model something" playground
  feature, streamed token-by-token rather than returned all at once.
- GET /api/models lists curated + uploaded model choices and the current
  switch status. POST /api/models/switch tears down the whole cluster and
  rebuilds it running a different model (see switch_model() below) --
  every daemon needs different weights, so this can't be a per-node patch.
  POST /api/models/upload accepts a caller's own GPT-2-format model files
  and registers the resulting local directory as a selectable model.
"""

import json
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mesh.cluster import Cluster
from mesh.coordinator import model_layer_count, plan_partition
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards
from mesh.rig import DeviceSpec, Rig, wait_until_ready

DASHBOARD_HTML_PATH = Path(__file__).parent / "dashboard.html"
MAX_GENERATE_TOKENS = 200  # demo sanity cap -- each token is a full pipeline pass
MAX_UPLOAD_BYTES = 2 * 1024**3  # 2GB -- whole-body-in-memory parser, see _parse_multipart
UPLOAD_DIR = Path.home() / ".cache" / "mesh" / "uploaded_models"

CURATED_MODELS = [
    {"id": "gpt2", "label": "gpt2 (124M) — fastest"},
    {"id": "distilgpt2", "label": "distilgpt2 (82M) — distilled, smaller than gpt2"},
    {"id": "gpt2-medium", "label": "gpt2-medium (355M)"},
    {"id": "gpt2-large", "label": "gpt2-large (774M)"},
    {"id": "gpt2-xl", "label": "gpt2-xl (1.5B) — slow, ~6GB first download"},
]


class ClusterHolder:
    """Mutable pointer to the "current" Cluster.

    Switching models tears down every daemon and builds a brand new
    Cluster from scratch (see switch_model()) -- request handlers need a
    level of indirection instead of closing over one fixed Cluster object,
    so a switch can swap it out under them.
    """

    def __init__(
        self,
        cluster: Cluster,
        primary_specs: list[DeviceSpec],
        standby_specs: list[DeviceSpec],
        heartbeat_interval: float,
        miss_threshold: int,
    ):
        self.cluster: Cluster | None = cluster
        self.primary_specs = primary_specs
        self.standby_specs = standby_specs
        self.heartbeat_interval = heartbeat_interval
        self.miss_threshold = miss_threshold
        self.switch_status = "idle"  # idle | switching | error
        self.switch_target: str | None = None
        self.switch_error: str | None = None
        self.uploaded_models: dict[str, str] = {}  # display name -> local directory path
        self.lock = threading.Lock()


def switch_model(holder: ClusterHolder, rig: Rig, model_name: str) -> None:
    """Rebuilds the whole cluster running a different model.

    There's no per-daemon "reload with different weights" RPC -- each
    daemon's model is fixed by its --model flag at process start. So a
    switch means: validate the new model's identifier fast (before
    touching anything), tear down every locally-spawned daemon, respawn
    the same topology under the new model, and re-run the full
    profile/partition/load sequence.

    Devices connected via mode="connect" (real external machines, not
    managed by `rig`) are NOT part of this and are simply lost -- their
    owners need to restart their own daemon with --model matching the new
    choice and reconnect.

    If teardown has already started when something fails (a daemon that
    won't start, an OOM loading a huge model), there's no rollback to the
    previous working cluster -- holder.cluster becomes None and the
    caller needs to restart the script. Validating the model identifier
    up front (model_layer_count(), a lightweight config-only fetch) before
    tearing anything down catches the most common failure -- a typo'd or
    nonexistent model name -- without paying that cost.
    """
    old_cluster = holder.cluster
    with holder.lock:
        holder.switch_status = "switching"
        holder.switch_target = model_name
        holder.switch_error = None
    if old_cluster is not None:
        old_cluster.log(f"model switch requested: {model_name!r} -- validating...")

    try:
        num_layers = model_layer_count(model_name)
    except Exception as exc:  # noqa: BLE001 -- reported via switch_error, not raised
        with holder.lock:
            holder.switch_status = "error"
            holder.switch_error = f"couldn't read {model_name!r}'s config: {exc}"
        if old_cluster is not None:
            old_cluster.log(f"model switch failed (validation): {exc}")
        return

    if old_cluster is not None:
        old_cluster.log(f"tearing down cluster to switch to {model_name!r}...")
        old_cluster.stop()
    rig.shutdown_all()

    try:
        handles = {}
        for spec in holder.primary_specs + holder.standby_specs:
            address = f"127.0.0.1:{rig.allocate_port()}"
            handles[spec.node_id] = NodeHandle(node_id=spec.node_id, address=address)
            rig.spawn(spec.node_id, address, spec.scale, model_name=model_name)
        for node in handles.values():
            wait_until_ready(node.address, timeout=180)

        primary_nodes = [handles[s.node_id] for s in holder.primary_specs]
        standby_nodes = [handles[s.node_id] for s in holder.standby_specs]
        profiles = benchmark_nodes(primary_nodes)
        assignments = plan_partition(num_layers, profiles)
        load_shards(primary_nodes, assignments)  # each daemon loads (and may download) the model here

        new_cluster = Cluster(
            primary_nodes, assignments, standby_nodes,
            heartbeat_interval=holder.heartbeat_interval, miss_threshold=holder.miss_threshold,
            model_name=model_name,
        )
        new_cluster.log(
            f"model switched to {model_name!r}: cluster rebuilt with "
            f"{len(holder.primary_specs)} primary, {len(holder.standby_specs)} standby"
        )
        with holder.lock:
            holder.cluster = new_cluster
            holder.switch_status = "idle"
            holder.switch_target = None
    except Exception as exc:  # noqa: BLE001 -- reported via switch_error, not raised
        with holder.lock:
            holder.switch_status = "error"
            holder.switch_error = f"cluster rebuild failed after teardown: {exc} -- restart the script to recover"
            holder.cluster = None


def _parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    """Minimal hand-rolled multipart/form-data parser: stdlib's `cgi`
    module (the traditional way to do this) is deprecated since 3.11 and
    removed in 3.13, and we only need to handle simple text fields plus a
    handful of file parts from a browser-generated FormData body, not the
    general MIME case. Returns (text_fields, [(filename, content), ...]).
    """
    match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type)
    if not match:
        raise ValueError("no multipart boundary in Content-Type")
    boundary = (match.group(1) or match.group(2)).strip().encode()
    delimiter = b"--" + boundary
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    for segment in body.split(delimiter):
        if segment.startswith(b"\r\n"):
            segment = segment[2:]
        if segment.endswith(b"\r\n"):
            segment = segment[:-2]
        if not segment or segment == b"--":
            continue
        if b"\r\n\r\n" not in segment:
            continue
        headers_blob, content = segment.split(b"\r\n\r\n", 1)
        headers_text = headers_blob.decode(errors="replace")
        name_match = re.search(r'name="([^"]*)"', headers_text)
        if not name_match:
            continue
        filename_match = re.search(r'filename="([^"]*)"', headers_text)
        if filename_match and filename_match.group(1):
            files.append((filename_match.group(1), content))
        else:
            fields[name_match.group(1)] = content.decode(errors="replace")
    return fields, files


def _make_handler(holder: ClusterHolder, rig: Rig):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 -- stdlib signature
            pass  # quiet; the dashboard polls every second, default logging is just noise

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _switch_snapshot(self) -> dict:
            with holder.lock:
                return {
                    "status": holder.switch_status,
                    "target": holder.switch_target,
                    "error": holder.switch_error,
                }

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                cluster = holder.cluster
                payload = cluster.status() if cluster is not None else {
                    "model_name": None, "num_layers": 0, "nodes": [], "standby_ids": [],
                    "needs_rebalance": False, "jobs_submitted": 0, "jobs_recovered": 0,
                    "last_job_ms": None, "latency_history": [], "events": [],
                }
                payload["model_switch"] = self._switch_snapshot()
                self._send_json(200, payload)
            elif parsed.path == "/api/models":
                with holder.lock:
                    uploaded = [{"name": n, "path": p} for n, p in holder.uploaded_models.items()]
                cluster = holder.cluster
                self._send_json(200, {
                    "current_model": cluster.model_name if cluster else None,
                    "curated": CURATED_MODELS,
                    "uploaded": uploaded,
                    "model_switch": self._switch_snapshot(),
                })
            elif parsed.path == "/api/generate/stream":
                self._handle_generate_stream(parse_qs(parsed.query))
            elif parsed.path in ("/", "/index.html"):
                body = DASHBOARD_HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:
            if self.path == "/api/models/upload":
                self._handle_upload_model()
                return

            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON body"})
                return

            if self.path == "/api/devices":
                self._handle_add_device(payload)
            elif self.path.startswith("/api/devices/") and self.path.endswith("/kill"):
                node_id = self.path[len("/api/devices/") : -len("/kill")]
                self._handle_kill(node_id)
            elif self.path == "/api/models/switch":
                self._handle_switch_model(payload)
            else:
                self.send_response(404)
                self.end_headers()

        def _known_node_ids(self) -> set[str]:
            cluster = holder.cluster
            if cluster is None:
                return set()
            status = cluster.status()
            return {n["node_id"] for n in status["nodes"]} | set(status["standby_ids"])

        def _handle_add_device(self, payload: dict) -> None:
            cluster = holder.cluster
            if cluster is None or holder.switch_status != "idle":
                self._send_json(503, {"error": "cluster is unavailable (a model switch is in progress or failed)"})
                return

            node_id = str(payload.get("node_id", "")).strip()
            join_as = payload.get("join_as", "active")
            mode = payload.get("mode", "spawn")

            if not node_id:
                self._send_json(400, {"error": "node_id is required"})
                return
            if join_as not in ("active", "standby"):
                self._send_json(400, {"error": "join_as must be 'active' or 'standby'"})
                return
            if mode not in ("spawn", "connect"):
                self._send_json(400, {"error": "mode must be 'spawn' or 'connect'"})
                return
            if node_id in self._known_node_ids() or node_id in rig.processes:
                self._send_json(409, {"error": f"a device named {node_id!r} already exists"})
                return

            if mode == "connect":
                address = str(payload.get("address", "")).strip()
                if not address:
                    self._send_json(400, {"error": "address is required when mode='connect'"})
                    return
            else:
                try:
                    scale = float(payload.get("scale", 1.0))
                except (TypeError, ValueError):
                    self._send_json(400, {"error": "scale must be a number"})
                    return
                address = f"127.0.0.1:{rig.allocate_port()}"
                # Same model as the cluster is already running -- a locally
                # spawned "simulated" device has no reason to load anything
                # else, and add_device()'s model-mismatch check would just
                # reject it otherwise.
                rig.spawn(node_id, address, scale, model_name=cluster.model_name)

            def onboard() -> None:
                try:
                    cluster.add_device(NodeHandle(node_id=node_id, address=address), join_as=join_as)
                except Exception as exc:  # noqa: BLE001 -- surfaced to the dashboard event log, not raised
                    cluster.log(f"device join failed: {node_id} ({exc})")

            threading.Thread(target=onboard, daemon=True).start()
            status_word = "spawning" if mode == "spawn" else "connecting"
            self._send_json(202, {"status": status_word, "node_id": node_id, "address": address})

        def _handle_kill(self, node_id: str) -> None:
            cluster = holder.cluster
            if cluster is not None:
                cluster.log(f"kill requested via dashboard: {node_id}")
            killed = rig.kill(node_id)
            if killed:
                self._send_json(202, {"status": "killed", "node_id": node_id})
            else:
                self._send_json(404, {"error": f"no running process for {node_id!r} (not a locally-spawned device)"})

        def _handle_generate_stream(self, query: dict) -> None:
            cluster = holder.cluster
            if cluster is None or holder.switch_status != "idle":
                self._send_json(503, {"error": "cluster is unavailable (a model switch is in progress or failed)"})
                return

            prompt = (query.get("prompt", [""])[0] or "").strip()
            if not prompt:
                self._send_json(400, {"error": "prompt is required"})
                return
            try:
                max_new_tokens = int(query.get("max_new_tokens", ["40"])[0])
            except ValueError:
                max_new_tokens = 40
            max_new_tokens = max(1, min(max_new_tokens, MAX_GENERATE_TOKENS))
            try:
                temperature = float(query.get("temperature", ["0"])[0])
            except ValueError:
                temperature = 0.0

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            cluster.log(f"playground: generating for prompt {prompt[:40]!r}")
            try:
                for piece in cluster.generate_stream(prompt, max_new_tokens=max_new_tokens, temperature=temperature):
                    self.wfile.write(f"data: {json.dumps({'piece': piece})}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(f"data: {json.dumps({'done': True})}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # client navigated away mid-stream
            except Exception as exc:  # noqa: BLE001 -- reported to the client as an SSE event, not raised
                try:
                    self.wfile.write(f"data: {json.dumps({'error': str(exc)})}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _handle_switch_model(self, payload: dict) -> None:
            model_name = str(payload.get("model_name", "")).strip()
            if not model_name:
                self._send_json(400, {"error": "model_name is required"})
                return
            with holder.lock:
                if holder.switch_status == "switching":
                    self._send_json(409, {"error": "a model switch is already in progress"})
                    return
            threading.Thread(target=switch_model, args=(holder, rig, model_name), daemon=True).start()
            self._send_json(202, {"status": "switching", "model_name": model_name})

        def _handle_upload_model(self) -> None:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_json(400, {"error": "expected multipart/form-data"})
                return
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_UPLOAD_BYTES:
                self._send_json(413, {"error": f"upload too large (max {MAX_UPLOAD_BYTES // 1024**3}GB)"})
                return
            body = self.rfile.read(length) if length else b""
            try:
                fields, files = _parse_multipart(body, content_type)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            raw_name = fields.get("model_name", "").strip()
            slug = re.sub(r"[^a-zA-Z0-9_-]", "-", raw_name)[:64].strip("-")
            if not slug:
                self._send_json(400, {"error": "model_name is required"})
                return
            if not files:
                self._send_json(400, {"error": "no files were uploaded"})
                return

            target_dir = UPLOAD_DIR / slug
            target_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in files:
                safe_name = Path(filename).name  # strip any directory components
                if safe_name:
                    (target_dir / safe_name).write_bytes(content)

            try:
                # Reuses the same validation switch_model() will run before
                # tearing anything down -- including its explicit
                # config.json check (see model_layer_count()'s docstring
                # for why GPT2Config.from_pretrained() alone isn't enough).
                model_layer_count(str(target_dir))
            except Exception as exc:  # noqa: BLE001 -- reported to the client, not raised
                shutil.rmtree(target_dir, ignore_errors=True)
                self._send_json(400, {"error": f"uploaded files don't look like a valid GPT-2 model: {exc}"})
                return

            with holder.lock:
                holder.uploaded_models[slug] = str(target_dir)
            self._send_json(200, {"status": "uploaded", "name": slug, "path": str(target_dir)})

    return Handler


def serve(holder: ClusterHolder, rig: Rig, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _make_handler(holder, rig))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
