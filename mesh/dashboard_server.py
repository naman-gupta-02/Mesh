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
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mesh.cluster import Cluster
from mesh.net_coordinator import NodeHandle
from mesh.rig import Rig

DASHBOARD_HTML_PATH = Path(__file__).parent / "dashboard.html"
MAX_GENERATE_TOKENS = 200  # demo sanity cap -- each token is a full pipeline pass


def _make_handler(cluster: Cluster, rig: Rig):
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

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                self._send_json(200, cluster.status())
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
            else:
                self.send_response(404)
                self.end_headers()

        def _known_node_ids(self) -> set[str]:
            status = cluster.status()
            return {n["node_id"] for n in status["nodes"]} | set(status["standby_ids"])

        def _handle_add_device(self, payload: dict) -> None:
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
            cluster.log(f"kill requested via dashboard: {node_id}")
            killed = rig.kill(node_id)
            if killed:
                self._send_json(202, {"status": "killed", "node_id": node_id})
            else:
                self._send_json(404, {"error": f"no running process for {node_id!r} (not a locally-spawned device)"})

        def _handle_generate_stream(self, query: dict) -> None:
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

    return Handler


def serve(cluster: Cluster, rig: Rig, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _make_handler(cluster, rig))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
