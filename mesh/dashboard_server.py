"""Lightweight HTTP server exposing Cluster.status() as JSON and serving a
self-contained animated dashboard page. Stdlib only (http.server) -- no new
dependency for a small local dev tool.

Two POST endpoints make the dashboard interactive rather than read-only:
/api/devices spawns a brand new local daemon and onboards it via
Cluster.add_device() (the "add a new device" feature); /api/devices/<id>/kill
kills a node's process to trigger the fault-tolerance path live. Both only
work because our demo owns every daemon process as a local subprocess (see
mesh/rig.py) -- see that module's docstring for why this isn't part of the
"real" architecture.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mesh.cluster import Cluster
from mesh.net_coordinator import NodeHandle
from mesh.rig import Rig

DASHBOARD_HTML_PATH = Path(__file__).parent / "dashboard.html"


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
            if self.path == "/api/status":
                self._send_json(200, cluster.status())
            elif self.path in ("/", "/index.html"):
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

        def _handle_add_device(self, payload: dict) -> None:
            node_id = str(payload.get("node_id", "")).strip()
            join_as = payload.get("join_as", "active")
            try:
                scale = float(payload.get("scale", 1.0))
            except (TypeError, ValueError):
                self._send_json(400, {"error": "scale must be a number"})
                return
            if not node_id:
                self._send_json(400, {"error": "node_id is required"})
                return
            if join_as not in ("active", "standby"):
                self._send_json(400, {"error": "join_as must be 'active' or 'standby'"})
                return
            if node_id in rig.processes:
                self._send_json(409, {"error": f"a device named {node_id!r} already exists"})
                return

            address = f"127.0.0.1:{rig.allocate_port()}"
            rig.spawn(node_id, address, scale)

            def onboard() -> None:
                try:
                    cluster.add_device(NodeHandle(node_id=node_id, address=address), join_as=join_as)
                except Exception as exc:  # noqa: BLE001 -- surfaced to the dashboard event log, not raised
                    cluster.log(f"device join failed: {node_id} ({exc})")

            threading.Thread(target=onboard, daemon=True).start()
            self._send_json(202, {"status": "spawning", "node_id": node_id, "address": address})

        def _handle_kill(self, node_id: str) -> None:
            cluster.log(f"kill requested via dashboard: {node_id}")
            killed = rig.kill(node_id)
            if killed:
                self._send_json(202, {"status": "killed", "node_id": node_id})
            else:
                self._send_json(404, {"error": f"no running process for {node_id!r}"})

    return Handler


def serve(cluster: Cluster, rig: Rig, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _make_handler(cluster, rig))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
