"""Process management for local demo/dashboard scripts: spawn and kill
node-daemon subprocesses on localhost, and hand out ports for new ones.

Not part of the "real" architecture -- a real coordinator has no business
killing a volunteer's laptop process. This exists only because our demos
simulate an entire cluster as local subprocesses on one machine, and the
dashboard's "add device" / "kill node" buttons need something to actually
spawn and kill.
"""

import subprocess
import sys
import threading


class Rig:
    def __init__(self, base_port: int):
        self._next_port = base_port
        self.processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def allocate_port(self) -> int:
        with self._lock:
            port = self._next_port
            self._next_port += 1
            return port

    def spawn(self, node_id: str, address: str, scale: float, model_name: str = "gpt2") -> subprocess.Popen:
        proc = subprocess.Popen(
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
                "--model",
                model_name,
            ]
        )
        with self._lock:
            self.processes[node_id] = proc
        return proc

    def kill(self, node_id: str) -> bool:
        with self._lock:
            proc = self.processes.get(node_id)
        if proc is None or proc.poll() is not None:
            return False
        proc.kill()
        return True

    def shutdown_all(self) -> None:
        with self._lock:
            procs = list(self.processes.values())
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
