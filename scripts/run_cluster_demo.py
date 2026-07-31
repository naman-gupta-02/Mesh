"""Phase 3.5 demo: the Cluster wrapper heals routing across jobs, not just
the one job that failed.

Runs three jobs against the same cluster:
  1. A normal job -- baseline, no failures.
  2. A job during which we SIGKILL a node mid-request. Cluster.submit()
     catches the failure internally and recovers automatically.
  3. Another normal job, submitted *after* the dead node has been fully
     replaced by the standby -- the point being that this one succeeds on
     the first try, with no recovery needed, because the pipeline's actual
     wiring was healed after job 2 (not just that one job patched over).
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from mesh.cluster import Cluster
from mesh.coordinator import plan_partition
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards
from mesh.proto import mesh_pb2, mesh_pb2_grpc

PRIMARY_DEVICES = [
    ("thinkpad-cpu", 0.5),
    ("old-macbook", 0.8),
    ("m1-air", 1.4),
    ("gaming-laptop-4060", 2.3),
]
STANDBY_DEVICE = ("spare-chromebook", 1.0)
BASE_PORT = 55000
DOOMED_NODE_ID = "m1-air"
DOOMED_ARTIFICIAL_DELAY = 3.0
KILL_AFTER = 0.5
PROMPT = "The best way to distribute a language model across a campus network is"


def wait_until_ready(address: str, timeout: float = 60.0) -> None:
    channel = grpc.insecure_channel(address)
    grpc.channel_ready_future(channel).result(timeout=timeout)
    mesh_pb2_grpc.NodeDaemonStub(channel).Heartbeat(mesh_pb2.HeartbeatRequest(), timeout=timeout)


def spawn_daemon(node_id: str, address: str, scale: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "mesh.daemon", "--node-id", node_id, "--address", address,
         "--simulated-scale", str(scale)]
    )


def check(label: str, result, reference_logits: torch.Tensor, tokenizer) -> None:
    max_abs_diff = (result.logits - reference_logits).abs().max().item()
    token = tokenizer.decode([result.logits[0, -1].argmax().item()])
    ok = max_abs_diff < 1e-3
    status = "PASS" if ok else "FAIL"
    print(f"[{label}] {status} -- max abs diff {max_abs_diff:.3e}, next token {token!r}, "
          f"stages: {[t.node_id for t in result.stage_timings]}")
    if not ok:
        sys.exit(1)


def main() -> None:
    all_devices = PRIMARY_DEVICES + [STANDBY_DEVICE]
    handles = {
        node_id: NodeHandle(node_id=node_id, address=f"127.0.0.1:{BASE_PORT + i}")
        for i, (node_id, _scale) in enumerate(all_devices)
    }
    primary_nodes = [handles[node_id] for node_id, _ in PRIMARY_DEVICES]
    standby = handles[STANDBY_DEVICE[0]]

    print(f"Starting {len(all_devices)} node daemons ({len(PRIMARY_DEVICES)} primary + 1 standby)...")
    processes = {node_id: spawn_daemon(node_id, handles[node_id].address, scale) for node_id, scale in all_devices}

    try:
        for node in handles.values():
            wait_until_ready(node.address)
        print("All daemons ready.\n")

        profiles = benchmark_nodes(primary_nodes)
        assignments = plan_partition(12, profiles)
        print("Initial partition plan:")
        for a in assignments:
            print(f"  {a.node_id:<22} layers [{a.layer_start:>2}, {a.layer_end:>2})")
        load_shards(primary_nodes, assignments)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        with torch.no_grad():
            reference_logits = model(input_ids).logits

        cluster = Cluster(primary_nodes, assignments, [standby], heartbeat_interval=1.0, miss_threshold=2)

        print("\n--- Job 1: normal ---")
        result1 = cluster.submit(input_ids, job_id="job-1")
        check("job-1", result1, reference_logits, tokenizer)

        print(f"\n--- Job 2: killing {DOOMED_NODE_ID} mid-request ---")
        cluster.inject_delay(DOOMED_NODE_ID, DOOMED_ARTIFICIAL_DELAY)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(cluster.submit, input_ids, "job-2")
            time.sleep(KILL_AFTER)
            print(f"Killing {DOOMED_NODE_ID} (pid {processes[DOOMED_NODE_ID].pid})...")
            processes[DOOMED_NODE_ID].kill()
            processes[DOOMED_NODE_ID].wait(timeout=10)
            result2 = future.result()  # Cluster.submit() recovers internally -- should not raise
        check("job-2", result2, reference_logits, tokenizer)
        print(f"Healed partition plan (assignments after recovery):")
        for a in cluster.assignments:
            print(f"  {a.node_id:<22} layers [{a.layer_start:>2}, {a.layer_end:>2})")

        print("\n--- Job 3: normal again, should NOT need recovery ---")
        result3 = cluster.submit(input_ids, job_id="job-3")
        check("job-3", result3, reference_logits, tokenizer)
        if DOOMED_NODE_ID in {t.node_id for t in result3.stage_timings}:
            print(f"FAIL: job-3 still routed through dead node {DOOMED_NODE_ID}")
            sys.exit(1)
        if standby.node_id not in {t.node_id for t in result3.stage_timings}:
            print(f"FAIL: job-3 did not route through standby {standby.node_id}")
            sys.exit(1)
        print(f"Confirmed: job-3 routed through {standby.node_id} directly, no recovery needed.")

        time.sleep(2.5)  # let the heartbeat loop independently notice the dead node
        print(f"\ncluster.is_alive({DOOMED_NODE_ID!r}) = {cluster.is_alive(DOOMED_NODE_ID)}")
        print(f"cluster.is_alive({standby.node_id!r}) = {cluster.is_alive(standby.node_id)}")

        cluster.stop()
        print("\nPASS: all three jobs produced correct output; routing was healed after job 2.")
    finally:
        print("\nShutting down node daemons...")
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
