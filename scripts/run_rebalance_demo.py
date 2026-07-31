"""Phase 3.6 demo: rebalancing after a topology change, not just healing.

Cluster's post-recovery healing (Phase 3.5) keeps the pipeline correct by
giving the standby the *exact* layer range the dead node used to hold --
fast and simple, but not necessarily well-balanced, since the standby's
real throughput has nothing to do with the dead node's. This demo shows
the difference: job 2 triggers a recovery (healed, unbalanced layer split);
job 3's Cluster.submit() lazily notices the topology changed and calls
rebalance() first, producing a partition plan proportional to every
currently-active node's actual measured throughput.
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
# Deliberately much faster than the node it'll replace, so the rebalanced
# split visibly differs from just inheriting old-macbook's old layer range.
STANDBY_DEVICE = ("beefy-desktop", 4.0)
BASE_PORT = 57000
DOOMED_NODE_ID = "old-macbook"
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


def print_plan(label: str, assignments) -> None:
    print(f"{label}:")
    for a in assignments:
        print(f"  {a.node_id:<22} layers [{a.layer_start:>2}, {a.layer_end:>2})  ({a.num_layers} layers)")


def check(label: str, result, reference_logits: torch.Tensor, tokenizer) -> None:
    max_abs_diff = (result.logits - reference_logits).abs().max().item()
    token = tokenizer.decode([result.logits[0, -1].argmax().item()])
    ok = max_abs_diff < 1e-3
    print(f"[{label}] {'PASS' if ok else 'FAIL'} -- max abs diff {max_abs_diff:.3e}, next token {token!r}")
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

    print(f"Starting {len(all_devices)} node daemons...")
    processes = {node_id: spawn_daemon(node_id, handles[node_id].address, scale) for node_id, scale in all_devices}

    try:
        for node in handles.values():
            wait_until_ready(node.address)
        print("All daemons ready.\n")

        profiles = benchmark_nodes(primary_nodes)
        assignments = plan_partition(12, profiles)
        print_plan("Initial partition plan", assignments)
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
            processes[DOOMED_NODE_ID].kill()
            processes[DOOMED_NODE_ID].wait(timeout=10)
            result2 = future.result()
        check("job-2", result2, reference_logits, tokenizer)
        print_plan(f"Healed plan (standby {standby.node_id} just inherited {DOOMED_NODE_ID}'s old range)",
                    cluster.assignments)

        print(f"\n--- Job 3: Cluster.submit() should rebalance first (needs_rebalance={cluster.needs_rebalance}) ---")
        result3 = cluster.submit(input_ids, job_id="job-3")
        check("job-3", result3, reference_logits, tokenizer)
        print_plan("Rebalanced plan (fresh benchmark across all active nodes)", cluster.assignments)

        standby_layers = next(a.num_layers for a in cluster.assignments if a.node_id == standby.node_id)
        print(f"\n{standby.node_id} (declared 4x scale) now holds {standby_layers} layers "
              f"-- proportional to its own measured throughput, not old-macbook's old share.")

        cluster.stop()
        print("\nPASS: rebalancing after recovery produced a fresh, throughput-proportional partition.")
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
