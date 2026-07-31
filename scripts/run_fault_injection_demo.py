"""Phase 3 demo: the "laptop closes mid-class" scenario.

Starts 4 primary node daemons plus 1 idle standby, submits a job, then
SIGKILLs one of the primaries partway through -- before it produces any
output. The coordinator detects the broken chain, reassigns the dead node's
layer range to the standby, and resumes from the last surviving node's
cached checkpoint instead of restarting the whole job. Final output is
checked against a monolithic forward pass.
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from mesh.coordinator import plan_partition
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards, recover_job, submit_job
from mesh.proto import mesh_pb2, mesh_pb2_grpc

PRIMARY_DEVICES = [
    ("thinkpad-cpu", 0.5),
    ("old-macbook", 0.8),
    ("m1-air", 1.4),
    ("gaming-laptop-4060", 2.3),
]
STANDBY_DEVICE = ("spare-chromebook", 1.0)
BASE_PORT = 52000
DOOMED_NODE_ID = "old-macbook"  # the node we'll kill mid-request
DOOMED_ARTIFICIAL_DELAY = 3.0  # seconds -- gives us a window to kill it
KILL_AFTER = 0.5  # seconds after submitting the job
PROMPT = "The best way to distribute a language model across a campus network is"


def wait_until_ready(address: str, timeout: float = 60.0) -> None:
    channel = grpc.insecure_channel(address)
    grpc.channel_ready_future(channel).result(timeout=timeout)
    mesh_pb2_grpc.NodeDaemonStub(channel).Heartbeat(mesh_pb2.HeartbeatRequest(), timeout=timeout)


def spawn_daemon(node_id: str, address: str, scale: float) -> subprocess.Popen:
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


def main() -> None:
    all_devices = PRIMARY_DEVICES + [STANDBY_DEVICE]
    handles = {
        node_id: NodeHandle(node_id=node_id, address=f"127.0.0.1:{BASE_PORT + i}")
        for i, (node_id, _scale) in enumerate(all_devices)
    }
    primary_nodes = [handles[node_id] for node_id, _ in PRIMARY_DEVICES]
    standby = handles[STANDBY_DEVICE[0]]

    print(f"Starting {len(all_devices)} node daemons ({len(PRIMARY_DEVICES)} primary + 1 standby)...")
    processes = {
        node_id: spawn_daemon(node_id, handles[node_id].address, scale) for node_id, scale in all_devices
    }

    try:
        for node_id, node in handles.items():
            wait_until_ready(node.address)
        print("All daemons ready.\n")

        profiles = benchmark_nodes(primary_nodes)
        num_layers = 12  # gpt2
        assignments = plan_partition(num_layers, profiles)
        print(f"Partition plan ({num_layers} layers total):")
        for a in assignments:
            print(f"  {a.node_id:<22} layers [{a.layer_start:>2}, {a.layer_end:>2})  ({a.num_layers} layers)")

        load_shards(primary_nodes, assignments, artificial_delays={DOOMED_NODE_ID: DOOMED_ARTIFICIAL_DELAY})
        print(f"\n{DOOMED_NODE_ID} configured with a {DOOMED_ARTIFICIAL_DELAY}s artificial delay "
              f"(fault-injection window).")

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
        entry_node = next(n for n in primary_nodes if n.node_id == assignments[0].node_id)
        job_id = "fault-injection-demo"

        print(f"\nSubmitting job (entry: {entry_node.node_id})...")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(submit_job, entry_node, job_id, input_ids)

            time.sleep(KILL_AFTER)
            print(f"Killing {DOOMED_NODE_ID} (pid {processes[DOOMED_NODE_ID].pid}) mid-request...")
            processes[DOOMED_NODE_ID].kill()
            processes[DOOMED_NODE_ID].wait(timeout=10)
            print(f"{DOOMED_NODE_ID} is dead.\n")

            try:
                result = future.result()
                print("Unexpected: job succeeded without recovery.")
            except grpc.RpcError as exc:
                print(f"Job failed as expected: {exc.code()}")
                print(f"\nRecovering: reassigning {DOOMED_NODE_ID}'s shard to {standby.node_id}...")
                result = recover_job(primary_nodes, assignments, standby, job_id, input_ids)

        print("\nStage timings (from the point of recovery onward):")
        for t in result.stage_timings:
            print(f"  {t.node_id:<22} {t.elapsed_seconds * 1000:7.1f} ms")

        print("\nRunning monolithic model for comparison...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        with torch.no_grad():
            reference_logits = model(input_ids).logits

        max_abs_diff = (result.logits - reference_logits).abs().max().item()
        distributed_next_token = result.logits[0, -1].argmax().item()
        reference_next_token = reference_logits[0, -1].argmax().item()

        print(f"\nMax abs logit diff:     {max_abs_diff:.6e}")
        print(f"Recovered next token:   {tokenizer.decode([distributed_next_token])!r}")
        print(f"Reference next token:   {tokenizer.decode([reference_next_token])!r}")

        tolerance = 1e-3
        if max_abs_diff < tolerance and distributed_next_token == reference_next_token:
            print(f"\nPASS: recovered output matches monolithic model despite node death (tol={tolerance})")
        else:
            print(f"\nFAIL: recovered output diverges from monolithic model (tol={tolerance})")
            sys.exit(1)
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
