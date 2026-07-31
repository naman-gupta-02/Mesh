"""Phase 2 demo: coordinator drives real node daemons over gRPC.

Each daemon is a separate OS process listening on its own localhost port.
Unlike Phase 1 (multiprocessing.Queue, shared memory on one machine), traffic
between nodes now goes through the OS TCP/IP stack and protobuf
serialization -- the same path it takes between two real machines on a
dorm network. Point --address at a LAN IP instead of 127.0.0.1 to actually
run a node on separate hardware.
"""

import subprocess
import sys
import time

import grpc
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from mesh.coordinator import plan_partition
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards, submit_job
from mesh.proto import mesh_pb2, mesh_pb2_grpc

SIMULATED_DEVICES = [
    ("thinkpad-cpu", 0.5),
    ("old-macbook", 0.8),
    ("m1-air", 1.4),
    ("gaming-laptop-4060", 2.3),
]
BASE_PORT = 51000
PROMPT = "The best way to distribute a language model across a campus network is"


def wait_until_ready(address: str, timeout: float = 60.0) -> None:
    channel = grpc.insecure_channel(address)
    grpc.channel_ready_future(channel).result(timeout=timeout)
    mesh_pb2_grpc.NodeDaemonStub(channel).Heartbeat(mesh_pb2.HeartbeatRequest(), timeout=timeout)


def main() -> None:
    nodes = [
        NodeHandle(node_id=node_id, address=f"127.0.0.1:{BASE_PORT + i}")
        for i, (node_id, _scale) in enumerate(SIMULATED_DEVICES)
    ]

    print(f"Starting {len(nodes)} node daemons (separate OS processes)...")
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mesh.daemon",
                "--node-id",
                node.node_id,
                "--address",
                node.address,
                "--simulated-scale",
                str(scale),
            ]
        )
        for node, (_node_id, scale) in zip(nodes, SIMULATED_DEVICES)
    ]

    try:
        for node in nodes:
            wait_until_ready(node.address)
        print("All daemons ready.\n")

        print("Benchmarking nodes over gRPC...")
        profiles = benchmark_nodes(nodes)
        for p in profiles:
            print(f"  {p.node_id:<22} {p.throughput:8.1f} GFLOPS")

        num_layers = 12  # gpt2
        assignments = plan_partition(num_layers, profiles)
        print(f"\nPartition plan ({num_layers} layers total):")
        for a in assignments:
            print(f"  {a.node_id:<22} layers [{a.layer_start:>2}, {a.layer_end:>2})  ({a.num_layers} layers)")

        print("\nLoading shards onto nodes over gRPC...")
        load_shards(nodes, assignments)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

        entry_node = next(n for n in nodes if n.node_id == assignments[0].node_id)

        print("\nSubmitting job over the network...")
        result = submit_job(entry_node, job_id="demo-job-1", input_ids=input_ids)

        print("\nStage timings:")
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
        print(f"Distributed next token: {tokenizer.decode([distributed_next_token])!r}")
        print(f"Reference next token:   {tokenizer.decode([reference_next_token])!r}")

        tolerance = 1e-3
        if max_abs_diff < tolerance and distributed_next_token == reference_next_token:
            print(f"\nPASS: networked distributed output matches monolithic model (tol={tolerance})")
        else:
            print(f"\nFAIL: networked distributed output diverges from monolithic model (tol={tolerance})")
            sys.exit(1)
    finally:
        print("\nShutting down node daemons...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
