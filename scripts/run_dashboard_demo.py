"""Phase 4/5 demo: a live, animated dashboard over a running Cluster.

Spawns a handful of primary daemons plus a couple of idle standbys, wires
them into a Cluster, starts the dashboard's HTTP server, and opens it in
your browser. A background thread keeps submitting jobs every couple of
seconds so the dashboard stays alive with real activity -- job counts,
latency, and the pipeline flow animation all reflect real RPCs, not
canned data. From the browser you can add a brand new device (spawns a
fresh local daemon and onboards it live) or kill an existing one
(SIGKILLs it to exercise the same recovery + rebalance path the earlier
fault-tolerance demos used), ask the model something in the Playground
tab, or switch the whole cluster to a different model from the Model tab
(picking from a curated list, typing any Hugging Face Hub identifier, or
uploading your own GPT-2-format checkpoint) -- and watch the dashboard
react in real time.

Runs until Ctrl+C.
"""

import argparse
import threading
import time
import webbrowser

from mesh.cluster import Cluster
from mesh.coordinator import model_layer_count, plan_partition
from mesh.dashboard_server import ClusterHolder, serve
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards
from mesh.rig import DeviceSpec, Rig, wait_until_ready

PRIMARY_DEVICES = [
    DeviceSpec("thinkpad-cpu", 0.5),
    DeviceSpec("old-macbook", 0.8),
    DeviceSpec("m1-air", 1.4),
    DeviceSpec("gaming-laptop-4060", 2.3),
]
STANDBY_DEVICES = [
    DeviceSpec("spare-chromebook", 1.0),
    DeviceSpec("spare-desktop", 2.0),
]
BASE_PORT = 59000
DASHBOARD_PORT = 8080
JOB_INTERVAL_SECONDS = 2.5
PROMPT = "The best way to distribute a language model across a campus network is"
# heartbeat_interval/miss_threshold are looser than the fault-tolerance
# tests use: this demo runs 6+ daemons sharing one physical CPU, and a
# tight deadline reads transient contention (e.g. a rebalance's benchmark
# step) as a false node death. Real, separate devices wouldn't need this.
HEARTBEAT_INTERVAL = 3.0
MISS_THRESHOLD = 3


def job_loop(holder: ClusterHolder, prompt: str, stop: threading.Event) -> None:
    """Reads holder.cluster fresh every iteration (not a fixed Cluster
    reference) and re-tokenizes the prompt against whatever model that
    cluster is currently running, so a model switch (see
    dashboard_server.py's switch_model()) is picked up automatically on
    the next tick instead of needing the loop restarted.
    """
    while not stop.is_set():
        cluster = holder.cluster
        if cluster is not None and holder.switch_status == "idle":
            try:
                input_ids = cluster.tokenize(prompt)
                cluster.submit(input_ids)
            except Exception as exc:  # noqa: BLE001 -- keep the loop alive; log and move on
                cluster.log(f"job failed: {exc}")
        stop.wait(JOB_INTERVAL_SECONDS)


def build_cluster(rig: Rig, model_name: str) -> Cluster:
    """Spawns the demo's fixed device topology under `model_name`, waits
    for it to be ready, and profiles/partitions/loads it into a fresh
    Cluster. Used both for the initial startup and (via
    dashboard_server.switch_model, which reimplements this same sequence
    against holder.primary_specs/standby_specs) after a model switch.
    """
    handles = {}
    for spec in PRIMARY_DEVICES + STANDBY_DEVICES:
        address = f"127.0.0.1:{rig.allocate_port()}"
        handles[spec.node_id] = NodeHandle(node_id=spec.node_id, address=address)
        rig.spawn(spec.node_id, address, spec.scale, model_name=model_name)

    for node in handles.values():
        wait_until_ready(node.address)

    primary_nodes = [handles[s.node_id] for s in PRIMARY_DEVICES]
    standby_nodes = [handles[s.node_id] for s in STANDBY_DEVICES]

    num_layers = model_layer_count(model_name)
    profiles = benchmark_nodes(primary_nodes)
    assignments = plan_partition(num_layers, profiles)
    load_shards(primary_nodes, assignments)

    return Cluster(
        primary_nodes, assignments, standby_nodes,
        heartbeat_interval=HEARTBEAT_INTERVAL, miss_threshold=MISS_THRESHOLD, model_name=model_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Mesh live dashboard demo.")
    parser.add_argument(
        "--model", default="gpt2",
        help="Any GPT-2-family checkpoint on the HF Hub (gpt2, gpt2-medium, gpt2-large, distilgpt2, ...), "
             "or a local directory path. Can also be changed later from the dashboard's Model tab.",
    )
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT, help="Dashboard HTTP port.")
    parser.add_argument("--prompt", default=PROMPT, help="Prompt the background job loop keeps submitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rig = Rig(base_port=BASE_PORT)

    print(f"Starting {len(PRIMARY_DEVICES) + len(STANDBY_DEVICES)} node daemons...")
    server = None
    holder = None
    stop = threading.Event()
    try:
        cluster = build_cluster(rig, args.model)
        print("All daemons ready.\n")

        holder = ClusterHolder(
            cluster, PRIMARY_DEVICES, STANDBY_DEVICES,
            heartbeat_interval=HEARTBEAT_INTERVAL, miss_threshold=MISS_THRESHOLD,
        )

        server = serve(holder, rig, port=args.port)
        url = f"http://127.0.0.1:{args.port}"
        print(f"Dashboard running at {url}")
        webbrowser.open(url)

        loop_thread = threading.Thread(target=job_loop, args=(holder, args.prompt, stop), daemon=True)
        loop_thread.start()

        print("Submitting a job every "
              f"{JOB_INTERVAL_SECONDS}s. Open the dashboard and try adding/killing a device, "
              "asking the model something, or switching models.")
        print("Press Ctrl+C to stop.\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop.set()
        if server is not None:
            server.shutdown()
        if holder is not None and holder.cluster is not None:
            holder.cluster.stop()
        rig.shutdown_all()
        print("Done.")


if __name__ == "__main__":
    main()
