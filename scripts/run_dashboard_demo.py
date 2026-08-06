"""Phase 4 demo: a live, animated dashboard over a running Cluster.

Spawns a handful of primary daemons plus a couple of idle standbys, wires
them into a Cluster, starts the dashboard's HTTP server, and opens it in
your browser. A background thread keeps submitting jobs every couple of
seconds so the dashboard stays alive with real activity -- job counts,
latency, and the pipeline flow animation all reflect real RPCs, not
canned data. From the browser you can add a brand new device (spawns a
fresh local daemon and onboards it live) or kill an existing one
(SIGKILLs it to exercise the same recovery + rebalance path the earlier
fault-tolerance demos used) and watch the dashboard react in real time.

Runs until Ctrl+C.
"""

import threading
import time
import webbrowser

import grpc
import torch
from transformers import GPT2Tokenizer

from mesh.cluster import Cluster
from mesh.coordinator import plan_partition
from mesh.dashboard_server import serve
from mesh.net_coordinator import NodeHandle, benchmark_nodes, load_shards
from mesh.proto import mesh_pb2, mesh_pb2_grpc
from mesh.rig import Rig

PRIMARY_DEVICES = [
    ("thinkpad-cpu", 0.5),
    ("old-macbook", 0.8),
    ("m1-air", 1.4),
    ("gaming-laptop-4060", 2.3),
]
STANDBY_DEVICES = [
    ("spare-chromebook", 1.0),
    ("spare-desktop", 2.0),
]
BASE_PORT = 59000
DASHBOARD_PORT = 8080
JOB_INTERVAL_SECONDS = 2.5
PROMPT = "The best way to distribute a language model across a campus network is"


def wait_until_ready(address: str, timeout: float = 60.0) -> None:
    channel = grpc.insecure_channel(address)
    grpc.channel_ready_future(channel).result(timeout=timeout)
    mesh_pb2_grpc.NodeDaemonStub(channel).Heartbeat(mesh_pb2.HeartbeatRequest(), timeout=timeout)


def job_loop(cluster: Cluster, input_ids: torch.Tensor, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            cluster.submit(input_ids)
        except Exception as exc:  # noqa: BLE001 -- keep the loop alive; log and move on
            cluster.log(f"job failed: {exc}")
        stop.wait(JOB_INTERVAL_SECONDS)


def main() -> None:
    rig = Rig(base_port=BASE_PORT)
    all_devices = PRIMARY_DEVICES + STANDBY_DEVICES
    handles = {}
    for node_id, scale in all_devices:
        # Goes through rig.allocate_port() (not a hand-computed offset) so
        # its internal counter stays consistent -- devices added later via
        # the dashboard's "Add device" form get ports that don't collide.
        address = f"127.0.0.1:{rig.allocate_port()}"
        handles[node_id] = NodeHandle(node_id=node_id, address=address)
        rig.spawn(node_id, address, scale)

    print(f"Starting {len(all_devices)} node daemons...")
    cluster = None
    server = None
    stop = threading.Event()
    try:
        for node in handles.values():
            wait_until_ready(node.address)
        print("All daemons ready.\n")

        primary_nodes = [handles[node_id] for node_id, _ in PRIMARY_DEVICES]
        standby_nodes = [handles[node_id] for node_id, _ in STANDBY_DEVICES]

        profiles = benchmark_nodes(primary_nodes)
        assignments = plan_partition(12, profiles)
        load_shards(primary_nodes, assignments)

        cluster = Cluster(
            # heartbeat_interval/miss_threshold are looser than the fault
            # -tolerance tests use: this demo runs 6+ daemons sharing one
            # physical CPU, and a tight deadline reads transient
            # contention (e.g. a rebalance's benchmark step) as a false
            # node death. Real, separate devices wouldn't need this slack.
            primary_nodes, assignments, standby_nodes,
            heartbeat_interval=3.0, miss_threshold=3, model_name="gpt2",
        )

        server = serve(cluster, rig, port=DASHBOARD_PORT)
        url = f"http://127.0.0.1:{DASHBOARD_PORT}"
        print(f"Dashboard running at {url}")
        webbrowser.open(url)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

        loop_thread = threading.Thread(target=job_loop, args=(cluster, input_ids, stop), daemon=True)
        loop_thread.start()

        print("Submitting a job every "
              f"{JOB_INTERVAL_SECONDS}s. Open the dashboard and try adding or killing a device.")
        print("Press Ctrl+C to stop.\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop.set()
        if server is not None:
            server.shutdown()
        if cluster is not None:
            cluster.stop()
        rig.shutdown_all()
        print("Done.")


if __name__ == "__main__":
    main()
