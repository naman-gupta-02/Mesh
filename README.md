# Mesh

Distributed LLM inference across heterogeneous volunteer devices (dorm-room
laptops, not a datacenter). Layer-parallel pipeline: the model is split into
contiguous ranges of transformer blocks, each held by a different node,
chained so activations flow node-to-node through the network.

## Phase 1 — single-machine simulation

Before any real networking: prove the sharding + reassembly logic is correct
by faking multiple "nodes" as local processes on one machine.

- `mesh/profiler.py` — per-node throughput micro-benchmark (real matmul
  timing), scaled to simulate the hardware heterogeneity a single dev
  machine can't otherwise produce.
- `mesh/coordinator.py` — greedy weighted layer partition: nodes get a layer
  count proportional to their measured throughput share, so every pipeline
  stage takes roughly the same wall-clock time.
- `mesh/model_shard.py` — a contiguous slice of a GPT-2 model's transformer
  blocks (entry shard holds the embedding, exit shard holds the LM head).
- `mesh/node.py` — a simulated node: a `multiprocessing.Process` that holds
  its shard warm and runs forward passes on activations from the previous
  stage.
- `mesh/pipeline.py` — orchestrates one process per shard, wired via queues
  in pipeline order; returns final logits plus per-stage timing.

### Run it

```
pip install -e .
python3 scripts/run_simulation.py
```

Splits real GPT-2 (12 layers) across 4 simulated devices with different
throughput profiles, runs a prompt through the distributed pipeline, and
checks the result against a plain (unsplit) forward pass. Passes iff the
distributed and monolithic logits match.

### Test

```
pip install -r requirements.txt
pytest tests/
```

`tests/test_partition.py` covers the partition heuristic (even split, faster
nodes get more layers, every node gets ≥1 layer, contiguous coverage, input
validation). `tests/test_pipeline_correctness.py` checks the distributed
pipeline's output against a monolithic forward pass on a small random model,
across single-node, heterogeneous, and uneven-partition cases.

## Phase 2 — real networking (gRPC, localhost)

Replaces `multiprocessing.Queue` with an actual network transport: each node
is a separate OS process running a gRPC server, reachable at its own
`host:port`. Traffic goes through the real TCP/IP stack and protobuf
serialization. Everything below still runs on localhost (no other machines
available in this environment) — point `--address` at a LAN IP to put a node
on real separate hardware; the protocol doesn't change.

- `mesh/proto/mesh.proto` — the `NodeDaemon` gRPC service: `Benchmark`,
  `LoadShard`, `Forward`, `Heartbeat`. Compiled to `mesh_pb2.py` /
  `mesh_pb2_grpc.py` (checked in; regenerate after editing the `.proto` with
  `python -m grpc_tools.protoc -I mesh/proto --python_out=mesh/proto
  --grpc_python_out=mesh/proto mesh/proto/mesh.proto`, then change the
  generated `import mesh_pb2` to `from . import mesh_pb2` in
  `mesh_pb2_grpc.py`).
- `mesh/tensor_codec.py` — tensor ⇄ bytes for gRPC message fields
  (`torch.save`/`load` round-trip).
- `mesh/daemon.py` — the node daemon: a gRPC server that loads a shard on
  `LoadShard` and, on `Forward`, computes its layers then **synchronously
  relays to its next hop** and returns whatever comes back with its own
  timing prepended. A single `Forward` call to the entry node therefore
  returns the final result once the whole chain has run — no separate
  telemetry channel needed.
- `mesh/net_coordinator.py` — coordinator client: benchmarks nodes over RPC,
  reuses `mesh/coordinator.py`'s partition heuristic, pushes `LoadShard` to
  each node (wiring `next_hop_address` in pipeline order), then submits the
  job to the entry node.

### Run it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/run_networked_simulation.py
```

Spawns 4 node-daemon subprocesses on `127.0.0.1:51000-51003`, benchmarks
them over gRPC, partitions GPT-2's 12 layers across them, loads shards, runs
a prompt through the real network chain, and checks it against a monolithic
forward pass.

**Dependency note:** `mesh/model_shard.py` calls `GPT2Block.forward`
directly (bypassing `GPT2Model`), which is private API. `transformers` 5.x
changed that call signature and breaks it — `torch`/`transformers` are
pinned in `pyproject.toml`/`requirements.txt` to versions known to work.
Use a venv for this project rather than installing into a shared/global
Python environment, to avoid dependency conflicts with other tools.

### Test

```
pytest tests/
```

`tests/test_networked_pipeline.py` spins up real daemon subprocesses,
drives them through the full RPC flow, and checks output against a
monolithic forward pass — the networked equivalent of
`test_pipeline_correctness.py`.

## What's next (not built yet)

Per the original build order: real hardware (not just localhost), fault
injection (kill nodes mid-job, checkpoint/reassignment recovery), then a
closed campus beta. Also pending: a heartbeat loop driving failure
detection (the `Heartbeat` RPC exists but nothing polls it yet on a timer),
sandboxed node daemons, signed/checksummed weight distribution, and
latency-aware scheduling.
