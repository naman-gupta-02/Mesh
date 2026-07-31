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

## Phase 3 — fault tolerance

The headline failure mode from the plan: a volunteer's laptop dies mid-job.
Every node now caches its own output per `job_id` in memory; when the chain
breaks, the coordinator doesn't restart the whole job — it finds the last
node that finished, reassigns the dead node's layer range to a standby, and
resumes from that node's cached checkpoint.

- `mesh/proto/mesh.proto` — added `GetCheckpoint` (returns a node's cached
  output for a `job_id`, or `available=false`) and `artificial_delay_seconds`
  on `LoadShardRequest` (a testing knob: sleep before computing, so a
  fault-injection harness has a deterministic window to kill the process).
- `mesh/daemon.py` — `Forward` now caches `encode(output)` per `job_id`
  before relaying downstream, and serves it back via `GetCheckpoint`.
- `mesh/net_coordinator.py` — `find_last_checkpoint()` walks the pipeline
  querying each node's checkpoint to find how far a broken job got;
  `recover_job()` loads the failed shard's spec onto a standby (same layer
  range, same original `next_hop_address`) and resumes from there.

Recovery limitation, by design for now: if a node dies *after* computing its
own output but *before* its downstream RPC call returns, that in-flight
output is lost with it (it only lived in the dead process's memory) — the
standby just recomputes that one shard from the last node that's still
alive. Also, stage timings from before the failure aren't preserved across
recovery (the original blocking call raised before returning anything);
only timings from the standby onward are reported. Both are "correctness
over speed" trade-offs, matching the plan's stated priority for v1.

### Run it

```
python3 scripts/run_fault_injection_demo.py
```

Starts 4 primary daemons plus 1 idle standby, submits a job, `SIGKILL`s one
primary node partway through its artificial delay (before it produces any
output), then shows the coordinator catching the failed RPC, reassigning
that node's shard to the standby, and resuming — with final output still
checked against a monolithic forward pass.

### Test

```
pytest tests/
```

`tests/test_fault_tolerance.py` automates the same kill-mid-request scenario
and asserts the recovered output matches the monolithic model, and that the
dead node's ID doesn't appear in the final stage timings while the
standby's does.

## What's next (not built yet)

Per the original build order: run on real separate hardware (not just
localhost — architecture doesn't change, just point `--address` at a LAN
IP), then a closed campus beta. Also pending: a heartbeat loop that
proactively detects failures on a timer rather than reacting to a failed
RPC (the `Heartbeat` RPC exists but nothing polls it yet), re-planning the
partition after recovery so future jobs don't route through the dead node
again, sandboxed node daemons, signed/checksummed weight distribution, and
latency-aware scheduling.
