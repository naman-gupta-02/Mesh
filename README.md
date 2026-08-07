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

## Phase 3.5 — durable healing + heartbeats (`mesh/cluster.py`)

`recover_job()` fixes the one job that was in flight when a node died, but
it doesn't touch the pipeline's actual wiring: the node immediately
upstream of the dead one is still pointed at the now-dead address, so the
very next job would hit the exact same failure and need recovery all over
again. `Cluster` closes that loop and adds proactive liveness tracking:

- Runs a background heartbeat thread (`heartbeat_interval`, `miss_threshold`)
  polling every tracked node's `Heartbeat` RPC independent of job
  submission — `cluster.is_alive(node_id)` reflects this without needing a
  job to fail first.
- `Cluster.submit()` wraps `submit_job()`; on `grpc.RpcError` it recovers
  internally (callers never see the exception) and — the actual fix — also
  updates `self.assignments` and calls the new `rewire_next_hop()` on the
  surviving upstream node, so it now points at the standby instead of the
  dead node. The *next* job submitted routes through the standby on the
  first try, no recovery needed.
- `Cluster.inject_delay()` is the fault-injection testing knob (reissues
  `LoadShard` with a new `artificial_delay_seconds`, everything else
  unchanged) used by the demo/tests to get a deterministic window to kill a
  node.

### Run it

```
python3 scripts/run_cluster_demo.py
```

Three jobs against the same cluster: job 1 normal (baseline), job 2 kills a
node mid-request (`Cluster.submit()` recovers internally), job 3 is
submitted fresh afterward and is checked to route through the standby
*without* touching the dead node at all — proving the healing persisted
past the one recovered job.

### Test

```
pytest tests/
```

`tests/test_cluster.py` automates the same three-job scenario and asserts
job 3's stage timings and `cluster.assignments` no longer reference the
dead node.

## Phase 3.6 — rebalancing after a topology change

Phase 3.5's healing keeps the pipeline correct by giving the standby the
dead node's *exact* old layer range — fast, but that range was sized for
the dead node's throughput, not the standby's. A spare that's 4x faster
than what it replaced would sit underused forever under healing alone.

- `Cluster.rebalance()` — re-profiles every currently-active node
  (`benchmark_nodes()`) and recomputes the partition from scratch
  (`plan_partition()`), then reissues `LoadShard` to all of them
  (`load_shards()`). Deliberately *not* called during recovery itself —
  recovery stays fast and correctness-first; a full re-plan is heavier and
  can wait.
- `Cluster.submit()` now checks a `needs_rebalance` flag (set whenever
  `_recover_and_heal` changes the topology) and calls `rebalance()` first
  if it's set, before submitting the job — matching the original plan's
  "you don't need to reshard immediately, but the next job submission
  should re-profile and re-plan."

### Run it

```
python3 scripts/run_rebalance_demo.py
```

Same kill-mid-request scenario as Phase 3.5, but the standby is configured
much faster than the node it replaces. Job 2's healed plan gives it the
dead node's old (small) share; job 3 shows `Cluster.submit()` rebalancing
first, and the standby's layer count grows to match its own measured
throughput — in one run, from 2 layers (inherited) to 6 layers (rebalanced,
and promoted to the new entry shard).

### Test

```
pytest tests/
```

`tests/test_rebalance.py` asserts `needs_rebalance` flips true after a
recovery and false again after the next `submit()`, that partition
invariants (contiguous, covers all layers) still hold post-rebalance, and
that the standby ends up with strictly more layers than the naive healed
assignment gave it.

## Phase 4 — live dashboard + adding a device

A browser view of the running cluster, plus a real way to onboard a new
device into a session that's already going — not just at startup.

- `mesh/rig.py` — spawns/kills node-daemon subprocesses and hands out
  ports. Not part of the "real" architecture (a real coordinator can't kill
  someone's laptop) — it exists so the dashboard's buttons have a live demo
  process to act on.
- `Cluster` gained an event log (`Cluster.log()`, capped at 200 entries),
  job counters, and `status()` — a JSON-serializable snapshot (model,
  per-node role/liveness/layer-range, standby pool, job/recovery counts,
  recent events) that's the dashboard's entire data source.
- `Cluster.add_device(node, join_as="active"|"standby")` — the "a
  volunteer's laptop joins mid-session" feature. Waits for the node's own
  daemon to be reachable, then either folds it into the active pool and
  flags a rebalance (so the *next* job re-profiles everyone including the
  newcomer — same lazy pattern as Phase 3.6), or holds it in reserve as a
  standby without touching the current partition.
- `mesh/dashboard_server.py` — stdlib-only HTTP server (no new
  dependency): `GET /api/status` for the JSON snapshot, `GET /` serves
  `mesh/dashboard.html`, `POST /api/devices` spawns a new local daemon and
  calls `add_device()` on it, `POST /api/devices/<id>/kill` kills a node's
  process to trigger the fault-tolerance path live.
- `mesh/dashboard.html` — single self-contained page (no CDN, no
  frameworks), polling `/api/status` every second: an animated pipeline
  view (node cards, flowing-dot connectors, a throughput-proportional
  layer-distribution bar), live stat tiles, an "Add a device" form with a
  copy-pasteable real-world CLI snippet, per-node kill buttons, and a
  color-coded, auto-scrolling event log. Colors follow the dataviz skill's
  palette: status colors (good/critical) for node liveness, a fixed
  8-slot categorical palette (hashed per `node_id`) for identity, both
  light- and dark-mode selected (not just an OS-setting flip).

### Run it

```
python3 scripts/run_dashboard_demo.py
```

Starts 4 primary + 2 standby daemons, opens `http://127.0.0.1:8080` in your
browser, and keeps submitting a real job every 2.5s in the background so
the dashboard stays alive with real activity — not canned data. From the
page you can add a brand-new device (spawns a fresh local daemon and
onboards it live, triggering a real rebalance on the next job) or kill an
existing one (exercises the exact same recovery + rebalance path as the
Phase 3 demos, just triggered from a button instead of a script). Ctrl+C
to stop; it tears down every daemon it spawned.

Note: on a single dev machine, 6+ daemons plus the job loop all compete for
one CPU. `heartbeat_interval`/`miss_threshold` are looser here than in the
fault-tolerance tests, and daemon gRPC servers use more worker threads
(`mesh/daemon.py`'s `serve()`), so a busy `Forward` call doesn't starve a
concurrent `Heartbeat` and read as a false node death. Real, separate
devices wouldn't need this slack.

### Test

```
pytest tests/
```

`tests/test_cluster_status.py` covers `status()`'s shape and both
`add_device()` join modes. `tests/test_dashboard_server.py` drives the
actual HTTP endpoints (`urllib`, no new dependency) against a real running
cluster: adding a device, killing one, and the error paths (missing
`node_id`, duplicate device, unknown node).

## Phase 5 — real devices, choosing a model, and asking it something

Three things Phase 4 didn't have: a real (not just simulated) way to add
someone else's machine, a choice of which model to run, and an actual way
to query the model instead of just checking next-token logits. The
dashboard was also restructured around these — one long page stopped being
the right shape once there was a device-management flow *and* a chat-style
feature to fit in.

### Adding a real device (a friend's laptop)

`mesh/dashboard_server.py`'s `POST /api/devices` now takes a `mode`:
- `"spawn"` (default) — what Phase 4 already did: spawns a local daemon
  process as a stand-in for hardware you don't have handy.
- `"connect"` — onboards a daemon **already running somewhere else**, by
  address alone. Nothing is spawned; `Cluster.add_device()` just connects
  to it. This is the real path: your friend runs
  `python -m mesh.daemon --node-id friends-laptop --address 0.0.0.0:PORT --model <same model>`
  on their own machine (same WiFi/LAN — a firewall or a different VPN can
  block the port), finds their LAN IP
  (`ipconfig getifaddr en0` / `hostname -I` / `ipconfig`), and you set
  `Address` in the dashboard's "Connect real device" form to
  `<their-lan-ip>:PORT`.
- Verified for real in this environment the only way two genuinely
  separate processes can be verified here: a daemon started completely
  independently of the dashboard's own process-spawning code (`mesh/rig.py`),
  onboarded purely by address through the "connect" form, immediately
  picked up real jobs after the next rebalance. True cross-machine
  networking (two physical computers) isn't something this sandboxed
  environment can exercise, but the code path — gRPC over a real
  `host:port`, no shared process, no shared filesystem — is identical to
  what runs when the second machine is actually a second machine.

**Model-mismatch guard:** `HeartbeatResponse` now carries the daemon's
configured `--model`. `add_device()` checks it against the cluster's model
before onboarding — mixing shards from two different checkpoints wouldn't
error, it'd just silently produce garbage output, so this is checked up
front (`ValueError`, surfaced in the dashboard's event log) rather than
discovered from bad generations later.

### Choosing a model

Anything in the GPT-2 family works — `gpt2`, `gpt2-medium`, `gpt2-large`,
`distilgpt2`, or any compatible checkpoint on the HF Hub:

```
python3 scripts/run_dashboard_demo.py --model gpt2-medium
```

`mesh/coordinator.py`'s new `model_layer_count(model_name)` reads the
model's config (not its weights) to size the partition, so nothing is
hardcoded to GPT-2-small's 12 layers anymore. Swapping models is a
per-session choice made at startup — there's no live "change model" button,
since every daemon in the pipeline holds weights for one specific model and
swapping would mean tearing down and rebuilding the whole cluster, not
patching a running one.

### Asking the model something

`Cluster.generate()` / `Cluster.generate_stream()` do real autoregressive
decoding — greedy (`temperature=0`, deterministic) or sampled
(`temperature>0`) — yielding each decoded piece as it's produced. Every new
token costs one full pipeline pass (no cross-request KV-cache — `ModelShard`
re-runs the whole sequence-so-far from scratch each step), so this is slow
relative to a real LLM chat UI; that's an honest v1 trade-off, not a bug.
`GET /api/generate/stream` streams it to the browser over Server-Sent
Events.

### The redesigned dashboard

Still one self-contained HTML file, no CDN, no frameworks — but now four
tabs instead of one long scroll:
- **Overview** — stat tiles, an SVG pipeline topology diagram (nodes as
  animated circles, flowing dots along the connectors, click a node for a
  detail panel — the modal links straight to the Devices tab, not a
  duplicate kill button), and a latency sparkline (line + filled area,
  recovered jobs marked as distinct points) built from a new
  `Cluster.status()["latency_history"]` field.
- **Devices** — the node grid, standby pool, and the add-device form with
  its "Simulate here" / "Connect real device" toggle (the CLI snippet
  updates live to match whichever mode is selected, with the cluster's
  actual model name interpolated in).
- **Playground** — the "ask the model" feature: a prompt box, max-tokens
  and temperature controls, a streamed response with a blinking cursor, and
  a running history of past exchanges for the session.
- **Activity** — the event log, with filter chips (jobs / recoveries /
  rebalance / devices / failures / playground) — moved off the main view
  so it's there when you want it instead of dominating the page.

A small always-visible ticker under the header shows the latest event
regardless of which tab is open.

### Run it

```
python3 scripts/run_dashboard_demo.py [--model gpt2] [--port 8080] [--prompt "..."]
```

### Test

```
pytest tests/
```

`tests/test_generation.py` covers `generate()`/`generate_stream()`
(produces text, greedy decoding is deterministic, streamed pieces match the
blocking call, output never exceeds `max_new_tokens`) and the model-mismatch
rejection (using a daemon started with a nonexistent model name — the
mismatch check only needs the *configured* name from `--model`, never
actually loads it, so this doesn't download anything extra).
`tests/test_dashboard_server.py` gained coverage for `mode="connect"`
(onboarding a daemon spawned completely outside the dashboard's own `Rig`)
and the SSE streaming endpoint.

## What's next (not built yet)

Per the original build order: run on real separate hardware (not just
localhost — architecture doesn't change, just point `--address` at a LAN
IP for a daemon, or a real IP into `add_device()`), then a closed campus
beta. Also pending: sandboxed node daemons, signed/checksummed weight
distribution, latency-aware scheduling (preferring low-latency hops when
chaining nodes — meaningful once nodes are on real, distinct networks
instead of all being localhost), and a KV-cache across generation steps so
the playground doesn't re-run the whole prompt on every token.
