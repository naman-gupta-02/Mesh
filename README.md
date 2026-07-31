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

## What's next (not built yet)

See the full implementation plan for the remaining build order: LAN mesh
with real devices and network hops, fault injection (kill nodes mid-job,
tune checkpoint/reassignment recovery), then a closed campus beta. Also
pending: real networking (gRPC/WebSocket) in place of `multiprocessing`
queues, heartbeats + node failure handling, sandboxed node daemons, and
latency-aware scheduling.
