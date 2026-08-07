"""Stateful cluster wrapper: durable liveness tracking + assignment healing
across multiple jobs, built on top of net_coordinator's stateless per-call
RPC helpers.

recover_job() (see mesh/net_coordinator.py) fixes a single in-flight job
but leaves the pipeline's actual wiring untouched -- the node immediately
upstream of a dead one is still pointed at its now-dead address, so the
very next job would hit the same failure and need recovery all over again.
Cluster closes that loop: after a recovery it also updates the assignment
table and rewires the surviving upstream node, so subsequent jobs route
through the standby directly, and it runs a background heartbeat loop to
track node liveness independent of job submission.

Also tracks enough state (event log, job counters, per-node role) for
mesh/dashboard_server.py to expose a live view of what the cluster is
doing, and supports onboarding a brand new device mid-session via
add_device() -- the "volunteer's laptop joins partway through" case.
"""

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

import grpc
import torch

from mesh.coordinator import ShardAssignment, plan_partition
from mesh.net_coordinator import (
    JobResult,
    NodeHandle,
    benchmark_nodes,
    find_last_checkpoint,
    load_shards,
    recover_job,
    replace_node_in_assignments,
    rewire_next_hop,
    submit_job,
)
from mesh.proto import mesh_pb2, mesh_pb2_grpc


@dataclass
class _NodeStatus:
    alive: bool = True
    consecutive_misses: int = 0


class Cluster:
    def __init__(
        self,
        primary_nodes: list[NodeHandle],
        assignments: list[ShardAssignment],
        standby_nodes: list[NodeHandle],
        heartbeat_interval: float = 2.0,
        miss_threshold: int = 2,
        model_name: str = "gpt2",
    ):
        self.assignments = list(assignments)
        self.model_name = model_name
        self._nodes_by_id: dict[str, NodeHandle] = {n.node_id: n for n in primary_nodes}
        self._standby_pool: list[NodeHandle] = list(standby_nodes)
        self.heartbeat_interval = heartbeat_interval
        self.miss_threshold = miss_threshold
        self._status: dict[str, _NodeStatus] = {n.node_id: _NodeStatus() for n in primary_nodes}
        self._num_layers = max(a.layer_end for a in self.assignments)
        # Set after a recovery: the standby that took over a dead node's
        # shard inherited its *old* layer range, which may not suit the
        # standby's actual throughput. True as soon as the topology
        # changes, cleared once a full re-plan has run.
        self._needs_rebalance = False
        self._jobs_submitted = 0
        self._jobs_recovered = 0
        self._last_job_ms: float | None = None
        # (timestamp, message) pairs, newest last. deque.append is atomic
        # under the GIL, so this is safe to touch without self._lock.
        self._events: deque[tuple[float, str]] = deque(maxlen=200)
        # (timestamp, elapsed_ms, recovered) per completed job -- the
        # dashboard's latency sparkline reads this straight from status()
        # instead of regex-scraping the event log text.
        self._latency_history: deque[tuple[float, float, bool]] = deque(maxlen=50)
        self._tokenizer = None  # lazily loaded on first generate() call
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        self._log(f"cluster started: {len(primary_nodes)} primary, {len(standby_nodes)} standby, model={model_name}")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _log(self, message: str) -> None:
        self._events.append((time.time(), message))

    def log(self, message: str) -> None:
        """Public hook for callers (dashboard server, demo scripts) to add
        their own annotations to the event feed -- e.g. "kill requested via
        dashboard" -- alongside the cluster's own internal events.
        """
        self._log(message)

    @property
    def needs_rebalance(self) -> bool:
        with self._lock:
            return self._needs_rebalance

    def is_alive(self, node_id: str) -> bool:
        with self._lock:
            status = self._status.get(node_id)
            return status.alive if status else False

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            with self._lock:
                node_ids = list(self._status.keys())
            for node_id in node_ids:
                self._check_one(node_id)

    def _check_one(self, node_id: str) -> None:
        with self._lock:
            node = self._nodes_by_id.get(node_id)
            status = self._status.get(node_id)
        if node is None or status is None or not status.alive:
            return
        try:
            channel = grpc.insecure_channel(node.address)
            mesh_pb2_grpc.NodeDaemonStub(channel).Heartbeat(
                mesh_pb2.HeartbeatRequest(), timeout=self.heartbeat_interval
            )
            with self._lock:
                status.consecutive_misses = 0
        except grpc.RpcError:
            with self._lock:
                status.consecutive_misses += 1
                went_dead = status.consecutive_misses >= self.miss_threshold and status.alive
                if went_dead:
                    status.alive = False
            if went_dead:
                self._log(f"heartbeat lost: {node_id} marked dead")

    def inject_delay(self, node_id: str, delay_seconds: float) -> None:
        """Testing hook: makes `node_id` sleep before computing its next
        Forward call, without touching its shard assignment or wiring. Lets
        a fault-injection harness create a deterministic window to kill it
        mid-job.
        """
        with self._lock:
            idx = next(i for i, a in enumerate(self.assignments) if a.node_id == node_id)
            assignment = self.assignments[idx]
            next_hop_address = (
                self._nodes_by_id[self.assignments[idx + 1].node_id].address
                if idx + 1 < len(self.assignments)
                else ""
            )
            node = self._nodes_by_id[node_id]

        rewire_next_hop(
            node,
            assignment,
            include_embed=(idx == 0),
            include_head=(idx == len(self.assignments) - 1),
            next_hop_address=next_hop_address,
            artificial_delay_seconds=delay_seconds,
        )

    def add_device(self, node: NodeHandle, join_as: str = "active", timeout: float = 30.0) -> None:
        """Onboards a new node into a running cluster -- the "new volunteer
        laptop joins mid-session" case from the plan. The node's own daemon
        must already be running and reachable at node.address (see
        mesh/daemon.py) before calling this.

        join_as="active" (default): folds the node into the active pool and
        flags a rebalance, so the next submit() re-profiles everyone
        including the newcomer and re-partitions across all of them --
        "you don't need to reshard immediately, but the next job
        submission should re-profile and re-plan."
        join_as="standby": holds it in reserve for future recover_job()
        calls instead, without touching the current partition.

        Raises ValueError if the node loaded a different base model than
        this cluster -- mixing shards from two different models wouldn't
        error loudly, it'd just silently produce garbage output, so this is
        checked up front rather than discovered from bad generations later.
        """
        if join_as not in ("active", "standby"):
            raise ValueError(f"join_as must be 'active' or 'standby', got {join_as!r}")

        self._log(f"device joining: {node.node_id} ({join_as}) at {node.address}")
        channel = grpc.insecure_channel(node.address)
        grpc.channel_ready_future(channel).result(timeout=timeout)
        heartbeat = mesh_pb2_grpc.NodeDaemonStub(channel).Heartbeat(mesh_pb2.HeartbeatRequest(), timeout=timeout)

        if heartbeat.model_name and heartbeat.model_name != self.model_name:
            self._log(
                f"device rejected: {node.node_id} loaded {heartbeat.model_name!r}, "
                f"cluster expects {self.model_name!r}"
            )
            raise ValueError(
                f"{node.node_id} loaded model {heartbeat.model_name!r} but this cluster runs "
                f"{self.model_name!r} -- restart its daemon with --model {self.model_name}"
            )

        with self._lock:
            if join_as == "standby":
                self._standby_pool.append(node)
            else:
                self._nodes_by_id[node.node_id] = node
                self._status[node.node_id] = _NodeStatus()
                self._needs_rebalance = True
        self._log(f"device joined: {node.node_id} ({join_as})")

    def rebalance(self) -> None:
        """Re-profiles every currently active node and recomputes the
        partition from scratch, rather than leaving a standby stuck with
        whatever layer range the node it replaced happened to hold.

        Deliberately not called during recovery itself -- recovery stays
        fast and correctness-first (matching the plan's stated v1
        priority); rebalancing is heavier (a fresh benchmark + LoadShard
        round trip to every active node) and can afford to happen lazily,
        on the next job submission after a topology change.
        """
        with self._lock:
            active_nodes = [self._nodes_by_id[nid] for nid, status in self._status.items() if status.alive]
        if not active_nodes:
            raise RuntimeError("no active nodes to rebalance across")

        self._log(f"rebalancing: re-profiling {len(active_nodes)} active nodes")
        profiles = benchmark_nodes(active_nodes)
        new_assignments = plan_partition(self._num_layers, profiles)
        load_shards(active_nodes, new_assignments)

        with self._lock:
            self.assignments = new_assignments
            self._needs_rebalance = False
        summary = ", ".join(f"{a.node_id}={a.num_layers}L" for a in new_assignments)
        self._log(f"rebalanced: {summary}")

    def submit(self, input_ids: torch.Tensor, job_id: str | None = None) -> JobResult:
        job_id = job_id or str(uuid.uuid4())
        with self._lock:
            needs_rebalance = self._needs_rebalance
        if needs_rebalance:
            self.rebalance()

        with self._lock:
            entry_node = self._nodes_by_id[self.assignments[0].node_id]
        start = time.perf_counter()
        try:
            result = submit_job(entry_node, job_id, input_ids)
        except grpc.RpcError:
            self._log(f"job {job_id[:8]} failed mid-flight, recovering...")
            result = self._recover_and_heal(job_id, input_ids)
            elapsed_ms = (time.perf_counter() - start) * 1000
            with self._lock:
                self._jobs_submitted += 1
                self._jobs_recovered += 1
                self._last_job_ms = elapsed_ms
                self._latency_history.append((time.time(), elapsed_ms, True))
            self._log(f"job {job_id[:8]} recovered in {elapsed_ms:.0f}ms")
            return result

        elapsed_ms = (time.perf_counter() - start) * 1000
        with self._lock:
            self._jobs_submitted += 1
            self._last_job_ms = elapsed_ms
            self._latency_history.append((time.time(), elapsed_ms, False))
        self._log(f"job {job_id[:8]} completed in {elapsed_ms:.0f}ms")
        return result

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import GPT2Tokenizer

            self._tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def generate_stream(self, prompt: str, max_new_tokens: int = 40, temperature: float = 0.0):
        """Actually "asks the model something": greedy (temperature=0) or
        sampled (temperature>0) autoregressive decoding, yielding each
        decoded text piece as it's produced.

        Every new token costs one full pipeline pass -- submit() re-runs
        the whole sequence-so-far through every shard from scratch, since
        ModelShard has no cross-request KV-cache. That's fine for a short
        demo prompt; it's the honest cost of this being v1 (a real
        optimization would cache attention keys/values across steps, the
        same way HF's `generate()` does internally).
        """
        tokenizer = self._get_tokenizer()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        eos_token_id = tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            result = self.submit(input_ids)
            next_token_logits = result.logits[0, -1]

            if temperature and temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = next_token_logits.argmax().item()

            if next_token == eos_token_id:
                break

            piece = tokenizer.decode([next_token])
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            yield piece

    def generate(self, prompt: str, max_new_tokens: int = 40, temperature: float = 0.0) -> str:
        """Blocking convenience wrapper around generate_stream() for callers
        that just want the final text (see mesh/dashboard_server.py's
        streaming SSE endpoint for the token-by-token version).
        """
        return "".join(self.generate_stream(prompt, max_new_tokens=max_new_tokens, temperature=temperature))

    def _recover_and_heal(self, job_id: str, input_ids: torch.Tensor) -> JobResult:
        with self._lock:
            if not self._standby_pool:
                raise RuntimeError("no standby nodes available for recovery")
            standby = self._standby_pool.pop(0)
            assignments = self.assignments
            all_known = list(self._nodes_by_id.values()) + [standby]

        last_index, _ = find_last_checkpoint(all_known, assignments, job_id)
        failed_index = last_index + 1
        if failed_index >= len(assignments):
            raise RuntimeError(f"job {job_id}: submit_job failed but the chain looks complete")

        failed_node_id = assignments[failed_index].node_id
        self._log(f"recovering: reassigning {failed_node_id}'s shard to {standby.node_id}")
        result = recover_job(all_known, assignments, standby, job_id, input_ids)

        with self._lock:
            self.assignments = replace_node_in_assignments(self.assignments, failed_index, standby.node_id)
            self._nodes_by_id[standby.node_id] = standby
            self._status[standby.node_id] = _NodeStatus()
            if failed_node_id in self._status:
                self._status[failed_node_id].alive = False
            self._needs_rebalance = True
            healed_assignments = self.assignments
            upstream_node = (
                self._nodes_by_id[healed_assignments[failed_index - 1].node_id] if failed_index > 0 else None
            )

        if upstream_node is not None:
            rewire_next_hop(
                upstream_node,
                healed_assignments[failed_index - 1],
                include_embed=(failed_index - 1 == 0),
                include_head=(failed_index - 1 == len(healed_assignments) - 1),
                next_hop_address=standby.address,
            )

        self._log(f"recovered: {standby.node_id} now serving {failed_node_id}'s former shard")
        return result

    def status(self) -> dict:
        """A JSON-serializable snapshot for mesh/dashboard_server.py."""
        with self._lock:
            entry_id = self.assignments[0].node_id if self.assignments else None
            exit_id = self.assignments[-1].node_id if self.assignments else None
            assignment_by_node = {a.node_id: a for a in self.assignments}

            nodes = []
            for node_id, node in self._nodes_by_id.items():
                node_status = self._status[node_id]
                assignment = assignment_by_node.get(node_id)
                role = "idle"
                if assignment is not None:
                    role = "entry" if node_id == entry_id else ("exit" if node_id == exit_id else "primary")
                nodes.append(
                    {
                        "node_id": node_id,
                        "address": node.address,
                        "alive": node_status.alive,
                        "role": role,
                        "layer_start": assignment.layer_start if assignment else None,
                        "layer_end": assignment.layer_end if assignment else None,
                        "num_layers": assignment.num_layers if assignment else 0,
                    }
                )
            # Stable pipeline order for the frontend, dead nodes trail at the end.
            order = {node_id: i for i, node_id in enumerate(a.node_id for a in self.assignments)}
            nodes.sort(key=lambda n: order.get(n["node_id"], len(order)))

            standby_ids = [n.node_id for n in self._standby_pool]
            events = [{"time": t, "message": m} for t, m in list(self._events)[-60:]]
            latency_history = [
                {"time": t, "elapsed_ms": ms, "recovered": recovered}
                for t, ms, recovered in self._latency_history
            ]

            return {
                "model_name": self.model_name,
                "num_layers": self._num_layers,
                "nodes": nodes,
                "standby_ids": standby_ids,
                "needs_rebalance": self._needs_rebalance,
                "jobs_submitted": self._jobs_submitted,
                "jobs_recovered": self._jobs_recovered,
                "last_job_ms": self._last_job_ms,
                "latency_history": latency_history,
                "events": events,
            }
