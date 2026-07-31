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
"""

import threading
import uuid
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
    ):
        self.assignments = list(assignments)
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
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

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
                if status.consecutive_misses >= self.miss_threshold:
                    status.alive = False

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

        profiles = benchmark_nodes(active_nodes)
        new_assignments = plan_partition(self._num_layers, profiles)
        load_shards(active_nodes, new_assignments)

        with self._lock:
            self.assignments = new_assignments
            self._needs_rebalance = False

    def submit(self, input_ids: torch.Tensor, job_id: str | None = None) -> JobResult:
        job_id = job_id or str(uuid.uuid4())
        with self._lock:
            needs_rebalance = self._needs_rebalance
        if needs_rebalance:
            self.rebalance()

        with self._lock:
            entry_node = self._nodes_by_id[self.assignments[0].node_id]
        try:
            return submit_job(entry_node, job_id, input_ids)
        except grpc.RpcError:
            return self._recover_and_heal(job_id, input_ids)

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

        return result
