"""Simulated volunteer device: owns one shard, holds it warm, executes
forward passes on activations handed to it by the previous pipeline stage.

Real nodes will talk over gRPC/WebSocket (see the architecture plan); Phase
1 fakes the network with multiprocessing.Queue so we can validate the
sharding + reassembly logic without any of that machinery.
"""

import multiprocessing as mp
import time

import torch

from mesh.model_shard import ModelShard


class NodeProcess(mp.Process):
    def __init__(
        self,
        node_id: str,
        shard: ModelShard,
        in_queue: "mp.Queue",
        out_queue: "mp.Queue",
        telemetry_queue: "mp.Queue",
        artificial_delay: float = 0.0,
    ):
        super().__init__(name=f"node-{node_id}")
        self.node_id = node_id
        self.shard = shard
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.telemetry_queue = telemetry_queue
        self.artificial_delay = artificial_delay

    def run(self) -> None:
        # Simulated nodes share one physical CPU; without this they'd fight
        # each other for BLAS threads and the timing numbers would be noise.
        torch.set_num_threads(1)

        while True:
            payload = self.in_queue.get()
            if payload is None:  # sentinel: shut down
                break

            start = time.perf_counter()
            if self.shard.include_embed:
                output = self.shard(hidden_states=None, input_ids=payload)
            else:
                output = self.shard(hidden_states=payload)

            if self.artificial_delay > 0:
                time.sleep(self.artificial_delay)
            elapsed = time.perf_counter() - start

            # Data channel: just the tensor, handed to the next stage (or
            # the coordinator, for the last stage).
            self.out_queue.put(output)
            # Control channel: timing telemetry, separate from the data
            # hand-off — mirrors the real design's split between a
            # gRPC shard-transfer path and a WebSocket control channel.
            self.telemetry_queue.put((self.node_id, elapsed))
