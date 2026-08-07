"""gRPC node daemon: the real-network counterpart to mesh/node.py.

Each daemon holds one shard warm and exposes it over gRPC instead of a
multiprocessing.Queue, so traffic between "nodes" goes through the OS
TCP/IP stack and protobuf serialization -- the same path it takes between
two real machines on a dorm network, just both ends on localhost for now.
"""

import argparse
import time
from concurrent import futures

import grpc

from mesh.model_shard import ModelShard
from mesh.profiler import measure_throughput
from mesh.proto import mesh_pb2, mesh_pb2_grpc
from mesh.tensor_codec import decode, encode


class NodeDaemonServicer(mesh_pb2_grpc.NodeDaemonServicer):
    def __init__(self, node_id: str, model_name: str = "gpt2", simulated_scale: float = 1.0):
        self.node_id = node_id
        self.model_name = model_name
        self.simulated_scale = simulated_scale
        self._model = None  # lazily loaded; every shard slices from these weights
        self.shard: ModelShard | None = None
        self._next_hop_stub: mesh_pb2_grpc.NodeDaemonStub | None = None
        self._artificial_delay = 0.0
        # job_id -> this node's own encoded output. Lets the coordinator
        # resume a broken chain from the last node that finished (see
        # GetCheckpoint) instead of restarting the whole job.
        self._checkpoints: dict[str, bytes] = {}

    def _load_model(self):
        if self._model is None:
            from transformers import GPT2LMHeadModel

            self._model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._model

    def Benchmark(self, request, context):
        throughput = measure_throughput(simulated_scale=self.simulated_scale)
        return mesh_pb2.BenchmarkResponse(node_id=self.node_id, throughput=throughput)

    def LoadShard(self, request, context):
        try:
            model = self._load_model()
            spec = request.shard
            self.shard = ModelShard(
                model,
                layer_start=spec.layer_start,
                layer_end=spec.layer_end,
                include_embed=spec.include_embed,
                include_head=spec.include_head,
            )
            if request.next_hop_address:
                channel = grpc.insecure_channel(request.next_hop_address)
                self._next_hop_stub = mesh_pb2_grpc.NodeDaemonStub(channel)
            else:
                self._next_hop_stub = None
            self._artificial_delay = request.artificial_delay_seconds
            return mesh_pb2.LoadShardResponse(ok=True)
        except Exception as exc:
            return mesh_pb2.LoadShardResponse(ok=False, error=str(exc))

    def Forward(self, request, context):
        if self.shard is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"{self.node_id}: no shard loaded")

        if self._artificial_delay > 0:
            # Sleep before computing (not after), so a fault-injection
            # harness has a deterministic window to kill this process
            # mid-request and see the coordinator's recovery path exercised.
            time.sleep(self._artificial_delay)

        tensor = decode(request.tensor)
        start = time.perf_counter()
        if self.shard.include_embed:
            output = self.shard(hidden_states=None, input_ids=tensor)
        else:
            output = self.shard(hidden_states=tensor)
        elapsed = time.perf_counter() - start
        my_timing = mesh_pb2.StageTiming(node_id=self.node_id, elapsed_seconds=elapsed)

        encoded_output = encode(output)
        self._checkpoints[request.job_id] = encoded_output

        if self._next_hop_stub is not None:
            downstream = self._next_hop_stub.Forward(
                mesh_pb2.ForwardRequest(job_id=request.job_id, tensor=encoded_output)
            )
            return mesh_pb2.ForwardResponse(
                job_id=request.job_id,
                logits=downstream.logits,
                timings=[my_timing, *downstream.timings],
            )

        return mesh_pb2.ForwardResponse(
            job_id=request.job_id,
            logits=encoded_output,
            timings=[my_timing],
        )

    def Heartbeat(self, request, context):
        return mesh_pb2.HeartbeatResponse(node_id=self.node_id, alive=True, model_name=self.model_name)

    def GetCheckpoint(self, request, context):
        checkpoint = self._checkpoints.get(request.job_id)
        if checkpoint is None:
            return mesh_pb2.GetCheckpointResponse(available=False)
        return mesh_pb2.GetCheckpointResponse(available=True, tensor=checkpoint)


def serve(node_id: str, address: str, model_name: str = "gpt2", simulated_scale: float = 1.0) -> grpc.Server:
    # A busy Forward call shouldn't starve Heartbeat/GetCheckpoint of a
    # worker thread -- on a loaded single dev machine (many daemons sharing
    # one CPU) a too-small pool here reads as a false-positive node death.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    mesh_pb2_grpc.add_NodeDaemonServicer_to_server(
        NodeDaemonServicer(node_id, model_name=model_name, simulated_scale=simulated_scale), server
    )
    server.add_insecure_port(address)
    server.start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Mesh node daemon.")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--address", required=True, help="host:port to listen on")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--simulated-scale", type=float, default=1.0)
    args = parser.parse_args()

    server = serve(args.node_id, args.address, model_name=args.model, simulated_scale=args.simulated_scale)
    print(f"[{args.node_id}] listening on {args.address}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
