"""Phase 1 demo: single-machine simulation of Mesh's layer-sharding pipeline.

Splits a real GPT-2 model across simulated heterogeneous "nodes" (fake local
processes, each declaring a different throughput profile), runs a real
prompt through the distributed pipeline, and checks the result against the
monolithic model to prove the sharding + reassembly logic is correct before
any real networking gets involved.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from mesh.coordinator import NodeProfile, plan_partition
from mesh.pipeline import run_pipeline
from mesh.profiler import measure_throughput

# Stand-ins for real volunteer devices. simulated_scale fakes the
# hardware heterogeneity a single dev machine can't otherwise produce
# (see mesh/profiler.py).
SIMULATED_DEVICES = [
    ("thinkpad-cpu", 0.5),
    ("old-macbook", 0.8),
    ("m1-air", 1.4),
    ("gaming-laptop-4060", 2.3),
]

PROMPT = "The best way to distribute a language model across a campus network is"


def main() -> None:
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    num_layers = len(model.transformer.h)

    print(f"\nProfiling {len(SIMULATED_DEVICES)} simulated nodes...")
    profiles = []
    for node_id, scale in SIMULATED_DEVICES:
        throughput = measure_throughput(simulated_scale=scale)
        profiles.append(NodeProfile(node_id=node_id, throughput=throughput))
        print(f"  {node_id:<22} {throughput:8.1f} GFLOPS")

    assignments = plan_partition(num_layers, profiles)
    print(f"\nPartition plan ({num_layers} layers total):")
    for a in assignments:
        print(f"  {a.node_id:<22} layers [{a.layer_start:>2}, {a.layer_end:>2})  " f"({a.num_layers} layers)")

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    print("\nRunning distributed pipeline...")
    result = run_pipeline(model, input_ids, assignments)

    print("\nStage timings:")
    for t in result.stage_timings:
        print(f"  {t.node_id:<22} {t.elapsed_seconds * 1000:7.1f} ms")

    print("\nRunning monolithic model for comparison...")
    with torch.no_grad():
        reference_logits = model(input_ids).logits

    max_abs_diff = (result.logits - reference_logits).abs().max().item()
    distributed_next_token = result.logits[0, -1].argmax().item()
    reference_next_token = reference_logits[0, -1].argmax().item()

    print(f"\nMax abs logit diff:     {max_abs_diff:.6e}")
    print(f"Distributed next token: {tokenizer.decode([distributed_next_token])!r}")
    print(f"Reference next token:   {tokenizer.decode([reference_next_token])!r}")

    tolerance = 1e-3
    if max_abs_diff < tolerance and distributed_next_token == reference_next_token:
        print(f"\nPASS: distributed output matches monolithic model (tol={tolerance})")
    else:
        print(f"\nFAIL: distributed output diverges from monolithic model (tol={tolerance})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
