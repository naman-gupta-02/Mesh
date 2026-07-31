"""Distributed pipeline output must match a plain forward pass on the
unsplit model, for any partition plan. Uses a tiny randomly-initialized
GPT-2 (not the pretrained 124M model) so tests run fast with no network
or download dependency.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mesh.coordinator import NodeProfile, plan_partition
from mesh.pipeline import run_pipeline

torch.manual_seed(0)


def make_tiny_model() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=64,
        n_positions=16,
        n_embd=32,
        n_layer=6,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def reference_logits(model: GPT2LMHeadModel, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids).logits


def test_single_node_matches_monolithic():
    model = make_tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    profiles = [NodeProfile("solo", throughput=100.0)]
    assignments = plan_partition(model.config.n_layer, profiles)

    result = run_pipeline(model, input_ids, assignments)
    expected = reference_logits(model, input_ids)

    assert torch.allclose(result.logits, expected, atol=1e-5)


def test_heterogeneous_multi_node_matches_monolithic():
    model = make_tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    profiles = [
        NodeProfile("thinkpad", throughput=50.0),
        NodeProfile("m1-air", throughput=140.0),
        NodeProfile("gaming-laptop", throughput=230.0),
    ]
    assignments = plan_partition(model.config.n_layer, profiles)

    result = run_pipeline(model, input_ids, assignments)
    expected = reference_logits(model, input_ids)

    assert torch.allclose(result.logits, expected, atol=1e-5)
    assert {t.node_id for t in result.stage_timings} == {p.node_id for p in profiles}


def test_uneven_layer_count_partition_matches_monolithic():
    model = make_tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
    # 6 layers across 4 nodes: forces some nodes to get 1 layer, others 2+
    profiles = [
        NodeProfile("tiny-share-a", throughput=10.0),
        NodeProfile("tiny-share-b", throughput=10.0),
        NodeProfile("mid", throughput=100.0),
        NodeProfile("dominant", throughput=400.0),
    ]
    assignments = plan_partition(model.config.n_layer, profiles)

    result = run_pipeline(model, input_ids, assignments)
    expected = reference_logits(model, input_ids)

    assert torch.allclose(result.logits, expected, atol=1e-5)
