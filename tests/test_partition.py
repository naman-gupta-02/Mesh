import pytest

from mesh.coordinator import NodeProfile, plan_partition


def test_equal_throughput_splits_evenly():
    profiles = [NodeProfile(f"node-{i}", throughput=100.0) for i in range(4)]
    assignments = plan_partition(12, profiles)

    assert [a.num_layers for a in assignments] == [3, 3, 3, 3]


def test_assignments_are_contiguous_and_cover_all_layers():
    profiles = [
        NodeProfile("thinkpad", throughput=50.0),
        NodeProfile("m1-air", throughput=140.0),
        NodeProfile("gaming-laptop", throughput=230.0),
    ]
    assignments = plan_partition(12, profiles)

    assert assignments[0].layer_start == 0
    assert assignments[-1].layer_end == 12
    for prev, cur in zip(assignments, assignments[1:]):
        assert prev.layer_end == cur.layer_start


def test_faster_node_gets_more_layers():
    profiles = [
        NodeProfile("slow", throughput=50.0),
        NodeProfile("fast", throughput=200.0),
    ]
    assignments = plan_partition(10, profiles)
    by_id = {a.node_id: a.num_layers for a in assignments}

    assert by_id["fast"] > by_id["slow"]


def test_every_node_gets_at_least_one_layer():
    # one node's throughput share alone would round to 0 layers here
    profiles = [
        NodeProfile("tiny-share", throughput=1.0),
        NodeProfile("dominant", throughput=1000.0),
    ]
    assignments = plan_partition(12, profiles)

    assert all(a.num_layers >= 1 for a in assignments)
    assert sum(a.num_layers for a in assignments) == 12


def test_rejects_more_nodes_than_layers():
    profiles = [NodeProfile(f"node-{i}", throughput=100.0) for i in range(5)]

    with pytest.raises(ValueError):
        plan_partition(3, profiles)


def test_rejects_empty_profiles():
    with pytest.raises(ValueError):
        plan_partition(12, [])


def test_rejects_non_positive_throughput():
    profiles = [NodeProfile("a", throughput=100.0), NodeProfile("b", throughput=0.0)]

    with pytest.raises(ValueError):
        plan_partition(12, profiles)
