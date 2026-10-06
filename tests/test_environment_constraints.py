import networkx as nx
import numpy as np

from environment.graph_env import GraphEnv, MultiRobotEnv
from evaluation.common_protocol import (
    add_dynamic_disruptions,
    evaluate_single_policy,
    generate_protocol,
    summarize,
)


def graph_and_embeddings():
    graph = nx.Graph()
    graph.add_node(0, risk=0.0, capacity=1)
    graph.add_node(1, risk=0.25, capacity=1)
    graph.add_node(2, risk=0.75, capacity=2)
    graph.add_edge(0, 1, weight=2.0, passable=True, width=0.9, type="door")
    graph.add_edge(1, 2, weight=3.0, passable=True, width=1.2, type="door")
    embeddings = {node: np.full(4, node, dtype=np.float32) for node in graph.nodes}
    return graph, embeddings


def test_closed_and_narrow_edges_are_masked():
    graph, embeddings = graph_and_embeddings()
    env = GraphEnv(graph, embeddings, reward_cfg={"min_door_width": 1.0})
    obs, _ = env.reset(start=0, target=2)
    assert obs["mask"].sum() == 0

    env = GraphEnv(graph, embeddings)
    obs, _ = env.reset(start=0, target=2, options={"closed_edges": [(0, 1)]})
    assert obs["mask"].sum() == 0


def test_static_risk_and_path_length_are_accumulated():
    graph, embeddings = graph_and_embeddings()
    env = GraphEnv(graph, embeddings, reward_cfg={"risk_factor": 1.0})
    obs, _ = env.reset(start=0, target=2)
    _, _, _, _, info = env.step(0)
    assert info["cumulative_risk"] == 0.25
    assert info["path_length"] == 2.0


def test_multi_robot_observation_contains_occupancy():
    graph, embeddings = graph_and_embeddings()
    env = MultiRobotEnv(graph, embeddings, n_robots=2, seed=42)
    observations = env.reset(starts=[0, 1], targets=[2, 0])
    assert all("occupancy" in obs for obs in observations)


def test_protocol_is_reproducible_and_reports_ci95():
    graph, _ = graph_and_embeddings()
    first = generate_protocol(graph, seed=7, n_single=3, n_multi=2, robot_counts=(1,))
    second = generate_protocol(graph, seed=7, n_single=3, n_multi=2, robot_counts=(1,))
    assert first == second
    report = summarize([
        {"success": 1, "reward": 1, "steps": 2, "spl": 1, "collisions": 0,
         "deadlock": 0, "cumulative_risk": 0.25, "detour": 1},
        {"success": 0, "reward": -1, "steps": 4, "spl": 0, "collisions": 1,
         "deadlock": 1, "cumulative_risk": 0.75, "detour": 2},
    ])
    assert report["success_mean"] == 0.5
    assert report["success_ci95"] > 0


def test_dynamic_protocol_closes_a_non_bridge_edge_and_applies_it():
    graph = nx.cycle_graph(4)
    for node in graph.nodes:
        graph.nodes[node].update(risk=0.0, capacity=1)
    for a, b in graph.edges:
        graph.edges[a, b].update(
            weight=1.0, passable=True, width=1.0, type="door"
        )
    embeddings = {
        node: np.full(4, node, dtype=np.float32) for node in graph.nodes
    }
    static = {
        "seed": 7,
        "regime": "static",
        "single": [{
            "scenario_id": 0,
            "start": 0,
            "target": 2,
            "closed_edges": [],
            "blocked_nodes": [],
        }],
        "multi": [],
    }
    dynamic = add_dynamic_disruptions(graph, static, seed=8)
    scenario = dynamic["single"][0]
    assert scenario["dynamic_closed_edges"]
    available = graph.copy()
    available.remove_edges_from(scenario["dynamic_closed_edges"])
    assert nx.has_path(available, 0, 2)

    env = GraphEnv(graph, embeddings, max_steps=10)
    rows = evaluate_single_policy(
        env,
        lambda obs: int(np.where(obs["mask"])[0][0]),
        dynamic["single"],
    )
    assert rows[0]["disruption_applied"] == 1.0
