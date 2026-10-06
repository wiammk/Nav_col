"""Frozen, learning-free evaluation protocols shared by every algorithm."""

import copy
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np


def _available_graph(graph, blocked_nodes=(), closed_edges=()):
    available = graph.copy()
    available.remove_nodes_from(
        node for node in blocked_nodes if node in available
    )
    available.remove_edges_from(
        (edge[0], edge[1])
        for edge in closed_edges
        if len(edge) == 2 and available.has_edge(edge[0], edge[1])
    )
    available.remove_edges_from([
        (u, v) for u, v, data in available.edges(data=True)
        if not bool(data.get("passable", True))
        or float(data.get("width", 1.0)) < 0.6
    ])
    return available


def _keeps_route_available(graph, start, target, blocked_nodes=(), closed_edges=()):
    available = _available_graph(graph, blocked_nodes, closed_edges)
    return (
        start in available
        and target in available
        and nx.has_path(available, start, target)
    )


def generate_protocol(graph, seed=42, n_single=200, n_multi=200, robot_counts=(3, 5, 10)):
    rng = np.random.default_rng(seed)
    nodes = list(sorted(graph.nodes()))

    def pair():
        start, target = rng.choice(nodes, size=2, replace=False)
        return [start.item() if hasattr(start, "item") else start,
                target.item() if hasattr(target, "item") else target]

    single = []
    door_edges = [
        (a, b) for a, b, data in graph.edges(data=True)
        if data.get("type") == "door"
    ]
    for scenario_id in range(n_single):
        start, target = pair()
        candidates = [node for node in nodes if node not in {start, target}]
        closed_edges = []
        if door_edges and scenario_id % 4 == 0:
            edge = door_edges[int(rng.integers(0, len(door_edges)))]
            proposed = [list(edge)]
            if _keeps_route_available(graph, start, target, closed_edges=proposed):
                closed_edges = proposed
        blocked_nodes = []
        if candidates and scenario_id % 5 == 0:
            blocked = rng.choice(candidates)
            proposed = [blocked.item() if hasattr(blocked, "item") else blocked]
            if _keeps_route_available(
                graph,
                start,
                target,
                blocked_nodes=proposed,
                closed_edges=closed_edges,
            ):
                blocked_nodes = proposed
        single.append({
            "scenario_id": scenario_id,
            "start": start,
            "target": target,
            "closed_edges": closed_edges,
            "blocked_nodes": blocked_nodes,
        })

    multi = []
    for robots in robot_counts:
        for scenario_id in range(n_multi):
            selected = rng.choice(nodes, size=2 * robots, replace=False).tolist()
            multi.append({
                "scenario_id": scenario_id,
                "robots": robots,
                "starts": selected[:robots],
                "targets": selected[robots:],
            })
    return {"seed": seed, "regime": "static", "single": single, "multi": multi}


def _dynamic_edge_for_route(
    graph,
    start,
    target,
    rng,
    blocked_nodes=(),
    closed_edges=(),
):
    """Select a non-bridge edge on the current shortest route.

    Removing a non-bridge edge keeps the available graph connected, so a
    dynamic test measures rerouting rather than an impossible mission.
    """
    available = _available_graph(graph, blocked_nodes, closed_edges)
    if start not in available or target not in available:
        return None, 1
    try:
        path = nx.shortest_path(available, start, target, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, 1
    if len(path) < 2:
        return None, 1

    bridge_keys = {
        frozenset(edge) for edge in nx.bridges(available)
    }
    candidates = [
        (path[index], path[index + 1])
        for index in range(len(path) - 1)
        if frozenset((path[index], path[index + 1])) not in bridge_keys
    ]
    if not candidates:
        return None, 1
    order = rng.permutation(len(candidates))
    edge = candidates[int(order[0])]
    event_step = max(1, min(5, (len(path) - 1) // 2))
    return [edge[0], edge[1]], event_step


def add_dynamic_disruptions(graph, protocol, seed=43):
    """Create a paired dynamic version of an existing frozen protocol.

    Starts, targets and initial constraints remain identical. During each
    episode a non-bridge edge on an initial shortest route is closed. This
    guarantees that an alternative route still exists in the test graph.
    """
    rng = np.random.default_rng(seed)
    dynamic = copy.deepcopy(protocol)
    dynamic["seed"] = seed
    dynamic["paired_static_seed"] = protocol.get("seed")
    dynamic["regime"] = "dynamic_edge_closure"

    for scenario in dynamic.get("single", []):
        edge, event_step = _dynamic_edge_for_route(
            graph,
            scenario["start"],
            scenario["target"],
            rng,
            blocked_nodes=scenario.get("blocked_nodes", []),
            closed_edges=scenario.get("closed_edges", []),
        )
        scenario["dynamic_step"] = event_step
        scenario["dynamic_closed_edges"] = [edge] if edge else []
        scenario["disruption_kind"] = "edge_closure" if edge else "none"

    for scenario in dynamic.get("multi", []):
        chosen_edge = None
        event_step = 1
        robot_order = rng.permutation(scenario["robots"])
        for robot_index in robot_order:
            edge, candidate_step = _dynamic_edge_for_route(
                graph,
                scenario["starts"][int(robot_index)],
                scenario["targets"][int(robot_index)],
                rng,
                blocked_nodes=scenario.get("blocked_nodes", []),
                closed_edges=scenario.get("closed_edges", []),
            )
            if edge:
                chosen_edge = edge
                event_step = candidate_step
                break
        scenario["dynamic_step"] = event_step
        scenario["dynamic_closed_edges"] = [chosen_edge] if chosen_edge else []
        scenario["disruption_kind"] = "edge_closure" if chosen_edge else "none"
    return dynamic


def save_protocol(protocol, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def load_protocol(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_single_policy(env, action_fn, scenarios):
    """Evaluate a frozen policy. ``action_fn`` must never update the model."""
    rows = []
    for scenario in scenarios:
        obs, _ = env.reset(
            start=scenario["start"],
            target=scenario["target"],
            options={
                "closed_edges": scenario.get("closed_edges", []),
                "blocked_nodes": scenario.get("blocked_nodes", []),
            },
        )
        shortest = env._sp_dist(env.current_node, env.target_node)
        reward = 0.0
        collisions = 0
        done = False
        info = {}
        disruption_applied = False
        while not done:
            if (
                not disruption_applied
                and scenario.get("dynamic_closed_edges")
                and env._step_count >= int(scenario.get("dynamic_step", 1))
            ):
                env.set_disruptions(
                    blocked_nodes=[
                        *scenario.get("blocked_nodes", []),
                        *scenario.get("dynamic_blocked_nodes", []),
                    ],
                    closed_edges=[
                        *scenario.get("closed_edges", []),
                        *scenario.get("dynamic_closed_edges", []),
                    ],
                )
                obs = env._build_obs()
                disruption_applied = True
            obs, step_reward, terminated, truncated, info = env.step(action_fn(obs))
            reward += float(step_reward)
            collisions += int(bool(info.get("collision", False)))
            done = terminated or truncated
        success = env.current_node == env.target_node
        path_length = float(env.path_length)
        spl = float(shortest / max(shortest, path_length)) if success and shortest > 0 else float(success)
        rows.append({
            "scenario_id": scenario["scenario_id"],
            "success": float(success),
            "reward": reward,
            "steps": env._step_count,
            "spl": spl,
            "collisions": collisions,
            "deadlock": float(not success and env._stuck_count >= env.rcfg["stuck_limit"]),
            "cumulative_risk": float(env.cumulative_risk),
            "path_length": path_length,
            "shortest_path_length": shortest,
            "detour": path_length / shortest if shortest > 0 else 1.0,
            "disruption_applied": float(disruption_applied),
        })
    return rows


def evaluate_multi_policies(multi_env, action_fns, scenarios):
    """Evaluate simultaneous frozen policies on the exact supplied scenarios."""
    rows = []
    for scenario in scenarios:
        if scenario["robots"] != multi_env.n_robots:
            continue
        base_options = {
            "closed_edges": scenario.get("closed_edges", []),
            "blocked_nodes": scenario.get("blocked_nodes", []),
        }
        observations = multi_env.reset(
            scenario["starts"],
            scenario["targets"],
            options=[dict(base_options) for _ in range(multi_env.n_robots)],
        )
        dones = [False] * multi_env.n_robots
        totals = [0.0] * multi_env.n_robots
        collisions = [0] * multi_env.n_robots
        no_progress = [0] * multi_env.n_robots
        previous = [env.current_node for env in multi_env.envs]
        disruption_applied = False
        elapsed_steps = 0
        while not all(dones):
            if (
                not disruption_applied
                and scenario.get("dynamic_closed_edges")
                and elapsed_steps >= int(scenario.get("dynamic_step", 1))
            ):
                for env in multi_env.envs:
                    env.set_disruptions(
                        blocked_nodes=[
                            *scenario.get("blocked_nodes", []),
                            *scenario.get("dynamic_blocked_nodes", []),
                        ],
                        closed_edges=[
                            *scenario.get("closed_edges", []),
                            *scenario.get("dynamic_closed_edges", []),
                        ],
                    )
                observations = multi_env._mask_occupied_neighbors([
                    env._build_obs() for env in multi_env.envs
                ])
                disruption_applied = True
            actions = [
                0 if dones[index] else action_fns[index](observations[index])
                for index in range(multi_env.n_robots)
            ]
            observations, rewards, step_dones, infos = multi_env.step(actions)
            for index, env in enumerate(multi_env.envs):
                if dones[index]:
                    continue
                totals[index] += float(rewards[index])
                collisions[index] += int(bool(infos[index].get("collision", False)))
                no_progress[index] = no_progress[index] + 1 if env.current_node == previous[index] else 0
                previous[index] = env.current_node
                dones[index] = bool(step_dones[index])
            elapsed_steps += 1

        for index, env in enumerate(multi_env.envs):
            shortest = env._initial_dist
            success = env.current_node == env.target_node
            path_length = float(env.path_length)
            rows.append({
                "scenario_id": scenario["scenario_id"],
                "robot_id": index,
                "robots": multi_env.n_robots,
                "success": float(success),
                "reward": totals[index],
                "steps": env._step_count,
                "spl": shortest / max(shortest, path_length) if success and shortest > 0 else float(success),
                "collisions": collisions[index],
                "deadlock": float(not success and no_progress[index] >= env.rcfg["stuck_limit"]),
                "cumulative_risk": float(env.cumulative_risk),
                "path_length": path_length,
                "shortest_path_length": shortest,
                "detour": path_length / shortest if shortest > 0 else 1.0,
                "disruption_applied": float(disruption_applied),
            })
    return rows


def summarize(rows):
    if not rows:
        return {}
    summary = {"n_scenarios": len(rows)}
    for key in (
        "success", "reward", "steps", "spl", "collisions", "deadlock",
        "cumulative_risk", "detour",
    ):
        values = np.asarray([row[key] for row in rows], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        summary[f"{key}_ci95"] = ci95
    return summary
