"""Joint collision-aware local training for FedAvg clients."""

from collections import defaultdict

import numpy as np
import torch

from federated.client import DQNClient, PPOClient, QLearningClient


def _new_ppo_buffer():
    return {key: [] for key in ("obs", "actions", "rewards", "values", "log_probs", "dones")}


def train_clients_multi_robot(clients, multi_env, episodes_per_client: int):
    if len(clients) != multi_env.n_robots:
        raise ValueError("Un client FedAvg est requis pour chaque robot du MultiRobotEnv")

    rewards = [[] for _ in clients]
    successes = [[] for _ in clients]
    collisions = [0 for _ in clients]
    transitions = [0 for _ in clients]
    update_metrics = [[] for _ in clients]

    for _ in range(episodes_per_client):
        sampled = [client.sample_start_target() for client in clients]
        starts = [item[0] for item in sampled]
        targets = [item[1] for item in sampled]
        observations = multi_env.reset(
            starts=starts if all(node is not None for node in starts) else None,
            targets=targets if all(node is not None for node in targets) else None,
        )
        dones = [False] * len(clients)
        episode_rewards = [0.0] * len(clients)
        ppo_buffers = [_new_ppo_buffer() for _ in clients]

        while not all(dones):
            actions = []
            ppo_pending = {}
            for robot_id, (client, obs) in enumerate(zip(clients, observations)):
                if dones[robot_id]:
                    actions.append(0)
                    continue
                if isinstance(client, QLearningClient):
                    action = client._choose_action(
                        client.env.current_node,
                        client.env.target_node,
                        obs["mask"],
                    )
                elif isinstance(client, DQNClient):
                    action = client.agent.act(obs)
                elif isinstance(client, PPOClient):
                    current, target, neighbors, context, valid = client.agent._obs_to_tensors(obs)
                    if not valid:
                        action = 0
                    else:
                        with torch.no_grad():
                            dist, value = client.agent.model(
                                current, target, neighbors, context
                            )
                            local_action = dist.sample()
                            log_prob = dist.log_prob(local_action)
                        action = valid[int(local_action.item())]
                        ppo_pending[robot_id] = (local_action.detach(), log_prob.detach(), value.detach())
                else:
                    raise TypeError(type(client))
                actions.append(action)

            next_observations, step_rewards, step_dones, infos = multi_env.step(actions)
            for robot_id, client in enumerate(clients):
                if dones[robot_id]:
                    continue
                transitions[robot_id] += 1
                reward = float(step_rewards[robot_id])
                done = bool(step_dones[robot_id])
                info = infos[robot_id]
                terminal = bool(info.get("terminated", done and not info.get("truncated", False)))
                episode_rewards[robot_id] += reward
                collisions[robot_id] += int(bool(info.get("collision", False)))

                if isinstance(client, QLearningClient):
                    current = client.env.path_taken[-2] if len(client.env.path_taken) >= 2 else client.env.current_node
                    client._update(
                        current,
                        client.env.target_node,
                        actions[robot_id],
                        reward,
                        client.env.current_node,
                        client.env.target_node,
                        next_observations[robot_id]["mask"],
                        done,
                    )
                elif isinstance(client, DQNClient):
                    client.agent.observe_transition(
                        observations[robot_id], actions[robot_id], reward,
                        next_observations[robot_id], done,
                    )
                elif robot_id in ppo_pending:
                    local_action, log_prob, value = ppo_pending[robot_id]
                    buffer = ppo_buffers[robot_id]
                    buffer["obs"].append(observations[robot_id])
                    buffer["actions"].append(local_action)
                    buffer["rewards"].append(reward)
                    buffer["values"].append(value)
                    buffer["log_probs"].append(log_prob)
                    buffer["dones"].append(float(terminal))
                    if len(buffer["rewards"]) >= client.agent.rollout_len or done:
                        metrics = client.agent._update(
                            buffer,
                            next_obs=next_observations[robot_id],
                            terminal=terminal,
                        )
                        update_metrics[robot_id].extend(metrics)
                        for values in buffer.values():
                            values.clear()
                dones[robot_id] = done

            observations = next_observations

        for robot_id, client in enumerate(clients):
            success = float(client.env.current_node == client.env.target_node)
            rewards[robot_id].append(episode_rewards[robot_id])
            successes[robot_id].append(success)
            client._episode_count += 1
            if isinstance(client, QLearningClient):
                client.epsilon = max(client.epsilon_min, client.epsilon * client.epsilon_decay)
            elif isinstance(client, DQNClient):
                client.agent.end_episode()
            elif isinstance(client, PPOClient):
                client.agent.total_episodes += 1

    results = []
    for robot_id, client in enumerate(clients):
        result = {
            "robot_id": robot_id,
            "rewards": rewards[robot_id],
            "successes": successes[robot_id],
            "n_episodes": episodes_per_client,
            "n_transitions": transitions[robot_id],
            "collisions": collisions[robot_id],
        }
        if isinstance(client, DQNClient):
            result["epsilon"] = client.agent.epsilon
        if isinstance(client, PPOClient):
            for key in ("loss_actor", "loss_critic", "entropy", "kl", "clip_frac", "grad_norm"):
                values = [item[key] for item in update_metrics[robot_id] if key in item]
                result[key] = float(np.mean(values)) if values else 0.0
        results.append(result)
    return results
