import gymnasium as gym
import numpy as np
from pathlib import Path
from ray.rllib.algorithms.algorithm import Algorithm
from metadrive import MultiAgentIntersectionEnv
from pathlib import Path
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_MODULE_TO_ENV_CONNECTOR,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.core.columns import Columns
import os
from pprint import pprint
import torch
from tqdm import tqdm
import numpy as np
from utils import actions_from_distributions, execute_one_episode
import pandas as pd


def compute_episode_metrics(infos):
    agent_data = {}

    for step_info in infos:
        for agent_id, data in step_info.items():
            if agent_id not in agent_data:
                agent_data[agent_id] = {
                    "velocity": [],
                    "acceleration": [],
                    "route_completion": [],
                    "energy": [],
                }
            # guardar valores
            agent_data[agent_id]["velocity"].append(float(data.get("velocity", 0.0)))
            agent_data[agent_id]["acceleration"].append(
                float(data.get("acceleration", 0.0))
            )
            agent_data[agent_id]["route_completion"].append(
                float(data.get("route_completion", 0.0))
            )
            agent_data[agent_id]["energy"].append(float(data.get("step_energy", 0.0)))

    # calcular promedios
    agent_metrics = {}

    for agent_id, values in agent_data.items():
        agent_metrics[agent_id] = {
            "avg_velocity": np.mean(values["velocity"]),
            "avg_acceleration": np.mean(values["acceleration"]),
            "route_completion": values["route_completion"][-1],
            "total_energy": np.sum(values["energy"]),
        }

    # promedio sobre todos los agentes
    global_metrics = {
        "avg_velocity": np.mean([m["avg_velocity"] for m in agent_metrics.values()]),
        "avg_acceleration": np.mean(
            [m["avg_acceleration"] for m in agent_metrics.values()]
        ),
        "avg_route_completion": np.mean(
            [m["route_completion"] for m in agent_metrics.values()]
        ),
        "total_energy": np.sum([m["total_energy"] for m in agent_metrics.values()]),
    }

    return global_metrics


def calculate_metrics(epochs, envclass, envconfig, modelpath, seed=42):

    env = envclass(envconfig)

    # carga el algoritmo desde un checkpoint
    try:
        algo = Algorithm.from_checkpoint(modelpath)
    except Exception as e:
        print(f"====ERROR: no se pudo cargar el checkpoint.====\n {e}")
        raise

    # obtiene politicas de agentes
    module_dict = {}
    for i in range(envconfig["num_agents"]):
        module_dict[f"agent{i}"] = algo.get_module(f"policy_{i}")

    print("Iniciando evaluación...")

    global_metrics = {
        "avg_velocity": [],
        "avg_acceleration": [],
        "avg_route_completion": [],
        "total_energy": [],
    }

    for epoch in tqdm(range(epochs)):
        env, infos_list = execute_one_episode(env, module_dict)

        # añadir metricas
        metrics = compute_episode_metrics(infos_list)
        global_metrics["avg_velocity"].append(metrics["avg_velocity"])
        global_metrics["avg_acceleration"].append(metrics["avg_acceleration"])
        global_metrics["avg_route_completion"].append(metrics["avg_route_completion"])
        global_metrics["total_energy"].append(metrics["total_energy"])

    final_metrics = {
        "avg_velocity": np.mean(global_metrics["avg_velocity"]),
        "avg_acceleration": np.mean(global_metrics["avg_acceleration"]),
        "avg_route_completion": np.mean(global_metrics["avg_route_completion"]),
        "total_energy": np.mean(global_metrics["total_energy"]),
    }

    algo.stop()
    env.close()

    return final_metrics


if __name__ == "__main__":
    exp_dir = Path.cwd() / "experimentos" / "exp5"
    checkpoint_freq = 20
    last_checkpoint = 500
    num_agents = 5

    metrics_dict = {}
    route_completion_list = []
    for i in range(checkpoint_freq, last_checkpoint + 1, checkpoint_freq):
        modelpath = exp_dir / "checkpoints" / str(i)

        metrics = calculate_metrics(
            epochs=20,
            envclass=MultiAgentIntersectionEnv,
            envconfig=dict(
                num_agents=num_agents, allow_respawn=False, traffic_density=0.1
            ),
            modelpath=modelpath,
        )
        avg_route_completion = metrics["avg_route_completion"]
        route_completion_list.append(avg_route_completion)

    data = {
        "Iteration": [
            i for i in range(checkpoint_freq, last_checkpoint + 1, checkpoint_freq)
        ],
        "Average route completion": route_completion_list,
    }
    pd.DataFrame(data).to_csv(exp_dir / "checkpoint_metrics.csv", index=False)

    # print("####### Metrics ####### ")
    # metrics = calculate_metrics(epochs = 20,
    #                                envclass=MultiAgentIntersectionEnv,
    #                                envconfig=dict(
    #                                num_agents=3,
    #                                allow_respawn=False),
    #                                modelpath = modelpath
    #
    # pprint(metrics)
