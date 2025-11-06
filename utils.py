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

def actions_from_distributions(module_dict, obs, env):
    action_dict = {}

    for agent in obs:
        module = module_dict[agent]
        input_dict = {Columns.OBS: torch.from_numpy(obs[agent]).unsqueeze(0)}
        out_dict = module.forward_inference(input_dict)
        dist_params_np = out_dict["action_dist_inputs"].detach().cpu().numpy()[0]
        raw_means = dist_params_np[[0, 1]]
        greedy_act = np.clip(raw_means, -1.0, 1.0)

        action_dict[agent] = greedy_act

    return action_dict

# Esto hay que cambiarlo de archivo despues jejej
def execute_one_episode(env, module_dict, title = None, enable_render = False):

    obs, info = env.reset()

    terminateds = {"__all__": False}
    truncateds = {"__all__": False}

    infos_list = []
    while not (terminateds["__all__"] or truncateds["__all__"]):
        # obtener acciones dic {agent : action}
        actions = actions_from_distributions(module_dict, obs, env)

        # ejecutar step
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        infos_list.append(infos)

        # configuraciones de renderizado y renderizar
        if (enable_render):
            env.render(
                window=False,
                mode="topdown",
                scaling=2,
                camera_position=(100, 0),
                screen_size=(500, 500),
                screen_record=True,
                text={"episode_step": env.engine.episode_step, "method": title},
            )

    return env, infos_list


