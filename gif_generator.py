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


def generate_gif(envclass, envconfig, modelpath, savepath, title, seed=42):
    """
    genera gif de un episodio utilizando un algoritmo
    Args:
        envclass: entorno multiagente de MetaDrive en el que hacer una ejecución de prueba.
        seed: semilla de generación del entorno. Por defecto: 42.
        modelpath: donde se encuentra el checkpoint de entrenamiento
        savepath: donde se guardaran los gifs.
    """

    # inicializar entorno / con config
    env = envclass(envconfig)

    # obtener el algoritmo completo desde el checkpoint
    try:
        algo = Algorithm.from_checkpoint(modelpath)
        # rl_module = MultiRLModule.from_checkpoint(
        #    os.path.join(
        #        modelpath,  # (or directly a string path to the checkpoint dir)
        #        COMPONENT_LEARNER_GROUP,
        #        COMPONENT_LEARNER,
        #        COMPONENT_RL_MODULE,
        #    )
        # )

        # rl_module_path = os.path.join(
        #     modelpath, COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER, COMPONENT_RL_MODULE
        # )
        # print(rl_module_path)
        # rl_module = MultiRLModule.from_checkpoint(rl_module_path)

    except Exception as e:
        print(f"====ERROR: no se pudo cargar el checkpoint.====\n {e}")
        raise

    module_dict = {}
    for i in range(envconfig["num_agents"]):
        module_dict[f"agent{i}"] = algo.get_module(f"policy_{i}")

    env, _ = execute_one_episode(env, module_dict, title, enable_render = True)

    env.top_down_renderer.generate_gif(savepath)

    algo.stop()

    print("Rendering environment to gif")
    env.top_down_renderer.generate_gif(savepath)
    print("Rendering done, stopping algorithm and closing environment...")
    env.close()


if __name__ == "__main__":
    exp_dir = Path.cwd() / "experimentos" / "exp2"
    modelpath = exp_dir / "checkpoints" / "final"

    generate_gif(
        envclass=MultiAgentIntersectionEnv,
        envconfig=dict(
            num_agents=3,
            allow_respawn=False,
        ),
        seed=0,
        modelpath=modelpath,
        savepath=str(exp_dir / "example.gif"),
       title="IPPO",
    )
