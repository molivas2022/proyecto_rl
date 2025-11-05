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

def actions_from_distributions(module_dict, obs):
    action_dict = {}

    for agent in obs:
        module = module_dict[agent]
        input_dict = {Columns.OBS: torch.from_numpy(obs[agent]).unsqueeze(0)}
        out_dict = module.forward_inference(input_dict)
        # Las medias de las distribuciones para cada acción.
        act1_mean = out_dict["action_dist_inputs"][0][0].item()
        act2_mean = out_dict["action_dist_inputs"][0][2].item()
        action_dict[agent] = [act1_mean, act2_mean]


    return action_dict



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


    # El formato de obs es: obs = dic {agent : observation}
    obs, info = env.reset()


    terminateds = {"__all__": False}
    truncateds = {"__all__": False}

    while not (terminateds["__all__"] or truncateds["__all__"]):

        # obtener acciones dic {agent : action}
        actions = actions_from_distributions(module_dict, obs)

        # ejecutar step
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # configuraciones de renderizado y renderizar
        env.render(
            window=False,
            mode="topdown",
            scaling=2,
            camera_position=(100, 0),
            screen_size=(500, 500),
            screen_record=True,
            text={"episode_step": env.engine.episode_step, "method": title},
        )

    algo.stop()

    print("Rendering environment to gif")
    env.top_down_renderer.generate_gif(savepath)
    print("Rendering done, stopping algorithm and closing environment...")
    env.close()


if __name__ == "__main__":
    exp_dir = Path.cwd() / "experimentos" / "exp1"
    modelpath = exp_dir / "checkpoints" / "final"

    generate_gif(
        envclass=MultiAgentIntersectionEnv,
        envconfig=dict(num_agents=3, allow_respawn=False),
        seed=0,
        modelpath=modelpath,
        savepath=str(exp_dir / "example.gif"),
        title="IPPO",
    )
