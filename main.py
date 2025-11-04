from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import MultiAgentEnv
from metadrive import (
    MultiAgentIntersectionEnv,
)  # probemos mientras solo con intersección
from pprint import pprint  # imprimir eedd con formato
from ray.tune import register_env
from metadrive_wrapper import MetadriveEnvWrapper
import warnings

# No mostrar warnings que no van al caso edl proyecto

import torch


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

env_config = dict(
    base_env_class=MultiAgentIntersectionEnv,
    num_agents=2,
    allow_respawn=False,
    # delay_done=5
)

print(env_config)

# Esto es para ver la cantidad de agentes que hay en un env test que se crea
# con la configuración
try:
    temp_env = MetadriveEnvWrapper(env_config)

    temp_env.reset()

    policies_dict = {}

    for agent_id, obs_space in temp_env.observation_spaces.items():

        act_space = temp_env.action_spaces[agent_id]

        # crear un id de policy para cada agente
        policy_id = f"policy_{agent_id}"
        policies_dict[policy_id] = (None, obs_space, act_space, {})

    # función de mapeo agent -> policy
    policy_mapping_fn = lambda agent_id, episode, **kwargs: f"policy_{agent_id}"

finally:
    if "temp_env" in locals() and hasattr(temp_env, "env"):
        temp_env.env.close()
    del temp_env

# Usar ppo
config = PPOConfig()

# Esto para configura multiagente
# esto es para IPPO, cada agente con una politica
config = config.multi_agent(
    # diccionario de politicas
    policies=policies_dict,
    # mapeo politica - agente
    policy_mapping_fn=policy_mapping_fn,
)

# Configurar entorno
config = config.environment(env=MetadriveEnvWrapper, env_config=env_config)

# que se usara como framework
config.framework("torch")
config.resources(num_gpus=1)

# Paralelización
config.env_runners(num_env_runners=2)

ppo = config.build_algo()

# esto no entrena bien, es solo prueba
for _ in range(4):
    pprint(ppo.train())


