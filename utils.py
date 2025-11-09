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


def transfer_module_weights(source_algo, target_algo, module_ids_to_transfer, idol):
    """
    Transfiere los módulos desde un source algorithm, sobreescribiendo los pesos del target algorithm.
    Útil para no tener que hacer uso de Algorithm.from_checkpoint(), el cual importa todo el config.
    En caso de que cardinalidad de policies sea distinto en ambos modelos, para simplificar las cosas,
    se define una policy "idol" en source_algo, el cual sobreescribe todas las politicas del target.

    Es importante actualizar el LearnerGroup (quiene sostienen la copia maestra de los pesos), y sincronizar
    con el EnvRunnerGroup
    """

    print("Iniciando transferencia de pesos")

    # Obtener los estados de los módulos del learner de origen
    print("Obteniendo estado del LearnerGroup de origen...")
    try:
        source_learner_state = source_algo.learner_group.get_state()
        source_module_states = source_learner_state[COMPONENT_LEARNER][
            COMPONENT_RL_MODULE
        ]
    except Exception as e:
        print(f"Error al obtener el estado del LearnerGroup de origen: {e}")
        return

    # Mapear los estados de origen a los módulos de destino
    states_to_transfer = {}
    print("Mapeando estados de origen a módulos de destino...")
    for module_id in module_ids_to_transfer:
        idol_id = idol if idol else module_id

        if idol_id in source_module_states:
            states_to_transfer[module_id] = source_module_states[idol_id]
            print(f"  - Mapeando '{module_id}' (destino) <-- '{idol_id}' (origen)")
        else:
            print(
                f"ADVERTENCIA: Módulo '{idol_id}' no encontrado en el Learner de origen. Saltando '{module_id}'."
            )

    if not states_to_transfer:
        print("No hay estados para transferir. Abortando.")
        return

    # Definimos una función de actualización para los Learners de destino
    def update_learner_modules(learner, *, states_to_set):
        # learner es la instancia del objeto Learner
        print(f"[Learner] Aplicando {len(states_to_set)} estados de módulo...")
        for module_id, state_to_load in states_to_set.items():
            if module_id in learner.module:
                try:
                    learner.module[module_id].set_state(state_to_load)
                    print(f"[Learner] Estado aplicado exitosamente a {module_id}")
                except Exception as e:
                    print(f"[Learner] Error al aplicar estado a {module_id}: {e}")
            else:
                print(
                    f"[Learner] ADVERTENCIA: Módulo {module_id} no encontrado en este Learner."
                )

    # Ejecutar la actualización en *todos* los workers del LearnerGroup de destino

    # Aplicar a todos los LEARNERS REMOTOS (si existen)
    print(
        "Ejecutando 'set_state' en todos los workers remotos del LearnerGroup (si existen)..."
    )
    target_algo.learner_group.foreach_learner(
        update_learner_modules, states_to_set=states_to_transfer
    )

    # Aplicar al LEARNER LOCAL (si existe)
    if target_algo.learner_group._learner:
        print("Aplicando 'set_state' al worker local del LearnerGroup...")
        update_learner_modules(
            target_algo.learner_group._learner, states_to_set=states_to_transfer
        )

    print("Actualización del LearnerGroup completada.")

    # Esto es lo último: Sincronizar los pesos del LearnerGroup al EnvRunnerGroup
    print("Sincronizando pesos del LearnerGroup actualizado al EnvRunnerGroup...")
    try:
        updated_weights = target_algo.learner_group.get_weights()

        # Aplicarlos a todos los EnvRunners (incluido el local)
        # 'local_worker=True' aquí SÍ es correcto para EnvRunnerGroup
        target_algo.env_runner_group.set_weights(updated_weights, local_worker=True)

        print("Sincronización con EnvRunnerGroup completada.")
    except Exception as e:
        print(f"Error durante la sincronización de pesos a EnvRuners: {e}")

    print("--- Transferencia de pesos completada ---")


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
def execute_one_episode(env, module_dict, title=None, enable_render=False):

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
        if enable_render:
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
