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

from ray.rllib.core.learner.learner import (
    COMPONENT_OPTIMIZER,
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
    Transfiere los módulos Y LOS ESTADOS DEL OPTIMIZADOR desde un source algorithm,
    sobreescribiendo los pesos del target algorithm.

    Esta función actualiza el LearnerGroup (la "fuente de verdad" de los pesos) y
    luego sincroniza esos pesos con el EnvRunnerGroup (para inferencia y verificación).
    """
    print("--- Iniciando transferencia de pesos y optimizador ---")

    # --- PASO 1: Obtener el ESTADO COMPLETO DEL LEARNER de origen ---
    print("Obteniendo estado del LearnerGroup de origen...")
    try:
        # Obtenemos el estado del primer (o único) Learner.
        # Esto incluye RL_MODULE, OPTIMIZER, y más.
        source_learner_state = source_algo.learner_group.get_state()[COMPONENT_LEARNER]
        source_module_states = source_learner_state[COMPONENT_RL_MODULE]
        source_optim_states = source_learner_state[COMPONENT_OPTIMIZER]
    except Exception as e:
        print(f"Error al obtener el estado del LearnerGroup de origen: {e}")
        print("Asegúrese de que el 'source_algo' esté usando la nueva pila de API.")
        return

    # --- PASO 2: Construir un ESTADO DE LEARNER PARCIAL para transferir ---
    # Necesitamos transferir tanto los pesos del módulo COMO el estado del optimizador.

    partial_learner_state_to_transfer = {
        COMPONENT_RL_MODULE: {},
        COMPONENT_OPTIMIZER: {},
    }

    print("Mapeando estados de módulo y optimizador...")
    for module_id in module_ids_to_transfer:
        idol_id = idol if idol else module_id

        # El estado del optimizador está indexado por el nombre completo del optimizador,
        # p.ej., "policy_0_default_optimizer".
        # Asumimos el formato de nombre de optimizador por defecto.
        source_optim_name = f"{idol_id}_default_optimizer"

        if idol_id in source_module_states and source_optim_name in source_optim_states:
            # Mapear los pesos del módulo
            partial_learner_state_to_transfer[COMPONENT_RL_MODULE][module_id] = (
                source_module_states[idol_id]
            )

            # Mapear el estado del optimizador para ese módulo
            # La clave para el dict de estado de *destino* también debe ser
            # el nombre completo del optimizador que el Learner de destino espera.
            target_optim_name = f"{module_id}_default_optimizer"
            partial_learner_state_to_transfer[COMPONENT_OPTIMIZER][
                target_optim_name
            ] = source_optim_states[source_optim_name]

            print(
                f"  - Mapeando '{target_optim_name}' (destino) <-- '{source_optim_name}' (origen) [Módulo+Optimizador]"
            )
        else:
            print(
                f"  - ADVERTENCIA: Módulo '{idol_id}' (o su optimizador '{source_optim_name}') no encontrado en el Learner de origen. Saltando '{module_id}'."
            )

    if not partial_learner_state_to_transfer[COMPONENT_RL_MODULE]:
        print("No hay estados para transferir. Abortando.")
        return

    # --- PASO 3: Definir la función de actualización para los Learners de destino ---
    # Esta función ahora usa learner.set_state() para actualizar MÚLTIPLES componentes.
    def update_learner_state(learner, *, partial_state_to_set):
        # 'learner' es la instancia del objeto Learner
        print(
            f"[Learner] Aplicando estado parcial del Learner (Módulos y Optimizadores)..."
        )
        try:
            # learner.set_state() es lo suficientemente inteligente como para
            # fusionar este estado parcial con su estado existente.
            learner.set_state(partial_state_to_set)
            print(f"[Learner] Estado parcial aplicado exitosamente.")
        except Exception as e:
            print(f"[Learner] Error al aplicar estado parcial: {e}")

    # --- PASO 4: Ejecutar la actualización en *todos* los workers del LearnerGroup de destino ---

    # 4.1. Aplicar a todos los LEARNERS REMOTOS (si existen)
    print(
        "Ejecutando 'set_state' en todos los workers remotos del LearnerGroup (si existen)..."
    )
    target_algo.learner_group.foreach_learner(
        update_learner_state, partial_state_to_set=partial_learner_state_to_transfer
    )

    # 4.2. Aplicar al LEARNER LOCAL (si existe)
    if target_algo.learner_group._learner:
        print("Aplicando 'set_state' al worker local del LearnerGroup...")
        update_learner_state(
            target_algo.learner_group._learner,
            partial_state_to_set=partial_learner_state_to_transfer,
        )

    print("Actualización del LearnerGroup completada.")

    # --- PASO 5: Sincronizar los pesos del LearnerGroup al EnvRunnerGroup ---
    # Este paso sigue siendo crucial para que tu 'algo.get_module()'
    # (que lee desde el EnvRunner) vea los pesos actualizados ANTES de 'algo.train()'.
    print("Sincronizando pesos del LearnerGroup actualizado al EnvRunnerGroup...")
    try:
        # 'sync_weights' extrae los pesos directamente desde el LearnerGroup.
        target_algo.env_runner_group.sync_weights(
            from_worker_or_learner_group=target_algo.learner_group,
            inference_only=True,  # Los EnvRunners solo necesitan pesos de inferencia
        )

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
