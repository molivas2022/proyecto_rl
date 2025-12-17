import torch
import argparse
import yaml
import re
from pathlib import Path
from typing import Callable
from ray.rllib.algorithms.algorithm import Algorithm
from metadrive import (
    MultiAgentMetaDrive,
    MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv,
    MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv,
    MultiAgentParkingLotEnv,
)
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.engine.engine_utils import close_engine
from src.utils.execute_episode import execute_one_episode, get_base_env

from src.envs import StackedLidarObservation
from src.envs.mappo_wrapper import MAPPOEnvWrapper
from pprint import pprint
import ray
import platform

ENVS = {
    "Roundabout": MultiAgentRoundaboutEnv,
    "Intersection": MultiAgentIntersectionEnv,
    "Tollgate": MultiAgentTollgateEnv,
    "Bottleneck": MultiAgentBottleneckEnv,
    "Parkinglot": MultiAgentParkingLotEnv,
    "PGMA": MultiAgentMetaDrive,
}

OBSERVATIONS = {
    "StackedLidar": StackedLidarObservation,
    "Lidar": LidarStateObservation,
}


# --- Patgon Factogy ---
def get_policy_mapping_fn(parameter_sharing: bool) -> Callable:
    def mapping_fn(agent_id, episode=None, **kwargs):
        # If sharing is ON, everyone goes to policy_0
        if parameter_sharing:
            return "policy_0"

        match = re.search(r"(\d+)", str(agent_id))
        if match:
            return f"policy_{match.group(1)}"
        return f"policy_{agent_id}"

    return mapping_fn


def generate_gif(
    envclass, envconfig, modelpath, savepath, title, parameter_sharing, algo_name="IPPO"
):
    print(f"Iniciando entorno: {envclass.__name__}")

    try:
        if algo_name == "MAPPO":
            print("MAPPO detectado: utilizando MAPPOEnvWrapper...")
            wrapper_config = envconfig.copy()
            wrapper_config["base_env_class"] = envclass
            env = MAPPOEnvWrapper(wrapper_config)
        else:
            envconfig = envclass.default_config().update(envconfig)
            env = envclass(envconfig)
    except Exception as e:
        raise RuntimeError(f"Fallo instanciando entorno: {e}")

    print(f"Cargando checkpoint: {modelpath.name}")
    try:
        algo = Algorithm.from_checkpoint(modelpath)
    except Exception as e:
        # Cleanup if checkpoint load fails
        if env:
            env.close()
        close_engine()  # FIX: Force close engine
        raise RuntimeError(f"Fallo cargando checkpoint: {e}")

    module_dict = {}
    mapping_fn = get_policy_mapping_fn(parameter_sharing)

    print("Mapeando agentes a políticas...")
    try:
        # Note: Depending on wrapper, envconfig["num_agents"] might not be available directly on env
        # Safer to use the sanitized dict passed in or just loop strict count
        num_agents = envconfig.get("num_agents", 1)

        for i in range(num_agents):
            agent_id = f"agent{i}"
            policy_id = mapping_fn(agent_id)
            module = algo.get_module(policy_id)
            if module is None:
                raise ValueError(
                    f"Política '{policy_id}' no encontrada para '{agent_id}'."
                )
            module_dict[agent_id] = module

    except Exception as e:
        algo.stop()
        env.close()
        close_engine()  # FIX: Force close engine
        raise RuntimeError(f"Error en asignación de módulos: {e}")

    try:
        print(f"Ejecutando episodio y guardando en: {savepath}")

        env, _ = execute_one_episode(env, module_dict, title, enable_render=True)

        # FIX: Unwrap env to find the renderer
        base_env = get_base_env(env)
        if base_env and hasattr(base_env, "top_down_renderer"):
            base_env.top_down_renderer.generate_gif(savepath)
            print(f"GIF GENERADO CORRECTAMENTE!")
        else:
            print("ERROR: No se encontró top_down_renderer en el entorno base.")

    except Exception as e:
        print(f"!!! Error renderizando GIF: {e}")
        import traceback

        traceback.print_exc()  # Useful to see real error
    finally:
        if algo:
            algo.stop()
        if env:
            try:
                env.close()
            except:
                pass

        # FIX: CRITICAL - Kill the singleton engine so the next loop doesn't crash
        print("Forzando cierre de motor MetaDrive...")
        close_engine()


if __name__ == "__main__":
    if not ray.is_initialized():
        if platform.system() == "Windows":
            print("Inicializando Ray con fix USE_LIBUV='0' para Windows...")
            ray.init(
                ignore_reinit_error=True,
                runtime_env={"env_vars": {"USE_LIBUV": "0"}},
                log_to_driver=False,
            )
        else:
            ray.init(ignore_reinit_error=True)

    parser = argparse.ArgumentParser(
        description="Generador de GIFs para MetaDrive/RLlib"
    )
    parser.add_argument(
        "exp_name", type=str, help="Nombre de la carpeta del experimento (ej: ippo_mlp)"
    )
    parser.add_argument(
        "ckpt_num",
        type=str,
        help="Nombre de la carpeta del checkpoint (ej: 500, final)",
    )

    args = parser.parse_args()

    base_dir = Path.cwd() / "experiments"
    exp_dir = base_dir / args.exp_name
    config_path = exp_dir / "config.yaml"
    ckpt_path = exp_dir / "checkpoints" / args.ckpt_num

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experimento no encontrado: {exp_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuración no encontrada: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")

    print(f"Leyendo config de: {config_path}")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # obtener parámetros importantes
    env_params = config_dict["environment"]
    agent_params = config_dict["agent"]
    param_sharing = agent_params.get("parameter_sharing", False)

    # Resolución de Clases
    env_type = env_params.get("type", "Intersection")
    if env_type not in ENVS:
        raise ValueError(f"Tipo de entorno '{env_type}' no registrado en ENVS.")
    EnvClass = ENVS[env_type]

    obs_str = agent_params.get("observation")
    if obs_str in OBSERVATIONS:
        # Inyección de la clase de observación para MetaDrive
        env_params["agent_observation"] = OBSERVATIONS[obs_str]

    # MetaDrive is strict. We must remove keys that RLlib uses (like 'num_env_runners')
    # but that MetaDrive doesn't know about.

    # Get all valid keys for this specific environment class
    valid_keys = set(EnvClass.default_config().keys())

    # Create a new dict keeping ONLY keys that exist in the valid_keys set
    sanitized_params = {k: v for k, v in env_params.items() if k in valid_keys}

    # Debug: Print what we removed
    removed_keys = set(env_params.keys()) - set(sanitized_params.keys())
    if removed_keys:
        print(f"Cleaned config. Removed RLlib keys: {removed_keys}")
    # --- CRITICAL FIX END ---

    algo_name = config_dict["agent"].get("algorithm", "IPPO")
    # Generación (3 ejemplos por defecto)
    print(f"Generando GIFs para {args.exp_name} | Checkpoint: {args.ckpt_num}")
    for i in range(3):
        seed = env_params.get("start_seed", 42) + i + 100
        current_env_config = sanitized_params.copy()
        current_env_config["start_seed"] = seed

        output_name = f"eval_{args.exp_name}_{args.ckpt_num}_seed{seed}.gif"
        output_path = exp_dir / output_name

        generate_gif(
            envclass=EnvClass,
            envconfig=current_env_config,
            modelpath=ckpt_path,
            savepath=str(output_path),
            title=f"{args.exp_name} | {args.ckpt_num}",
            parameter_sharing=param_sharing,
            algo_name=algo_name,
        )
