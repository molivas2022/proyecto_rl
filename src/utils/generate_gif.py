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
from src.utils.execute_episode import execute_one_episode 

from src.envs import StackedLidarObservation

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


def generate_gif(envclass, envconfig, modelpath, savepath, title, parameter_sharing):
    print(f"Iniciando entorno: {envclass.__name__}")
    
    try:
        env = envclass(envconfig)
    except Exception as e:
        raise RuntimeError(f"Fallo instanciando entorno: {e}")

    print(f"Cargando checkpoint: {modelpath.name}")
    try:
        algo = Algorithm.from_checkpoint(modelpath)
    except Exception as e:
        raise RuntimeError(f"Fallo cargando checkpoint: {e}")

    module_dict = {}
    mapping_fn = get_policy_mapping_fn(parameter_sharing)

    print("Mapeando agentes a políticas...")
    try:
        for i in range(envconfig["num_agents"]):
            agent_id = f"agent{i}"
            policy_id = mapping_fn(agent_id)
            
            module = algo.get_module(policy_id)
            
            if module is None:
                raise ValueError(f"Política '{policy_id}' no encontrada en checkpoint para '{agent_id}'.")
            
            module_dict[agent_id] = module
            
    except Exception as e:
        algo.stop()
        env.close()
        raise RuntimeError(f"Error en asignación de módulos: {e}")

    try:
        print(f"Ejecutando episodio y guardando en: {savepath}")

        # OJO: revisar si funciona con MAPPO

        env, _ = execute_one_episode(env, module_dict, title, enable_render=True)

        env.top_down_renderer.generate_gif(savepath)
        print(f"GIF GENERADO CORRECTAMENTE!")
    except Exception as e:
        print(f"!!! Error renderizando GIF: {e}")
    finally:
        algo.stop()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de GIFs para MetaDrive/RLlib")
    parser.add_argument("exp_name", type=str, help="Nombre de la carpeta del experimento (ej: ippo_mlp)")
    parser.add_argument("ckpt_num", type=str, help="Nombre de la carpeta del checkpoint (ej: 500, final)")
    
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

    # Generación (3 ejemplos por defecto)
    print(f"Generando GIFs para {args.exp_name} | Checkpoint: {args.ckpt_num}")
    for i in range(3):
        seed = env_params.get("start_seed", 42) + i + 100
        current_env_config = env_params.copy()
        current_env_config["start_seed"] = seed
        
        output_name = f"eval_{args.exp_name}_{args.ckpt_num}_seed{seed}.gif"
        output_path = exp_dir / output_name
        
        generate_gif(
            envclass=EnvClass,
            envconfig=current_env_config,
            modelpath=ckpt_path,
            savepath=str(output_path),
            title=f"{args.exp_name} | {args.ckpt_num}",
            parameter_sharing=param_sharing
        )
