import argparse
import yaml
import re
import json
import traceback
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


def get_policy_mapping_fn(parameter_sharing: bool) -> Callable:
    def mapping_fn(agent_id, episode=None, **kwargs):
        if parameter_sharing:
            return "policy_0"
        match = re.search(r"(\d+)", str(agent_id))
        return f"policy_{match.group(1)}" if match else f"policy_{agent_id}"

    return mapping_fn


def generate_gif(
    envclass, envconfig, modelpath, savepath, title, parameter_sharing, algo_name="IPPO"
):
    print(
        f"Iniciando entorno: {envclass.__name__} | Seed: {envconfig.get('start_seed')} | Mapa: {envconfig.get('map', 'Default')}"
    )

    try:
        if algo_name == "MAPPO":
            wrapper_config = envconfig.copy()
            wrapper_config["base_env_class"] = envclass
            env = MAPPOEnvWrapper(wrapper_config)
        else:
            env = envclass(envconfig)
    except Exception as e:
        raise RuntimeError(f"Fallo instanciando entorno: {e}")

    # Headless Render Patch
    original_render = env.render

    def headless_render(*args, **kwargs):
        kwargs["mode"] = "top_down"
        return original_render(*args, **kwargs)

    env.render = headless_render

    print(f"Cargando checkpoint: {modelpath.name}")
    try:
        algo = Algorithm.from_checkpoint(modelpath)
    except Exception as e:
        if env:
            env.close()
        close_engine()
        raise RuntimeError(f"Fallo cargando checkpoint: {e}")

    module_dict = {}
    mapping_fn = get_policy_mapping_fn(parameter_sharing)

    try:
        # Check actual number of agents in env if possible, or use config
        num_agents = envconfig.get("num_agents", 10)
        available_policies = list(algo.get_weights().keys())

        # Logic to map agents to policies
        for i in range(num_agents):
            agent_id = f"agent{i}"
            policy_id = mapping_fn(agent_id)
            if policy_id not in available_policies:
                if "default_policy" in available_policies:
                    policy_id = "default_policy"
                elif "policy_0" in available_policies:
                    policy_id = "policy_0"

            if policy_id in available_policies:
                module_dict[agent_id] = algo.get_module(policy_id)

        print(f"Ejecutando episodio -> {savepath}")

        env, _ = execute_one_episode(env, module_dict, title, enable_render=True)

        base_env = get_base_env(env)
        if (
            base_env
            and hasattr(base_env, "top_down_renderer")
            and base_env.top_down_renderer
        ):
            base_env.top_down_renderer.generate_gif(savepath)
            print(f"GIF guardado: {savepath}")
        else:
            print("ERROR: top_down_renderer no se inicializó correctamente.")

    except Exception as e:
        raise e
    finally:
        if algo:
            algo.stop()
        if env:
            try:
                env.close()
            except:
                pass
        close_engine()


if __name__ == "__main__":
    if not ray.is_initialized():
        if platform.system() == "Windows":
            ray.init(
                ignore_reinit_error=True,
                runtime_env={"env_vars": {"USE_LIBUV": "0"}},
                log_to_driver=False,
            )
        else:
            ray.init(ignore_reinit_error=True, log_to_driver=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("ckpt_num", type=str)
    args = parser.parse_args()

    base_dir = Path.cwd() / "experiments"
    exp_dir = base_dir / args.exp_name
    ckpt_path = exp_dir / "checkpoints" / args.ckpt_num
    params_json_path = ckpt_path / "params.json"
    config_yaml_path = exp_dir / "config.yaml"

    # --- 1. LOAD RAW CONFIG ---
    raw_params = {}
    algo_name = "IPPO"
    param_sharing = False

    if params_json_path.exists():
        with open(params_json_path, "r") as f:
            data = json.load(f)
            raw_params = (
                data.get("env_config")
                or data.get("environment", {}).get("env_config", {})
                or data
            )
    elif config_yaml_path.exists():
        with open(config_yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
            if isinstance(config_dict, list):
                config_dict = config_dict[-1]
            raw_params = config_dict["environment"]
            param_sharing = config_dict["agent"].get("parameter_sharing", False)
            algo_name = config_dict["agent"].get("algorithm", "IPPO")
    else:
        raise FileNotFoundError("Config no encontrada.")

    # --- 2. CONFIGURACIÓN BASE ---
    # Usamos una configuración limpia y forzamos parámetros de visualización
    base_env_config = {
        "num_agents": raw_params.get("num_agents", 10),
        # start_seed no importa mucho en primitivas fijas, pero lo mantenemos
        "start_seed": 42,
        "horizon": 1000,
        "traffic_density": 0,  # Un poco de tráfico para ver interacción
        "use_render": False,
        "disable_collision": False,
        "random_spawn_lane_index": True,
        "random_lane_width": False,
        "random_lane_num": True,
        "allow_respawn": False,
        "agent_observation": LidarStateObservation,
        # Importante: NO incluimos 'agent_configs' para que se autogeneren
    }

    # --- 3. LISTA DE PRIMITIVAS A GENERAR ---
    targets = [
        ("Intersection", MultiAgentIntersectionEnv),
        ("Roundabout", MultiAgentRoundaboutEnv),
        ("Bottleneck", MultiAgentBottleneckEnv),
    ]

    print("DEBUG: Iniciando generación de GIFs para primitivas...")

    for env_name, EnvClass in targets:
        print(f"\n>>> Generando para: {env_name}")

        # Copia fresca para no contaminar el siguiente loop
        current_config = base_env_config.copy()

        # IMPORTANTE: Las primitivas CRASHEAN si ven "map" en la config
        if "map" in current_config:
            del current_config["map"]

        output_path = exp_dir / f"eval_{args.exp_name}_{args.ckpt_num}_{env_name}.gif"

        try:
            generate_gif(
                envclass=EnvClass,
                envconfig=current_config,
                modelpath=ckpt_path,
                savepath=str(output_path),
                title=f"{args.exp_name} - {env_name}",
                parameter_sharing=param_sharing,
                algo_name=algo_name,
            )
        except Exception as e:
            print(f"!!! Error generando {env_name}: {e}")
            # traceback.print_exc()
            try:
                close_engine()
            except:
                pass

    print("\n>>> Proceso finalizado.")
