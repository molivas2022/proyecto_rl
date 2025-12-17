import torch
import yaml
import warnings
from pathlib import Path

from metadrive import (
    MultiAgentMetaDrive,
    MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv,
    MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv,
    MultiAgentParkingLotEnv,
)
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from metadrive.obs.state_obs import LidarStateObservation

# Imports del proyecto
# from src.utils import generate_gif
from src.models import IPPOCNN, MetaDriveStackedCNN, MAPPOCNN, MAPPOMLP
from src.envs import StackedLidarObservation
from src.trainers import IPPOTrainer, MAPPOTrainer

from ray.rllib.utils.framework import try_import_torch
from metadrive.engine.engine_utils import close_engine

ALGORITHMS = {"IPPO": IPPOTrainer, "MAPPO": MAPPOTrainer}

OBSERVATIONS = {
    "StackedLidar": StackedLidarObservation,
    "Lidar": LidarStateObservation,
}

MODELS = {
    "IPPOCNN": IPPOCNN,
    "StackedCNN": MetaDriveStackedCNN,
    "IPPOMLP": PPOTorchRLModule,
    "MAPPOCNN": MAPPOCNN,
    "MAPPOMLP": MAPPOMLP,
}

ENVS = {
    "Roundabout": MultiAgentRoundaboutEnv,
    "Intersection": MultiAgentIntersectionEnv,
    "Tollgate": MultiAgentTollgateEnv,
    "Bottleneck": MultiAgentBottleneckEnv,
    "Parkinglot": MultiAgentParkingLotEnv,
    "PGMA": MultiAgentMetaDrive,
}

warnings.filterwarnings("ignore")


def run_experiments():
    current_dir = Path.cwd()
    CONFIG_PATH = current_dir / "experiments.yml"
    BASE_OUTPUT_DIR = current_dir / "experiments"

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No se encontró experiment.yml en: {CONFIG_PATH}")

    print(f"Cargando configuración desde: {CONFIG_PATH}")

    # Cargar YAML
    with open(CONFIG_PATH) as f:
        experiments_list = yaml.load(f, Loader=yaml.SafeLoader)

    if not isinstance(experiments_list, list):
        raise TypeError("El YAML debe ser una lista de experimentos.")

    for i, exp_config in enumerate(experiments_list):
        exp_name = exp_config.get("experiment_name", f"exp_run_{i}")
        CURRENT_EXP_DIR = BASE_OUTPUT_DIR / exp_name
        CURRENT_EXP_DIR.mkdir(parents=True, exist_ok=True)
        final_ckpt = CURRENT_EXP_DIR / "checkpoints" / "final"
        if final_ckpt.exists():
            print(f"\n>>> Saltando {exp_name}: Ya existe checkpoint final.")
            continue
        print(f"\n>>> Preparando: {exp_name}")

        with open(CURRENT_EXP_DIR / "config.yaml", "w") as f_out:
            yaml.dump(exp_config, f_out)

        # ---------------------------------------------------------
        # 1. EXTRACCIÓN Y VALIDACIÓN DE STRINGS DEL YAML
        # ---------------------------------------------------------
        try:
            algo_str = exp_config["agent"]["algorithm"]
            # Asumimos que agregaste 'type' en environment en el YAML.
            # Si no, por defecto usa Intersection.
            env_str = exp_config["environment"]["type"]
            obs_str = exp_config["agent"]["observation"]
            policy_str = exp_config["agent"]["policy_type"]
        except KeyError as e:
            raise KeyError(
                f"Falta la clave {e} en la configuración del experimento {exp_name}"
            )

        # ---------------------------------------------------------
        # 2. RESOLUCIÓN DE CLASES (Mapping String -> Class)
        # ---------------------------------------------------------
        if algo_str not in ALGORITHMS:
            raise ValueError(f"Algoritmo '{algo_str}' no definido en ALGORITHMS.")

        if env_str not in ENVS:
            raise ValueError(f"Environment '{env_str}' no definido en ENVS.")
        EnvClass = ENVS[env_str]

        # ---------------------------------------------------------
        # 3. INTERCAMBIO EN EL DICCIONARIO (Mutación del Config)
        # ---------------------------------------------------------
        # Aquí reemplazamos el string 'StackedLidarObservation' por la clase real
        if policy_str in MODELS:
            exp_config["agent"]["policy_type"] = MODELS[policy_str]
        else:
            print(
                f"Advertencia: Modelo '{policy_str}' no encontrado en mapa, se mantiene como string."
            )

        if obs_str in OBSERVATIONS:
            exp_config["agent"]["observation"] = OBSERVATIONS[obs_str]
        else:
            print(
                f"Advertencia: Observación '{obs_str}' no encontrada en mapa, se mantiene como string."
            )

        # Lógica para determinar use_cnn dinámicamente basado en el string original
        use_cnn_flag = "CNN" in policy_str
        # Guardamos la config PURA (con strings) antes de mutarla con clases
        # (porque YAML no puede serializar clases de Python fácilmente)
        # Nota: Si quieres guardar lo que ejecutaste, haz esto ANTES del paso 3.
        # Aquí guardamos una versión "limpia" reconstruyendo el dict si fuera necesario,
        # pero para simplificar, asumimos que el dump ya se hizo o se hace con cuidado.

        # ---------------------------------------------------------
        # 4. EJECUCIÓN
        # ---------------------------------------------------------
        try:
            print(
                f">>> Ejecutando con: Algo={algo_str}, Env={env_str}, CNN={use_cnn_flag}"
            )

            # Instanciamos la clase dinámica (TrainerClass)
            # Pasamos la clase de entorno dinámica (EnvClass)
            if algo_str == "IPPO":

                exp = IPPOTrainer(
                    exp_config=exp_config,
                    env_class=EnvClass,
                    exp_dir=CURRENT_EXP_DIR,
                    start_from_checkpoint=False,
                    use_cnn=use_cnn_flag,
                )
            elif algo_str == "MAPPO":
                exp = MAPPOTrainer(
                    exp_config=exp_config,
                    env_class=EnvClass,
                    exp_dir=CURRENT_EXP_DIR,
                    use_cnn=use_cnn_flag,
                )

            _ = exp.train()
            print(f">>> Finalizado: {exp_name}")

        except Exception as e:
            print(f"!!! Error fatal en {exp_name}: {e}")
            raise e
        finally:
            close_engine()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Cuda available.")
    else:
        print("Cuda not available.")
    run_experiments()
