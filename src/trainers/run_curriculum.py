import yaml
import shutil
import torch
from pathlib import Path
import warnings
import ray
import os
import time
import gc

# --- 1. Imports ---
from metadrive import (
    MultiAgentMetaDrive,  # Phase 1
    MultiAgentIntersectionEnv,  # Phase 4
    MultiAgentRoundaboutEnv,  # Phase 3
    MultiAgentBottleneckEnv,  # Phase 2
)
from metadrive.obs.state_obs import LidarStateObservation
from src.envs import StackedLidarObservation

# Import Models
from src.models import IPPOCNN, MetaDriveStackedCNN, MAPPOCNN, MAPPOMLP
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

# Import Trainer
from src.trainers import IPPOTrainer
from metadrive.engine.engine_utils import close_engine

# --- 2. Mappings ---
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

warnings.filterwarnings("ignore")


def run_curriculum():
    # Setup Paths
    current_dir = Path.cwd()
    CONFIG_PATH = current_dir / "experiments.yml"
    BASE_OUTPUT_DIR = current_dir / "experiments"

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No se encontró experiment.yml en: {CONFIG_PATH}")

    print(f"Cargando configuración base desde: {CONFIG_PATH}")

    with open(CONFIG_PATH) as f:
        experiments_list = yaml.load(f, Loader=yaml.SafeLoader)

    exp_config = experiments_list[0]
    exp_name = exp_config.get("experiment_name", "curriculum")

    CURRICULUM_ROOT_DIR = BASE_OUTPUT_DIR / exp_name
    if CURRICULUM_ROOT_DIR.exists():
        print(f"\n>>> Limpiando directorio anterior: {exp_name}")
        shutil.rmtree(CURRICULUM_ROOT_DIR)
    CURRICULUM_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 3. Base Class Resolution ---
    try:
        obs_str = exp_config["agent"]["observation"]
        policy_str = exp_config["agent"]["policy_type"]
    except KeyError as e:
        raise KeyError(f"Falta clave {e} en config.")

    if obs_str in OBSERVATIONS:
        exp_config["agent"]["observation"] = OBSERVATIONS[obs_str]
    if policy_str in MODELS:
        exp_config["agent"]["policy_type"] = MODELS[policy_str]

    use_cnn_flag = "CNN" in policy_str
    start_seed = exp_config["environment"].get("start_seed", 0)
    target_agents = exp_config["environment"].get("num_agents", 10)

    # --- RESOURCE HARD LIMITS ---
    # We trust the config or default to 24.
    # We explicitly do NOT trust os.cpu_count() inside the container.
    manual_runners = exp_config["environment"].get("num_env_runners", 24)

    print(f"HARD LIMIT: Using {manual_runners} Env Runners")

    # --- 4. DEFINE PHASES ---
    phases = [
        {
            "name": "Phase1_SoloPhysics",
            "env_class": MultiAgentMetaDrive,
            "overrides": {
                "environment": {
                    "num_agents": 1,
                    "start_seed": start_seed,
                    "num_scenarios": 50,
                    "map": 3,
                    "traffic_density": 0.0,
                    "horizon": 500,
                },
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "n_epochs": 1,
                    "entropy_coeff": 0.02,
                },
            },
        },
        {
            "name": "Phase2_ObstacleCourse",
            "env_class": MultiAgentBottleneckEnv,  # BottleNeck
            "overrides": {
                "environment": {
                    "num_agents": 1,
                    "start_seed": start_seed,
                    "num_scenarios": 1,
                    "traffic_density": 0.10,
                    "crash_vehicle_penalty": 5.0,
                    "horizon": 1000,
                },
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "n_epochs": 150,
                    "entropy_coeff": 0.01,
                },
            },
        },
        {
            "name": "Phase3_TheCloneWar",
            "env_class": MultiAgentRoundaboutEnv,
            "overrides": {
                "environment": {
                    "num_agents": 3,
                    "start_seed": start_seed,
                    "num_scenarios": 1,
                    "traffic_density": 0.05,
                    "crash_vehicle_penalty": 15.0,
                },
                "hyperparameters": {
                    "learning_rate": 1e-4,
                    "n_epochs": 200,
                    "entropy_coeff": 0.015,
                },
            },
        },
        {
            "name": "Phase4_FullScale",
            "env_class": MultiAgentIntersectionEnv,
            "overrides": {
                "environment": {
                    "num_agents": target_agents,
                    "start_seed": start_seed,
                    "num_scenarios": 1,
                    "traffic_density": exp_config["environment"].get(
                        "traffic_density", 0.1
                    ),
                    "horizon": 1000,
                },
                "hyperparameters": {
                    "learning_rate": 5e-5,
                    "n_epochs": 500,
                    "entropy_coeff": 0.001,
                },
            },
        },
    ]

    last_checkpoint = None

    # --- 5. EXECUTION LOOP ---
    try:
        for phase in phases:
            print(f"\n{'='*60}")
            print(f"STARTING {phase['name']} using {phase['env_class'].__name__}")
            print(f"{'='*60}")

            # A. PRE-FLIGHT CLEANUP
            if ray.is_initialized():
                print(">>> Cleaning up previous Ray session...")
                ray.shutdown()

            gc.collect()
            print(">>> Waiting 5s for resources to settle...")
            time.sleep(5)

            ray.init(
                num_cpus=24,
                ignore_reinit_error=True,
                runtime_env={"env_vars": {"USE_LIBUV": "0"}},
                log_to_driver=False,
                include_dashboard=False,  # Save memory
            )

            # C. CONFIG PREP
            current_config = exp_config.copy()
            current_config["environment"] = current_config["environment"].copy()
            current_config["hyperparameters"] = current_config["hyperparameters"].copy()

            # Enforce the manual limit
            current_config["num_env_runners"] = manual_runners

            # Apply Overrides
            for section, params in phase["overrides"].items():
                current_config[section].update(params)

            # Surgical Cleanup for Primitives
            is_pgma = phase["env_class"] == MultiAgentMetaDrive
            if not is_pgma:
                if "map" in current_config["environment"]:
                    del current_config["environment"]["map"]
                if "map_config" in current_config["environment"]:
                    del current_config["environment"]["map_config"]

            phase_dir = CURRICULUM_ROOT_DIR / phase["name"]
            phase_dir.mkdir(exist_ok=True)

            # D. TRAIN
            trainer = IPPOTrainer(
                exp_config=current_config,
                env_class=phase["env_class"],
                exp_dir=phase_dir,
                base_checkpoint_path=last_checkpoint,
                use_cnn=use_cnn_flag,
            )

            final_checkpoint_path = trainer.train()
            last_checkpoint = final_checkpoint_path
            print(f"Finished {phase['name']}. Saved to {last_checkpoint}")

            # E. POST-PHASE CLEANUP
            print(">>> Stopping Trainer algorithm...")
            try:
                if hasattr(trainer, "algorithm") and trainer.algorithm:
                    trainer.algorithm.stop()
                elif hasattr(trainer, "algo") and trainer.algo:
                    trainer.algo.stop()
                elif hasattr(trainer, "_trainer") and trainer._trainer:
                    trainer._trainer.stop()
            except Exception as e:
                print(f"Warning during algo stop: {e}")

            del trainer
            close_engine()
            ray.shutdown()

    except Exception as e:
        print(f"!!! Error fatal: {e}")
        ray.shutdown()
        close_engine()
        raise e

    print(f"\n>>> Curriculum Finalizado Exitosamente.")


if __name__ == "__main__":
    run_curriculum()
