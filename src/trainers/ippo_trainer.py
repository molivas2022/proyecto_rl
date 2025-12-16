import re
from pathlib import Path
from typing import Any, Dict, Optional

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import UnifiedLogger

from src.envs import MetadriveEnvWrapper
from src.utils import transfer_module_weights
from .ppo_metrics_logger import PPOMetricsLogger
import logging
import torch


class IPPOTrainer:
    def __init__(
        self,
        exp_config: Dict[str, Any],
        env_class: Any,
        exp_dir: Path,
        start_from_checkpoint: bool = False,
        use_cnn: bool = False,
    ):
        self.exp_config = exp_config.copy()
        self.exp_dir = exp_dir
        self.start_from_checkpoint = start_from_checkpoint
        self.use_cnn = use_cnn

        # Configuración del entorno
        self.env_config = self._build_env_config(env_class)

        # Inicialización de espacios y especificaciones
        self.policies = {}
        self.rl_module_spec: Optional[MultiRLModuleSpec] = None
        self._initialize_spaces_and_specs()

    def _build_env_config(self, env_class: Any) -> Dict[str, Any]:
        """Centraliza la creación de la configuración del entorno."""
        env_params = self.exp_config["environment"]
        hyperparameters = self.exp_config["hyperparameters"]

        return {
            "base_env_class": env_class,
            "num_agents": env_params["num_agents"],
            "allow_respawn": env_params["allow_respawn"],
            "horizon": env_params["horizon"],
            "traffic_density": env_params["traffic_density"],
            "agent_observation": self.exp_config["agent"]["observation"],
            "normalize_reward": env_params.get("normalize_reward", False),
            # --- OPCIONALES (solo se añaden si existen en el YAML) ---
            **(
                {"start_seed": env_params["start_seed"]}
                if "start_seed" in env_params
                else {}
            ),
            **(
                {"num_scenarios": env_params["num_scenarios"]}
                if "num_scenarios" in env_params
                else {}
            ),
            **({"map": env_params["map"]} if "map" in env_params else {}),
            **(
                {"test_start_seed": env_params["test_start_seed"]}
                if "test_start_seed" in env_params
                else {}
            ),
            **(
                {"test_num_scenarios": env_params["test_num_scenarios"]}
                if "test_num_scenarios" in env_params
                else {}
            ),
            # --- SIGUE TODO TAL CUAL ---
            "gamma": hyperparameters["gamma"],
            # Rewards
            "crash_vehicle_penalty": env_params.get("crash_vehicle_penalty", 5.0),
            "crash_object_penalty": env_params.get("crash_object_penalty", 5.0),
            "out_of_road_penalty": env_params.get("out_of_road_penalty", 5.0),
            "driving_reward": env_params.get("driving_reward", 1.0),
            "speed_reward": env_params.get("speed_reward", 0.1),
            "success_reward": env_params.get("success_reward", 10.0),
        }

    @staticmethod
    def policy_mapping_fn(agent_id: str, episode: Any = None, **kwargs) -> str:
        """
        Mapea agentes a políticas. Debe ser estático o externo para serialización en Ray.
        """
        match = re.search(r"(\d+)", str(agent_id))
        if match:
            return f"policy_{match.group(1)}"
        return f"policy_{agent_id}"

    def _initialize_spaces_and_specs(self) -> None:
        """
        Instancia un entorno temporal para obtener espacios de acción/observación
        y construir las especificaciones de políticas y RLModules.
        """
        # Context manager implícito o try/finally para asegurar cierre
        temp_env = MetadriveEnvWrapper(self.env_config)
        try:
            temp_env.reset()
            # Número de lasers
            # print(temp_env.env.config["vehicle_config"]["lidar"])

            rl_module_specs_dict = {}

            for agent_id, obs_space in temp_env.observation_spaces.items():
                act_space = temp_env.action_spaces[agent_id]
                policy_id = self.policy_mapping_fn(agent_id, None)

                # Definición de Policies
                if policy_id not in self.policies:
                    self.policies[policy_id] = PolicySpec(
                        observation_space=obs_space, action_space=act_space, config={}
                    )

                # Definición de RLModules
                if policy_id not in rl_module_specs_dict:
                    # TODO: revisar si esto funciona, cual es valor por defecto
                    # sin CNN
                    model_config = {"hidden_dim": 256} if self.use_cnn else {}

                    rl_module_specs_dict[policy_id] = RLModuleSpec(
                        module_class=self.exp_config["agent"]["policy_type"],
                        observation_space=obs_space,
                        action_space=act_space,
                        model_config=model_config,
                    )

            self.rl_module_spec = MultiRLModuleSpec(
                rl_module_specs=rl_module_specs_dict
            )

        finally:
            if hasattr(temp_env, "env"):
                temp_env.env.close()

    def _build_algorithm_config(self) -> PPOConfig:
        """Construye y retorna el objeto PPOConfig."""
        hyperparams = self.exp_config["hyperparameters"]
        num_gpus = 1 if torch.cuda.is_available() else 0

        config = (
            PPOConfig()
            .training(
                gamma=hyperparams["gamma"],
                clip_param=hyperparams["clip_param"],
                # Se samplean train_batch_size env steps por training iteration. El default es 4000
                # Tener esto en cuenta. En la config los estoy dejando las reducciones con target
                # en epochs 200, 1000.
                lr=hyperparams["learning_rate"],
                entropy_coeff=hyperparams["entropy_coeff"],
                lambda_=hyperparams["lambda"],
                train_batch_size_per_learner=hyperparams["train_batch_size"],
                minibatch_size=hyperparams["minibatch_size"],
                vf_clip_param=hyperparams["vf_clip_param"],
                grad_clip=hyperparams["grad_clip"],
            )
            .multi_agent(
                policies=self.policies, policy_mapping_fn=self.policy_mapping_fn
            )
            .environment(env=MetadriveEnvWrapper, env_config=self.env_config)
            .framework("torch")
            .resources(num_gpus=1)
            .env_runners(
                num_env_runners=self.exp_config["environment"]["num_env_runners"],
                num_envs_per_env_runner=self.exp_config["environment"][
                    "num_envs_per_env_runner"
                ],
                num_cpus_per_env_runner=self.exp_config["environment"][
                    "num_cpus_per_env_runner"
                ],  # vCPU
                # Prefiero no cambiar esto (por ahora) la verdad
                # rollout_fragment_length=self.exp_config["hyperparameters"]["rollout_fragment_length"],
            )
            .learners(num_learners=1, num_gpus_per_learner=num_gpus)
            .evaluation(
                evaluation_interval=self.exp_config["experiment"].get(
                    "evaluation_interval", 50
                ),
                evaluation_duration=self.exp_config["experiment"].get(
                    "evaluation_duration", 5
                ),
                evaluation_duration_unit="episodes",
                evaluation_num_env_runners=1,
                evaluation_config={"explore": False},
            )
            .callbacks(PPOMetricsLogger)
            .update_from_dict(
                {
                    "callback_args": {
                        "PPOMetricsLogger": {
                            "exp_dir": self.exp_dir,
                            "log_save_frequency": self.exp_config["experiment"].get(
                                "log_save_freq", 100
                            ),
                        }
                    }
                }
            )
            .rl_module(rl_module_spec=self.rl_module_spec)
        )
        return config

    def _load_weights_if_needed(self, algo: Algorithm) -> None:
        """Maneja la lógica de transferencia de pesos desde checkpoints base."""
        if not self.start_from_checkpoint:
            return

        checkpoint_path = self.exp_dir / "base_checkpoint"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint base no encontrado en: {checkpoint_path}"
            )

        print(f"Loading weights from {checkpoint_path}...")
        source_algo = Algorithm.from_checkpoint(checkpoint_path)

        # Lógica de mismatch de políticas
        idol = (
            "policy_0"
            if len(self.policies) != len(source_algo.config.policies)
            else None
        )

        transfer_module_weights(source_algo, algo, self.policies, idol)
        source_algo.stop()

    def train(self) -> Path:
        """Ejecuta el bucle de entrenamiento."""
        if not ray.is_initialized():
            ray.init(
                # log_to_driver=False,
                ignore_reinit_error=True,
                # logging_level=logging.ERROR,
            )

        print("Building PPO Algorithm...")
        algo_config = self._build_algorithm_config()

        # Para tensorboard
        def logger_creator(config):
            log_dir = self.exp_dir / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            return UnifiedLogger(config, str(log_dir), loggers=None)

        algo = algo_config.build(logger_creator=logger_creator)

        print("Finished building PPO Algorithm...")

        self._load_weights_if_needed(algo)

        # Checkpoint inicial
        save_dir = self.exp_dir / "checkpoints"
        algo.save_to_path(save_dir / "0")

        print("Starting training loop")
        n_epochs = self.exp_config["hyperparameters"]["n_epochs"]
        checkpoint_freq = self.exp_config["experiment"]["checkpoint_freq"]

        for i in range(n_epochs):
            algo.train()

            epoch = i + 1
            if epoch % checkpoint_freq == 0:
                print(f"Saving checkpoint at epoch {epoch}")
                algo.save_to_path(save_dir / str(epoch))

        print("Training complete. Saving final state.")
        final_dir = algo.save_to_path(save_dir / "final")

        algo.stop()
        ray.shutdown()

        return final_dir
