from pathlib import Path
import copy
from typing import Any, Dict

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
import re
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.logger import UnifiedLogger

from .ppo_metrics_logger import PPOMetricsLogger
from ray.rllib.callbacks.callbacks import RLlibCallback

from src.envs.mappo_wrapper import MAPPOEnvWrapper
from src.models.mappo_model import MAPPOMLP
import logging
import torch

# TODO: Mover despues del refactor
# TODO: Agregar on_algo_end (creo que se debe llamar algo asi), por ahora simplemente puse frequencia de guardado 1.
# TODO: Dejar de medir métricas en base a training_iteration y hacerlo en base a time_steps


class CentralizedCallback(RLlibCallback):

    def _sync_critics(self, algorithm):
        """
        Called at the end of Algorithm.train().
        We use this to Average the Critic weights across all agents.
        """

        # 1. Get the current weights from the Learner Group (CPU or GPU)
        # Structure: {"policy_0": {"layer_name": tensor}, "policy_1": ...}
        weights = algorithm.learner_group.get_weights()

        # Filter for actual agent policies (ignore default_policy if unused)
        policy_ids = [pid for pid in weights.keys() if "policy_" in pid]

        if len(policy_ids) < 2:
            return  # No need to sync if only 1 agent

        # 2. Identify which weights belong to the Critic
        # We look for keys containing 'critic_encoder' or 'vf_head' based on your module definition
        # We take the first policy as a template to find the keys.
        template_weights = weights[policy_ids[0]]
        critic_keys = [
            k
            for k in template_weights.keys()
            if "critic_encoder" in k or "vf_head" in k
        ]

        if not critic_keys:
            print("Warning: No critic keys found to sync.")
            return

        # 3. Calculate the Mean (Average) for every critic parameter
        avg_critic_weights = {}
        n_agents = len(policy_ids)

        for k in critic_keys:
            # Sum the tensors for this specific layer across all policies
            total_weight = sum(weights[pid][k] for pid in policy_ids)
            # Divide by N
            avg_critic_weights[k] = total_weight / n_agents

        # 4. Create the update dictionary
        # We must preserve the unique Actor weights for each agent,
        # so we copy their existing weights and overwrite ONLY the critic.
        new_weights_dict = {}

        for pid in policy_ids:
            # Start with the agent's specific weights (Actor + Old Critic)
            # We use copy to ensure we don't modify the original dict structure mid-loop
            agent_weights = copy.deepcopy(weights[pid])

            # Overwrite the Critic parts with the Global Average
            for k in critic_keys:
                agent_weights[k] = avg_critic_weights[k]

            new_weights_dict[pid] = agent_weights

        # 5. Set the weights back to the Learner
        # The Learner will automatically broadcast these new weights to
        # EnvRunners (workers) at the start of the next iteration.
        algorithm.learner_group.set_weights(new_weights_dict)

        print("Synchronized!")

    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        self._sync_critics(algorithm)


# EXPERIMENT
# TODO: start for checkpoint
# TODO: cnn
class MAPPOTrainer:
    def __init__(
        self,
        exp_config,
        env_class,
        exp_dir,
        use_cnn: bool = False,
    ):
        self.exp_config = exp_config.copy()
        self.exp_dir = exp_dir
        self.use_cnn = use_cnn

        # Configuración del entorno
        self.env_config = self._build_env_config(env_class)

        # Inicialización de espacios y especificaciones
        self.policies = {}
        self.rl_module_spec = None
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
    def policy_mapping_fn(agent_id, episode=None, **kwargs):
        match = re.search(r"(\d+)", str(agent_id))
        if match:
            return f"policy_{match.group(1)}"
        return f"policy_{agent_id}"

    def _initialize_spaces_and_specs(self):
        """
        Instancia entorno temporal MAPPO para obtener espacios.
        Crucial: El wrapper debe retornar un Dict space con keys 'obs' y 'state'.
        """
        temp_env = MAPPOEnvWrapper(self.env_config)
        try:
            temp_env.reset()

            rl_module_specs_dict = {}

            for agent_id, obs_space in temp_env.observation_spaces.items():
                act_space = temp_env.action_spaces[agent_id]
                policy_id = self.policy_mapping_fn(agent_id, None)

                # Definición de Policies
                if policy_id not in self.policies:
                    self.policies[policy_id] = PolicySpec(
                        observation_space=obs_space, action_space=act_space, config={}
                    )

                # Definición de RLModules (Específico MAPPO)
                if policy_id not in rl_module_specs_dict:
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
                temp_env.env.env.close()

    def _build_algorithm_config(self):
        hyperparams = self.exp_config["hyperparameters"]
        num_gpus = 1 if torch.cuda.is_available() else 0

        config = (
            PPOConfig()
            .training(
                gamma=hyperparams["gamma"],
                clip_param=hyperparams["clip_param"],
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
            .environment(env=MAPPOEnvWrapper, env_config=self.env_config)
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
            # Reincorporo la lógica de evaluación que tenías en IPPO
            # Es inconsistente tenerla en uno y no en el otro si es para comparar.
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
            .callbacks([PPOMetricsLogger, CentralizedCallback])
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

    def train(self) -> Path:
        if not ray.is_initialized():
            ray.init(
                # log_to_driver=False,
                ignore_reinit_error=True,
                # logging_level=logging.ERROR,
            )

        print("Building MAPPO Algorithm...")
        algo_config = self._build_algorithm_config()

        # Para tensorboard
        def logger_creator(config):
            log_dir = self.exp_dir / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            return UnifiedLogger(config, str(log_dir), loggers=None)

        algo = algo_config.build(logger_creator=logger_creator)

        # Checkpoint inicial
        save_dir = self.exp_dir / "checkpoints"
        algo.save_to_path(save_dir / "0")

        print("Starting MAPPO training loop")
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
