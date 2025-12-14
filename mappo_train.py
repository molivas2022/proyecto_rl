from pathlib import Path
current_dir = Path.cwd()

### ARGUMENTOS

EXP_DIR = current_dir / "experimentos" / "mappoexp"
LOG_SAVE_FREQUENCY = 5

### TODOS LOS IMPORTS.

from mappo_wrapper import MAPPOEnvWrapper
from mappo_module import MAPPOTorchRLModule

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.callbacks.callbacks import RLlibCallback
import pandas as pd
from metadrive import (
    MultiAgentIntersectionEnv,
)  # probemos mientras solo con intersección
import re
import yaml
import warnings
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

warnings.filterwarnings("ignore")
current_dir = Path.cwd()

# TODO: Mover despues del refactor
# TODO: Agregar on_algo_end (creo que se debe llamar algo asi), por ahora simplemente puse frequencia de guardado 1.
# TODO: Dejar de medir métricas en base a training_iteration y hacerlo en base a time_steps
class PPOMetricsLogger(RLlibCallback):
    
    def on_algorithm_init(self, *, algorithm, **kwargs):
        custom_args = algorithm.config.get("callback_args", {})
        
        self.save_path = custom_args.get("exp_dir")
        self.save_frequency = custom_args.get("log_save_frequency")
        
        print(f"Callback initialized with: {self.save_path}, {self.save_frequency}")
        
    def __init__(self):
        super().__init__()
        self.rewards_df = pd.DataFrame(columns=["training_iteration", "reward_mean"])
        self.policy_data_to_log = [
            "mean_kl_loss",
            "policy_loss",
            "vf_loss",
            "total_loss",
        ]
        self.policy_data_df = pd.DataFrame(
            columns=["training_iteration", "policy"] + self.policy_data_to_log
        )

        self.calls_since_last_save = 0

    def update_reward_data(self, training_iteration, reward_mean):
        new_row = pd.Series(
            {"training_iteration": training_iteration, "reward_mean": reward_mean}
        )
        self.rewards_df = pd.concat([self.rewards_df, new_row.to_frame().T])

    def update_policy_data(self, training_iteration, policy_data):
        for policy in policy_data.keys():
            logs = policy_data[policy]
            data = {key: logs[key] for key in self.policy_data_to_log}
            data["training_iteration"] = training_iteration
            data["policy"] = policy
            new_row = pd.Series(data)
            self.policy_data_df = pd.concat([self.policy_data_df, new_row.to_frame().T])

    def save_data(self):
        if self.calls_since_last_save < self.save_frequency:
            return

        self.rewards_df.to_csv(self.save_path / "rewards_log.csv", index=False)
        self.policy_data_df.to_csv(self.save_path / "policy_log.csv", index=False)
        self.calls_since_last_save = 0

    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        training_iteration = result["training_iteration"]
        policies = set(result["config"]["policies"].keys())
        policy_data = {}

        reward_mean = result["env_runners"]["episode_return_mean"] / len(policies)
        print(f"Iteration {training_iteration} finished, with reward: {reward_mean}")

        for policy, logs in result["learners"].items():
            if policy in policies:
                policy_data[policy] = logs

        self.update_reward_data(training_iteration, reward_mean)
        self.update_policy_data(training_iteration, policy_data)
        self.calls_since_last_save += 1
        self.save_data()

### EXPERIMENT
# TODO: start for checkpoint
# TODO: cnn
class MAPPOExperiment:

    def __init__(self, exp_config, env_class, exp_dir):

        self.exp_dir = exp_dir
        self.exp_config = exp_config.copy()
        
        ## CONFIGURACION DEL ENTORNO

        self.env_config = {}
        self.env_config["base_env_class"] = env_class
        self.env_config["num_agents"] = exp_config["environment"]["num_agents"]
        self.env_config["allow_respawn"] = exp_config["environment"]["allow_respawn"]
        self.env_config["horizon"] = exp_config["environment"]["horizon"]
        self.env_config["traffic_density"] = exp_config["environment"]["traffic_density"]

        ## CONFIGURACION DE LAS POLICIES.
        
        self.policies = {}
        rl_module_specs_dict = {} # Mapeo en politica_id -> la politica como tal
        def policy_mapping_fn(agent_id, episode, **kwargs):
            match = re.search(r"(\d+)", str(agent_id))
            if match:
                return f"policy_{match.group(1)}"
            else:
                return f"policy_{agent_id}"
        self.policy_mapping_fn = policy_mapping_fn

        # antes: obs[agent_id] -> agent_obs
        # despues new_obs[agent_id] -> {"obs": agent_obs, "state": global_state}
        # acciones siguen iguales
        temp_env = MAPPOEnvWrapper(self.env_config)
        temp_env.reset()

        # Exactamente igual que original
        for agent_id, obs_space in temp_env.observation_spaces.items():
            act_space = temp_env.action_spaces[agent_id]
            policy_id = self.policy_mapping_fn(agent_id, None)

            if policy_id not in self.policies:
                self.policies[policy_id] = PolicySpec(
                    observation_space=obs_space, action_space=act_space, config={}
                )

            if policy_id not in rl_module_specs_dict:
                rl_module_specs_dict[policy_id] = RLModuleSpec(
                    module_class=MAPPOTorchRLModule,
                    observation_space=obs_space,
                    action_space=act_space,
                    model_config={"hidden_dim": 256},
                )
        
        """
        RLModule: define la politica de un agente, es decir, arquitectura, pesos, hiperparametros
        MultiRLModule: muchas politicas, por afuera debe haber un mapeo de agentes a politicas
        MultiRLModuleSpec: configuracion
        """
        self.spec = MultiRLModuleSpec(rl_module_specs=rl_module_specs_dict)

        if "temp_env" in locals() and hasattr(temp_env, "env"):
            temp_env.env.close()
        del temp_env
    
    def train(self):

        ray.init(ignore_reinit_error=True)

        # Exactamente igual que original
        config = (
            PPOConfig()
            .training(
                gamma=self.exp_config["hyperparameters"]["gamma"],
                clip_param=self.exp_config["hyperparameters"]["clip_param"],
                lr=[
                    [0, self.exp_config["hyperparameters"]["learning_rate"]],
                    [8e5, self.exp_config["hyperparameters"]["learning_rate"]],
                    [4e6, self.exp_config["hyperparameters"]["learning_rate"]],
                ],
                entropy_coeff=[
                    [0, self.exp_config["hyperparameters"]["entropy_coeff"]],
                    [8e5, self.exp_config["hyperparameters"]["entropy_coeff"] / 2],
                    [4e6, self.exp_config["hyperparameters"]["entropy_coeff"] / 5],
                ],
                lambda_=self.exp_config["hyperparameters"]["lambda"],
                train_batch_size_per_learner=self.exp_config["hyperparameters"]["train_batch_size"],
                minibatch_size=self.exp_config["hyperparameters"]["minibatch_size"],
                vf_clip_param=self.exp_config["hyperparameters"]["vf_clip_param"],
                grad_clip=self.exp_config["hyperparameters"]["grad_clip"]
            )
            .multi_agent(
                policies=self.policies, policy_mapping_fn=self.policy_mapping_fn
            )
            .environment(env=MAPPOEnvWrapper, env_config=self.env_config)
            .framework("torch")
            .resources(num_gpus=1)
            .env_runners(
                num_env_runners=self.exp_config["environment"]["num_env_runners"],
                # Prefiero no cambiar esto (por ahora) la verdad
                # rollout_fragment_length=self.exp_config["hyperparameters"]["rollout_fragment_length"],
            )
            .callbacks(PPOMetricsLogger)
            .update_from_dict({
                "callback_args": {
                    "exp_dir": EXP_DIR,
                    "log_save_frequency": LOG_SAVE_FREQUENCY
                }
            })
            .rl_module(
                rl_module_spec=self.spec,
            )
        )

        algo = config.build()

        print("Commencing MAPPO training")
        for i in range(self.exp_config["hyperparameters"]["n_epochs"]):
            result = algo.train()

            if (i + 1) % self.exp_config["experiment"]["checkpoint_freq"] == 0:
                checkpoint_dir = algo.save_to_path(
                    self.exp_dir / "checkpoints" / f"{i + 1}"
                )

        print("Training complete, saving final state and logs")
        self.final_checkpoint_dir = algo.save_to_path(
            self.exp_dir / "checkpoints" / "final"
        )

        algo.stop()
        ray.shutdown()

        return self.final_checkpoint_dir

with open(EXP_DIR / "exp.yaml") as f:
    exp_config = yaml.load(f, Loader=yaml.SafeLoader)

exp = MAPPOExperiment(exp_config, MultiAgentIntersectionEnv, EXP_DIR)
_ = exp.train()