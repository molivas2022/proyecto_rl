import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env import MultiAgentEnv
from metadrive import (
    MultiAgentIntersectionEnv,
)  # probemos mientras solo con intersección
from pprint import pprint  # imprimir eedd con formato
from ray.tune import register_env
from metadrive_wrapper import MetadriveEnvWrapper
import re
import torch
import yaml
from pathlib import Path
import warnings
from plot import plot_reward_curve
from gif_generator import generate_gif
from ray.rllib.callbacks.callbacks import RLlibCallback
import pandas as pd


warnings.filterwarnings("ignore")
current_dir = Path.cwd()

EXP_DIR = current_dir / "experimentos" / "exp2"

# Guardamos logs en memoria persistente cada 10 calls de logger (en este caso, cada 10 iteraciones de training)
LOG_SAVE_FREQUENCY = 1

# TODO: Agregar on_algo_end (creo que se debe llamar algo asi), por ahora simplemente puse frequencia de guardado 1.
class PPOMetricsLogger(RLlibCallback):
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

        self.save_path = EXP_DIR
        self.save_frequency = LOG_SAVE_FREQUENCY
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

        reward_mean = result["env_runners"]["episode_return_mean"]
        print(f"Iteration {training_iteration} finished, with reward: {reward_mean}")

        for policy, logs in result["learners"].items():
            if policy in policies:
                policy_data[policy] = logs

        self.update_reward_data(training_iteration, reward_mean)
        self.update_policy_data(training_iteration, policy_data)
        self.calls_since_last_save += 1
        self.save_data()


class IPPOExperiment:
    def __init__(self, exp_config, env_class, exp_dir):
        self.exp_dir = exp_dir
        self.exp_config = exp_config.copy()
        self.env_config = {}
        self.env_config["base_env_class"] = env_class
        self.env_config["base_env_class"] = env_class
        self.env_config["num_agents"] = self.exp_config["environment"]["num_agents"]
        self.env_config["allow_respawn"] = self.exp_config["environment"][
            "allow_respawn"
        ]
        self.env_config["horizon"] = self.exp_config["environment"]["horizon"]

        self.policies = {}

        def policy_mapping_fn(agent_id, episode, **kwargs):
            match = re.search(r"(\d+)", str(agent_id))
            if match:
                return f"policy_{match.group(1)}"
            else:
                return f"policy_{agent_id}"

        self.policy_mapping_fn = policy_mapping_fn

        temp_env = MetadriveEnvWrapper(self.env_config)
        temp_env.reset()

        for agent_id, obs_space in temp_env.observation_spaces.items():
            act_space = temp_env.action_spaces[agent_id]
            policy_id = self.policy_mapping_fn(agent_id, None)

            if policy_id not in self.policies:
                self.policies[policy_id] = PolicySpec(
                    observation_space=obs_space, action_space=act_space, config={}
                )

        if "temp_env" in locals() and hasattr(temp_env, "env"):
            temp_env.env.close()
        del temp_env

    def train(self):
        ray.init(ignore_reinit_error=True)
        config = (
            PPOConfig()
            .training(
                # lr=self.exp_config["hyperparameters"]["learning_rate"],
                gamma=self.exp_config["hyperparameters"]["gamma"],
                clip_param=self.exp_config["hyperparameters"]["clip_param"],
                lr=[
                    [0, self.exp_config["hyperparameters"]["learning_rate"]],
                    [15000, self.exp_config["hyperparameters"]["learning_rate"] / 10],
                ],
            )
            .multi_agent(
                policies=self.policies, policy_mapping_fn=self.policy_mapping_fn
            )
            .environment(env=MetadriveEnvWrapper, env_config=self.env_config)
            .framework("torch")
            .resources(num_gpus=1)
            .env_runners(
                num_env_runners=self.exp_config["environment"]["num_env_runners"]
            )
            .callbacks(PPOMetricsLogger)
        )

        algo = config.build()

        print("Commencing training")
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


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA GPU: {torch.cuda.get_device_name()}")


with open(EXP_DIR / "exp.yaml") as f:
    exp_config = yaml.load(f, Loader=yaml.SafeLoader)

exp = IPPOExperiment(exp_config, MultiAgentIntersectionEnv, EXP_DIR)
_ = exp.train()

# plot_reward_curve(rewards_csv_dir, exp_dir / "resultados.png")
