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
                lr=self.exp_config["hyperparameters"]["learning_rate"],
                gamma=self.exp_config["hyperparameters"]["gamma"],
                clip_param=self.exp_config["hyperparameters"]["clip_param"],
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
        )

        algo = config.build()
        rewards = []
        rewards_csv_dir = self.exp_dir / "rewards_log.csv"

        print("Commencing training")
        for i in range(self.exp_config["hyperparameters"]["n_epochs"]):
            result = algo.train()
            reward = (
                result["env_runners"]["episode_return_mean"]
                / self.env_config["num_agents"]
            )
            print(f"Iteration: {i + 1} finished, with reward: {reward}")
            # Dividir por número de agentes, esto es cosa de gustos en verdad.
            rewards.append(reward)

            if (i + 1) % self.exp_config["experiment"]["checkpoint_freq"] == 0:
                pd.DataFrame(
                    {"Iteration": range(1, len(rewards) + 1), "Avg reward per episode": rewards}
                ).to_csv(rewards_csv_dir, index=False)

                checkpoint_dir = algo.save_to_path(
                    self.exp_dir / "checkpoints" / f"{i + 1}"
                )

        print("Training complete, saving final state and logs")
        pd.DataFrame(
            {"Iteration": range(1, len(rewards) + 1), "Avg reward per episode": rewards}
        ).to_csv(rewards_csv_dir, index=False)
        self.final_checkpoint_dir = algo.save_to_path(
            self.exp_dir / "checkpoints" / "final"
        )

        algo.stop()
        ray.shutdown()

        return self.final_checkpoint_dir, rewards_csv_dir

    def evaluate(self, checkpoint_path, num_episodes=10):
        ray.init(ignore_reinit_error=True)


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA GPU: {torch.cuda.get_device_name()}")


exp_dir = current_dir / "experimentos" / "exp1"

with open(exp_dir / "exp1.yaml") as f:
    exp_config = yaml.load(f, Loader=yaml.SafeLoader)

exp = IPPOExperiment(exp_config, MultiAgentIntersectionEnv, exp_dir)
_, rewards_csv_dir = exp.train()

plot_reward_curve(rewards_csv_dir, exp_dir / "resultados.png")
