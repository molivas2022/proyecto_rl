import pandas as pd
from pathlib import Path
from ray.rllib.callbacks.callbacks import RLlibCallback
from pprint import pprint


# TODO: Agregar on_algo_end (creo que se debe llamar algo asi), por ahora simplemente puse frequencia de guardado 1.
# TODO: Dejar de medir métricas en base a training_iteration y hacerlo en base a time_steps


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
        self.save_path = None
        self.save_frequency = None

        self.calls_since_last_save = 0

    def on_algorithm_init(self, *, algorithm, **kwargs):
        custom_args = algorithm.config.get("callback_args", {})

        self.save_path = custom_args.get("exp_dir")
        self.save_frequency = custom_args.get("log_save_frequency")

        print(f"Callback initialized with: {self.save_path}, {self.save_frequency}")

    def update_reward_data(self, total_steps, reward_mean):
        new_row = pd.Series(
            {"training_iteration": total_steps, "reward_mean": reward_mean}
        )
        self.rewards_df = pd.concat([self.rewards_df, new_row.to_frame().T])

    def update_policy_data(self, total_steps, policy_data):
        for policy in policy_data.keys():
            logs = policy_data[policy]
            data = {key: logs[key] for key in self.policy_data_to_log}
            data["training_iteration"] = total_steps
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
        total_steps = result.get(
            "num_env_steps_sampled_lifetime",
            result["env_runners"].get("num_env_steps_sampled_lifetime"),
        )
        policies = set(result["config"]["policies"].keys())
        policy_data = {}

        current_lr = result["learners"]["policy_0"].get(
            "default_optimizer_learning_rate"
        )
        # print(current_lr)

        reward_mean = result["env_runners"]["episode_return_mean"] / len(policies)
        print(
            f"Iteration: {training_iteration}, total steps: {total_steps}, finished with reward: {reward_mean}"
        )

        for policy, logs in result["learners"].items():
            if policy in policies:
                policy_data[policy] = logs

        self.update_reward_data(total_steps, reward_mean)
        self.update_policy_data(total_steps, policy_data)
        self.calls_since_last_save += 1
        self.save_data()
