import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
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
from utils import transfer_module_weights
from custom_networks import MetaDriveCNN
from custom_networks import MetaDriveStackedCNN
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from observation import StackedLidarObservation


warnings.filterwarnings("ignore")
current_dir = Path.cwd()

EXP_DIR = current_dir / "experimentos" / "exp8"

# Guardamos logs en memoria persistente cada 10 calls de logger (en este caso, cada 10 iteraciones de training)
LOG_SAVE_FREQUENCY = 10


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

        reward_mean = result["env_runners"]["episode_return_mean"] / len(policies)
        print(f"Iteration {training_iteration} finished, with reward: {reward_mean}")

        for policy, logs in result["learners"].items():
            if policy in policies:
                policy_data[policy] = logs

        self.update_reward_data(training_iteration, reward_mean)
        self.update_policy_data(training_iteration, policy_data)
        self.calls_since_last_save += 1
        self.save_data()


class IPPOExperiment:
    def __init__(
        self, exp_config, env_class, exp_dir, start_from_checkpoint=False, use_cnn=False
    ):
        self.exp_dir = exp_dir
        self.exp_config = exp_config.copy()
        self.env_config = {}

        # TODO: Dejar de hardcodear todos los fields del env_config
        self.env_config["base_env_class"] = env_class
        self.env_config["num_agents"] = self.exp_config["environment"]["num_agents"]
        self.env_config["allow_respawn"] = self.exp_config["environment"][
            "allow_respawn"
        ]
        self.env_config["horizon"] = self.exp_config["environment"]["horizon"]
        self.env_config["traffic_density"] = self.exp_config["environment"][
            "traffic_density"
        ]

        # Por ahora hardcodeado
        # self.env_config["agent_observation"] = StackedLidarObservation

        self.start_from_checkpoint = start_from_checkpoint

        self.policies = {}

        self.policy_specs = {}
        rl_module_specs_dict = {}

        def policy_mapping_fn(agent_id, episode, **kwargs):
            match = re.search(r"(\d+)", str(agent_id))
            if match:
                return f"policy_{match.group(1)}"
            else:
                return f"policy_{agent_id}"

        self.policy_mapping_fn = policy_mapping_fn

        temp_env = MetadriveEnvWrapper(self.env_config)
        temp_env.reset()

        # Número de lasers
        # print(temp_env.env.config["vehicle_config"]["lidar"])

        # self.spec = RLModuleSpec(
        #        observation_space=temp_env.observation_spaces["agent0"],
        #        action_space=temp_env.action_spaces["agent0"],
        #        module_class=MetaDriveCNN,
        #        model_config={"hidden_dim": 256},
        #    )
        # test_module = self.spec.build()
        # print("########### HELLO ###############")
        # print(sum(p.numel() for p in test_module.parameters() if p.requires_grad))
        # print("########### BYE #################")

        for agent_id, obs_space in temp_env.observation_spaces.items():
            act_space = temp_env.action_spaces[agent_id]
            policy_id = self.policy_mapping_fn(agent_id, None)

            if policy_id not in self.policies:
                self.policies[policy_id] = PolicySpec(
                    observation_space=obs_space, action_space=act_space, config={}
                )

            # TODO: Generalizar esto, quizás pasar un enum al constructor con la red a utilizar.
            if policy_id not in rl_module_specs_dict:
                if use_cnn:
                    rl_module_specs_dict[policy_id] = RLModuleSpec(
                        module_class=MetaDriveCNN,
                        observation_space=obs_space,  #
                        action_space=act_space,  #
                        model_config={"hidden_dim": 256},
                    )
                else:
                    rl_module_specs_dict[policy_id] = RLModuleSpec(
                        module_class=PPOTorchRLModule,
                        observation_space=obs_space,
                        action_space=act_space,
                        model_config={"hidden_dim": 256},
                    )
        self.spec = MultiRLModuleSpec(rl_module_specs=rl_module_specs_dict)

        if "temp_env" in locals() and hasattr(temp_env, "env"):
            temp_env.env.close()
        del temp_env

    def train(self):
        ray.init(ignore_reinit_error=True)
        # TODO: Refactorizar esto para que no esté hardcodeado
        config = (
            PPOConfig()
            .training(
                gamma=self.exp_config["hyperparameters"]["gamma"],
                clip_param=self.exp_config["hyperparameters"]["clip_param"],
                # Se samplean train_batch_size env steps por training iteration. El default es 4000
                # Tener esto en cuenta. En la config los estoy dejando las reducciones con target
                # en epochs 200, 1000.
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
                train_batch_size_per_learner=self.exp_config["hyperparameters"][
                    "train_batch_size"
                ],
                minibatch_size=self.exp_config["hyperparameters"]["minibatch_size"],
                vf_clip_param=self.exp_config["hyperparameters"]["vf_clip_param"],
                grad_clip=self.exp_config["hyperparameters"]["grad_clip"],
            )
            .multi_agent(
                policies=self.policies, policy_mapping_fn=self.policy_mapping_fn
            )
            .environment(env=MetadriveEnvWrapper, env_config=self.env_config)
            .framework("torch")
            .resources(num_gpus=1)
            .env_runners(
                num_env_runners=self.exp_config["environment"]["num_env_runners"],
                # Prefiero no cambiar esto (por ahora) la verdad
                # rollout_fragment_length=self.exp_config["hyperparameters"]["rollout_fragment_length"],
            )
            .callbacks(PPOMetricsLogger)
            .rl_module(
                rl_module_spec=self.spec,
            )
        )

        algo = config.build()

        # Si empezamos de un checkpoint, cargamos los parámetros de los RLModules, no usar from_checkpoint directamente, eso NO nos gusta!
        if self.start_from_checkpoint:

            source_algo = Algorithm.from_checkpoint(self.exp_dir / "base_checkpoint")
            module_source = source_algo.get_module("policy_0")
            # print("### Source module state ###")
            # pprint(module_source.get_state())

            # Si hay mismatch entre cantidad de políticas, definimos un idol arbitrario.
            idol = None
            if len(self.policies) != len(source_algo.config.policies):
                idol = "policy_0"

            transfer_module_weights(source_algo, algo, self.policies, idol)
            source_algo.stop()

            # module_now = algo.get_module("policy_0")
            # print("### After transfer target module state ###")
            # pprint(module_now.get_state())

        # "Checkpoint" inicial para debugging, más que nada
        checkpoint_dir = algo.save_to_path(self.exp_dir / "checkpoints" / "0")

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

exp = IPPOExperiment(exp_config, MultiAgentIntersectionEnv, EXP_DIR, False, False)
_ = exp.train()
