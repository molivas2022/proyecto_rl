import pandas as pd
from pathlib import Path
from ray.rllib.callbacks.callbacks import RLlibCallback
from pprint import pprint
import numpy as np


# TODO: Agregar on_algo_end (creo que se debe llamar algo asi), por ahora simplemente puse frequencia de guardado 0 mod numero de iteraciones


class PPOMetricsLogger(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.rewards_df = pd.DataFrame(
            columns=[
                "training_iteration",
                "total_steps",
                "reward_mean",
                "reward_raw_mean",
                "lr",
            ]
        )

        self.policy_data_to_log = [
            "mean_kl_loss",
            "policy_loss",
            "vf_loss",
            "total_loss",
        ]
        self.policy_data_df = pd.DataFrame(
            columns=["total_steps", "policy"] + self.policy_data_to_log
        )

        # esto es para las métricas
        self.policy_data_df = pd.DataFrame(
            columns=["training_iteration", "policy"] + self.policy_data_to_log
        )

        self.eval_metrics_cols = [
            "avg_velocity",
            "avg_acceleration",
            "avg_route_completion",
            "total_energy"
        ]

        self.eval_data_df = pd.DataFrame()

        self._episode_storage = {}
        self.save_path = None
        self.save_frequency = None
        self.calls_since_last_save = 0

    def on_algorithm_init(self, *, algorithm, **kwargs):
        custom_args = algorithm.config.get("callback_args", {})
        self.save_path = custom_args.get("exp_dir")
        self.save_frequency = custom_args.get("log_save_frequency")
        print(f"Callback initialized with: {self.save_path}, {self.save_frequency}")

    def on_episode_start(self, *, episode, **kwargs):
        # En la nueva API, episode.id_ es la propiedad correcta
        self._episode_storage[episode.id_] = {
            "velocity": [],
            "acceleration": [],
            "energy": [],
            "route_completion": []
        }

    def on_episode_step(self, *, episode, **kwargs):
        if episode.id_ not in self._episode_storage:
            self._episode_storage[episode.id_] = {
                "velocity": [],
                "acceleration": [],
                "energy": [],
                "route_completion": []
            }

        storage = self._episode_storage[episode.id_]

        for agent_id in episode.get_agents_that_stepped():
            # episodio del agente individual
            sa_episode = episode.agent_episodes[agent_id]
            
            # última info disponible
            infos = sa_episode.get_infos()
            info = infos[-1] if infos else {}

            storage["velocity"].append(float(info.get("velocity", 0.0)))
            storage["acceleration"].append(float(info.get("acceleration", 0.0)))
            storage["energy"].append(float(info.get("step_energy", 0.0)))
            storage["route_completion"].append(float(info.get("route_completion", 0.0)))


    def on_episode_end(self, *, episode, env_runner, metrics_logger, **kwargs):
        """
        Captura los retornos raw de la info del entorno, cuando un agente termina un episodio.
        """
        # return de recompensasraw
        last_infos = episode.get_infos(indices=-1)

        for agent_id in episode.agent_ids:
            if agent_id in last_infos:
                last_info = last_infos[agent_id]

                if "episode_return_raw" in last_info:
                    val = last_info["episode_return_raw"]
                    metrics_logger.log_value("episode_return_raw", val)
        
        # logs de métricas
        if episode.id_ not in self._episode_storage:
            return

        storage = self._episode_storage[episode.id_]
        vels = storage["velocity"]
        accs = storage["acceleration"]
        energies = storage["energy"]
        routes = storage["route_completion"]

        avg_velocity = np.mean(vels) if vels else 0.0
        avg_acceleration = np.mean(accs) if accs else 0.0
        total_energy = np.sum(energies)
        avg_route_completion = np.mean(routes) if routes else 0.0

        if metrics_logger:
            # Registramos las métricas
            metrics_logger.log_value("avg_velocity", avg_velocity)
            metrics_logger.log_value("avg_acceleration", avg_acceleration)
            metrics_logger.log_value("total_energy", total_energy)
            metrics_logger.log_value("avg_route_completion", avg_route_completion)

        # Limpieza de memoria
        del self._episode_storage[episode.id_]

    def update_reward_data(
        self, training_iteration, total_steps, reward_mean, reward_raw_mean, lr
    ):
        new_row = pd.Series(
            {
                "training_iteration": training_iteration,
                "total_steps": total_steps,
                "reward_mean": reward_mean,
                "reward_raw_mean": reward_raw_mean,
                "lr": lr,
            }
        )
        self.rewards_df = pd.concat([self.rewards_df, new_row.to_frame().T])

    def update_policy_data(self, training_iteration, total_steps, policy_data):
        for policy in policy_data.keys():
            logs = policy_data[policy]
            data = {key: logs[key] for key in self.policy_data_to_log}
            data["training_iteration"] = training_iteration
            data["total_steps"] = total_steps
            data["policy"] = policy
            new_row = pd.Series(data)
            self.policy_data_df = pd.concat([self.policy_data_df, new_row.to_frame().T])

    def save_data(self):
        if self.calls_since_last_save < self.save_frequency:
            return

        self.rewards_df.to_csv(self.save_path / "rewards_log.csv", index=False)
        self.policy_data_df.to_csv(self.save_path / "policy_log.csv", index=False)

        if not self.eval_data_df.empty:
            self.eval_data_df.to_csv(self.save_path / "evaluation_log.csv", index=False)
        self.calls_since_last_save = 0

    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        training_iteration = result["training_iteration"]

        # 1. Steps totales
        total_steps = result.get("num_env_steps_sampled_lifetime")
        if total_steps is None:
            total_steps = result.get("env_runners", {}).get(
                "num_env_steps_sampled_lifetime"
            )
        if total_steps is None:
            total_steps = 0

        policies = set(result["config"]["policies"].keys())
        policy_data = {}

        # 2. Learning Rate
        current_lr = 0.0
        learners = result.get("learners", {})
        if "policy_0" in learners:
            current_lr = learners["policy_0"].get(
                "default_optimizer_learning_rate", 0.0
            )

        env_runners_metrics = result.get("env_runners", {})

        reward_raw_mean = env_runners_metrics.get("episode_return_raw", None)

        # Recompesas normalizadas
        episode_return_mean = env_runners_metrics.get("episode_return_mean", 0.0)
        num_policies = (
            len(result["config"]["policies"])
            if len(result["config"]["policies"]) > 0
            else 1
        )
        reward_mean = episode_return_mean / num_policies

        # No hay normalización, entonces reward_mean es en verdad reward_raw
        if reward_raw_mean == None:
            reward_raw_mean = reward_mean

        print(
            f"Iter: {training_iteration} | Steps: {total_steps} | "
            f"Rew(Norm): {reward_mean:.2f} | Rew(Raw): {reward_raw_mean:.2f} | LR: {current_lr:.2e}"
        )

        for policy, logs in learners.items():
            if policy in policies:
                policy_data[policy] = logs

        self.update_reward_data(
            training_iteration, total_steps, reward_mean, reward_raw_mean, current_lr
        )
        self.update_policy_data(training_iteration, total_steps, policy_data)
        self.update_eval_data(total_steps, result)
        self.calls_since_last_save += 1
        self.save_data()

    def update_eval_data(self, total_steps, result):
        # Buscar métricas personalizadas en la ubicación estándar del nuevo stack
        # Usualmente bajo env_runners si se usó metrics_logger
        source = result.get("env_runners", {})
        
        # Si es evaluación, a veces está en 'evaluation'
        if "evaluation" in result:
            source = result["evaluation"].get("env_runners", source)

        row_data = {"training_iteration": total_steps}
        has_data = False
        
        for col in self.eval_metrics_cols:
            # metrics_logger suele no agregar '_mean' si usas log_value directamente,
            # pero depende de la configuración de windowing. RLlib suele promediar automáticamente.
# Verificamos ambas llaves.
            val = source.get(col)
            if val is None:
                val = source.get(f"{col}_mean") # Fallback
            
            if val is not None:
                row_data[col] = val
                has_data = True

        if has_data:
            self.eval_data_df = pd.concat(
                [self.eval_data_df, pd.DataFrame([row_data])], 
                ignore_index=True
            )
