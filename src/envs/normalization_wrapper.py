import numpy as np
import gymnasium as gym


class RunningMeanStd:
    """Lleva la media y varianza usando el algoritmo de Welford."""

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = 1 if np.isscalar(x) else x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        return new_mean, new_var, tot_count


class MetadriveMARLNormalizedRewardEnv(gym.Wrapper):
    """Wrapper para normalizar los rewards en un entorno MARL de MetaDrive"""

    def __init__(self, env, gamma):
        super().__init__(env)

        self.gamma = gamma
        self.epsilon = 1e-8
        # Estimador de varianza compartido
        self.return_rms = RunningMeanStd(shape=())
        # Trackea los retornos individuales de los agentes
        self.agents_returns = {}

    def step(self, actions):
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

        normalized_rewards = {}

        # 1. Actualizamos y normalizamos
        for agent_id, reward in rewards.items():
            if agent_id not in self.agents_returns:
                self.agents_returns[agent_id] = 0.0

            self.agents_returns[agent_id] = (
                self.agents_returns[agent_id] * self.gamma + reward
            )
            self.return_rms.update(np.array([self.agents_returns[agent_id]]))

            normalized_rewards[agent_id] = reward / np.sqrt(
                self.return_rms.var + self.epsilon
            )

            if agent_id not in infos:
                infos[agent_id] = {}

        # 2. Manejamos los agentes que han terminado
        done_agents = set(terminateds.keys()) | set(truncateds.keys())

        for agent_id in done_agents:
            if agent_id == "__all__":
                continue

            is_done = terminateds.get(agent_id, False) or truncateds.get(
                agent_id, False
            )

            # Si el agente ha terminado...
            if is_done:
                # A. Logeamos el reward raw
                if agent_id in infos:
                    infos[agent_id]["episode_return_raw"] = self.agents_returns.get(
                        agent_id, 0.0
                    )

                # B. Lo reseteamos para el siguiente episodio
                self.agents_returns[agent_id] = 0.0

        return obs, normalized_rewards, terminateds, truncateds, infos

    def reset(self, **kwargs):
        self.agents_returns = {}
        return self.env.reset(**kwargs)
