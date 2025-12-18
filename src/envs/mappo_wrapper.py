import numpy as np
import gymnasium as gym

# Import your existing utilities
from .metadrive_env_wrapper import MetadriveEnvWrapper


# --- 1. Wrapper MAPPO para obs global ---
class MAPPOEnvWrapper(gym.Wrapper):
    """
    Wraps MetadriveEnvWrapper to provide a Dict observation:
    {
        "obs": Local Observation (for the Actor/Policy),
        "state": Global State (Concatenated obs of all agents for the Critic)
    }
    """

    def __init__(self, config):
        env = MetadriveEnvWrapper(config)
        super().__init__(env)
        self.env = env
        self.num_agents = config["num_agents"]

        # Determine shapes
        # We assume all agents have the same observation space
        sample_agent = list(self.env.observation_spaces.keys())[0]
        self.local_obs_space = self.env.observation_spaces[sample_agent]
        self.local_obs_dim = np.prod(self.local_obs_space.shape)

        # Global state = concatenation of all agents
        self.global_state_dim = self.num_agents * self.local_obs_dim

        # Define the new Dict observation space for RLlib
        self.observation_spaces = {}
        for agent_id, obs_space in self.env.observation_spaces.items():
            self.observation_spaces[agent_id] = gym.spaces.Dict(
                {
                    "obs": obs_space,
                    "state": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.global_state_dim,),
                        dtype=np.float32,
                    ),
                }
            )
        # Update action spaces just in case (usually unchanged)
        self.action_spaces = self.env.action_spaces

    def _get_global_state(self, obs_dict):
        state_list = []
        # We iterate by index to ensure consistent order (agent0, agent1, ...)
        # If an agent is missing (not spawned or dead), we pad with zeros.
        for i in range(self.num_agents):
            key = f"agent{i}"
            if key in obs_dict:
                flat_obs = obs_dict[key].flatten()
            else:
                flat_obs = np.zeros(self.local_obs_dim, dtype=np.float32)
            state_list.append(flat_obs)

        return np.concatenate(state_list).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        global_state = self._get_global_state(obs)

        new_obs = {}
        for agent_id, agent_obs in obs.items():
            new_obs[agent_id] = {"obs": agent_obs, "state": global_state}
        return new_obs, info

    def step(self, action_dict):
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)
        global_state = self._get_global_state(obs)

        new_obs = {}
        for agent_id, agent_obs in obs.items():
            new_obs[agent_id] = {"obs": agent_obs, "state": global_state}
        return new_obs, rewards, terminated, truncated, info
