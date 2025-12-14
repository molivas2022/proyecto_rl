from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from metadrive import MetaDriveEnv


class MetadriveEnvWrapper(MultiAgentEnv):
    """
        Importante acerca de ambiente:
        debe tener respawn desactivado, si terminated o truncated == True para todos
        termina el episodio.
    """

    def __init__(self, config=None):
        """
        - env: environment de metadrive
        """
        super().__init__()

        config_copy = config.copy()

        # Clase base (MultiAgentIntersectionEnv, etc.)
        BaseEnvClass = config_copy.pop("base_env_class", None)

        # 🔹 LEEMOS, PERO NO POPPEAMOS
        self.start_seed = config_copy.get("start_seed", 0)
        self.num_scenarios = config_copy.get("num_scenarios", 1)
        self.current_seed = self.start_seed

        # map_pieces → map (por si lo usas con PG)
        if "map_pieces" in config_copy:
            config_copy["map"] = config_copy.pop("map_pieces")

        # Ahora MetaDrive recibe start_seed y num_scenarios en su config
        self.env = BaseEnvClass(config_copy)

        self.possible_agents = list(self.env.observation_space.keys())
        self.observation_spaces = self.env.observation_space
        self.action_spaces = self.env.action_space
        self.agents = []

    def reset(self, *, seed=None, options=None):
        # RLlib rara vez pasa seed, así que normalmente usamos nuestro contador
        if seed is not None:
            next_seed = seed
        else:
            next_seed = self.current_seed
            self.current_seed += 1
            if self.current_seed >= self.start_seed + self.num_scenarios:
                self.current_seed = self.start_seed

        # DEBUG (si quieres)
        print(f"[Wrapper] reset with seed={next_seed}, range=[{self.start_seed}, {self.start_seed + self.num_scenarios})")

        obs, info = self.env.reset(seed=next_seed)
        self.agents = list(obs.keys())
        return obs, info

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = self.env.step(action_dict)

        if terminated["__all__"] or truncated["__all__"]:
            self.agents = []
        else:
            self.agents = list(obs.keys())

        return obs, rew, terminated, truncated, info
