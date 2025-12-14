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

        # Copiar config para evitar modificar original
        config_copy = config.copy()

        # Obtener la clase custom
        BaseEnvClass = config_copy.pop("base_env_class", None)
        
        # Asegurarse de incluir map_pieces en la configuración del entorno
        if 'map_pieces' in config_copy:
            map_pieces = config_copy.pop('map_pieces')
            config_copy['map'] = map_pieces  # Añadir el parámetro 'map' en el config del entorno

        # Crear el entorno con la configuración proporcionada
        self.env = BaseEnvClass(config_copy)

        self.possible_agents = list(self.env.observation_space.keys())
        self.observation_spaces = self.env.observation_space
        self.action_spaces = self.env.action_space
        self.agents = []

    def reset(self, *, seed=None, options=None):
        # Manejo del seed
        if seed is not None:
            try:
                obs_dict, info_dict = self.env.reset(seed=seed)
            except TypeError:
                obs_dict, info_dict = self.env.reset()
        else:
            obs_dict, info_dict = self.env.reset()

        self.agents = list(obs_dict.keys())

        return obs_dict, info_dict

    def step(self, action_dict):
        obs_dict, rewards_dict, terminateds_dict, truncateds_dict, infos_dict = self.env.step(action_dict)

        # Actualizar lista de agentes si se termina o trunca el episodio
        if terminateds_dict["__all__"] or truncateds_dict["__all__"]:
            self.agents = []
        else:
            self.agents = list(obs_dict.keys())

        return obs_dict, rewards_dict, terminateds_dict, truncateds_dict, infos_dict
