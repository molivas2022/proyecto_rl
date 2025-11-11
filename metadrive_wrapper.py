from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from metadrive import MetaDriveEnv


class MetadriveEnvWrapper(MultiAgentEnv):
    """
        Importante acerca de ambiente:
        debe tener respawn desactivado, si terminated o truncated == True para todos
        termina el episodio.
    """

    def __init__(self, config = None):
        """
        - env: enviroment de metadrive
        """
        super().__init__()

        # copiar config para evitar modificar original
        config_copy = config.copy()

        # obtener la clase custom
        BaseEnvClass = config_copy.pop("base_env_class", None)
        self.env = BaseEnvClass(config_copy)


        self.possible_agents = list(self.env.observation_space.keys())

        self.observation_spaces = self.env.observation_space

        self.action_spaces= self.env.action_space

        self.agents = []

    def reset(self, *, seed=None, options=None):

        # FIXME: Esto debe aceptar seed, por alguno razon lo hace y da un error
        # options no porque el entorno no lo acepta jeje
        if seed is not None:
            #obs_dict, info_dict = self.env.reset(seed=seed)
            obs_dict, info_dict = self.env.reset()
        else:
            obs_dict, info_dict = self.env.reset()
        
        self.agents = list(obs_dict.keys())

        return obs_dict, info_dict


    def step(self, action_dict):

        obs_dict, rewards_dict, terminateds_dict, truncateds_dict, infos_dict = self.env.step(action_dict)

        if terminateds_dict["__all__"] or truncateds_dict["__all__"]:
            self.agents = []
        else:
            self.agents = list(obs_dict.keys())

        return obs_dict, rewards_dict, terminateds_dict, truncateds_dict, infos_dict


