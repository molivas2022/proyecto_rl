from ray.rllib.env.multi_agent_env import MultiAgentEnv
from .normalization_wrapper import MetadriveMARLNormalizedRewardEnv


class MetadriveEnvWrapper(MultiAgentEnv):
    """
    Importante acerca de ambiente:
    - Debe tener respawn desactivado.
    - Si terminated["__all__"] o truncated["__all__"] == True,
      el episodio termina para todos.
    """

    def __init__(self, config=None):
        """
        - config:
            - base_env_class: clase base de MetaDrive (MultiAgentIntersectionEnv, etc.)
            - start_seed: seed inicial para los escenarios (opcional)
            - num_scenarios: cantidad de escenarios distintos (opcional)
            - normalize_reward: bool, si normalizamos o no
            - gamma: descuento para normalización de reward
            - map_pieces: (opcional) se renombra a "map" por compatibilidad
        """
        super().__init__()

        # Evitar modificar la config original
        config_copy = (config or {}).copy()

        # Clase base del entorno
        BaseEnvClass = config_copy.pop("base_env_class", None)
        if BaseEnvClass is None:
            raise ValueError("config['base_env_class'] es obligatorio para MetadriveEnvWrapper")

        # Flags de normalización
        normalize_reward = config_copy.pop("normalize_reward", False)
        gamma = config_copy.pop("gamma", 0.0)

        # ---------- SEEDS (tu lógica + flag de compatibilidad) ----------
        # NUEVO: bandera para saber si debemos usar la lógica de seeds o no
        self._use_seed_cycle = ("start_seed" in config_copy) and ("num_scenarios" in config_copy)

        if self._use_seed_cycle:
            # Modo "nuevo": usamos el ciclo de seeds
            self.start_seed = config_copy.get("start_seed", 0)
            self.num_scenarios = config_copy.get("num_scenarios", 1)
            self.current_seed = self.start_seed
        else:
            # Modo "refactor antiguo": no tocamos seeds, se ignorará el seed en reset
            self.start_seed = None
            self.num_scenarios = None
            self.current_seed = None

        # map_pieces → map (por si lo usas con otras configs)
        if "map_pieces" in config_copy and "map" not in config_copy:
            config_copy["map"] = config_copy.pop("map_pieces")

        # Crear entorno base
        base_env = BaseEnvClass(config_copy)

        # Normalizar rewards si corresponde
        if normalize_reward:
            if gamma <= 0:
                raise ValueError(
                    "Si normalize_reward=True debes pasar un gamma > 0 en la config."
                )
            self.env = MetadriveMARLNormalizedRewardEnv(base_env, gamma)
        else:
            self.env = base_env

        # Espacios RLlib
        self.possible_agents = list(self.env.observation_space.keys())
        self.observation_spaces = self.env.observation_space
        self.action_spaces = self.env.action_space
        self.agents = []

    # --------- Helpers internos ---------
    def _next_seed(self, explicit_seed):
        """
        Devuelve la seed a usar cuando estamos en modo "seed cycle".
        """
        # Si por alguna razón se llama en modo compatibilidad, no hacemos magia extra
        if not self._use_seed_cycle:
            return explicit_seed

        if explicit_seed is not None:
            return explicit_seed

        next_seed = self.current_seed
        self.current_seed += 1

        if self.current_seed >= self.start_seed + self.num_scenarios:
            self.current_seed = self.start_seed

        return next_seed

    # --------- API MultiAgentEnv ---------
    def reset(self, *, seed=None, options=None):
        """
        Resetea el entorno.
        - Si _use_seed_cycle=True: usa la lógica de seeds incremental (modo nuevo).
        - Si _use_seed_cycle=False: se comporta como el refactor antiguo
          (ignora seed y llama self.env.reset()).
        """

        if self._use_seed_cycle:
            # MODO NUEVO (con seeds del YAML)
            next_seed = self._next_seed(seed)

            print(
                f"[Wrapper] reset with seed={next_seed}, "
                f"range=[{self.start_seed}, {self.start_seed + self.num_scenarios})"
            )

            # Si el wrapper de normalización no acepta seed en reset, esto lo maneja el try/except
            try:
                obs_dict, info_dict = self.env.reset(seed=next_seed)
            except TypeError:
                obs_dict, info_dict = self.env.reset()
        else:
            # MODO COMPATIBILIDAD (como el refactor original):
            # ignoramos el seed que llega y simplemente reseteamos.
            # Equivalente a:
            #   if seed is not None:
            #       obs_dict, info_dict = self.env.reset()
            #   else:
            #       obs_dict, info_dict = self.env.reset()
            obs_dict, info_dict = self.env.reset()

        self.agents = list(obs_dict.keys())
        return obs_dict, info_dict

    def step(self, action_dict):
        """
        Paso estándar multi-agente: passthrough al env interno.
        """
        (
            obs_dict,
            rewards_dict,
            terminateds_dict,
            truncateds_dict,
            infos_dict,
        ) = self.env.step(action_dict)

        if terminateds_dict.get("__all__", False) or truncateds_dict.get("__all__", False):
            self.agents = []
        else:
            self.agents = list(obs_dict.keys())

        return obs_dict, rewards_dict, terminateds_dict, truncateds_dict, infos_dict
