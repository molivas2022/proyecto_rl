import numpy as np
import gymnasium as gym
from collections import deque
from metadrive.obs.state_obs import LidarStateObservation

class StackedLidarObservation(LidarStateObservation):
    """
    Extiende LidarStateObservation para que retorne un stack de las últimas k obervaciones.
    """
    def __init__(self, config):
        # Un stack size por defecto conservativo, de 3.
        if "stack_size" not in config:
            config["stack_size"] = 3
            
        super(StackedLidarObservation, self).__init__(config)
        
        self.stack_size = config["stack_size"]
        # FIFO buffer FIFO buffer FIFO buffer FIFO buffer FIFO buffer
        self.observation_buffer = deque(maxlen=self.stack_size)

    @property
    def observation_space(self):
        # Necesitamos lel observation_space original, el nuevo espacio será una extensión de este
        # en k canales.
        original_space = super(StackedLidarObservation, self).observation_space
        feature_dim = original_space.shape[0]

        return gym.spaces.Box(
            low=0.0, 
            high=1.0,
            shape=(self.stack_size, feature_dim),
            dtype=np.float32
        )

    def observe(self, vehicle):
        # Obtenemos la observación actual
        current_frame = super(StackedLidarObservation, self).observe(vehicle)

        # Casos cold start, se llena el buffer con k copias de la observación
        if len(self.observation_buffer) == 0:
            for _ in range(self.stack_size):
                self.observation_buffer.append(current_frame)
        else:
            # Actualizamos el buffer
            self.observation_buffer.append(current_frame)

        # (k, Feature_Dim) 
        return np.array(self.observation_buffer, dtype=np.float32)

    def reset(self, env, vehicle=None):
        """
        Un override de reset para limpiar el buffer interno cuando el episodio reinicia.
        """
        self.observation_buffer.clear()
        super(StackedLidarObservation, self).reset(env, vehicle)
