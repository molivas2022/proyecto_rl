import gymnasium as gym
import numpy as np
from pathlib import Path
from ray.rllib.algorithms.algorithm import Algorithm
from metadrive import MultiAgentIntersectionEnv
from pathlib import Path

def generate_gif(envclass, envconfig, modelpath, savepath, title, seed=42):
    """
    genera gif de un episodio utilizando un algoritmo
    Args:
        envclass: entorno multiagente de MetaDrive en el que hacer una ejecución de prueba.
        seed: semilla de generación del entorno. Por defecto: 42.
        modelpath: donde se encuentra el checkpoint de entrenamiento
        savepath: donde se guardaran los gifs.
    """

    # inicializar entorno / con config
    env = envclass(envconfig)

    # obtener el algoritmo completo desde el checkpoint
    try:
        algo = Algorithm.from_checkpoint(modelpath)
    except Exception as e:
        print(f"====ERROR: no se pudo cargar el checkpoint.====\n {e}")
        raise

    episode_return_total = 0.0
    
    obs, info = env.reset(seed=seed)
    # obs = dic {agent : observation} 
    
    terminateds = {"__all__": False}
    truncateds = {"__all__": False}

    while not (terminateds["__all__"] or truncateds["__all__"]):
        
        # obtener acciones dic {agent : action} 
        actions = algo.compute_actions(
            observations=obs,
            explore=False  # no hay que explorar
        )
        
        # ejecutar step
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # configuraciones de renderizado y renderizar
        env.render(window=False,
                   mode="topdown",
                   scaling=2,
                   camera_position=(100, 0),
                   screen_size=(1000, 1000),
                   screen_record=True,
                   text={"episode_step": env.engine.episode_step,
                         "method": title})
        
    env.top_down_renderer.generate_gif(savepath)

    algo.stop()
    env.close()
    

if __name__ == "__main__":
    modelpath = Path("")

    generate_gif(envclass = MultiAgentIntersectionEnv,
                 envconfig = dict(num_agents = 2,
                                  allow_respawn = False
                                  ),
                 seed=42,
                 modelpath = modelpath,
                 savepath = "example.gif",
                 title = "TEST")
