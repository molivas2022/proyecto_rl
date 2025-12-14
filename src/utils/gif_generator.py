from ray.rllib.algorithms.algorithm import Algorithm
from .execute_episode import execute_one_episode
# from metadrive import MultiAgentIntersectionEnv
# from pathlib import Path


def generate_gif(envclass, envconfig, modelpath, savepath, title, seed=42):
    """
    genera gif de un episodio utilizando un algoritmo
    Args:
        envclass: entorno multiagente de MetaDrive en el que hacer una
        ejecución de prueba.
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

    module_dict = {}
    for i in range(envconfig["num_agents"]):
        module_dict[f"agent{i}"] = algo.get_module(f"policy_{i}")
        print(algo.get_module(f"policy_{i}"))

    env, _ = execute_one_episode(env, module_dict, title, enable_render=True)

    env.top_down_renderer.generate_gif(savepath)

    algo.stop()

    print("Rendering environment to gif")
    env.top_down_renderer.generate_gif(savepath)
    print("Rendering done, stopping algorithm and closing environment...")
    env.close()


# if __name__ == "__main__":
#     exp_dir = Path.cwd() / "experimentos" / "exp7"
#     modelpath = exp_dir / "checkpoints" / "640"
# 
#     for i in range(3):
#         generate_gif(
#             envclass=MultiAgentIntersectionEnv,
#             envconfig=dict(
#                 num_agents=5,
#                 allow_respawn=False,
#                 random_spawn_lane_index=True,
#                 start_seed=57,
#                 traffic_density=0,
#                 agent_observation=StackedLidarObservation
#             ),
#             seed=0,
#             modelpath=modelpath,
#             savepath=str(exp_dir / f"example_{i}.gif"),
#             title="IPPO",
#         )
