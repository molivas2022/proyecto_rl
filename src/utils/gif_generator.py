from ray.rllib.algorithms.algorithm import Algorithm
from src.utils.execute_episode import execute_one_episode
from metadrive import MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv
from pathlib import Path


def generate_gif(envclass, envconfig, modelpath, savepath, title):
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


if __name__ == "__main__":
    exp_dir = Path.cwd() / "experiments" / "ippo_norm_cnn"
    modelpath = exp_dir / "checkpoints" / "500"

    for i in range(3):
        curr_seed = 57 + i
        generate_gif(
            envclass=MultiAgentIntersectionEnv,
            envconfig=dict(
                num_agents=10,
                random_lane_num=True,
                random_spawn_lane_index=True,
                traffic_density=0,
                start_seed=curr_seed,
                allow_respawn=False,
                # agent_observation=StackedLidarObservation
            ),
            modelpath=modelpath,
            savepath=str(exp_dir / f"example_{i}.gif"),
            title="IPPO_NORM_CNN",
        )
