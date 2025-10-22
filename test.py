import torch
from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from IPython.display import clear_output
import random


print("Probando torch")
print("PyTorch versión:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Dispositivo:", torch.cuda.get_device_name(0))
print("---------------------------")


print("Probando metadrive")
# modo basico
env_config = dict(use_render=False,
                  traffic_mode="respawn", # default: basic
                  map="O",
                  traffic_density=0.05,
                  manual_control=False,
                  controller="keyboard",
                  )

env = MetaDriveEnv(env_config)

# solo se mueven los bots cuando se acerca el ego car
#env = MetaDriveEnv(dict(traffic_mode="trigger", map="O"))

# los bots respawnean
#env = MetaDriveEnv(dict(traffic_mode="respawn", map="O", traffic_density=0.05))

# mezcla de ambos modos
#env = MetaDriveEnv(dict(traffic_mode="hybrid", map="O")) # Default, traffic density=0.1

env.reset(seed=0)

try:
    steering = 0
    throttle = 0
    for i in range(1000):
        # actions: steering, throttle (las acciones son continuas)
        # steering \in [-1, 1]  -> dirección -1 es izquierda y +1 es derecha jejeje
        # throttle \in [-1, 1]   -> -1 frenar, +1 acelelerar
        # creo que esto esta mal y son 3 acciones, pero soy chorooooo wuajajajajajajajajajaajajajajaj

        # Elige de forma aleatoria una dirección
        if (i%25 == 0):
            if (random.random() < 0.5):
                steering = -0.2
            else:
                steering = +0.2

        if (i%100 == 0):
            if (random.random() < 0.5):
                throttle = -0.2
            else:
                throttle = +0.2

        action = [steering, throttle]

        obs, reward, terminated, truncated, info = env.step(action)

        if i == 2:
            print("Observation:\n", obs)
            print("Reward:\n", reward)
            print("Terminated:\n", terminated)
            print("Truncated:\n", truncated)
            for k,v in info.items():
                print(k, ": ", v)

        env.render(window=True,
                   mode="topdown",
                   scaling=2,
                   camera_position=(100, 0),
                   screen_size=(1000, 1000),
                   screen_record=True,
                   text={"episode_step": env.engine.episode_step,
                         "mode": "basic"})
        

    #env.top_down_renderer.generate_gif()
finally:
    env.close()
    clear_output()
print("---------------------------")
