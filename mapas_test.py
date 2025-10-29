import torch
from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from IPython.display import clear_output
import random

class experiment:
    env: MetaDriveEnv
    start_seed: int
    ammount: int

    def __init__(self, env, start_seed, ammount):
        self.env = env
        self.start_seed = start_seed
        self.ammount = ammount

    @property
    def end_seed(self):
        return self.start_seed + self.ammount



# modo basico
def split_train_test(ammount_train:int = 800, ammount_test:int = 200, num_piezas:int = 4):
    env_config_train = dict(
        use_render=False,
        traffic_mode="respawn",  # default: basic
        map=num_piezas,
        traffic_density=0.05,
        manual_control=False,
        controller="keyboard",
        start_seed=1,
        num_scenarios=ammount_train,
        random_lane_width=True,
        random_lane_num=True,
    )
    # crear una copia para test
    env_config_test = env_config_train.copy()
    env_config_test["start_seed"] = ammount_train + 1
    env_config_test["num_scenarios"] = ammount_test

    env_train = MetaDriveEnv(env_config_train)
    env_test = MetaDriveEnv(env_config_test)
    train_set = experiment(env_train, 1, ammount_train)
    test_set = experiment(env_test, ammount_train + 1, ammount_test)

    return (train_set, test_set)


def execute_simulation(data, render:bool = False, timesteps=1000):
    env, start_seed, ammount = data.env, data.start_seed, data.ammount
    for i in range(start_seed, start_seed + ammount):
        env.reset(seed=i)
        steering = 0
        throttle = 0
        for i in range(timesteps):
            if (i % 25 == 0):
                if (random.random() < 0.5):
                    steering = -0.2
                else:
                    steering = +0.2

            if (i % 100 == 0):
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
                for k, v in info.items():
                    print(k, ": ", v)
            if render:
                env.render(window=True,
                           mode="topdown",
                           scaling=2,
                           camera_position=(100, 100),
                           screen_size=(750, 750),
                           screen_record=True,
                           text={"episode_step": env.engine.episode_step,
                                 "mode": "basic"})
            print(i)
    env.close()
    clear_output()
    print("---------------------------")



if __name__ == "__main__":
    env_train, env_test = split_train_test(ammount_train=40, ammount_test=200)
    execute_simulation(env_train, timesteps=2)
    execute_simulation(env_train, True)