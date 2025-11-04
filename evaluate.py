# Ejemplo de como sería hacer tests

from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule

# Create only the neural network (RLModule) from our algorithm checkpoint.
# See here (https://docs.ray.io/en/master/rllib/checkpoints.html)
# to learn more about checkpointing and the specific "path" used.
rl_module = RLModule.from_checkpoint(
    Path(best_checkpoint.path)
    / "learner_group"
    / "learner"
    / "rl_module"
    / "default_policy"
)

# Create the RL environment to test against (same as was used for training earlier).
env = gym.make("Pendulum-v1", render_mode="human")

episode_return = 0.0
done = False

# Reset the env to get the initial observation.
obs, info = env.reset()

while not done:
    # Uncomment this line to render the env.
    # env.render()

    # Compute the next action from a batch (B=1) of observations.
    obs_batch = torch.from_numpy(obs).unsqueeze(0)  # add batch B=1 dimension
    model_outputs = rl_module.forward_inference({"obs": obs_batch})

    # Extract the action distribution parameters from the output and dissolve batch dim.
    action_dist_params = model_outputs["action_dist_inputs"][0].numpy()

    # We have continuous actions -> take the mean (max likelihood).
    greedy_action = np.clip(
        action_dist_params[0:1],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
        a_min=env.action_space.low[0],
        a_max=env.action_space.high[0],
    )
    # For discrete actions, you should take the argmax over the logits:
    # greedy_action = np.argmax(action_dist_params)

    # Send the action to the environment for the next step.
    obs, reward, terminated, truncated, info = env.step(greedy_action)

    # Perform env-loop bookkeeping.
    episode_return += reward
    done = terminated or truncated

print(f"Reached episode return of {episode_return}.")
