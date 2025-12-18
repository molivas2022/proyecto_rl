import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def get_base_env(env):
    """Recursively unwraps the environment to find the base MetaDriveEnv."""
    # If this is the base env (has engine), return it
    if hasattr(env, "engine") and env.engine is not None:
        return env
    # If it's a wrapper, dig deeper
    if hasattr(env, "env"):
        return get_base_env(env.env)
    if hasattr(env, "base_env"):
        return get_base_env(env.base_env)
    return None


def actions_from_distributions(module_dict, obs, env):
    action_dict = {}

    for agent in obs:
        if agent == "__all__":
            continue

        module = module_dict.get(agent)
        if module is None:
            continue

        agent_obs = obs[agent]

        # --- Polimorfico ---
        if isinstance(agent_obs, dict):
            obs_batch = {
                k: torch.from_numpy(v).float().unsqueeze(0)
                for k, v in agent_obs.items()
            }
            input_dict = {Columns.OBS: obs_batch}
        else:
            obs_batch = torch.from_numpy(agent_obs).float().unsqueeze(0)
            input_dict = {Columns.OBS: obs_batch}

        # Inferencia
        with torch.no_grad():
            out_dict = module.forward_inference(input_dict)
            dist_params_np = out_dict["action_dist_inputs"].cpu().numpy()[0]

        # Parseamos acciones
        raw_means = dist_params_np[[0, 1]]
        greedy_act = np.clip(raw_means, -1.0, 1.0)

        action_dict[agent] = greedy_act

    return action_dict


def execute_one_episode(env, module_dict, title=None, enable_render=False):
    obs, info = env.reset()

    base_env = get_base_env(env)
    if base_env is None and enable_render:
        print("Warning: Could not find base MetaDriveEnv for rendering info.")

    terminateds = {"__all__": False}
    truncateds = {"__all__": False}
    infos_list = []

    if enable_render and base_env:
        base_env.render(
            window=False,
            mode="topdown",
            scaling=2,
            camera_position=(100, 0),
            screen_size=(500, 500),
            screen_record=True,
            text={"episode_step": base_env.engine.episode_step, "method": title or ""},
        )

    while not (terminateds["__all__"] or truncateds["__all__"]):
        actions = actions_from_distributions(module_dict, obs, env)

        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        infos_list.append(infos)

        if enable_render and base_env:
            base_env.render(
                window=False,
                mode="topdown",
                scaling=2,
                camera_position=(100, 0),
                screen_size=(500, 500),
                screen_record=True,
                text={"episode_step": base_env.engine.episode_step, "method": title},
            )

    return env, infos_list
