import gymnasium as gym
from typing import Any, Dict, Optional, Union
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import DefaultModelConfig

from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class MAPPOTorchRLModule(TorchRLModule, ValueFunctionAPI):
    """
    Standard MAPPO Module: Contains its OWN Actor and its OWN Critic.
    We will synchronize the Critic weights externally via Callbacks.
    """

    def __init__(
            self,
            *,
            observation_space: Optional[gym.Space] = None,
            action_space: Optional[gym.Space] = None,
            inference_only: Optional[bool] = None,
            learner_only: bool = False,
            model_config: Optional[Union[dict, DefaultModelConfig]] = None,
            **kwargs,
            ):
        # 1. Pass explicit arguments to super() to avoid Deprecation/Config errors
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            **kwargs,
        )

        # Validation
        if not isinstance(self.observation_space, gym.spaces.Dict):
             raise ValueError(f"MAPPO requires Dict observation space. Got: {type(self.observation_space)}")

        self.local_obs_dim = self.observation_space["obs"].shape[0]
        self.global_state_dim = self.observation_space["state"].shape[0]
        self.action_dim = self.action_space.shape[0]
        
        # Safe config access
        hidden_dim = 256
        if self.model_config:
            hidden_dim = self.model_config.get("hidden_dim", 256)

        # --- ACTOR (Unique per agent) ---
        self.actor_encoder = nn.Sequential(
            nn.Linear(self.local_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.pi_head = nn.Linear(hidden_dim, self.action_dim * 2)

        # --- CRITIC (To be synchronized) ---
        self.critic_encoder = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.vf_head = nn.Linear(hidden_dim, 1)

        # Initialize
        self.actor_encoder.apply(init_weights)
        self.critic_encoder.apply(init_weights)
        self.pi_head.apply(init_weights)
        self.vf_head.apply(init_weights)

    def _forward_inference(self, batch: Dict[str, Any], **kwargs):
        obs = batch[Columns.OBS]["obs"]
        actor_features = self.actor_encoder(obs)
        action_logits = self.pi_head(actor_features)
        return {Columns.ACTION_DIST_INPUTS: action_logits}

    def _forward_exploration(self, batch: Dict[str, Any], **kwargs):
        return self._forward_inference(batch, **kwargs)

    def _forward_train(self, batch: Dict[str, Any], **kwargs):
        # 1. Actor Pass
        obs = batch[Columns.OBS]["obs"]
        actor_features = self.actor_encoder(obs)
        action_logits = self.pi_head(actor_features)

        # 2. Critic Pass
        state = batch[Columns.OBS]["state"]
        critic_features = self.critic_encoder(state)
        value_pred = self.vf_head(critic_features).squeeze(-1)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.VF_PREDS: value_pred,
        }

    def compute_values(self, batch: Dict[str, Any], embeddings=None):
        state = batch[Columns.OBS]["state"]
        critic_features = self.critic_encoder(state)
        value_pred = self.vf_head(critic_features).squeeze(-1)
        return value_pred
