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


class MAPPOMLP(TorchRLModule, ValueFunctionAPI):
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
            raise ValueError(
                f"MAPPO requires Dict observation space. Got: {type(self.observation_space)}"
            )

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
            nn.ReLU(),
        )
        self.pi_head = nn.Linear(hidden_dim, self.action_dim * 2)

        # --- CRITIC (To be synchronized) ---
        self.critic_encoder = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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


class MAPPOCNN(TorchRLModule, ValueFunctionAPI):
    """
    MAPPO Module with Hybrid CNN (1D) Actor and MLP Critic.

    Actor: Processes local 'obs' (Scalar + LIDAR) using the MetaDrive-specific hybrid architecture.
    Critic: Processes global 'state' using a standard MLP.
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
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            **kwargs,
        )

        # 1. Validation
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError(
                f"MAPPO requires Dict observation space. Got: {type(self.observation_space)}"
            )

        # 2. Dimensions
        self.local_obs_dim = self.observation_space["obs"].shape[0]
        self.global_state_dim = self.observation_space["state"].shape[0]
        self.action_dim = self.action_space.shape[0] * 2  # *2 for mean/log_std

        # MetaDrive specific Lidar config
        self.lidar_dim = 72
        self.non_lidar_dim = self.local_obs_dim - self.lidar_dim

        # Config access
        hidden_dim = 256
        if self.model_config:
            hidden_dim = self.model_config.get("hidden_dim", 256)

        # -----------------------------------------------------------
        # ACTOR (Unique per agent) - Hybrid CNN + MLP
        # -----------------------------------------------------------
        # Matches IPPOCNN architecture for feature extraction
        self.actor_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Input to MLP is non_lidar features + CNN output (128)
        self.actor_mlp = nn.Sequential(
            nn.Linear(self.non_lidar_dim + 128, hidden_dim),
            nn.Tanh(),
        )

        self.pi_head = nn.Linear(hidden_dim, self.action_dim)

        # -----------------------------------------------------------
        # CRITIC (Centralized/Global) - MLP
        # -----------------------------------------------------------
        # Processes global state (likely flattened), so we use MLP
        self.critic_encoder = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.vf_head = nn.Linear(hidden_dim, 1)

        # -----------------------------------------------------------
        # Initialization
        # -----------------------------------------------------------
        self.actor_cnn.apply(init_weights)
        self.actor_mlp.apply(init_weights)

        # Initialize Heads
        nn.init.xavier_uniform_(self.pi_head.weight)
        nn.init.zeros_(self.pi_head.bias)

        self.critic_encoder.apply(init_weights)
        nn.init.xavier_uniform_(self.vf_head.weight)
        nn.init.zeros_(self.vf_head.bias)

    def _actor_forward(self, obs: torch.Tensor):
        """Helper to process local observations via Hybrid CNN."""
        # Split observation
        non_lidar_feats = obs[:, 0 : self.non_lidar_dim]
        lidar_feats = obs[:, self.non_lidar_dim :]

        # CNN Pass (Add channel dim)
        lidar_input = lidar_feats.unsqueeze(1)
        cnn_out = self.actor_cnn(lidar_input)

        # Concat & MLP Pass
        combined = torch.cat((non_lidar_feats, cnn_out), dim=1)
        actor_features = self.actor_mlp(combined)

        return actor_features

    def _forward_inference(self, batch: Dict[str, Any], **kwargs):
        obs = batch[Columns.OBS]["obs"]
        actor_features = self._actor_forward(obs)
        action_logits = self.pi_head(actor_features)
        return {Columns.ACTION_DIST_INPUTS: action_logits}

    def _forward_exploration(self, batch: Dict[str, Any], **kwargs):
        return self._forward_inference(batch, **kwargs)

    def _forward_train(self, batch: Dict[str, Any], **kwargs):
        # 1. Actor Pass (Local Obs -> Hybrid CNN)
        obs = batch[Columns.OBS]["obs"]
        actor_features = self._actor_forward(obs)
        action_logits = self.pi_head(actor_features)

        # 2. Critic Pass (Global State -> MLP)
        state = batch[Columns.OBS]["state"]
        critic_features = self.critic_encoder(state)
        value_pred = self.vf_head(critic_features).squeeze(-1)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.VF_PREDS: value_pred,
        }

    def compute_values(self, batch: Dict[str, Any], embeddings=None):
        # Note: In MAPPO, 'embeddings' usually refers to actor embeddings.
        # However, value function relies on Global State, so we typically
        # recompute from state rather than reusing actor embeddings.
        state = batch[Columns.OBS]["state"]
        critic_features = self.critic_encoder(state)
        value_pred = self.vf_head(critic_features).squeeze(-1)
        return value_pred
