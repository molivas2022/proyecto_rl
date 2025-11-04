# ============ ejemplo de como customizar una red ===================
import torch

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule

# Define your custom env class by subclassing `TorchRLModule`:
class CustomTorchRLModule(TorchRLModule):
    def setup(self):
        # You have access here to the following already set attributes:
        # self.observation_space
        # self.action_space
        # self.inference_only
        # self.model_config  # <- a dict with custom settings
        input_dim = self.observation_space.shape[0]
        hidden_dim = self.model_config["hidden_dim"]
        output_dim = self.action_space.n

        # Define and assign your torch subcomponents.
        self._policy_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def _forward(self, batch, **kwargs):
        # Push the observations from the batch through our `self._policy_net`.
        action_logits = self._policy_net(batch[Columns.OBS])
        # Return parameters for the default action distribution, which is
        # `TorchCategorical` (due to our action space being `gym.spaces.Discrete`).
        return {Columns.ACTION_DIST_INPUTS: action_logits}
