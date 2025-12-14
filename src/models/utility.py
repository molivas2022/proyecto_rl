import copy
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def make_target_network(module):
    """Utility para copiar target networks."""
    return copy.deepcopy(module)


def init_weights_xavier_bias_zero(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
