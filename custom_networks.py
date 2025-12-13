from typing import Any, Dict, Optional, Union
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.rl_module import (
    RLModuleConfig,
    DefaultModelConfig,
)
from ray.rllib.core.rl_module.apis import (
    TARGET_NETWORK_ACTION_DIST_INPUTS,
    TargetNetworkAPI,
    ValueFunctionAPI,
)
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    valid_padding,
)
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.framework import try_import_torch
import gymnasium as gym
import copy


torch, nn = try_import_torch()

def make_target_network(module):
    """Utility para copiar target networks."""
    return copy.deepcopy(module)

def init_weights_xavier_bias_zero(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# En metadrive, asumiendo que no se está usando random agent model, módulos de navegación adicionales, y
# los side detectors, lane detectors, no estén usando LIDAR, el state observation es un vector de una dimensión donde:
# - Los primeros 6 elementos son el estado ego base, que contiene velocidad actual, ultimas acciones,
#   guiñada, etc.
# - Los 10 siguientes son información de navigación.
# - 2 * num_lasers correspondiente a los side detectors, (sin lasers es 2).
# - num_lasers correspondiente a el lane detector, (sin lasers es 1).
# - num_lasers correspondiente al LIDAR.
# Por ejemplo, con la configuración estándar multiagente, se configuran cero lasers para los detectores
# de lados y lane, y 72 para el LIDAR. Luego, el vector tendría
# 6 + 10 + 2 + 1 + 72 = 91 elementos
# Verdaderamente solo nos interesa que los últimos 72 elementos sean PCD.
class MetaDriveCNN(TorchRLModule, ValueFunctionAPI, TargetNetworkAPI):
    """
    Módulo de CNN Hibrida (1D) + MLP para MetaDrice con una única observación Lidar (sin stack)
    """
    def __init__(
            self,
            *,
            observation_space: Optional[gym.Space],
            action_space: Optional[gym.Space],
            model_config: Optional[Union[dict, DefaultModelConfig]],
            inference_only: Optional[bool] = None,
            learner_only: bool = False,
            catalog_class=None,
            **kwargs,
        ):
        # 1. Llamamos al constructor padre PRIMERO con todos los kwargs.
        # Esto ejecuta RLModule.__init__ (que llama a setup() vacío)
        # y LUEGO nn.Module.__init__() (que inicializa el dict _modules).
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )

        # 2. AHORA, después de que super().__init__() ha retornado,
        # nn.Module está inicializado y podemos registrar capas.

        # Extraemos los hiperparámetros de model_config
        # _get_model_config() ya fue llamado por RLModuleSpec.build()
        hidden_dim = self.model_config["hidden_dim"]

        self.input_dim = self.observation_space.shape[0]
        output_dim = self.action_space.shape[0] * 2
        self.lidar_dim = 72
        self.non_lidar_dim = self.input_dim - self.lidar_dim

        # 3. Definimos las capas como atributos de `self`.
        # nn.Module.__setattr__ las registrará automáticamente.
        self._base_cnn_stack = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self._base_cnn_stack.apply(init_weights_xavier_bias_zero)

        self._shared_mlp = nn.Sequential(
            nn.Linear(self.non_lidar_dim + 128, hidden_dim),
            nn.Tanh(),
        )
        self._shared_mlp.apply(init_weights_xavier_bias_zero)

        self._action_dist_inputs = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self._action_dist_inputs.weight)
        nn.init.zeros_(self._action_dist_inputs.bias)

        self._values = nn.Linear(hidden_dim, 1)
        normc_initializer(0.01)(self._values.weight)

    @override(TorchRLModule)
    def setup(self):
        # Dejamos setup() vacío.
        # Es llamado por RLModule.__init__ *antes* de
        # que nn.Module.__init__ sea llamado, por lo que no podemos
        # registrar capas aquí.
        pass


    def _compute_shared_embeddings(self, obs: TensorType) -> Dict[str, TensorType]:
        """
        Computa los embeddings compartidos del CNN + MLP
        """
        non_lidar_feats = obs[:, 0 : self.non_lidar_dim]
        lidar_feats = obs[:, self.non_lidar_dim :]
        lidar_input = lidar_feats.unsqueeze(1)

        cnn_output = self._base_cnn_stack(lidar_input)
        final_feats = torch.cat((non_lidar_feats, cnn_output), dim=1)
        embeddings = self._shared_mlp(final_feats)
        
        return {"embeddings": embeddings, "cnn_output": cnn_output}

    # --- API TorchRLModule ---

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        result = self._compute_shared_embeddings(batch[Columns.OBS])
        embeddings = result["embeddings"]
        action_dist_inputs = self._action_dist_inputs(embeddings)
        return {Columns.ACTION_DIST_INPUTS: action_dist_inputs}

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        result = self._compute_shared_embeddings(batch[Columns.OBS])
        embeddings = result["embeddings"]
        action_dist_inputs = self._action_dist_inputs(embeddings)
        return {
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.EMBEDDINGS: embeddings,
        }

    # --- ValueFunctionAPI ---

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[TensorType] = None,
    ) -> TensorType:
        
        if embeddings is None:
            result = self._compute_shared_embeddings(batch[Columns.OBS])
            embeddings = result["embeddings"]
        
        values = self._values(embeddings)
        return values.squeeze(-1)

    # --- TargetNetworkAPI ---

    @override(TargetNetworkAPI)
    def make_target_networks(self) -> None:
        """Creamos las target networks."""
        self._target_base_cnn_stack = make_target_network(self._base_cnn_stack)
        self._target_shared_mlp = make_target_network(self._shared_mlp)
        self._target_action_dist_inputs = make_target_network(self._action_dist_inputs)

    @override(TargetNetworkAPI)
    def get_target_network_pairs(self):
        """
        Retorna pares de redes (source, target).
        No se incluye la red de valor.
        """
        return [
            (self._base_cnn_stack, self._target_base_cnn_stack),
            (self._shared_mlp, self._target_shared_mlp),
            (self._action_dist_inputs, self._target_action_dist_inputs),
        ]

    @override(TargetNetworkAPI)
    def forward_target(self, batch: Dict[str, Any], **kw) -> Dict[str, Any]:
        """
        Computa la distribución de acciones usando las redes target.
        """
        result = self._compute_shared_embeddings_target(batch[Columns.OBS])
        embeddings = result["embeddings"]
        
        # Usamos la target policy head
        action_dist_inputs = self._target_action_dist_inputs(embeddings)
        
        return {TARGET_NETWORK_ACTION_DIST_INPUTS: action_dist_inputs}



def init_weights_xavier_bias_zero(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MetaDriveStackedCNN(TorchRLModule, ValueFunctionAPI, TargetNetworkAPI):
    """
    CNN Híbrida (1D) + MLP para MetaDrive con Frame Stacking.
    Procesa una observación de forma (B, K, N), donde K es el stack_size.
    Utiliza K como canales de entrada para la convolución 1D sobre el LIDAR.
    NO usar con otros tipos de observación.
    """
    def __init__(
            self,
            *,
            observation_space: Optional[gym.Space],
            action_space: Optional[gym.Space],
            model_config: Optional[Union[dict, DefaultModelConfig]],
            inference_only: Optional[bool] = None,
            learner_only: bool = False,
            catalog_class=None,
            **kwargs,
        ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )

        hidden_dim = self.model_config["hidden_dim"]
        output_dim = self.action_space.shape[0] * 2

        # Analizamos la forma de la observación apilada: (Stack_Size, Feature_Dim)
        self.stack_size = self.observation_space.shape[0]
        self.full_feat_dim = self.observation_space.shape[1]
        
        # Definición de dimensiones específicas de MetaDrive
        self.lidar_dim = 72
        self.non_lidar_dim = self.full_feat_dim - self.lidar_dim # e.g. 19

        # La entrada a la MLP será: (Historia de estados escalares) + (Salida CNN)
        # Flattened state history: stack_size * non_lidar_dim
        self.mlp_input_dim = (self.stack_size * self.non_lidar_dim)

        # --- Definición de Capas ---

        # 1. CNN para LIDAR
        # Entrada: (Batch, Stack_Size, 72) -> Tratamos Stack_Size como in_channels
        self._base_cnn_stack = nn.Sequential(
            nn.Conv1d(in_channels=self.stack_size, out_channels=64, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1), # Forzamos salida espacial a 1
            nn.Flatten(),            # Salida: (Batch, 128)
        )
        self._base_cnn_stack.apply(init_weights_xavier_bias_zero)

        # 2. MLP Compartido
        # Concatenamos la historia de estados escalares aplanada con los features del LIDAR
        self._shared_mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim + 128, hidden_dim),
            nn.Tanh(),
        )
        self._shared_mlp.apply(init_weights_xavier_bias_zero)

        # 3. Cabezales (Policy & Value)
        self._action_dist_inputs = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self._action_dist_inputs.weight)
        nn.init.zeros_(self._action_dist_inputs.bias)

        self._values = nn.Linear(hidden_dim, 1)
        normc_initializer(0.01)(self._values.weight)

    @override(TorchRLModule)
    def setup(self):
        pass

    def _compute_shared_embeddings(self, obs: TensorType) -> Dict[str, TensorType]:
        """
        Procesa el tensor de observación (Batch, Stack, Features).
        """
        # Obs shape: (Batch, Stack_Size, 91)
        
        # 1. Separar componentes (Slicing)
        # Escalares: (Batch, Stack_Size, 19)
        non_lidar_history = obs[..., :self.non_lidar_dim] 
        # Lidar: (Batch, Stack_Size, 72)
        lidar_history = obs[..., self.non_lidar_dim:]     

        # 2. Procesar Escalares
        # Aplanamos la historia temporal para que la MLP vea la evolución de velocidad/acciones
        # Shape: (Batch, Stack_Size * 19)
        non_lidar_flat = non_lidar_history.reshape(non_lidar_history.shape[0], -1)

        # 3. Procesar LIDAR (CNN 1D)
        # Conv1d espera: (Batch, Channels, Length)
        # Nuestra lidar_history ya es (Batch, Stack_Size, 72), que calza perfecto.
        cnn_output = self._base_cnn_stack(lidar_history)

        # 4. Concatenar y MLP
        final_feats = torch.cat((non_lidar_flat, cnn_output), dim=1)
        embeddings = self._shared_mlp(final_feats)
        
        return {"embeddings": embeddings, "cnn_output": cnn_output}
    
    def _compute_shared_embeddings_target(self, obs: TensorType) -> Dict[str, TensorType]:
        """Versión para las target networks (código duplicado por claridad/API constraints)."""
        non_lidar_history = obs[..., :self.non_lidar_dim]
        lidar_history = obs[..., self.non_lidar_dim:]

        non_lidar_flat = non_lidar_history.reshape(non_lidar_history.shape[0], -1)
        
        # Usamos _target_base_cnn_stack
        cnn_output = self._target_base_cnn_stack(lidar_history)

        final_feats = torch.cat((non_lidar_flat, cnn_output), dim=1)
        # Usamos _target_shared_mlp
        embeddings = self._target_shared_mlp(final_feats)
        
        return {"embeddings": embeddings}

    # --- API TorchRLModule ---

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        result = self._compute_shared_embeddings(batch[Columns.OBS])
        embeddings = result["embeddings"]
        action_dist_inputs = self._action_dist_inputs(embeddings)
        return {Columns.ACTION_DIST_INPUTS: action_dist_inputs}

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        result = self._compute_shared_embeddings(batch[Columns.OBS])
        embeddings = result["embeddings"]
        action_dist_inputs = self._action_dist_inputs(embeddings)
        return {
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.EMBEDDINGS: embeddings,
        }

    # --- ValueFunctionAPI ---

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[TensorType] = None,
    ) -> TensorType:
        
        if embeddings is None:
            result = self._compute_shared_embeddings(batch[Columns.OBS])
            embeddings = result["embeddings"]
        
        values = self._values(embeddings)
        return values.squeeze(-1)

    # --- TargetNetworkAPI ---

    @override(TargetNetworkAPI)
    def make_target_networks(self) -> None:
        self._target_base_cnn_stack = make_target_network(self._base_cnn_stack)
        self._target_shared_mlp = make_target_network(self._shared_mlp)
        self._target_action_dist_inputs = make_target_network(self._action_dist_inputs)

    @override(TargetNetworkAPI)
    def get_target_network_pairs(self):
        return [
            (self._base_cnn_stack, self._target_base_cnn_stack),
            (self._shared_mlp, self._target_shared_mlp),
            (self._action_dist_inputs, self._target_action_dist_inputs),
        ]

    @override(TargetNetworkAPI)
    def forward_target(self, batch: Dict[str, Any], **kw) -> Dict[str, Any]:
        result = self._compute_shared_embeddings_target(batch[Columns.OBS])
        embeddings = result["embeddings"]
        action_dist_inputs = self._target_action_dist_inputs(embeddings)
        return {TARGET_NETWORK_ACTION_DIST_INPUTS: action_dist_inputs}
