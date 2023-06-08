import gymnasium as gym
import numpy as np
import torch
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class TanhTorchDeterministic(TorchDistributionWrapper):
    """This distribution stores the deterministic policy, with takes logtis as self.inputs, and gives the tanh value, that contains the full gradient.
    """

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        # Note that this will also be used when sample a trajactory
        return self.deterministic_sample()

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        return torch.tanh(self.inputs)

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        return self.sampled_action_logp()

    @override(TorchDistributionWrapper)
    def sampled_action_logp(self) -> TensorType:
        return torch.zeros((self.inputs.size()[0], ), dtype=torch.float32, device=self.inputs.device)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space: gym.Space,
        model_config: ModelConfigDict,
    ) -> int | np.ndarray:
        return np.prod(action_space.shape, dtype=np.int32)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        return -torch.ones((self.inputs.size()[0], ), dtype=torch.float32, device=self.inputs.device) * np.inf

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        return torch.zeros((self.inputs.size()[0], ), dtype=torch.float32, device=self.inputs.device)


class SigmoidTorchDeterministic(TorchDistributionWrapper):
    """This distribution stores the deterministic policy, with takes logtis as self.inputs, and gives the tanh value, that contains the full gradient.
    """

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        # Note that this will also be used when sample a trajactory
        return self.deterministic_sample()

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        return torch.sigmoid(self.inputs)

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        return self.sampled_action_logp()

    @override(TorchDistributionWrapper)
    def sampled_action_logp(self) -> TensorType:
        return torch.zeros((self.inputs.size()[0], ), dtype=torch.float32, device=self.inputs.device)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space: gym.Space,
        model_config: ModelConfigDict,
    ) -> int | np.ndarray:
        return np.prod(action_space.shape, dtype=np.int32)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        return -torch.ones((self.inputs.size()[0], ), dtype=torch.float32, device=self.inputs.device) * np.inf

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        return torch.zeros((self.inputs.size()[0], ), dtype=torch.float32, device=self.inputs.device)
