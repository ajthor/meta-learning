import torch
import torch.func as func
from typing import Any
from collections import OrderedDict


class AdaptedParameterModel(torch.nn.Module):
    """Wrapper that allows using adapted parameters with the original architecture."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        adapted_params: OrderedDict[str, torch.Tensor],
    ):
        """Initialize with base model and adapted parameters.

        Args:
            base_model: The original model architecture
            adapted_params: OrderedDict of adapted parameter tensors
        """
        self.base_model = base_model
        self.adapted_params = adapted_params

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass using adapted parameters.

        Returns:
            Model output using adapted parameters
        """
        return func.functional_call(self.base_model, self.adapted_params, args, kwargs)
