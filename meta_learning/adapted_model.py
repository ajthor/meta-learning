import torch
import torch.func as func
from typing import Any
from collections import OrderedDict


class AdaptedParameterModel:
    """A wrapper that allows using adapted parameters with the original model architecture.
    
    This class provides a clean interface for evaluating a model with different parameters
    while maintaining the original model's forward pass logic.
    """
    
    def __init__(self, base_model: torch.nn.Module, adapted_params: OrderedDict[str, torch.Tensor]):
        """Initialize with base model and adapted parameters.
        
        Args:
            base_model: The original model architecture
            adapted_params: OrderedDict of adapted parameter tensors
        """
        self.base_model = base_model
        self.adapted_params = adapted_params
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass using adapted parameters.
        
        Returns:
            Model output using adapted parameters
        """
        return func.functional_call(self.base_model, self.adapted_params, args, kwargs)
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Alternative forward method for consistency with nn.Module interface."""
        return self(*args, **kwargs)
    
    def parameters(self):
        """Return adapted parameters (for compatibility)."""
        return self.adapted_params.values()
    
    def named_parameters(self):
        """Return named adapted parameters."""
        return self.adapted_params.items()
    
    def state_dict(self):
        """Return adapted parameters as state dict."""
        return self.adapted_params.copy()
    
    def to(self, device):
        """Move adapted parameters to device."""
        adapted_params = {
            name: param.to(device) if hasattr(param, 'to') else param
            for name, param in self.adapted_params.items()
        }
        return AdaptedParameterModel(self.base_model, adapted_params)
    
    @property
    def device(self):
        """Get device of adapted parameters."""
        first_param = next(iter(self.adapted_params.values()))
        return first_param.device if hasattr(first_param, 'device') else None