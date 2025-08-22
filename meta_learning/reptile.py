import torch
import copy
from typing import Tuple, Callable
from collections import OrderedDict


def adapt_model(
    model: torch.nn.Module,
    example_data: Tuple[torch.Tensor, ...],
    loss_fn: Callable,
    inner_lr: float,
    inner_steps: int
) -> OrderedDict[str, torch.Tensor]:
    """Adapt model to a specific task using standard SGD (Reptile approach).
    
    Args:
        model: The model to adapt
        example_data: Tuple of (inputs, targets, ...) for adaptation
        loss_fn: Loss function that takes (model, data) -> loss
        inner_lr: Learning rate for inner loop adaptation
        inner_steps: Number of gradient steps for adaptation
        
    Returns:
        OrderedDict of adapted parameters
    """
    # Clone model for task-specific adaptation
    adapted_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)
    
    # Standard SGD training for Reptile
    for _ in range(inner_steps):
        optimizer.zero_grad()
        loss = loss_fn(adapted_model, example_data)
        
        # Check for NaN/Inf in loss
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss detected: {loss.item()}")
            
        loss.backward()
        optimizer.step()
    
    # Return adapted parameters as OrderedDict
    return OrderedDict(adapted_model.named_parameters())


def meta_update_step(
    model: torch.nn.Module,
    query_data: Tuple[torch.Tensor, ...],
    example_data: Tuple[torch.Tensor, ...],
    inner_lr: float,
    meta_lr: float,
    inner_steps: int,
    loss_fn: Callable,
    meta_optimizer: torch.optim.Optimizer
) -> float:
    """Single meta-learning update step across a batch of tasks.
    
    Args:
        model: The meta-model to update
        query_data: Batch of query data for meta-loss computation
        example_data: Batch of example data for adaptation
        inner_lr: Learning rate for inner loop adaptation
        meta_lr: Learning rate for meta updates (Reptile step size)
        inner_steps: Number of inner loop steps
        loss_fn: Loss function that takes (model, data) -> loss
        meta_optimizer: Optimizer for meta-parameters (unused in Reptile)
        
    Returns:
        Average meta loss across the batch
    """
    # Get batch size from first tensor
    batch_size = query_data[0].shape[0]
    meta_losses = []
    adapted_params_list = []
    
    # Adapt to each task and collect adapted parameters
    for i in range(batch_size):
        # Extract single task data
        task_example_data = tuple(tensor[i] for tensor in example_data)
        task_query_data = tuple(tensor[i] for tensor in query_data)
        
        # Adapt to current task
        adapted_params = adapt_model(model, task_example_data, loss_fn, inner_lr, inner_steps)
        adapted_params_list.append(adapted_params)
        
        # Compute meta loss on query set for monitoring (using original model)
        meta_loss = loss_fn(model, task_query_data)
        meta_losses.append(meta_loss)
    
    # Reptile update: move towards average of adapted parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Average the parameter across all adapted models
            adapted_param_values = [adapted_params[name] for adapted_params in adapted_params_list]
            avg_adapted_param = torch.stack(adapted_param_values).mean(dim=0)
            
            # Reptile update: interpolate between original and average adapted
            param.data = param.data + meta_lr * (avg_adapted_param - param.data)
    
    # Return average meta loss for monitoring
    avg_meta_loss = torch.stack(meta_losses).mean()
    
    # Check for NaN/Inf in meta loss
    if not torch.isfinite(avg_meta_loss):
        raise ValueError(f"Non-finite meta loss detected: {avg_meta_loss.item()}")
        
    return avg_meta_loss.item()