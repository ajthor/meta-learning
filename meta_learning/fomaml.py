import torch
from typing import Tuple, Callable
from collections import OrderedDict
from .adapted_model import AdaptedParameterModel


def adapt_model(
    model: torch.nn.Module,
    example_data: Tuple[torch.Tensor, ...],
    loss_fn: Callable,
    inner_lr: float,
    inner_steps: int
) -> AdaptedParameterModel:
    """Adapt model to a specific task using example data (first-order only).
    
    Args:
        model: The model to adapt
        example_data: Tuple of (inputs, targets, ...) for adaptation
        loss_fn: Loss function that takes (model, data) -> loss
        inner_lr: Learning rate for inner loop adaptation
        inner_steps: Number of gradient steps for adaptation
        
    Returns:
        AdaptedParameterModel with updated parameters
    """
    # Get initial parameters as OrderedDict
    params = OrderedDict(model.named_parameters())
    
    for _ in range(inner_steps):
        # Create adapted model for current parameters
        adapted_model = AdaptedParameterModel(model, params)
        
        # Compute loss using adapted model
        loss = loss_fn(adapted_model, example_data)
        
        # Check for NaN/Inf in loss
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss detected: {loss.item()}")
        
        # Compute gradients without computational graph for FO-MAML
        grads = torch.autograd.grad(
            loss, 
            params.values(), 
            create_graph=False,  # First-order only - no second derivatives
            retain_graph=False
        )
        
        # Update parameters (no computational graph maintained)
        with torch.no_grad():
            params = OrderedDict({
                name: param - inner_lr * grad
                for (name, param), grad in zip(params.items(), grads)
            })
    
    return AdaptedParameterModel(model, params)


def meta_update_step(
    model: torch.nn.Module,
    query_data: Tuple[torch.Tensor, ...],
    example_data: Tuple[torch.Tensor, ...],
    inner_lr: float,
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
        inner_steps: Number of inner loop steps
        loss_fn: Loss function that takes (model, data) -> loss
        meta_optimizer: Optimizer for meta-parameters
        
    Returns:
        Average meta loss across the batch
    """
    meta_optimizer.zero_grad()
    
    # Get batch size from first tensor
    batch_size = query_data[0].shape[0]
    meta_losses = []
    
    # Process each task in the batch
    for i in range(batch_size):
        # Extract single task data
        task_example_data = tuple(tensor[i] for tensor in example_data)
        task_query_data = tuple(tensor[i] for tensor in query_data)
        
        # Adapt to current task
        adapted_model = adapt_model(model, task_example_data, loss_fn, inner_lr, inner_steps)
        
        # Detach adapted parameters to ensure first-order approximation
        detached_params = OrderedDict({
            name: param.detach().requires_grad_(True)
            for name, param in adapted_model.adapted_params.items()
        })
        detached_adapted_model = AdaptedParameterModel(model, detached_params)
        
        # Compute meta loss on query set using detached adapted model
        meta_loss = loss_fn(detached_adapted_model, task_query_data)
        meta_losses.append(meta_loss)
    
    # Average meta loss across tasks
    avg_meta_loss = torch.stack(meta_losses).mean()
    
    # Check for NaN/Inf in meta loss
    if not torch.isfinite(avg_meta_loss):
        raise ValueError(f"Non-finite meta loss detected: {avg_meta_loss.item()}")
    
    # Backpropagate through the meta-parameters (first-order only)
    avg_meta_loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    meta_optimizer.step()
    
    return avg_meta_loss.item()