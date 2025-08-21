import torch
import copy
from typing import List, Tuple, Callable, Any


def adapt_model(
    model: torch.nn.Module,
    support_data: Tuple[torch.Tensor, ...],
    loss_fn: Callable,
    inner_lr: float,
    inner_steps: int
) -> torch.nn.Module:
    """Adapt model to a specific task using support data.
    
    Args:
        model: The model to adapt
        support_data: Tuple of (inputs, targets, ...) for adaptation
        loss_fn: Loss function for the task
        inner_lr: Learning rate for inner loop adaptation
        inner_steps: Number of gradient steps for adaptation
        
    Returns:
        Adapted model
    """
    # Clone model for task-specific adaptation
    adapted_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)
    
    # Standard SGD training for Reptile
    for _ in range(inner_steps):
        optimizer.zero_grad()
        loss = loss_fn(adapted_model, *support_data)
        loss.backward()
        optimizer.step()
    
    return adapted_model


def meta_update_step(
    model: torch.nn.Module,
    batch: Tuple[torch.Tensor, ...],
    inner_lr: float,
    meta_lr: float,
    inner_steps: int,
    loss_fn: Callable,
    meta_optimizer: torch.optim.Optimizer
) -> float:
    """Single meta-learning update step across a batch of tasks.
    
    Args:
        model: The meta-model to update
        batch: Batch from DataLoader (mu, y0, dt, y1, y0_example, dt_example, y1_example)
        inner_lr: Learning rate for inner loop adaptation
        meta_lr: Learning rate for meta updates (Reptile step size)
        inner_steps: Number of inner loop steps
        loss_fn: Loss function for tasks
        meta_optimizer: Optimizer for meta-parameters (unused in Reptile)
        
    Returns:
        Average meta loss across the batch
    """
    mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
    
    batch_size = y0.shape[0]
    meta_losses = []
    adapted_models = []
    
    # Adapt to each task and collect adapted parameters
    for i in range(batch_size):
        support_data = (y0_example[i], dt_example[i], y1_example[i])
        query_data = (y0[i], dt[i], y1[i])
        
        # Adapt to current task
        adapted_model = adapt_model(model, support_data, loss_fn, inner_lr, inner_steps)
        adapted_models.append(adapted_model)
        
        # Compute meta loss on query set for monitoring
        meta_loss = loss_fn(adapted_model, *query_data)
        meta_losses.append(meta_loss)
    
    # Reptile update: move towards average of adapted parameters
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            # Average the i-th parameter across all adapted models
            adapted_params = [list(adapted_model.parameters())[i] for adapted_model in adapted_models]
            avg_adapted_param = torch.stack(adapted_params).mean(dim=0)
            
            # Reptile update: interpolate between original and average adapted
            param.data = param.data + meta_lr * (avg_adapted_param - param.data)
    
    # Return average meta loss for monitoring
    avg_meta_loss = torch.stack(meta_losses).mean()
    return avg_meta_loss.item()