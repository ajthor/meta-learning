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
    
    for _ in range(inner_steps):
        loss = loss_fn(adapted_model, *support_data)
        
        # Compute gradients without computational graph for FOMAML
        grads = torch.autograd.grad(
            loss, 
            adapted_model.parameters(), 
            create_graph=False,  # First-order only - no second derivatives
            retain_graph=False
        )
        
        # Manual parameter update
        with torch.no_grad():
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data = param.data - inner_lr * grad
    
    return adapted_model


def meta_update_step(
    model: torch.nn.Module,
    batch: Tuple[torch.Tensor, ...],
    inner_lr: float,
    inner_steps: int,
    loss_fn: Callable,
    meta_optimizer: torch.optim.Optimizer
) -> float:
    """Single meta-learning update step across a batch of tasks.
    
    Args:
        model: The meta-model to update
        batch: Batch from DataLoader (mu, y0, dt, y1, y0_example, dt_example, y1_example)
        inner_lr: Learning rate for inner loop adaptation
        inner_steps: Number of inner loop steps
        loss_fn: Loss function for tasks
        meta_optimizer: Optimizer for meta-parameters
        
    Returns:
        Average meta loss across the batch
    """
    meta_optimizer.zero_grad()
    mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
    
    batch_size = y0.shape[0]
    meta_losses = []
    
    # Process each task in the batch
    for i in range(batch_size):
        support_data = (y0_example[i], dt_example[i], y1_example[i])
        query_data = (y0[i], dt[i], y1[i])
        
        # Adapt to current task
        adapted_model = adapt_model(model, support_data, loss_fn, inner_lr, inner_steps)
        
        # Compute meta loss on query set
        meta_loss = loss_fn(adapted_model, *query_data)
        meta_losses.append(meta_loss)
    
    # Average meta loss across tasks
    avg_meta_loss = torch.stack(meta_losses).mean()
    
    # Backpropagate through the meta-parameters (first-order only)
    avg_meta_loss.backward()
    meta_optimizer.step()
    
    return avg_meta_loss.item()