import torch
import copy
from typing import List, Tuple, Callable, Any


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    """Create a deep copy of model parameters.
    
    Args:
        model: The model to clone
        
    Returns:
        A deep copy of the model
    """
    return copy.deepcopy(model)


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
    adapted_model = clone_model(model)
    
    for _ in range(inner_steps):
        loss = loss_fn(adapted_model, *support_data)
        
        # Compute gradients with respect to adapted model parameters
        grads = torch.autograd.grad(
            loss, 
            adapted_model.parameters(), 
            create_graph=True,  # Important for MAML second-order gradients
            retain_graph=True
        )
        
        # Manual parameter update
        with torch.no_grad():
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data = param.data - inner_lr * grad
    
    return adapted_model


def compute_meta_loss(
    model: torch.nn.Module,
    query_data: Tuple[torch.Tensor, ...],
    loss_fn: Callable
) -> torch.Tensor:
    """Compute loss on query data after adaptation.
    
    Args:
        model: The adapted model
        query_data: Tuple of (inputs, targets, ...) for evaluation
        loss_fn: Loss function for the task
        
    Returns:
        Loss tensor
    """
    return loss_fn(model, *query_data)


def meta_update_step(
    model: torch.nn.Module,
    task_batch: List[Tuple],
    inner_lr: float,
    meta_lr: float,
    inner_steps: int,
    loss_fn: Callable,
    meta_optimizer: torch.optim.Optimizer
) -> float:
    """Single meta-learning update step across a batch of tasks.
    
    Args:
        model: The meta-model to update
        task_batch: List of tasks, each task is (support_data, query_data)
        inner_lr: Learning rate for inner loop adaptation
        meta_lr: Learning rate for meta updates (unused, handled by optimizer)
        inner_steps: Number of inner loop steps
        loss_fn: Loss function for tasks
        meta_optimizer: Optimizer for meta-parameters
        
    Returns:
        Average meta loss across the batch
    """
    meta_optimizer.zero_grad()
    meta_losses = []
    
    for support_data, query_data in task_batch:
        # Adapt to current task
        adapted_model = adapt_model(model, support_data, loss_fn, inner_lr, inner_steps)
        
        # Compute meta loss on query set
        meta_loss = compute_meta_loss(adapted_model, query_data, loss_fn)
        meta_losses.append(meta_loss)
    
    # Average meta loss across tasks
    avg_meta_loss = torch.stack(meta_losses).mean()
    
    # Backpropagate through the meta-parameters
    avg_meta_loss.backward()
    meta_optimizer.step()
    
    return avg_meta_loss.item()