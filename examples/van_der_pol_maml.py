import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
sys.path.append('/workspaces/meta-learning')

from meta_learning.maml import adapt_model, meta_update_step, clone_model
from meta_learning.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from meta_learning.model.mlp import MLP
from examples.datasets.van_der_pol import VanDerPolDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model setup - Neural ODE
hidden_dim = 64
mlp_backbone = MLP(
    layer_sizes=[3, hidden_dim, hidden_dim, 2],  # [time, x1, x2] -> [dx1/dt, dx2/dt]
    activation=nn.Tanh()
)
ode_func = ODEFunc(mlp_backbone)
model = NeuralODE(ode_func=ode_func, integrator=rk4_step).to(device)

# Loss function for van der Pol regression
def loss_fn(model, y0, dt, y1):
    """Loss function for van der Pol ODE prediction."""
    pred_y1 = model((y0, dt))
    return nn.MSELoss()(pred_y1, y1)

# Dataset and hyperparameters
dataset = VanDerPolDataset(
    n_points=1000,
    n_example_points=100,
    mu_range=(0.5, 2.5),
    y0_range=(-3.5, 3.5),
    dt_range=(0.01, 0.1)
)

# Meta-learning hyperparameters
inner_lr = 1e-2
meta_lr = 1e-3
inner_steps = 5
meta_batch_size = 16
num_epochs = 1000

# Meta optimizer
meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

# Training loop
print("Starting MAML training...")
meta_losses = []

for epoch in range(num_epochs):
    # Sample batch of tasks
    task_batch = []
    for _ in range(meta_batch_size):
        mu, y0, dt, y1, y0_example, dt_example, y1_example = next(iter(dataset))
        
        # Move to device
        y0 = y0.to(device)
        dt = dt.to(device)
        y1 = y1.to(device)
        y0_example = y0_example.to(device)
        dt_example = dt_example.to(device)
        y1_example = y1_example.to(device)
        
        support_data = (y0_example, dt_example, y1_example)
        query_data = (y0, dt, y1)
        task_batch.append((support_data, query_data))
    
    # Meta update step
    meta_loss = meta_update_step(
        model, task_batch, inner_lr, meta_lr, inner_steps, loss_fn, meta_optimizer
    )
    meta_losses.append(meta_loss)
    
    # Logging
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Meta Loss: {meta_loss:.6f}")

# Evaluation: Test adaptation performance
print("\nEvaluating adaptation performance...")
test_tasks = []
adaptation_losses = []

for i in range(10):
    mu, y0, dt, y1, y0_example, dt_example, y1_example = next(iter(dataset))
    
    # Move to device
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)
    
    support_data = (y0_example, dt_example, y1_example)
    query_data = (y0, dt, y1)
    
    # Test adaptation
    adapted_model = adapt_model(model, support_data, loss_fn, inner_lr, inner_steps)
    test_loss = loss_fn(adapted_model, *query_data).item()
    adaptation_losses.append(test_loss)
    
    print(f"Task {i+1} (μ={mu.item():.2f}): Adaptation loss = {test_loss:.6f}")

print(f"\nAverage adaptation loss: {torch.tensor(adaptation_losses).mean():.6f}")

# Plot training curve
plt.figure(figsize=(10, 6))
plt.plot(meta_losses)
plt.title('MAML Training - Meta Loss')
plt.xlabel('Epoch')
plt.ylabel('Meta Loss')
plt.yscale('log')
plt.grid(True)
plt.savefig('/workspaces/meta-learning/maml_training_curve.png')
plt.show()

print("MAML training completed!")
print(f"Final meta loss: {meta_losses[-1]:.6f}")
print(f"Model saved and training curve plotted.")
