import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
sys.path.append('/workspaces/meta-learning')

from meta_learning.maml import adapt_model, meta_update_step
from meta_learning.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from meta_learning.model.mlp import MLP
from examples.datasets.van_der_pol import VanDerPolDataset

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset setup
dataset = VanDerPolDataset(
    n_points=1000,
    n_example_points=100,
    mu_range=(0.5, 2.5),
    y0_range=(-3.5, 3.5),
    dt_range=(0.01, 0.1)
)

# DataLoader setup
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size)
dataloader_iter = iter(dataloader)

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

# Meta-learning hyperparameters
inner_lr = 1e-2
meta_lr = 1e-3
inner_steps = 5
num_steps = 1000

# Meta optimizer
meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

# Train the model
print("Starting meta-learning training...")
meta_losses = []

for step in range(num_steps):
    # Get batch of tasks
    batch = next(dataloader_iter)
    batch = tuple(tensor.to(device) for tensor in batch)
    
    # Meta update step
    meta_loss = meta_update_step(
        model, batch, inner_lr, inner_steps, loss_fn, meta_optimizer
    )
    meta_losses.append(meta_loss)
    
    # Logging
    if step % 100 == 0:
        print(f"Step {step}, Meta Loss: {meta_loss:.6f}")

print(f"Training completed. Final meta loss: {meta_losses[-1]:.6f}")

# Evaluate adaptation performance
print("\nEvaluating adaptation performance...")
adaptation_losses = []

for i in range(10):
    # Get a test batch
    test_batch = next(dataloader_iter)
    test_batch = tuple(tensor.to(device) for tensor in test_batch)
    mu, y0, dt, y1, y0_example, dt_example, y1_example = test_batch
    
    # Use first task from batch for evaluation
    support_data = (y0_example[0], dt_example[0], y1_example[0])
    query_data = (y0[0], dt[0], y1[0])
    
    # Test adaptation
    adapted_model = adapt_model(model, support_data, loss_fn, inner_lr, inner_steps)
    test_loss = loss_fn(adapted_model, *query_data).item()
    adaptation_losses.append(test_loss)
    
    print(f"Task {i+1} (μ={mu[0].item():.2f}): Adaptation loss = {test_loss:.6f}")

print(f"Average adaptation loss: {torch.tensor(adaptation_losses).mean():.6f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(meta_losses)
plt.title('Meta-Learning Training - Meta Loss')
plt.xlabel('Training Step')
plt.ylabel('Meta Loss')
plt.yscale('log')
plt.grid(True)
plt.savefig('/workspaces/meta-learning/training_curve.png')
plt.show()
