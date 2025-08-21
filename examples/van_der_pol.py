import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from meta_learning.maml import adapt_model, meta_update_step
from meta_learning.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from meta_learning.model.mlp import MLP
from examples.datasets.van_der_pol import VanDerPolDataset

import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)

# Load dataset

dataset = VanDerPolDataset(
    n_points=1000,
    n_example_points=100,
    mu_range=(0.5, 2.5),
    y0_range=(-3.5, 3.5),
    dt_range=(0.01, 0.1),
)

dataloader = DataLoader(dataset, batch_size=16)
dataloader_iter = iter(dataloader)


# Create model

model = NeuralODE(
    ode_func=ODEFunc(model=MLP(layer_sizes=[3, 64, 64, 2])),
    integrator=rk4_step,
).to(device)


# Train model


def loss_fn(model, data):
    y0, dt, y1 = data
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)

    pred = model((y0, dt))

    return torch.nn.functional.mse_loss(pred, y1)


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

with tqdm.trange(num_steps) as tqdm_bar:
    for step in tqdm_bar:
        # Get batch of tasks
        batch = next(dataloader_iter)
        _, y0, dt, y1, y0_example, dt_example, y1_example = batch

        query_data = (y0, dt, y1)
        example_data = (y0_example, dt_example, y1_example)

        # Meta update step
        meta_loss = meta_update_step(
            model=model,
            query_data=query_data,
            example_data=example_data,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            loss_fn=loss_fn,
            meta_optimizer=meta_optimizer,
        )
        meta_losses.append(meta_loss)

        tqdm_bar.set_postfix_str(f"loss: {meta_loss:.2e}")


# Evaluate adaptation

adaptation_losses = []

dataloader = DataLoader(dataset, batch_size=1)
dataloader_iter = iter(dataloader)

for i in range(10):
    batch = next(dataloader_iter)
    _, y0, dt, y1, y0_example, dt_example, y1_example = batch

    query_data = (y0, dt, y1)
    example_data = (y0_example, dt_example, y1_example)

    # Test adaptation
    adapted_model = adapt_model(
        model=model,
        query_data=query_data,
        example_data=example_data,
        loss_fn=loss_fn,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
    )

    # Compute test loss using adapted model
    test_loss = loss_fn(adapted_model, *query_data).item()
    adaptation_losses.append(test_loss)

# Plot results

plt.figure(figsize=(10, 6))
plt.plot(meta_losses)
plt.title("Meta-Learning Training - Meta Loss")
plt.xlabel("Training Step")
plt.ylabel("Meta Loss")
plt.yscale("log")
plt.grid(True)
plt.savefig("/workspaces/maml-meta-learning/training_curve.png")
plt.show()
