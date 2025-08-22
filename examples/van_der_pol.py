import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from meta_learning.maml import adapt_model, meta_update_step
from meta_learning.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from meta_learning.model.mlp import MLP
from datasets.van_der_pol import VanDerPolDataset, van_der_pol

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
    dt_range=(0.1, 0.1),
)

dataloader = DataLoader(dataset, batch_size=16)
dataloader_iter = iter(dataloader)


# Create model

model = NeuralODE(
    ode_func=ODEFunc(model=MLP(layer_sizes=[3, 128, 128, 2])),
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


# Save model
torch.save(model.state_dict(), "van_der_pol_model.pth")

# Load model
model = NeuralODE(
    ode_func=ODEFunc(model=MLP(layer_sizes=[3, 128, 128, 2])),
    integrator=rk4_step,
).to(device)
model.load_state_dict(torch.load("van_der_pol_model.pth"))
model.eval()


# Get a random batch of 9 tasks for evaluation
eval_dataloader = DataLoader(dataset, batch_size=9)
batch = next(iter(eval_dataloader))
mu, y0, dt, y1, y0_example, dt_example, y1_example = batch

# Move to device
mu = mu.to(device)
y0 = y0.to(device)
dt = dt.to(device)
y1 = y1.to(device)
y0_example = y0_example.to(device)
dt_example = dt_example.to(device)
y1_example = y1_example.to(device)

# Match function-encoder repo parameters
s = 0.1  # time step
n = int(10 / s)  # number of steps = 100
_dt = torch.tensor([s]).to(device)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for idx in range(9):
    _mu = mu[idx].item()
    _y0 = torch.empty(1, 2, device=device).uniform_(*dataloader.dataset.y0_range)

    # Adapt model to this specific task
    example_data = (y0_example[idx], dt_example[idx], y1_example[idx])
    adapted_model = adapt_model(
        model=model,
        example_data=example_data,
        loss_fn=loss_fn,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
    )

    # Generate true trajectory rollout (matching function-encoder repo exactly)
    x = _y0.clone()
    y = [x]  # Remove batch dimension for list
    for k in range(n):
        x = rk4_step(van_der_pol, x, _dt, mu=_mu) + x
        y.append(x)  # Remove batch dimension for list
    true_trajectory = torch.stack(y, dim=0)  # Use stack instead of cat

    # Generate predicted trajectory rollout
    x = _y0.clone()
    pred = [x]  # Remove batch dimension for list
    for k in range(n):
        with torch.no_grad():
            x_next = adapted_model((x, _dt))
            x = x_next + x
            pred.append(x)  # Remove batch dimension for list
    pred_trajectory = torch.stack(pred, dim=0)  # Use stack instead of cat

    # Plot phase portrait
    ax = axes[idx]

    # Convert to numpy for plotting
    y_np = true_trajectory.cpu().numpy().squeeze()
    pred_np = pred_trajectory.cpu().numpy().squeeze()

    # Plot exactly like function-encoder repo
    ax.plot(y_np[:, 0], y_np[:, 1], label="True")
    ax.plot(pred_np[:, 0], pred_np[:, 1], label="Predicted")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Compute trajectory error
    traj_error = torch.mean((true_trajectory - pred_trajectory) ** 2).item()
    print(f"Trajectory MSE: {traj_error:.6f}")

# Add legend exactly like function-encoder repo
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=2,
    frameon=False,
)

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Make room for legend
plt.show()
