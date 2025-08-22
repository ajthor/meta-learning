import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from meta_learning.maml import (
    adapt_model as maml_adapt,
    meta_update_step as maml_meta_step,
)
from meta_learning.fomaml import (
    adapt_model as fomaml_adapt,
    meta_update_step as fomaml_meta_step,
)
from meta_learning.reptile import (
    adapt_model as reptile_adapt,
    meta_update_step as reptile_meta_step,
)
from meta_learning.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from meta_learning.model.mlp import MLP
from meta_learning.adapted_model import AdaptedParameterModel
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


# Create models for each algorithm
def create_neural_ode():
    return NeuralODE(
        ode_func=ODEFunc(model=MLP(layer_sizes=[3, 64, 64, 2])),
        integrator=rk4_step,
    ).to(device)


maml_model = create_neural_ode()
fomaml_model = create_neural_ode()
reptile_model = create_neural_ode()
plain_model = create_neural_ode()


# Loss function
def loss_fn(model, data):
    y0, dt, y1 = data
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)

    pred = model((y0, dt))
    return torch.nn.functional.mse_loss(pred, y1)


# Hyperparameters
inner_lr = 1e-2
meta_lr = 1e-3
inner_steps = 5
num_steps = 1000

# Optimizers
maml_optimizer = torch.optim.Adam(maml_model.parameters(), lr=meta_lr)
fomaml_optimizer = torch.optim.Adam(fomaml_model.parameters(), lr=meta_lr)
plain_optimizer = torch.optim.Adam(plain_model.parameters(), lr=meta_lr)

print("Training MAML model...")
maml_losses = []
dataloader_iter = iter(dataloader)

with tqdm.trange(num_steps, desc="MAML") as tqdm_bar:
    for step in tqdm_bar:
        batch = next(dataloader_iter)
        _, y0, dt, y1, y0_example, dt_example, y1_example = batch

        query_data = (y0, dt, y1)
        example_data = (y0_example, dt_example, y1_example)

        meta_loss = maml_meta_step(
            model=maml_model,
            query_data=query_data,
            example_data=example_data,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            loss_fn=loss_fn,
            meta_optimizer=maml_optimizer,
        )
        maml_losses.append(meta_loss)
        tqdm_bar.set_postfix_str(f"loss: {meta_loss:.2e}")

torch.save(maml_model.state_dict(), "van_der_pol_maml_model.pth")

print("Training FO-MAML model...")
fomaml_losses = []
dataloader_iter = iter(dataloader)

with tqdm.trange(num_steps, desc="FO-MAML") as tqdm_bar:
    for step in tqdm_bar:
        batch = next(dataloader_iter)
        _, y0, dt, y1, y0_example, dt_example, y1_example = batch

        query_data = (y0, dt, y1)
        example_data = (y0_example, dt_example, y1_example)

        meta_loss = fomaml_meta_step(
            model=fomaml_model,
            query_data=query_data,
            example_data=example_data,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            loss_fn=loss_fn,
            meta_optimizer=fomaml_optimizer,
        )
        fomaml_losses.append(meta_loss)
        tqdm_bar.set_postfix_str(f"loss: {meta_loss:.2e}")

torch.save(fomaml_model.state_dict(), "van_der_pol_fomaml_model.pth")

print("Training Reptile model...")
reptile_losses = []
dataloader_iter = iter(dataloader)

with tqdm.trange(num_steps, desc="Reptile") as tqdm_bar:
    for step in tqdm_bar:
        batch = next(dataloader_iter)
        _, y0, dt, y1, y0_example, dt_example, y1_example = batch

        query_data = (y0, dt, y1)
        example_data = (y0_example, dt_example, y1_example)

        meta_loss = reptile_meta_step(
            model=reptile_model,
            query_data=query_data,
            example_data=example_data,
            inner_lr=inner_lr,
            meta_lr=meta_lr,
            inner_steps=inner_steps,
            loss_fn=loss_fn,
        )
        reptile_losses.append(meta_loss)
        tqdm_bar.set_postfix_str(f"loss: {meta_loss:.2e}")

torch.save(reptile_model.state_dict(), "van_der_pol_reptile_model.pth")

print("Training plain Neural ODE...")
plain_losses = []
dataloader_iter = iter(dataloader)

with tqdm.trange(num_steps, desc="Plain Neural ODE") as tqdm_bar:
    for step in tqdm_bar:
        batch = next(dataloader_iter)
        _, y0, dt, y1, y0_example, dt_example, y1_example = batch

        query_data = (y0, dt, y1)
        example_data = (y0_example, dt_example, y1_example)

        plain_optimizer.zero_grad()

        loss = loss_fn(plain_model, query_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plain_model.parameters(), max_norm=1.0)
        plain_optimizer.step()

        plain_losses.append(loss.item())
        tqdm_bar.set_postfix_str(f"loss: {loss.item():.2e}")

torch.save(plain_model.state_dict(), "van_der_pol_plain_model.pth")

# Set all models to eval mode
maml_model.eval()
fomaml_model.eval()
reptile_model.eval()
plain_model.eval()

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

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for idx in range(9):
    _mu = mu[idx].item()
    _y0 = torch.empty(1, 2, device=device).uniform_(*dataloader.dataset.y0_range)

    # Adapt meta-learning models to this specific task
    example_data = (y0_example[idx], dt_example[idx], y1_example[idx])

    maml_adapted = maml_adapt(
        model=maml_model,
        example_data=example_data,
        loss_fn=loss_fn,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
    )

    fomaml_adapted = fomaml_adapt(
        model=fomaml_model,
        example_data=example_data,
        loss_fn=loss_fn,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
    )

    reptile_adapted_params = reptile_adapt(
        model=reptile_model,
        example_data=example_data,
        loss_fn=loss_fn,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
    )
    reptile_adapted = AdaptedParameterModel(reptile_model, reptile_adapted_params)

    # Generate true trajectory rollout
    x = _y0.clone()
    y = [x]
    for k in range(n):
        x = rk4_step(van_der_pol, x, _dt, mu=_mu) + x
        y.append(x)
    true_trajectory = torch.stack(y, dim=0)

    # Generate predicted trajectory rollouts for each model
    def generate_trajectory(model):
        x = _y0.clone()
        pred = [x]
        for k in range(n):
            with torch.no_grad():
                x_next = model((x, _dt))
                x = x_next + x
                pred.append(x)
        return torch.stack(pred, dim=0)

    maml_trajectory = generate_trajectory(maml_adapted)
    fomaml_trajectory = generate_trajectory(fomaml_adapted)
    reptile_trajectory = generate_trajectory(reptile_adapted)
    plain_trajectory = generate_trajectory(plain_model)

    # Plot phase portrait
    ax = axes[idx]

    # Convert to numpy for plotting
    true_np = true_trajectory.cpu().numpy().squeeze()
    maml_np = maml_trajectory.cpu().numpy().squeeze()
    fomaml_np = fomaml_trajectory.cpu().numpy().squeeze()
    reptile_np = reptile_trajectory.cpu().numpy().squeeze()
    plain_np = plain_trajectory.cpu().numpy().squeeze()

    ax.plot(true_np[:, 0], true_np[:, 1], "k-", linewidth=2, label="True")
    ax.plot(maml_np[:, 0], maml_np[:, 1], "r--", linewidth=1.5, label="MAML")
    ax.plot(fomaml_np[:, 0], fomaml_np[:, 1], "b--", linewidth=1.5, label="FO-MAML")
    ax.plot(reptile_np[:, 0], reptile_np[:, 1], "g--", linewidth=1.5, label="Reptile")
    ax.plot(
        plain_np[:, 0], plain_np[:, 1], "m:", linewidth=1.5, label="Plain Neural ODE"
    )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    # ax.set_title(f"μ = {_mu:.2f}")

    # Compute trajectory errors
    maml_error = torch.mean((true_trajectory - maml_trajectory) ** 2).item()
    fomaml_error = torch.mean((true_trajectory - fomaml_trajectory) ** 2).item()
    reptile_error = torch.mean((true_trajectory - reptile_trajectory) ** 2).item()
    plain_error = torch.mean((true_trajectory - plain_trajectory) ** 2).item()

    print(
        f"Task {idx+1} (μ={_mu:.2f}) - MAML MSE: {maml_error:.6f}, FO-MAML MSE: {fomaml_error:.6f}, Reptile MSE: {reptile_error:.6f}, Plain MSE: {plain_error:.6f}"
    )

# Add legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=5,
    frameon=False,
)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
