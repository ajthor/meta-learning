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


def evaluate_rollout_accuracy(model, n_rollouts, k_steps, y0_range, dt, mu):
    """Evaluate model accuracy via k-step rollouts from random initial conditions.

    Args:
        model: The adapted model to evaluate
        n_rollouts: Number of random initial conditions to test
        k_steps: Length of each rollout
        y0_range: Range for sampling random initial conditions
        dt: Time step tensor
        mu: Van der Pol parameter

    Returns:
        Average MSE across all rollouts
    """
    total_error = 0.0

    for _ in range(n_rollouts):
        # Sample random initial condition
        y0_random = torch.empty(1, 2, device=device).uniform_(*y0_range)

        # Generate true trajectory
        x_true = y0_random.clone()
        true_traj = [x_true]
        for step in range(k_steps):
            x_true = rk4_step(van_der_pol, x_true, dt, mu=mu) + x_true
            true_traj.append(x_true)
        true_trajectory = torch.stack(true_traj, dim=0)

        # Generate predicted trajectory
        x_pred = y0_random.clone()
        pred_traj = [x_pred]
        for step in range(k_steps):
            with torch.no_grad():
                x_next = model((x_pred, dt))
                x_pred = x_next + x_pred
                pred_traj.append(x_pred)
        pred_trajectory = torch.stack(pred_traj, dim=0)

        # Compute rollout error
        rollout_error = torch.mean((true_trajectory - pred_trajectory) ** 2).item()
        total_error += rollout_error

    return total_error / n_rollouts


# Hyperparameters
inner_lr = 1e-2
meta_lr = 1e-3
inner_steps = 5
num_steps = 1000

# Online evaluation parameters
n_rollouts = 10  # Number of random ICs per evaluation (reduced for testing)
k_steps = 10  # Rollout length for evaluation (reduced for testing)
online_adaptation_steps = 2  # Quick adaptation steps for online updates

# Optimizers
maml_optimizer = torch.optim.Adam(maml_model.parameters(), lr=meta_lr)
fomaml_optimizer = torch.optim.Adam(fomaml_model.parameters(), lr=meta_lr)
plain_optimizer = torch.optim.Adam(plain_model.parameters(), lr=meta_lr)

# print("Training MAML model...")
# maml_losses = []
# dataloader_iter = iter(dataloader)

# with tqdm.trange(num_steps, desc="MAML") as tqdm_bar:
#     for step in tqdm_bar:
#         batch = next(dataloader_iter)
#         _, y0, dt, y1, y0_example, dt_example, y1_example = batch

#         query_data = (y0, dt, y1)
#         example_data = (y0_example, dt_example, y1_example)

#         meta_loss = maml_meta_step(
#             model=maml_model,
#             query_data=query_data,
#             example_data=example_data,
#             inner_lr=inner_lr,
#             inner_steps=inner_steps,
#             loss_fn=loss_fn,
#             meta_optimizer=maml_optimizer,
#         )
#         maml_losses.append(meta_loss)
#         tqdm_bar.set_postfix_str(f"loss: {meta_loss:.2e}")

# torch.save(maml_model.state_dict(), "van_der_pol_maml_model.pth")

# print("Training FO-MAML model...")
# fomaml_losses = []
# dataloader_iter = iter(dataloader)

# with tqdm.trange(num_steps, desc="FO-MAML") as tqdm_bar:
#     for step in tqdm_bar:
#         batch = next(dataloader_iter)
#         _, y0, dt, y1, y0_example, dt_example, y1_example = batch

#         query_data = (y0, dt, y1)
#         example_data = (y0_example, dt_example, y1_example)

#         meta_loss = fomaml_meta_step(
#             model=fomaml_model,
#             query_data=query_data,
#             example_data=example_data,
#             inner_lr=inner_lr,
#             inner_steps=inner_steps,
#             loss_fn=loss_fn,
#             meta_optimizer=fomaml_optimizer,
#         )
#         fomaml_losses.append(meta_loss)
#         tqdm_bar.set_postfix_str(f"loss: {meta_loss:.2e}")

# torch.save(fomaml_model.state_dict(), "van_der_pol_fomaml_model.pth")

# print("Training Reptile model...")
# reptile_losses = []
# dataloader_iter = iter(dataloader)

# with tqdm.trange(num_steps, desc="Reptile") as tqdm_bar:
#     for step in tqdm_bar:
#         batch = next(dataloader_iter)
#         _, y0, dt, y1, y0_example, dt_example, y1_example = batch

#         query_data = (y0, dt, y1)
#         example_data = (y0_example, dt_example, y1_example)

#         meta_loss = reptile_meta_step(
#             model=reptile_model,
#             query_data=query_data,
#             example_data=example_data,
#             inner_lr=inner_lr,
#             meta_lr=meta_lr,
#             inner_steps=inner_steps,
#             loss_fn=loss_fn,
#         )
#         reptile_losses.append(meta_loss)
#         tqdm_bar.set_postfix_str(f"loss: {meta_loss:.2e}")

# torch.save(reptile_model.state_dict(), "van_der_pol_reptile_model.pth")

# print("Training plain Neural ODE...")
# plain_losses = []
# dataloader_iter = iter(dataloader)

# with tqdm.trange(num_steps, desc="Plain Neural ODE") as tqdm_bar:
#     for step in tqdm_bar:
#         batch = next(dataloader_iter)
#         _, y0, dt, y1, y0_example, dt_example, y1_example = batch

#         query_data = (y0, dt, y1)
#         example_data = (y0_example, dt_example, y1_example)

#         plain_optimizer.zero_grad()

#         loss = loss_fn(plain_model, query_data)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(plain_model.parameters(), max_norm=1.0)
#         plain_optimizer.step()

#         plain_losses.append(loss.item())
#         tqdm_bar.set_postfix_str(f"loss: {loss.item():.2e}")

# torch.save(plain_model.state_dict(), "van_der_pol_plain_model.pth")

# Load trained models for evaluation
maml_model.load_state_dict(torch.load("van_der_pol_maml_model.pth"))
fomaml_model.load_state_dict(torch.load("van_der_pol_fomaml_model.pth"))
reptile_model.load_state_dict(torch.load("van_der_pol_reptile_model.pth"))
plain_model.load_state_dict(torch.load("van_der_pol_plain_model.pth"))

# Set all models to eval mode
maml_model.eval()
fomaml_model.eval()
reptile_model.eval()
plain_model.eval()

# Online adaptation evaluation
print("Starting online adaptation evaluation...")

# Get a random batch of 9 tasks for evaluation
eval_dataloader = DataLoader(dataset, batch_size=9)
batch = next(iter(eval_dataloader))
mu_batch, y0_batch, dt_batch, y1_batch, y0_example, dt_example, y1_example = batch

# Move to device
mu_batch = mu_batch.to(device)
y0_batch = y0_batch.to(device)

# Evaluation parameters
s = 0.1  # time step
n_trajectory_steps = (
    50  # Number of trajectory steps to generate measurements from (reduced for testing)
)
_dt = torch.tensor([s]).to(device)

# Storage for learning curves
n_tasks = 3  # Reduced for testing
maml_learning_curves = []
fomaml_learning_curves = []
reptile_learning_curves = []
plain_learning_curves = []

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
if n_tasks == 1:
    axes = [axes]

for idx in range(n_tasks):
    _mu = mu_batch[idx].item()
    _y0 = y0_batch[idx, 0:1]  # Use first query point as initial condition

    print(f"Evaluating task {idx+1}/9 (μ={_mu:.2f})")

    # Generate true trajectory for this task (source of measurements)
    x_true = _y0.clone()
    true_measurements = [x_true]
    for step in range(n_trajectory_steps):
        x_true = rk4_step(van_der_pol, x_true, _dt, mu=_mu) + x_true
        true_measurements.append(x_true)

    # Initialize models for this task (no initial adaptation)
    maml_current = maml_model
    fomaml_current = fomaml_model
    reptile_current = reptile_model
    plain_current = plain_model

    # Track learning curves for this task
    task_maml_errors = []
    task_fomaml_errors = []
    task_reptile_errors = []
    task_plain_errors = []

    # Online adaptation loop: at each time step, adapt using measurement and evaluate
    for t in range(n_trajectory_steps):
        # Current measurement (y_t, dt, y_t+1)
        if t == 0:
            continue  # Skip first step (no previous measurement)

        y_t = true_measurements[t - 1]
        y_t_plus_1 = true_measurements[t]
        dy = rk4_step(
            van_der_pol, y_t, _dt, mu=_mu
        )  # Compute delta using true dynamics
        measurement_data = (y_t, _dt, dy)  # (state, dt, delta)

        # Adapt each model using current measurement
        maml_current = maml_adapt(
            model=maml_current,
            example_data=measurement_data,
            loss_fn=loss_fn,
            inner_lr=inner_lr,
            inner_steps=online_adaptation_steps,
        )

        fomaml_current = fomaml_adapt(
            model=fomaml_current,
            example_data=measurement_data,
            loss_fn=loss_fn,
            inner_lr=inner_lr,
            inner_steps=online_adaptation_steps,
        )

        # For Reptile, adapt from base model each time (as per Reptile philosophy)
        reptile_adapted_params = reptile_adapt(
            model=reptile_model,
            example_data=measurement_data,
            loss_fn=loss_fn,
            inner_lr=inner_lr,
            inner_steps=online_adaptation_steps,
        )
        reptile_current = AdaptedParameterModel(reptile_model, reptile_adapted_params)

        # For plain model, just do standard gradient step
        plain_optimizer.zero_grad()
        loss = loss_fn(plain_current, measurement_data)
        loss.backward()
        plain_optimizer.step()

        # Evaluate current models via rollout accuracy
        maml_error = evaluate_rollout_accuracy(
            maml_current, n_rollouts, k_steps, dataloader.dataset.y0_range, _dt, _mu
        )
        fomaml_error = evaluate_rollout_accuracy(
            fomaml_current, n_rollouts, k_steps, dataloader.dataset.y0_range, _dt, _mu
        )
        reptile_error = evaluate_rollout_accuracy(
            reptile_current, n_rollouts, k_steps, dataloader.dataset.y0_range, _dt, _mu
        )
        plain_error = evaluate_rollout_accuracy(
            plain_current, n_rollouts, k_steps, dataloader.dataset.y0_range, _dt, _mu
        )

        # Store errors for learning curves
        task_maml_errors.append(maml_error)
        task_fomaml_errors.append(fomaml_error)
        task_reptile_errors.append(reptile_error)
        task_plain_errors.append(plain_error)

    # Store learning curves for this task
    maml_learning_curves.append(task_maml_errors)
    fomaml_learning_curves.append(task_fomaml_errors)
    reptile_learning_curves.append(task_reptile_errors)
    plain_learning_curves.append(task_plain_errors)

    # Plot learning curve for this task
    ax = axes[idx]
    time_steps = list(range(1, len(task_maml_errors) + 1))

    ax.plot(time_steps, task_maml_errors, "r-", linewidth=1.5, label="MAML")
    ax.plot(time_steps, task_fomaml_errors, "b-", linewidth=1.5, label="FO-MAML")
    ax.plot(time_steps, task_reptile_errors, "g-", linewidth=1.5, label="Reptile")
    ax.plot(
        time_steps, task_plain_errors, "m-", linewidth=1.5, label="Plain Neural ODE"
    )

    ax.set_xlabel("Measurements Seen")
    ax.set_ylabel("Rollout MSE")
    ax.set_title(f"μ = {_mu:.2f}")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Print final errors
    print(
        f"Task {idx+1} (μ={_mu:.2f}) - Final MAML MSE: {task_maml_errors[-1]:.6f}, "
        f"FO-MAML MSE: {task_fomaml_errors[-1]:.6f}, "
        f"Reptile MSE: {task_reptile_errors[-1]:.6f}, "
        f"Plain MSE: {task_plain_errors[-1]:.6f}"
    )

# Convert learning curves to numpy for averaging
import numpy as np

maml_curves = np.array(maml_learning_curves)
fomaml_curves = np.array(fomaml_learning_curves)
reptile_curves = np.array(reptile_learning_curves)
plain_curves = np.array(plain_learning_curves)

# Add average learning curve plot
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
time_steps = list(range(1, maml_curves.shape[1] + 1))

# Plot mean and standard deviation
maml_mean = np.mean(maml_curves, axis=0)
maml_std = np.std(maml_curves, axis=0)
fomaml_mean = np.mean(fomaml_curves, axis=0)
fomaml_std = np.std(fomaml_curves, axis=0)
reptile_mean = np.mean(reptile_curves, axis=0)
reptile_std = np.std(reptile_curves, axis=0)
plain_mean = np.mean(plain_curves, axis=0)
plain_std = np.std(plain_curves, axis=0)

ax2.plot(time_steps, maml_mean, "r-", linewidth=2, label="MAML")
ax2.fill_between(
    time_steps, maml_mean - maml_std, maml_mean + maml_std, alpha=0.2, color="red"
)

ax2.plot(time_steps, fomaml_mean, "b-", linewidth=2, label="FO-MAML")
ax2.fill_between(
    time_steps,
    fomaml_mean - fomaml_std,
    fomaml_mean + fomaml_std,
    alpha=0.2,
    color="blue",
)

ax2.plot(time_steps, reptile_mean, "g-", linewidth=2, label="Reptile")
ax2.fill_between(
    time_steps,
    reptile_mean - reptile_std,
    reptile_mean + reptile_std,
    alpha=0.2,
    color="green",
)

ax2.plot(time_steps, plain_mean, "m-", linewidth=2, label="Plain Neural ODE")
ax2.fill_between(
    time_steps,
    plain_mean - plain_std,
    plain_mean + plain_std,
    alpha=0.2,
    color="magenta",
)

ax2.set_xlabel("Measurements Seen")
ax2.set_ylabel("Average Rollout MSE")
ax2.set_title("Average Online Learning Performance")
ax2.set_yscale("log")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add legend to individual task plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=4,
    frameon=False,
)

plt.figure(fig.number)
plt.tight_layout()
plt.subplots_adjust(top=0.9)

plt.figure(fig2.number)
plt.tight_layout()

plt.show()
