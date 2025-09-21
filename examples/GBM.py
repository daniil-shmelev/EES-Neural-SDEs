METHOD = "ees25"
DIR = "plots/options_" + METHOD

import os
if not os.path.exists(DIR):
    os.makedirs(DIR)

import os
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchsde

# Setup seed for reproducibility
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Parameters
NUM_SAMPLES = 250000
T = 25.0
R = 0.5
SIGMA = 1.5
S0 = 100.
STRIKES = np.linspace(90, 110, 20)
MATURITIES = np.linspace(0, T, 100)
BATCH_SIZE = 125000
SEED = 42
EPOCHS = 250

N = len(MATURITIES)
DT = T / N

def BS_call(S, K, T, r, sigma):

    if T == 0.:
        return max(S - K, 0.)

    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    #return S * scipy.stats.norm.cdf(d1) - K * np.exp(-r*T)* scipy.stats.norm.cdf(d2)
    return np.exp(r * T) * S * scipy.stats.norm.cdf(d1) - K * scipy.stats.norm.cdf(d2)

def options_data(S0, strikes, maturities, r, sigma):
    """
    Simulate the GBM

    Parameters:
    T (float): Total time.
    N (int): Number of time steps.
    mu (float): Long-term mean.
    sigma (float): Volatility.
    X0 (float): Initial value.

    Returns:
    np.ndarray: Simulated values of the OU process.
    """
    calls = torch.empty((len(strikes), len(maturities)))

    for i in range(len(strikes)):
        for j in range(len(maturities)):
            calls[i,j] = BS_call(S0, strikes[i], maturities[j], r, sigma)

    return calls

# Ensure reproducibility
seed_everything(SEED)

# Generate data
call_data = options_data(S0, STRIKES, MATURITIES, R, SIGMA)


class LipSwish(nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.nn.functional.sigmoid(x)


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim, num_layers, tanh=False, activation='lipswish'):
        super().__init__()

        if activation == 'lipswish':
            activation_fn = LipSwish()
        elif activation == 'sigmoid':
            activation_fn = Sigmoid()
        else:
            activation_fn = nn.ReLU()

        model = [nn.Linear(in_size, hidden_dim), activation_fn]
        for _ in range(num_layers - 1):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(activation_fn)
        model.append(nn.Linear(hidden_dim, out_size))
        if tanh:
            model.append(nn.Tanh())
        self._model = nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


class NeuralSDEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation='lipswish'):
        super(NeuralSDEFunc, self).__init__()
        self.sde_type = "stratonovich" # NSDE will learn the ito correction
        self.noise_type = "diagonal"  # or "scalar"

        self.f_net = MLP(input_dim + 1, input_dim, hidden_dim, num_layers, activation=activation)
        self.g_net = MLP(input_dim + 1, input_dim, hidden_dim, num_layers, activation=activation)

    def f(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        return self.f_net(torch.cat((t, y), dim=-1))

    def g(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        return self.g_net(torch.cat((t, y), dim=-1))


class NDE_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation='lipswish', vector_field=None):
        super(NDE_model, self).__init__()
        self.func = vector_field(input_dim, hidden_dim, num_layers, activation=activation)

    def forward(self, batch, times):
        y0 = torch.tensor([[S0]] * batch).to(times.device)

        z = torchsde.sdeint(sde=self.func,
                            y0=y0,
                            ts=times,
                            dt=DT,
                            method=METHOD)

        return z.permute(1, 0, 2)

input_dim = 1
hidden_dim = 8
num_layers = 2

model = NDE_model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, vector_field=NeuralSDEFunc, activation="lipswish").to(device)

num_epochs = EPOCHS
lr = 1e-2

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # decay factor


def my_loss(pred_paths, call_data):
    #pred_paths of shape (batch, T, dim)
    pred_calls = torch.empty(size = (len(STRIKES), len(MATURITIES)))

    assert (len(MATURITIES) == pred_paths.shape[1])

    for i in range(len(STRIKES)):
        for j in range(len(MATURITIES)):
            pred_calls[i,j] = torch.relu(pred_paths[:, j] - STRIKES[i]).mean()

    loss = pred_calls - call_data
    loss *= torch.exp(-R * torch.tensor(MATURITIES))
    loss = loss[:, ::10]
    loss = loss ** 2
    return loss.mean() / 2.



mse_loss = []

for epoch in range(1, num_epochs + 1):
    model.train()

    num_batches = NUM_SAMPLES // BATCH_SIZE
    total_loss = 0
    all_preds = []

    for i in range(num_batches):
        times = torch.linspace(0, 1, len(MATURITIES)).to(device)

        optimizer.zero_grad()
        pred = model(BATCH_SIZE, times).squeeze(-1)
        loss = my_loss(pred, call_data)
        loss.backward()
        optimizer.step()

        total_loss += loss
        all_preds.append(pred.detach().cpu())

    mse_loss.append(total_loss.detach().cpu())

    # decay LR once per epoch
    scheduler.step()

    # monitor current LR
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}, Loss: {total_loss:.6f}, LR: {current_lr:.6e}")

    if epoch % 10 == 0:
        plt.figure(figsize=(8, 4))
        for i in range(100):
            plt.plot(times.cpu(), pred.detach().cpu()[i, :], color='r', alpha = 0.1)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Model Predictions at epoch ' + str(epoch))
        plt.savefig(DIR + "/model_pred_" + str(epoch) + ".png")
        plt.clf()


import pickle
with open(DIR + '/mse.pickle', 'wb') as handle:
    pickle.dump(mse_loss, handle)

plt.plot(mse_loss)
plt.savefig(DIR + "/loss.png")
plt.clf()

from scipy.stats import entropy


def calculate_kl_divergence(true_values, pred_values, num_bins=50):
    # Compute histogram for true and predicted values
    hist_true, bin_edges = np.histogram(true_values, bins=num_bins, density=True)
    hist_pred, _ = np.histogram(pred_values, bins=bin_edges, density=True)

    # Avoid division by zero and log(0) issues
    hist_true = np.where(hist_true == 0, 1e-10, hist_true)
    hist_pred = np.where(hist_pred == 0, 1e-10, hist_pred)

    # Calculate KL divergence
    kl_div = entropy(hist_true, hist_pred)
    return kl_div


def compare_distributions(true_data, pred_data, points, num_bins=50):
    time_points = [int(p * true_data.shape[1]) for p in points]

    fig, axes = plt.subplots(1, len(points), figsize=(20, 5), sharey=True)

    for ax, point, time_point in zip(axes, points, time_points):
        true_values = true_data[:, time_point]
        pred_values = pred_data[:, time_point]

        kl_div = calculate_kl_divergence(true_values, pred_values, num_bins)

        bins = np.histogram(np.hstack((true_values, pred_values)), bins=num_bins)[1]
        ax.hist([round(v, 5) for v in true_values], bins=bins, alpha=0.5, label='True', color='r')
        ax.hist([round(v, 5) for v in pred_values], bins=bins, alpha=0.5, label='Pred', color='b')
        ax.set_title(f'{int(point * 100)}% Point\nKL: {kl_div:.4f}')
        ax.set_xlabel('Value')
        # ax.set_xlim(-0.75, 1.25)
        # ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
        if ax == axes[0]:
            ax.set_ylabel('Frequency')
        ax.legend()

    plt.suptitle('Distribution Comparison at Specific Points')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(DIR + "/distr.png")
    plt.savefig(DIR + "/distr.pdf")
    plt.savefig(DIR + "/distr.eps")
    plt.clf()

W = np.cumsum(
    np.random.normal(0., np.sqrt(DT), size=(NUM_SAMPLES, N)),
    axis = 1
)
t = np.linspace(0, T, N)
t = np.tile(t, (NUM_SAMPLES, 1))
all_trues = S0 * np.exp(
    (R - 0.5 * SIGMA**2) * t + SIGMA * W
)

all_preds = torch.cat(all_preds, axis = 0)
all_preds = np.array(all_preds)

points_to_compare = [0.2, 0.4, 0.6, 0.8]
compare_distributions(all_trues, all_preds, points_to_compare)

with open(DIR + '/distr.pickle', 'wb') as handle:
    pickle.dump((all_trues, all_preds), handle)
