"""
This file contains code fitting a Neural Langevin SDE (LSDE) to high volatility
Ornstein-Uhlenbeck dynamics. The code is largely based on the repository:

https://github.com/yongkyung-oh/Stable-Neural-SDEs/blob/main/tutorial/simple%20OU%20process%20-%20Neural%20LSDE.ipynb

supporting the paper "Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data".

Once this file is run for both reversible_heun and ees25, the training loss can be plotted using plot_OU.py
"""

import os
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torchcde
import torchsde
from torch.utils.data import Dataset, DataLoader
from scipy.stats import entropy

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

def ou_process(T, N, theta, mu, sigma, X0):
    dt = T / N
    t = np.linspace(0, T, N)
    X = np.zeros(N)
    X[0] = X0

    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW

    return t, X

def generate_data(config):
    data_list = []
    for _ in range(config['num_samples']):
        t, X = ou_process(config['T'], config['N'], config['theta'], config['mu'], config['sigma'], config['X0'])
        data_list.append([t, X])

    total_data = torch.Tensor(np.array(data_list))  # [Batch size, Dimension, Length]
    total_data = total_data.permute(0, 2, 1)  # [Batch size, Length, Dimension]

    max_len = total_data.shape[1]
    times = torch.linspace(0, 1, max_len)
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(total_data, times)

    return total_data, coeffs, times

class OU_Dataset(Dataset):
    def __init__(self, data, coeffs):
        self.data = data
        self.coeffs = coeffs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx, ...],
            self.coeffs[idx, ...],
        )

def split_data(data, coeffs, train_ratio=0.8):
    total_size = len(data)
    train_size = int(total_size * train_ratio)

    train_idx = np.random.choice(range(total_size), train_size, replace=False)
    test_idx = np.array([i for i in range(total_size) if i not in train_idx])

    train_data = data[train_idx, ...]
    test_data = data[test_idx, ...]
    train_coeffs = coeffs[train_idx, ...]
    test_coeffs = coeffs[test_idx, ...]

    return train_data, train_coeffs, test_data, test_coeffs

def create_data_loaders(train_data, train_coeffs, test_data, test_coeffs, batch_size=16):
    train_dataset = OU_Dataset(train_data, train_coeffs)
    test_dataset = OU_Dataset(test_data, test_coeffs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class LipSwish(nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim, num_layers, tanh=False, activation='lipswish'):
        super().__init__()

        if activation == 'lipswish':
            activation_fn = LipSwish()
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

class NeuralLSDEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers, activation='lipswish'):
        super().__init__()
        self.sde_type = "stratonovich"
        self.noise_type = "diagonal"  # or "scalar"

        self.linear_X = nn.Linear(input_dim, hidden_dim)
        self.emb = nn.Linear(hidden_dim * 2, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.noise_in = nn.Linear(1, hidden_dim)
        self.g_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)

    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)

    def f(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.linear_X(Xt)
        z = self.emb(torch.cat([y, Xt], dim=-1))
        z = self.f_net(z)
        return self.linear_out(z)

    def g(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)

        tt = self.noise_in(t)
        return self.g_net(tt)

class NDE_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, method, activation='lipswish', vector_field=None,
                 dt=None):
        super().__init__()
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers, activation=activation)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.method = method
        self.dt = dt

    def forward(self, coeffs, times, adjoint=False, bm_entropy=None):
        # control module
        self.func.set_X(coeffs, times)

        y0 = self.func.X.evaluate(times)
        y0 = self.initial(y0)[:, 0, :]

        dt = self.dt
        if dt is None:
            dt = float((times[1] - times[0]).detach().cpu())

        bm = None
        if bm_entropy is not None:
            bm = torchsde.BrownianInterval(
                t0=float(times[0].detach().cpu()),
                t1=float(times[-1].detach().cpu()),
                size=(y0.size(0), y0.size(1)),
                dtype=y0.dtype,
                device=y0.device,
                entropy=int(bm_entropy),
                dt=dt,
            )

        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        z = sdeint(sde=self.func,
                   y0=y0,
                   ts=times,
                   bm=bm,
                   dt=dt,
                   method=self.method)
        z = z.permute(1, 0, 2)
        return self.decoder(z)

def flattened_grads(model):
    grads = []
    for param in model.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param).reshape(-1))
        else:
            grads.append(param.grad.detach().reshape(-1))
    return torch.cat(grads)

def gradient_for_loss(model, criterion, coeffs, times, true, adjoint, bm_entropy):
    model.zero_grad(set_to_none=True)
    pred = model(coeffs, times, adjoint=adjoint, bm_entropy=bm_entropy).squeeze(-1)
    loss = criterion(pred, true)
    loss.backward()
    grad = flattened_grads(model)
    return loss.detach(), grad

def relative_gradient_error(model, criterion, coeffs, times, true, bm_entropy):
    _, direct_grad = gradient_for_loss(
        model, criterion, coeffs, times, true, adjoint=False, bm_entropy=bm_entropy
    )
    _, adjoint_grad = gradient_for_loss(
        model, criterion, coeffs, times, true, adjoint=True, bm_entropy=bm_entropy
    )
    direct_grad = direct_grad.double()
    adjoint_grad = adjoint_grad.double()
    return (adjoint_grad - direct_grad).norm() / direct_grad.norm().clamp_min(1e-12)

def save_training_histories(dir_, mse_loss, grad_error_epochs, grad_error):
    with open(dir_ + '/mse.pickle', 'wb') as handle:
        pickle.dump(mse_loss, handle)
    with open(dir_ + '/grad_error.pickle', 'wb') as handle:
        pickle.dump({'epochs': grad_error_epochs, 'values': grad_error}, handle)

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

def compare_distributions(true_data, pred_data, points, dir_, num_bins=50):
    time_points = [int(p * true_data.shape[1]) for p in points]

    _, axes = plt.subplots(1, len(points), figsize=(20, 5), sharey=True)

    for ax, point, time_point in zip(axes, points, time_points):
        true_values = true_data[:, time_point].numpy()
        pred_values = pred_data[:, time_point].numpy()

        kl_div = calculate_kl_divergence(true_values, pred_values, num_bins)

        ax.hist([round(v, 5) for v in true_values], bins=num_bins, alpha=0.5, label='True', color='r')
        ax.hist([round(v, 5) for v in pred_values], bins=num_bins, alpha=0.5, label='Pred', color='b')
        ax.set_title(f'{int(point * 100)}% Point\nKL: {kl_div:.4f}')
        ax.set_xlabel('Value')
        if ax == axes[0]:
            ax.set_ylabel('Frequency')
        ax.legend()

    plt.suptitle('Distribution Comparison at Specific Points')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(dir_ + "/distr.png")
    plt.savefig(dir_ + "/distr.pdf")
    plt.savefig(dir_ + "/distr.eps")
    plt.clf()

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Parameters
    config = {
        'method': "reversible_heun", # "reversible_heun" or "ees25"
        'num_samples': 50000,
        'T': 10.0,
        'N': 20,
        'theta': 0.2,
        'mu': 0.1,
        'sigma': 2.0,
        'X0': 1.0,
        'train_ratio': 0.8,
        'batch_size': 50000,
        'seed': 42,
        'num_epochs': 250,
        'input_dim': 2,
        'output_dim': 1,
        'hidden_dim': 32,
        'num_layers': 1,
        'lr': 1e-3,
        'solver_dt': None,
        'use_adjoint_training': True,
        'measure_grad_error': True,
        'grad_error_interval': 1,
        'grad_error_batch_size': 512,
    }

    DIR = "plots/" + config['method']

    if not os.path.exists(DIR):
        os.makedirs(DIR)

    seed_everything(config['seed'])

    # Generate data
    total_data, coeffs, times = generate_data(config)

    # Split data
    train_data, train_coeffs, test_data, test_coeffs = split_data(total_data, coeffs, config['train_ratio'])

    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_data, train_coeffs, test_data, test_coeffs, config['batch_size'])

    # Plot the first sample for verification
    plt.plot(times.numpy(), total_data[0, :, 1].numpy())
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('OU Process Sample Path')
    plt.grid(True)
    plt.savefig(DIR + "/sample_path.png")
    plt.savefig(DIR + "/sample_path.pdf")
    plt.savefig(DIR + "/sample_path.eps")
    plt.clf()

    # Plot the whole samples for verification
    for data in total_data:
        plt.plot(times.numpy(), data[:,1].numpy(), alpha=0.05)
    plt.plot(times.numpy(), total_data[0, :, 1].numpy(), color='k')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('OU Process Total Sample Path')
    plt.grid(True)
    plt.savefig(DIR + "/total_sample_path.png")
    plt.savefig(DIR + "/total_sample_path.pdf")
    plt.savefig(DIR + "/total_sample_path.eps")
    plt.clf()


    model = NDE_model(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'], output_dim=config['output_dim'],
                      num_layers=config['num_layers'], method = config['method'], vector_field=NeuralLSDEFunc,
                      dt=config['solver_dt']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.MSELoss()

    model.eval()
    total_loss = 0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in test_loader:
            coeffs = batch[1].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

            true = batch[0][:,:,1].to(device)
            pred = model(coeffs, times).squeeze(-1)
            loss = criterion(pred, true)
            total_loss += loss.item()

            all_preds.append(pred.cpu())
            all_trues.append(true.cpu())

    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}', flush=True)

    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    num_samples = 5

    plt.figure(figsize=(8, 4))
    for i in range(num_samples):
        plt.plot(all_trues[i].numpy(), color='r')
        plt.plot(all_preds[i].numpy(), color='b')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.ylim(-0.75,1.25)
    plt.title('Model Predictions vs True Values')
    plt.savefig(DIR + "/model_pred.png")
    plt.savefig(DIR + "/model_pred.pdf")
    plt.savefig(DIR + "/model_pred.eps")
    plt.clf()

    mse_loss = []
    grad_error_epochs = []
    grad_error = []

    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            coeffs = batch[1].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

            true = batch[0][:, :, 1].to(device)

            if (
                config['measure_grad_error']
                and batch_idx == 0
                and epoch % config['grad_error_interval'] == 0
            ):
                grad_batch_size = min(config['grad_error_batch_size'], coeffs.shape[0])
                grad_bm_entropy = config['seed'] + 1000003 * epoch + batch_idx
                rel_grad_error = relative_gradient_error(
                    model=model,
                    criterion=criterion,
                    coeffs=coeffs[:grad_batch_size],
                    times=times,
                    true=true[:grad_batch_size],
                    bm_entropy=grad_bm_entropy,
                )
                grad_error_epochs.append(epoch)
                grad_error.append(rel_grad_error.item())
                print(f'Epoch {epoch}, Grad Error: {rel_grad_error.item()}', flush=True)

            optimizer.zero_grad()
            pred = model(coeffs, times, adjoint=config['use_adjoint_training']).squeeze(-1)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        mse_loss.append(total_loss)

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}, Loss: {avg_loss}', flush=True)
            save_training_histories(DIR, mse_loss, grad_error_epochs, grad_error)

            model.eval()
            total_loss = 0
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for batch in test_loader:
                    coeffs = batch[1].to(device)
                    times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

                    true = batch[0][:, :, 1].to(device)
                    pred = model(coeffs, times).squeeze(-1)
                    loss = criterion(pred, true)
                    total_loss += loss.item()

                    all_preds.append(pred.cpu())
                    all_trues.append(true.cpu())

            avg_loss = total_loss / len(test_loader)
            print(f'Test Loss: {avg_loss}', flush=True)

            all_preds = torch.cat(all_preds, dim=0)
            all_trues = torch.cat(all_trues, dim=0)

            plt.figure(figsize=(8, 4))
            for i in range(num_samples):
                plt.plot(all_trues[i].numpy(), color='r')
                plt.plot(all_preds[i].numpy(), color='b')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.ylim(-0.75, 1.25)
            plt.title('Model Predictions vs True Values')
            plt.savefig(DIR + "/model_pred" + str(epoch) + ".png")
            plt.savefig(DIR + "/model_pred" + str(epoch) + ".pdf")
            plt.savefig(DIR + "/model_pred" + str(epoch) + ".eps")
            plt.clf()



    save_training_histories(DIR, mse_loss, grad_error_epochs, grad_error)

    plt.plot(mse_loss)
    plt.savefig(DIR + "/loss.png")
    plt.clf()

    if grad_error:
        plt.plot(grad_error_epochs, grad_error)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Relative Gradient Error")
        plt.savefig(DIR + "/grad_error.png")
        plt.clf()

    points_to_compare = [0.2, 0.4, 0.6, 0.8]
    compare_distributions(all_trues, all_preds, points_to_compare, DIR)
