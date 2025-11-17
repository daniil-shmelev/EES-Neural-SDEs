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
import timeit
from tqdm import tqdm
from scipy.linalg import qr
torch.set_default_dtype(torch.float64)

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

def make_lambdas(dim_, A_range):
    return np.linspace(A_range[0], A_range[1], dim_)

def get_A(dim_, A_range, seed=42):
    rng = np.random.default_rng(seed)
    D = np.diag(make_lambdas(dim_, A_range))
    # random orthogonal matrix via QR
    X = rng.normal(size=(dim_, dim_))
    Q, _ = qr(X)  # Q is orthogonal
    A = Q @ D @ Q.T
    return A

def gbm_process(T, N, A, K, sigma, X0_):
    d = config['input_dim'] - 1
    X0 = np.ones(d) * X0_
    dt = T / (N * K)
    t = np.linspace(0, T, N)
    X = np.zeros((N, d))
    X[0] = X0

    X_curr = X0

    for i in range(1, N):
        for _ in range(K):
            dW = np.random.normal(0, np.sqrt(dt), d)
            drift = (A @ X_curr) * dt
            diffusion = X_curr * sigma * dW
            X_curr += drift + diffusion

        X[i] = X_curr

    return torch.tensor(t).unsqueeze(-1), torch.tensor(X)

def generate_data(config):
    A = get_A(config['input_dim'] - 1, config['A_range'])
    data_list = []
    print("Generating data...")
    for _ in tqdm(range(config['num_samples'])):
        t, X = gbm_process(config['T'], config['N'], A, config['K'], config['sigma'], config['X0'])
        data_list.append(torch.concatenate((t, X), dim = -1).unsqueeze(0))

    total_data = torch.concatenate(data_list, dim=0)  # [Batch size, Dimension, Length]
    #total_data = total_data.permute(0, 2, 1)  # [Batch size, Length, Dimension]

    max_len = total_data.shape[1]
    times = torch.linspace(0, config['T'], max_len)
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(total_data, times)

    return total_data, coeffs, times

class GBM_Dataset(Dataset):
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
    train_dataset = GBM_Dataset(train_data, train_coeffs)
    test_dataset = GBM_Dataset(test_data, test_coeffs)

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

class NeuralSDEFunc(nn.Module):
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
        Xt = self.X.evaluate(t)
        Xt = self.linear_X(Xt)
        z = self.emb(torch.cat([y, Xt], dim=-1))
        z = self.g_net(z)
        return self.linear_out(z)

class NDE_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, method, activation='lipswish', vector_field=None, options=None):
        super().__init__()
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers, activation=activation)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.method = method
        self.options = options

    def forward(self, coeffs, times, dt):
        # control module
        self.func.set_X(coeffs, times)

        y0 = self.func.X.evaluate(times[0])
        y0 = self.initial(y0)

        z = torchsde.sdeint(sde=self.func,
                            y0=y0,
                            ts=times,
                            dt=dt,
                            method=self.method,
                            options=self.options)
        z = z.permute(1, 0, 2)
        return self.decoder(z)

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

def main(METHOD, DT, config, total_data, coeffs, times):
    config['method'] = METHOD
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    options = {'lam' : 0.99}

    DIR = "plots/gbm_" + config['method']

    if not os.path.exists(DIR):
        os.makedirs(DIR)

    seed_everything(config['seed'])

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
                      num_layers=config['num_layers'], method = config['method'], vector_field=NeuralSDEFunc, options=options).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.MSELoss()

    model.eval()
    total_loss = 0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in test_loader:
            coeffs = batch[1].to(device)
            times = torch.linspace(0, config['T'], batch[0].shape[1]).to(device)

            true = batch[0][:,:,1:].to(device)
            pred = model(coeffs, times, DT)
            loss = criterion(pred, true)
            total_loss += loss.item()

            all_preds.append(pred.cpu())
            all_trues.append(true.cpu())

    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}')

    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    num_samples = 5

    plt.figure(figsize=(8, 4))
    for i in range(num_samples):
        plt.plot(all_trues[i, :, 0].numpy(), color='r')
        plt.plot(all_preds[i, :, 0].numpy(), color='b')
    plt.xlabel('Time')
    plt.ylabel('Value')
    #plt.ylim(-0.75,1.25)
    plt.title('Model Predictions vs True Values')
    plt.savefig(DIR + "/model_pred.png")
    plt.savefig(DIR + "/model_pred.pdf")
    plt.savefig(DIR + "/model_pred.eps")
    plt.clf()

    mse_loss = []

    start = timeit.default_timer()

    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            coeffs = batch[1].to(device)
            times = torch.linspace(0, config['T'], batch[0].shape[1]).to(device)

            optimizer.zero_grad()
            true = batch[0][:, :, 1:].to(device)
            pred = model(coeffs, times, DT)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        mse_loss.append(total_loss)

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}, Loss: {avg_loss}')

            ##
            model.eval()
            total_loss = 0
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for batch in test_loader:
                    coeffs = batch[1].to(device)
                    times = torch.linspace(0, config['T'], batch[0].shape[1]).to(device)

                    true = batch[0][:, :, 1:].to(device)
                    pred = model(coeffs, times, DT)
                    loss = criterion(pred, true)
                    total_loss += loss.item()

                    all_preds.append(pred.cpu())
                    all_trues.append(true.cpu())

            avg_loss = total_loss / len(test_loader)
            print(f'Test Loss: {avg_loss}')

            all_preds = torch.cat(all_preds, dim=0)
            all_trues = torch.cat(all_trues, dim=0)

            ##
            plt.figure(figsize=(8, 4))
            for i in range(num_samples):
                plt.plot(all_trues[i, :, 0].numpy(), color='r')
                plt.plot(all_preds[i, :, 0].numpy(), color='b')
            plt.xlabel('Time')
            plt.ylabel('Value')
            #plt.ylim(-0.75, 1.25)
            plt.title('Model Predictions vs True Values')
            plt.savefig(DIR + "/model_pred" + str(epoch) + ".png")
            plt.savefig(DIR + "/model_pred" + str(epoch) + ".pdf")
            plt.savefig(DIR + "/model_pred" + str(epoch) + ".eps")
            plt.clf()

    end = timeit.default_timer()

    with open(DIR + "/time.txt", "w") as f:
        f.write(str(end - start))

    with open(DIR + '/mse.pickle', 'wb') as handle:
        pickle.dump(mse_loss, handle)

    plt.plot(mse_loss)
    plt.savefig(DIR + "/loss.png")
    plt.clf()

    points_to_compare = [0.2, 0.4, 0.6, 0.8]

if __name__ == "__main__":

    dim = 25

    # Parameters
    config = {  # "reversible_heun" or "ees25"
        'num_samples': 1000,
        'T': 1.0,
        'N': 100,
        'K': 20,
        'A_range' : (-100, 0),
        'sigma': 0.1,
        'X0': 1.0,
        'train_ratio': 0.8,
        'batch_size': 1000,
        'seed': 42,
        'num_epochs': 250,
        'input_dim': dim + 1,
        'output_dim': dim,
        'hidden_dim': 32,
        'num_layers': 1,
        'lr': 2e-2
    }

    data = generate_data(config)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    dt = {
        "reversible_heun" : 2. / 120,
        "ees25" : 2. / 40,
        "ees27" : 2. / 30,
        "mcf_euler" : 2. / 60,
        "mcf_midpoint" : 2. / 30
    }[args.method]

    main(args.method, dt, config, *data)

    # main("reversible_heun", 2. / 120, config, *args)
    # main("ees25", 2. / 40, config, *args)
    # main("mcf_midpoint", 2. / 30, config, *args)
    # main("ees27", 2. / 30, config, *args)
    # main("mcf_euler", 2. / 60, config, *args)
