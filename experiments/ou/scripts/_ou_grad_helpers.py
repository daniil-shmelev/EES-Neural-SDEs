"""Adjoint-parameterised re-implementation of the OU Neural Langevin SDE
model classes that live in ``experiments/ou/OU.py``.

The production ``OU.py`` hard-codes ``torchsde.sdeint(...)`` in
``NDE_model.forward`` (full-autograd discretise-then-optimise). For the
gradient-error comparison we need to call ``torchsde.sdeint_adjoint(...)``
on the same forward, so we mirror the model classes here with an
``adjoint`` flag and an ``integrate`` helper that picks the corresponding
torchsde entry point.

Default behaviour (``adjoint='autograd'``) is bit-equivalent to ``OU.py``'s
forward, so the grad-error script can train a reference model in this file
and the trained weights are interchangeable with one trained via ``OU.py``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torchcde
import torchsde
from torch import nn

ADJOINT_METHODS = {
    "ees25": "adjoint_ees25",
    "ees27": "adjoint_ees27",
    "reversible_heun": "adjoint_reversible_heun",
}


class LipSwish(nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim, num_layers, tanh=False, activation="lipswish"):
        super().__init__()
        activation_fn = LipSwish() if activation == "lipswish" else nn.ReLU()
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
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers, activation="lipswish"):
        super().__init__()
        self.sde_type = "stratonovich"
        self.noise_type = "diagonal"
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
    """OU Neural-LSDE wrapper with an adjoint switch.

    For ``adjoint='autograd'`` (default) calls ``torchsde.sdeint``, which is
    what ``OU.py`` does. For ``adjoint='reversible'`` calls
    ``torchsde.sdeint_adjoint`` with the matching ``adjoint_<method>``.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        method,
        activation="lipswish",
        vector_field=None,
        dt: float = 0.05,
        adjoint: str = "autograd",
    ):
        super().__init__()
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers, activation=activation)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.method = method
        self.dt = dt
        if adjoint not in ("autograd", "reversible"):
            raise ValueError(f"adjoint must be 'autograd' or 'reversible', got {adjoint!r}")
        self.adjoint = adjoint

    def integrate(self, y0, ts, bm: Optional["torchsde.BrownianInterval"] = None):
        if self.adjoint == "autograd":
            return torchsde.sdeint(
                sde=self.func, y0=y0, ts=ts, dt=self.dt,
                method=self.method, bm=bm,
            )
        return torchsde.sdeint_adjoint(
            sde=self.func, y0=y0, ts=ts, dt=self.dt,
            method=self.method, adjoint_method=ADJOINT_METHODS[self.method],
            adjoint_params=tuple(self.func.parameters()),
            bm=bm,
        )

    def forward(self, coeffs, times, bm=None):
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)
        y0 = self.initial(y0)[:, 0, :]
        z = self.integrate(y0, times, bm=bm)
        z = z.permute(1, 0, 2)
        return self.decoder(z)


def strip_spline_state(state_dict: dict) -> dict:
    """Remove ``torchcde.CubicSpline`` buffers from a saved state dict."""
    return {k: v for k, v in state_dict.items() if not k.startswith("func.X")}
