"""
Runge-Kutta Schemes
"""

# Adapted from https://github.com/daniil-shmelev/kauri/blob/main/kauri/rk.py

import copy
from typing import Union, Callable, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt

class StochRK:
    def __init__(self, a, b, name = None):
        if not isinstance(a, (list, np.ndarray)):
            raise TypeError("a must be a list or array, not " + str(type(a)))
        if not isinstance(b, (list, np.ndarray)):
            raise TypeError("b must be a list or array, not " + str(type(a)))

        self.name = name
        self.s = len(b)
        if len(a) != self.s or len(a[0]) != self.s:
            raise ValueError("Parameter 'a' must be a square s x s matrix and b a vector of length s")

        self.a = a
        self.b = b
        self.c = [sum(a[i][j] for j in range(self.s)) for i in range(self.s)]

        self.explicit = self._check_explicit()

        if not self.explicit:
            raise NotImplementedError("StochRK only supports explicit schemes")

        self.deriv_dict = {}  # {repr(None) : 1, repr([]) : 1}
        for i in range(self.s):
            self.deriv_dict[(i, repr(None))] = 1
            self.deriv_dict[(i, repr([]))] = 1

        self.np_a = np.array(a)
        self.np_b = np.array(b)

    def __repr__(self):
        out = "["
        for i in range(self.s - 1):
            out += repr(self.a[i]) + ",\n"
        out += repr(self.a[-1]) + "]\n"
        out += repr(self.b)
        return out

    def _check_explicit(self):
        for i in range(self.s):
            for j in range(i, self.s):
                if self.a[i][j]:
                    return False
        return True

    def _explicit_step(self, y0, f, dX, d):
        Y_h = [None] * self.s

        for i in range(self.s):
            Y_h[i] = y0 + sum(sum(self.a[i][j] * f(Y_h[j])[k] * dX[k] for j in range(i)) for k in range(d))

        y_next = y0 + sum(sum(self.b[i] * f(Y_h[i])[k] * dX[k] for i in range(self.s)) for k in range(d))
        return y_next

    def step(self,
             y0 : Union[list, np.ndarray],
             f : Callable[[float], Union[list, np.ndarray]],
             dX : Union[list, np.ndarray]
             ) -> Union[list, np.ndarray]:

        if not isinstance(y0, (list, np.ndarray)):
            raise TypeError("y0 must be a list or array, not " + str(type(y0)))
        if not callable(f):
            raise TypeError("f must be callable")
        if not isinstance(dX, (list, np.ndarray)):
            raise TypeError("X must be a list or array, not " + str(type(dX)))

        def f_(t_, y_):
            return np.array(f(t_,y_))
        y0_ = np.array(y0).copy()

        return self._explicit_step(y0_, f_, dX, dX.shape[1])

    def run(self,
            y0 : Union[list, np.ndarray],
            f : Callable[[float], Union[list, np.ndarray]],
            X: Union[list, np.ndarray],
            plot : bool = False,
            plot_dims : Union[list, np.ndarray] = None,
            plot_kwargs : dict = None
            ) -> Tuple[list, list]:

        if not isinstance(y0, (list, np.ndarray)):
            raise TypeError("y0 must be a list or array, not " + str(type(y0)))
        if not callable(f):
            raise TypeError("f must be callable")
        if not (isinstance(plot, bool) or plot is None):
            raise TypeError("plot must be a bool, not " + str(type(plot)))
        if not (isinstance(plot_dims, (list, np.ndarray)) or plot_dims is None):
            raise TypeError("plot_dims must be a list or array, not " + str(type(plot_dims)))
        if not (isinstance(plot_kwargs, dict) or plot_kwargs is None):
            raise TypeError("plot_kwargs must be a dict, not " + str(type(plot_kwargs)))

        if plot_kwargs is None:
            plot_kwargs = {}
        if plot_dims is None:
            plot_dims = list(range(len(y0)))

        def f_(y_):
            return np.array(f(y_))
        y0_ = np.array(y0).copy()

        X_ = np.array(X)
        dX = X_[1:, :] - X_[:-1, :]
        n = dX.shape[0]
        d = dX.shape[1]

        y_vals = [y0_]
        y = y0_.copy()

        for incr in dX:
            y = self._explicit_step(y, f_, incr, d)
            y_vals.append(copy.deepcopy(y))

        if plot:
            plt.plot(range(n+1), np.array(y_vals)[:, plot_dims], **plot_kwargs)

        return y_vals