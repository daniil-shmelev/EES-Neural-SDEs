"""Fixed-grid CFEES25 integration with replayable Brownian increments.

This module intentionally does not use Diffrax Brownian paths. The fixed-grid
solver generates one Gaussian Lie-algebra increment per step from
``jax.random.fold_in(base_key, step_index)``. The custom VJP replays those same
increments while walking the CFEES recurrence backward, so it avoids storing the
per-step solver tape.
"""

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array


EES25_A = (-7.0 / 15.0, -35.0 / 32.0)
EES25_B = (1.0 / 3.0, 15.0 / 16.0, 2.0 / 5.0)
EES25_C = (0.0, 1.0 / 3.0, 5.0 / 6.0)
DEFAULT_CHART_DEGREE = 6
SphereAction = Literal["cayley", "taylor"]


def step_noise(key: Array, step_index: Array | int, shape: tuple[int, ...], dtype) -> Array:
    """Generate the replayable standard normal for one fixed solver step."""

    step_key = jax.random.fold_in(key, step_index)
    return jax.random.normal(step_key, shape=shape, dtype=dtype)


def _normalize(x: Array, eps: float = 1e-7) -> Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def _coords_to_alg(coeffs: Array, basis: Array) -> Array:
    return jnp.einsum("...d,dij->...ij", coeffs, basis.astype(coeffs.dtype))


def _apply_taylor_increment(
    z: Array,
    coeffs: Array,
    basis: Array,
    *,
    degree: int,
) -> Array:
    omega = _coords_to_alg(coeffs, basis)
    term = z
    out = z
    for k in range(1, int(degree) + 1):
        term = jnp.einsum("...ij,...j->...i", omega, term) / jnp.asarray(k, z.dtype)
        out = out + term
    return _normalize(out)


def _apply_cayley_increment(z: Array, coeffs: Array, basis: Array) -> Array:
    omega = _coords_to_alg(coeffs, basis)
    ident = jnp.eye(basis.shape[-1], dtype=z.dtype)
    rhs = jnp.einsum("...ij,...j->...i", ident + 0.5 * omega, z)
    return jnp.linalg.solve(ident - 0.5 * omega, rhs[..., None])[..., 0]


def _apply_increment(
    z: Array,
    coeffs: Array,
    basis: Array,
    *,
    action: SphereAction,
    degree: int,
) -> Array:
    if action == "cayley":
        return _apply_cayley_increment(z, coeffs, basis)
    if action == "taylor":
        return _apply_taylor_increment(z, coeffs, basis, degree=degree)
    raise ValueError(f"unknown sphere action {action!r}")


def _time_fn_batched(time_fn: Any, h: Array, times: Array) -> Array:
    if h.ndim == 1:
        return time_fn(h, times)
    flat_h = h.reshape((-1, h.shape[-1]))
    drift = jax.vmap(lambda one_h: time_fn(one_h, times))(flat_h)
    return drift.reshape(h.shape[:-1] + drift.shape[1:])


def cfees25_step(
    time_fn: Any,
    h: Array,
    z: Array,
    sigma: Array,
    t0: Array,
    t1: Array,
    basis: Array,
    noise: Array,
    *,
    chart_degree: int = DEFAULT_CHART_DEGREE,
    action: SphereAction = "cayley",
) -> Array:
    """One low-storage CFEES25 step on the sphere."""

    dt = t1 - t0
    abs_sqrt_dt = jnp.sqrt(jnp.abs(dt))
    sign_dt = jnp.sign(dt)
    diffusion = sigma.astype(z.dtype) * sign_dt.astype(z.dtype) * abs_sqrt_dt.astype(z.dtype)
    diffusion = diffusion * noise

    stage_times = jnp.stack(
        [t0 + jnp.asarray(c, dtype=dt.dtype) * dt for c in EES25_C]
    )
    drift = _time_fn_batched(time_fn, h, stage_times)
    drift = jnp.broadcast_to(drift, z.shape[:-1] + drift.shape[-2:])
    raws = drift * dt.astype(drift.dtype) + diffusion[..., None, :]

    tmp = raws[..., 0, :]
    z = _apply_increment(
        z,
        jnp.asarray(EES25_B[0], dtype=z.dtype) * tmp,
        basis,
        action=action,
        degree=chart_degree,
    )
    for stage_index in range(1, len(EES25_B)):
        tmp = (
            jnp.asarray(EES25_A[stage_index - 1], dtype=z.dtype) * tmp
            + raws[..., stage_index, :]
        )
        z = _apply_increment(
            z,
            jnp.asarray(EES25_B[stage_index], dtype=z.dtype) * tmp,
            basis,
            action=action,
            degree=chart_degree,
        )
    return z


def cfees25_integrate_scan(
    time_fn: Any,
    h: Array,
    z0: Array,
    sigma: Array,
    times: Array,
    basis: Array,
    key: Array,
    *,
    chart_degree: int = DEFAULT_CHART_DEGREE,
    action: SphereAction = "cayley",
) -> Array:
    """Forward fixed-grid CFEES25 scan with ordinary autodiff."""

    z0 = _normalize(z0)
    noise_shape = z0.shape[:-1] + (basis.shape[0],)

    def step(z, inputs):
        step_index, t0, t1 = inputs
        noise = step_noise(key, step_index, noise_shape, z.dtype)
        z_next = cfees25_step(
            time_fn,
            h,
            z,
            sigma,
            t0,
            t1,
            basis,
            noise,
            chart_degree=chart_degree,
            action=action,
        )
        return z_next, z_next

    step_indices = jnp.arange(times.shape[0] - 1)
    _, zs = jax.lax.scan(step, z0, (step_indices, times[:-1], times[1:]))
    zs = jnp.concatenate([z0[None], zs], axis=0)
    return jnp.moveaxis(zs, 0, 2)


@eqx.filter_custom_vjp
def cfees25_integrate_reversible(
    vjp_arg,
    times: Array,
    basis: Array,
    key: Array,
    *,
    chart_degree: int = DEFAULT_CHART_DEGREE,
    action: SphereAction = "cayley",
) -> Array:
    """Fixed-grid CFEES25 scan with a replaying custom VJP.

    ``vjp_arg`` is ``(time_fn, h, z0, sigma)``. Gradients are returned for all
    differentiable leaves in that tuple; ``times``, ``basis``, ``key`` and the
    chart degree are treated as nondifferentiated solver metadata.
    """

    time_fn, h, z0, sigma = vjp_arg
    return cfees25_integrate_scan(
        time_fn,
        h,
        z0,
        sigma,
        times,
        basis,
        key,
        chart_degree=chart_degree,
        action=action,
    )


@cfees25_integrate_reversible.def_fwd
def _cfees25_integrate_reversible_fwd(
    perturbed,
    vjp_arg,
    times: Array,
    basis: Array,
    key: Array,
    *,
    chart_degree: int = DEFAULT_CHART_DEGREE,
    action: SphereAction = "cayley",
):
    del perturbed
    out = cfees25_integrate_reversible(
        vjp_arg,
        times,
        basis,
        key,
        chart_degree=chart_degree,
        action=action,
    )
    terminal = out[:, :, -1, :]
    return out, terminal


def _tree_add(a, b):
    return jtu.tree_map(
        lambda x, y: y if x is None else x if y is None else x + y,
        a,
        b,
        is_leaf=lambda x: x is None,
    )


def _tree_zeros_like(tree):
    return jtu.tree_map(lambda x: jnp.zeros_like(x), tree)


@cfees25_integrate_reversible.def_bwd
def _cfees25_integrate_reversible_bwd(
    residuals,
    grad_path: Array,
    perturbed,
    vjp_arg,
    times: Array,
    basis: Array,
    key: Array,
    *,
    chart_degree: int = DEFAULT_CHART_DEGREE,
    action: SphereAction = "cayley",
):
    del perturbed
    time_fn, h, z0, sigma = vjp_arg
    noise_shape = z0.shape[:-1] + (basis.shape[0],)

    grad_time_fn = _tree_zeros_like(time_fn)
    grad_h = jnp.zeros_like(h)
    grad_sigma = jnp.zeros_like(sigma)
    grad_z_next = grad_path[:, :, -1, :]

    def step(carry, step_index):
        z_next, grad_z_next, grad_time_fn, grad_h, grad_sigma = carry
        noise = step_noise(key, step_index, noise_shape, z_next.dtype)
        t0 = times[step_index]
        t1 = times[step_index + 1]
        z_prev = cfees25_step(
            time_fn,
            h,
            z_next,
            sigma,
            t1,
            t0,
            basis,
            noise,
            chart_degree=chart_degree,
            action=action,
        )
        z_prev = jax.lax.stop_gradient(z_prev)

        def local_step(local_time_fn, local_h, local_z, local_sigma):
            return cfees25_step(
                local_time_fn,
                local_h,
                local_z,
                local_sigma,
                t0,
                t1,
                basis,
                noise,
                chart_degree=chart_degree,
                action=action,
            )

        _, pullback = jax.vjp(local_step, time_fn, h, z_prev, sigma)
        g_time_fn, g_h, grad_z_prev, g_sigma = pullback(grad_z_next)
        grad_time_fn = _tree_add(grad_time_fn, g_time_fn)
        grad_h = grad_h + g_h
        grad_sigma = grad_sigma + g_sigma
        grad_z_next = grad_z_prev + grad_path[:, :, step_index, :]
        return (z_prev, grad_z_next, grad_time_fn, grad_h, grad_sigma), None

    step_indices = jnp.arange(times.shape[0] - 1)
    carry = (residuals, grad_z_next, grad_time_fn, grad_h, grad_sigma)
    final_carry, _ = jax.lax.scan(
        step,
        carry,
        step_indices,
        reverse=True,
    )
    _, grad_z_next, grad_time_fn, grad_h, grad_sigma = final_carry

    _, normalise_pullback = jax.vjp(_normalize, z0)
    (grad_z0,) = normalise_pullback(grad_z_next)
    return grad_time_fn, grad_h, grad_z0, grad_sigma
