import jax
import jax.numpy as jnp
from diffrax import (
    ControlTerm,
    Dopri5,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)


def drift(t, y, args):
    del t, args
    theta = 0.8
    mu = 0.0
    return theta * (mu - y)


def diffusion(t, y, args):
    del t, y, args
    sigma = 0.3
    return jnp.array([[sigma]])


def main():
    key = jax.random.key(0)

    t0 = 0.0
    t1 = 1.0
    dt = 0.05
    y0 = jnp.array([1.0])
    ts = jnp.arange(t0, t1 + dt, dt)

    brownian = VirtualBrownianTree(
        t0=t0,
        t1=t1,
        tol=dt / 2.0,
        shape=(1,),
        key=key,
    )

    term = MultiTerm(
        ODETerm(drift),
        ControlTerm(diffusion, brownian),
    )

    sol = diffeqsolve(
        term,
        Dopri5(),
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        saveat=SaveAt(ts=ts),
        max_steps=256,
    )

    print("times:")
    print(sol.ts)
    print("solution:")
    print(sol.ys[:, 0])


if __name__ == "__main__":
    main()
