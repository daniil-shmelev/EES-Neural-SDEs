"""Short, matched training check with the production model and tiny batches.

The reference adjoint is used only for CG2 training and the final derivative
check. Every EES update uses ReversibleAdjoint, with no state checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from diffrax import RecursiveCheckpointAdjoint
from georax import CFEES25

from experiments.kuramoto.datasets.kuramoto_dataset import KuramotoDataset, default_npz_path
from experiments.kuramoto.experiment.config import DEFAULT_CONFIG, Solvers, load_config
from experiments.kuramoto.experiment.factories import make_model
from experiments.kuramoto.experiment.losses import make_multi_horizon_energy_score


def _flatten(tree):
    return np.concatenate([
        np.asarray(x, dtype=np.float64).ravel() for x in jax.tree.leaves(tree)
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=['cg2', 'ees'],
                        default=['cg2', 'ees'])
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10, help='Adaptive EES maximum step is T/n_steps; CG2 uses 75 steps.')
    parser.add_argument('--adaptive-atol', type=float, default=.03)
    parser.add_argument('--adaptive-dtmin', type=float, default=1e-4)
    parser.add_argument('--initial-checkpoint', type=Path,
                        help='Start every method at the same saved weights; use a fresh optimizer.')
    args = parser.parse_args()
    if args.steps < 2 or min(args.batch_size, args.samples, args.n_steps) < 1:
        parser.error('Require steps >= 2 and positive batch, sample and solver step counts.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = load_config(args.config)
    path = default_npz_path(Path(base.data_dir), base.N, base.data_seed)
    data = KuramotoDataset(path, 'train')
    data.theta, data.omega = data.theta[:128], data.omega[:128]
    train = data.as_array_dict()
    held_out = KuramotoDataset(path, 'val')
    held_out.theta, held_out.omega = held_out.theta[:8], held_out.omega[:8]
    validation = held_out.as_array_dict()
    loss = make_multi_horizon_energy_score(horizons=base.loss_horizons, n_samples=args.samples)
    results = dict(
        command_options={key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
        train_examples=len(data), validation_examples=len(held_out),
        devices=[str(device) for device in jax.devices()],
        scope='Small-batch diagnostic; losses are not interchangeable with the full four-sample experiment.',
        runs=[],
    )
    for mode in args.modes:
        reversible = mode != 'cg2'
        config = dataclasses.replace(
            base, solver=Solvers.CFEES25 if reversible else Solvers.CG2,
            adjoint='reversible' if reversible else 'checkpoint_full',
            n_steps=args.n_steps if reversible else 75, adaptive=reversible,
            adaptive_atol=args.adaptive_atol, adaptive_dtmin=args.adaptive_dtmin, adaptive_max_steps=512,
            batch_size=args.batch_size, loss_n_samples=args.samples, seed=args.seed,
        )
        model = make_model(config, data.metadata(), jax.random.key(args.seed))
        if reversible:
            object.__setattr__(model, 'solver', CFEES25())
        if args.initial_checkpoint is not None:
            model = eqx.tree_deserialise_leaves(args.initial_checkpoint, model)
        schedule = optax.warmup_cosine_decay_schedule(
            1e-5, config.learning_rate, min(200, args.steps // 2), args.steps,
            end_value=1e-5,
        )
        optimizer = optax.chain(optax.clip_by_global_norm(config.grad_clip_norm), optax.adam(schedule))
        state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(model, state, batch, key):
            value, gradient = eqx.filter_value_and_grad(loss)(
                model, batch, jnp.ones(args.batch_size), key,
            )
            norm = optax.global_norm(gradient)
            updates, state = optimizer.update(gradient, state, model)
            return eqx.apply_updates(model, updates), state, value, norm

        evaluate = eqx.filter_jit(lambda model: loss(
            model, validation, jnp.ones(len(held_out)), jax.random.key(2026),
        ))
        start = time.perf_counter()
        history = []
        key = jax.random.key(args.seed + 500)
        for index in range(args.steps):
            key, batch_key, noise_key = jax.random.split(key, 3)
            indices = jax.random.randint(batch_key, (args.batch_size,), 0, len(data))
            batch = jax.tree.map(lambda value: value[indices], train)
            model, state, value, norm = step(model, state, batch, noise_key)
            if (index + 1) % (25 if args.steps <= 200 else 100) == 0 or index == args.steps - 1:
                val_loss = float(evaluate(model))
                cell = dict(mode=mode, step=index + 1, train_loss=float(value),
                            val_loss=val_loss, grad_norm=float(norm),
                            elapsed_s=time.perf_counter() - start)
                history.append(cell)
                print(json.dumps(cell), flush=True)
                if not np.isfinite([cell['train_loss'], val_loss, cell['grad_norm']]).all():
                    raise FloatingPointError(f'Nonfinite training result: {cell}')
        eqx.tree_serialise_leaves(args.output_dir / f'{mode}.eqx', model)
        derivative = None
        if reversible:
            batch = jax.tree.map(lambda value: value[:2], validation)
            value_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss))
            reference = copy.copy(model)
            object.__setattr__(reference, 'adjoint', RecursiveCheckpointAdjoint())
            value, gradient = value_grad(model, batch, jnp.ones(2), jax.random.key(777))
            reference_value, reference_gradient = value_grad(reference, batch, jnp.ones(2), jax.random.key(777))
            gradient, reference_gradient = _flatten(gradient), _flatten(reference_gradient)
            scale = np.linalg.norm(reference_gradient)
            derivative = dict(
                relative_error=float(np.linalg.norm(gradient - reference_gradient) / scale),
                cosine=float(np.dot(gradient, reference_gradient) / (np.linalg.norm(gradient) * scale)),
                loss_difference=float(value - reference_value),
            )
            print(mode, derivative, flush=True)
        results['runs'].append(dict(mode=mode, config=dataclasses.asdict(config),
                                    history=history, derivative=derivative))
        (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
