"""Train one CFEES25 cell and save test predictions plus context angles.

Regenerates ``results/rna_predictions_demo.npz`` so the figure in
``report/generate_figures.py::fig_predictions`` is self-contained and
aligned with the current dataset cache.

Usage::

    python scripts/train_and_save_predictions_demo.py

Environment-variable overrides for quick hyperparameter sweeps:
    RNA_LR, RNA_EPOCHS, RNA_WARMUP -- override learning_rate, epochs, warmup_steps.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.rna.datasets.dataset import RNATorsionDataset
from experiments.rna.experiment.config import Devices, Experiments, Solvers, make_config
from experiments.rna.experiment.factories import make_loader, make_model
from experiments.rna.experiment.rna_losses import (
    make_wrapped_energy_score_loss,
    make_wrapped_mae_metric,
    make_wrapped_mse_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "results" / "rna_predictions_demo.npz"


def main() -> int:
    config = make_config(
        experiment=Experiments.RNA,
        epochs=int(os.environ.get("RNA_EPOCHS", 20)),
        batch_size=int(os.environ.get("RNA_BATCH_SIZE", 64)),
        learning_rate=float(os.environ.get("RNA_LR", 3e-4)),
        seed=int(os.environ.get("RNA_SEED", 0)),
        device=Devices.GPU,
        hidden_dim=int(os.environ.get("RNA_HIDDEN_DIM", 256)),
        ctx_dim=int(os.environ.get("RNA_CTX_DIM", 128)),
        n_steps=int(os.environ.get("RNA_N_STEPS", 20)),
        dt=float(os.environ.get("RNA_DT", 0.125)),
        solver=Solvers.CFEES25,
        diffusion_scale=float(os.environ.get("RNA_DIFFUSION_SCALE", 0.1)),
        context_length=int(os.environ.get("RNA_CONTEXT_LENGTH", 20)),
        residues_per_state=1,
        future_bases_window=int(os.environ.get("RNA_FUTURE_BASES_WINDOW", 0)),
        future_ctx_dim=int(os.environ.get("RNA_FUTURE_CTX_DIM", 32)),
        activation=os.environ.get("RNA_ACTIVATION", "silu"),
        drift_depth=int(os.environ.get("RNA_DRIFT_DEPTH", 3)),
        diffusion_depth=int(os.environ.get("RNA_DIFFUSION_DEPTH", 2)),
        filter_canonical=os.environ.get("RNA_FILTER_CANONICAL", "0") == "1",
        canonical_threshold=float(os.environ.get("RNA_CANONICAL_THRESHOLD", 1.0)),
        max_chains=int(os.environ.get("RNA_MAX_CHAINS", 1000)),
    )
    n_mc_samples = int(os.environ.get("RNA_N_MC_SAMPLES", 16))
    warmup_steps = int(os.environ.get("RNA_WARMUP", 1000))
    weight_decay = float(os.environ.get("RNA_WEIGHT_DECAY", 1e-4))
    use_energy_score = True
    energy_score_samples = int(os.environ.get("RNA_ENERGY_SAMPLES", 4))

    train_ds = RNATorsionDataset(
        split="train",
        max_chains=config.max_chains,
        context_length=config.context_length,
        residues_per_state=config.residues_per_state,
        future_bases_window=config.future_bases_window,
        filter_canonical=config.filter_canonical,
        canonical_threshold=config.canonical_threshold,
    )
    metadata = train_ds.metadata()

    key = jax.random.key(config.seed)
    model = make_model(config, metadata, key)
    if use_energy_score:
        loss_fn = make_wrapped_energy_score_loss(n_samples=energy_score_samples)
    else:
        loss_fn = make_wrapped_mse_loss()
    metric_fn = make_wrapped_mae_metric()
    train_loader = make_loader(config, "train")
    val_loader = make_loader(config, "val")

    total_train_steps = train_loader.steps_per_epoch * config.epochs
    end_lr_frac = float(os.environ.get("RNA_END_LR_FRAC", 0.05))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=min(warmup_steps, total_train_steps // 4),
        decay_steps=max(total_train_steps - warmup_steps, 1),
        end_value=config.learning_rate * end_lr_frac,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    key, train_key, val_key = jax.random.split(key, 3)
    train_state = train_loader.init_state(train_key)
    val_state = val_loader.init_state(val_key)
    print(
        f"train={train_loader.steps_per_epoch * config.batch_size}"
        f" val={val_loader.steps_per_epoch * config.batch_size}"
        f" n_mc={n_mc_samples} warmup={warmup_steps}/{total_train_steps}",
        flush=True,
    )

    @eqx.filter_jit
    def run_train_epoch(model, opt_state, train_state, key):
        model_arr, model_static = eqx.partition(model, eqx.is_array)
        opt_arr, opt_static = eqx.partition(opt_state, eqx.is_array)

        def body(carry, batch, mask):
            m_arr, o_arr, k, loss_sum = carry
            m = eqx.combine(m_arr, model_static)
            opt_s = eqx.combine(o_arr, opt_static)
            k, step_key = jax.random.split(k)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(m, batch, mask, step_key)
            updates, new_opt_s = optimizer.update(grads, opt_s, m)
            new_m = eqx.apply_updates(m, updates)
            new_m_arr, _ = eqx.partition(new_m, eqx.is_array)
            new_o_arr, _ = eqx.partition(new_opt_s, eqx.is_array)
            return (new_m_arr, new_o_arr, k, loss_sum + loss), None

        init_carry = (model_arr, opt_arr, key, jnp.zeros(()))
        train_state, final_carry, _ = train_loader.scan_epoch(
            train_state, init_carry, body
        )
        new_m_arr, new_o_arr, new_key, total_loss = final_carry
        new_model = eqx.combine(new_m_arr, model_static)
        new_opt_state = eqx.combine(new_o_arr, opt_static)
        return new_model, new_opt_state, train_state, new_key, total_loss

    @eqx.filter_jit
    def run_val_epoch(model, val_state, key):
        model_arr, model_static = eqx.partition(model, eqx.is_array)

        def body(carry, batch, mask):
            m_arr, k, metric_sum = carry
            m = eqx.combine(m_arr, model_static)
            k, step_key = jax.random.split(k)
            metric = metric_fn(m, batch, mask, step_key)
            return (m_arr, k, metric_sum + metric), None

        init_carry = (model_arr, key, jnp.zeros(()))
        val_state, final_carry, _ = val_loader.scan_epoch(
            val_state, init_carry, body
        )
        _, _, total_metric = final_carry
        return val_state, total_metric

    best_ckpt_path = PROJECT_ROOT / "results" / "rna_best_model.eqx"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_model = model
    best_val = float("inf")
    val_base_key = jax.random.key(config.seed + 1000)
    initial_val_state = val_state
    for epoch in range(config.epochs):
        epoch_start = time.time()
        model, opt_state, train_state, key, train_loss_sum = run_train_epoch(
            model, opt_state, train_state, key
        )
        _, val_metric_sum = run_val_epoch(model, initial_val_state, val_base_key)
        train_loss_sum.block_until_ready()
        val_metric_sum.block_until_ready()
        elapsed = time.time() - epoch_start
        train_loss = float(train_loss_sum) / train_loader.steps_per_epoch
        val_metric = float(val_metric_sum) / val_loader.steps_per_epoch
        if val_metric < best_val:
            best_val = val_metric
            best_model = model
            eqx.tree_serialise_leaves(best_ckpt_path, best_model)
        print(
            f"epoch {epoch + 1:2d}/{config.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_mae={val_metric:.4f}  best_val={best_val:.4f}  "
            f"wall={elapsed:.1f}s",
            flush=True,
        )
    model = best_model

    test_ds = RNATorsionDataset(
        split="test",
        max_chains=config.max_chains,
        context_length=config.context_length,
        residues_per_state=config.residues_per_state,
        future_bases_window=config.future_bases_window,
        filter_canonical=config.filter_canonical,
        canonical_threshold=config.canonical_threshold,
    )
    arrays = test_ds.as_array_dict()
    context_angles = jnp.asarray(arrays["context_angles"])
    context_bases = jnp.asarray(arrays["context_bases"])
    target_angles = jnp.asarray(arrays["target_angles"])
    target_bases = jnp.asarray(arrays["target_bases"])
    future_bases = jnp.asarray(arrays["future_bases"])
    future_mask = jnp.asarray(arrays["future_mask"])

    # Test-time averaging: draw n_mc_samples stochastic rollouts per context
    # and circular-mean them on the torus.
    base_key = jax.random.key(config.seed + 99)
    per_sample_keys = jax.random.split(base_key, n_mc_samples)

    @jax.jit
    def predict_ensemble(ctx_a, ctx_b, tgt_b, fut_b, fut_m, keys):
        def one_sample(k):
            sub_keys = jax.random.split(k, ctx_a.shape[0])
            return jax.vmap(
                lambda a, cb, tb, fb, fm, kk: model(a, cb, tb, fb, fm, kk)
            )(ctx_a, ctx_b, tgt_b, fut_b, fut_m, sub_keys)
        samples = jax.vmap(one_sample)(keys)  # (n_mc, n_test, d)
        mean_sin = jnp.mean(jnp.sin(samples), axis=0)
        mean_cos = jnp.mean(jnp.cos(samples), axis=0)
        return jnp.arctan2(mean_sin, mean_cos)

    predicted = jax.device_get(
        predict_ensemble(
            context_angles, context_bases, target_bases, future_bases, future_mask, per_sample_keys
        )
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        predicted=np.asarray(predicted),
        actual=np.asarray(target_angles),
        context_angles=np.asarray(context_angles),
        context_bases=np.asarray(context_bases),
        target_bases=np.asarray(target_bases),
        future_bases=np.asarray(future_bases),
        future_mask=np.asarray(future_mask),
    )
    print(f"saved {OUT_PATH}", flush=True)
    print(f"n_test = {predicted.shape[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
