"""Evaluation uses one ensemble and counts each held-out example once."""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.kuramoto.experiment.config import ExperimentConfig
from experiments.kuramoto.experiment.factories import make_loader
from experiments.kuramoto.experiment.losses import (
    make_eval_metrics,
    make_eval_step,
    make_multi_horizon_energy_score,
)
from experiments.kuramoto.experiment.train_kuramoto import _eval_split, fit


class Forecast(eqx.Module):
    scale: jax.Array

    def __call__(self, theta, omega, key, *, save_indices=None):
        noise = 0.1 * jax.random.normal(key, (3, theta.shape[0]))
        th, om = theta[None] + self.scale * noise, omega[None] + self.scale * noise
        if save_indices is not None:
            th, om = th[save_indices], om[save_indices]
        return th, om


def test_shared_evaluation_matches_separate_reductions():
    batch = {
        "theta0": jnp.array([[0.1], [0.2]]),
        "omega0": jnp.array([[0.3], [0.4]]),
        "theta_traj": jnp.zeros((2, 3, 1)),
        "omega_traj": jnp.zeros((2, 3, 1)),
    }
    model = Forecast(jnp.array(1.0))
    key = jax.random.key(7)
    mask = jnp.array([True, True])
    loss, metrics = make_eval_step()(model, batch, mask, key)
    expected_loss = make_multi_horizon_energy_score()(model, batch, mask, key)
    expected_metrics = make_eval_metrics()(model, batch, key)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    for name in metrics:
        np.testing.assert_allclose(metrics[name], expected_metrics[name], rtol=1e-6)


def test_padding_does_not_change_loss_or_metrics():
    # Deterministic forecasts make the expected loss and MAEs analytic.
    batch = {
        "theta0": jnp.array([[0.2], [0.4], [20.0]]),
        "omega0": jnp.array([[0.5], [0.7], [100.0]]),
        "theta_traj": jnp.zeros((3, 3, 1)),
        "omega_traj": jnp.zeros((3, 3, 1)),
    }
    mask = jnp.array([True, True, False])
    loss, metrics = make_eval_step()(
        Forecast(jnp.array(0.0)), batch, mask, jax.random.key(0)
    )
    np.testing.assert_allclose(loss, 0.9, atol=1e-6)
    np.testing.assert_allclose(metrics["theta_mae_per_horizon"], 0.3, atol=1e-6)
    np.testing.assert_allclose(metrics["omega_mae_per_horizon"], 0.6, atol=1e-6)


def _dataset(tmp_path):
    theta = np.broadcast_to(
        np.array([1.0, 2.0, 9.0], dtype=np.float32)[:, None, None], (3, 3, 1)
    )
    arrays = {
        f"{name}_{split}": theta
        for name in ("theta", "omega")
        for split in ("train", "val", "test")
    }
    path = tmp_path / "kuramoto_N1_seed0.npz"
    np.savez(path, **arrays, t_grid=np.linspace(0, 0.2, 3, dtype=np.float32))
    path.with_suffix(".json").write_text(json.dumps({"N": 1, "T": 0.2, "n_obs": 3}))
    return ExperimentConfig(N=1, batch_size=2, epochs=1, data_dir=str(tmp_path))


def test_evaluation_covers_split_once_and_weights_last_batch(tmp_path, capsys):
    config = _dataset(tmp_path)
    loader, _ = make_loader(config, "val")
    next_batch = jax.jit(loader.next)
    state = loader.init_state(jax.random.key(0))
    counts = []

    def evaluate(model, batch, mask, key):
        counts.append(int(jnp.sum(mask)))
        value = jnp.sum(batch["theta0"][:, 0] * mask) / jnp.sum(mask)
        return value, {"mae": value, "per_horizon": jnp.array([value, value])}

    for _ in range(2):
        loss, metrics, _, state = _eval_split(
            None,
            loader,
            next_batch,
            evaluate,
            state,
            jax.random.key(0),
            label="val",
        )
        assert loss == pytest.approx(4.0)
        assert metrics["mae"] == pytest.approx(4.0)
        np.testing.assert_allclose(metrics["per_horizon"], [4.0, 4.0])
    assert counts == [2, 1, 2, 1]
    assert "val batch=2/2 loss=" in capsys.readouterr().out


def test_training_weights_saved_before_interrupted_evaluation(tmp_path):
    config = _dataset(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    model = Forecast(jnp.array(1.0))

    def loss_fn(model, batch, mask, key):
        return model.scale**2

    def interrupted_eval(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fit(
            model,
            loss_fn=loss_fn,
            eval_step=interrupted_eval,
            config=config,
            output_dir=output,
        )
    restored = eqx.tree_deserialise_leaves(output / "model_epoch_000.eqx", model)
    assert float(restored.scale) < 1.0
    assert (output / "opt_state_latest.eqx").is_file()


def test_resume_after_failure_uses_last_completed_epoch(tmp_path):
    from experiments.kuramoto.experiment.train_kuramoto import _resume_checkpoints

    (tmp_path / "history.json").write_text(json.dumps({"val_loss": [3.0, 2.0]}))
    model = tmp_path / "model_epoch_001.eqx"
    optimiser = tmp_path / "opt_state_latest.eqx"
    model.touch()
    optimiser.touch()
    assert _resume_checkpoints(tmp_path) == (model, optimiser)
    # An optimiser saved before an interrupted evaluation belongs to a newer
    # epoch than the history. Never pair it with the earlier model.
    (tmp_path / "model_epoch_002.eqx").touch()
    with pytest.raises(ValueError, match="matching completed evaluation history"):
        _resume_checkpoints(tmp_path)
