"""Partial evaluation batches must not outweigh full batches in model selection."""

import jax.numpy as jnp
import pytest
from cyreal.loader import DataLoader
from cyreal.sources import ArraySource
from cyreal.transforms import BatchTransform

from experiments.sphere_latent_nsde import train_activity
from experiments.sphere_latent_nsde.config import ActivityConfig
from experiments.sphere_latent_nsde.losses import masked_mean


def test_partial_batch_weighting_and_best_checkpoint(monkeypatch):
    def make_loader(config, split):
        return DataLoader(
            [
                ArraySource(
                    {
                        "loss": jnp.array([2.0, 2.0, 8.0]),
                        "accuracy": jnp.array([1.0, 1.0, 0.0]),
                        "work": jnp.array([10.0, 10.0, 20.0]),
                    },
                    ordering="sequential",
                ),
                BatchTransform(batch_size=2, drop_last=False, pad_last_batch=True),
            ]
        )

    def loss(model, batch, mask, key, **kwargs):
        return model**2, {"num_steps": jnp.array(10)}

    def metrics(model, batch, mask, key, **kwargs):
        # Adam takes the scalar model from 1 to ~.99, then ~.98. Epoch 1 has
        # accuracy 2/3 and epoch 2 has .6; averaging batch means picks epoch 2.
        accuracy = jnp.where(model < 0.985, 0.6, batch["accuracy"])
        accuracy = masked_mean(accuracy, mask)
        return {
            "loss": masked_mean(batch["loss"], mask),
            "aux_acc": accuracy,
            "aux_acc_pct": 100.0 * accuracy,
            "solver_num_steps": batch["work"][0],
        }

    monkeypatch.setattr(train_activity, "make_loader", make_loader)
    monkeypatch.setattr(train_activity, "activity_loss_value", loss)
    monkeypatch.setattr(train_activity, "activity_metrics_value", metrics)
    result = train_activity.fit(
        jnp.array(1.0),
        ActivityConfig(
            epochs=2,
            batch_size=2,
            learning_rate=0.01,
            max_train_batches=1,
        ),
    )
    for row in result.history["val"] + result.history["test"]:
        assert row["loss"] == pytest.approx(4.0)
        assert row["solver_num_steps"] == pytest.approx(15.0)
        assert row["best_val_epoch"] == 1
    assert result.history["val"][0]["aux_acc"] == pytest.approx(2 / 3)
    assert result.history["val"][1]["aux_acc"] == pytest.approx(0.6)
    assert result.history["test"][0]["aux_acc"] == pytest.approx(2 / 3)
    assert float(result.best_model) > 0.985
