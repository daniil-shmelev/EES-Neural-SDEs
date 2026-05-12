"""Runtime helpers (output dirs, JSON / NPZ persistence, dtype) for the Kuramoto experiment."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def make_output_dir(tag: str, base: Path | str | None = None) -> Path:
    base_dir = Path(base) if base is not None else Path("experiments/kuramoto/results/runs/")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = base_dir / f"{stamp}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return np.asarray(obj).tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def save_json(path: Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(_to_jsonable(payload), indent=2))


def save_npz(path: Path, **arrays: Any) -> None:
    converted = {k: np.asarray(jax.device_get(v)) for k, v in arrays.items()}
    np.savez_compressed(path, **converted)
