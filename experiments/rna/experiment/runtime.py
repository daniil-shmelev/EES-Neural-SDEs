"""Helpers shared by the SPD and RNA training entry points."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import equinox as eqx

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def model_name(model: eqx.Module) -> str:
    raw = getattr(model, "name", None)
    name = str(raw() if callable(raw) else raw or model.__class__.__name__).strip()
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return sanitized or model.__class__.__name__.lower()


def make_output_dir(model: eqx.Module) -> Path:
    timestamp = datetime.now().astimezone().strftime("%H%M%S_%Y%m%d")
    return PROJECT_ROOT / "results" / f"{model_name(model)}_{timestamp}"
