"""Cyreal dataset for the HumanActivity classification task."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import jax
import numpy as np
from cyreal.datasets.dataset_protocol import DatasetProtocol
from cyreal.datasets.utils import to_host_jax_array
from cyreal.sources import ArraySource


INPUT_DIM = 12
NUM_TIMEPOINTS = 228
NUM_CLASSES = 7
MAX_SEQ_LENGTH = 50

TAG_IDS = (
    "010-000-024-033",
    "010-000-030-096",
    "020-000-033-111",
    "020-000-032-221",
)
TAG_DICT = {tag: i for i, tag in enumerate(TAG_IDS)}

LABEL_NAMES = (
    "walking",
    "falling",
    "lying down",
    "lying",
    "sitting down",
    "sitting",
    "standing up from lying",
    "on all fours",
    "sitting on the ground",
    "standing up from sitting",
    "standing up from sit on grnd",
)
LABEL_DICT = {
    "walking": 0,
    "falling": 1,
    "lying": 2,
    "lying down": 2,
    "sitting": 3,
    "sitting down": 3,
    "standing up from lying": 4,
    "standing up from sitting": 4,
    "standing up from sit on grnd": 4,
    "on all fours": 5,
    "sitting on the ground": 6,
}


def remap_time_ids(tid: np.ndarray, num_timepoints: int) -> np.ndarray:
    """Map raw 228-point activity time IDs onto a model output grid."""

    num_timepoints = int(num_timepoints)
    if num_timepoints < 2:
        raise ValueError("num_timepoints must be at least 2.")
    if num_timepoints == NUM_TIMEPOINTS:
        return tid.astype(np.int32, copy=True)

    scale = (num_timepoints - 1) / (NUM_TIMEPOINTS - 1)
    mapped = np.rint(tid.astype(np.float32) * scale)
    return np.clip(mapped, 0, num_timepoints - 1).astype(np.int32)


def _fractional_split_lengths(n: int, fractions: tuple[float, ...]) -> list[int]:
    raw = np.asarray(fractions, dtype=np.float64) * n
    lengths = np.floor(raw).astype(np.int64)
    remainder = n - int(lengths.sum())
    for i in range(remainder):
        lengths[i % len(lengths)] += 1
    return [int(x) for x in lengths]


def _split_indices(
    n: int,
    *,
    split: Literal["train", "val", "test"],
    seed: int,
    strategy: Literal["auto", "numpy", "torch"] = "auto",
) -> np.ndarray:
    lengths = _fractional_split_lengths(n, (0.64, 0.16, 0.20))
    if strategy in ("auto", "torch"):
        try:
            import torch
        except ModuleNotFoundError as exc:
            if strategy == "torch":
                raise ModuleNotFoundError(
                    "split_strategy='torch' requires torch. Use the original repo's "
                    "venv for exact split parity, or split_strategy='numpy'."
                ) from exc
        else:
            generator = torch.Generator().manual_seed(seed)
            full_idx = torch.arange(n)
            train, val, test = torch.utils.data.random_split(
                full_idx, lengths, generator=generator
            )
            parts = {
                "train": np.asarray(train.indices, dtype=np.int32),
                "val": np.asarray(val.indices, dtype=np.int32),
                "test": np.asarray(test.indices, dtype=np.int32),
            }
            return parts[split]

    perm = np.random.default_rng(seed).permutation(n).astype(np.int32)
    n_train, n_val, _ = lengths
    if split == "train":
        return perm[:n_train]
    if split == "val":
        return perm[n_train : n_train + n_val]
    return perm[n_train + n_val :]


def _empty_step() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vals = np.zeros((len(TAG_IDS), 3), dtype=np.float32)
    mask = np.zeros((len(TAG_IDS), 3), dtype=np.float32)
    nobs = np.zeros((len(TAG_IDS),), dtype=np.float32)
    labels = np.zeros((len(LABEL_NAMES),), dtype=np.float32)
    return vals, mask, nobs, labels


def _append_windows(
    records: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    record_id: str,
    tt: list[int],
    vals: list[np.ndarray],
    masks: list[np.ndarray],
    labels: list[np.ndarray],
    *,
    max_seq_length: int,
) -> None:
    times = np.asarray(tt, dtype=np.float32)
    val_arr = np.asarray(vals, dtype=np.float32).reshape(len(tt), -1)
    mask_arr = np.asarray(masks, dtype=np.float32).reshape(len(tt), -1)
    label_arr = np.asarray(labels, dtype=np.float32)

    seq_length = len(times)
    offset = 0
    slide = max_seq_length // 2
    while offset + max_seq_length < seq_length:
        idx = slice(offset, offset + max_seq_length)
        first_tp = times[idx][0]
        records.append(
            (
                record_id,
                times[idx] - first_tp,
                val_arr[idx],
                mask_arr[idx],
                label_arr[idx],
            )
        )
        offset += slide


def _parse_raw_file(raw_file: Path, *, max_seq_length: int) -> dict[str, np.ndarray]:
    records: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    record_id: str | None = None
    first_tp = 0.0
    prev_time = -1
    tt: list[int] = []
    vals: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    nobs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with raw_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 8:
                raise ValueError(f"Unexpected row with {len(parts)} columns in {raw_file}")
            cur_record_id, tag_id, time_s, _date, val1, val2, val3, label = parts
            value_vec = np.asarray((float(val1), float(val2), float(val3)), dtype=np.float32)
            absolute_time = float(time_s)

            if cur_record_id != record_id:
                if record_id is not None:
                    _append_windows(
                        records,
                        record_id,
                        tt,
                        vals,
                        masks,
                        labels,
                        max_seq_length=max_seq_length,
                    )
                record_id = cur_record_id
                first_tp = absolute_time
                time = round((absolute_time - first_tp) / 10**5)
                prev_time = time
                tt = [0]
                val0, mask0, nobs0, labels0 = _empty_step()
                vals = [val0]
                masks = [mask0]
                nobs = [nobs0]
                labels = [labels0]
            else:
                time = round((absolute_time - first_tp) / 10**5)

            if time != prev_time:
                tt.append(int(time))
                val_t, mask_t, nobs_t, labels_t = _empty_step()
                vals.append(val_t)
                masks.append(mask_t)
                nobs.append(nobs_t)
                labels.append(labels_t)
                prev_time = time

            if tag_id in TAG_DICT:
                tag_idx = TAG_DICT[tag_id]
                n_observations = nobs[-1][tag_idx]
                if n_observations > 0:
                    vals[-1][tag_idx] = (
                        vals[-1][tag_idx] * n_observations + value_vec
                    ) / (n_observations + 1.0)
                else:
                    vals[-1][tag_idx] = value_vec

                masks[-1][tag_idx] = 1.0
                nobs[-1][tag_idx] += 1.0
                if label in LABEL_DICT and labels[-1][LABEL_DICT[label]] == 0:
                    labels[-1][LABEL_DICT[label]] = 1.0
            elif tag_id != "RecordID":
                raise ValueError(f"Read unexpected tag id {tag_id!r}")

    if record_id is not None:
        _append_windows(
            records,
            record_id,
            tt,
            vals,
            masks,
            labels,
            max_seq_length=max_seq_length,
        )

    tids = np.stack([r[1] for r in records]).astype(np.int32)
    obs = np.stack([r[2] for r in records]).astype(np.float32)
    masks_arr = np.stack([r[3] for r in records]).astype(np.float32)
    labels_arr = np.stack([r[4][:, :NUM_CLASSES].argmax(axis=1) for r in records]).astype(
        np.int32
    )
    return {
        "obs": obs,
        "mask": masks_arr,
        "tid": tids,
        "labels": labels_arr,
    }


def _load_with_torch(processed_file: Path) -> dict[str, np.ndarray]:
    import torch

    data = torch.load(processed_file, map_location="cpu")[:8000]
    obs = torch.stack([r[2] for r in data]).numpy().astype(np.float32)
    masks = torch.stack([r[3] for r in data]).numpy().astype(np.float32)
    tids = torch.stack([r[1] for r in data]).long().numpy().astype(np.int32)
    labels = (
        torch.stack([r[4] for r in data])[:, :, :NUM_CLASSES]
        .argmax(dim=2)
        .long()
        .numpy()
        .astype(np.int32)
    )
    return {"obs": obs, "mask": masks, "tid": tids, "labels": labels}


@lru_cache(maxsize=4)
def _load_activity_arrays_cached(
    data_dir: str,
    source: Literal["auto", "raw", "torch"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(data_dir)
    processed_file = root / "PersonActivity" / "processed" / "data.pt"
    raw_file = root / "PersonActivity" / "raw" / "ConfLongDemo_JSI.txt"

    if source in ("auto", "torch") and processed_file.exists():
        try:
            arrays = _load_with_torch(processed_file)
            return arrays["obs"], arrays["mask"], arrays["tid"], arrays["labels"]
        except ModuleNotFoundError:
            if source == "torch":
                raise

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Could not load activity data. Missing raw file {raw_file}; "
            f"processed torch file is {processed_file}."
        )
    arrays = _parse_raw_file(raw_file, max_seq_length=MAX_SEQ_LENGTH)
    return arrays["obs"], arrays["mask"], arrays["tid"], arrays["labels"]


def load_activity_arrays(
    data_dir: str | Path,
    *,
    source: Literal["auto", "raw", "torch"] = "auto",
) -> dict[str, np.ndarray]:
    obs, mask, tid, labels = _load_activity_arrays_cached(str(Path(data_dir)), source)
    return {"obs": obs, "mask": mask, "tid": tid, "labels": labels}


@dataclass
class HumanActivityDataset(DatasetProtocol):
    split: Literal["train", "val", "test"] = "train"
    data_dir: Path = Path("experiments/sphere_latent_sde/data_dir")
    seed: int = 42
    source: Literal["auto", "raw", "torch"] = "auto"
    split_strategy: Literal["auto", "numpy", "torch"] = "auto"
    output_num_timepoints: int = NUM_TIMEPOINTS
    ordering: Literal["sequential", "shuffle"] = field(init=False)

    def __post_init__(self) -> None:
        self.ordering = "shuffle" if self.split == "train" else "sequential"
        arrays = load_activity_arrays(self.data_dir, source=self.source)
        idx = _split_indices(
            len(arrays["obs"]),
            split=self.split,
            seed=self.seed,
            strategy=self.split_strategy,
        )

        obs = arrays["obs"][idx]
        mask = arrays["mask"][idx]
        tid = arrays["tid"][idx]
        labels = arrays["labels"][idx]
        output_tid = remap_time_ids(tid, self.output_num_timepoints)

        self._obs = to_host_jax_array(obs)
        self._mask = to_host_jax_array(mask)
        self._tid = to_host_jax_array(tid)
        self._output_tid = to_host_jax_array(output_tid)
        self._labels = to_host_jax_array(labels)
        self._tps = to_host_jax_array(tid.astype(np.float32) / NUM_TIMEPOINTS)

    def __len__(self) -> int:
        return int(self._obs.shape[0])

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        return {key: val[index] for key, val in self.as_array_dict().items()}

    def as_array_dict(self) -> dict[str, jax.Array]:
        return {
            "inp_obs": self._obs,
            "inp_msk": self._mask,
            "inp_tid": self._tid,
            "inp_tps": self._tps,
            "evd_obs": self._obs,
            "evd_msk": self._mask,
            "evd_tid": self._output_tid,
            "aux_obs": self._labels,
            "aux_tid": self._output_tid,
        }

    def metadata(self) -> dict[str, int]:
        return {
            "input_dim": INPUT_DIM,
            "num_timepoints": int(self.output_num_timepoints),
            "raw_num_timepoints": NUM_TIMEPOINTS,
            "num_classes": NUM_CLASSES,
            "dataset_size": len(self),
            "max_seq_length": MAX_SEQ_LENGTH,
        }

    def make_array_source(self) -> ArraySource:
        return ArraySource(self.as_array_dict(), ordering=self.ordering)
