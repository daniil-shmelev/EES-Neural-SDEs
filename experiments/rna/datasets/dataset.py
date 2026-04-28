"""Cyreal dataset for one-step RNA torsion-angle forecasting on T^d."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jax
import numpy as np
from cyreal.datasets.dataset_protocol import DatasetProtocol
from cyreal.datasets.utils import to_host_jax_array
from cyreal.sources import ArraySource

from experiments.rna.datasets.download import (
    CACHE_DIR,
    PDB_CACHE_DIR,
    ensure_pdb_files,
    fetch_bgsu_nr_list,
    parse_chain_token,
)
from experiments.rna.datasets.preprocessing import (
    NUM_ANGLES,
    NUM_BASES,
    compute_chain_torsions,
    load_chain_atoms,
    split_into_contiguous_runs,
)

logger = logging.getLogger(__name__)

# Bumped when the on-disk cache layout changes. Old caches are ignored.
CACHE_LAYOUT_VERSION = "v2"


@dataclass
class RNATorsionDataset(DatasetProtocol):
    """One-step-ahead forecasting dataset on the n-torus.

    Each sample is a dict with:
    - ``context_angles``: ``(context_length, 7 * residues_per_state)`` float32
    - ``context_bases``:  ``(context_length * residues_per_state,)`` int32 in 0..3
    - ``target_angles``:  ``(7 * residues_per_state,)`` float32
    - ``target_bases``:   ``(residues_per_state,)`` int32 in 0..3

    Splits are over chain identity (not residue index) so that no residue
    leaks between train/val/test.
    """

    split: Literal["train", "val", "test"] = "train"
    context_length: int = 20
    residues_per_state: int = 1
    future_bases_window: int = 0
    filter_canonical: bool = False
    canonical_threshold: float = 1.0
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    cutoff_angstrom: float = 3.0
    max_chains: int | None = 200
    cache_dir: Path = CACHE_DIR
    pdb_cache_dir: Path = PDB_CACHE_DIR
    min_chain_length: int = 30
    seed: int = 0
    ordering: Literal["sequential", "shuffle"] = field(init=False)

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")
        if self.residues_per_state <= 0:
            raise ValueError("residues_per_state must be positive.")
        if self.future_bases_window < 0:
            raise ValueError("future_bases_window must be non-negative.")
        self.ordering = "shuffle" if self.split == "train" else "sequential"

        chains = _load_or_build_chain_angles(
            cutoff_angstrom=self.cutoff_angstrom,
            max_chains=self.max_chains,
            cache_dir=Path(self.cache_dir),
            pdb_cache_dir=Path(self.pdb_cache_dir),
            min_chain_length=self.min_chain_length,
        )
        chain_ids = sorted(chains.keys())

        train_ids, val_ids, test_ids = _chain_level_split(
            chain_ids,
            train_fraction=self.train_fraction,
            val_fraction=self.val_fraction,
            seed=self.seed,
        )
        if self.split == "train":
            split_ids = train_ids
        elif self.split == "val":
            split_ids = val_ids
        else:
            split_ids = test_ids

        c_angles, c_bases, t_angles, t_bases, f_bases, f_mask = _build_pairs(
            {cid: chains[cid] for cid in split_ids},
            context_length=self.context_length,
            residues_per_state=self.residues_per_state,
            future_bases_window=self.future_bases_window,
        )

        if self.filter_canonical and len(t_angles) > 0:
            circ_mean = np.arctan2(
                np.mean(np.sin(t_angles), axis=0),
                np.mean(np.cos(t_angles), axis=0),
            )
            diff = t_angles - circ_mean[None]
            wrapped = np.arctan2(np.sin(diff), np.cos(diff))
            dist = np.linalg.norm(wrapped, axis=1)
            keep = dist >= self.canonical_threshold
            c_angles = c_angles[keep]
            c_bases = c_bases[keep]
            t_angles = t_angles[keep]
            t_bases = t_bases[keep]
            f_bases = f_bases[keep]
            f_mask = f_mask[keep]

        self._context_angles = to_host_jax_array(c_angles)
        self._context_bases = to_host_jax_array(c_bases)
        self._target_angles = to_host_jax_array(t_angles)
        self._target_bases = to_host_jax_array(t_bases)
        self._future_bases = to_host_jax_array(f_bases)
        self._future_mask = to_host_jax_array(f_mask)
        self._num_angles = int(NUM_ANGLES * self.residues_per_state)
        self._n_chains = len(split_ids)

    def __len__(self) -> int:
        return int(self._context_angles.shape[0])

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        return {
            "context_angles": self._context_angles[index],
            "context_bases": self._context_bases[index],
            "target_angles": self._target_angles[index],
            "target_bases": self._target_bases[index],
            "future_bases": self._future_bases[index],
            "future_mask": self._future_mask[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        return {
            "context_angles": self._context_angles,
            "context_bases": self._context_bases,
            "target_angles": self._target_angles,
            "target_bases": self._target_bases,
            "future_bases": self._future_bases,
            "future_mask": self._future_mask,
        }

    def metadata(self) -> dict[str, int]:
        return {
            "num_angles": self._num_angles,
            "num_bases": NUM_BASES,
            "context_length": self.context_length,
            "residues_per_state": self.residues_per_state,
            "future_bases_window": self.future_bases_window,
            "dataset_size": len(self),
            "n_chains": self._n_chains,
        }

    def make_array_source(self) -> ArraySource:
        return ArraySource(self.as_array_dict(), ordering=self.ordering)


def _chain_level_split(
    chain_ids: list[str],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    train, val, test = [], [], []
    for cid in chain_ids:
        h = hashlib.sha256(f"{seed}:{cid}".encode()).digest()
        bucket = int.from_bytes(h[:8], "big") / 2**64
        if bucket < train_fraction:
            train.append(cid)
        elif bucket < train_fraction + val_fraction:
            val.append(cid)
        else:
            test.append(cid)
    return train, val, test


def _build_pairs(
    chains: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    *,
    context_length: int,
    residues_per_state: int,
    future_bases_window: int = 0,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Build all six arrays needed for the RNA one-step forecasting task.

    Returns ``(context_angles, context_bases, target_angles, target_bases,
    future_bases, future_mask)``. The first four behave as before; the last
    two expose the next ``future_bases_window`` states' base identities to
    the model as free inference-time information (known from the chain
    sequence). ``future_mask[i, j] == 1`` iff the ``j``-th future state of
    pair ``i`` falls inside the same contiguous run; pad entries are zero.
    When ``future_bases_window == 0`` the future arrays have a zero-sized
    window axis and the mask is empty.
    """
    state_dim = NUM_ANGLES * residues_per_state
    bases_per_state = residues_per_state
    f = future_bases_window
    c_angles_chunks: list[np.ndarray] = []
    c_bases_chunks: list[np.ndarray] = []
    t_angles_chunks: list[np.ndarray] = []
    t_bases_chunks: list[np.ndarray] = []
    f_bases_chunks: list[np.ndarray] = []
    f_mask_chunks: list[np.ndarray] = []

    for runs in chains.values():
        for run_angles, run_bases in runs:
            n_states = run_angles.shape[0] // residues_per_state
            if n_states <= context_length:
                continue
            usable = n_states * residues_per_state
            grouped_angles = run_angles[:usable].reshape(n_states, state_dim)
            grouped_bases = run_bases[:usable].reshape(n_states, bases_per_state)
            num_pairs = n_states - context_length
            row_idx = np.arange(num_pairs)[:, None] + np.arange(context_length)[None, :]
            c_angles_chunks.append(grouped_angles[row_idx])
            c_bases_chunks.append(grouped_bases[row_idx])
            t_angles_chunks.append(
                grouped_angles[context_length : context_length + num_pairs]
            )
            t_bases_chunks.append(
                grouped_bases[context_length : context_length + num_pairs]
            )

            if f > 0:
                # Future window of pair i spans states [i+L+1 .. i+L+f].
                future_idx = (
                    np.arange(num_pairs)[:, None]
                    + (context_length + 1)
                    + np.arange(f)[None, :]
                )
                valid = future_idx < n_states
                safe_idx = np.where(valid, future_idx, 0)
                f_bases_chunks.append(grouped_bases[safe_idx])
                f_mask_chunks.append(valid.astype(np.int8))

    if not c_angles_chunks:
        return (
            np.zeros((0, context_length, state_dim), dtype=np.float32),
            np.zeros((0, context_length, bases_per_state), dtype=np.int32),
            np.zeros((0, state_dim), dtype=np.float32),
            np.zeros((0, bases_per_state), dtype=np.int32),
            np.zeros((0, f, bases_per_state), dtype=np.int32),
            np.zeros((0, f), dtype=np.int8),
        )

    c_angles = np.concatenate(c_angles_chunks, axis=0).astype(np.float32)
    c_bases = np.concatenate(c_bases_chunks, axis=0).astype(np.int32)
    t_angles = np.concatenate(t_angles_chunks, axis=0).astype(np.float32)
    t_bases = np.concatenate(t_bases_chunks, axis=0).astype(np.int32)
    n = c_angles.shape[0]
    if f > 0:
        f_bases = np.concatenate(f_bases_chunks, axis=0).astype(np.int32)
        f_mask = np.concatenate(f_mask_chunks, axis=0).astype(np.int8)
    else:
        f_bases = np.zeros((n, 0, bases_per_state), dtype=np.int32)
        f_mask = np.zeros((n, 0), dtype=np.int8)
    return c_angles, c_bases, t_angles, t_bases, f_bases, f_mask


def _cache_key(*, cutoff_angstrom: float, max_chains: int | None) -> str:
    payload = (
        f"{CACHE_LAYOUT_VERSION}|cutoff={cutoff_angstrom}|max_chains={max_chains}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_or_build_chain_angles(
    *,
    cutoff_angstrom: float,
    max_chains: int | None,
    cache_dir: Path,
    pdb_cache_dir: Path,
    min_chain_length: int,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Return ``{chain_token: [(angles, base_ids), ...]}`` for valid runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"chain_angles_{_cache_key(cutoff_angstrom=cutoff_angstrom, max_chains=max_chains)}.npz"
    chains: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    if cache_path.exists():
        logger.info("Loading cached RNA torsion data from %s", cache_path)
        data = np.load(cache_path, allow_pickle=False)
        for key in data.files:
            if not key.startswith("angles__"):
                continue
            sanitized = key[len("angles__") :]
            token = sanitized.replace("__", "|")
            run_lengths = data[f"runs__{sanitized}"]
            angles = data[key]
            bases = data[f"bases__{sanitized}"]
            offset = 0
            runs: list[tuple[np.ndarray, np.ndarray]] = []
            for length in run_lengths:
                length = int(length)
                runs.append(
                    (angles[offset : offset + length], bases[offset : offset + length])
                )
                offset += length
            chains[token] = runs
        return chains

    logger.info(
        "Building RNA torsion cache (cutoff=%sA, max_chains=%s)",
        cutoff_angstrom,
        max_chains,
    )
    tokens = fetch_bgsu_nr_list(cutoff_angstrom=cutoff_angstrom)
    if max_chains is not None:
        tokens = tokens[:max_chains]
    pdb_ids = sorted({parse_chain_token(t)[0] for t in tokens})
    paths = ensure_pdb_files(pdb_ids, cache_dir=pdb_cache_dir)

    for token in tokens:
        try:
            pdb_id, model, chain_id = parse_chain_token(token)
        except ValueError as exc:
            logger.warning("Skipping malformed token %r: %s", token, exc)
            continue
        path = paths.get(pdb_id)
        if path is None:
            continue
        atoms = load_chain_atoms(path, chain_id, model=model)
        if atoms is None:
            continue
        angles, res_ids, base_ids = compute_chain_torsions(atoms)
        if len(angles) < min_chain_length:
            continue
        runs = [
            (a, b)
            for (a, b) in split_into_contiguous_runs(angles, res_ids, base_ids)
            if len(a) >= min_chain_length
        ]
        if not runs:
            continue
        chains[token] = runs

    payload: dict[str, np.ndarray] = {}
    for token, runs in chains.items():
        sanitized = token.replace("|", "__")
        run_angles = [a for (a, _) in runs]
        run_bases = [b for (_, b) in runs]
        all_angles = np.concatenate(run_angles, axis=0).astype(np.float32)
        all_bases = np.concatenate(run_bases, axis=0).astype(np.int8)
        run_lengths = np.asarray([len(a) for a in run_angles], dtype=np.int64)
        payload[f"angles__{sanitized}"] = all_angles
        payload[f"bases__{sanitized}"] = all_bases
        payload[f"runs__{sanitized}"] = run_lengths

    np.savez(cache_path, **payload)
    logger.info("Cached %d chains to %s", len(chains), cache_path)
    return chains
