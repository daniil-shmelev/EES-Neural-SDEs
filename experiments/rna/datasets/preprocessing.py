"""Compute RNA backbone + glycosidic torsion angles per residue using biotite."""

from __future__ import annotations

import logging
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
import biotite.structure.io.pdbx as pdbx_io
import numpy as np

logger = logging.getLogger(__name__)

# Standard nucleobase residue names recognised across PDB files.
RNA_RESIDUES: frozenset[str] = frozenset({"A", "G", "C", "U"})
PURINES: frozenset[str] = frozenset({"A", "G"})
PYRIMIDINES: frozenset[str] = frozenset({"C", "U"})

# Integer encoding of the four RNA bases used as a model feature.
BASE_TO_INDEX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}
NUM_BASES: int = len(BASE_TO_INDEX)

# Atom name conventions for the glycosidic chi torsion.
PURINE_CHI_ATOMS: tuple[str, str, str, str] = ("O4'", "C1'", "N9", "C4")
PYRIMIDINE_CHI_ATOMS: tuple[str, str, str, str] = ("O4'", "C1'", "N1", "C2")

ANGLE_NAMES: tuple[str, ...] = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi",
)
NUM_ANGLES: int = len(ANGLE_NAMES)


def load_chain_atoms(
    structure_path: Path, chain_id: str, *, model: int = 1
) -> struc.AtomArray | None:
    """Return the nucleotide atoms of ``chain_id`` in ``structure_path``.

    Accepts both legacy ``.pdb`` files and modern ``.cif``/``.pdbx`` files.
    """
    suffix = structure_path.suffix.lower()
    try:
        if suffix in (".cif", ".pdbx", ".mmcif"):
            structure = pdbx_io.get_structure(
                pdbx_io.CIFFile.read(str(structure_path)), model=model
            )
        else:
            structure = pdb_io.PDBFile.read(str(structure_path)).get_structure(
                model=model
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse %s: %s", structure_path, exc)
        return None

    chain = structure[structure.chain_id == chain_id]
    if chain.array_length() == 0:
        return None
    nucleotides = chain[struc.filter_nucleotides(chain)]
    if nucleotides.array_length() == 0:
        return None
    return nucleotides


def _build_atom_index(
    chain: struc.AtomArray,
) -> tuple[dict[int, str], dict[tuple[int, str], np.ndarray]]:
    """Return ``(res_id -> res_name, (res_id, atom_name) -> coord)``."""
    res_name_by_id: dict[int, str] = {}
    coord_by_key: dict[tuple[int, str], np.ndarray] = {}
    for i in range(chain.array_length()):
        res_id = int(chain.res_id[i])
        res_name = str(chain.res_name[i])
        atom_name = str(chain.atom_name[i])
        coord_by_key[(res_id, atom_name)] = chain.coord[i]
        if res_id not in res_name_by_id:
            res_name_by_id[res_id] = res_name
    return res_name_by_id, coord_by_key


def _dihedral(coords: list[np.ndarray | None]) -> float:
    if any(c is None for c in coords):
        return float("nan")
    return float(struc.dihedral(*coords))


def compute_chain_torsions(
    chain: struc.AtomArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(angles, res_ids, base_ids)`` for valid interior residues only.

    ``angles`` is ``(n_valid, 7)`` in radians, ordered as
    ``(alpha, beta, gamma, delta, epsilon, zeta, chi)``. ``base_ids`` is
    ``(n_valid,)`` int8 with values in ``{0, 1, 2, 3}`` mapping
    ``(A, C, G, U)``. Residues whose res_ids do not form a contiguous run,
    residues missing any required atom, and the leading/trailing residues
    for which alpha/beta or zeta are undefined are dropped. ``res_ids``
    records the kept residue numbers.
    """
    res_name_by_id, coord_by_key = _build_atom_index(chain)
    sorted_res_ids = sorted(res_name_by_id)

    def get(res_id: int, atom_name: str) -> np.ndarray | None:
        return coord_by_key.get((res_id, atom_name))

    angles: list[list[float]] = []
    kept_res_ids: list[int] = []
    kept_base_ids: list[int] = []

    for idx, res_id in enumerate(sorted_res_ids):
        prev_id = res_id - 1
        next_id = res_id + 1
        if prev_id not in res_name_by_id or next_id not in res_name_by_id:
            continue

        res_name = res_name_by_id[res_id]
        if res_name not in RNA_RESIDUES:
            continue

        chi_atoms = (
            PURINE_CHI_ATOMS if res_name in PURINES else PYRIMIDINE_CHI_ATOMS
        )

        per_residue = [
            _dihedral([get(prev_id, "O3'"), get(res_id, "P"), get(res_id, "O5'"), get(res_id, "C5'")]),
            _dihedral([get(res_id, "P"), get(res_id, "O5'"), get(res_id, "C5'"), get(res_id, "C4'")]),
            _dihedral([get(res_id, "O5'"), get(res_id, "C5'"), get(res_id, "C4'"), get(res_id, "C3'")]),
            _dihedral([get(res_id, "C5'"), get(res_id, "C4'"), get(res_id, "C3'"), get(res_id, "O3'")]),
            _dihedral([get(res_id, "C4'"), get(res_id, "C3'"), get(res_id, "O3'"), get(next_id, "P")]),
            _dihedral([get(res_id, "C3'"), get(res_id, "O3'"), get(next_id, "P"), get(next_id, "O5'")]),
            _dihedral([get(res_id, chi_atoms[0]), get(res_id, chi_atoms[1]), get(res_id, chi_atoms[2]), get(res_id, chi_atoms[3])]),
        ]
        if any(np.isnan(x) for x in per_residue):
            continue

        angles.append(per_residue)
        kept_res_ids.append(res_id)
        kept_base_ids.append(BASE_TO_INDEX[res_name])

    if not angles:
        return (
            np.zeros((0, NUM_ANGLES), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int8),
        )

    angles_arr = np.asarray(angles, dtype=np.float32)
    res_ids_arr = np.asarray(kept_res_ids, dtype=np.int64)
    base_ids_arr = np.asarray(kept_base_ids, dtype=np.int8)
    return angles_arr, res_ids_arr, base_ids_arr


def split_into_contiguous_runs(
    angles: np.ndarray,
    res_ids: np.ndarray,
    base_ids: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split ``(angles, base_ids)`` into runs along monotonically-spaced ``res_ids``.

    Returns a list of ``(run_angles, run_base_ids)`` pairs, one per contiguous
    stretch of residue numbers.
    """
    if len(angles) == 0:
        return []
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for i in range(1, len(res_ids)):
        if res_ids[i] - res_ids[i - 1] != 1:
            runs.append((angles[start:i], base_ids[start:i]))
            start = i
    runs.append((angles[start:], base_ids[start:]))
    return runs
