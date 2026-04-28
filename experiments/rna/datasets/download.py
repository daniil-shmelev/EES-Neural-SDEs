"""BGSU non-redundant RNA chain list + RCSB PDB fetcher with on-disk cache."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from biotite.database import rcsb

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/rna")
PDB_CACHE_DIR = CACHE_DIR / "pdbs"

BGSU_URL_TEMPLATE = (
    "https://rna.bgsu.edu/rna3dhub/nrlist/download/current/{cutoff}A/csv"
)

# Fallback list used when the BGSU NR fetch fails. Hand-picked to span
# tRNAs, riboswitches, ribozymes, and small ribosomal fragments. Each entry
# is "PDB_ID|MODEL|CHAIN" in BGSU notation.
FALLBACK_CHAINS: tuple[str, ...] = (
    "1EHZ|1|A", "4TNA|1|A", "1F27|1|A", "1FFK|1|9", "1J5E|1|A",
    "1KXK|1|A", "1Y26|1|X", "1Y27|1|X", "2AVY|1|A", "2GIS|1|A",
    "2GO5|1|A", "2HOJ|1|A", "2HOK|1|A", "2HOL|1|A", "2HOM|1|A",
    "2HOO|1|A", "2HOP|1|A", "2NZ4|1|A", "2OE5|1|A", "2OEU|1|A",
    "2QUS|1|A", "2QUW|1|A", "2QUX|1|B", "2R8S|1|R", "2TRA|1|A",
    "3DIG|1|A", "3DIL|1|A", "3DIM|1|A", "3DIQ|1|A", "3DIR|1|A",
    "3DIS|1|A", "3DIX|1|A", "3DIY|1|A", "3DIZ|1|A", "3F4G|1|A",
    "3F4H|1|A", "3FU2|1|A", "3GS1|1|A", "3GS5|1|A", "3GS8|1|A",
    "3IRW|1|R", "3IZD|1|A", "3IZE|1|A", "3IZF|1|A", "3OWI|1|A",
    "3OWZ|1|A", "3Q3Z|1|A", "3SD1|1|A", "3SD3|1|A", "3SKI|1|A",
)


def _bgsu_url(cutoff_angstrom: float) -> str:
    if cutoff_angstrom not in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 20.0):
        raise ValueError(
            f"BGSU resolution cutoff must be one of "
            f"1.5/2.0/2.5/3.0/3.5/4.0/20.0; got {cutoff_angstrom}"
        )
    return BGSU_URL_TEMPLATE.format(cutoff=cutoff_angstrom)


def fetch_bgsu_nr_list(
    cutoff_angstrom: float = 3.0,
    *,
    timeout_seconds: float = 30.0,
) -> list[str]:
    """Return a list of representative chains as ``"PDB|MODEL|CHAIN"`` strings.

    A representative may be a multi-chain RNA joined with ``+``; we split on
    ``+`` and keep each chain individually. On any failure (network, parse,
    HTTP error) the function falls back to ``FALLBACK_CHAINS``.
    """
    try:
        url = _bgsu_url(cutoff_angstrom)
        response = requests.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": "CFEES-experiment/0.1"},
        )
        response.raise_for_status()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("BGSU NR fetch failed (%s); falling back to hardcoded list.", exc)
        return list(FALLBACK_CHAINS)

    chains: list[str] = []
    for line in response.text.splitlines():
        parts = [p.strip().strip('"') for p in line.split('","')]
        if len(parts) < 2:
            continue
        representative = parts[1].strip('"')
        for chain in representative.split("+"):
            chain = chain.strip()
            if chain:
                chains.append(chain)

    if not chains:
        logger.warning("BGSU NR list parsed empty; falling back to hardcoded list.")
        return list(FALLBACK_CHAINS)

    return chains


def parse_chain_token(token: str) -> tuple[str, int, str]:
    """Parse ``"PDB|MODEL|CHAIN"`` into ``(pdb_id, model, chain_id)``."""
    parts = token.split("|")
    if len(parts) != 3:
        raise ValueError(f"unexpected chain token format: {token!r}")
    pdb_id, model_str, chain_id = parts
    return pdb_id.upper(), int(model_str), chain_id


def ensure_pdb_files(
    pdb_ids: list[str],
    *,
    cache_dir: Path = PDB_CACHE_DIR,
    max_workers: int = 8,
) -> dict[str, Path]:
    """Fetch missing structures into ``cache_dir`` and return a path map.

    Tries CIF first (most modern large RNA structures are CIF-only), then
    falls back to legacy PDB. The path map values are the cached file paths.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for pdb_id in pdb_ids:
        cif = cache_dir / f"{pdb_id.upper()}.cif"
        pdb = cache_dir / f"{pdb_id.upper()}.pdb"
        if cif.exists() and cif.stat().st_size > 0:
            paths[pdb_id.upper()] = cif
        elif pdb.exists() and pdb.stat().st_size > 0:
            paths[pdb_id.upper()] = pdb
        else:
            missing.append(pdb_id.upper())

    if not missing:
        return paths

    logger.info("Fetching %d structures into %s", len(missing), cache_dir)

    def fetch_one(pdb_id: str) -> tuple[str, Path | None]:
        try:
            time.sleep(0.05)  # gentle rate limit
            result = rcsb.fetch(pdb_id, "cif", str(cache_dir))
            return pdb_id, Path(result if isinstance(result, str) else result[0])
        except Exception as cif_exc:  # noqa: BLE001
            try:
                result = rcsb.fetch(pdb_id, "pdb", str(cache_dir))
                return pdb_id, Path(result if isinstance(result, str) else result[0])
            except Exception as pdb_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to fetch %s (cif: %s; pdb: %s)", pdb_id, cif_exc, pdb_exc
                )
                return pdb_id, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for future in as_completed(pool.submit(fetch_one, p) for p in missing):
            pdb_id, path = future.result()
            if path is not None:
                paths[pdb_id] = path

    return paths
