"""Compare key JAX reimplementation pieces with the original Torch repo.

This is intentionally not a full training parity test. It checks the behavioral
contracts that should be exact or near-exact across implementations:

* PersonActivity preprocessing arrays.
* Chebyshev drift evaluation.
* Sphere ``so(n)`` action convention.
* Per-timepoint auxiliary cross entropy and accuracy.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.sphere_latent_nsde.dataset import load_activity_arrays
from experiments.sphere_latent_nsde.geometry import Sphere, SphereExpChart
from experiments.sphere_latent_nsde.losses import _cross_entropy_per_time, gather_time
from experiments.sphere_latent_nsde.models import Chebyshev


ORIGINAL_REPO = Path("experiments/sphere_latent_sde").resolve()
ORIGINAL_PYTHON = Path(sys.executable)
DATA_DIR = ORIGINAL_REPO / "data_dir"


def _default_original_python(repo: Path) -> Path:
    candidates = [
        repo / ".venv" / "bin" / "python",
        repo / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _run_original(code: str, out_file: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ORIGINAL_REPO)
    subprocess.run(
        [str(ORIGINAL_PYTHON), "-c", code, str(out_file)],
        check=True,
        cwd=str(ORIGINAL_REPO),
        env=env,
    )


def _assert_close(name: str, actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-6) -> None:
    max_abs = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if not np.allclose(actual, expected, atol=atol, rtol=atol):
        raise AssertionError(f"{name} mismatch: max_abs={max_abs:.3e}, atol={atol}")
    print(f"{name}: ok max_abs={max_abs:.3e}")


def check_dataset(tmp: Path) -> None:
    out = tmp / "activity_original.npz"
    _run_original(
        r'''
import sys
import numpy as np
import torch
data = torch.load("data_dir/PersonActivity/processed/data.pt", map_location="cpu")[:8000]
obs = torch.stack([r[2] for r in data]).numpy().astype(np.float32)
mask = torch.stack([r[3] for r in data]).numpy().astype(np.float32)
tid = torch.stack([r[1] for r in data]).long().numpy().astype(np.int32)
labels = torch.stack([r[4] for r in data])[:, :, :7].argmax(dim=2).long().numpy().astype(np.int32)
np.savez(sys.argv[1], obs=obs, mask=mask, tid=tid, labels=labels)
''',
        out,
    )
    original = np.load(out)
    ours = load_activity_arrays(DATA_DIR, source="raw")
    for key in ("obs", "mask", "tid", "labels"):
        print(f"dataset {key}: original={original[key].shape} ours={ours[key].shape}")
        np.testing.assert_equal(ours[key].shape, original[key].shape)
        if key in ("tid", "labels"):
            np.testing.assert_array_equal(ours[key], original[key])
            print(f"dataset {key}: ok exact")
        else:
            _assert_close(f"dataset {key}", ours[key], original[key], atol=1e-6)


def check_chebyshev(tmp: Path) -> None:
    out = tmp / "chebyshev_original.npz"
    _run_original(
        r'''
import sys
import numpy as np
import torch
from core.models import Chebyshev
torch.set_default_dtype(torch.float32)
mod = Chebyshev(5, 4, 6, time_min=0.0, time_max=1.98)
with torch.no_grad():
    mod.map.weight.copy_(torch.linspace(-0.3, 0.4, mod.map.weight.numel()).view_as(mod.map.weight))
    mod.map.bias.copy_(torch.linspace(-0.2, 0.2, mod.map.bias.numel()))
h = torch.linspace(-1.0, 1.0, 15).view(3, 5)
t = torch.linspace(0.0, 0.99, 7)
out = mod(h, t)
np.savez(
    sys.argv[1],
    weight=mod.map.weight.detach().numpy(),
    bias=mod.map.bias.detach().numpy(),
    h=h.numpy(),
    t=t.numpy(),
    out=out.detach().numpy(),
)
''',
        out,
    )
    original = np.load(out)
    cheb = Chebyshev(5, 4, 6, time_min=0.0, time_max=1.98, key=jax.random.key(0))
    cheb = eqx.tree_at(lambda m: m.linear.weight, cheb, jnp.asarray(original["weight"]))
    cheb = eqx.tree_at(lambda m: m.linear.bias, cheb, jnp.asarray(original["bias"]))
    ours = jax.vmap(lambda h: cheb(h, jnp.asarray(original["t"])))(jnp.asarray(original["h"]))
    _assert_close("chebyshev", np.asarray(ours), original["out"], atol=1e-3)


def check_sphere_action(tmp: Path) -> None:
    out = tmp / "sphere_original.npz"
    _run_original(
        r'''
import sys
import numpy as np
import torch
from utils.misc import vec_to_matrix
torch.manual_seed(0)
n = 5
group_dim = n * (n - 1) // 2
idx = torch.tril_indices(row=n, col=n, offset=-1)
basis = torch.zeros(group_dim, n, n)
basis[:, idx[0], idx[1]] = torch.eye(group_dim)
basis = basis - basis.permute(0, 2, 1)
z = torch.randn(n)
z = z / z.norm()
a = torch.linspace(-0.2, 0.3, group_dim)
omega = vec_to_matrix(a, basis)
y = torch.matrix_exp(omega) @ z
np.savez(sys.argv[1], z=z.numpy(), a=a.numpy(), y=y.numpy(), basis=basis.numpy())
''',
        out,
    )
    original = np.load(out)
    sphere = Sphere(int(original["z"].shape[0]), chart=SphereExpChart())
    ours = sphere.apply_increment(jnp.asarray(original["z"]), jnp.asarray(original["a"]))
    _assert_close("sphere action", np.asarray(ours), original["y"], atol=2e-6)
    _assert_close("sphere basis", np.asarray(sphere.basis), original["basis"], atol=0.0)


def check_aux_loss(tmp: Path) -> None:
    out = tmp / "aux_original.npz"
    _run_original(
        r'''
import sys
import numpy as np
import torch
from core.models import PerTimePointCrossEntropyLoss
torch.manual_seed(0)
logits = torch.randn(2, 3, 10, 4)
target = torch.tensor([[0, 1, 2, 3, 1], [3, 2, 1, 0, 0], [1, 1, 2, 2, 3]])
tids = torch.tensor([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9], [0, 1, 8, 8, 9]])
loss = PerTimePointCrossEntropyLoss(reduction="none")(logits, target, tids)
idx = tids.view(1, 3, 5, 1).repeat(2, 1, 1, 4).long()
acc = (logits.gather(2, idx).mean(dim=0).flatten(0, 1).argmax(1) == target.flatten()).float().mean()
np.savez(sys.argv[1], logits=logits.numpy(), target=target.numpy(), tids=tids.numpy(), loss=loss.detach().numpy(), acc=np.asarray(acc.item(), dtype=np.float32))
''',
        out,
    )
    original = np.load(out)
    logits = jnp.asarray(original["logits"])
    target = jnp.asarray(original["target"])
    tids = jnp.asarray(original["tids"])
    gathered = gather_time(logits, tids)
    mean_logits = jnp.mean(gathered, axis=0)
    ce = _cross_entropy_per_time(mean_logits, target)
    loss = jnp.mean(ce, axis=1, keepdims=True)
    acc = jnp.mean((jnp.argmax(mean_logits, axis=-1) == target).astype(jnp.float32))
    _assert_close("aux loss", np.asarray(loss), original["loss"], atol=2e-6)
    _assert_close("aux accuracy", np.asarray(acc), original["acc"], atol=2e-6)


def main() -> int:
    global ORIGINAL_REPO, ORIGINAL_PYTHON, DATA_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-repo", type=Path, default=ORIGINAL_REPO)
    parser.add_argument("--original-python", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--skip-dataset", action="store_true")
    args = parser.parse_args()

    ORIGINAL_REPO = args.original_repo.resolve()
    ORIGINAL_PYTHON = (
        args.original_python.resolve()
        if args.original_python is not None
        else _default_original_python(ORIGINAL_REPO)
    )
    DATA_DIR = (
        args.data_dir.resolve()
        if args.data_dir is not None
        else ORIGINAL_REPO / "data_dir"
    )

    if not ORIGINAL_PYTHON.exists():
        raise FileNotFoundError(f"Original repo Python not found: {ORIGINAL_PYTHON}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        if not args.skip_dataset:
            check_dataset(tmp)
        check_chebyshev(tmp)
        check_sphere_action(tmp)
        check_aux_loss(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
