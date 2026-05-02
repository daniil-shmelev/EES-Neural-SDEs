# Sphere Latent NSDE Activity Classification

JAX/Equinox reimplementation of the HumanActivity experiment from
`/home/luke/EES-LatentSDEonHS/activity_classification.py`.

The latent state lives on `S^{z_dim-1}` and is integrated through a georax
`Manifold` implementation in `geometry.py`. Its frame coordinates match the
original PyTorch lower-triangular `so(n)` convention. Training uses cheap
Cayley/Taylor charts selected by the georax solver; the parity script uses the
exact exponential chart to reproduce the baseline homogeneous-space action.

## Data

The cyreal dataset reads the PersonActivity data from:

```bash
/home/luke/EES-LatentSDEonHS/data_dir
```

It first tries the processed Torch `data.pt` if `torch` is importable. In the
JAX-only environment it rebuilds the same arrays from the raw text file. The
parity script confirms that this raw NumPy preprocessing is exactly equal to
the original processed artifact. Split strategy defaults to `auto`: exact Torch
`random_split` when `torch` is importable, otherwise a deterministic NumPy split
with the same split sizes.

## Checks

Compare behavior against the original Torch repo:

```bash
uv run python experiments/sphere_latent_nsde/scripts/check_original_parity.py
```

Run a small end-to-end smoke train:

```bash
uv run python -m experiments.sphere_latent_nsde.train_activity \
  --smoke \
  --data-source raw \
  --solver geometric_euler \
  --output-dir /tmp/activity_jax_smoke
```

For the intended solver comparison, set `--solver cfees25 --adjoint reversible`
or keep `--solver geometric_euler`.

The exact original output grid has 228 points. For faster JAX development runs,
use `--num-timepoints 64` or `--num-timepoints 32`; the dataloader remaps
reconstruction/classification time IDs onto that grid. Omit this flag for the
full original grid.


```bash
uv run python -m experiments.sphere_latent_nsde.train_activity \
  --data-source raw \
  --solver geometric_euler \
  --adjoint direct \
  --epochs 100 \
  --batch-size 64 \
  --num-timepoints 64 \
  --output-dir experiments/sphere_latent_nsde/results/geometric_euler_activity
```

```bash
uv run python -m experiments.sphere_latent_nsde.train_activity \
  --data-source raw \
  --solver cfees25 \
  --adjoint reversible \
  --epochs 100 \
  --batch-size 64 \
  --num-timepoints 64 \
  --output-dir experiments/sphere_latent_nsde/results/cfees25_activity
```
