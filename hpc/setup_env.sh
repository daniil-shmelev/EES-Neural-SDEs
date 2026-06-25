#!/bin/bash
# One-time environment build for the Imperial RCS HPC (run on a LOGIN node).
#
#   bash hpc/setup_env.sh
#
# Login nodes have internet access, which is needed for the git-pinned
# dependencies (diffrax/georax/cyreal) and the JAX CUDA wheels. Compute nodes
# may not, so always build the env here first.
#
# Docs: https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/applications/guides/conda/
set -euo pipefail

ENV_NAME="${ENV_NAME:-ees}"
PY_VERSION="${PY_VERSION:-3.13}"

echo "[setup] loading miniforge"
module load miniforge/3
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup] creating conda env '${ENV_NAME}' (python ${PY_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi

conda activate "${ENV_NAME}"

# Install the two experiments we launch on the cluster. jax[cuda12] (Linux)
# pulls self-contained CUDA wheels, so no `module load CUDA` is required.
# Plain pip handles the git+https direct references in pyproject.toml.
echo "[setup] installing ees-neural-sdes[kuramoto,sphere-jax]"
python -m pip install --upgrade pip
python -m pip install -e ".[kuramoto,sphere-jax]"

echo "[setup] verifying JAX sees CUDA wheels (CPU on login node is expected)"
python - <<'PY'
import jax, jaxlib
print("jax", jax.__version__, "jaxlib", jaxlib.__version__)
print("backend on login node:", jax.default_backend(), "(will be 'gpu' inside a GPU job)")
PY

echo "[setup] done. Env '${ENV_NAME}' is ready."
