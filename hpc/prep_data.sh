#!/bin/bash
# Data preparation (run on a LOGIN node -- needs internet).
#
#   bash hpc/prep_data.sh
#
# - Sphere/UCI: downloads the raw HumanActivity file. The JAX loader rebuilds
#   its arrays from this raw file, so torch / the PyTorch submodule are not
#   required.
# - Kuramoto N=1000: the *training set* is large and GPU-generated, so it is
#   NOT done here. Submit hpc/gen_kuramoto_data.pbs instead (a GPU job).
set -euo pipefail

SPHERE_RAW="experiments/sphere_latent_sde/data_dir/PersonActivity/raw"
UCI_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt"

mkdir -p "${SPHERE_RAW}"
if [ ! -s "${SPHERE_RAW}/ConfLongDemo_JSI.txt" ]; then
    echo "[prep] downloading UCI HumanActivity -> ${SPHERE_RAW}/ConfLongDemo_JSI.txt"
    curl -fL --retry 3 -o "${SPHERE_RAW}/ConfLongDemo_JSI.txt" "${UCI_URL}"
else
    echo "[prep] UCI HumanActivity already present, skipping"
fi

echo "[prep] sphere data ready."
echo "[prep] NOTE: for Kuramoto, submit the GPU data-gen job next:"
echo "       qsub hpc/gen_kuramoto_data.pbs"
