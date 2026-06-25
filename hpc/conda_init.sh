#!/bin/bash
# Sourceable helper: make `conda` available on the Imperial RCS HPC.
#
# On RCS there is no system conda on PATH. `module load miniforge/3` provides a
# one-time `miniforge-setup` helper that installs miniforge into your HOME; conda
# then lives at ~/miniforge3/bin/conda. This script loads the module, bootstraps
# miniforge on first use, and initialises conda for the current (possibly
# non-interactive) shell so `conda activate` works.
#
# Docs: https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/applications/guides/conda/

module load miniforge/3 2>/dev/null || true

# One-time bootstrap: install miniforge into $HOME if it isn't there yet.
if [ ! -x "$HOME/miniforge3/bin/conda" ] && command -v miniforge-setup >/dev/null 2>&1; then
    echo "[conda_init] first-time miniforge-setup (installs ~/miniforge3, takes a few minutes)..."
    miniforge-setup
fi

if [ -x "$HOME/miniforge3/bin/conda" ]; then
    eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "[conda_init] ERROR: conda unavailable. Run once: module load miniforge/3 && miniforge-setup" >&2
    return 1 2>/dev/null || exit 1
fi
