#!/bin/bash
# M5 pipeline: generate N=1000 training data, then run the hero training matrix.
# Sequential, one GPU job at a time, all logs persisted.
#
# Usage: bash run_m5_pipeline.sh
#
# Args (envvars with defaults):
#   N                 = 1000
#   N_TRAIN, N_VAL, N_TEST = 2000, 500, 500
#   N_FINE            = 4096   (less than the small-N case for tractability)
#   N_OBS             = 100
#   T                 = 5.0
#   BATCH_SIM         = 32     (data-gen batch; tune to GPU memory)
#   BATCH_TRAIN       = 32     (training batch)
#   N_STEPS           = 1000   (training integrator steps; tune from M4 results)
#   EPOCHS            = 20
#   SEEDS             = "0 1 2"
#   LR                = 1e-3

set -euo pipefail

cd "$(dirname "$0")/../../.."

N=${N:-1000}
N_TRAIN=${N_TRAIN:-1500}    # reduced from 5000 for tractability at N=1000
N_VAL=${N_VAL:-300}
N_TEST=${N_TEST:-300}
N_FINE=${N_FINE:-2048}      # halved from small-N case; Heun converges at ~1024
N_OBS=${N_OBS:-100}
T=${T:-5.0}
BATCH_SIM=${BATCH_SIM:-32}
BATCH_TRAIN=${BATCH_TRAIN:-32}
N_STEPS=${N_STEPS:-500}     # M5 training integrator steps; bump if M4 says we have headroom
EPOCHS=${EPOCHS:-10}        # short; the long pole is per-step compute, not per-epoch count
SEEDS=${SEEDS:-"0 1 2"}
LR=${LR:-1e-3}

LOG_DIR=experiments/kuramoto/results/m5_logs
mkdir -p "$LOG_DIR"

DATA_NPZ=experiments/kuramoto/data/kuramoto_N${N}_seed0.npz

echo "[m5] === step 1/2: data gen (N=${N}, ${N_TRAIN}/${N_VAL}/${N_TEST}, n_fine=${N_FINE}) ==="
if [ -f "$DATA_NPZ" ]; then
    echo "[m5] data already present at $DATA_NPZ; skipping gen"
else
    .venv/bin/python -m experiments.kuramoto.scripts.run_m1 \
        --skip-verify --no-gif \
        --n-list ${N} \
        --n-train ${N_TRAIN} --n-val ${N_VAL} --n-test ${N_TEST} \
        --T ${T} --n-fine ${N_FINE} --n-obs ${N_OBS} \
        --batch-sim ${BATCH_SIM} \
        2>&1 | tee "${LOG_DIR}/datagen.log"
fi

NFE_BUDGET=${NFE_BUDGET:-1500}

echo "[m5] === step 2/3: runtime-parity check at M5 config (NFE_BUDGET=${NFE_BUDGET}) ==="
.venv/bin/python -m experiments.kuramoto.scripts.runtime_parity_check \
    --N ${N} --batch-size ${BATCH_TRAIN} --hidden-dim 128 \
    --T ${T} --nfe-budget ${NFE_BUDGET} \
    2>&1 | tee "${LOG_DIR}/parity.log"

echo "[m5] === step 3/4: hero training (NFE-parity, NFE_BUDGET=${NFE_BUDGET}, epochs=${EPOCHS}) ==="
.venv/bin/python -m experiments.kuramoto.scripts.run_hero_training \
    --config experiments/kuramoto/configs/kuramoto.toml \
    --N ${N} \
    --nfe-budget ${NFE_BUDGET} \
    --batch-size ${BATCH_TRAIN} \
    --epochs ${EPOCHS} \
    --seeds ${SEEDS} \
    --lr ${LR} \
    2>&1 | tee "${LOG_DIR}/hero.log" || true   # don't abort the pipeline if a cell crashes

echo "[m5] === step 4/4: M6 plot + aggregate ==="
# Render the M4 memory-scaling figure into the manuscript's figures/ dir.
.venv/bin/python -m experiments.kuramoto.scripts.plot_memory_sweep \
    --input experiments/kuramoto/results/memory_sweep_N${N}.json \
    2>&1 | tee "${LOG_DIR}/plot_memory.log" || true

# Render a per-variant predictions panel from the best-by-val model.
for run_dir in experiments/kuramoto/results/hero_N${N}/*_seed0; do
    [ -d "$run_dir" ] || continue
    label=$(basename "$run_dir")
    [ -f "$run_dir/predictions_demo.npz" ] || { echo "[m6] skip $label (no demo file)"; continue; }
    out=experiments/kuramoto/results/hero_N${N}/predictions_${label}.pdf
    .venv/bin/python -m experiments.kuramoto.scripts.plot_predictions \
        --run-dir "$run_dir" --output "$out" \
        2>&1 | tee -a "${LOG_DIR}/plot_predictions.log" || true
done

# Aggregate everything into one JSON for manuscript fill-in.
.venv/bin/python -m experiments.kuramoto.scripts.aggregate_results \
    2>&1 | tee "${LOG_DIR}/aggregate.log" || true

# Surgically substitute the manuscript placeholders.
.venv/bin/python -m experiments.kuramoto.scripts.fill_manuscript \
    --aggregate experiments/kuramoto/results/aggregate.json \
    2>&1 | tee "${LOG_DIR}/fill_manuscript.log" || true

# Final: rebuild the PDF so the user comes back to fresh artifacts.
MANUSCRIPT_DIR=/mnt/c/Users/Shmelev/source/overleaf/EES_Neural_SDEs_Overleaf/neurips2026
if command -v latexmk >/dev/null 2>&1 && [ -d "$MANUSCRIPT_DIR" ]; then
    (cd "$MANUSCRIPT_DIR" && latexmk -pdf -g -interaction=nonstopmode \
        ees_neurips_manuscript.tex >> "${LOG_DIR}/latex.log" 2>&1) || true
    echo "[m6] manuscript rebuilt at $MANUSCRIPT_DIR/ees_neurips_manuscript.pdf"
fi

echo "[m5] === pipeline complete ==="
echo "[m5] artifacts:"
echo "[m5]   M4 results: experiments/kuramoto/results/memory_sweep_N${N}.json"
echo "[m5]   M4 figure : (overleaf)/figures/fig_kuramoto_memory_scaling.pdf"
echo "[m5]   M5 hero   : experiments/kuramoto/results/hero_N${N}/"
echo "[m5]   aggregate : experiments/kuramoto/results/aggregate.json"
echo "[m5]   logs      : ${LOG_DIR}/"
