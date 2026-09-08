# Launching the cluster runs (Imperial RCS HPC)

Scripts to run the two compute-limited experiments on the Imperial RCS HPC
(PBS scheduler, `gpu72` queue). These were previously run on a single 16 GB
RTX 5080; the cluster lets us scale them to the regime that actually answers
the reviewers.

| Experiment | Why scale it | What changes vs. the workstation run |
|---|---|---|
| **Kuramoto quality @ N=1000** | Memory claim is at N=1000, but quality was only shown at N≤8 (and CF-EES trailed CG2 with ±15.55 variance). | Train the quality comparison at **N=1000**, **5 seeds** (was ≤3), 30 epochs, NFE-matched. |
| **Sphere/UCI HumanActivity** | Paper ran 30 epochs / 64 timepoints / 2 seeds → 88.30% < Zeng's published 90.56%. | **Native 228-grid**, **~990 epochs**, **5 seeds**, 3 georax methods. |

## Compute capacity needed

The cluster's GPU queue (`gpu72`): 1 node, ≤64 cores, ≤920 GB RAM, **≤72 h walltime**,
**max 12 GPUs per user** at once. GPU types: **L40S (48 GB, default)**, A100 (40 GB, scarce),
RTX6000 (24 GB). All sizings below are **1 GPU per job**.

| Job | PBS resource line | Walltime | Count |
|---|---|---|---|
| Kuramoto data-gen (once) | `select=1:ncpus=8:mem=96gb:ngpus=1:gpu_type=L40S` | ~2–4 h | 1 |
| Kuramoto training | `select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S` | ~6–12 h / task | array of 5 |
| Sphere data download (once) | login node, no GPU | minutes | 1 |
| Sphere training | `select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S` | ~12–36 h / task | array of 15 |

Notes on sizing:
- **Memory is not the binding constraint for training** at these grids (CG2-Full
  at N=1000, 50 steps is < 1 GB; sphere at the 228 grid is < 1 GB). L40S is
  recommended for **headroom and availability**, and because 48 GB also lets you
  later re-measure the memory-sweep cells that OOM'd on the 16 GB card
  (CG2-Full Kuramoto @ 10k steps; Geo-Euler sphere @ 5k steps ≈ 19.7 GB).
  **RTX6000 (24 GB) is sufficient for the training itself** and usually queues
  faster — change `gpu_type=RTX6000` in the `#PBS` line if the L40S queue is long.
- The **12-GPU cap throttles the arrays automatically**: the 15 sphere tasks run
  ~12 at a time. No action needed.
- The expensive one is the full sphere run (228 steps × 990 epochs). If wall-clock
  is tight, start cheaper with `qsub -v EPOCHS=500 hpc/run_sphere_activity.pbs`
  or drop to 3 seeds (edit `seed = [...]` in `configs/activity/full.toml`).

## One-time setup (login node)

```bash
cd ~/EES-Neural-SDEs          # wherever you cloned it
bash hpc/setup_env.sh          # builds conda env 'ees' + installs deps
bash hpc/prep_data.sh          # downloads the UCI HumanActivity file
```

## Run it (submit from the repo root)

```bash
# --- Kuramoto N=1000 ---
qsub hpc/gen_kuramoto_data.pbs                 # 1) generate the N=1000 dataset
qsub -W depend=afterok:<JOBID> hpc/run_kuramoto_n1000.pbs   # 2) train (use the data-gen job id)
# or just submit step 2 after step 1 finishes.

# --- Sphere / UCI HumanActivity ---
qsub hpc/run_sphere_activity.pbs
```

`qsub` prints a job id (e.g. `403036[].pbs-7` for an array). Monitor / cancel:

```bash
qstat -u "$USER"          # S column: Q=queued, R=running
qstat -t <JOBID>          # per-array-task status
qdel <JOBID>
```

Stdout+stderr land next to the script as `<name>.o<id>` (merged via `#PBS -j oe`).

## Knobs (override without editing files)

```bash
qsub -v EPOCHS=50 hpc/run_kuramoto_n1000.pbs       # more/fewer epochs
qsub -v EPOCHS=500 hpc/run_sphere_activity.pbs      # cheaper sphere pass
qsub -v ENV_NAME=myenv hpc/run_kuramoto_n1000.pbs   # different conda env
```
GPU type, walltime, and the seed/method grid (sphere) live in the `#PBS` headers
and `configs/activity/full.toml` respectively.

## Where results land

- Kuramoto: `experiments/kuramoto/results/runtime_parity_N1000_quality/<method>_seed<N>/{metrics,history,config}.json`
  plus `runtime_parity_summary.json`.
- Sphere: `experiments/sphere_latent_nsde/results/activity__<solver>_<adjoint>__seed<N>__<timestamp>/{metrics,history,config}.json` + `nsde.eqx`.

## Caveats

- Only the **three georax methods** (Geo-Euler, CG2, CF-EES(2,5)) are wired up
  here; the paper's SRKMK-ShARK sphere row uses a separate solver path not in
  this JAX harness.
- The committed paper *memory* figures come from the PyTorch submodule
  (`experiments/sphere_latent_sde`, extra `[sphere]`); this setup drives the
  JAX/georax **accuracy/quality** runs, which is what the two scaled experiments
  need.
- Compute nodes may lack internet — that's why `setup_env.sh` and `prep_data.sh`
  run on the **login node**. If a training job can't import a git-pinned dep,
  the env wasn't built on the login node first.
