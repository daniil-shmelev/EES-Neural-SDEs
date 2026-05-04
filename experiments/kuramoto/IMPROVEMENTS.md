# Kuramoto NSDE — Model Improvement Hand-off

You are picking up a research project on neural SDEs on Lie groups. The
project as a whole already works and has shipped its main result. Your
job is to improve a specific subset: the **forecast quality** of the
trained models, **without touching the integrator-memory or runtime
claims**. Read this document end-to-end before doing anything.

---

## 1. The bigger project (orientation)

The repo at `~/repos/EES-Neural-SDEs/` contains the code for a paper on
**Explicit and Effectively Symmetric (EES) numerical schemes for neural
SDEs on Lie groups**. The headline contribution is `CFEES(2,5)`: a
commutator-free, near-reversible Runge–Kutta scheme that runs neural
SDEs on homogeneous spaces with $\mathcal{O}(1)$-memory adjoint
backpropagation.

The paper has several experiments. The one you are working on is at:

```
experiments/kuramoto/
```

This is the **Kuramoto-network manifold-SDE example** that demonstrates
the integrator's memory advantage in a power-grid–like regime. You do
**not** need to touch any other experiment.

### 1.1 Stack

- **Python 3.13** in a venv at `.venv/` (use `.venv/bin/python`)
- **JAX 0.10** with CUDA 13 (works on a GPU; CPU fallback is fine for
  smoke tests)
- **diffrax** (sammccallum fork, ~0.7.1) — SDE/ODE solvers + adjoints
- **georax** (`luke-a-thompson/georax`) — Lie-group integrators
  (`CFEES25`, `CFEES27`, `CG2`, `CG4`, `RKMK`) and the `LieGroup` /
  `GeometricTerm` abstractions
- **equinox** — model framework (`eqx.Module`, `eqx.filter_jit`,
  `eqx.filter_grad`)
- **optax** — optimisers
- **cyreal** — array-based dataloaders (used by training)

If imports break, run:

```bash
.venv/bin/python -m pip install --no-deps \
    "diffrax @ git+https://github.com/sammccallum/diffrax.git@aeb1335b5a6278e85a270231e0f97b8db4453ae6" \
    "georax @ git+https://github.com/luke-a-thompson/georax.git@30ecbb9bb45806e0bf6ecf24bcf6da1de9173aea" \
    "cyreal @ git+https://github.com/luke-a-thompson/cyreal_dynamics.git@88f02657989fc4f9c85df0afb1452085fc3f2f8b" \
    "diffrax-lowstorage @ git+https://github.com/luke-a-thompson/diffrax-lowstorage.git@74fd4dba492d96dcfd6148ad7d2f71f2a72f77ee"
.venv/bin/python -m pip install "jax[cuda13]" matplotlib  # GPU + plotting
```

The project's `pyproject.toml` extras include `[jax-base]` which pulls
all of these, but it has a transitively-broken `jax[cuda]` constraint
on Linux that fails resolver. The `--no-deps` install above is the
known-working workaround.

### 1.2 Manuscript location

```
/mnt/c/Users/Shmelev/source/overleaf/EES_Neural_SDEs_Overleaf/
├── neurips2026/
│   ├── ees_neurips_manuscript.tex   # main file
│   ├── neurips_references.bib       # bib
│   └── latexmkrc                    # build config
└── figures/                         # figure pdfs go here
```

Compile from inside `neurips2026/`:

```bash
cd /mnt/c/Users/Shmelev/source/overleaf/EES_Neural_SDEs_Overleaf/neurips2026
latexmk -pdf -g -interaction=nonstopmode ees_neurips_manuscript.tex
```

The output PDF is `ees_neurips_manuscript.pdf` in that directory.

### 1.3 The Kuramoto experiment

We integrate a **stochastic 2nd-order Kuramoto network** matching
Olmi & Torcini 2024 eq.(1) with their higher-order coupling switched
off ($K_2 = 0$):

$$m\,\ddot{\theta}_i = -\dot{\theta}_i + \Omega_i + \frac{K}{N}\sum_j \sin(\theta_j - \theta_i) + \xi_i(t),$$

with $\langle \xi_i(t)\xi_j(s)\rangle = 2D\,\delta_{ij}\delta(t-s)$ and
**bimodal** natural frequencies $\Omega_i \in \{+P, -P\}$ (Filatrella
generator/consumer convention). Default parameters: $K=2$, $P=0.5$,
$D=0.05$, $m=1$, $T=5$ s. The state lives on the cotangent bundle
$T\mathbb{T}^N = \mathbb{T}^N \times \mathbb{R}^N$ as the product Lie
group.

We train a neural SDE on $T\mathbb{T}^N$ to forecast trajectories. The
training loss is a **multi-horizon wrapped energy score** at horizons
$\{T/8, T/4, T/2, T\}$ with 4 Monte Carlo rollouts per horizon. The
energy score is strictly proper but doesn't directly penalise
distributional moments — a key fact for one of the improvements below.

### 1.4 Files you'll touch

| Purpose | Path |
|---|---|
| Drift + diffusion fields (the architectures to replace) | `experiments/kuramoto/models/kuramoto_nsde.py` |
| Euclidean baseline (similar architecture for comparison) | `experiments/kuramoto/models/euclidean_baseline.py` |
| Multi-horizon energy score loss | `experiments/kuramoto/experiment/losses.py` |
| Training loop with full logging | `experiments/kuramoto/experiment/train_kuramoto.py` |
| TOML hyperparameter config | `experiments/kuramoto/configs/kuramoto.toml` |
| Hero training driver (matrix of 3 variants × 3 seeds) | `experiments/kuramoto/scripts/run_hero_training.py` |
| End-to-end pipeline (data + parity + train + plot + fill) | `experiments/kuramoto/scripts/run_m5_pipeline.sh` |
| Diagnostic plot (per-cell prediction quality) | `experiments/kuramoto/scripts/plot_predictions.py` |

### 1.5 Files you should NOT touch

| Path | Why |
|---|---|
| `experiments/kuramoto/datasets/{kuramoto,simulator,pipeline}.py` | Data-generation simulator — well-tested, used to produce the cached training data |
| `experiments/kuramoto/models/product_torus.py` | The `LieGroup` subclass for $T\mathbb{T}^N$ — correct as-is |
| `experiments/kuramoto/scripts/{run_memory_sweep,memory_sweep_single}.py` | M4 benchmark scripts — already produced the headline figure |
| `experiments/kuramoto/scripts/runtime_parity_check.py` | M5 pre-flight; already validated |
| `experiments/kuramoto/scripts/{aggregate_results,fill_manuscript,plot_memory_sweep}.py` | M6 plumbing |
| Any other experiment dir under `experiments/` | Unrelated |
| `pyproject.toml` | Don't change project deps |
| `experiments/kuramoto/data/*.npz` | Cached training data; do not regenerate unless you need to |

---

## 2. The specific problem you're solving

We trained the model with the integrator we're trying to sell. The
training works (loss descends, models converge). But inspecting the
trained model's predictions reveals two specific failure modes that
**reduce credibility of the secondary "no-quality-regression" claim**
in the paper. The headline integrator-memory and runtime claims are
unaffected.

### 2.1 Diagnostic numbers (from `cfees25_reversible_seed1`)

These are computed against the held-out test split using the
`predictions_demo.npz` saved by training. For comparison: the wrapped
angle metric is bounded above by $\pi/2 \approx 1.57$ for "random"
predictions.

| Horizon | NSDE wrapped MAE | predict-$\theta_0$ baseline | random baseline |
|---|---|---|---|
| 0.05 s | **0.020** | 0.019 | $\pi/2 \approx 1.57$ |
| 1.26 s | **0.354** | 0.389 | 1.57 |
| 2.53 s | **0.561** | 0.856 | 1.57 |
| 3.79 s | **0.683** | 1.426 | 1.57 |
| 5.00 s | **0.800** | 1.946 | 1.57 |

So the model has learned the **local kinematics** (short horizons are
near-perfect, MAE grows sub-linearly, beats predict-$\theta_0$ by 2.4×
at $T=5$s). What it has NOT learned:

1. **Slow synchronisation onset.** The Kuramoto order parameter
   $r(t) = \big| \tfrac{1}{N}\sum_j e^{i\theta_j} \big|$ in the data
   slowly grows from $\approx 0.03$ at $t=0$ to $\approx 0.07$ at
   $t=5$s — the partial-sync collective behaviour the model is supposed
   to capture. The NSDE's samples stay flat at $r \approx 0.02-0.03$
   throughout. The model entirely misses this.
2. **Omega over-diffusion.** Data $\sigma_\omega = 0.50$, NSDE samples
   $\sigma_\omega = 0.72$ — the diffusion field is producing too-noisy
   velocities, ~46% over.

You can reproduce these numbers by running the analysis script in
section 4.4 below.

### 2.2 Hypothesised root causes

(In order of importance — addressed by improvements 1, 2, 3 below.)

a) **No permutation-equivariance inductive bias.** The current
   `KuramotoDriftField` is a plain MLP that sees
   $(\sin\theta, \cos\theta, \omega) \in \mathbb{R}^{3N}$ as a flat
   3000-dimensional vector at $N=1000$. It has to learn the
   $(K/N)\sum_j \sin(\theta_j - \theta_i)$ coupling structure from raw
   features. With a width-128 MLP and 1500 training trajectories at
   that input dimension, this is hopeless.

b) **Over-large diffusion at init.** The diffusion field's softplus
   output is initialised with bias $-2$ and scaled by $0.3$, giving
   initial diffusion magnitude $\approx 0.3 \times \mathrm{softplus}(-2)
   \approx 0.04$ per Brownian dimension. At $N=1000$ this aggregates
   into a velocity-noise floor that's too high.

c) **Energy-score loss is moment-blind.** The strictly-proper energy
   score evaluates pairwise sample distances; it doesn't directly
   penalise the mean order parameter, the mean velocity, or any other
   moment. The slow $r(t)$ growth is therefore "invisible" to the
   gradient signal until very late in training.

---

## 3. The improvements

Apply in this order. Each improvement says exactly which file to touch
and what to change. After each one, run the smoke check in §4.

### Improvement 1 — Permutation-equivariant drift (BIGGEST WIN)

**File**: `experiments/kuramoto/models/kuramoto_nsde.py`

Replace the current `KuramotoDriftField` (a plain MLP that flattens the
$N$-oscillator state into a $3N$-dim vector) with a DeepSets /
message-passing architecture that is permutation-equivariant by
construction and matches Kuramoto's natural mean-field structure.

**Concretely**, the new drift, applied to per-oscillator state
$(\theta_i, \omega_i)$ given the full-state context $(\theta, \omega)$:

```python
class EquivariantDriftField(eqx.Module):
    phi_pair: eqx.nn.MLP   # per-pair edge features
    rho_node: eqx.nn.MLP   # per-node aggregation -> output
    geometry: ProductTorusEuclidean
    N: int = eqx.field(static=True)

    def __init__(self, geometry, hidden_dim, *, key, depth_pair=2, depth_node=2,
                 activation=jax.nn.silu):
        N = geometry.N
        kp, kn = jax.random.split(key, 2)
        # phi_pair input: (sin(dtheta), cos(dtheta), domega) ∈ R^3
        # phi_pair output: hidden_dim per-pair feature
        self.phi_pair = eqx.nn.MLP(
            in_size=3, out_size=hidden_dim,
            width_size=hidden_dim, depth=depth_pair,
            activation=activation, key=kp,
        )
        # rho_node input: (sin theta_i, cos theta_i, omega_i, mean-pooled phi)
        # rho_node output: 2 (per-node Lie-algebra increment: dot-theta, dot-omega)
        self.rho_node = eqx.nn.MLP(
            in_size=3 + hidden_dim, out_size=2,
            width_size=hidden_dim, depth=depth_node,
            activation=activation, key=kn,
        )
        # Match the existing zero-init-last-layer pattern of KuramotoDriftField:
        last = self.rho_node.layers[-1]
        last = eqx.tree_at(lambda l: l.bias, last, jnp.zeros_like(last.bias))
        last = eqx.tree_at(lambda l: l.weight, last, jnp.zeros_like(last.weight))
        self.rho_node = eqx.tree_at(lambda m: m.layers[-1], self.rho_node, last)
        self.geometry = geometry
        self.N = N

    def __call__(self, t, y, args):
        del t, args
        N = self.N
        theta = y[:N]      # (N,)
        omega = y[N:]      # (N,)
        # Pair features: phi(sin(theta_j - theta_i), cos(...), omega_j - omega_i)
        # vmap over both i and j.
        def pair_input(theta_i, omega_i, theta_j, omega_j):
            return jnp.stack([jnp.sin(theta_j - theta_i),
                              jnp.cos(theta_j - theta_i),
                              omega_j - omega_i])
        # (N, N, 3) full grid, then vmap phi_pair across the last axis
        ti = theta[:, None]; tj = theta[None, :]
        oi = omega[:, None]; oj = omega[None, :]
        pair_in = jnp.stack([jnp.sin(tj - ti), jnp.cos(tj - ti), oj - oi], axis=-1)  # (N, N, 3)
        # Apply phi_pair to each (i, j) entry; flatten leading dims for vmap.
        pair_feats = jax.vmap(jax.vmap(self.phi_pair))(pair_in)  # (N, N, hidden)
        # Mean-pool over j (the (1/N) sum):
        pooled = jnp.mean(pair_feats, axis=1)  # (N, hidden)
        # Per-node: feed (sin theta_i, cos theta_i, omega_i, pooled_i) to rho.
        node_in = jnp.concatenate([jnp.sin(theta)[:, None],
                                    jnp.cos(theta)[:, None],
                                    omega[:, None],
                                    pooled], axis=-1)  # (N, 3 + hidden)
        out = jax.vmap(self.rho_node)(node_in)  # (N, 2)
        dtheta = out[:, 0]
        domega = out[:, 1]
        return jnp.concatenate([dtheta, domega])
```

**Wire it in**: change `KuramotoNSDE.__init__` to use
`EquivariantDriftField` instead of `KuramotoDriftField`. Keep
`KuramotoDriftField` in the file for back-compat (existing checkpoints
need it for deserialisation if the user wants to load them).

**Make it opt-in via the config**: add a string field
`drift_kind: str = "equivariant"` to `ExperimentConfig` in
`experiments/kuramoto/experiment/config.py` (default to the new
architecture; legacy "mlp" runs the old one). Plumb it through
`make_model` in `experiments/kuramoto/experiment/factories.py`.

**Sanity check**: at $N=4$, the new drift's parameter count should be
*independent of $N$* (because `phi_pair` and `rho_node` are
applied per-pair / per-node). Verify with:

```python
model = KuramotoNSDE(N=4, hidden_dim=64, n_steps=10, dt=0.05, n_obs=10, key=jr.PRNGKey(0))
print(sum(int(x.size) for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))))
```

Both $N=4$ and $N=1000$ should print the same number. If they don't,
the architecture isn't truly equivariant and you need to find the
$N$-dependent layer.

### Improvement 2 — Smaller diffusion init + bound

**Files**:
- `experiments/kuramoto/configs/kuramoto.toml` — change
  `diffusion_scale = 0.3` to `diffusion_scale = 0.1`.
- `experiments/kuramoto/models/kuramoto_nsde.py` — in
  `KuramotoDiffusionField.__init__`, change the softplus bias from
  `jnp.full_like(last.bias, -2.0)` to `jnp.full_like(last.bias, -4.0)`.
- (Optional) Add `loss + 0.001 * jnp.mean(scales ** 2)` to the loss in
  `experiment/losses.py` to L2-penalise the diffusion's output.

These act on the diffusion field's effective initial scale. After
applying, re-run a single training cell at $N=2$ (10 epochs, 200 train
trajectories) and verify in `predictions_demo.npz` that the omega std
of the NSDE samples is now within ~10% of the data's omega std.

### Improvement 3 — Auxiliary $r(t)$ moment-matching loss

**File**: `experiments/kuramoto/experiment/losses.py`

Modify `make_multi_horizon_energy_score` to include an additional term
that matches the **mean order parameter $r(t)$** between the NSDE's
sample ensemble and the data trajectories at each horizon:

```python
def order_r(theta_btn):
    # theta_btn shape (B, T_obs, N) -> (B, T_obs)
    return jnp.abs(jnp.mean(jnp.exp(1j * theta_btn), axis=-1))

# Inside loss_fn, after sampling sample_theta of shape (S, B, T_obs, N):
r_sample = order_r(sample_theta.reshape(-1, *sample_theta.shape[-2:]))  # (S*B, T_obs)
r_target = order_r(target_theta)  # (B, T_obs)
# Mean over the S*B sample axis at each horizon:
r_pred_mean_at_h = jnp.mean(r_sample[:, idx], axis=0)  # (H,)
r_data_mean_at_h = jnp.mean(r_target[:, idx], axis=0)  # (H,)
aux = jnp.sum((r_pred_mean_at_h - r_data_mean_at_h) ** 2)
return main_es_loss + 0.1 * aux
```

The `0.1` weight is a starting point; tune in $\{0.01, 0.1, 1.0\}$ if
the trade-off looks wrong.

### Improvement 4 — Numerical-stability fixes

**File**: `experiments/kuramoto/experiment/train_kuramoto.py`

Two patches inside `fit()`. Both targeted at the sporadic
`grad_norm = inf` we observed in the existing M5 run.

**(a) Skip non-finite gradient steps.** Wrap the optimiser update in a
`jax.lax.cond` that no-ops when the gradient is non-finite:

```python
@eqx.filter_jit
def train_step(current_model, opt_state, batch, mask, key):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(current_model, batch, mask, key)
    flat_grads = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    grad_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in flat_grads))
    update_ok = jnp.isfinite(grad_norm)
    updates, new_opt_state = optim.update(grads, opt_state, current_model)
    # If non-finite, leave model + opt_state unchanged.
    new_model = jax.lax.cond(
        update_ok,
        lambda: eqx.apply_updates(current_model, updates),
        lambda: current_model,
    )
    new_opt_state = jax.lax.cond(
        update_ok, lambda: new_opt_state, lambda: opt_state,
    )
    return new_model, new_opt_state, loss, grad_norm
```

**(b) LR warmup.** Replace the bare `optax.adam(config.learning_rate)`
with a warmup-cosine schedule:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-5,
    peak_value=config.learning_rate,
    warmup_steps=200,
    decay_steps=config.epochs * train_loader.steps_per_epoch,
    end_value=1e-5,
)
optim = optax.chain(
    optax.clip_by_global_norm(config.grad_clip_norm),
    optax.adam(schedule),
)
```

### Improvement 5 — More compute (boring but works)

**File**: `experiments/kuramoto/configs/kuramoto.toml`

```toml
epochs = 50           # was 10
hidden_dim = 256      # was 128
```

Plus regenerate the training data with more trajectories:

```bash
.venv/bin/python -m experiments.kuramoto.scripts.run_m1 \
    --skip-verify --no-gif --n-list 1000 \
    --n-train 5000 --n-val 1000 --n-test 1000 \
    --T 5.0 --n-fine 2048 --n-obs 100 --batch-sim 32
```

This will overwrite `experiments/kuramoto/data/kuramoto_N1000_seed0.npz`
(takes ~1 hour on the GPU).

---

## 4. How to validate

### 4.1 Smoke test (5 min, after each change)

```bash
.venv/bin/python -m experiments.kuramoto.experiment.train_kuramoto \
    experiments/kuramoto/configs/kuramoto.toml \
    --output-dir /tmp/smoke_run \
    --N 4 --batch-size 8 --epochs 2 --n-steps 50
```

This trains a tiny model in ~1 min. Look for:
- No exceptions
- Loss is finite and decreasing
- `/tmp/smoke_run/history.json` exists and has `train_loss`, `step_grad_norm`, etc.
- `/tmp/smoke_run/predictions_demo.npz` exists

### 4.2 Single-cell quality check (~90 min, after improvements 1-3)

```bash
.venv/bin/python -m experiments.kuramoto.experiment.train_kuramoto \
    experiments/kuramoto/configs/kuramoto.toml \
    --output-dir experiments/kuramoto/results/improved_seed0 \
    --N 1000 --n-steps 500 --batch-size 32 --epochs 10 \
    --seed 0 --model kuramoto_nsde --solver cfees25 --adjoint reversible
```

Then run the diagnostic script (§4.4) on the resulting cell. Compare to
the baseline numbers in §2.1.

### 4.3 Full hero re-run (~9 hours)

Once you're confident the improvements work, re-run the full hero
matrix:

```bash
.venv/bin/python -m experiments.kuramoto.scripts.run_hero_training \
    --config experiments/kuramoto/configs/kuramoto.toml \
    --N 1000 \
    --nfe-budget 1500 \
    --batch-size 32 \
    --epochs 10 \
    --seeds 0 1 2 \
    --output-base experiments/kuramoto/results/hero_N1000_v2/
```

Then aggregate + plot:

```bash
.venv/bin/python -m experiments.kuramoto.scripts.aggregate_results \
    --output experiments/kuramoto/results/aggregate_v2.json
```

### 4.4 Diagnostic script (drop in to a Python REPL)

Run this against any trained cell's `predictions_demo.npz` to reproduce
the numbers in §2.1:

```python
import numpy as np
d = np.load("experiments/kuramoto/results/<cell-dir>/predictions_demo.npz")
target_th, target_om = d["target_theta"], d["target_omega"]
sample_th, sample_om = d["sample_theta"], d["sample_omega"]
t = d["t_grid"]
T_obs, N = target_th.shape[1], target_th.shape[2]

def wrap(x): return np.mod(x + np.pi, 2*np.pi) - np.pi
def order(theta): return np.abs(np.mean(np.exp(1j * theta), axis=-1))

mae = np.abs(wrap(sample_th - target_th[None])).mean(axis=(0, 1, 3))
mae_zero = np.abs(wrap(target_th[:, :1] - target_th)).mean(axis=(0, 2))
print("Wrapped MAE per horizon:")
for i in [1, T_obs//4, T_obs//2, 3*T_obs//4, T_obs-1]:
    print(f"  t={t[i]:5.2f}s: NSDE={mae[i]:.3f}  predict-θ0={mae_zero[i]:.3f}")

print("\nOrder parameter r(t) data vs NSDE:")
r_data = order(target_th).mean(axis=0)
r_samp = order(sample_th).mean(axis=(0,1))
for i in [0, T_obs//4, T_obs//2, T_obs-1]:
    print(f"  t={t[i]:5.2f}s: data={r_data[i]:.3f}  NSDE={r_samp[i]:.3f}")

print(f"\nOmega std: data={target_om.std():.3f}  NSDE={sample_om.std():.3f}")
```

### 4.5 Pass criteria

The improvements are working when:

1. **Synchronisation onset captured**: NSDE's mean $r(t)$ tracks the
   data's mean $r(t)$ to within $\pm 0.02$ at $t=5$s (currently:
   data $\approx 0.07$, NSDE $\approx 0.03$; gap $0.04$).
2. **Omega scale right**: NSDE samples' $\sigma_\omega$ is within
   $\pm 10\%$ of the data's (currently 46% over).
3. **Long-horizon MAE drops**: at $t=5$s, wrapped MAE goes from $0.80$
   to $\le 0.60$.
4. **Test ES drops**: from current $\approx 661.7 \pm 2.0$ at
   $N=1000$ to something noticeably lower.

If you hit (1) and (2), you're done. If only (3) and (4) move, you're
just compute-bound — improvement 5 will help.

---

## 5. What you must NOT change

These are sacred for the headline paper claims and any change risks
invalidating already-completed experiments:

- The data-generation simulator
  (`experiments/kuramoto/datasets/{kuramoto,simulator,pipeline}.py`).
- The `ProductTorusEuclidean` Lie group
  (`experiments/kuramoto/models/product_torus.py`).
- The CFEES integrator itself (which lives in the `georax` package, not
  in this repo).
- Anything outside `experiments/kuramoto/`.
- The pyproject extras / dependency pins.
- The M4 memory sweep results
  (`experiments/kuramoto/results/memory_sweep_N1000.json`) — those are
  the headline.
- The figure `figures/fig_kuramoto_memory_scaling.pdf` — generated
  from M4 data.

If you find yourself wanting to touch any of these, stop and ask the
human reviewer first.

---

## 6. After you're done

Once your improvements pass the §4.5 criteria:

1. Stage your code changes (`git add experiments/kuramoto/...`) but do
   **not** commit unless explicitly asked.
2. Write a short summary in `experiments/kuramoto/IMPROVEMENT_RESULTS.md`
   listing: which improvements you applied, the diagnostic numbers
   before/after (using the §4.4 script), and any caveats.
3. Re-run the manuscript fill + recompile so the table cells reflect
   the new numbers:

   ```bash
   .venv/bin/python -m experiments.kuramoto.scripts.aggregate_results \
       --output experiments/kuramoto/results/aggregate.json
   .venv/bin/python -m experiments.kuramoto.scripts.fill_manuscript \
       --aggregate experiments/kuramoto/results/aggregate.json
   cd /mnt/c/Users/Shmelev/source/overleaf/EES_Neural_SDEs_Overleaf/neurips2026
   latexmk -pdf -g -interaction=nonstopmode ees_neurips_manuscript.tex
   ```

That's it. Hand back to the human reviewer with the summary file.
