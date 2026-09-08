# SIAM manuscript verification

This directory contains the exact ordered-forest verifier and the archived learning measurements supporting the SIAM manuscript. The paper text and all tables remain in its master `SIAM/ees_simods.tex`.

From this directory, run:

```sh
python scripts/verify_cfees.py --manuscript /path/to/SIAM/ees_simods.tex
python scripts/summarize_records.py --manuscript /path/to/SIAM/ees_simods.tex
```

The first command requires SymPy and checks the three-stage family over `Q(x)`, the selected four-stage method over `Q(sqrt(2))`, all 24 printed tree coefficients, and the embedded companions. The second uses the Python standard library, checks both learning tables against the records, and prints summary statistics. Add `--update` to the second command to update only those two tabular environments in the master.

`experiment_records/` contains the configurations and measurements used in the paper. Its `sources.json` records original source paths and checksums; figure destination names in that manifest are relative to the manuscript's SIAM directory. `verification.txt` records the exact checks, and `validation_environment.json` records software versions.

The ODE experiment remains in `experiments/convergence_ode`, using Diffrax/georax and the shared `experiments.convergence_plotting` helper. The manuscript is built from its project root with `latexmk SIAM/ees_simods.tex`; the root `.latexmkrc` supplies the class, bibliography, and macro search paths. Overleaf should use `SIAM/ees_simods.tex` as its main document.

Before submission, confirm the action and software revision used by the saved sphere training runs, the hardware/software manifests if a controlled runtime comparison is intended, and the invocation settings for the existing rough-driver figure. The paper states the limitations of the archived metadata. The Poincaré/energy experiment remains deferred.

This directory was moved from the local `experiments/order_verification/siam` tree during repository unification. Archived paths in `sources.json` and environment records are retained verbatim as provenance. The manuscript is an external input; it is not bundled or changed by repository validation.

Both scripts also run without `--manuscript`: the verifier then checks the algebraic identities and companions, and the record script validates and summarises archived measurements. Supplying `--manuscript` additionally checks the expected table labels and contents; a later manuscript layout may require updating the table parser.
