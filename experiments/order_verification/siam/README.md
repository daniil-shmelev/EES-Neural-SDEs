# SIAM manuscript verification

This directory contains the exact ordered-forest order verifier, built on [kauri](https://github.com/daniil-shmelev/kauri), and the archived learning measurements supporting the SIAM manuscript. The paper text and all tables remain in its master `SIAM/ees_simods.tex`.

From this directory, run:

```sh
python scripts/verify_cfees.py
python scripts/summarize_records.py --manuscript /path/to/SIAM/ees_simods.tex
```

The first command uses kauri's Owren pseudo-stage character and MKW symmetry defect to check the three-, four- and five-stage cosine schemes exactly over `Q`, `Q(sqrt(2))` and `Q(sqrt(5))`: forward order two, antisymmetric order exactly 5, 7 and 9 on every ordered forest, and the 24 printed tree coefficients of the three-stage scheme. SymPy only supplies the number fields, and the verifier does not read the manuscript: the printed tree coefficients are held in the script. The second uses the Python standard library, checks both learning tables against the records, and prints summary statistics. Add `--update` to the second command to update only those two tabular environments in the master.

`experiment_records/` contains the sphere configurations and measurements, and the separate historical Kuramoto memory sweep used in the paper. The current Kuramoto learning comparison is in `../../kuramoto/results/unified/reversible_second_order_verified_20260925/`; the table checker reads that comparison directly. The historical `sources.json` records original source paths and checksums; moved or deleted entries are mapped in `../../../docs/cleanup_20261027.json`; figure destination names in that manifest are relative to the manuscript's SIAM directory. `verification.txt` records the exact checks, and `validation_environment.json` records software versions.

The ODE experiment remains in `experiments/convergence_ode`, using Diffrax/georax and the shared convergence plotter. The manuscript is built from its project root with `latexmk SIAM/ees_simods.tex`; the root `.latexmkrc` supplies the class, bibliography, and macro search paths. Overleaf should use `SIAM/ees_simods.tex` as its main document.

Before submission, confirm the action and software revision used by the saved sphere training runs, the hardware/software manifests if a controlled runtime comparison is intended, and the invocation settings for the existing rough-driver figure. The paper states the limitations of the archived metadata. The Poincaré/energy experiment remains deferred.
