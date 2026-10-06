# Validation

Validated on 6 October 2026 with Python 3.12, JAX/JAXlib 0.5.3,
PyTorch Lightning 2.5.5, PyTorch 2.14.1 (CPU execution), NumPy 2.3.5,
SciPy 1.17.0, and iminuit 2.33.0. Toolkit source:
`fc09848fc6540fd32310faebbe9db6eea7ecd17b`.

## Executed checks

- **30 automated tests passed.** These include coherent amplitude positivity,
  Gaussian overlap normalization, exact SBI sampling moments, analytic score
  finite differences, selected-rate normalization, degree-six interpolation
  and its derivatives, stable Poisson likelihoods near their minimum,
  multi-minimum scalar fits, fraction spline positivity/zero preservation,
  actual toolkit JAX/NumPy likelihood agreement, actual toolkit training and
  checkpoint reload, and stale-selection protection.
- **All 33 code cells in all five notebooks executed successfully** with the
  smoke configuration. This included generating every nominal/varied sample,
  training the selector and all nine ratio tasks, constructing workspaces,
  fitting the Asimov scans, filling and fitting histograms, fitting splines,
  and producing all eight PDF figures. Notebook files remain free of outputs.
- Notebook format and Python syntax were validated. Models and generated
  samples are runtime products and are not committed to the repository.

The execution was in a local runtime using the same scientific code and the
notebooks' local bootstrap path. Interactive Google Drive mounting and private
GitHub authorization must be performed by the user in Colab; they were not
claimed as an authenticated Colab execution here.

## Smoke-run numerical results

The smoke run generates 12,000 events per sample, trains the selector for up
to 20 epochs, trains each ratio for only 5 epochs, and uses 1,500 proposed
integration points per S/B/NI stratum. These budgets test execution rather
than establish precise density-ratio calibration.

- Selection calibration: S/NI = **0.10137**, retaining **86.17%** of S.
- Independent holdout S/NI: **0.11621**, with small-sample fluctuations.
- Exact-component unbinned Asimov fits recover **mu = 1** with and without
  nuisance profiling.
- At eta = 1, 8, 16, and 32 score bins retain approximately **67.4%, 84.3%,
  and 92.0%** of the extended Fisher information in this small quadrature.
- With 31 spline-grid points plus the truth anchor, the largest tested
  spline/direct-histogram difference is **0.279** in the stat-only test
  statistic and **0.278** after profiling. Six of the ten tested mu values
  lie between spline anchors. These are measured residual errors, not zero
  by construction.

The deliberately undertrained smoke density ratios **do not achieve physics
closure**: their fitted mu is approximately 0.274 without nuisances and 0.147
with profiling, whereas the generating value is 1. The notebooks expose these
discrepancies using analytical truth; they never substitute a learned-model
Asimov sample to conceal them. An optimizer's valid minimum alone does not
demonstrate a calibrated model.

Production defaults are 5 million events in each of ten samples, longer
training, three-member ratio ensembles, finer histograms, and 61 spline-grid
points. **That full production training has not been executed here.** Its
calibration, positive intensities over the fitted domain, integration precision,
and between-anchor spline errors must be judged from the generated diagnostics.

## Reproduce

```bash
python -m pip install -r requirements-colab.txt
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/smoke.py --root /tmp/poodemo-smoke
```

Use a new directory for a changed configuration. The tutorial does not claim
frequentist coverage validation from these Asimov calculations.
