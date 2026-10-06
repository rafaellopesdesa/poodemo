# Validation

Validated on 6 October 2026 with Python 3.12, JAX/JAXlib 0.5.3,
PyTorch Lightning 2.5.5, PyTorch 2.14.1 (CPU execution), NumPy 2.3.5,
SciPy 1.17.0, and iminuit 2.33.0. Toolkit source:
`fc09848fc6540fd32310faebbe9db6eea7ecd17b`.

## Notebook 5 spline-anchor regression

The committed notebook stopped after eta=0.4625 with `profile fit failed:
conditional=No free parameters; unrestricted=No scalar bracket converged`.
This was reproduced at the next scan point, `0.5349999999999999`, which is
one floating-point step below the spline anchor `0.535`. PCHIP left tiny
residues in otherwise empty bins, differing among the nominal and varied
templates. The exponential-polynomial interpolation then rejected their
inconsistent support or nonpositive ratios, making every likelihood
evaluation invalid.

Spline queries within roundoff of an anchor now restore its stored fractions,
including exact zeros. This also applies at endpoints. No positive floor is
added to empty bins; interpolation away from anchors and the prohibition on
extrapolation are unchanged.

**23 targeted tests passed**, including four new sparse-template regression
cases covering exact and one-step-adjacent anchors, scalar and batched queries,
stat-only/profiled fits, and strict rejection of extrapolation.

The failure was reproduced using the existing smoke quadrature with 128 bins
and the production eta grids (61 spline grid points and 41 scan grid points,
plus the truth anchor). After the fix, **all notebook 5 analysis cells executed**
and all **84 stat-only/profiled direct-versus-spline comparisons were finite**.
This checks execution and numerical stability using smoke data, not a rerun
of the user's production samples. The largest spline/direct test-statistic
difference in this finer-grid smoke check was 0.800; interpolation accuracy
remains a separate diagnostic. The saved failed-cell output was cleared;
other completed notebook outputs were retained.

## Colab JAX plugin regression

Notebook 3's committed output showed an incompatible CUDA 13 JAX plugin calling
the missing `jaxlib.xla_client.register_custom_type_handler` API. The traceback
was logged during plugin discovery; no completed unbinned-fit output was saved.
The initial repair selected CPU fits. The current setup selects GPU when an
NVIDIA device is detected, installs matching JAX/JAXlib/CUDA 12 plugin/PJRT
packages at 0.5.3, and removes incompatible plugins. CPU remains available for
CPU runtimes or an explicit choice. A GPU initialization error is reported
instead of silently falling back. The live kernel checks double-precision JIT
values and gradients on the requested device before training. Saved training
products remain compatible; restart once after the earlier error or CPU setup.

The regression reproduces the reported plugin failure using real JAX with an
isolated incompatible plugin fixture, including when `JAX_PLATFORMS=cpu` is
set. Setup recovery and CPU JIT execution are checked without a GPU or live
Colab account; GPU selection and failure handling use simulated devices in the
bootstrap tests. The four-package CUDA stack resolves successfully for Python
3.13 Linux x86_64, and matching wheels are available. The explicit hyphenated
`with-cuda` extra follows the JAX maintainers' documented 0.5.3 installation fix.

**16 targeted tests passed for this GPU setup revision:** five JAX environment
checks, three notebook bootstrap checks, five toolkit likelihood checks, and
three pipeline checks. All five notebooks pass format and Python syntax
validation; analysis cells and their saved outputs are unchanged.

**No physical GPU is available in this validation environment.** CUDA execution
and CPU/GPU timing have not been measured here. The Colab preflight verifies
actual GPU execution when the notebooks run there. Completed notebook
diagnostics and figures are retained.

## Colab import regression

The output committed in notebook 1 reproduced `ModuleNotFoundError: No module
named 'poodemo.pipeline'` after installation. The shared setup now explicitly
adds the checkout to the running kernel's import path and clears an incorrectly
cached `poodemo` package when necessary.

**Three targeted checks passed:** initial import in the Colab directory layout,
recovery after a failed import cached the outer checkout as a namespace package,
and consistency of the setup cell across all five notebooks. Both import checks
also rerun setup after success and confirm that the correctly loaded package is
retained. These run in isolated Python processes with Drive mounting stubbed and
installation skipped; they do not claim a live authenticated Colab session.
All regenerated notebooks also pass format and Python syntax validation.

## NI-only scientific workflow checks

The following results are from the NI-only revision before the setup-only
fixes; the scientific code and configuration are unchanged by those fixes.

- **33 automated tests passed.** These include coherent amplitude positivity,
  Gaussian overlap normalization, exact SBI sampling moments, analytic score
  finite differences, selected-rate normalization, degree-six interpolation
  and its derivatives, stable Poisson likelihoods near their minimum,
  multi-minimum scalar fits, fraction spline positivity/zero preservation,
  actual toolkit JAX/NumPy likelihood agreement, actual toolkit training and
  checkpoint reload, and stale-selection protection. The NI-only generation
  inventory and rejection of older run manifests are also covered.
- The NI Gaussian constraint is checked explicitly with nuisance-neutral
  templates: its contribution is exactly **alpha_NI² to -2 log L** in the
  actual toolkit JAX objective, the NumPy unbinned likelihood, both direct
  histogram morph orders, and held-out spline templates. The JAX nuisance
  derivative is also checked against 2 alpha_NI.
- **All 33 code cells in all five notebooks executed successfully** with the
  smoke configuration. This included generating every nominal/varied sample,
  training the selector and all five ratio tasks, constructing workspaces,
  fitting the Asimov scans, filling and fitting histograms, fitting splines,
  and producing all eight PDF figures. This was a fresh NI-only run with six
  samples and one nuisance parameter. Those test executions did not save outputs.
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
  statistic and **0.281** after profiling NI. Six of the ten tested mu values
  lie between spline anchors. These are measured residual errors, not zero
  by construction.

The deliberately undertrained smoke density ratios **do not achieve physics
closure**: their fitted mu is approximately 0.274 without nuisances and 0.260
with NI profiling, whereas the generating value is 1. The notebooks expose these
discrepancies using analytical truth; they never substitute a learned-model
Asimov sample to conceal them. An optimizer's valid minimum alone does not
demonstrate a calibrated model.

Production defaults are 5 million events in each of six samples, longer
training, three-member ratio ensembles, finer histograms, and 61 spline-grid
points. **That full production training has not been executed here.** Its
calibration, positive intensities over the fitted domain, integration precision,
and between-anchor spline errors must be judged from the generated diagnostics.

## Reproduce

```bash
python -m pip install -r requirements-colab.txt
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/smoke.py --root /tmp/poodemo-smoke-ni
```

Use a new directory for a changed configuration. The tutorial does not claim
frequentist coverage validation from these Asimov calculations.
