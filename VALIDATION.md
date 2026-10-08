# Validation

## Larger NI bank and ratio retraining v4 (8 October 2026)

New run: `paper-distinct-s-v4`, schema 7. Physics amplitudes and physical yields
are unchanged. Rerun 01–03, then 04–09 for updated inference results. Existing
v3 runs remain separate. Committed notebook outputs are cleared to avoid mixing
old fitted results with the new configuration; scientific plotting/binning
choices are retained.

- Production generates 5 million S/SBI/B events each and 50 million events each
  for NI, NI_up and NI_down. Source-specific manifests, splitting, selected
  efficiency accounting and stale-cache checks use the actual sample lengths.
  NI physical yields do not change when the Monte Carlo count increases.
- Production ratios have three 1024-unit SiLU hidden layers, a linear logit and
  an explicit final Sigmoid, with batch size 1024. The public output is in [0,1].
  BCE uses logits for numerical stability and raw ratios use exp(logit); the
  loss has no dropout, weight decay, smoothing or additional penalty.
- The LR decreases from 1e-3 to exactly 1e-9 by zero-based epoch 90, then trains
  at that floor for ten epochs. Ratio early stopping is disabled. The selected
  checkpoint remains the best validation epoch, which can precede the final
  training epoch. Histories and metadata report the LR actually used.
- **126 tests passed; 11 optional JAX checks skipped.** Actual CPU Lightning
  tests cover the full-width architecture, explicit sigmoid, BCE values and
  gradients, extreme-logit stability, legacy/new checkpoint loading, and the
  real learning rates used in training. An interrupted/resumed four-epoch test
  preserves earlier history, replaces replayed rows and uses rates
  1e-3, 1e-5, 1e-7, 1e-9, with completed metadata reporting that floor.
- Callback tests enforce fit → raw member plots → display before the next
  member starts. Only the reserved calibration partition normalizes the final
  ratio; its raw/normalized plots then appear before the next task. A failed
  diagnostic interrupts the sequence. Member checks never modify model weights
  or averaging conventions.
- The revised notebook cells were exercised on a fresh smoke run with 12,000
  S/SBI/B events and 120,000 events per NI source, a trained selector and all
  five ratio tasks. The independent bank and immediate member/ensemble checks
  produced 20 PDF/PNG figures and 20 diagnostic CSVs. Representative calibration
  residual and reweighting ratio figures were visually inspected.
- All nine notebook formats and code cells validate. Notebook 03's updated
  cells match the generator and place bank preparation before training.

The smoke run checks execution, not calibration precision. The 165-million-event
production generation and full 1024-wide production training were not run here.

## Notebook 03 density and MLE diagnostics (8 October 2026)

The physics, run configuration, tag and trained-network behavior are unchanged.
Existing checkpoints gain architecture metadata without changing their weights.

- **102 tests passed, 11 skipped.** The skipped tests require optional JAX
  inference dependencies not installed for this check. Real CPU PyTorch and
  the pinned toolkit were installed for training tests: a two-epoch fit,
  checkpoint reload, BCE value/gradient comparison and metadata enrichment
  preserving checkpoint bytes all passed. The actual network has no dropout,
  batch normalization, weight decay or extra loss penalty.
- Reweighting tests retain deliberate normalization bias and probability mass
  outside the plotting window. Calibration tests check equal class priors with
  unequal sample counts and score/fraction covariance in the residual errors.
  Sparse bins remain unavailable, not artificial zero-error agreement.
- Closure tests compare the independent likelihood with the existing morphing
  model; an oracle recovers mu=1 while an intentionally distorted density does
  not. Observation mass is not renormalized. Tests also cover frozen workspace
  normalizers, integration-node positivity, raw ratio mean/SE/ESS, local scan
  refinement and rejection of a changed selector checkpoint.
- The three added notebook code cells executed with an actual trained smoke
  selector and all five toolkit ratio classifiers. A fresh bank generated
  20,000 events per source; 35,000 selected S/B/NI mixture nodes supplied the
  independent expectation and a separate Poisson sample retained 6,902 events.
  Architecture JSON, ten reweighting/calibration figures, expectation
  convergence and overview/zoom MLE scans were generated as PDF/PNG with CSV
  tables. The rendered layouts were visually inspected.
- Notebook 03 passes format and Python syntax validation; its new cells match
  the notebook generator. Preexisting cells and all other notebooks are
  preserved, including the user's saved outputs in notebooks 01–02.

This reduced run validates execution and diagnostics, not production closure.
Its undertrained ratios and small original quadrature produce visible biases,
which the new plots retain. The independent validation samples never normalize
the fitted model. The full production diagnostic budgets and the user's saved
production networks have not been run here. The added independent likelihood
fits use NumPy/SciPy on CPU; the existing JAX inference path is unchanged.

## Selected-distribution plots in notebook 02 (8 October 2026)

Plot-only addition; no physics, training, configuration or tag changes.

- Reuses the common positive quadrature after the frozen selection. Shapes
  divide by full selected rates; stacks keep accepted yields and combine
  coherent intensities before histogramming. Inclusive notebook-01 behavior
  and output names are preserved.
- **10 plotting tests passed**, including selected yields, finite-window tail
  losses, mu=0/B and mu=1/SBI identities, unchanged NI, positivity and an
  independent complex-amplitude comparison near destructive cancellation.
- The actual new notebook cell executed with a 30,000-event per component
  Gaussian sample and a fixed analytical cut, yielding 54,148 selected
  integration nodes. Only loading the trained selector/cache was substituted
  by the analytical cut in this check. It produced six PDF/PNG files and the
  selected-yield CSV; rendered process and stack layouts were inspected.
- Notebook 02 validates and all its code cells parse. Existing notebook cells
  and outputs are retained, and the generator includes the two added cells.

## Distinct-signal benchmark and notebook 01 (8 October 2026)

Fresh run name: `paper-distinct-s-v3` (schema 6). Samples and networks from the
previous physics model must be regenerated/retrained. Old run directories are
untouched; notebook output cells are cleared and their source settings retained,
except for the shared run name and scan zoom lower bound.

- B mean is shifted by -0.35 along the NI direction; cov_B=1.25*cov_S,
  phase_cos=-0.24, exposure=50. S/B full-space total-variation distance is about
  0.17, versus 0.004 for SBI/B. There are still six samples and five ratio tasks.
- Independent ideal-selector integration uses 2^16 Sobol nodes per S/B/NI
  process, scramble seeds 1011–1013, and the frozen threshold 0.2843 calibrated
  independently with 2^18 nodes per process (seeds 900–902). It retains 118,529
  quadrature nodes, approximately 91.5% of S, and selected S/NI=0.09997.
- Fixed-NI secondary minimum: mu=0.11977, q=4.2902; barrier q=7.3276.
  Profiled-NI secondary: mu=0.06889, q=2.3134; barrier q=7.1063.
  The primary remains at mu=1. Local sigma_mu increases 0.13759→0.14775.
  The complete selected scan is saved in `validation/distinct_s_v3.json`.
- A separate full-phase-space calculation also has secondary minima at about
  0.126 (NI fixed) and 0.111 (profiled). The effect is not created by selection.
  These checks use analytical selection, not a retrained network; the actual
  production selection and learned fits can shift the extrema.
- All five notebook-01 scientific code cells executed on a fresh 12,000-event
  per sample run. The process overlay and linear/log coherent stacks produced
  six PDF/PNG outputs; all were rendered and visually inspected. No ATLAS label
  is added. The stacks use exact Gaussian marginal bin integrals to avoid
  negative bins from subtraction of independent MC samples.
- **77 tests passed, 12 skipped**. Skipped tests require optional toolkit,
  training or JAX dependencies unavailable in this environment. New plotting
  checks cover full-space integrals, positivity, mu=0 yielding B, and an
  independent MC marginal check. Benchmark tests verify shape contrast and
  the lifted secondary basin with and without the constrained NI nuisance.

The numerical benchmark can be regenerated with:

```bash
python scripts/validate_benchmark.py --threshold .2843 --power 16 --seed 1011 --output benchmark_validation.json
```

Production training and notebooks 02–09 have not been rerun for this model.
The scan starts at 0.02. The score scale changes to 0.04 to accommodate its
broader distribution; leaving the old 0.005 scale would cause artificial
finite-bin saturation in comparisons with the ratio observables.

## Physical-reference ratio notebooks (7 October 2026)

Notebooks 08/09 mirror the current 06/07 sources with the analytical observable
R/(1+R), R=p_selected(x;eta,0)/p_selected(x;1,0). All seven earlier notebooks
are unchanged. No new training or production rerun was performed.

- **72 tests passed, 10 skipped** in this environment. Skips concern optional
  training/toolkit dependencies; this run does not establish GPU compatibility.
- Tests verify cancellation of the S reference, selected-density normalization,
  exact z=1/2 at eta=1, zero shape Fisher information at that anchor, and
  preservation of pairwise likelihood differences in resolved categorical bins.
  Real SciPy binning/spline/estimator workflows verify finite fits, exact Asimov
  t(1)=0 and separate artifact names.
- Scientific and plotting cells ran on 4,500 independent Gaussian mixture
  quadrature nodes without preselection, using the physical model, 17 uniform
  scan points plus truth, all 15 coarsenings from 4 to 60, 32 estimator bins,
  and 20 spline bins with the smoke anchor grid. Setup and checkpoint loading
  were bypassed; analytical unbinned references were fitted with SciPy, and the
  learned-model refresh was replaced by these references. All figures rendered.
  This checks notebook execution, not neural closure or selected production data.
- The midpoint checks explicitly expose the collapse: the largest bin-fraction
  error was about 0.47 near eta=1. The coarse scan happened to coincide with
  spline anchors, so its tiny likelihood differences are not evidence of
  off-anchor interpolation accuracy. Notebook 09 surfaces the worst midpoint
  errors and warns that continuous PCHIP cannot represent the bin discontinuity.

Uniform bins, the singular observable, and the original spline method are kept
intentionally for a controlled comparison. No adaptive rescaling, jitter or
silent direct-template fallback masks this behavior.

## Parallel ratio-observable notebooks (7 October 2026)

Notebooks 6–7 repeat the score workflow with the analytical selected-density
ratio to S, transformed as r/(1+r). The selected rates, integration sample,
physical coefficients, NI interpolation and Gaussian constraint are shared.
Ratio products have separate filenames. Existing notebooks 1–5, including the
user's executed outputs and plot settings, were preserved byte for byte.

- **78 tests passed, 1 skipped** with the pinned toolkit/JAX/Minuit stack.
  The skipped test requires optional PyTorch training dependencies.
- New checks cover normalized selected densities, common-exposure invariance,
  independence from `score_scale`, and physical-parameter changes inside frozen
  ratio bins. A categorical interference example verifies genuine local
  information loss under ratio compression even with arbitrarily fine bins.
- Real small SciPy studies exercise score and ratio coarsening, splines and
  estimator diagnostics in the same run, verifying that every score product
  remains unchanged and every corresponding ratio product is written.
- Every scientific code cell in both new notebooks was executed on an
  independent 3,938-node analytical-selector quadrature. The check used six
  scan points, all 15 coarsenings from 4 to 60 bins, 512 bins for estimator
  diagnostics, and 20 bins with 52 anchors for the reduced spline study.
  It exercised the actual toolkit/JAX/Minuit unbinned references with analytical
  surrogate fields, the SciPy binned fits, and all nine saved figures.
  All fits and local-width diagnostics were valid; the largest fixed-truth
  estimator displacement was 6.3e-8. Sentinel score files remained unchanged.
- All seven notebooks validate and their Python cells parse; the new notebooks
  are committed without execution outputs.

The small execution check validates the workflow, not production interpolation
precision. Its deliberately coarse spline grid gave a maximum direct/spline
test-statistic difference of about 1.15; the production grid is unchanged from
the score study and its precision must be judged from the saved comparisons.
No additional network training or full production rerun was performed here.

## Finite learned-model Asimov construction (7 October 2026)

The three unbinned comparisons now distinguish analytical generating weights
from weights of the finite, normalized learned model. Notebook 4 refreshes these
fits using existing checkpoints; it does not retrain networks or alter the
analytical histogram data.

- **66 tests passed, 1 skipped** with the pinned toolkit, JAX and Minuit. The
  skipped test requires the optional PyTorch training stack.
- Deliberately distorted learned shapes at generating mu=0.7 and 1.3 close at
  their own generating point: the objective and both parameter gradients vanish,
  and fixed/profiled fits recover the input value. The negative S or B coefficient
  is retained. Morphed shapes integrate to their prescribed signed total rate at
  intermediate and extrapolated nuisance values.
- The same distorted model evaluated on analytical generating weights still
  fails closure. Arbitrary constant rescalings of all five predicted ratios
  cancel through their finite-sample normalizers.
- An independent small pipeline execution exercised all three workspaces and
  the saved scan/fit tables with surrogate ratio predictors and the actual
  toolkit/JAX/Minuit fits. At mu=0.7, learned-model Asimov fits returned
  0.700000 (fixed NI) and 0.700014 (profiled NI); the objective at truth was
  below 1e-27. The deliberately wrong learned model retained a nonzero
  analytical-truth objective of about 31.36.

This verifies the finite construction and its integration into the workflow.
It does not validate the user's production networks or claim that self-closure
removes learned-shape errors in curvature or the secondary likelihood basin.

## Nearby-minimum benchmark (schema 5, 7 October 2026)

Use `RUN_NAME = "paper-nearby-v2"` and rerun notebooks 1–5. The new model
keeps truth mu=1, sets cos(phase)=-0.273, reduces the S/B mean separation
to 0.03 along the existing direction, and uses covariance_B=1.0045*covariance_S.
Exposure is 2000; the inclusive S/B and B/NI ratios and generated Monte Carlo
sample counts are unchanged. Only NI is varied, with its existing 30% mean
shift and unit-Gaussian auxiliary constraint.

The phase targets a second branch near mu=0.5. Nearby equal-rate roots alone
would leave a shallow barrier, so the increased exposure makes it visible.
The distinct S/B shapes lift the second minimum above the generating minimum.
The physical scan is 0.2–1.4, with fits over 0.001–2.0 and score scale 0.005.
The analytical landscape figure focuses on test-statistic values up to 10;
the overview and saved tables retain the full range.

An ideal balanced-classifier selector, with a frozen cut calibrated on separate
Sobol draws, gives these **full-shape** analytical results. Profiling uses the
same normalized exp-poly model and Gaussian constraint as the notebooks.

| Analytical oracle diagnostic | NI fixed | NI profiled |
|---|---:|---:|
| Secondary minimum mu | 0.50956 | 0.49616 |
| Secondary minimum statistic | 0.44452 | 0.22539 |
| Intervening barrier statistic | 5.9924 | 4.8388 |
| Local sigma_mu at truth | 0.05919 | 0.06773 |

The table uses 65,536 Sobol nodes per proposal component and independent
scrambles 1011–1013. A separate 32,768-node/component calculation agrees to
about 0.001 in the reported statistics. These are design checks with an
analytical selector, not a guarantee of the precise minima after a learned
selection. Reproduce the independent calculation with:

```bash
python scripts/validate_benchmark.py --power 16 --seed 1011 --output /tmp/poodemo-nearby.json
```

Actual toolkit/JAX/Minuit fits were also run on an independent 15,727-node
selected quadrature. Both global fits recovered (mu,alpha_NI)=(1,0), while
starts in the other basin found minima at 0.50956 and 0.49618. Checked toolkit
and NumPy likelihood values agreed within 1e-9. The toolkit objective now
integrates a stable per-node Poisson KL expression, avoiding cancellation
between large rates near the minimum. Regression checks cover exposure factors
2000 and 1,000,000, including JAX gradients.

The spline grid uses 2401 uniform knots plus 1201 knots within 0.1 of the
nominal selected-rate turnover, and the truth anchor. Knots depend on expected
nominal yields, not on the observed histogram. The 82-point regular scan is
intentionally mostly off these knots; saved `is_spline_anchor` flags distinguish
interpolation checks from exact-anchor agreement. Compact column construction
avoids millions of Python dictionaries in the denser template-yield table.

An independent spline audit with 503,109 selected Sobol nodes, 128 bins,
and 15 deliberately off-grid queries found no support or fit failures. Largest
absolute spline/direct statistic differences were 0.216 (NI fixed) and 0.309
(profiled); the latter occurred in a tail with statistic about 68.7. Near the
turnover the largest profiled difference was 0.142. Near the secondary basin,
a checked profiled value changed from 0.113 to 0.356: the approximation is
useful but is not exact, and these differences can matter for precision work.
The former 601-knot grid produced large distortions and is no longer the default.
Finite integration noise remains separate from interpolation resolution; denser
knots alone do not guarantee monotonic improvement. Notebook 5 reports its own
midpoint template errors and off-grid likelihood comparisons.

Validation scope for this revision:

- **59 automated tests passed; one optional neural-training roundtrip was
  skipped** because PyTorch was not installed in this execution environment.
- All five regenerated notebooks validate as notebooks, and all 38 code cells
  compile. The analytical landscape and stacked-histogram cells were executed
  with the new model and visually inspected.
- The actual binning, estimator, and spline pipeline stages completed with an
  independent analytical-oracle integration sample and smoke budgets. All fit
  and width diagnostics were valid; the largest Asimov MLE displacement was
  below 2e-7. Smoke grids can coincide with spline anchors, so their agreement
  is an execution check, not interpolation-accuracy evidence.
- Full production neural retraining and a live Colab GPU session were not run
  for this revision. Higher exposure makes ratio-shape errors more consequential;
  the learned-versus-analytical closure plots must establish production accuracy.
  No confidence-set coverage claim is made from the Asimov checks.

## Earlier validation history

The sections below describe earlier revisions and their then-current defaults;
those numerical results do not describe the schema-5 model above.

Validated on 6 October 2026 with Python 3.12, JAX/JAXlib 0.5.3,
PyTorch Lightning 2.5.5, PyTorch 2.14.1 (CPU execution), NumPy 2.3.5,
SciPy 1.17.0, and iminuit 2.33.0. Toolkit source:
`fc09848fc6540fd32310faebbe9db6eea7ecd17b`.

## Paper benchmark (schema 4)

The `paper-interference-v1` revision changes the physics model and requires a
fresh run of notebooks 1–5. S and B now have nearby but distinct Gaussian
shapes, while the destructive phase remains -0.65. NI alone has a 30% mean
displacement toward S, with the same unit-Gaussian auxiliary constraint.
The physical scan covers 0.1–12 and fits cover 0.001–14. The logistic score
scale is 0.075, avoiding unnecessarily narrow score histograms.

- **57 automated tests passed.** New checks cover the secondary full-shape
  likelihood basin, the stronger NI effect, Fisher versus numerical Asimov
  Hessians and profiled curvature, fixed-data estimator fits, reuse of checked
  toolkit baselines, and rejection of incompatible physics caches.
- **All 38 code cells in all five regenerated notebooks executed** against a
  fresh smoke run, including generation, selector and ratio training, toolkit
  fits, histograms, local-width/MLE diagnostics, splines, and all figures.
  Execution used the local bootstrap and CPU; the authenticated Colab session
  and the new full production training were not run here.
- The trained smoke selector achieved calibration S/NI = **0.10056**, with
  independent holdout S/NI = **0.12308** at its small integration budget.
  The analytical unbinned scans clearly show the second basin. At the coarse
  smoke grid point mu = 7.5375, the statistic is **2.485** with NI fixed and
  **1.192** with NI profiled; the intervening sampled peaks are about 7.88
  and 4.45, respectively.
- The analytical unbinned local widths are **0.3470** and **0.4227** (fixed
  and profiled NI). Across the frozen-eta direct histograms, the largest
  Asimov MLE displacement from truth was **4.8e-8**. All estimator/width rows
  were valid. These are local widths at the fixed generating truth, not
  global confidence intervals across the second minimum.
- The smoke ratio fits remain deliberately undertrained: fitted mu values
  were **1.425** (fixed NI) and **2.505** (profiled NI), versus generating mu=1.
  Production training must establish ratio closure independently; analytical
  results are never substituted for failed learned-model closure.
- The 32-bin, 61-anchor smoke spline/direct comparison had a largest absolute
  statistic difference of **2.16**. This is an approximation error, not a
  coverage result. The paper defaults use 601 spline grid points and 81 scan
  grid points, plus truth; 40 regular scan points remain between anchors.
  Denser anchors alone cannot remove finite quadrature noise. Notebook 5
  reports both held-out bin-fraction errors and fitted likelihood differences.

A separate higher-statistics spline check used 65,536 Sobol points per
proposal component (123,355 selected nodes), an ideal frozen selector, and
128 score bins. At 49 held-out or targeted eta points, 601 uniform anchors
plus truth gave no support or fit failures. The largest absolute statistic
differences were **0.080** with NI fixed and **0.105** after profiling. The
coarser 121-anchor trial had empty-support failures and larger distortions.
This supports the denser default but is not a uniform error guarantee; the
production run still validates its own learned selection and quadrature.

An independent analytical preflight uses an ideal balanced multiclass
selector with a separately calibrated, frozen cut; it does not use the
trained smoke selector. Reproduce it with:

```bash
python scripts/validate_benchmark.py --power 15 --seed 100 --output /tmp/poodemo-benchmark.json
```

| Analytical oracle diagnostic | NI fixed | NI profiled |
|---|---:|---:|
| Secondary minimum mu | 7.7652 | 7.1490 |
| Secondary minimum statistic | 2.6367 | 1.0982 |
| Intervening barrier statistic | 8.0686 | 4.3955 |
| Local sigma_mu at truth | 0.3434 | 0.4243 |

Doubling the quadrature with independent Sobol scrambles changes the secondary
minimum statistics by about 0.001. In this oracle study, 128/256/512 score bins
retain at least 99.08%/99.75%/99.94% of extended Fisher information across the
tested anchors. These numerical checks motivate the defaults; the production
notebooks report their own results after the learned selection.

The remaining sections record earlier schema-3 validation and runtime fixes.

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
discrepancies using analytical truth. A separately labelled learned-model finite
Asimov curve now tests self-closure without replacing that analytical-truth
comparison. An optimizer's valid minimum alone does not demonstrate a calibrated
model.

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
