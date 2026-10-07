# Parameterized optimal observables: an interference toy

Five Google Colab notebooks connect analytical quantum amplitudes, ATLAS-style
neural simulation-based inference, and a parameterized binned approximation
built from the likelihood score. Data and trained models persist in your Google
Drive; the source lives in this repository.
The nearby-minimum benchmark makes secondary-minimum and nuisance effects
visible. It is an illustrative toy, not a numerical reproduction of an ATLAS
measurement. The generating value is $\mu_*=1$, with a secondary branch designed
to lie near $\mu\simeq0.5$.

| Notebook | Purpose | Open in Colab |
|---|---|---|
| [01_generate_amplitudes.ipynb](notebooks/01_generate_amplitudes.ipynb) | Generate coherent S/B amplitudes, SBI and NI samples, and nuisance variations | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/01_generate_amplitudes.ipynb) |
| [02_preselection.ipynb](notebooks/02_preselection.ipynb) | Train S/B/NI multiclass preselection and select approximately S/NI = 0.1 | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/02_preselection.ipynb) |
| [03_unbinned_nsbi.ipynb](notebooks/03_unbinned_nsbi.ipynb) | Train density ratios, build workspaces, and compare analytical and learned Asimov fits | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/03_unbinned_nsbi.ipynb) |
| [04_score_histograms.ipynb](notebooks/04_score_histograms.ipynb) | Study score histograms, local widths, and fitted signal strength versus the observable anchor | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/04_score_histograms.ipynb) |
| [05_spline_templates.ipynb](notebooks/05_spline_templates.ipynb) | Interpolate moving bin fractions, validate splines, and compare profiled scans | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/05_spline_templates.ipynb) |

## Run in Colab

1. Open notebook 1, then run notebooks in numerical order. For a private
   repository, authorize GitHub access in Colab's notebook browser. If the link
   does not open, download the `.ipynb` and upload it to Colab.
2. Choose a GPU runtime for notebooks 2–3. Both PyTorch training and JAX
   likelihood fits use the GPU. `JAX_BACKEND = "auto"` detects an NVIDIA GPU;
   use `"gpu"` to require one, or `"cpu"` for an intentional CPU run.
   The other notebooks can use a CPU runtime.
3. In each notebook use the same `RUN_NAME` and `MODE`. Defaults are
   `RUN_NAME = "paper-nearby-v2"` and `MODE = "production"`. For an initial
   short check, choose `MODE = "smoke"` and a distinct run name such as
   `paper-nearby-smoke-v2`.
   Computational budgets can be changed through `CONFIG_OVERRIDES`, for example
   `{"quadrature_per_process": 500_000, "epochs": 150}`. Use identical overrides
   in every notebook and a new run name when changing the configuration.
4. Mount Google Drive when prompted. Run products are stored in
   `MyDrive/poodemo/runs/<RUN_NAME>/`.
5. If the repository is private, grant runtime download access using a
   fine-grained token with **Contents: read** permission for `poodemo`. Store it
   as the Colab secret `GITHUB_TOKEN` or enter it at the hidden prompt. It is
   used only in an HTTP authorization header: never in shell commands, Git
   remotes, notebook output, or saved configuration. Alternatively upload the
   repository ZIP to `/content/poodemo_source.zip` before running setup.

Each notebook can start in a fresh Colab runtime. Its bootstrap downloads the
source and installs the pinned toolkit. An existing `/content/poodemo` checkout
is reused; remove that temporary checkout to fetch a newer source version.
Completed data files and stage products can be reused after disconnection.
GPU setup installs JAX, JAXlib, the CUDA 12 plugin, and CUDA 12 PJRT at version
0.5.3. It removes incompatible preinstalled JAX plugins and checks a
double-precision JIT value and gradient on the selected device before training.
A detected/requested GPU that fails to initialize raises an error rather than
silently running the fit on CPU. Startup prints `backend=gpu` on a GPU runtime.
JAX memory preallocation is disabled by default so it can share the GPU with
PyTorch. For a runtime-only repair, such as switching from the previous CPU-only setup or recovering from the
`register_custom_type_handler` error, restart the runtime once, reopen the
updated notebook, and rerun setup with the same `RUN_NAME`.
Completed networks and samples in Drive are reused; retraining is not required.
Selected-data and quadrature caches record the selector fingerprint and reject
stale reuse after its model or threshold changes.
Use a new run name when changing the configuration: incompatible saved settings
are rejected rather than silently mixed.

For this changed physics benchmark, use the new run directory
`paper-nearby-v2` and rerun notebooks 1–5. Schema-5 manifests prevent reuse
of the earlier physics configuration.
If an earlier version was already run in Colab, start a fresh runtime, or
refresh `/content/poodemo` and restart the runtime, so that the new source is
loaded. Earlier Drive runs remain intact, but their samples and trained models
must not be reused for the changed amplitudes and exposure.

Production generates **5 million events in each of six samples**: S, SBI, B,
NI, NI_up, and NI_down, for 30 million events total. The raw three-coordinate
float32 arrays occupy about 360 MB; selected arrays, network ensembles, cached
quadrature, and workspaces need additional space. Data are generated and evaluated in chunks. Training
has configurable sample caps, early stopping, and saved checkpoints. Production
training is substantial and is not run automatically when opening a notebook.
Smoke mode uses 12,000 events per sample, 20 preselection epochs, and five
density-ratio epochs with single-model training; it
checks the workflow and does **not** establish final physics precision.

## The model

For normalized three-dimensional Gaussian densities
$\phi_j=\mathcal N_3(m_j,\Sigma_j)$, use

$$
A_S=\sqrt{\lambda_S\phi_S},\qquad
A_B=e^{i\varphi}\sqrt{\lambda_B\phi_B},\qquad
\cos\varphi<0.
$$

These are Gaussian wavefunctions. Inclusive yields, the destructive phase,
means, and correlated covariance matrices are displayed in notebook 1 and
recorded in `run.json`. S and B shapes are deliberately close to make the
interference ambiguity visible. The SBI template is a positive process with intensity
\(D_{SBI}=|A_S+A_B|^2\). Its integral is calculated analytically, including
interference, and is reported rather than assigned an independent normalization.
NI is incoherent.

Relative to the earlier distant-branch benchmark, the destructive phase has
smaller magnitude and S/B shapes are even more similar. The expected exposure
is increased to keep the barrier between the nearby branches visible. Exposure
scales physical event yields, preserving the process-yield ratios; it does not
increase the five-million-event Monte Carlo samples. Notebook 1 prints the
exposure multiplier and the actual inclusive yields.

The physical intensity is

$$
D(x;\mu)=|\sqrt\mu A_S+A_B|^2+D_{NI}
=(\mu-\sqrt\mu)D_S+\sqrt\mu D_{SBI}
 +(1-\sqrt\mu)D_B+D_{NI}.
$$

The Gaussian-amplitude construction makes this intensity positive for
$\mu\ge0$. SBI is sampled with rejection sampling using an exact envelope.
The number of generated Monte Carlo events is independent of the expected
number of events in the statistical experiment.

Destructive interference alone does not guarantee a second likelihood minimum:
shape information, preselection, and the scan range also matter. At nominal NI
nuisance, the selected total rate is

$$
\Lambda_\mu=\lambda_S\mu+\lambda_I\sqrt\mu+\lambda_B+\lambda_{NI},\qquad
\lambda_I=\lambda_{SBI}-\lambda_S-\lambda_B.
$$

Besides the generating value $\mu_*$, the same total rate occurs at
$\mu_{\rm rate}=(-\lambda_I/\lambda_S-\sqrt{\mu_*})^2$ when the term in
parentheses is nonnegative. Equal rates do not imply equal event distributions.
Notebook 3 calculates this rate diagnostic after selection and plots the actual
analytical likelihood, separately with NI fixed and profiled. Any secondary
minimum must be established from those curves rather than inferred from the
sign of interference alone.

For the target partner near $0.5$ with truth $1$, the rate relation is
$-\lambda_I/\lambda_S\simeq1+\sqrt{0.5}$. Its turnover lies near
$(-\lambda_I/(2\lambda_S))^2\simeq0.729$, between the two branches.
The score-distribution panels choose their anchors dynamically from the
selected-rate turnover and partner, so they display this nearby structure.
These are rate landmarks; the measured full-likelihood minima can move because
of shape information and NI profiling.
The analytical landscape figure defaults to $0\le t_A\le10$ so that the
nearby minima and barrier remain visible; set `LANDSCAPE_Q_MAX = None` in
notebook 3 for automatic vertical scaling. The overview plot and scan tables
retain the full likelihood range.

The only nuisance is $\alpha_{NI}$. It shifts the NI amplitude mean by
the configured magnitude along the direction toward the signal mean. This
deliberate direction makes the shape variation more relevant to signal-strength
inference. Notebook 1 prints the exact displacement vector and its magnitude;
notebook 2 shows the resulting NI± acceptance and selected yields.
S, B, and their coherent SBI process retain their nominal shapes and yields.
The inclusive NI yield stays fixed, while its selected yield can vary through
the selection efficiency. The nominal and ±1 NI anchors define an
exponential–polynomial interpolation; that interpolation is not assumed to be
the exact continuously shifted amplitude between anchors.

All profiled fits include a standard-normal auxiliary constraint,

$$
L_{\rm aux}(\alpha_{NI})\propto\exp(-\alpha_{NI}^2/2),
$$

which contributes $+\alpha_{NI}^2$ to $-2\log L$. The workspace parameters are
$(\mu,\alpha_{NI})$; statistical-only fits fix $\alpha_{NI}=0$.

## Statistical construction

Samples are divided into seven disjoint roles before training or selection:

| Role | Fraction |
|---|---:|
| Preselection training | 20% |
| Preselection validation | 5% |
| Selection threshold calibration | 5% |
| Density-ratio training | 40% |
| Density-ratio validation | 10% |
| Density-ratio calibration | 5% |
| Final integration | 15% |

No event used to train or tune the selector is reused for density-ratio
training, calibration, or final inference. The multiclass S output is cut using
selection-calibration events to approach a **physical** selected S/NI ratio of 0.1. The
cut is frozen and applied to every sample and nuisance variation. Selected
normalizations, including nuisance acceptance changes, are retained.

The NSBI stage uses the selected S density as reference. Process ratios and
systematic ratios are learned using the toolkit's cross-entropy neural
networks. This is neural **density-ratio estimation**, without an additional
normalizing-flow density model. The underlying analytical densities provide
an independent benchmark. The integration proposal is a positive mixture of
held-out S, B, and NI events; it is distinct from the S density-ratio reference.
There are five ratio tasks: SBI/S, B/S, NI/S, NI_up/NI, and NI_down/NI.
Production uses three networks per task, for fifteen ratio networks plus the
separate multiclass preselection network.

To isolate shape-estimation errors, learned and analytical likelihoods use the
same accepted component yields calculated on the analytical quadrature.
Sample-estimated selected rates are reported separately. Learned shapes are
normalized numerically on the common quadrature after independent ratio
calibration; the raw integrals are also saved. This deliberate use of the toy's
known densities is a benchmark, not a replacement for simulation-derived rate
estimates in a real analysis. The label **analytical unbinned** refers to exact
component and ±1-anchor densities followed by the same exponential–polynomial
nuisance model, rather than the continuously shifted Gaussian at every nuisance
value.

The unbinned comparison distinguishes three experiments:

| Curve | Likelihood model | Asimov generating intensity |
|---|---|---|
| `analytic unbinned` | Analytical | Analytical |
| `learned unbinned (analytic Asimov)` | Learned | Analytical; external closure check |
| `learned unbinned (model Asimov)` | Learned | The same finite, normalized learned model |

The last curve implements the [finite Asimov construction](https://arxiv.org/abs/2609.14136).
Let $a_m$ be the existing importance-weighted integration weights and
$q_m$ the S reference density normalized on those nodes. The finite reference
masses are $\omega_m=a_mq_m$, with $\sum_m\omega_m=1$. Each learned process
ratio is divided by $Z_j=\sum_m\omega_m r_j(x_m)$, giving
$\widehat\nu(x_m;\mu,\alpha)=q_m\sum_j c_j(\mu)\lambda_j(\alpha)
\bar r_j(x_m;\alpha)$, where $\bar r_j=r_j/Z_j$.
The complete nuisance-morphed shape is renormalized at every fitted $\alpha$.
The additional curve uses fixed generating weights
$w_m^A=a_m\widehat\nu(x_m;\mu_*,0)$; these are not recomputed for each tested
hypothesis. With positive total intensities and the Gaussian auxiliary centered
at zero, the generating point is a global likelihood maximum on this finite
sample, even when the learned shapes are imperfect. This ensures self-closure,
not agreement of curvature or the secondary minimum with analytical physics.
The separate analytical-Asimov learned fit retains that external check.

Notebook 4 refreshes all three unbinned fits from the saved ratio checkpoints
and integration sample before the histogram comparison. Existing Drive runs
remain compatible; no regeneration or retraining is required. Use a fresh
Colab runtime to load the updated repository code. The score histograms still
use analytical generating weights, so `analytic unbinned` remains their
like-for-like compression benchmark. The fit table records `asimov_source`,
the generating value and event count, and the objective at the generating point.

The nearby branches and increased exposure make density-ratio precision more
demanding: small neural shape errors can produce appreciable likelihood
distortions. Analytical benchmark agreement does not establish closure of the
trained networks. The five-epoch smoke fit is a workflow check; production
training must be assessed using the held-out diagnostics and the learned versus
analytical scans. More training or simulation may be needed before claiming a
specific closure precision.

For $D_\mu=\sum_j c_j(\mu)\lambda_jp_j$ and
$\Lambda_\mu=\sum_j c_j(\mu)\lambda_j$, notebook 4 uses

$$
o_\eta(x)=\left.\partial_\mu\log p(x;\mu)\right|_\eta
=\frac{D'_\eta(x)}{D_\eta(x)}
 -\frac{\Lambda'_\eta}{\Lambda_\eta},\qquad
z_\eta=\frac1{1+e^{-o_\eta/a}}.
$$

The positive scale $a$ is saved as `score_scale`; it is fixed throughout the run.
It preserves unbinned information while controlling the resolution of a finite
uniform histogram in $z$.

Every component includes its yield. The total-rate denominator contains only
yields, not a residual $p_{NI}(x)$ factor. Total-count information is retained
by the extended likelihood. Positive score anchors avoid the nonregular
$\sqrt\mu$ derivative at $\mu=0$.
The actual scan and fit domain are printed from `mu_min`, `mu_max`, and
`mu_fit_bounds`. The scan extends across the interference turnover and the
possible secondary minimum. The positive lower fit bound is a numerical floor for an interior
inference demonstration; testing the physical $\mu=0$ boundary needs a
separate construction.

When testing $\mu_0$, set $\eta=\mu_0$, fill its data and templates, and
**freeze that observable throughout both numerator and denominator fits**.
Spline interpolation is in the observable construction parameter $\eta$.
Physical $\mu$-dependence continues to use the exact coefficients above.
Exhaustive bins have process totals independent of $\eta$, so interpolated
bin fractions are constrained to sum to one and multiplied by separate yields.

Notebook 4 also varies $\eta$ while holding the **same generating Asimov
experiment** fixed. It compares $\hat\mu(\eta)$ and the local uncertainty

$$
\sigma_\mu^{\rm fixed}(\eta)=1/\sqrt{I_{\mu\mu}^{(\eta)}},\qquad
\sigma_\mu^{\rm profiled}(\eta)
=\sqrt{[(\mathbf I^{(\eta)})^{-1}]_{\mu\mu}}
$$

between direct score histograms and the analytical unbinned reference. Fisher
information is evaluated at the fixed generating truth and includes total-rate
information and the NI Gaussian constraint. These widths describe the local
curvature of each frozen-observable fit; they are not the curvature of the
moving-observable test-statistic scan or global multimodal confidence intervals.
For an identified, correctly modeled Asimov experiment, the fitted values should
remain at the generating value. This comparison does not simulate finite-sample
fluctuations. Profiled histogram widths use morphing after binning, with the
same morph-order distinction studied in notebook 5.

The diagnostic table is saved as `results/estimator_eta.csv`; its figure is
`plots/04_estimator_eta.pdf` (also PNG). The analytical likelihood panels are
saved as `plots/03_analytical_landscape.pdf` (also PNG).

## What the comparisons establish

The notebooks keep several effects distinct:

- Analytical versus learned unbinned curves with analytical Asimov weights:
  density-ratio estimation. The learned-model Asimov is a separate self-closure
  construction.
- Increasingly fine direct score histograms: finite binning.
- Spline versus direct templates at withheld anchors: interpolation accuracy.
- Interpolated binned anchors versus integrated unbinned morphing: nonlinear
  nuisance interpolation and bin integration generally do not commute.
- Score histograms versus the complete unbinned model: scalar compression and,
  when profiling, sensitivity to nuisance directions.

The exact unbinned score preserves the local Fisher information at its anchor
for a regular single-parameter model with nuisances fixed. It is not generally
globally sufficient in an interference model. Finer bins therefore need not
make a broad scan exactly equal to the original full-dimensional likelihood.
A scalar signal-strength score alone also does not guarantee preservation of
all profiled information; a joint score or efficient score is an extension.

The saved scans are **Asimov expectations approximated by weighted numerical
integration**, not coverage results from Poisson pseudo-experiments. Independent
Poisson bin likelihoods still require correctly predicted means. Ratio,
quadrature, and interpolation errors must be checked separately; a calibrated
Poisson probability does not fix an incorrect mean model.

Notebook 3 includes a numerical-integration uncertainty estimate and two
disjoint half-sample curves. Halves are split inside each proposal stratum
before selection, and rejected draws contribute zero when estimating the
integration variance. These errors describe numerical precision, not parameter
confidence intervals. Increase the integration budget in a separate run before
claiming a physics-level closure precision.

## Local use and implementation

```bash
python -m pip install -r requirements-colab.txt
python -m pip install -e '.[dev]'
export POODEMO_ROOT=/absolute/path/to/poodemo-runs/paper-nearby-smoke-v2
export POODEMO_MODE=smoke
```

For a local NVIDIA GPU installation, resolve both requirement files together:

```bash
python -m pip install -r requirements-colab.txt -r requirements-jax-cuda12.txt
```

The explicit `jax-cuda12-plugin[with-cuda]==0.5.3` requirement follows the
[JAX maintainers' workaround](https://github.com/jax-ml/jax/issues/27874) for
the extras-name issue in that release. JAX/JAXlib and both CUDA plugin packages
must have matching versions; incompatible preinstalled CUDA 13 plugins must be
removed. The Colab setup handles this automatically. Local users should select
their backend before importing JAX; the automatic setup described above runs
in Colab.

Run the notebooks in Jupyter with that environment, or use the high-level
functions in `poodemo.pipeline`. `create_run` validates and saves the run
configuration. All analysis outputs belong under the chosen run directory,
not in Git. Python helpers implement the physics, staged storage, toolkit
adapters, and numerical likelihood calculations; notebooks retain the
derivations and plotting steps.

For example, `create_run(path, mode="production", overrides={"epochs": 150})`
changes a configurable budget without editing the package. Unknown or protected
configuration keys are rejected, and existing runs must match their saved
settings.

Physics parameters can be varied with `physics_overrides`, for example
`CONFIG_OVERRIDES = {"physics_overrides": {"shift_fraction_ni": 0.4}}`.
Use a new run name for any such change. The production defaults use 82 scan
grid points and 2,401 uniform spline anchors, plus 1,201 focused anchors within
$\pm0.1$ of the selected nominal-rate turnover and the generating value.
Smoke mode uses 241 uniform and 121 focused anchors. Duplicate anchors are
removed and the focused interval is clipped to the scan range. The settings
are `spline_anchors`, `spline_focus_anchors`, and `spline_focus_halfwidth`.

The focused region is determined only from expected nominal process yields;
it is not selected from observed fluctuations or evaluated scan values. Near
the interference turnover, the score can change rapidly with its construction
parameter, so uniform anchors alone can miss substantial movement of bin
yields. A dense grid reduces this interpolation error but does not remove
finite quadrature noise or guarantee likelihood accuracy. Template fractions
are checked at intermediate points withheld from the spline fit. The
likelihood-comparison table labels points with `is_spline_anchor`: agreement at
an anchor does not test interpolation between anchors. Check additional
off-grid likelihood points when assessing precision. The 82-point production
scan deliberately differs from the uniform spline grid, so most of its points
test interpolation; coincident points and the included generating value are
still identified by `is_spline_anchor`. The shorter smoke scan primarily checks
workflow execution.

An analytical benchmark check can be run before training:

```bash
python scripts/validate_benchmark.py --output /tmp/poodemo-benchmark.json
```

This uses an ideal multiclass selector with a separately calibrated frozen
cut. It verifies the interference minima, NI profiling effect, and score
resolution; it does not validate trained ratios or replace notebook results.

Notebook sources are maintained in `scripts/build_notebooks.py`:

```bash
python scripts/build_notebooks.py
python -m pytest
python scripts/smoke.py --root /tmp/poodemo-paper-nearby-smoke
```

The notebook generator produces clean notebooks. This intentional physics
revision clears earlier numerical outputs so that all five notebooks can be
rerun consistently; targeted runtime fixes normally preserve user outputs. Toolkit
source is pinned to commit `fc09848fc6540fd32310faebbe9db6eea7ecd17b`;
`requirements-colab.txt` includes the dependencies its package metadata does
not declare. Compatible PyTorch ranges allow Colab's GPU-enabled build to be
reused. The experiment manifest records the toolkit commit and configuration.

See [VALIDATION.md](VALIDATION.md) for the current validation scope, runtime,
and numerical results. The automated checks cover amplitude algebra, sampling,
score derivatives, interpolation, normalization, the NI Gaussian constraint,
optimizer behavior, toolkit agreement, checkpoint reload, and cache safeguards.
The full 5-million-event production training is separate from the smoke checks.

## References

- [NSBI LHC toolkit](https://github.com/iris-hep/nsbi-lhc-toolkit), including
  multiclass and density-ratio networks, workspace construction, model morphing,
  and inference utilities.
- [ATLAS, *An implementation of neural simulation-based inference for parameter
  estimation in ATLAS*](https://arxiv.org/abs/2412.01600).
- [ATLAS, *Measurement of off-shell Higgs boson production in the four-lepton
  channel using neural simulation-based inference*](https://arxiv.org/abs/2412.01548).
- [Cranmer, Pavez and Louppe, *Approximating Likelihood Ratios with Calibrated
  Discriminative Classifiers*](https://arxiv.org/abs/1506.02169).
- [Alsing and Wandelt, *Generalized massive optimal data
  compression*](https://arxiv.org/abs/1712.00012).
- [Pollard, *A note on insufficiency and the preservation of Fisher
  information*](https://arxiv.org/abs/1107.3797).
- [Alsing and Wandelt, *Nuisance hardened data compression for fast
  likelihood-free inference*](https://arxiv.org/abs/1903.01473).
