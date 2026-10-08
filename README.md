# Parameterized optimal observables: an interference toy

Nine Google Colab notebooks connect analytical quantum amplitudes, ATLAS-style
neural simulation-based inference, and a parameterized binned approximation
built from either the likelihood score or a bounded density ratio. Data and trained models persist in your Google
Drive; the source lives in this repository.
The nearby-minimum benchmark makes secondary-minimum and nuisance effects
visible. It is an illustrative toy, not a numerical reproduction of an ATLAS
measurement. The generating value is $\mu_*=1$, with a secondary branch designed
to lie between zero and one while S has a visibly distinct shape.

| Notebook | Purpose | Open in Colab |
|---|---|---|
| [01_generate_amplitudes.ipynb](notebooks/01_generate_amplitudes.ipynb) | Generate coherent S/B amplitudes, SBI and NI samples, and nuisance variations | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/01_generate_amplitudes.ipynb) |
| [02_preselection.ipynb](notebooks/02_preselection.ipynb) | Train S/B/NI multiclass preselection and select approximately S/NI = 0.1 | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/02_preselection.ipynb) |
| [03_unbinned_nsbi.ipynb](notebooks/03_unbinned_nsbi.ipynb) | Train density ratios, build workspaces, and compare analytical and learned Asimov fits | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/03_unbinned_nsbi.ipynb) |
| [04_score_histograms.ipynb](notebooks/04_score_histograms.ipynb) | Study score histograms, local widths, and fitted signal strength versus the observable anchor | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/04_score_histograms.ipynb) |
| [05_spline_templates.ipynb](notebooks/05_spline_templates.ipynb) | Interpolate moving bin fractions, validate splines, and compare profiled scans | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/05_spline_templates.ipynb) |
| [06_ratio_histograms.ipynb](notebooks/06_ratio_histograms.ipynb) | Repeat notebook 4 using the bounded analytical ratio to the S reference | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/06_ratio_histograms.ipynb) |
| [07_ratio_spline_templates.ipynb](notebooks/07_ratio_spline_templates.ipynb) | Repeat notebook 5 with ratio-observable templates and profiled fits | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/07_ratio_spline_templates.ipynb) |
| [08_reference_ratio_histograms.ipynb](notebooks/08_reference_ratio_histograms.ipynb) | Repeat notebook 6 with the bounded physical ratio p(x;eta)/p(x;1) | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/08_reference_ratio_histograms.ipynb) |
| [09_reference_ratio_spline_templates.ipynb](notebooks/09_reference_ratio_spline_templates.ipynb) | Repeat notebook 7 with physical-reference ratio templates and profiled fits | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/09_reference_ratio_spline_templates.ipynb) |

## Run in Colab

1. Open notebook 1, then run notebooks in numerical order. For a private
   repository, authorize GitHub access in Colab's notebook browser. If the link
   does not open, download the `.ipynb` and upload it to Colab.
2. Choose a GPU runtime for notebooks 2–3. Both PyTorch training and JAX
   likelihood fits use the GPU. `JAX_BACKEND = "auto"` detects an NVIDIA GPU;
   use `"gpu"` to require one, or `"cpu"` for an intentional CPU run.
   The other notebooks can use a CPU runtime.
3. In each notebook use the same `RUN_NAME` and `MODE`. Defaults are
   `RUN_NAME = "paper-distinct-s-v4"` and `MODE = "production"`. For an initial
   short check, choose `MODE = "smoke"` and a distinct run name such as
   `paper-distinct-s-smoke-v4`.
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
Notebooks 6–7 reuse the same completed run and networks as notebooks 1–5.
Start notebook 6 in a fresh runtime after updating the repository, then run
notebook 7; no new data generation or training is required. Notebook 6 refreshes
the shared unbinned reference fits from saved checkpoints, just as notebook 4
does. The two histogram workflows save distinct results and plots.
Use a new run name when changing the configuration: incompatible saved settings
are rejected rather than silently mixed.

The enlarged NI bank and wider density-ratio networks require the new run
`paper-distinct-s-v4` (schema 7). The amplitudes and physical yields are unchanged.
Start a fresh Colab runtime, open updated notebook 01, and rerun **01–03** with
this run name. Rerun **04–09** to refresh downstream inference results. Old Drive
runs remain available; do not mix their selected arrays or checkpoints with v4.
A restarted/partially completed v4 run reuses compatible data and checkpoints.

Production generates **5 million events each for S, SBI and B**, and **50 million
each for NI, NI_up and NI_down**. The NI samples are ten times larger to compensate
for lower selection acceptance; physical yields do not change. The 165 million
three-coordinate float32 events occupy about 1.98 GB before selected arrays,
network checkpoints and workspaces. Generation and selection operate in chunks.
Preselection retains its original batch/width and early stopping. Ratio training
runs its full learning-rate schedule and saves the best-validation checkpoint.
Smoke mode uses 12,000 events for S/SBI/B and 120,000 for each NI source, with
small networks and five ratio epochs; it checks execution, not physics precision.

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
recorded in `run.json`. S is displaced to larger means and has narrower widths than B,
while SBI stays close to B. The destructive phase retains a second basin below 1. The SBI template is a positive process with intensity
\(D_{SBI}=|A_S+A_B|^2\). Its integral is calculated analytically, including
interference, and is reported rather than assigned an independent normalization.
NI is incoherent.

The distinct-signal benchmark uses exposure 50, phase cosine -0.24 and a
background covariance 1.25 times the signal covariance. The B mean is displaced
by 0.35 along the negative NI-nuisance direction. The normalized full-space
S/B total-variation distance is approximately 17%, versus 0.4% for SBI/B.
With an independently calibrated ideal selector, the secondary minima are near
mu=0.120 (NI fixed) and 0.069 (NI profiled); the primary is at 1. The second
basin also exists before selection. Production trained-selector results may
shift these values. See `VALIDATION.md` and `validation/distinct_s_v3.json`.

Notebook 01 now shows mplhep ATLAS-style histograms without an ATLAS label:
normalized process overlays with both linear/log axes, and physical coherent
SBI(mu)+NI stacks at mu=0,0.5,1,2 in both scales. Exact Gaussian marginal bin
integrals show tails and preserve coherent positivity; optional generated MC
markers check the sampling. Figures are saved as PDF/PNG.

Notebook 02 adds the same process overlays and coherent SBI(mu)+NI stacks after
frozen preselection, using the common selected quadrature for physical yields.
They retain linear/log views and save separate `02_selected_...` PDF/PNG files.
After setup, the new plot cell can reuse completed preselection without training
or fitting again within a completed run. These plots do not change the physics
or training configuration.

Notebook 03 records the actual layer-by-layer classifier architecture in each
model's `training.json` and in `results/03_architecture.json`. Production ratios
use **3→1024→1024→1024→1→Sigmoid**, SiLU hidden activations and **batch size 1024**.
The public network output is a probability in [0,1]. Ordinary BCE is evaluated
with its numerically stable logits implementation; ratio inference uses the
same pre-sigmoid logit to avoid loss of precision in saturated tails. Making the
sigmoid explicit does not itself change the mathematical classification loss.
NAdam uses zero weight decay; there is no dropout, batch normalization, label
smoothing or explicit loss penalty. The rate falls geometrically from 1e-3 to
**1e-9 at epoch 91** and stays there through epoch 100. Ratio early stopping is
disabled so those epochs really run, while the best-validation checkpoint is
retained. `history.json` records the rate actually used in each training epoch.
The preselection classifier remains at three hidden layers of 128 and batch size 8192.

Its additional diagnostics use a fresh simulation bank after the frozen
selection: reweighting plots with ratio panels, balanced-class calibration with
residual panels, and independent denominator-mean convergence toward one for all
five ratio tasks. The bank is prepared before training. Each ensemble member
immediately displays raw reweighting/calibration plots before the next member
starts; the completed task then displays raw and mean-normalized ensemble
checks. Per-member and ensemble PDF/PNG/CSV files have separate directories.
Independent expectation and high-statistics Poisson likelihood scans compare
analytical and learned MLEs at generating mu=1, with NI fixed and profiled.
The bank never normalizes the fitted model. Analytical shifts therefore also
expose finite original-workspace integration errors. Figures use mplhep styling
without an ATLAS label and save PDF/PNG plus numerical CSVs.
Notebook controls default to two million generated events per source for the
bank and ten million expected inclusive events for the Poisson experiment;
selected counts and exposure are reported. These diagnostic budgets can change
without changing a completed v4 run name or its networks. After the initial
v4 retraining, matching checkpoints are reused and diagnostics can be rerun
without retraining or recalibration.

The physical expected-yield exposure is separate from Monte Carlo counts;
there are still six generated samples and five ratio-training tasks. Use the
new run name `paper-distinct-s-v4` and rerun generation and training. Earlier
runs remain available. Notebook output cells are cleared to avoid presenting
results from the previous training configuration under the new defaults, while user binning choices
in notebooks 02–09 are retained (the standard zoom lower bound is extended to 0.02). The scan now begins at 0.02 and the score scale
is 0.04 to cover the lower branch and avoid saturating its broader score.


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

For the distinct-signal benchmark the rate turnover is
$(-\lambda_I/(2\lambda_S))^2$. This and the equal-rate partner guide the
distribution plots, but shape information and NI profiling move the actual
likelihood extrema. The scan starts at 0.02 to include the lower secondary
branch; zoom controls in notebooks 03–09 also include that branch.
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

The cancellation between coherent components makes density-ratio precision more
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

## Bounded-ratio comparison in notebooks 6–7

These notebooks parallel notebooks 4–5 with the observable

$$
r_\eta(x)=\frac{p(x;\eta,0)}{p_S(x)}
=\frac{D_\eta(x)/\Lambda_\eta}{D_S(x)/\lambda_S},
\qquad z_\eta(x)=\frac{r_\eta(x)}{1+r_\eta(x)}.
$$

Both densities are normalized after the same frozen preselection, on the same
integration sample. The reference is the selected nominal S density used in
the unbinned analysis. All physical component yields are included in
$D_\eta$ and $\Lambda_\eta$. This is the ratio of normalized densities, not
the ratio of unnormalized event intensities. There is no `score_scale` factor.
The observable uses the exact analytical densities, matching notebook 4's
analytical score; learned-ratio errors remain visible in the separate unbinned
curves rather than being mixed into this compression comparison.

Notebook 6 uses the current notebook 4 resolution grid (4–60 bins in steps of
4), shows scans with 4, 12, 24, 36 and 60 bins, and includes the same stacked
distributions and fixed-truth estimator/width study. The estimator study uses
the saved run's finest bin count, as notebook 4 does. Notebook 7 uses 20 bins,
the same spline anchors, NI rate/shape interpolation, and unit-Gaussian
constraint as notebook 5. Every observable is frozen at its tested anchor
throughout both likelihood fits, and the analytical Asimov data are re-binned
at each anchor.

A monotonic function of $r_\eta$ preserves the event ordering for the binary
comparison with the S reference. It need not retain the local Fisher
information for varying physical $\mu$, nor all information needed for NI
profiling. The Fisher fractions and likelihood comparisons measure this loss;
they do not assume the score's local optimality theorem applies to the ratio.

Ratio-specific products use `results/ratio_*.csv`,
`results/ratio_spline_templates.npz` and `results/ratio_spline_choice.json`;
figures use `plots/06_*` and `plots/07_*`. They do not overwrite the score
histograms, splines, or estimator diagnostics. The samples, quadrature and
three unbinned reference curves are shared.

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
export POODEMO_ROOT=/absolute/path/to/poodemo-runs/paper-distinct-s-smoke-v4
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
python scripts/smoke.py --root /tmp/poodemo-distinct-s-smoke
```

The notebook generator produces clean notebooks. This intentional physics
revision clears earlier numerical outputs so that all nine notebooks can be
rerun consistently; targeted runtime fixes normally preserve user outputs. Toolkit
source is pinned to commit `fc09848fc6540fd32310faebbe9db6eea7ecd17b`;
`requirements-colab.txt` includes the dependencies its package metadata does
not declare. Compatible PyTorch ranges allow Colab's GPU-enabled build to be
reused. The experiment manifest records the toolkit commit and configuration.

See [VALIDATION.md](VALIDATION.md) for the current validation scope, runtime,
and numerical results. The automated checks cover amplitude algebra, sampling,
score derivatives, interpolation, normalization, the NI Gaussian constraint,
optimizer behavior, toolkit agreement, checkpoint reload, and cache safeguards.
The full 165-million-event generation and production ratio training are separate from the smoke checks.

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


## Physical-reference ratio comparison (08–09)

Run 08 then 09 in fresh runtimes using the same completed `RUN_NAME`. No new
samples or training are required. Notebook 08 mirrors 06's current binning and
plot controls; 09 uses 20 bins. Results have the `reference_ratio_` prefix and
figures use `08_`/`09_`, leaving score and S-reference studies intact.

The analytical observable is `R/(1+R)`, where
`R = r(x;eta)/r(x;1) = p_selected(x;eta,0)/p_selected(x;1,0)`.
It preserves the continuous pairwise comparison with truth 1 at fixed NI.
Finite uniform binning and NI profiling have no corresponding optimality
guarantee. At eta=1 it is exactly 1/2: the frozen experiment retains only rate
information. Notebook 08 explicitly diagnoses the collapse. Notebook 09 keeps
the original PCHIP study and highlights its inability to reproduce discontinuous
bin yields across that anchor; inspect off-anchor interpolation errors.

Rebuild only these new notebooks with `python scripts/build_reference_notebooks.py`.
The builder reads the current 06/07 cell sources and clears outputs in 08/09;
it does not rewrite 01–07.
