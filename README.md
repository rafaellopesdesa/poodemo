# Parameterized optimal observables: an interference toy

The Google Colab notebook series connects analytical quantum amplitudes, ATLAS-style
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
| [08_reference_ratio_histograms.ipynb](notebooks/08_reference_ratio_histograms.ipynb) | Generate 10M selected events per source in Colab; validate no-floor learned histograms and MC convergence | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/08_reference_ratio_histograms.ipynb) |
| [09_reference_ratio_spline_templates.ipynb](notebooks/09_reference_ratio_spline_templates.ipynb) | Validate empirical CDFs in $u$; study the floor, binning and nuisance morph order | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/09_reference_ratio_spline_templates.ipynb) |
| [10_discriminator_histograms.ipynb](notebooks/10_discriminator_histograms.ipynb) | Compare fixed 36/36²/36³ process-ratio histograms with the moving 36-bin likelihood ratio | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/10_discriminator_histograms.ipynb) |
| [11_test_statistic_toys.ipynb](notebooks/11_test_statistic_toys.ipynb) | Compare stat-only likelihood-ratio distributions for simulator and model toys at mu=0 and 1.4 | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/11_test_statistic_toys.ipynb) |
| [12_toy_diagnostics.ipynb](notebooks/12_toy_diagnostics.ipynb) | Investigate branch switching, fit bounds, finite integration error, bin refinement, and coverage uncertainty in notebook 11 | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/12_toy_diagnostics.ipynb) |
| [13_learned_ratio_compression.ipynb](notebooks/13_learned_ratio_compression.ipynb) | Compare learned and analytical compression with independent simulator-derived 36-bin yields | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/13_learned_ratio_compression.ipynb) |

## Run in Colab

1. The physical-reference workflow is **01–03 → 08**, followed by the studies
   in 09, 10 and 11. Notebook 12 retains its historical diagnostic inputs;
   its source is not automatically switched to the new floored ensemble. Notebook 11
   builds its own direct bin yields and does not require 09's splines.
   Notebooks 04–07 are retired; their earlier studies remain in the history.
   For a private repository, authorize GitHub access in Colab's notebook browser. If the link
   does not open, download the `.ipynb` and upload it to Colab.
2. Choose a GPU runtime for notebooks 2–3. Both PyTorch training and JAX
   likelihood fits use the GPU. `JAX_BACKEND = "auto"` detects an NVIDIA GPU;
   use `"gpu"` to require one, or `"cpu"` for an intentional CPU run.
   Notebooks 11–12 also benefit from a GPU for evaluating the saved networks on
   fresh simulator toys. The histogram and scalar fitting code runs on the CPU.
   Other histogram notebooks can use a CPU runtime.
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
$D_{SBI}=|A_S+A_B|^2$. Its integral is calculated analytically, including
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

Calibration curves default to 15 approximately equal-occupancy bins in the
equal-prior score mixture, with axes zoomed symmetrically around $0.5$ to cover
the observed scores and empirical error bars. The legend reports the usable
point count if tied scores or sparse class counts prevent 15 points. Residual
panels retain $[-0.01,0.01]$, and reweighting ratio panels retain $[0.95,1.05]$.
This changes diagnostic binning only; the predicted ratios are unchanged.

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


## Practical learned observables and MC templates (08–11)

Run **08** from the top with the completed
`RUN_NAME="paper-distinct-s-v4"`, the original `MODE`/`CONFIG_OVERRIDES`, and
notebook 03's Unity directory and filtering policy. Notebook 10 also requires
the completed 08 baseline; notebook 11 can run independently after 01–03.
Each downloads its pinned backend into a fresh
Colab checkout and clears cached package imports. No new training, event
transfer, or physics run name is required. The default bundle directory is
`My Drive/poodemo/unity_models/paper-distinct-s-v4-unity100/`.
Notebook 08 generates/reuses a 10M-selected-event campaign in Colab. Its final
cell exports a baseline manifest; notebooks 09 and 10 require that manifest and reuse
the exact same bank, rates, learned-map normalizers and yield-zeroing policy.

The notebooks verify and reuse **all five frozen member selections from 03**,
including its pointwise outlier filtering. They verify input and artifact hashes
without re-screening using analytical densities. If the bundle or policy
changes, complete a matching notebook-03 filter first. `RATIO_SOURCE="unity"`
selects the transferred bundle; `"drive"` selects the original Drive models.
Both require a matching completed filter. All new products live in
`practical_v1/results/` and `practical_v1/plots/` within the selected filter
instance; notebook 03 and historical 08–12 results remain separate. Notebook 08's
larger campaign has its own sample and histogram fingerprints beneath this tree.

### What supplies each input

| Input | Source |
|---|---|
| Moving reference-ratio observable | Filtered learned ratios to $S$ |
| Fixed NI, SBI and B coordinates | The same filtered learned ratios |
| Selected rates and histogram component templates | Labelled held-out simulator events and generated yields |
| NI nuisance anchors | Separate selected NI-down and NI-up simulation samples |
| Binned Asimov observations | The same nominal MC templates at the generating truth |
| External inference benchmarks | Analytical unbinned; 10 also overlays the saved filtered learned-unbinned physical-Asimov result from 03 |
| Toy physics | The original simulator, with frozen preselection |

No analytical PDF supplies a coordinate or a binned template. In particular,
this is more than the older optional learned-axis setting, which still used
analytical templates. Notebook 03 retains its analytical-versus-learned closure
studies; notebooks 12 and 13 are historical studies with their own definitions.

Let $\hat r_S=1$ and let $\hat r_j$ be the deployed learned process ratios. With
$Z_j=\langle\hat r_j\rangle_{S,\mathrm{MC}}$, define

$$
\hat f_\eta(x)=\sum_j c_j(\eta)\nu_j\frac{\hat r_j(x)}{Z_j},\qquad
\hat R_\eta(x)=\frac{\hat f_\eta(x)/\nu_\eta}{\hat f_1(x)/\nu_1},\qquad
z_\eta(x)=\operatorname{sigmoid}(100\log\hat R_\eta(x)).
$$

Here $c_j$ are the unchanged physical coefficients and $\nu_\eta=\sum_jc_j(\eta)\nu_j$.
The unknown common $S$ density cancels. Rates and NN mean normalizers are frozen
on the integration samples, never on observations, individual toy samples, or
fit parameters. They do not modify the saved network calibration artifacts.
The baseline is **36 bins, power 100, no floor** in all four notebooks.
At $\eta=1$ the coordinate is exactly $1/2$; the shared bin edges place it
strictly inside one bin. Physical hypotheses and fit domains remain unchanged.
Learned ratios do not have an exact sufficiency guarantee.

In 10–11, `SIMULATION_EVENTS_PER_SAMPLE=250_000` caps **selected** integration events used
per source. All selected events determine its acceptance. A retained event in
source $j$ receives weight $\nu_j/n_j$, where

$$
\nu_j=\lambda_j\frac{N_{j,\mathrm{selected}}}{N_{j,\mathrm{generated}}}.
$$

The MC bank records source counts, rates, effective statistics, input hashes and
filter identity. NN predictions are cached. Notebook 08 now generates fresh
selected samples instead of increasing this cap. Same-template Asimov closure does not test
independent MC closure, and the fits currently condition on the estimated MC
templates rather than profiling finite-MC uncertainty. Information fractions
above one can reflect MC noise, not a gain over the full likelihood.

NI is interpolated **after binning**. Strictly positive bins retain the existing
exp-poly rate/shape morph. Where any NI anchor is zero, shape interpolation is
linear between $-1,0,+1$ and holds its boundary shape beyond that interval;
the rate still follows exp-poly extrapolation. This documented sparse-support
convention introduces no pseudo-counts or positive floor. Such bins are counted
in diagnostics; nonregular profiled Fisher widths are marked invalid. More MC
or coarser bins may be necessary. In 08/09, tiny negative **total** bin predictions
are set to zero only if the largest correction is at most 0.01 expected events
and the sum of corrections is at most 0.05 events at that parameter point.
Larger deficits remain invalid. These opt-in limits never clip signed components,
and the same corrected total enters both Poisson terms. Raw values and corrections
are retained. A zero predicted mean with a positive observation still gives zero
likelihood. Clipped/nonregular Fisher points remain invalid. The practical studies do not
include an analytical morph-before-bin control.

### Notebook 08

Use a **Colab GPU** and Run all. `SELECTED_EVENTS_PER_SOURCE=10_000_000` targets
ten million events **after the frozen preselection for each of the six sources**
(60 million selected events total). This is independent inference simulation,
not additional training. The existing filtered ensembles are evaluated in
8192-event GPU batches, with 250k-event selected chunks saved on Drive.

Generation, neural predictions and compact histogram counts resume after a
runtime interruption. Keep the seed, chunk and inference-batch controls fixed
when resuming; changing them creates a distinct cache. Increase
`SELECTED_EVENTS_PER_SOURCE` to extend the same streams. Three raw learned ratios
and selected coordinates are stored in chunks, requiring about 2.2 GB on Drive
at the default target; local model staging uses additional Colab disk. No array
containing all 60M events is allocated. No cluster job is submitted.

The 250k, 1M and 10M nested prefixes share frozen selected process yields and
learned-map normalizers from the original integration samples. Weights are
$w_j=\nu_j/N$, so larger MC statistics do not increase experimental exposure.
This controls **shape MC precision**; the original acceptance-rate uncertainty
is unchanged. The selected-target stopping count is never used to estimate
acceptance. Predictions use learned ratios; analytical densities only enter the
simulator and the explicit analytical unbinned benchmark.

Before fitting, the notebook reports negative physical yields and nonregular
points at truth, plus per-bin MC errors. Physical minima are checked exactly in
$\sqrt{\mu}$ at a declared NI grid; nuisance behaviour between those grid points
is not certified. Raw domain minima and bounded-zeroing corrections are reported
separately. No bins are merged. The baseline remains 36 bins, power 100, and no
observable-eta floor. Convergence plots include
both fixed and profiled NI, with invalid fits marked explicitly.

Paired controls compare powers 50/100/150 on the full cached bank, focused on
low strengths and the secondary minimum. They report endpoint-bin occupancy,
exact numerical saturation and likelihood scans. Power 100 remains the baseline;
these controls do not automatically tune against the analytical reference.
Counts checkpoint after each chunk; neither generation nor NN evaluation repeats.

Cache locations are printed. Selected samples and predictions live under
`practical_v1/results/colab_mc/<sample fingerprint>/`; histogram counts and new
results/plots live under its `histograms/<histogram fingerprint>/` directory,
with a separate `inference_<policy fingerprint>/` for each zeroing policy.
The final cell publishes `practical_v1/results/colab_08_baseline.json` for 09.

Inspect learned-coordinate distributions, the exact equality collapse, bin-count
scans, and local-width comparisons at fixed truth. The sole oracle overlay is
**analytical unbinned**. The no-floor baseline and explicit $0.2$ floor control
share the same MC bank and physical fit parameters. Direct Asimov scans check
that a feasible truth reaches the nonnegative NLL lower bound, which certifies
one unrestricted MLE; it does not establish uniqueness of the interference
branch. Local-width estimator fits still report their optimizer results.

### Notebook 09

Only **exact weighted empirical CDF in $u$** (`ecdf_u`, study version 5) remains:

$$
u_\eta(x)=\frac{\log\hat R_\eta(x)}{\eta-1},\qquad
u_1(x)=\left.\partial_\eta\log\hat p_\eta(x)\right|_{\eta=1}.
$$

Stable algebra evaluates the score limit. Raw MC cumulative weights, grouped
ties and correctly ordered boundaries reproduce direct fills of **the same
learned map** at anchors. Convex linear interpolation in $\eta$ uses the original
$z$ bin boundaries mapped into $u$. It preserves component totals and the exact
equality collapse without smoothing or changing component templates. The shared
bounded-zeroing policy is applied only to the final physical mixture. Raw component
and mixture interpolation errors remain available before that correction.

Run 08 through its final cell first. Notebook 09 checks its completed manifest
against the selected ensembles and compression/zeroing settings, then stages
about 1.4 GB of cached ratios on Colab local disk. It builds source CDFs separately;
two anchors need about 1 GB of coordinate arrays at 10M events per source.
No dense 60M-event/component matrix is created. Sorting/refinement can take
appreciable CPU time, but no simulation, neural inference or training is required.

Refinement uses template and likelihood checks on predeclared optimization
points. Reserved points then test interpolation independently. The existing
4,801-anchor/12-round limits and $|\Delta q|\leq0.01$ target remain; hitting a
budget is not acceptance. `direct_physical_failures` distinguishes invalid
finite-MC physical probes from interpolation errors. Separate direct studies
compare floors 0/0.2 and 36/144 bins. They use only MC morph-after-bin models.
The analytical unbinned curve remains the reference. Products are under
`reference_ratio_interpolation/` and `reference_ratio_floor_study/` in the new
evaluation; two full-event CDFs are cached to bound memory.

### Notebook 10

Compare fixed NI, NI × SBI, and NI × SBI × B coordinates against the moving
learned reference ratio. Each fixed axis has 36 bins (36, 1,296, 46,656 cells),
with powers $a_{NI}=1$, $a_{SBI}=a_B=2$. Run the updated **08 through its final
cell first**. Notebook 10 reuses that completed 10M-selected-event-per-source
bank, frozen rates/normalizers and filtered learned ratios. It stages cached
ratios and fills fixed histograms in chunks with resumable count checkpoints.
It never generates events, evaluates networks or increases the sample size;
10M selected events per source is a hard ceiling. The moving 36-bin, $a=100$,
no-floor histograms are loaded directly from 08 and checked against its scans.
A recap fills **36, 72, 108 and 144 moving bins** from the same cached ratios,
keeping 36 bins as the primary baseline. Every likelihood/information panel
includes analytical unbinned and the filtered notebook-03 learned-unbinned
physical-Asimov comparison (not its learned-model self-Asimov curve). The
learned reference retains 03's original quadrature normalization and rates;
its information line is local objective curvature at the common generating
truth, with a finite-difference step check, not a coverage guarantee.
NI morphing and bounded total-yield zeroing are identical to 08/09. The
comparison retains fixed/profiled NI, invalid-fit reports and local information.

A factorial study crosses **9, 18 and 36 bins per fixed 3D coordinate** with
**2.5M, 5M and 10M events per source**, all from the existing cached streams.
The coarse grids merge adjacent cells of the original 36-bin grid, retaining
its boundaries. Rates and ratio normalizers stay frozen; each case has a
same-template Asimov sample at the same physical exposure. Full/zoomed scans,
differences from analytical unbinned, and $q$ versus inverse MC size include
both unbinned references, with NI fixed and profiled. Prefixes are correlated;
no independent-point fit, infinite-MC extrapolation or bias subtraction is made.
NI anchors are merged before morphing, so the profiled comparison also tests
that approximation. The full-prefix fine-grid scan must reproduce the main
study. Failures and bounded-zeroing corrections remain visible.
The scans also identify nonpositive nominal predictions with positive Asimov
counts: a permitted tiny zeroing can still imply an infinite Poisson penalty,
and a profiled fit may shift NI to escape that boundary. Plot markers distinguish
invalid fits from valid fits with this nominal-support warning.

Additional nominal-NI tables record the per-cell hypothesis contrast, its
MC uncertainty with shared-template correlations, the raw Poisson contribution
to $q$, and a second-order MC-bias diagnostic on positive-yield cells. A scatter
plot near the secondary region compares contrast size with its MC uncertainty.
These are diagnostic quantities, not corrections or bin-selection rules.
New products are under `3d_convergence_v1/<configuration fingerprint>/` within
the notebook-10 output directory; count caches can resume interrupted fills.

Per-bin MC precision plots show raw total-yield relative uncertainties at
$\mu=0.02,0.054,0.122,1$ and every moving scan point at all four recap resolutions. Every fixed 3D cell appears
in labelled B-coordinate slices. Six-source tables include nominal/up/down NI;
`SOURCE_3D_PLOTS` optionally adds source-specific 3D atlases. Errors use
fixed-sample multinomial marginal variances, conditional on frozen rates and
the learned map, with squared physical coefficients for independent sources.
They exclude rate and neural-model uncertainty. Empty sampled cells and
nonpositive raw totals are explicitly marked; zeroing never enters the
uncertainty denominator. These diagnostics do not add MC nuisance parameters,
adaptively merge cells, change powers or trigger additional events. Full per-cell tables
are compressed CSVs and raw six-source counts are saved in `source_counts.npz`.
Products are in `results/10_colab_10M_precision_v1/` beneath the completed 08
study directory, with PNG/PDF plots under the corresponding `plots/` directory.
Regenerate only this notebook with `python scripts/build_colab_10.py`.

### Notebook 11

Retain five stat-only comparisons: analytical unbinned and learned unbinned on
paired fresh simulator experiments, learned unbinned on learned-bank toys, and
MC histograms on simulator and binned-model toys. The two-sided statistic is
$q_{\mu_0}=-2\log[L(\mu_0)/L(\hat\mu)]$, with hypotheses 0 and 1.4, physical fit
bounds $[0,3]$, and a 129-point $\sqrt\mu$ grid followed by local refinement.
Freeze the learned coordinate at each tested point throughout both fits.
Histogram simulator events once, without a second Poisson fluctuation.

The learned likelihood uses fields relative to $S$; that unknown common PDF
cancels from the statistic. Its own toys Poisson-resample an importance-weighted
S simulation bank, a finite-bank approximation rather than a continuous neural
sampler. All normalizers remain frozen. Binned-model toys use Poisson counts
from the direct MC bin means and require no notebook-09 interpolation products.
The default is 5,000 toys per hypothesis and case (12 in smoke mode). Increasing
`N_TOYS` resumes deterministic streams; changed templates, model identity,
normalizers or other scientific settings reject old checkpoints.

Even self-model toy IDs estimate critical values and odd IDs test acceptance.
Failures remain in coverage denominators/bounds. Errors condition on estimated
critical values. New products use `toy_learned_mc_36_no_floor`; notebook 12's old
tags and inputs are preserved. Do not repoint it at these products without a
separate diagnostic update.

Builders: `scripts/build_reference_notebooks.py` updates 08/09;
`scripts/build_reference_spline_notebook.py` updates 09 only;
`scripts/build_comparison_notebooks.py` updates 10/11. They clear stale notebook
outputs and preserve user run/model controls while applying the new baseline.

## Learned compression with simulator-derived yields (13)

Run **only notebook 13** in a fresh runtime with the completed v4 `RUN_NAME`,
`MODE`, and `CONFIG_OVERRIDES` from 01–03. It reuses the saved selector and
ratio networks without training. It does not require completed 08–12 outputs.
A GPU accelerates network evaluation; numerical likelihood fits run on CPU.
Build this notebook alone with `python scripts/build_learned_compression_notebook.py`.

The study repeats the principal reference-ratio comparison with two maps:
analytical ratios and the deployed learned ratios. Both use **36 bins**,
**`REFERENCE_RATIO_ETA_MIN=0.2`**, **`REFERENCE_RATIO_A=100`**, denominator mu=1,
and the existing edges that put 0.5 inside a bin. The original notebook-03
workspace fixes the learned normalizations and the selected-rate constants
used in both maps. They are never refitted on the new simulation banks.
Learned fields determine bin assignments only; every nominal and NI-varied
bin prediction comes from physical simulator intensities.

Two independently generated banks use the frozen preselection, with equal
numbers of generated S/B/NI proposal events before selection. Importance
weights account for rejected events and retain the simulated absolute yields.
The main **matched-bank** Asimov comparison uses bank A for both predictions
and observations, isolating compression and finite binning. A second
**independent-bank** comparison keeps those predictions and maps fixed while
using bank B for the weighted Asimov observations. It tests finite-MC template
closure; its counts are not rescaled to force the generating value to fit.
Neither comparison is a pseudo-experiment or a coverage measurement.

The three likelihood descriptions are analytical unbinned, analytical-map
histograms, and learned-map histograms. Each hypothesis freezes its observable
before the conditional and unrestricted fits. The physical mu scan, component
coefficients, truth, and fit bounds are unchanged by the observable floor.
Analytical unbinned references are recomputed on these banks rather than
reusing notebook-03 curves from a different integration sample. NI-profiled
histograms morph the integrated nuisance anchors, as in notebook 09, so their
comparison with unbinned profiling also includes that interpolation convention.

Inspect scan overlays and differences, local-information diagnostics, and
independent-bank bin-yield closure. Negative or nonfinite learned physical
intensities are reported as invalid maps rather than silently clipped or
removed. Unsupported bins in the independent closure likewise remain visible;
no artificial counts are inserted. Simulation errors describe the fixed
trained maps and banks, not network-training uncertainty.

New products use `results/13_learned_compression_36_floor02` and corresponding
`plots/13_learned_compression_36_floor02` figures. Cached banks carry provenance
for their physics, selection, networks, frozen normalizers, seeds and generated
counts. Changing those dependencies requires a new output tag. Earlier
notebooks, their execution outputs, and their study products are preserved.

## Investigating toy discrepancies (12)

Run **only notebook 12**, in a fresh runtime, with the same completed v4 run,
mode and configuration overrides. It reads notebook 11's direct-yield
`SOURCE_TAG="toy_study_direct"` products; it does not extend that ensemble or
retrain the networks. Build this notebook alone with
`python scripts/build_toy_diagnostics_notebook.py`.

The saved-table stage pairs analytical, learned and binned fits to each physical
toy, separates local and remote branches, and bootstraps coverage while
re-estimating the model-derived critical values. Simulator fits are resampled
together; independent toy sources stay independent. Even calibration IDs and
odd evaluation IDs remain disjoint. Failed and missing fits are reported.

The integration stage generates independent stratified S/B/NI proposals, with
1, 2 and 5 times the original generated count per source and two independent
replicas. It tests process and physical bin yields, total rates, and the expected
score in `kappa=sqrt(mu)`, particularly at zero. Networks, observable boundaries,
and reference normalizers remain frozen. Rejected proposals enter the MC
variance as zeros; process and bin correlations are retained where needed.
Errors on closure residuals are conditional on the original prediction and
partition, rather than a claimed exact combined old-plus-new MC pull.

Refits reconstruct the same simulator toys from their recorded seeds. Defaults
use 200 randomly selected toys per hypothesis plus all upper-bound cases,
compare the original and denser 129-point kappa searches, widen the upper mu
bound from 2 to 3, and compare 12 bins with a nested 36-bin refinement. Fresh
integration predictions are tested on the same events. Original-fit reproduction
is checked first; population summaries use only the random subset, since the
flagged subset is deliberately enriched. The denser grid remains a numerical
search, not a proof that every possible minimum has been found.

Both heavy stages can be disabled for a quick first look. Intermediate MC
prefixes and complete refit toys are checkpointed; changed settings or source
inputs require a new diagnostic `OUTPUT_TAG`. Tables/manifests and figures go to
`results/12_toy_diagnostics/<OUTPUT_TAG>` and
`plots/12_toy_diagnostics/<OUTPUT_TAG>`, separately from notebook 11. No model or
run tag changes are needed. These checks distinguish calibration and numerical
issues from differences in information: matching the analytical distribution
of q is not by itself a coverage criterion.

Notebook 12 also has an appended **section 6: integration convergence**. After
updating the checkout in a fresh Colab runtime, run the original setup and
common-import cells, then jump directly to section 6. Keep the same `RUN_NAME`,
`MODE`, `CONFIG_OVERRIDES`, `SOURCE_TAG` and original diagnostic `OUTPUT_TAG`.
The previous notebook 12 sections do not need to be rerun.

This section reloads the exact saved refit selection and both integration
replicas. It uses all completed prefixes (normally 1×, 2× and 5× the diagnostic
base budget), then continues each replica's saved RNG stream to 10× by default.
For the standard 250,000-event base, the new largest bank has 2.5 million
**generated** proposals per S/B/NI source per replica. Earlier integration
checkpoints are reused, and extending the banks evaluates only the frozen
preselection and analytical densities, without reevaluating ratio networks on
new bank points.

Each reconstructed physical toy is shared by every bank comparison. Analytical
unbinned rates and 12/36-bin yields are changed together while the observable,
its normalization, bin boundaries, fit range and optimizer remain fixed.
Convergence plots compare mean q, paired binned-minus-analytical differences,
branch and zero-q probabilities, and changes between bank prefixes and replicas.
Population summaries use the original random controls; flagged experiments
remain diagnostic examples. Error bars from the paired toy sample condition on
the chosen banks. Correlated prefixes and two independent replicas do not define
a full finite-MC uncertainty model or certify convergence automatically.

New outputs are under the original diagnostic directories in
`convergence/<CONVERGENCE_TAG>/`, with default
`CONVERGENCE_TAG="bank_convergence"`. Completed refits resume. To change the new
study's settings, choose a new `CONVERGENCE_TAG`, while retaining the original
`OUTPUT_TAG` so its saved banks and toy selection can be found. The notebook
builder appends this section without replacing earlier executed cells.

**Section 7: likelihood decomposition** reuses the completed section 6 run.
In a fresh runtime with updated source, execute the first two setup/import code
cells, then jump directly to section 7. Keep the same run configuration,
`SOURCE_TAG`, `OUTPUT_TAG` and `CONVERGENCE_TAG`. No earlier sections, training,
integration extension or fits need to be repeated.

With the observable frozen at the tested hypothesis eta and reference mu=1,
define `D(mu) = -2 log[L(mu)/L(1)]`. The paired difference splits exactly as
`delta_q = delta_D(eta) - delta_D(mu_hat)`, where each model uses its own saved
best fit. The first term tests the fixed hypothesis comparison; the second
measures the change in the freely fitted minimum. Both are evaluated directly
and checked against the saved q. The same bank supplies analytical rates and
binned yields, so the fixed-pair rate contribution cancels within each pair.

Defaults compare the two largest completed integration prefixes, both replicas,
12/36 bins and the original-bank control. Only the existing selected simulator
experiments are reconstructed; no optimizer or ratio-network evaluation is
needed. Read-only bank loading rejects missing section 6 checkpoints. The new
results authenticate the saved context, predictions, fits and event coordinates.
Per-toy tables retain flagged experiments and failures, while population means,
RMS values, paired bank changes and branch comparisons use random controls.
Reported standard errors condition on the banks and explicitly counted valid
fits. Signed decomposition plots use linear axes. This diagnoses compression
and numerical sensitivity; it does not recalibrate coverage.

New tables/checkpoints are saved below
`results/12_toy_diagnostics/<OUTPUT_TAG>/convergence/<CONVERGENCE_TAG>/decomposition/<DECOMPOSITION_TAG>`
with matching plot directories. The default `DECOMPOSITION_TAG="fixed_pair"`
resumes unchanged settings; choose a new decomposition tag if changing this
diagnostic. The builder appends section 7 while preserving all executed cells
and outputs above it.

**Section 8: full-family control and branch preference** adds analytical
`u=I/(B+NI)` and `v=S/(B+NI)` coordinates, whose unbinned likelihood ratios
retain the complete stat-only family. It checks that identity, then compares
12x12 and 36x36 histograms with the existing analytical and scalar 12/36-bin
models on the same saved toys. Axis boundaries are nested, parameter-independent
quantiles of the original bank weighted by `B+NI`, with infinite outer edges.

Run the first two setup/import code cells in a fresh runtime and jump directly
to section 8; keep the completed section 6 tags. The new `FAMILY_TAG="family_control"`
namespace supports resume. Rebinning requires replaying the saved integration
streams through frozen preselection (15 million proposals for the standard two
10x banks). Chunk/checkpoint boundaries are reproduced, and rates and scalar
predictions must match before the 2D templates are accepted. No ratio-network
inference or training is needed. Empty observed cells with zero prediction
invalidate the corresponding fit; yields are never floored to hide missing MC
support. Support tables and heatmaps report sparsity and integration precision.

For every model, separate constrained low/high fits measure
`G = min_high D - min_low D`, where negative values favor the high branch.
Evaluating each model at the same two analytical minima separates changes in
relative evidence from movement of the minima during refitting. Population
summaries use saved random controls; representative scans deliberately include
branch-switching examples. Ambiguous ties, split-boundary minima, failed fits
and bank dependence remain explicit. The 2D control diagnoses full-family
compression; its finite histograms still require binning and MC convergence.

Outputs are below
`results/12_toy_diagnostics/<OUTPUT_TAG>/convergence/<CONVERGENCE_TAG>/family/<FAMILY_TAG>`
with matching plot directories. All earlier notebook outputs and notebooks 08–11
are preserved. This study retains the scalar observable's fixed reference mu=1;
MLE-based reference selection and coverage calibration are separate tests.

**Section 9: per-toy unbinned-MLE reference** tests the first of those follow-ups.
In a fresh runtime with updated source, run the first two setup/import code
cells and jump directly to section 9. Keep the completed run and section 6
tags; no earlier section needs to be rerun. The default new
`REFERENCE_TAG="mle_reference"` keeps its cache and fits separate.

For each saved toy and bank, fit the analytical unbinned likelihood and replace
the observable's reference mu=1 by that analytical MLE. Freeze the resulting
mapping throughout the subsequent unconstrained binned fit, searching both
branches and explicitly including the analytical MLE as a candidate. Compare
fixed and MLE references using the same 12/36-bin edges, power and original-bank
density normalizers. Defaults use both replicas at their largest completed
prefix, with the same random controls and flagged examples. A changed reference
is not a substitute for the free binned fit; both mappings are freely fitted.

The new event-level cache replays the original integration streams once,
authenticating counts, RNG states, rates and fixed-reference moments. It stores
importance-weighted analytical component fields for rebinning at arbitrary
references, requiring additional disk space. No new integration stream or
ratio-network inference on new points is introduced. Cache construction and
complete toys resume. To include the previous bank prefix, set
`REFERENCE_LARGEST_PREFIXES=2` and choose a new reference tag.

The exact decomposition is `delta_q = delta_pair + refit_gain`, where
`delta_pair = D_bin(eta) - D_bin(mu_hat_analytic) - q_analytic`, and
`refit_gain = D_bin(mu_hat_analytic) - D_bin(mu_hat_bin) >= 0`.
The pair statistic remains signed. Exact eta=reference cases map to 0.5 and
are reported separately, with a one-bin yield-only closure check; their free
fit can still give positive q. Reference separation, occupied observed bins,
positive predicted bins, unsupported observations, branch changes and selected
full scans diagnose remaining discrepancies. No reference jitter, prediction
floors or forced zero-q values are applied.

The MLE reference is computed separately in each bank, so bank contrasts include
both changes in yield predictions and movement of the reference. Data-dependent
partitions define a test-statistic algorithm: any future calibration must repeat
the complete unbinned fit, mapping, histogramming and free binned fit for every
toy. This diagnostic does not establish coverage. Outputs are below
`results/12_toy_diagnostics/<OUTPUT_TAG>/convergence/<CONVERGENCE_TAG>/mle_reference/<REFERENCE_TAG>`
with matching plot directories. Earlier executed cells and notebooks 08–11 are
preserved; no model or run tag changes are needed.

**Section 10: shifted ratio endpoints** compares the unshifted observable with
numerator-only, denominator-only and simultaneous shifts. Run the first two
setup/import code cells in a fresh runtime with updated source, then jump to
section 10. No rerun of sections 1–9 or notebooks 01–11 is required. Keep
`CACHE_REFERENCE_TAG="mle_reference"` and `SHIFT_LARGEST_PREFIXES=1` to reuse
the default section-9 event-level integration cache. If that cache is absent,
the same saved streams are replayed to build it; the section-9 fits are not
needed. New results use the separate `SHIFT_TAG="shifted_reference"`.

With `NUMERATOR_SHIFT=DENOMINATOR_SHIFT=0.1`, the endpoint pairs are
`(eta,1)`, `(eta+0.1,1)`, `(eta,1.1)`, and `(eta+0.1,1.1)`. The shifts affect
only the observable: the physical toy/test hypotheses remain eta=0 and 1.4,
and the predicted yields are freely fitted over the saved physical mu range
and both branches. These use analytical component ratios to isolate compression
effects, the original density normalizers and power, and the saved 12/36-bin
edges. Templates are reused across toys for each fixed mapping and bank.

The same saved random controls provide paired comparisons with analytical
unbinned results and direct contrasts against the unshifted mapping. Inspect
RMS q differences, changes in absolute/squared error, branch agreement,
ambiguities, unsupported observations, and the signed pair-error/free-fit-gain
decomposition. Log-scale q distributions, occupancy plots and selected full
likelihood scans accompany the tables. Two bank replicas reuse the same toys;
their errors are conditional toy errors, not independent extra toy statistics.
Changing endpoints can alter both retained information and finite-bin occupancy.
This predefined comparison does not optimize offsets on the toys or establish
coverage. Change `SHIFT_TAG` if changing shifts or fit settings.

Outputs are below
`results/12_toy_diagnostics/<OUTPUT_TAG>/convergence/<CONVERGENCE_TAG>/shifted_reference/<SHIFT_TAG>`
with matching plot directories. Earlier executed notebook cells and the model
and run tags are preserved.

### Large ensembles on UMass Unity

See [unity/README.md](unity/README.md) for frozen Drive inputs, selected-event
sample generation, 100-member GPU Slurm arrays for all five ratio tasks,
resumable training, model collection for SCP transfer to Google Drive, and
notebook 03 evaluation on the existing Drive samples. Only inference models and metadata return from Unity;
Colab computes the event weights again.
