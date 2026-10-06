# Parameterized optimal observables: an interference toy

Five Google Colab notebooks connect analytical quantum amplitudes, ATLAS-style
neural simulation-based inference, and a parameterized binned approximation
built from the likelihood score. Data and trained models persist in your Google
Drive; the source lives in this repository.

| Notebook | Purpose | Open in Colab |
|---|---|---|
| [01_generate_amplitudes.ipynb](notebooks/01_generate_amplitudes.ipynb) | Generate coherent S/B amplitudes, SBI and NI samples, and nuisance variations | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/01_generate_amplitudes.ipynb) |
| [02_preselection.ipynb](notebooks/02_preselection.ipynb) | Train S/B/NI multiclass preselection and select approximately S/NI = 0.1 | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/02_preselection.ipynb) |
| [03_unbinned_nsbi.ipynb](notebooks/03_unbinned_nsbi.ipynb) | Train density ratios, build workspaces, and compare analytical and learned Asimov fits | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/03_unbinned_nsbi.ipynb) |
| [04_score_histograms.ipynb](notebooks/04_score_histograms.ipynb) | Compute the analytical score and study progressively finer Poisson histograms | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/04_score_histograms.ipynb) |
| [05_spline_templates.ipynb](notebooks/05_spline_templates.ipynb) | Interpolate moving bin fractions, validate splines, and compare profiled scans | [Open](https://colab.research.google.com/github/rafaellopesdesa/poodemo/blob/main/notebooks/05_spline_templates.ipynb) |

## Run in Colab

1. Open notebook 1, then run notebooks in numerical order. For a private
   repository, authorize GitHub access in Colab's notebook browser. If the link
   does not open, download the `.ipynb` and upload it to Colab.
2. Choose a GPU runtime for notebooks 2–3. The other notebooks can use a CPU.
3. In each notebook use the same `RUN_NAME` and `MODE`. Defaults are
   `RUN_NAME = "demo-v1"` and `MODE = "production"`. For an initial short check,
   choose `MODE = "smoke"` and a distinct run name such as `smoke-v1`.
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
Selected-data and quadrature caches record the selector fingerprint and reject
stale reuse after its model or threshold changes.
Use a new run name when changing the configuration: incompatible saved settings
are rejected rather than silently mixed.

Production generates **5 million events in each of ten samples**: S, SBI, B,
NI, B±, SBI±, and NI±. The raw three-coordinate float32 arrays occupy about
600 MB; selected arrays, network ensembles, cached quadrature, and workspaces
need additional space. Data are generated and evaluated in chunks. Training
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
\cos\varphi=-0.65.
$$

These are Gaussian wavefunctions. The expected inclusive yields are
$\lambda_S=100$, $\lambda_B=1000$, and $\lambda_{NI}=10000$.
Means and correlated covariance matrices differ between processes and are
recorded in `run.json`. The SBI template is a positive process with intensity
\(D_{SBI}=|A_S+A_B|^2\). Its integral is calculated analytically, including
interference; the default is approximately 820.26 events. NI is incoherent.

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

A B shape nuisance shifts its mean by
$\pm0.1\|m_B\|v_B$, where $v_B$ is a fixed, seeded random unit vector.
SBI is regenerated coherently with that shifted B amplitude, including the
changed SBI yield. A second, independent mean-shift nuisance is added for NI
to support the varied NI histogram study in notebook 5. The nominal and ±1
anchors define an exponential–polynomial interpolation; that interpolation
is not assumed to be the exact continuously shifted amplitude between anchors.

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

For $D_\mu=\sum_j c_j(\mu)\lambda_jp_j$ and
$\Lambda_\mu=\sum_j c_j(\mu)\lambda_j$, notebook 4 uses

$$
o_\eta(x)=\left.\partial_\mu\log p(x;\mu)\right|_\eta
=\frac{D'_\eta(x)}{D_\eta(x)}
 -\frac{\Lambda'_\eta}{\Lambda_\eta},\qquad
z_\eta=\frac1{1+e^{-o_\eta}}.
$$

Every component includes its yield. The total-rate denominator contains only
yields, not a residual $p_{NI}(x)$ factor. Total-count information is retained
by the extended likelihood. Positive score anchors avoid the nonregular
$\sqrt\mu$ derivative at $\mu=0$.
The default scan covers $0.1\le\mu\le3$, while the fit domain is
$0.001\le\mu\le4$. The lower fit bound is a numerical floor for an interior
inference demonstration; testing the physical $\mu=0$ boundary needs a
separate construction.

When testing $\mu_0$, set $\eta=\mu_0$, fill its data and templates, and
**freeze that observable throughout both numerator and denominator fits**.
Spline interpolation is in the observable construction parameter $\eta$.
Physical \(\mu\)-dependence continues to use the exact coefficients above.
Exhaustive bins have process totals independent of \(\eta\), so interpolated
bin fractions are constrained to sum to one and multiplied by separate yields.

## What the comparisons establish

The notebooks keep several effects distinct:

- Analytical versus learned unbinned curves: density-ratio estimation.
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
export POODEMO_ROOT=/absolute/path/to/poodemo-runs/smoke-v1
export POODEMO_MODE=smoke
```

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

Notebook sources are maintained in `scripts/build_notebooks.py`:

```bash
python scripts/build_notebooks.py
python -m pytest
python scripts/smoke.py --root /tmp/poodemo-smoke
```

Notebooks are committed without execution output or credentials. Toolkit
source is pinned to commit `fc09848fc6540fd32310faebbe9db6eea7ecd17b`;
`requirements-colab.txt` includes the dependencies its package metadata does
not declare. Compatible PyTorch ranges allow Colab's GPU-enabled build to be
reused. The experiment manifest records the toolkit commit and configuration.

Validation: all five notebooks (33 code cells) were executed with the small
smoke configuration, including real toolkit training, workspaces, fits, and
all plot cells. The 30 automated tests cover the amplitude algebra, sampling,
score derivatives, interpolation, normalization, optimizer behavior, toolkit
agreement, checkpoint reload, and cache safeguards. See [VALIDATION.md](VALIDATION.md)
for the scope and numerical results. The full 5-million-event production
training has not been run as part of this validation.

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
