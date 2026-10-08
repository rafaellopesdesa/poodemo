"""Build clean Colab notebooks from reviewable, version-controlled cell sources."""
from pathlib import Path
import textwrap

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]


def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


SETUP_INTRO = r"""
### Runtime and persistent run

In Colab, choose **Runtime → Change runtime type → GPU** for notebooks 2–3.
Both PyTorch training and JAX likelihood fits use the GPU when available.
`JAX_BACKEND = "auto"` detects the Colab NVIDIA GPU; choose `"gpu"` to require it,
or `"cpu"` for an explicit CPU run. Setup installs matching JAX/JAXlib and CUDA 12
plugin packages, then checks double-precision JIT computation and gradients on
the selected device before training. GPU initialization failures are reported
instead of silently switching to CPU. Restart the runtime once when switching
from the previous CPU-only setup or after a JAX plugin error.
For a runtime-only repair, keep the same run name: saved samples and trained
networks are reused. A changed physics model requires a new run, as below.
The source checkout lives in the temporary runtime; **datasets, checkpoints,
workspace files, and results live in Google Drive**. Use the same `RUN_NAME`
and `MODE` in every notebook. Set `MODE = "smoke"` for a short workflow check;
production generates five million events each for S, SBI and B, and fifty
million each for NI, NI_up and NI_down.
Use `CONFIG_OVERRIDES` to change a computational budget without editing source,
for example `{"quadrature_per_process": 500_000, "epochs": 150}`. Use the same
overrides in all notebooks, and choose a new `RUN_NAME` when changing them.

This repository may be private. Grant Colab access when opening it from GitHub.
For the runtime download, either save a GitHub fine-grained token with **Contents:
read** access to this repository as the Colab secret `GITHUB_TOKEN`, or enter it
in the hidden prompt. The token is used only in a Python HTTP header and is not
written into files, Git remotes, shell arguments, or notebook output. Alternatively,
upload a repository ZIP to `/content/poodemo_source.zip`; no token is then needed.
An existing `/content/poodemo` checkout is reused. Delete that checkout to fetch
new source after updating the repository; your Drive run is separate.
For the enlarged NI bank and wider ratio classifiers, use the new default
`RUN_NAME = "paper-distinct-s-v4"`. Rerun notebooks 01–03; then rerun 04–09
to refresh their downstream results. The physics amplitudes are unchanged, but
the sample/training configuration is new (schema 7). Start a fresh Colab runtime
or refresh the temporary source checkout and restart it before proceeding.
Use the same new run name in every notebook; previous Drive runs remain intact.
"""

BOOTSTRAP = r'''
import os
import sys
from pathlib import Path

RUN_NAME = "paper-distinct-s-v4"
MODE = os.environ.get("POODEMO_MODE", "production")  # production or smoke
CONFIG_OVERRIDES = {}  # e.g. {"quadrature_per_process": 500_000, "epochs": 150}
JAX_BACKEND = os.environ.get("POODEMO_JAX_BACKEND", "auto")  # auto, gpu, or cpu

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    import subprocess
    requested_backend = JAX_BACKEND.strip().lower()
    if requested_backend not in ("auto", "gpu", "cpu"):
        raise ValueError("JAX_BACKEND must be auto, gpu, or cpu")
    selected_backend = requested_backend
    if requested_backend == "auto":
        try:
            gpu_probe = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=10)
            gpu_present = gpu_probe.returncode == 0 and bool(gpu_probe.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            gpu_present = False
        selected_backend = "gpu" if gpu_present else "cpu"
    # CUDA-only selection fails visibly if initialization fails. Keep PyTorch's
    # CUDA visibility and avoid JAX reserving most GPU memory before training.
    jax_platform = "cuda" if selected_backend == "gpu" else "cpu"
    os.environ["JAX_PLATFORMS"] = jax_platform
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    ROOT = Path(os.environ.get("POODEMO_ROOT", f"/content/drive/MyDrive/poodemo/runs/{RUN_NAME}"))
    REPO_DIR = Path("/content/poodemo")
    if not (REPO_DIR / "pyproject.toml").exists():
        import getpass
        import io
        import shutil
        import tarfile
        import tempfile
        import urllib.error
        import urllib.request
        import zipfile

        uploaded_zip = Path("/content/poodemo_source.zip")
        if uploaded_zip.exists():
            archive_bytes = uploaded_zip.read_bytes()
            archive_kind = "zip"
        else:
            archive_url = "https://api.github.com/repos/rafaellopesdesa/poodemo/tarball/main"

            def fetch_source(token=None):
                headers = {"Accept": "application/vnd.github+json", "User-Agent": "poodemo-colab"}
                if token:
                    headers["Authorization"] = "Bearer " + token
                request = urllib.request.Request(archive_url, headers=headers)
                with urllib.request.urlopen(request, timeout=180) as response:
                    return response.read()

            try:
                archive_bytes = fetch_source()
            except urllib.error.HTTPError as error:
                if error.code not in (401, 403, 404):
                    raise RuntimeError(f"Repository download failed (HTTP {error.code}).") from None
                token = None
                try:
                    from google.colab import userdata
                    token = userdata.get("GITHUB_TOKEN")
                except Exception:
                    pass
                if not token:
                    token = getpass.getpass("GitHub token with read access to poodemo: ").strip()
                if not token:
                    raise RuntimeError("Upload /content/poodemo_source.zip or provide read access.")
                try:
                    archive_bytes = fetch_source(token)
                except urllib.error.HTTPError as auth_error:
                    raise RuntimeError(f"Repository download failed (HTTP {auth_error.code}); check token access.") from None
                finally:
                    token = None
            archive_kind = "tar"

        # Validate archive paths and reject links before extracting the source.
        with tempfile.TemporaryDirectory(prefix="poodemo-source-") as temp:
            staging = Path(temp).resolve()

            def safe_destination(name):
                destination = (staging / name).resolve()
                if not destination.is_relative_to(staging):
                    raise RuntimeError("Unsafe path in source archive.")
                return destination

            if archive_kind == "zip":
                with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                    for entry in archive.infolist():
                        safe_destination(entry.filename)
                        if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                            raise RuntimeError("Symbolic links are not supported in the source ZIP.")
                    archive.extractall(staging)
            else:
                with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
                    for entry in archive.getmembers():
                        safe_destination(entry.name)
                        if not (entry.isfile() or entry.isdir()):
                            raise RuntimeError("Links and special files are not supported in the source archive.")
                    archive.extractall(staging)
            candidates = [p.parent for p in staging.rglob("pyproject.toml") if (p.parent / "poodemo").is_dir()]
            if len(candidates) != 1:
                raise RuntimeError("The uploaded archive must contain one poodemo repository checkout.")
            shutil.copytree(candidates[0], REPO_DIR, dirs_exist_ok=True)
        del archive_bytes

    if os.environ.get("POODEMO_SKIP_INSTALL") != "1":
        from importlib.metadata import version, PackageNotFoundError
        # JAX discovers every installed plugin, even when selecting CPU.
        # Retain already-matched CUDA 12 packages when rerunning GPU setup.
        remove_plugins = []
        for package in ("jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin", "jax-cuda13-pjrt"):
            try:
                installed_version = version(package)
            except PackageNotFoundError:
                continue
            if not (selected_backend == "gpu" and package.startswith("jax-cuda12-") and installed_version == "0.5.3"):
                remove_plugins.append(package)
        if remove_plugins:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", *remove_plugins])
        gpu_requirements = (["jax==0.5.3", "jaxlib==0.5.3",
                             "jax-cuda12-plugin[with-cuda]==0.5.3", "jax-cuda12-pjrt==0.5.3"]
                            if selected_backend == "gpu" else [])
        # Resolve with the base requirements so PyTorch's CUDA dependencies are
        # considered together with JAX's. Avoid blanket --upgrade changes.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r",
                               str(REPO_DIR / "requirements-colab.txt"), *gpu_requirements])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)])

        # Check the live notebook kernel before any expensive network training.
        import jax
        import jaxlib
        import jax.numpy as jnp
        if jax.__version__ != "0.5.3" or jaxlib.__version__ != "0.5.3":
            raise RuntimeError("JAX packages are stale in this kernel. Restart the runtime and rerun setup.")
        if selected_backend == "gpu":
            for package in ("jax-cuda12-plugin", "jax-cuda12-pjrt"):
                if version(package) != "0.5.3":
                    raise RuntimeError(f"{package} must match JAX 0.5.3. Restart the runtime and rerun setup.")
        jax.config.update("jax_platforms", jax_platform)
        jax.config.update("jax_enable_x64", True)
        try:
            jax_devices = jax.devices()
            if not jax_devices or any(device.platform != selected_backend for device in jax_devices):
                raise RuntimeError(f"Expected {selected_backend} devices; found {jax_devices}")
            probe_value, probe_gradient = jax.jit(jax.value_and_grad(lambda x: jnp.sum(x*x)))(
                jnp.asarray([1., 2., 3.], dtype=jnp.float64))
            probe_gradient.block_until_ready()
            if (float(probe_value) != 14. or list(probe_gradient) != [2., 4., 6.]
                    or probe_gradient.dtype != jnp.float64
                    or any(device.platform != selected_backend for device in probe_gradient.devices())):
                raise RuntimeError("JAX double-precision value/gradient check failed")
        except Exception as error:
            raise RuntimeError(
                f"JAX {selected_backend} startup failed. Select a Colab GPU runtime if requesting GPU, "
                "then restart the runtime and rerun setup. Use JAX_BACKEND='cpu' only for an intentional CPU run."
            ) from error
        print(f"JAX {jax.__version__} / jaxlib {jaxlib.__version__}: backend={selected_backend}, {jax_devices}; float64 JIT/gradient OK")

    # A new editable install is not activated in an already-running kernel.
    # Expose the inner package directly, including after a failed import cached
    # the outer /content/poodemo checkout as a namespace package.
    import importlib
    source_root = str(REPO_DIR.resolve())
    if source_root in sys.path:
        sys.path.remove(source_root)
    sys.path.insert(0, source_root)
    importlib.invalidate_caches()
    cached_package = sys.modules.get("poodemo")
    cached_file = getattr(cached_package, "__file__", None)
    expected_init = REPO_DIR / "poodemo" / "__init__.py"
    if cached_package is not None and (cached_file is None or Path(cached_file).resolve() != expected_init.resolve()):
        for module_name in list(sys.modules):
            if module_name == "poodemo" or module_name.startswith("poodemo."):
                del sys.modules[module_name]
else:
    # Locally: pip install -r requirements-colab.txt && pip install -e .
    ROOT = Path(os.environ.get("POODEMO_ROOT", str(Path.home() / "poodemo-runs" / RUN_NAME)))
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "poodemo" / "pipeline.py").exists():
            sys.path.insert(0, str(candidate))
            break

ROOT.mkdir(parents=True, exist_ok=True)
print(f"Run directory: {ROOT}")
print(f"Mode: {MODE}")
'''

COMMON_IMPORTS = r'''
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from poodemo.pipeline import create_run

run = create_run(ROOT, mode=MODE, overrides=CONFIG_OVERRIDES)
plt.rcParams.update({"figure.figsize": (7, 4.5), "axes.grid": True, "grid.alpha": 0.2})
display(pd.Series(run.config, name="configuration"))
'''

PLOT_HELPER = r'''
def plot_scans(frame, title, filename, group_columns=None, *, xlim=None, ylim=None):
    """Plot saved scans with optional (min, max) axis limits.

    Omit xlim/ylim to retain the default view. A None endpoint keeps that
    bound unchanged, for example ylim=(0, None) or xlim=(None, 1.2).
    """
    x_column = next((c for c in ["mu", "mu_test", "mu0"] if c in frame.columns), None)
    y_column = next((c for c in ["q", "t_mu", "test_statistic", "ts"] if c in frame.columns), None)
    if x_column is None or y_column is None:
        raise ValueError(f"Expected mu and q scan columns; found {list(frame.columns)}")
    if group_columns is None:
        group_columns = [c for c in ["method", "model", "systematics", "nuisances", "n_bins", "bins", "observable"] if c in frame.columns]
    groups = frame.groupby(group_columns, dropna=False, sort=False) if group_columns else [("scan", frame)]
    fig, ax = plt.subplots()
    for label, group in groups:
        group = group.sort_values(x_column)
        values = label if isinstance(label, tuple) else (label,)
        parts = []
        for column, value in zip(group_columns, values):
            if pd.isna(value):
                continue
            if column == "systematics":
                parts.append("profiled" if value else "stat only")
            elif column in ("n_bins", "bins"):
                parts.append(f"{int(value)} bins")
            else:
                parts.append(str(value))
        ax.plot(group[x_column], group[y_column], marker=".", ms=3, label=" | ".join(parts))
    ax.axhline(1.0, color="0.45", ls=":", lw=1)
    ax.set(xlabel=r"Tested signal strength $\mu_0$", ylabel=r"$t(\mu_0)$", title=title)
    ax.set_ylim(bottom=0)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = ROOT / "plots"
    output.mkdir(exist_ok=True)
    fig.savefig(output / filename, bbox_inches="tight")
    plt.show()
    return fig
'''


SCORE_DISTRIBUTIONS_INTRO = r'''
### How the observable changes with eta

These panels show expected event yields after preselection, with both the
observable constructed at \(\eta\) and the physical prediction evaluated at
\(\mu=\eta\), at nominal \(\alpha_{NI}=0\). For each bin, the coherent layer is
\[
\nu_{\mathrm{SBI},i}^{(\eta)}
=(\eta-\sqrt\eta)\nu_{S,i}(\eta)
+\sqrt\eta\,\nu_{SBI,i}(\eta)
+(1-\sqrt\eta)\nu_{B,i}(\eta).
\]
It is stacked above \(\nu_{NI,i}(\eta)\). Combine the three coherent terms
**before stacking**: their individual coefficients can be negative, while
their physical sum is nonnegative. The curves are not normalized to unit area.
NI's total yield stays fixed even though its distribution in \(z_\eta\) changes.

The default anchors include the generating value, the nominal total-rate
turnover, and the other equal-rate solution when those lie in the scan range.
These rate landmarks need not coincide with full-likelihood extrema.
Edit `ETA_VALUES` and `HIST_BINS` below to explore other choices. All panels
use the same bins and axes; `LOG_Y = True` exposes the tails (set it to `False`
for linear axes). After running setup, this cell can run independently
of the likelihood scans: it reuses the cached analytical quadrature without
retraining or refitting. The figure uses ATLAS-style formatting for this toy
model and is saved as PDF and PNG in the run's `plots` directory on Drive.
'''



UNBINNED_ASIMOV_REFRESH_INTRO = r'''
### Refresh the unbinned references from the saved networks

This cell reloads the existing ratio checkpoints and rebuilds the workspaces
and fits. It does **not** train any network. It refreshes three distinct curves:

| Curve | Likelihood model | Asimov event weights |
|---|---|---|
| `analytic unbinned` | Analytical intensity | Analytical intensity |
| `learned unbinned (analytic Asimov)` | Learned intensity | Analytical intensity: external closure check |
| `learned unbinned (model Asimov)` | Learned intensity | The same learned intensity: finite-sample model Asimov |

The construction follows the finite-sample Asimov prescription discussed in
[arXiv:2609.14136](https://arxiv.org/abs/2609.14136). Let $a_i$ be the saved
quadrature weights, so that $\int f(x)\,dx\approx\sum_i a_i f(x_i)$, and let

$$
\omega_i=a_i q_S(x_i),\qquad
Z_j=\sum_i\omega_i r_j^{\mathrm{NN}}(x_i),\qquad
\widehat p_j(x_i)=q_S(x_i)\frac{r_j^{\mathrm{NN}}(x_i)}{Z_j}.
$$

Here $q_S$ is the selected S reference density normalized on these nodes, so
$\sum_i \omega_i=1$. The $\omega_i$ are reference-density weights, distinct
from the integration weights $a_i$. The process ratios are already normalized using these **same
final quadrature points**; nuisance-morphed shapes are also normalized on those
points. Those existing normalizers are retained. The model Asimov instead
changes which intensity supplies the event weights:

$$
w_{A,i}=a_i\, \widehat D(x_i;\mu_A,0).
$$

With valid positive predictions and the expected rate evaluated on the same
weighted points, the learned model's finite Asimov likelihood satisfies

$$
-2\log\frac{L_A(\mu,\alpha_{NI})}{L_A(\mu_A,0)}
=2\sum_i a_i\left[\widehat D_i-\widehat D_{A,i}
-\widehat D_{A,i}\log\frac{\widehat D_i}{\widehat D_{A,i}}\right]
+\alpha_{NI}^2 \ge 0.
$$

Thus the generating point is a minimum even for a finite set of nodes and an
imperfect learned ratio. This does not guarantee a unique minimum, agreement
with the analytical curvature, or agreement in the secondary basin. The
learned-model Asimov is an internal consistency check; the analytical-Asimov
curve remains the external test of learned-model closure.

The analytical histogram data below are unchanged. Their full-dimensional
comparison is **`analytic unbinned`**. The two learned curves show separately
the effect of changing the model and of constructing its own finite Asimov data.
'''

UNBINNED_ASIMOV_REFRESH = r'''
from poodemo.pipeline import run_unbinned

# Reuse existing trained checkpoints; only rebuild normalization, workspaces, and fits.
unbinned = run_unbinned(run)
display(unbinned["diagnostics"])
display(unbinned["scans"])
print("Unbinned references refreshed from the saved models; no networks were trained.")
'''


def observable_distribution_code(observable, prefix):
    label = r'Ratio observable $z_\eta=r_\eta/(1+r_\eta)$' if observable == 'ratio' else r'Score observable $z_\eta$'
    settings = f'OBSERVABLE = {observable!r}\nPLOT_PREFIX = {prefix!r}\nOBSERVABLE_LABEL = {label!r}\n'
    return settings + r'''
import mplhep as hep
from poodemo.pipeline import prepare_quadrature, observable_values
from poodemo.inference import histogram_components

quad = prepare_quadrature(run)
selected_s, selected_sbi, selected_b, _ = quad["nominal"] @ quad["weights"]
rate_kappa = (selected_s + selected_b - selected_sbi) / selected_s
rate_turnover = (rate_kappa / 2)**2 if rate_kappa > 0 else np.nan
other_root = rate_kappa - np.sqrt(run.config["asimov_mu"])
rate_partner = other_root**2 if other_root >= 0 else np.nan
candidate_anchors = [run.config["mu_min"], run.config["asimov_mu"], rate_turnover, rate_partner,
                     (run.config["mu_min"] + run.config["mu_max"]) / 2, run.config["mu_max"]]
ETA_VALUES = []
for anchor in candidate_anchors:
    if (np.isfinite(anchor) and run.config["mu_min"] <= anchor <= run.config["mu_max"]
            and not any(np.isclose(anchor, old) for old in ETA_VALUES)):
        ETA_VALUES.append(float(anchor))
    if len(ETA_VALUES) == 4:
        break
ETA_VALUES = sorted(ETA_VALUES)  # Replace with your own anchors to explore.
HIST_BINS = 128
LOG_Y = True
edges = np.linspace(0., 1., HIST_BINS + 1)
distributions = []
for eta in ETA_VALUES:
    z_eta = observable_values(run, quad, eta, observable=OBSERVABLE)
    # The cached fields already include the process yields (lambda factors).
    nu_s, nu_sbi, nu_b, nu_ni = histogram_components(
        z_eta, quad["nominal"], quad["weights"], edges)
    root_eta = np.sqrt(eta)
    coherent = (eta - root_eta)*nu_s + root_eta*nu_sbi + (1. - root_eta)*nu_b
    tolerance = 1e-12 * max(1., np.max(np.abs(coherent)))
    if np.any(coherent < -tolerance):
        raise ValueError(f"Negative coherent prediction at eta={eta}: {coherent.min()}")
    # Remove only floating-point roundoff after checking positivity.
    distributions.append((eta, np.maximum(coherent, 0.), nu_ni))

with plt.style.context(hep.style.ATLAS), plt.rc_context({"font.size": 16}):
    ncols = min(2, len(distributions))
    nrows = (len(distributions) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 4.8*nrows),
                             sharex=True, sharey=True, squeeze=False)
    ymax = max(np.max(coherent + ni) for _, coherent, ni in distributions)
    for ax, (eta, coherent, ni) in zip(axes.flat, distributions):
        hep.histplot([ni, coherent], bins=edges, stack=True, histtype="fill",
                     color=["#B8B8B8", "#56B4E9"], edgecolor="black", linewidth=0.7,
                     label=["Non-interfering background", r"Coherent SBI ($\mu=\eta$)"],
                     yerr=False, ax=ax)
        ax.text(0.04, 0.94, fr"$\eta={eta:g},\quad\alpha_{{NI}}=0$",
                transform=ax.transAxes, va="top", fontsize=16)
        ax.set(xlim=(0, 1),
               xlabel=OBSERVABLE_LABEL, ylabel="Expected events / bin")
        if LOG_Y:
            ax.set_yscale("log")
            ax.set_ylim(ymax*1e-5, ymax*30)
        else:
            ax.set_ylim(0, 1.45*ymax)
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(False)
        ax.legend(loc="upper right", fontsize=12, frameon=False)
    for ax in axes.flat[len(distributions):]:
        ax.set_visible(False)
    fig.suptitle("Gaussian-amplitude toy · after preselection", fontsize=19)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    (ROOT / "plots").mkdir(exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(ROOT / "plots" / f"{PLOT_PREFIX}_{OBSERVABLE}_distributions.{extension}",
                    bbox_inches="tight", dpi=160)
    plt.show()

display(pd.DataFrame([
    {"eta": eta, "coherent_SBI_yield": coherent.sum(), "NI_yield": ni.sum(),
     "total_yield": (coherent + ni).sum()}
    for eta, coherent, ni in distributions
]))
'''

def fidelity_plot_code(prefix):
    settings = f'PLOT_PREFIX = {prefix!r}\n'
    return settings + r'''
fidelity = binning["fidelity"]
display(fidelity)
eta_near_truth = min(fidelity["eta"].unique(), key=lambda e: abs(e - run.config["asimov_mu"]))
at_anchor = fidelity[np.isclose(fidelity["eta"], eta_near_truth)].sort_values("n_bins")
y_columns = [c for c in fidelity if "information_fraction" in c]
if y_columns:
    ax = at_anchor.plot(x="n_bins", y=y_columns, marker="o", logx=True)
    ax.set(xlabel="Number of bins", ylabel="Retained Fisher information fraction",
           title=fr"Information retained at $\eta={eta_near_truth:.3g}$")
    ax.figure.tight_layout()
    (ROOT / "plots").mkdir(exist_ok=True)
    ax.figure.savefig(ROOT / "plots" / f"{PLOT_PREFIX}_binning_diagnostics.pdf", bbox_inches="tight")
    plt.show()
'''

def estimator_plot_code(prefix):
    settings = f'PLOT_PREFIX = {prefix!r}\n'
    return settings + r'''
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, squeeze=False)
truth_mu = float(estimator_study["truth"].iloc[0])
for row, systematic in enumerate([False, True]):
    subset = estimator_study[estimator_study["systematics"] == systematic]
    group_columns = [c for c in ["model", "n_bins"] if c in subset.columns]
    for label, values in subset.groupby(group_columns, dropna=False, sort=False):
        values = values.sort_values("eta")
        label_values = label if isinstance(label, tuple) else (label,)
        parts = [f"{int(value)} bins" if name == "n_bins" else str(value)
                 for name, value in zip(group_columns, label_values)
                 if pd.notna(value) and (name != "n_bins" or float(value) > 0)]
        legend_label = " | ".join(parts)
        line_style = "--" if "unbinned" in str(label_values[0]).lower() else "-"
        sigma = values["sigma_mu"].replace([np.inf, -np.inf], np.nan).where(values["width_valid"])
        axes[row, 0].plot(values["eta"], sigma, line_style, label=legend_label)
        axes[row, 1].plot(values["eta"], values["mu_hat"].where(values["fit_valid"]), line_style, label=legend_label)
    nuisance_label = "NI profiled" if systematic else "NI fixed at zero"
    axes[row, 0].set(title=nuisance_label, ylabel=r"Local width $\sigma_\mu(\eta)$")
    axes[row, 1].axhline(truth_mu, color="black", ls=":", label="Generating value")
    axes[row, 1].set(title=nuisance_label + ": same Asimov experiment", ylabel=r"$\hat\mu(\eta)$")
    finite_estimates = subset.loc[subset["fit_valid"], "mu_hat"].to_numpy()
    finite_estimates = finite_estimates[np.isfinite(finite_estimates)]
    lower = min(truth_mu, finite_estimates.min()) if len(finite_estimates) else truth_mu
    upper = max(truth_mu, finite_estimates.max()) if len(finite_estimates) else truth_mu
    margin = max(.03 * (run.config["mu_max"] - run.config["mu_min"]), .1 * (upper - lower))
    axes[row, 1].set_ylim(lower - margin, upper + margin)
    for ax in axes[row]:
        ax.set_xlabel(r"Frozen observable anchor $\eta$")
        ax.legend(fontsize=7)
fig.tight_layout()
for extension in ("pdf", "png"):
    fig.savefig(ROOT / "plots" / f"{PLOT_PREFIX}_estimator_eta.{extension}", bbox_inches="tight", dpi=160)
plt.show()
'''

def template_yield_plot_code(prefix):
    settings = f'PLOT_PREFIX = {prefix!r}\n'
    return settings + r'''
from poodemo.data import PROCESSES

yields = splines["yields"]
display(yields.head(20))
# Points are histogram anchors; lines are interpolated expected yields.
spline = splines["spline"]
eta_dense = np.linspace(spline.etas[0], spline.etas[-1], 200)
dense_yields = spline(eta_dense)
variation_order = ["nominal", "NI_down", "NI_up"]
processes = list(yields["sample"].unique())
fig, axes = plt.subplots(len(processes), 1, figsize=(8, 3 * len(processes)), squeeze=False)
for process, ax in zip(processes, axes[:, 0]):
    subset = yields[yields["sample"] == process]
    shown_bins = subset.groupby("bin")["yield_value"].sum().nlargest(3).index
    variations = ["nominal"] + (["NI_down", "NI_up"] if process == "NI" else [])
    subset = subset[subset["bin"].isin(shown_bins) & subset["variation"].isin(variations)]
    for (variation, bin_number), values in subset.groupby(["variation", "bin"], sort=False):
        values = values.sort_values("eta")
        curve = dense_yields[:, variation_order.index(variation), PROCESSES.index(process), int(bin_number)]
        line, = ax.plot(eta_dense, curve, label=f"{variation}, bin {bin_number}")
        ax.plot(values["eta"], values["yield_value"], ".", ms=3, color=line.get_color())
    ax.set(xlabel=r"Observable anchor $\eta$", ylabel="Expected bin yield", title=process)
    ax.legend(fontsize=7, ncol=3)
fig.tight_layout()
(ROOT / "plots").mkdir(exist_ok=True)
fig.savefig(ROOT / "plots" / f"{PLOT_PREFIX}_template_yields.pdf", bbox_inches="tight")
plt.show()
'''

def physical_yield_plot_code(prefix):
    settings = f'PLOT_PREFIX = {prefix!r}\n'
    return settings + r'''
physical = splines["physical_yields"]
shown_bins = physical.groupby("bin")["yield_value"].sum().nlargest(6).index
fig, ax = plt.subplots()
for bin_number, values in physical[physical["bin"].isin(shown_bins)].groupby("bin"):
    ax.plot(values["mu"], values["yield_value"], label=f"bin {bin_number}")
ax.set(xlabel=r"Physical signal strength $\mu$", ylabel="Expected total bin yield",
       title=fr"Physical scaling with the observable fixed at $\eta={run.config['asimov_mu']:g}$")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "plots" / f"{PLOT_PREFIX}_physical_bin_yields.pdf", bbox_inches="tight")
plt.show()
'''

def spline_scan_plot_code(prefix, xlim=None, ylim=None):
    settings = f'PLOT_PREFIX = {prefix!r}\nDEFAULT_SCAN_XLIM = {xlim!r}\nDEFAULT_SCAN_YLIM = {ylim!r}\n'
    return settings + r'''
display(splines["scans"])
SCAN_XLIM = DEFAULT_SCAN_XLIM  # Set None for the full scan range.
SCAN_YLIM = DEFAULT_SCAN_YLIM  # Set None for the full vertical range.
plot_scans(splines["scans"], "Profiled fits: spline, direct histogram, and unbinned", f"{PLOT_PREFIX}_profiled_scans.pdf",
           xlim=SCAN_XLIM, ylim=SCAN_YLIM)
comparison = splines["comparison"]
display(comparison)
print("Actual unique spline anchors:", len(splines["spline"].etas))
print("Largest spline/direct test-statistic difference on the scan:", comparison["delta_q"].abs().max())
off_grid = comparison.loc[~comparison["is_spline_anchor"]]
if len(off_grid):
    print("Largest difference at evaluated non-anchor points:", off_grid["delta_q"].abs().max())
else:
    print("All evaluated scan points are spline anchors; their agreement does not test interpolation between anchors.")
    print("Inspect the midpoint template-validation table and additional off-grid likelihood checks.")
'''

ESTIMATOR_STUDY_INTRO = r'''
### Estimator and local uncertainty versus the frozen observable

Now hold the generating experiment fixed at \((\mu_*,\alpha_{NI})=(\mu_*,0)\)
and vary only the observable anchor \(\eta\). Each row refills the
**same Asimov expectation**, freezes that histogram, and fits physical
\(\mu\), with NI either fixed or profiled. For an identified, correctly
modeled Asimov experiment, \(\hat\mu(\eta)\) should remain at the
generating value. This is not a study of finite-sample fluctuations.

The width is evaluated from the local Fisher information at the fixed
truth, using the extended likelihood and the NI auxiliary constraint:
\[
\sigma_\mu^{\rm fixed}(\eta)=1/\sqrt{I_{\mu\mu}^{(\eta)}},\qquad
\sigma_\mu^{\rm profiled}(\eta)
=\sqrt{[(\mathbf I^{(\eta)})^{-1}]_{\mu\mu}}.
\]
This is the local Asimov curvature of **each frozen-observable fit**.
It is neither the curvature of the scan that changes \(\eta=\mu_0\)
at every point nor a global confidence interval. A second likelihood
minimum can produce additional accepted parameter regions that a local
width cannot describe.

The full unbinned reference does not depend on \(\eta\). For the binned
comparison, profiled widths use nuisance morphing after binning, as in
the spline notebook; its difference from integrating the unbinned morphing is a
separate model effect. Inspect anomalous fitted values for optimizer
failure or multiple equally good minima before interpreting them as
compression bias.
'''

SPLINE_COORDINATE_INTRO = r'''
### The spline coordinate is the construction parameter \(\eta\)

A process yield \(H_{jb}(\eta)\) in a fixed \(z\)-bin changes when
\(\eta\) changes because the observable moves events between bins.
The physical \(\mu\)-dependence still enters through the exact amplitude
coefficients. These two roles must stay separate:
\[
\nu_b(\mu;\eta)=(\mu-\sqrt\mu)H_{Sb}(\eta)
+\sqrt\mu H_{SBI,b}(\eta)+(1-\sqrt\mu)H_{Bb}(\eta)
+H_{NI,b}(\eta).
\]
Since the selected phase space is fixed and bins are exhaustive,
\(\sum_bH_{jb}(\eta)=\lambda_j\) is independent of \(\eta\).
We interpolate bin fractions with positivity and normalization enforced,
then restore the separate total yield. SBI interference enters through
the coherent templates, not an independently generated signed sample.
'''

SPLINE_NUISANCE_INTRO = r'''
### Vary the templates, then interpolate the nuisance

Build nominal and ±1 templates for NI and nominal templates for S, B,
and SBI. Fit their dependence on \(\eta\).
At any construction point, evaluate the splines first; then use the
same exponential–polynomial rule to interpolate in \(\alpha_{NI}\).
Every profiled fit retains the Gaussian auxiliary constraint
\(\exp(-\alpha_{NI}^2/2)\), or \(+\alpha_{NI}^2\) in \(-2\log L\).

The Asimov data histogram also depends on \(\eta\) and is refilled from
the integration events at each tested point, rather than interpolated.
All nuisance fits keep \(\eta=\mu_0\) fixed
in both numerator and denominator.

Integration and nonlinear nuisance interpolation generally do not
commute:
\[
\int_{B_b(\eta)}\!\mathrm{Interp}_{\alpha_{NI}}[D(x)]\,dx
\ne \mathrm{Interp}_{\alpha_{NI}}\!\left[\int_{B_b(\eta)}D(x)\,dx\right].
\]
This difference is recorded separately from spline interpolation error.
Agreement at nominal and ±1 templates alone does not establish equality
between the continuously morphed unbinned and binned models.
'''

PHYSICAL_YIELD_INTRO = r'''
For comparison, the next plot fixes \(\eta=\mu_*\) and varies the **physical**
\(\mu\). This dependence comes from the signal/interference coefficients,
not from the spline. The two plots answer different questions.
'''



NN_METADATA_INTRO = r"""
### Classifier architecture and training metadata

Each production ratio classifier has **3 inputs → 1024 → 1024 → 1024 →
1 logit → Sigmoid**, with a bias in each linear layer and **SiLU/Swish**,
$a\mapsto a\,\sigma(a)$, after each hidden layer. The public network output is
$s=\sigma(\ell)\in[0,1]$. Standard mean **binary cross entropy** is computed
using its numerically stable BCE-with-logits form on the pre-sigmoid logit.
This is mathematically the same sigmoid+BCE objective. It avoids saturation
when evaluating the loss and $r=s/(1-s)=\exp(\ell)$ in the tails; sigmoid
itself is not an additional calibration correction.

Production uses batch size **1024**, labels 0=denominator and 1=numerator,
and equal class minibatches. There is **no dropout, batch normalization,
weight decay, L1/L2 penalty, or label smoothing**. NAdam starts at $10^{-3}$;
the learning rate decreases geometrically to **$10^{-9}$ at epoch 91**
(1-based) and stays there through epoch 100. Ratio early stopping is disabled
so the final low-rate epochs really run; the best-validation checkpoint is
still retained. The used rate in every epoch is recorded in `history.json`.
Preselection retains its separate original network settings.

The table below reads the actual loaded layers, activations and parameter
counts; smoke/custom runs can differ from these production dimensions.
The three production members are combined by averaging ratios, not logits.
Mean normalization on separate calibration events is a later operation, not
a loss penalty or a guarantee of shape closure.

This configuration requires the new v4 run. Matching completed v4 checkpoints
can be reused. Per-task/per-member metadata is saved in each `training.json`
and in `results/03_architecture.json`.
"""

NN_METADATA_CODE = r'''
from poodemo.pipeline import RATIO_TASKS
from poodemo.training import load_predictor, training_metadata
from poodemo.data import save_json

architecture_reports, architecture_rows, layer_rows = {}, [], []
for numerator, denominator in RATIO_TASKS:
    task = f"{numerator}_over_{denominator}"
    predictor = load_predictor(run.path("models", "ratios", task))
    report = training_metadata(predictor)
    architecture_reports[task] = report
    members = report.get("members", [report])
    for member_index, member in enumerate(members):
        architecture = member["architecture"]
        optimizer = member["optimizer"]
        architecture_rows.append({
            "ratio": f"{numerator}/{denominator}", "member": member_index,
            "inputs": architecture["input_features"],
            "hidden_layers": architecture["hidden_layers"],
            "width": architecture["hidden_width"],
            "activation": architecture["hidden_activation"],
            "output": architecture["output_activation"],
            "parameters": architecture["trainable_parameters"],
            "loss": member["loss"]["name"], "optimizer": optimizer["type"],
            "batch_size": member.get("config", {}).get("batch_size"),
            "initial_lr": optimizer["defaults"]["lr"],
            "lowest_lr_used": member.get("lowest_learning_rate_used"),
            "final_lr_used": member.get("final_learning_rate_used"),
            "epochs_completed": member.get("epochs_completed"),
            "dropout": member["regularization"]["dropout"],
            "weight_decay": optimizer["parameter_group_weight_decay"],
        })
        layer_rows.extend({"ratio": f"{numerator}/{denominator}", "member": member_index,
                           **layer} for layer in architecture["layers"])
    del predictor
save_json(run.path("results", "03_architecture.json"), architecture_reports)
display(pd.DataFrame(architecture_rows))
with pd.option_context("display.max_rows", None):
    display(pd.DataFrame(layer_rows))
print("Full optimizer, scheduler, preprocessing and checkpoint details:")
example = architecture_reports[next(iter(architecture_reports))]
example = example.get("members", [example])[0]
print(json.dumps({key: example[key] for key in
                 ("optimizer", "scheduler", "preprocessing", "checkpoint_selection")}, indent=2))
'''

RATIO_VALIDATION_INTRO = r"""
### Independent density validation bank

These diagnostics generate **new events**, apply the frozen selector from
notebook 02, and are prepared **before ratio training**. The bank is separate
from training, checkpoint-selection validation, calibration and workspace normalization.
The production default is two million **generated** events per source
(S, SBI, B, NI and NI up/down); selected counts are printed below. Generation
is cached with the physics/selection identity and random seed. Re-running with
the same settings reuses it. These controls are diagnostic budgets and do not
change `RUN_NAME`, the model, or `CONFIG_OVERRIDES`.

Each reweighting figure compares numerator events with denominator events
weighted by the learned ratio, for all three coordinates. Lower panels show
reweighted/target density with Monte Carlo uncertainty. The full sample
counts define normalization; the displayed range is never renormalized to
force agreement. Bins with too few target or reference events are flagged, not
reported as successful closure.

Calibration figures compare the mean predicted score $s=r/(1+r)$ to the
observed numerator fraction using an equal class prior, with residual
$f_{\rm numerator}-\langle s\rangle$ underneath. Immediately after **each
member** finishes, its raw reweighting and calibration plots are displayed,
before training the next member. After each task, the final ratio is shown in
both raw and calibration-mean-normalized forms. Member checks do not alter
the ensemble weights or fit any calibration map. Calibration
here is a diagnostic: these cells do not fit a new calibration map. Error bars
describe the independent finite validation samples conditional on the trained
network; they do not include training or calibration uncertainty.
"""

RATIO_VALIDATION_CODE = r'''
from poodemo.closure_validation import prepare_validation_bank

DIAGNOSTIC_N_PER_SAMPLE = 20_000 if MODE == "smoke" else 2_000_000
DIAGNOSTIC_EDGES = np.linspace(-5., 6., 56)
CALIBRATION_BINS = 20
VALIDATION_LOG_SCALE = False  # Set True for logarithmic marginal-density panels.

validation_bank = prepare_validation_bank(run, n_per_sample=DIAGNOSTIC_N_PER_SAMPLE)
display(pd.DataFrame([
    {"sample": name, "generated": validation_bank["generated_counts"][name],
     "selected": len(sample)}
    for name, sample in validation_bank["samples"].items()
]))
'''

TRAIN_RATIOS_WITH_VALIDATION_CODE = r'''
from poodemo.pipeline import train_ratios

def show_ratio_validation(report):
    print(f"{report['ratio']} — {report['stage']}", flush=True)
    for figure in report["figures"].values():
        display(figure)
    # Canvases are saved and closed by the pipeline after this callback returns.
    if report["stage"] == "ensemble":
        display(report["reweighting"].groupby("stage", sort=False)[
            ["denominator_mean_ratio", "numerator_events", "denominator_events"]].first())

ratio_diagnostics = train_ratios(
    run, validation_samples=validation_bank["samples"],
    on_validation=show_ratio_validation,
    validation_options={"edges": DIAGNOSTIC_EDGES, "score_bins": CALIBRATION_BINS,
                        "log": VALIDATION_LOG_SCALE},
)
display(ratio_diagnostics)
print("Saved per-member and ensemble plots:", run.path("plots", "03_validation"))
'''

CLOSURE_VALIDATION_INTRO = r"""
### High-statistics expectation and MLE closure

For a normalized target-to-reference density ratio, an independent denominator
sample should satisfy
\[
\widehat{E_q[r]}_N=\frac{1}{N}\sum_{i=1}^{N}r(x_i)\longrightarrow 1.
\]
The convergence plot uses increasing, nested prefixes of the selected bank,
with Monte Carlo standard errors and effective sample sizes saved in the
table. Prefix estimates are correlated. Increasing statistics can reveal a
persistent offset; it does not force an inaccurate learned ratio toward one.
No mean correction is fitted to this bank.

The unbinned likelihood diagnostic compares the analytical and learned models
on an independent, simulation-weighted expectation at **$\mu_*=1$**, and on a
separately generated high-statistics Poisson sample from SBI+NI. It shows
NI-fixed and NI-profiled scans, a vertical line at one, and fitted $\hat\mu$
values. A high-statistics expectation checks systematic MLE closure; an
individual Poisson sample also fluctuates statistically. The auxiliary NI
measurement is held at its nominal value. The learned-model self-Asimov scan
later in the notebook is a separate consistency check.

Workspace rates and shape-normalization constants remain frozen from the
original workspace quadrature. The new data never normalize the fitted model.
Therefore analytical closure itself has finite original-quadrature and new-bank
uncertainty; learned-minus-analytical differences help isolate network errors.
The Poisson-sample exposure is increased independently of the bank size and is
reported with the selected observed count. Invalid learned intensities or fits
are flagged; they are never clipped into a closing scan. Both interference
branches are included in the fit search.

These diagnostics can be expensive. Increase the controls below and the bank
budget above to examine stability. All tables and PDF/PNG figures are saved;
none of these cells retrain or recalibrate the networks.
The independent-data fits use NumPy/SciPy on CPU; the standard toolkit/JAX
scans below retain their configured backend. Both an overview and a zoom around
the fitted minima are saved, with extra scan points near each minimum.
"""

CLOSURE_VALIDATION_CODE = r'''
from poodemo.closure_validation import run_closure_validation

CLOSURE_SCAN_POINTS = 17 if MODE == "smoke" else 81
POISSON_GENERATED_EVENTS = 50_000 if MODE == "smoke" else 10_000_000
CLOSURE_PROFILE_NI = True

closure_validation = run_closure_validation(
    run, validation_bank, scan_points=CLOSURE_SCAN_POINTS,
    poisson_generated_events=POISSON_GENERATED_EVENTS, profile=CLOSURE_PROFILE_NI,
)
display(closure_validation["convergence"])
display(closure_validation["fits"])
display(pd.Series(closure_validation["poisson"], name="Independent Poisson sample"))
print("Closure figures, tables and provenance:", closure_validation["directory"])
for figure in closure_validation["figures"]:
    display(figure)
    plt.close(figure)
'''


def start(title, introduction):
    introduction = textwrap.dedent(introduction).strip()
    return [md(f"# {title}\n\n{introduction}"), md(SETUP_INTRO), code(BOOTSTRAP), code(COMMON_IMPORTS)]


def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": name, "provenance": []},
    }
    for index, cell in enumerate(notebook.cells):
        cell.id = f"cell-{index:03d}"
    destination = REPO / "notebooks" / name
    destination.parent.mkdir(exist_ok=True)
    nbf.write(notebook, destination)


def build():
    cells = start("1. Gaussian amplitudes and simulated samples", r"""
    We construct a positive interference model with known densities and derivatives.
    This provides an analytical benchmark for learned density ratios, unbinned
    likelihoods, and parameterized score histograms. Run this notebook first.

    Five million events per S/SBI/B sample and fifty million per NI/NI-up/NI-down
    sample provide Monte Carlo precision; the
    expected event yields of the statistical experiment are separate quantities.
    This benchmark is designed to make interference and nuisance effects visible;
    it illustrates the method and is not a numerical reproduction of an ATLAS
    measurement. The second branch remains below \(\mu=1\), while S now has a visibly
    different shape from B. The generating signal strength is \(\mu_*=1\).
    """)
    cells += [
        md(r"""
        ### Amplitudes, densities, and yields

        Let \(\phi_j(x)=\mathcal N_3(x;m_j,\Sigma_j)\), normalized to one.
        Define amplitudes
        \[
        A_S(x)=\sqrt{\lambda_S\phi_S(x)},\qquad
        A_B(x)=e^{i\varphi}\sqrt{\lambda_B\phi_B(x)},
        \qquad A_{NI}(x)=\sqrt{\lambda_{NI}\phi_{NI}(x)}.
        \]
        Their square roots are Gaussian functions, so these are Gaussian
        wavefunctions. The configured phase gives destructive interference.
        The means, positive-definite covariance matrices, phase, and inclusive
        yields are displayed below and saved with this run. S has a higher
        mean and narrower widths than B; SBI remains close to B because the
        coherent process is background dominated. The model was retuned to
        retain a secondary minimum **below 1**, without requiring it near 0.5.
        In an independent analytical-selector validation, the secondary minimum
        is near 0.120 with NI fixed and 0.069 with NI profiled; the primary
        minimum is at 1. A trained selector can shift these values. The
        full-space analytical likelihood also has a secondary minimum.

        The exposure is now 50, independently of generated sample counts.
        The scan starts at 0.02 to include the second basin. The bounded-score
        scale is 0.04, adapted to its broader distribution, so later comparisons
        do not saturate the old score transformation.

        The coherent process at \(\mu=1\) is
        \[
        D_{SBI}=|A_S+A_B|^2=D_S+D_B+D_I,
        \quad D_I=2\cos\varphi\sqrt{D_SD_B}.
        \]
        The SBI density is \(p_{SBI}=D_{SBI}/\lambda_{SBI}\), with
        \(\lambda_{SBI}=\int D_{SBI}\,dx\). It is **not** assigned an arbitrary
        yield: interference fixes its normalization. NI is added incoherently.

        At any \(\mu\ge0\),
        \[
        D(x;\mu)=|\sqrt\mu A_S+A_B|^2+D_{NI}
        = (\mu-\sqrt\mu)D_S+\sqrt\mu D_{SBI}
          +(1-\sqrt\mu)D_B+D_{NI}.
        \]
        The negative coefficients in this basis are physical algebra; the total
        amplitude model remains positive.

        Destructive interference alone does **not** guarantee a second likelihood
        minimum. Its visibility also depends on the signal/background shapes,
        selection, event rates, and scanned parameter range. Shape information
        can separate two hypotheses with the same total expected yield. Notebook
        3 measures the actual unbinned likelihood landscape after preselection.
        """),
        md(r"""
        ### Shape uncertainties and statistically independent roles

        The only nuisance is \(\alpha_{NI}\). Its ±1 anchors shift the NI
        amplitude mean along the configured direction with the configured
        magnitude. The actual shift vector is displayed below, so its strength
        remains explicit when the benchmark is changed.
        S, B, and their coherent SBI process are fixed with respect to this
        nuisance. The exact NI shift vector and all model settings are saved in
        the run configuration. The inclusive NI yield stays fixed; its selected
        yield may change because the selection efficiency changes.

        This gives six samples: S, B, SBI, NI, NI_up, and NI_down. Production mode
        generates 5 million events each for S, B and SBI, and 50 million each for
        NI, NI_up and NI_down: 165 million three-dimensional events in total.
        The NI multiplier compensates for its lower selection acceptance without
        changing any physical yield. Before any
        learning or selection, each sample is divided into seven independent
        roles:

        | Role | Fraction |
        |---|---:|
        | Preselection training | 20% |
        | Preselection validation | 5% |
        | Selection threshold calibration | 5% |
        | Density-ratio training | 40% |
        | Density-ratio validation | 10% |
        | Density-ratio calibration | 5% |
        | Final integration | 15% |

        Events used to train or tune the selector are never reused for the
        subsequent ratio-training or inference stages. A sample's Monte Carlo
        size does not change its physical yield.
        """),
        code(r'''
        display(pd.Series(run.model.to_dict(), name="amplitude model"))
        display(pd.DataFrame({
            "process": ["S", "SBI", "B", "NI"],
            "inclusive_yield": [run.model.component_yield(p) for p in ["S", "SBI", "B", "NI"]],
        }))
        ni_nominal_mean = run.model.mean("NI", alpha_ni=0.)
        ni_shift = run.model.mean("NI", alpha_ni=1.) - ni_nominal_mean
        print("NI mean at alpha=0:", ni_nominal_mean)
        print("NI +1 sigma mean displacement:", ni_shift)
        print("Displacement magnitude:", np.linalg.norm(ni_shift))
        print("Expected-yield exposure multiplier:", run.model.exposure)
        '''),
        code("from poodemo.pipeline import generate_data\n\nsample_summary = generate_data(run)\ndisplay(sample_summary)"),
        md(r"""
        ### Shape and expected-yield views

        The first figure overlays normalized S, SBI, B and NI marginals in
        linear and logarithmic views. Exact model bin averages are shown as
        steps, with a subset of generated MC as markers. The log panels expose
        tail differences without relying on sparsely populated MC bins.

        The next figures show expected yields at mu=0, 0.5, 1 and 2, before
        preselection. Each stack contains the positive coherent SBI(mu) sum
        and NI. Signed S/SBI/B basis terms are combined **before** stacking.
        At mu=0 the coherent component is exactly B. Exact Gaussian marginal
        integrals preserve normalization and positivity; no yield is inferred
        by subtracting independently fluctuating generated samples.
        """),
        code(r'''
        from poodemo.plotting import plot_process_marginals, plot_coherent_stacks

        # Plot-only controls: changing these never regenerates or retrains samples.
        PLOT_MAX_EVENTS = 100_000
        PLOT_EDGES = np.linspace(-5, 6, 67)  # Extend the range to inspect farther tails.
        STACK_MUS = [0., 0.5, 1., 2.]
        PLOT_SAMPLES = {
            process: np.load(ROOT / "raw" / f"{process}.npy", mmap_mode="r")[:PLOT_MAX_EVENTS]
            for process in ["S", "SBI", "B", "NI"]
        }
        plot_process_marginals(run.model, PLOT_EDGES, samples=PLOT_SAMPLES,
                               output=ROOT / "plots")
        plt.show()
        for log_y in [False, True]:
            plot_coherent_stacks(run.model, PLOT_EDGES, STACK_MUS, log=log_y,
                                 output=ROOT / "plots")
            plt.show()

        # Full-space yields, not just the finite visible plotting window.
        display(pd.DataFrame({
            "mu": STACK_MUS,
            "coherent_SBI_yield": [run.model.total_yield(mu) - run.model.component_yield("NI")
                                    for mu in STACK_MUS],
            "NI_yield": run.model.component_yield("NI"),
            "total_yield": [run.model.total_yield(mu) for mu in STACK_MUS],
        }))
        '''),
        md(r"""
        ### Inspect the saved experiment

        The generator writes samples in chunks and records normalization and
        sampling information. Re-running a completed stage reuses its saved
        products; changing the experiment should use a new `RUN_NAME`.
        Selected-data and quadrature caches are tied to the selector's
        fingerprint, preventing reuse after the selection model changes.
        Inspect the summary above for the generated counts and physical yields.
        The selected yields in notebook 2 will differ because the classifier cut
        has different efficiencies for each process and each systematic anchor.
        """),
        code(r'''
        print("Persistent products:")
        for path in sorted(ROOT.iterdir()):
            print(path.name + ("/" if path.is_dir() else ""))
        '''),
        md("Next: open **02_preselection.ipynb** with the same run name and mode."),
    ]
    write("01_generate_amplitudes.ipynb", cells)

    cells = start("2. Multiclass preselection", r"""
    Train a three-class classifier on S, B, and NI, then freeze a cut on its S
    output. The aim is to improve the physical S/NI yield ratio toward the
    configured target, without using the eventual inference sample to choose
    the cut. Run notebook 1 first. A GPU is recommended in production mode.
    """)
    cells += [
        md(r"""
        ### Why this is a separate classifier

        The preselection network uses multiclass cross entropy and the toolkit's
        neural-network implementation. Its output is a selection variable,
        not the final likelihood ratio or optimal score. Class-balanced training
        avoids allowing the dominant NI yield alone to determine the classifier.

        A threshold \(c\) is chosen on held-out calibration events to satisfy
        \[
        \frac{\lambda_S\epsilon_S(c)}
             {\lambda_{NI}\epsilon_{NI}(c)}\simeq\rho_{\rm target}.
        \]
        Within the available threshold choices, preserving signal efficiency
        matters. The target is approximate because efficiencies are estimated
        with finite samples. The same frozen cut is applied to every process,
        every nuisance variation, and all subsequent parameter hypotheses.
        """),
        code("from poodemo.pipeline import run_preselection\n\nprint('Inclusive S/NI:', run.model.component_yield('S') / run.model.component_yield('NI'))\nprint('Selected S/NI target:', run.config['target_s_over_ni'])\nselection = run_preselection(run)\ndisplay(selection)"),
        md(r"""
        ### Selection is part of the statistical experiment

        A cut changes both rates and normalized shapes. For process \(j\),
        \[
        \lambda_j^{\mathrm{sel}}=\lambda_j\epsilon_j,\qquad
        p_j^{\mathrm{sel}}(x)
        =\frac{\mathbf1_{C}(x)p_j(x)}{\epsilon_j}.
        \]
        Nuisance variations generally change \(\epsilon_j\). We retain those
        rate effects; renormalizing every selected variation to the nominal
        rate would define a different systematic model.

        Preselection training, selection tuning, density-ratio training, ratio
        calibration, and final integration use separate sample roles. Only the
        four density-ratio/integration partitions are materialized as selected
        arrays. The learned cut is held fixed during every likelihood fit.
        """),
        code(r'''
        print("Selection outputs are saved under:", ROOT)
        for key, value in selection.items():
            if isinstance(value, pd.DataFrame):
                print(key)
                display(value)
        '''),
        md(r"""
        ### Inspect the NI variation after selection

        The stronger NI shape variation can change the accepted NI yield even
        when its inclusive yield is fixed. The table and plot below use the
        independent integration partition, with the same frozen cut for all
        three NI anchors. These finite-sample acceptance estimates are a
        diagnostic; the later analytical comparison uses the common quadrature.
        """),
        code(r'''
        ni_selection = selection["selection"].query("split == 'integration'").set_index("sample")
        ni_selection = ni_selection.loc[["NI_down", "NI", "NI_up"]].copy()
        ni_selection["relative_to_nominal"] = ni_selection["expected_yield"] / ni_selection.loc["NI", "expected_yield"]
        display(ni_selection[["efficiency", "expected_yield", "relative_to_nominal"]])
        fig, ax = plt.subplots()
        ax.bar([-1, 0, 1], ni_selection["expected_yield"], width=.6, color=["#56B4E9", "0.55", "#E69F00"])
        ax.set(xticks=[-1, 0, 1], xlabel=r"NI nuisance anchor $\alpha_{NI}$",
               ylabel="Expected selected NI events", title="NI acceptance variation after the frozen cut")
        fig.tight_layout()
        (ROOT / "plots").mkdir(exist_ok=True)
        fig.savefig(ROOT / "plots" / "02_ni_selection_yields.pdf", bbox_inches="tight")
        plt.show()
        '''),
        md(r"""
        ### Process shapes and coherent stacks after preselection

        These are the notebook-01 views after the **same frozen classifier cut**:
        normalized S/SBI/B/NI marginals with linear/log axes, plus coherent SBI(mu)+NI
        stacks at mu=0,0.5,1,2 in both scales. Styling uses mplhep without an ATLAS label.

        Steps use the common held-out integration sample, with the selected analytical
        process densities evaluated on its weighted nodes. Unlike the inclusive plots,
        these selected marginal integrals are finite-quadrature estimates. Each shape
        is divided by its complete selected yield, not by the yield within the plotting
        window. Markers show selected integration-sample events; they can overlap the
        quadrature sample and are not an independent closure test.

        The stacks retain physical accepted yields. Coherent terms are combined on
        common nodes before histogramming; signed S/SBI/B terms are never stacked
        individually. At mu=0 the coherent layer is B; NI stays fixed. Changing mu does
        not retrain or move the selection cut. The table reports selected yields over
        all coordinates, including events outside the plotting window.

        Once preselection has completed, this cell can run after setup without rerunning
        training or fits. It saves `02_selected_...` PDF/PNG figures and a yield CSV,
        leaving notebook-01 figures intact.
        """),
        code(r'''
        from poodemo.data import selected_split
        from poodemo.pipeline import prepare_quadrature
        from poodemo.plotting import plot_process_marginals, plot_coherent_stacks

        PLOT_MAX_EVENTS = 100_000  # Per process; only controls optional MC markers.
        PLOT_EDGES = np.linspace(-5., 6., 89)
        STACK_MUS = [0., 0.5, 1., 2.]
        SHOW_MC_MARKERS = True

        selected_quad = prepare_quadrature(run)  # Reuses the saved frozen selector and cache.
        selected_plot_samples = ({
            process: selected_split(run, process, "integration")[:PLOT_MAX_EVENTS]
            for process in ["S", "SBI", "B", "NI"]
        } if SHOW_MC_MARKERS else None)
        plot_process_marginals(run.model, PLOT_EDGES, samples=selected_plot_samples,
                               quadrature=selected_quad, output=ROOT / "plots")
        plt.show()
        for log_y in [False, True]:
            plot_coherent_stacks(run.model, PLOT_EDGES, mus=STACK_MUS, log=log_y,
                                 quadrature=selected_quad, output=ROOT / "plots")
            plt.show()

        selected_rates = selected_quad["nominal"] @ selected_quad["weights"]
        selected_yields = pd.DataFrame([
            {"mu": mu, "coherent_SBI": run.model.coefficients(mu)[:3] @ selected_rates[:3],
             "NI": selected_rates[3], "total": run.model.coefficients(mu) @ selected_rates}
            for mu in STACK_MUS
        ])
        display(selected_yields)
        selected_yields.to_csv(ROOT / "results" / "02_selected_yields.csv", index=False)
        '''),
        md(r"""
        ### What to inspect

        Compare the achieved S/NI ratio with the target and inspect the retained
        signal efficiency. A smoke run checks interfaces, not attainable
        classifier performance. The next notebook uses the selected S sample as
        the common density-ratio reference and retains all selected process
        yields explicitly.
        """),
        md("Next: **03_unbinned_nsbi.ipynb**."),
    ]
    write("02_preselection.ipynb", cells)

    cells = start("3. Learned ratios and unbinned Asimov fits", r"""
    Train process-to-S and systematic-variation density ratios with the pinned
    NSBI toolkit, construct the interference model, and compare learned and
    analytical unbinned Asimov scans. Run notebooks 1–2 first.

    Here NSBI ratio estimation uses classifiers trained with cross entropy; it
    does not require a separate normalizing-flow density estimator because the
    S reference can be sampled directly.
    """)
    cells += [
        md(r"""
        ### Density ratios and the physical model

        For the fixed selected phase space, set \(q=p_S^{\rm sel}\), and learn
        \(r_j=p_j^{\rm sel}/q\) for the other processes. Shape ratios and physical
        yields are distinct. At the nominal nuisance value \(\alpha_{NI}=0\),
        \[
        \frac{D(x;\mu)}{q(x)}
        =(\mu-\sqrt\mu)\lambda_S
         +\sqrt\mu\lambda_{SBI}r_{SBI}(x)
         +(1-\sqrt\mu)\lambda_Br_B(x)
         +\lambda_{NI}r_{NI}(x),
        \]
        where every \(\lambda\) is now a **selected** yield.

        We also train the NI up/down variation-to-nominal ratios. There are five
        ratio tasks: SBI/S, B/S, NI/S, NI_up/NI, and NI_down/NI. The production
        three-member ensembles therefore contain fifteen ratio networks, plus
        the separate preselection network. Only NI is morphed; S, B, and SBI
        retain their nominal templates. The workspace parameters are
        \((\mu,\alpha_{NI})\).

        The nuisance interpolation is the toolkit's exponential–polynomial
        interpolation. A standard-normal auxiliary measurement supplies
        \[
        L_{\rm aux}(\alpha_{NI})\propto\exp(-\alpha_{NI}^2/2),
        \]
        so its contribution to \(-2\log L\) is \(+\alpha_{NI}^2\).
        This constraint is included in every profiled likelihood. The
        interpolated nuisance model need not equal a continuously shifted
        Gaussian amplitude away from its nominal and ±1 anchors.
        """),
        md(NN_METADATA_INTRO),
        md(RATIO_VALIDATION_INTRO),
        code(RATIO_VALIDATION_CODE),
        code(TRAIN_RATIOS_WITH_VALIDATION_CODE),
        code(NN_METADATA_CODE),
        md(r"""
        ### Validate ratios before interpreting the inference

        The analytical density is available in this toy. Held-out diagnostics
        distinguish finite training errors from score compression and histogram
        errors. Ratio normalization alone does not guarantee that the ratio is
        correct point by point. Signed interference coefficients make numerical
        positivity of the total learned intensity a particularly important
        diagnostic; a failed physical-domain check must not be silently hidden
        by clipping the final likelihood.

        To isolate neural-network shape errors in this demonstration, the
        learned and analytical likelihoods use the same accepted component
        yields from the common analytical quadrature. Sample-estimated selected
        rates are reported separately. Learned shapes are numerically normalized
        on that quadrature after independent calibration, and their raw
        integrals are saved. This uses the toy's known density for validation
        and rate control; an analysis without an oracle would estimate those
        quantities from its simulation and auxiliary measurements.

        The interference cancellations make this a
        more stringent test of learned ratios. Small shape-estimation errors
        can now have a substantial likelihood effect. Agreement of the
        analytical benchmark with the target landscape does not establish
        neural-network closure. Inspect the held-out ratio diagnostics and
        learned-versus-analytical curves; the short smoke training is only a
        workflow check, and production closure must be demonstrated separately.
        """),
        code("from poodemo.pipeline import prepare_quadrature\n\nquadrature = prepare_quadrature(run)\nprint('Quadrature products:', list(quadrature))"),
        md(CLOSURE_VALIDATION_INTRO),
        code(CLOSURE_VALIDATION_CODE),
        md(r"""
        ### Asimov integration and profiling

        We approximate expectations with a held-out weighted integration sample.
        These are Asimov likelihood curves, not a coverage study with random
        Poisson pseudo-experiments. The extended likelihood retains the total
        expected yield and includes the Gaussian NI auxiliary measurement.

        At the configured generating point \((\mu_*,\alpha_{NI,*})=(\mu_*,0)\),
        an exact intensity model gives
        \[
        t_A(\mu,\alpha_{NI})=2\int\!dx\,
        \left[D(x;\mu,\alpha_{NI})-D_*(x)
        -D_*(x)\log\frac{D(x;\mu,\alpha_{NI})}{D_*(x)}\right]
        +\alpha_{NI}^2,
        \]
        followed by profiling. Finite quadrature and learned-ratio errors are
        separate effects. The pipeline saves both analytical and learned
        reference curves so that later comparisons can identify their origin.
        """),
        code("from poodemo.pipeline import run_unbinned\n\nunbinned = run_unbinned(run)\ndisplay(unbinned['scans'])"),
        code(PLOT_HELPER),
        code("SCAN_XLIM = None  # e.g. (0.02, 1.15); None keeps the default x-axis range.\nSCAN_YLIM = None  # e.g. (0, 8); None keeps the default y-axis range.\nplot_scans(unbinned['scans'], 'Unbinned Asimov likelihood scans', '03_unbinned_scans.pdf',\n           xlim=SCAN_XLIM, ylim=SCAN_YLIM);"),
        md(r"""
        ### Why a second minimum can appear

        At nominal NI nuisance and with the selection fixed, write the selected
        total yield as
        \[
        \Lambda_\mu=\lambda_S\mu+\lambda_I\sqrt\mu+\lambda_B+\lambda_{NI},
        \qquad \lambda_I=\lambda_{SBI}-\lambda_S-\lambda_B<0.
        \]
        Equating this yield to the generating yield gives
        \[
        \Lambda_\mu-\Lambda_{\mu_*}
        =(\sqrt\mu-\sqrt{\mu_*})
        [\lambda_S(\sqrt\mu+\sqrt{\mu_*})+\lambda_I]=0.
        \]
        Besides \(\mu=\mu_*\), there is an equal-rate solution
        \[
        \mu_{\rm rate}=\left(-\lambda_I/\lambda_S-\sqrt{\mu_*}\right)^2
        \]
        when the expression in parentheses is nonnegative. This is an exact
        statement about the **total rate at nominal nuisance**, not an exact
        degeneracy of the full event distribution. Similar signal and
        interference shapes can leave a second likelihood minimum nearby;
        shape information can lift it. Profiling NI can change its depth and
        location, with the Gaussian constraint penalizing the required shift.

        The distinct-signal benchmark keeps the second basin below 1 rather
        than forcing it near 0.5. The selected-rate turnover is
        \[
        \mu_{\rm turnover}=\left(-\frac{\lambda_I}{2\lambda_S}\right)^2.
        \]
        Neither this turnover nor the equal-rate partner is generally a
        full-likelihood extremum. The analytical-selector validation gives
        secondary minima near 0.120 (NI fixed) and 0.069 (NI profiled);
        remeasure these locations with the trained selector below.

        The following analytical-only panels separate fixed-NI and profiled-NI
        curves. Dots mark interior minima on the evaluated grid, rather than
        claiming a more precise optimizer location. `LANDSCAPE_Q_MAX` focuses
        the vertical range on the nearby minima and intervening barrier; set
        it to `None` for automatic scaling. The preceding overview figure and
        saved scan table retain the full test-statistic range.
        """),
        code(r'''
        from poodemo.pipeline import prepare_quadrature

        LANDSCAPE_Q_MAX = 10.0  # Set None for the full vertical range.
        landscape_quad = prepare_quadrature(run)
        selected_s, selected_sbi, selected_b, selected_ni = landscape_quad["nominal"] @ landscape_quad["weights"]
        selected_interference = selected_sbi - selected_s - selected_b
        truth_mu = run.config["asimov_mu"]
        other_root = -selected_interference / selected_s - np.sqrt(truth_mu)
        equal_rate_mu = float(other_root**2) if other_root >= 0 else np.nan
        display(pd.Series({"selected_lambda_S": selected_s,
                           "selected_lambda_I": selected_interference,
                           "generating_mu": truth_mu,
                           "other_equal_rate_mu_at_alpha_NI_0": equal_rate_mu}, name="rate diagnostic"))
        analytical_scans = unbinned["scans"].query("model == 'analytic unbinned'")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
        minima = []
        for ax, systematic in zip(axes, [False, True]):
            frame = analytical_scans[analytical_scans["systematics"] == systematic].sort_values("mu")
            mu_values = frame["mu"].to_numpy()
            q_values = frame["q"].to_numpy()
            ax.plot(mu_values, q_values, color="#D55E00" if systematic else "#0072B2", lw=2)
            for index in range(1, len(frame) - 1):
                nearby = q_values[index-1:index+2]
                if np.all(np.isfinite(nearby)) and nearby[1] <= min(nearby[0], nearby[2]):
                    ax.scatter(mu_values[index], q_values[index], color="black", zorder=4)
                    minima.append({"systematics": systematic, "mu_grid_minimum": mu_values[index], "q": q_values[index]})
            ax.axvline(truth_mu, color="0.35", ls=":", label="Generating value")
            if np.isfinite(equal_rate_mu) and run.config["mu_min"] <= equal_rate_mu <= run.config["mu_max"]:
                ax.axvline(equal_rate_mu, color="0.6", ls="--", label="Other nominal equal-rate value")
            ax.axhline(1., color="0.7", ls=":")
            ax.set(xlabel=r"Tested signal strength $\mu_0$", ylabel=r"$t_A(\mu_0)$", ylim=(0, LANDSCAPE_Q_MAX),
                   title="Analytical: NI profiled" if systematic else "Analytical: NI fixed at zero")
            ax.legend(fontsize=8)
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(ROOT / "plots" / f"03_analytical_landscape.{extension}", bbox_inches="tight", dpi=160)
        plt.show()
        display(pd.DataFrame(minima))
        print("Scan range:", (run.config["mu_min"], run.config["mu_max"]))
        print("Fit bounds:", run.config["mu_fit_bounds"])
        '''),
        md(r"""
        ### Check the numerical expectation itself

        The next diagnostic evaluates the nominal-nuisance analytical Asimov
        objective using the full integration sample and two disjoint halves.
        The split is made **within each proposal stratum before selection**;
        rejected events contribute zero to the uncertainty calculation.
        Error bars estimate Monte Carlo integration uncertainty, not statistical
        confidence intervals on \(\mu\). Half-sample agreement and small
        estimated errors are checks on the numerical expectation; a precision
        study should also increase the integration budget in a separate run.
        """),
        code(r'''
        integration_check = unbinned["quadrature"].sort_values("mu")
        display(integration_check)
        fig, ax = plt.subplots()
        ax.errorbar(integration_check["mu"], integration_check["q"],
                    yerr=integration_check["standard_error"], fmt="o-", ms=3,
                    capsize=2, label="Full sample ± integration SE")
        ax.plot(integration_check["mu"], integration_check["q_half_even"], "--", label="Even half")
        ax.plot(integration_check["mu"], integration_check["q_half_odd"], ":", label="Odd half")
        ax.set(xlabel=r"Signal strength $\mu$", ylabel="Asimov likelihood objective",
               title="Integration stability: fixed nominal NI nuisance")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ROOT / "plots" / "03_quadrature_stability.pdf", bbox_inches="tight")
        plt.show()
        '''),
        code(r'''
        for key, value in unbinned.items():
            if key not in ("scans", "quadrature"):
                print(key)
                display(value)
        '''),
        md(r"""
        The horizontal line at \(t=1\) is a visual likelihood-scale reference.
        These Asimov curves alone do not demonstrate interval coverage or
        establish Wilks' theorem at a boundary. The scan uses positive signal
        strengths; the derivative of the \(\sqrt\mu\) model is singular at zero.
        The scan range and fit bounds are printed above from the run configuration.
        The positive lower fit bound is a numerical floor for
        this regular interior demonstration, not an implementation of a
        discovery test at the physical boundary \(\mu=0\).

        Short smoke training may give visibly inaccurate learned curves or
        invalid learned intensities. Such points are flagged and appear as gaps;
        they do not count as successful learned-model closure. Inspect fit
        validity and ratio diagnostics before increasing the training budget.
        """),
        md("Next: **04_score_histograms.ipynb**."),
    ]
    write("03_unbinned_nsbi.ipynb", cells)

    cells = start("4. Analytical scores and histogram coarsening", r"""
    Compute the analytical score at each tested signal strength, transform it to
    a bounded scalar, and compare increasingly fine Poisson histograms with the
    unbinned likelihood. The construction parameter is called \(\eta\); the
    fitted physical parameter remains \(\mu\). Run notebooks 1–3 first.
    """)
    cells += [
        md(r"""
        ### The score, including the yield factors

        In the selected phase space define normalized process densities \(p_j\)
        and their selected yields \(\lambda_j\). At \(\alpha_{NI}=0\),
        \[
        D_\mu(x)=(\mu-\sqrt\mu)\lambda_Sp_S(x)
        +\sqrt\mu\lambda_{SBI}p_{SBI}(x)
        +(1-\sqrt\mu)\lambda_Bp_B(x)+\lambda_{NI}p_{NI}(x),
        \]
        \[
        \Lambda_\mu=(\mu-\sqrt\mu)\lambda_S
        +\sqrt\mu\lambda_{SBI}+(1-\sqrt\mu)\lambda_B+\lambda_{NI},
        \qquad p(x;\mu)=D_\mu(x)/\Lambda_\mu.
        \]
        In particular, **there is no density factor in \(\Lambda_\mu\)**.
        For \(\eta>0\),
        \[
        D'_\eta(x)=\frac{(2\sqrt\eta-1)\lambda_Sp_S(x)
               +\lambda_{SBI}p_{SBI}(x)-\lambda_Bp_B(x)}{2\sqrt\eta},
        \]
        \[
        \boxed{o_\eta(x)=\left.\partial_\mu\log p(x;\mu)\right|_\eta
                 =\frac{D'_\eta(x)}{D_\eta(x)}
                  -\frac{\Lambda'_\eta}{\Lambda_\eta}.}
        \]
        The constant normalization term ensures zero mean under the selected
        normalized distribution. Keeping the event count separately preserves
        rate information. The bounded observable is
        \(z_\eta=1/[1+\exp(-o_\eta/a)]\), with one fixed positive scale
        \(a=\texttt{run.config['score\_scale']}\) for the run. This invertible
        transformation preserves the unbinned score information. Its scale
        affects how efficiently a finite uniform binning resolves the score.
        """),
        md(SCORE_DISTRIBUTIONS_INTRO),
        code(observable_distribution_code("score", "04")),
        md(r"""
        ### Freeze the observable inside each likelihood fit

        For the test of \(\mu_0\), construct \(z_{\eta=\mu_0}\) at nominal
        \(\alpha_{NI}=0\), then keep its bin edges, event assignments, and Asimov data
        fixed while fitting all physical alternatives \(\mu\). Thus,
        \[
        t(\mu_0)=-2\log\frac{L(\mu_0;\eta=\mu_0)}
                                   {\max_\mu L(\mu;\eta=\mu_0)}.
        \]
        The Asimov data must also be re-histogrammed when \(\eta\) changes.
        The binned likelihood is an independent Poisson likelihood for disjoint
        bins of a Poisson point process. This notebook fixes \(\alpha_{NI}=0\),
        where the scalar score has its clean local guarantee.
        """),
        md(UNBINNED_ASIMOV_REFRESH_INTRO),
        code(UNBINNED_ASIMOV_REFRESH),
        code("from dataclasses import replace\nfrom poodemo.pipeline import run_binning_study\n\nBIN_COUNTS = list(range(4, 61, 4))\n# Change only this resolution study; reuse the saved experiment and models.\nbinning_run = replace(run, config={**run.config, \"bin_counts\": BIN_COUNTS})\nbinning = run_binning_study(binning_run)\ndisplay(binning['scans'])\ndisplay(binning['fidelity'])"),
        code(PLOT_HELPER),
        code("SCAN_BIN_COUNTS = [4, 12, 24, 36, 60]\nscans = binning[\"scans\"]\nshown_scans = scans.loc[\n    (scans[\"model\"].eq(\"direct score histogram\") & scans[\"n_bins\"].isin(SCAN_BIN_COUNTS))\n    | scans[\"model\"].eq(\"analytic unbinned\")\n    | scans[\"model\"].str.startswith(\"learned unbinned\", na=False)\n].copy()\nSCAN_XLIM = (0.02, 1.15)  # Set None to keep the default x-axis range.\nSCAN_YLIM = (0, 8)  # Set None to keep the default y-axis range.\nscan_title = 'Score histograms: ' + ', '.join(map(str, SCAN_BIN_COUNTS)) + ' bins'\nplot_scans(shown_scans, scan_title, '04_binning_scans.pdf',\n           xlim=SCAN_XLIM, ylim=SCAN_YLIM);"),
        md(r"""
        ### What convergence can and cannot show

        Two information losses must be distinguished:

        1. **Compression** from \(x\) to \(o_\eta\). Before binning, the exact
           score retains the local Fisher information at \(\mu=\eta\).
        2. **Finite binning** of that score. At the anchor its shape-information
           loss is \(E_\eta[\operatorname{Var}_\eta(o_\eta\mid\text{bin})]\).

        Finer nested bins reduce the second loss. They do not automatically
        restore global likelihood information removed by scalar compression.
        Consequently, fine-bin curves should approach the **unbinned score
        experiment**, and can approximate the full unbinned scan very well
        locally; exact equality over a broad interference scan is not guaranteed.
        The tables report deviations rather than assuming convergence to zero.

        In a fixed generating Asimov experiment, \(\mu_*\) and the tested
        \(\eta\) differ away from the minimum. This is precisely where local
        optimality must not be mistaken for global sufficiency.
        """),
        code(fidelity_plot_code('04')),
        md(ESTIMATOR_STUDY_INTRO),
        code(r'''
        from poodemo.diagnostics import run_estimator_study

        estimator_study = run_estimator_study(run)
        display(estimator_study)
        invalid_rows = estimator_study[~estimator_study["fit_valid"] | ~estimator_study["width_valid"]]
        if len(invalid_rows):
            print("Invalid fits or local widths (omitted from the corresponding curves):")
            display(invalid_rows)
        print("Saved estimator table:", ROOT / "results" / "estimator_eta.csv")
        '''),
        code(estimator_plot_code('04')),
        md("Next: **05_spline_templates.ipynb**."),
    ]
    write("04_score_histograms.ipynb", cells)

    cells = start("5. Spline templates and nuisance-profiled fits", r"""
    Replace repeated histogram filling with a parameterized template model.
    Compare spline predictions with independent construction-parameter anchors,
    then compare direct binned, spline binned, and unbinned profiled scans.
    Run notebooks 1–4 first.
    """)
    cells += [
        md(SPLINE_COORDINATE_INTRO),
        md(SPLINE_NUISANCE_INTRO),
        code("from poodemo.pipeline import run_spline_study\n\nsplines = run_spline_study(run, n_bins=20)\ndisplay(splines['validation'])\ndisplay(splines['yields'])"),
        md(r"""
        We use **20 bins** explicitly, matching the finest resolution in notebook 4.
        The saved binning diagnostic reports the retained extended Fisher information
        across the tested anchors, including whether this choice reaches 99% at every
        anchor. The splines are shape-preserving piecewise cubic
        Hermite interpolants (PCHIP) of nonnegative bin fractions, followed by
        explicit normalization. This preserves zero-valued anchors without a
        logarithmic floor. Queries that differ from an anchor only by
        floating-point roundoff reuse its stored bin fractions, so nominal
        and varied templates keep their exact empty-bin support. No
        extrapolation beyond the anchor range is permitted.

        The score can change rapidly as \(\eta\) crosses the interference
        turnover, moving expected yields between fixed bins. The spline grid
        therefore combines uniform anchors over the scan range with a focused
        grid around the turnover calculated from **expected nominal yields**.
        Production uses 2,401 uniform anchors and 1,201 focused anchors within
        \(\pm0.1\) of that point; smoke mode uses 241 and 121. The generating
        value is included, duplicates are removed, and the focused interval is
        clipped to the scan range. These settings are configurable through
        `spline_anchors`, `spline_focus_anchors`, and `spline_focus_halfwidth`.

        The extra anchors use no observed fluctuations or fitted scan values.
        Their density addresses rapid observable changes, but cannot eliminate
        finite integration noise or guarantee a small likelihood error. The
        template-validation table compares predictions with direct histograms
        at intermediate points not used as spline anchors.
        The default 82-point production scan also differs from the uniform
        spline grid, so most scan points directly test likelihood interpolation.
        The comparison table's `is_spline_anchor` flag identifies coincident
        points and the included generating value; the shorter smoke scan is
        primarily a workflow check.
        """),
        code(template_yield_plot_code('05')),
        md(PHYSICAL_YIELD_INTRO),
        code(physical_yield_plot_code('05')),
        code(PLOT_HELPER),
        code(spline_scan_plot_code('05')),
        md(r"""
        ### Read the comparisons separately

        - **Spline vs direct histogram:** tests amortization of the moving
          observable. The midpoint template comparison evaluates interpolation;
          scan points flagged as spline anchors check agreement at stored knots.
        - **Coarse vs fine direct histograms:** tests finite binning.
        - **Binned nuisance interpolation vs integrated unbinned interpolation:**
          tests the distinction between two morphing prescriptions.
        - **Learned vs analytical unbinned:** tests ratio estimation.
        - **Score histograms vs the full unbinned model:** includes the effects
          of scalar compression, particularly when \(\alpha_{NI}\) is profiled.

        A scalar \(\mu\)-score at \(\alpha_{NI}=0\) does not automatically
        preserve all profiled information. A joint score vector, or a suitable
        efficient score, is a natural extension. This example measures the
        agreement obtained by the requested scalar construction.

        Notebook 4's \(\sigma_\mu(\eta)\) and \(\hat\mu(\eta)\) study uses the
        same frozen-fit convention. Those local widths help explain sensitivity
        near the generating minimum; this complete profile scan also reveals
        the secondary minimum and any disconnected low-test-statistic regions.
        """),
        code(r'''
        print("All stages finished. Results are stored under:", ROOT)
        print("Use a new RUN_NAME for a changed configuration or a fresh statistical study.")
        for key, value in splines.items():
            if key not in ("scans", "yields", "validation", "spline"):
                print(key)
                display(value)
        '''),
        md(r"""
        ### Further experiments

        Increase the integration statistics to test numerical stability; widen
        the scan to expose interference effects; compare a fixed \(\eta=\mu_*\)
        score with the moving score; or add nuisance-score components. An
        interval-coverage study would require genuine pseudo-experiments and
        calibration of the chosen statistic, beyond these Asimov comparisons.
        """),
    ]
    write("05_spline_templates.ipynb", cells)


def ratio_histogram_cells():
    """Parallel of notebook 4 with explicit ratio-specific statistical claims."""
    cells = start("6. Analytical density-ratio histograms", r"""
    Repeat notebook 4 with the exact density-ratio observable
    \(z_\eta=r_\eta/(1+r_\eta)\). Use the same selected events, analytical
    Asimov data, physical model, binning study, and fit convention. Only the
    scalar observable changes. Run notebooks 1–3 first; no network retraining
    or new Monte Carlo generation is needed. Running notebook 4 also permits
    a direct comparison with its saved score results.
    """)
    cells += [
        md(r"""
        ### The ratio of normalized, selected densities

        Use the same selected phase space and yield factors as notebook 4:
        \[
        D_\eta(x)=(\eta-\sqrt\eta)\lambda_Sp_S(x)
        +\sqrt\eta\lambda_{SBI}p_{SBI}(x)
        +(1-\sqrt\eta)\lambda_Bp_B(x)+\lambda_{NI}p_{NI}(x),
        \]
        \[
        \Lambda_\eta=(\eta-\sqrt\eta)\lambda_S
        +\sqrt\eta\lambda_{SBI}+(1-\sqrt\eta)\lambda_B+\lambda_{NI}.
        \]
        Here every \(p_j\) is a normalized **selected** process density and
        every \(\lambda_j\) is its selected yield. At nominal \(\alpha_{NI}=0\),
        \[
        r_\eta(x)=\frac{p(x;\eta,0)}{p_S(x)}
        =\frac{D_\eta(x)/\Lambda_\eta}{D_S(x)/\lambda_S},
        \qquad D_S(x)=\lambda_Sp_S(x),\qquad
        \boxed{z_\eta(x)=\frac{r_\eta(x)}{1+r_\eta(x)}}.
        \]
        The analytical Gaussian components and selected rates are evaluated
        on the same final quadrature. This is an exact analytical construction
        within that numerical integration, **not a learned ratio**. There is
        no score-scale factor. The monotone map places positive ratios between
        zero and one without changing their event ordering.

        For each fixed \(\eta\), this ordering is optimal for the binary
        comparison of the selected \(p(x;\eta,0)\) with the selected S reference.
        That binary statement does not make the scalar sufficient for the
        physical \(\mu\)-family. In particular, its unbinned local Fisher
        information at \(\mu=\eta\) need not equal that of the full event: the
        score \(\partial_\mu\log p\) need not be a function of \(r_\eta\).
        Keeping the Poisson event count retains the separate rate information.
        """),
        md(SCORE_DISTRIBUTIONS_INTRO),
        code(observable_distribution_code("ratio", "06")),
        md(r"""
        ### Freeze the observable inside each likelihood fit

        For the test of \(\mu_0\), construct \(z_{\eta=\mu_0}\) at
        \(\alpha_{NI}=0\), then freeze its bin edges, event assignments, and
        Asimov data while fitting all physical alternatives \(\mu\):
        \[
        t(\mu_0)=-2\log\frac{L(\mu_0;\eta=\mu_0)}
                                  {\max_\mu L(\mu;\eta=\mu_0)}.
        \]
        Refill the same analytical Asimov expectation when \(\eta\) changes.
        The coarsening scan below fixes NI at nominal, as in notebook 4.
        Moving the construction point to the tested hypothesis does not create
        the score's local information-preservation guarantee for this ratio.
        """),
        md(UNBINNED_ASIMOV_REFRESH_INTRO),
        code(UNBINNED_ASIMOV_REFRESH),
        code(r'''
        from dataclasses import replace
        from poodemo.pipeline import run_binning_study

        BIN_COUNTS = list(range(4, 61, 4))
        # The resolution study shares all other settings with notebook 4.
        binning_run = replace(run, config={**run.config, "bin_counts": BIN_COUNTS})
        binning = run_binning_study(binning_run, observable="ratio")
        display(binning["scans"])
        display(binning["fidelity"])
        print("Saved ratio study:", ROOT / "results" / "ratio_binning_scans.csv")
        '''),
        code(PLOT_HELPER),
        code(r'''
        SCAN_BIN_COUNTS = [4, 12, 24, 36, 60]
        scans = binning["scans"]
        shown_scans = scans.loc[
            (scans["model"].eq("direct ratio histogram") & scans["n_bins"].isin(SCAN_BIN_COUNTS))
            | scans["model"].eq("analytic unbinned")
            | scans["model"].str.startswith("learned unbinned", na=False)
        ].copy()
        SCAN_XLIM = (0.02, 1.15)  # Set None to show the full scan range.
        SCAN_YLIM = (0, 8)  # Set None to show the full vertical range.
        scan_title = "Ratio histograms: " + ", ".join(map(str, SCAN_BIN_COUNTS)) + " bins"
        plot_scans(shown_scans, scan_title, "06_binning_scans.pdf",
                   xlim=SCAN_XLIM, ylim=SCAN_YLIM);
        '''),
        md(r"""
        ### Distinguish ratio compression from finite binning

        Compression from \(x\) to \(r_\eta(x)\) can lose local information
        before any bins are introduced. At a fixed truth, its per-event shape
        Fisher-information loss is the conditional score variance
        \[
        I_X^{\rm shape}-I_{r_\eta}^{\rm shape}
        =E[\operatorname{Var}(\partial_\mu\log p(X;\mu)\mid r_\eta(X))].
        \]
        It vanishes only when the relevant score is determined by the ratio.
        The extended diagnostic multiplies the per-event information by the
        expected event count and adds the Poisson rate information.
        Finite bins can lose additional information. Refining **nested** bins
        reduces this second loss; the chosen counts need not form a nested
        sequence, so every successive point need not improve monotonically.

        Fine bins approach the unbinned **ratio experiment**, which may differ
        from the full unbinned event experiment even at \(\mu=\eta\). The Fisher
        table measures that total loss without assuming it vanishes. Compare
        with notebook 4's corresponding score table at the same anchors and
        resolutions. Both histogram studies use analytical Asimov data, so their
        full-dimensional reference is `analytic unbinned`; the two learned
        references retain their separate closure/consistency interpretations.
        """),
        code(fidelity_plot_code("06")),
        md(ESTIMATOR_STUDY_INTRO),
        md(r"""
        This estimator study uses the finest bin count in the original saved
        run configuration, just as notebook 4 does. The local `BIN_COUNTS`
        override above affects only the coarsening study. Using the same
        estimator resolution makes the score/ratio comparison controlled.
        Changing \(\eta\) changes a frozen observable, not the generating data.
        A flat \(\hat\mu(\eta)\) does not establish information sufficiency:
        the local width can still increase after ratio compression.
        """),
        code(r'''
        from poodemo.diagnostics import run_estimator_study

        print("Estimator-study bin count:", max(run.config["bin_counts"]))
        estimator_study = run_estimator_study(run, observable="ratio")
        display(estimator_study)
        invalid_rows = estimator_study[~estimator_study["fit_valid"] | ~estimator_study["width_valid"]]
        if len(invalid_rows):
            print("Invalid fits or local widths (omitted from the corresponding curves):")
            display(invalid_rows)
        print("Saved estimator table:", ROOT / "results" / "ratio_estimator_eta.csv")
        '''),
        code(estimator_plot_code("06")),
        md("Next: **07_ratio_spline_templates.ipynb**. Ratio results use a `ratio_` prefix; score-study results are retained."),
    ]
    return cells


def ratio_spline_cells():
    """Parallel of notebook 5 with the same ratio construction as notebook 6."""
    cells = start("7. Density-ratio spline templates and profiled fits", r"""
    Repeat notebook 5 with the analytical normalized-density ratio observable
    \(z_\eta=r_\eta/(1+r_\eta)\) from notebook 6. The physical model, selected
    samples, analytical Asimov data, spline method, NI nuisance interpolation,
    and Gaussian auxiliary constraint are unchanged. Run notebooks 1–3 and 6
    first. The outputs are separate from the score-template study.
    """)
    cells += [
        md(SPLINE_COORDINATE_INTRO),
        md(r"""
        Here the frozen scalar is
        \[
        z_\eta(x)=\frac{r_\eta(x)}{1+r_\eta(x)},\qquad
        r_\eta(x)=\frac{D_\eta(x)/\Lambda_\eta}{D_S(x)/\lambda_S}
        \quad(\alpha_{NI}=0).
        \]
        Both densities are normalized after the same fixed preselection, using
        the common quadrature. There is no score-scale parameter and no neural
        estimate in this observable. Changing the histogram coordinate does not
        change the physical amplitude coefficients or expected process rates.
        """),
        md(SPLINE_NUISANCE_INTRO),
        code(r'''
        from poodemo.pipeline import run_spline_study

        splines = run_spline_study(run, n_bins=20, observable="ratio")
        display(splines["validation"])
        display(splines["yields"])
        '''),
        md(r"""
        ### Hold the spline study settings fixed

        Use **20 bins** explicitly, one of the resolutions measured in notebook
        6 and the same choice used in notebook 5. This does not assume 20 bins
        retain 99% of the full information: the saved selection metadata reports
        the measured Fisher fraction and whether that target is reached.

        The shape-preserving piecewise cubic Hermite interpolants (PCHIP)
        interpolate nonnegative bin fractions and then renormalize them. The
        total selected process yield is restored separately. Exact empty-bin
        support is kept at stored anchors, and no extrapolation is permitted.
        The shared anchor grid combines uniform points with extra points near
        the turnover of the nominal selected total rate, plus the generating
        value. This matches the score study's computational settings without
        using observed fluctuations to choose anchors. The actual grid is
        controlled by `spline_anchors`, `spline_focus_anchors`, and
        `spline_focus_halfwidth` in the saved run configuration.

        A dense grid alone does not guarantee a small likelihood error. Inspect
        direct-versus-spline templates at intermediate anchors and the likelihood
        differences at scan points flagged `is_spline_anchor=False`. Agreement
        at stored knots does not test interpolation between knots. Compare the
        results with notebook 5 to see how the change of observable affects both
        information retention and template interpolation.
        """),
        code(template_yield_plot_code("07")),
        md(PHYSICAL_YIELD_INTRO),
        code(physical_yield_plot_code("07")),
        code(PLOT_HELPER),
        code(spline_scan_plot_code("07", xlim=(0.02, 1.15), ylim=(0, 8))),
        md(r"""
        ### Read the comparisons separately

        - **Spline vs direct ratio histogram:** tests amortization of the
          moving observable, with off-grid points testing interpolation.
        - **Coarse vs fine direct ratio histograms:** tests finite binning.
        - **Binned nuisance interpolation vs integrated unbinned interpolation:**
          tests noncommuting morphing and integration; inspect `morph_comparison`.
        - **Learned unbinned on analytical Asimov vs analytical unbinned:**
          tests learned-model closure on the same generating experiment.
        - **Learned unbinned on its own finite model Asimov:** tests internal
          consistency, not agreement with analytical curvature or secondary minima.
        - **Ratio histograms vs analytical unbinned:** includes scalar
          compression loss, as well as finite binning and the nuisance-morph
          difference. It has no automatic local Fisher-preservation guarantee.

        The ratio gives the binary ordering for \(\eta\) versus the S reference.
        A scalar preserving that ordering need not retain all information about
        the physical \(\mu\)-family, particularly with NI profiled. Notebook
        6's \(\sigma_\mu(\eta)\) and \(\hat\mu(\eta)\) curves describe the same
        fixed-truth frozen-fit convention. Those local widths are not global
        confidence intervals across separate likelihood minima.
        """),
        code(r'''
        print("Ratio spline results are stored under:", ROOT / "results")
        print("Tables use a ratio_ prefix; figures use 07_. Score results are retained.")
        for key, value in splines.items():
            if key not in ("scans", "yields", "validation", "spline"):
                print(key)
                display(value)
        '''),
        md(r"""
        ### Further experiments

        Compare ratio and score results at matching bin counts and anchors;
        increase integration statistics to separate numerical fluctuations from
        scalar-compression loss; or compare a fixed \(\eta=\mu_*\) ratio with
        the moving ratio. A coverage study needs pseudo-experiments and
        calibration beyond these analytical and model-Asimov comparisons.
        """),
    ]
    return cells


def build_ratio_notebooks():
    """Write only the additional ratio notebooks; preserve existing notebooks."""
    write("06_ratio_histograms.ipynb", ratio_histogram_cells())
    write("07_ratio_spline_templates.ipynb", ratio_spline_cells())


if __name__ == "__main__":
    build()
    build_ratio_notebooks()
