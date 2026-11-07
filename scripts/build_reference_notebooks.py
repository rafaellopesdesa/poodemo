"""Update notebooks 08/09 in place, retaining user-edited study/plot settings.

The checked-in notebooks are the source for their cells. Unlike the original
bootstrap from 06/07, this migration preserves later edits to the reference
studies. Only 08/09 are written, and stale execution outputs are cleared.
"""
import re

import nbformat
from build_notebooks import REPO

MODEL_SOURCE_INTRO = r"""
### Reuse notebook 03's filtered density ratios

Run notebook 03 with the transferred Unity models first. Keep the same
`RUN_NAME`, `MODE`, `CONFIG_OVERRIDES`, `UNITY_MODEL_DIR`, and
`ENSEMBLE_FILTER_POLICY` here. The default directory is
`My Drive/poodemo/unity_models/paper-distinct-s-v4-unity100/`; it contains
`bundle.json`, `campaign.json`, and `ratios/` directly inside it.

The next cell verifies the transferred bundle and applies the same pointwise
outlier policy to all five ensembles. With unchanged inputs and policy it
reuses notebook 03's cached retained-member lists. Changed inputs require a
new screening pass; no networks are trained. `RATIO_SOURCE = "drive"` explicitly
selects the original Drive models instead, with the same filtering rules.

All tables, figures, and refreshed unbinned workspaces go to the selected
evaluation's directory, printed below. The base Drive samples, preselection,
and quadrature remain the inputs. The reference-ratio observable and histogram
templates are **analytical**; the new neural ratios update the learned unbinned
comparison. Changing the ensembles does not change the rationale for the floor.

Start a fresh Colab runtime to load the updated repository. Run this notebook
from the top; an existing `/content/poodemo` checkout must be refreshed first.
"""

MODEL_SOURCE_CODE = '''
import os
from pathlib import Path
from poodemo.data import save_json
from poodemo.ensemble_bundle import use_bundle, BundleEvaluationRun
from poodemo.member_filter import MemberFilterPolicy, filter_ratio_ensembles, unfiltered_run

RATIO_SOURCE = "unity"  # "drive" explicitly selects the original saved models.
UNITY_MODEL_DIR = Path(os.environ.get(
    "POODEMO_UNITY_MODEL_DIR",
    "/content/drive/MyDrive/poodemo/unity_models/paper-distinct-s-v4-unity100",
))
ENSEMBLE_FILTER_POLICY = MemberFilterPolicy(
    catastrophic_peer_factor=100.0,
)  # Match notebook 03, including any custom policy settings.

# Unwrap a previous selection so rerunning this cell never nests evaluations.
source_run = unfiltered_run(run)
source_run = source_run.base_run if isinstance(source_run, BundleEvaluationRun) else source_run
if RATIO_SOURCE == "unity":
    if not (UNITY_MODEL_DIR / "bundle.json").is_file():
        raise FileNotFoundError(
            f"Copy the complete Unity model bundle to {UNITY_MODEL_DIR}. "
            "Expected bundle.json, campaign.json and ratios/ directly inside it. "
            "Use the same directory as notebook 03; see unity/README.md.")
    ensemble_run = use_bundle(source_run, UNITY_MODEL_DIR)
elif RATIO_SOURCE == "drive":
    ensemble_run = source_run
else:
    raise ValueError("RATIO_SOURCE must be 'drive' or 'unity'")
run = filter_ratio_ensembles(ensemble_run, policy=ENSEMBLE_FILTER_POLICY)
filter_summary = pd.read_csv(run.path("results", "ensemble_filter", "summary.csv"))
display(filter_summary)
print("Ratio source:", RATIO_SOURCE)
print("Retained ratio models:", run.path("models", "ratios"))
print("Evaluation outputs:", run.evaluation_root)
save_json(run.path("results", "reference_ratio_model_source.json"), {
    "ratio_source": RATIO_SOURCE,
    "bundle_directory": str(ensemble_run.bundle) if RATIO_SOURCE == "unity" else None,
    "campaign_id": ensemble_run.bundle_manifest["campaign_id"] if RATIO_SOURCE == "unity" else None,
    "filter_identity": run.filter_manifest["identity"],
    "filter_policy": run.filter_manifest["inputs"]["policy"],
    "evaluation_root": str(run.evaluation_root),
})
'''

DEFINITION = r"""
### Ratio to the fixed physical reference

At nominal NI nuisance, use the analytical **selected, normalized** densities:
$$
\eta_{\mathrm{eff}}=\max(\eta,\eta_{\min}),\qquad
R_\eta(x)=\frac{r(x;\eta_{\mathrm{eff}})}{r(x;1)}
=\frac{p(x;\eta_{\mathrm{eff}},0)}{p(x;1,0)},
\qquad z_\eta(x)=\frac{R_\eta(x)^a}{1+R_\eta(x)^a}
=\operatorname{sigmoid}\!\left(a\log R_\eta(x)\right).
$$
The $S$ reference cancels. Both normalizations use the same final quadrature;
this observable is analytical, not learned. The denominator is always at
$(\mu,\alpha_{NI})=(1,0)$, independent of the fit, anchor and Asimov truth settings.
The event count retains the Poisson rate information.

Use **36 bins**, **`REFERENCE_RATIO_ETA_MIN = 0.2`** and
**`REFERENCE_RATIO_A = 100`** throughout notebooks 08–11. The floor changes only
which observable partitions the events. **Physical test points, generated truth,
component coefficients and the unrestricted physical-$\mu$ fit are not floored.**
All $\eta$ below the floor share a partition; their physical bin yields still vary
with $\mu$. Above the floor the original observable is unchanged. These settings
require rerunning the downstream studies, without sample generation, retraining
or changing the saved physics run. The power stays fixed throughout every fit.

For fixed NI, the continuous ratio retains the comparison between
$\eta_{\mathrm{eff}}$ and $1$. Below the floor this is not the pairwise-optimal ratio for
physical $\eta$ versus $1$: the deliberate change targets the competing-minimum
problem seen in notebook 12. Even above the floor, pairwise sufficiency does
not imply sufficiency for every freely fitted $\mu$ or for NI profiling. Compare
to **analytical unbinned**; learned references have separate closure meanings.

The value $0.2$ is an example-specific regularization, not a universal optimum.
The matched no-floor study below measures local widths at fixed truth $\mu_*=1$;
it does not establish competing-branch stability or confidence-interval coverage
at low physical $\mu$. Those require paired toys with the complete free fit.

At $\eta=1$, $R=1$ and $z=1/2$ exactly for every event. The separate equality
collapse remains: no jitter, shifted high-eta endpoint or score substitution is
introduced. That frozen histogram measures only the total rate. Near 1 the
distribution still contracts for any finite positive $a$. Larger $a$ spreads ratios
near 1 across more of $[0,1]$; $a=1$ gives the original $R/(1+R)$ transformation.

Every histogram uses `observable_bin_edges`: **0.5 is strictly inside a bin**.
Odd bin counts use uniform edges. For even counts, the internal edges shift
left by half a nominal bin width, so the central bins keep equal widths and
0.5 is a bin center (for at least four bins). Only the endpoint bins have widths
0.5 and 1.5 times the usual width. The count and endpoints 0 and 1 are preserved.
The same edges are used in plots, direct fits, estimator diagnostics and splines.
These are event-yield plots per bin, not densities divided by bin width.
"""

SPLINE_WARNING = r"""
### Interpolation at the floor and collapsed anchor

Use **`SPLINE_BINS = 36`**, with the same power and floor as notebook 08.
The saved 08 study is checked for matching observable settings and bin edges;
rerun 08 first after changing them. PCHIP interpolation includes explicit knots
at eta_min and eta=1, independently of the Asimov truth. No physical process
yields or data are replaced by a floored physical hypothesis.

Below eta_min, every nominal and NI-varied process histogram has the same
partition. Identical template anchors at eta=0 and eta_min reproduce the **exact
plateau** under PCHIP interpolation. The floor is an explicit knot, so the
interpolation never spans its kink in one interval. The physical mu coefficients still vary throughout the fit. Inspect
off-anchor validation immediately above the floor as well as near eta=1.

The shared bin edges place 1/2 strictly inside a bin, avoiding a numerical split
of the eta=1 events. For a finite bank, sufficiently nearby anchors also occupy
that bin: a local rate-only plateau is a real finite-binning effect. Stretching
can sharpen template changes outside it. Finite-bank histograms can require
additional eta knots. Data are always refilled directly at each physical eta
with the floored observable prescription.

Agreement at a stored anchor tests reproduction of that anchor, not accuracy
between knots. Spline discrepancies are interpolation errors. Independently,
the floored ratio below eta_min is no longer the pairwise-optimal ratio for the
physical tested eta versus 1. A spline fit can shift its minimum if its templates
differ from the directly filled data.
"""

REFERENCE_SETUP = '''
# Observable-only settings: keep these identical in notebooks 08–11.
# The floor changes the observable, never the physical mu or its fit bounds.
N_BINS = 36
REFERENCE_RATIO_ETA_MIN = 0.2
REFERENCE_RATIO_A = 100.0
if not np.isfinite(REFERENCE_RATIO_A) or REFERENCE_RATIO_A <= 0:
    raise ValueError("REFERENCE_RATIO_A must be finite and positive")
if not np.isfinite(REFERENCE_RATIO_ETA_MIN) or not 0 <= REFERENCE_RATIO_ETA_MIN < 1:
    raise ValueError("REFERENCE_RATIO_ETA_MIN must lie in [0, 1)")
run = run.with_config(
    reference_ratio_power=float(REFERENCE_RATIO_A),
    reference_ratio_eta_floor=float(REFERENCE_RATIO_ETA_MIN),
)
print(f"Reference ratio: {N_BINS} bins, eta floor={REFERENCE_RATIO_ETA_MIN:g}, "
      f"a={REFERENCE_RATIO_A:g}; 0.5 is inside a bin")
'''

COLLAPSE_CODE = '''
from poodemo.pipeline import observable_values, prepare_quadrature, observable_bin_edges
quad = prepare_quadrature(run)
COLLAPSE_BINS = N_BINS
collapse_edges = observable_bin_edges(COLLAPSE_BINS, "reference_ratio")
central_bin = np.searchsorted(collapse_edges, .5, side="right") - 1
assert collapse_edges[central_bin] < .5 < collapse_edges[central_bin + 1]
collapse_rows = []
check_etas = sorted(set([0., REFERENCE_RATIO_ETA_MIN / 2, REFERENCE_RATIO_ETA_MIN,
                         0.99, 0.999, 1.0, 1.001, 1.01]))
for eta in check_etas:
    z = observable_values(run, quad, eta, "reference_ratio")
    occupied = np.count_nonzero(np.histogram(z, collapse_edges)[0])
    collapse_rows.append(dict(eta=eta, eta_effective=max(eta, REFERENCE_RATIO_ETA_MIN),
                              a=REFERENCE_RATIO_A, n_bins=COLLAPSE_BINS,
                              z_min=z.min(), z_max=z.max(), occupied_bins=occupied,
                              central_low=collapse_edges[central_bin],
                              central_high=collapse_edges[central_bin + 1]))
collapse = pd.DataFrame(collapse_rows)
display(collapse)
np.testing.assert_array_equal(
    observable_values(run, quad, 0., "reference_ratio"),
    observable_values(run, quad, REFERENCE_RATIO_ETA_MIN, "reference_ratio"))
assert np.all(observable_values(run, quad, 1., "reference_ratio") == .5)
assert collapse.loc[collapse.eta.eq(1.), "occupied_bins"].item() == 1
collapse.to_csv(run.path("results", "reference_ratio_collapse.csv"), index=False)
'''

SPLINE_CHECKS = '''
# Check both special construction points, independently of the physical fit.
print("Largest off-anchor bin-fraction errors (inspect eta near the floor and 1):")
display(splines["validation"].nlargest(12, "max_fraction_error"))
print("Direct-versus-spline likelihood checks near the floor or eta=1:")
comparison = splines["comparison"]
near_special_anchor = ((abs(comparison["mu"] - REFERENCE_RATIO_ETA_MIN) < .05)
                       | (abs(comparison["mu"] - 1.) < .05))
display(comparison.loc[near_special_anchor])
if REFERENCE_RATIO_ETA_MIN > 0:
    assert np.any(np.isclose(splines["spline"].etas, REFERENCE_RATIO_ETA_MIN))
    plateau = splines["spline"](np.array([0., REFERENCE_RATIO_ETA_MIN / 2,
                                        REFERENCE_RATIO_ETA_MIN]))
    np.testing.assert_allclose(plateau[0], plateau[-1], rtol=0., atol=1e-12)
    np.testing.assert_allclose(plateau[1], plateau[-1], rtol=0., atol=1e-12)
    print("All nominal and NI-varied process templates are constant below the floor.")
else:
    print("No positive floor: the legacy spline grid has no low-eta plateau to check.")
'''

RUNTIME_NOTE = '''
For the 36-bin/floored-observable update, keep the completed v4 run and rerun
**08, then 09**, followed by **10 and 11**. Keep `REFERENCE_RATIO_A` and
`REFERENCE_RATIO_ETA_MIN` identical throughout. No sample generation,
retraining, or new physics run name is required. Old notebook outputs are
cleared because they used a different observable or resolution.
'''


ESTIMATOR_STUDY_CODE = '''
from poodemo.diagnostics import run_estimator_study

print("Estimator-study bin count:", N_BINS)
# Primary study: keep the configured floor and its usual downstream outputs.
estimator_study = run_estimator_study(run, observable="reference_ratio", n_bins=N_BINS)
# Matched control: only the observable floor changes. Truth, integration bank,
# eta grid, bin edges, power, physical coefficients and free-fit bounds stay fixed.
no_floor_run = run.with_config(reference_ratio_eta_floor=0.0)
estimator_no_floor = run_estimator_study(
    no_floor_run, observable="reference_ratio", n_bins=N_BINS,
    output_tag="reference_ratio_estimator_eta_no_floor")
np.testing.assert_array_equal(np.sort(estimator_study["eta"].unique()),
                              np.sort(estimator_no_floor["eta"].unique()))
FLOOR_LABEL = f"{N_BINS} bins, floor={REFERENCE_RATIO_ETA_MIN:g}"
NO_FLOOR_LABEL = f"{N_BINS} bins, no floor"
# Include a single analytical baseline per eta and NI treatment.
is_unbinned = estimator_study["model"].eq("analytic unbinned")
estimator_comparison = pd.concat([
    estimator_study.loc[~is_unbinned].assign(observable_variant=FLOOR_LABEL),
    estimator_no_floor.loc[~estimator_no_floor["model"].eq("analytic unbinned")]
        .assign(observable_variant=NO_FLOOR_LABEL),
    estimator_study.loc[is_unbinned].assign(observable_variant="Analytical unbinned"),
], ignore_index=True)
display(estimator_comparison)
invalid_rows = estimator_comparison[
    ~estimator_comparison["fit_valid"] | ~estimator_comparison["width_valid"]]
if len(invalid_rows):
    print("Invalid fits or local widths (omitted from the corresponding curves):")
    display(invalid_rows)
estimator_comparison.to_csv(
    run.path("results", "reference_ratio_estimator_eta_floor_comparison.csv"), index=False)
print("Primary floored estimator table:", run.path("results", "reference_ratio_estimator_eta.csv"))
print("Independent no-floor control:", run.path("results", "reference_ratio_estimator_eta_no_floor.csv"))
'''

ESTIMATOR_PLOT_CODE = r'''
PLOT_PREFIX = '08'
ESTIMATOR_ZOOM_XLIM = (0.0, 0.5)  # Low-eta width comparison; edit to change the zoom.

import mplhep as hep

truth_mu = float(estimator_study["truth"].iloc[0])
estimator_styles = {
    FLOOR_LABEL: dict(color="#0072B2", linestyle="-", linewidth=2.2),
    NO_FLOOR_LABEL: dict(color="#D55E00", linestyle="--", linewidth=1.8),
    "Analytical unbinned": dict(color="black", linestyle=":", linewidth=2.0),
}

def estimator_curves(systematic):
    subset = estimator_comparison.loc[estimator_comparison["systematics"].eq(systematic)]
    for label, style in estimator_styles.items():
        values = subset.loc[subset["observable_variant"].eq(label)].sort_values("eta")
        yield label, values, style

def mark_estimator_floor(ax):
    if REFERENCE_RATIO_ETA_MIN > 0:
        ax.axvline(REFERENCE_RATIO_ETA_MIN, color="0.55", ls="--", lw=1,
                   label=fr"Floor position $\eta={REFERENCE_RATIO_ETA_MIN:g}$")
    ax.grid(False)

with plt.style.context(hep.style.ATLAS), plt.rc_context({"font.size": 14}):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, squeeze=False)
    for row, systematic in enumerate([False, True]):
        for label, values, style in estimator_curves(systematic):
            sigma = values["sigma_mu"].replace([np.inf, -np.inf], np.nan).where(values["width_valid"])
            axes[row, 0].plot(values["eta"], sigma, label=label, **style)
            axes[row, 1].plot(values["eta"], values["mu_hat"].where(values["fit_valid"]),
                              label=label, **style)
        nuisance_label = "NI profiled" if systematic else "NI fixed at zero"
        axes[row, 0].set(title=nuisance_label, ylabel=r"Local width $\sigma_\mu(\eta)$")
        axes[row, 1].axhline(truth_mu, color="0.65", ls="-.", lw=1, label="Generating value")
        axes[row, 1].set(title=nuisance_label + ": same Asimov experiment", ylabel=r"$\hat\mu(\eta)$")
        subset = estimator_comparison.loc[estimator_comparison["systematics"].eq(systematic)]
        finite_estimates = subset.loc[subset["fit_valid"], "mu_hat"].to_numpy()
        finite_estimates = finite_estimates[np.isfinite(finite_estimates)]
        lower = min(truth_mu, finite_estimates.min()) if len(finite_estimates) else truth_mu
        upper = max(truth_mu, finite_estimates.max()) if len(finite_estimates) else truth_mu
        margin = max(.03 * (run.config["mu_max"] - run.config["mu_min"]), .1 * (upper - lower))
        axes[row, 1].set_ylim(lower - margin, upper + margin)
        for ax in axes[row]:
            mark_estimator_floor(ax)
            ax.set_xlabel(r"Frozen observable anchor $\eta$")
            ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    (ROOT / "plots").mkdir(exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(ROOT / "plots" / f"{PLOT_PREFIX}_estimator_eta.{extension}",
                    bbox_inches="tight", dpi=160)
    plt.show()

    # Same values, with vertical limits set from the displayed low-eta region.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True, squeeze=False)
    for ax, systematic in zip(axes.flat, [False, True]):
        shown_widths = []
        for label, values, style in estimator_curves(systematic):
            sigma = values["sigma_mu"].replace([np.inf, -np.inf], np.nan).where(values["width_valid"])
            ax.plot(values["eta"], sigma, label=label, **style)
            in_view = values["eta"].between(*ESTIMATOR_ZOOM_XLIM)
            shown_widths.extend(sigma.loc[in_view & sigma.notna()].tolist())
        if shown_widths:
            lo, hi = min(shown_widths), max(shown_widths)
            pad = max(.08 * (hi - lo), .02 * max(abs(lo), abs(hi)), 1e-8)
            ax.set_ylim(max(0., lo - pad), hi + pad)
        mark_estimator_floor(ax)
        ax.set(xlim=ESTIMATOR_ZOOM_XLIM, xlabel=r"Frozen observable anchor $\eta$",
               ylabel=r"Local width $\sigma_\mu(\eta)$",
               title="NI profiled" if systematic else "NI fixed at zero")
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(fr"Same Asimov truth $\mu_*={truth_mu:g}$: floor versus no floor", fontsize=16)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(ROOT / "plots" / f"{PLOT_PREFIX}_estimator_eta_floor_comparison_zoom.{extension}",
                    bbox_inches="tight", dpi=160)
    plt.show()
'''

def configure_reference_cells(cells, number):
    """Migrate only relevant controls/text; retain custom axes and bin sweeps."""
    if number == 8 and not any(cell.get('id') == 'reference-model-source' for cell in cells):
        setup_index = next(i for i, cell in enumerate(cells)
                           if 'run = create_run(ROOT,' in cell.source)
        cells[setup_index + 1:setup_index + 1] = [
            nbformat.v4.new_markdown_cell(MODEL_SOURCE_INTRO.strip(), id='reference-model-intro'),
            nbformat.v4.new_code_cell(MODEL_SOURCE_CODE.strip(), id='reference-model-source'),
        ]
    for cell in cells:
        source = cell.source
        if cell.cell_type == "code":
            if "run = create_run(ROOT, mode=MODE, overrides=CONFIG_OVERRIDES)" in source:
                # Keep user-edited numeric controls when regenerating again.
                setup = REFERENCE_SETUP.strip()
                for name in ("N_BINS", "REFERENCE_RATIO_ETA_MIN", "REFERENCE_RATIO_A"):
                    match = re.search(rf"^{name} = (.+)$", source, flags=re.M)
                    if match:
                        setup = re.sub(rf"^{name} = .+$", f"{name} = {match.group(1)}",
                                       setup, flags=re.M)
                line = "run = create_run(ROOT, mode=MODE, overrides=CONFIG_OVERRIDES)"
                begin = source.index(line) + len(line)
                end = source.index("plt.rcParams.update", begin)
                source = source[:begin] + "\n" + setup + "\n" + source[end:]
            if "HIST_BINS =" in source:
                source = re.sub(r"^HIST_BINS = .+$", "HIST_BINS = N_BINS", source, flags=re.M)
                source = source.replace(
                    '"eta": eta, "coherent_SBI_yield"',
                    '"eta": eta, "eta_effective": max(eta, REFERENCE_RATIO_ETA_MIN), "coherent_SBI_yield"')
                source = source.replace(
                    "    z_eta = observable_values", "    # The observable is floored; the physical coherent coefficients below are not.\n    z_eta = observable_values"
                ) if "# The observable is floored" not in source else source
            if "collapse_rows = []" in source:
                source = COLLAPSE_CODE.strip()
            if "run_estimator_study(run," in source:
                source = ESTIMATOR_STUDY_CODE.strip()
            if "binning_run = replace(run," in source:
                source = source.replace('from dataclasses import replace\n', '')
                source = source.replace(
                    'binning_run = replace(run, config={**run.config, "bin_counts": BIN_COUNTS})',
                    'binning_run = run.with_config(bin_counts=BIN_COUNTS)')
                source = source.replace('shares all other settings with notebook 4.',
                                        'preserves the selected models and all other settings.')
            if 'truth_mu = float(estimator_study["truth"].iloc[0])' in source:
                plot_source = ESTIMATOR_PLOT_CODE.strip()
                zoom = re.search(r"^ESTIMATOR_ZOOM_XLIM = (.+)$", source, flags=re.M)
                if zoom:
                    plot_source = re.sub(r"^ESTIMATOR_ZOOM_XLIM = .+$",
                                         "ESTIMATOR_ZOOM_XLIM = " + zoom.group(1),
                                         plot_source, flags=re.M)
                source = plot_source
            if "splines = run_spline_study" in source:
                source = re.sub(r"^SPLINE_BINS = 12.*$",
                                "SPLINE_BINS = 36  # Must appear in notebook 08 BIN_COUNTS.",
                                source, flags=re.M)
            if "# Explicitly surface the singular-anchor" in source or "near_special_anchor =" in source:
                source = SPLINE_CHECKS.strip()
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
        elif source.startswith("# 8."):
            source = r'''# 8. Floored reference-ratio histograms

Use $R_\eta=r(x;\max(\eta,0.2))/r(x;1)$, with power $a=100$ and a primary
resolution of **36 bins**. The binning sweep remains available to compare
coarsening. Physical $\mu$, Asimov truth and free fits retain their full domains.
Reuse the completed run from notebooks 01–03; no generation or training is
needed. Results use `reference_ratio_` and figures `08_`.'''
        elif source.startswith("# 9."):
            source = r'''# 9. Floored reference-ratio splines and profiled fits

Interpolate **36-bin** templates for R_eta=r(x;max(eta,0.2))/r(x;1), with
the same power a=100 as notebook 08. The observable floor leaves the physical
model and NI constraint unchanged. Run 08 first with the same completed run.
Tables use `reference_ratio_` and figures `09_`.'''
        elif source.startswith("### Ratio to the fixed physical reference"):
            source = DEFINITION.strip()
        elif source.startswith(("### Interpolation at the collapsed anchor",
                                "### Interpolation at the floor and collapsed anchor")):
            source = SPLINE_WARNING.strip()
        elif source.startswith("### Runtime and persistent run"):
            for marker in ("For the reference-ratio stretch/bin-edge update alone",
                           "For the 36-bin/floored-observable update", "For this refresh"):
                if marker in source:
                    source = source[:source.index(marker)].rstrip()
            source += "\n\n" + RUNTIME_NOTE.strip()
            if number == 8:
                source = source[:source.index("For the 36-bin/floored-observable update")].rstrip()
                source += "\n\n" + r'''For this refresh, keep the completed v4 run and rerun **08 from the top**.
Select the same Unity directory and member-filter policy as notebook 03 below.
Keep the observable floor at $0.2$, power at $100$, and primary bin count at $36$.
No sample generation, retraining, or new physics run name is required. Stale
notebook outputs have been cleared. Before continuing to downstream notebooks,
select the same ensemble/filter evaluation there as well; matching only the
base run name does not select these new results.'''
        elif source.startswith("### How the observable changes with eta"):
            source = source.replace("observable constructed at $\\eta$",
                                    "observable constructed at $\\eta_{\\mathrm{eff}}=\\max(\\eta,0.2)$")
            source = source.replace(r"\max(\eta,0.1)", r"\max(\eta,0.2)")
            source = source.replace("Edit `ETA_VALUES` and `HIST_BINS` below", "Edit `ETA_VALUES` and `N_BINS` below")
            source = source.replace("Eta=0 is supported", r"$\eta=0$ is supported")
        elif number == 8 and source.startswith("Next: **09_reference_ratio_spline_templates.ipynb**"):
            source = '''Next: **09_reference_ratio_spline_templates.ipynb**. Before using these
results there, configure the same ensemble/filter evaluation and observable
settings. The base run name alone is insufficient. Earlier outputs remain available.'''
        elif source.startswith("### Freeze the observable inside each fit"):
            source = r'''### Freeze the observable inside each fit

For a physical test $\mu_0$, set $\eta=\mu_0$ and construct $z$ using
$\eta_{\mathrm{eff}}=\max(\eta,\eta_{\min})$, at nominal NI. Freeze event assignments, edges and
Asimov counts while fitting physical $\mu$ and, where enabled, NI. The physical
$\mu$ coefficients and fit bounds are never floored. The density denominator
stays at $(1,0)$, independently of the best fit. The binning study fixes NI.
Below $\eta_{\min}$, infinitely fine bins need not recover the physical $\eta$-versus-$1$
unbinned comparison: the numerator endpoint is deliberately $\eta_{\min}$. Above
the floor the original pairwise argument applies, with no guarantee of full
local Fisher information or of preserving every freely fitted alternative.'''
        elif source.startswith("### Finite-separation fidelity versus local information"):
            source = r'''### Finite-separation fidelity versus local information

The likelihood scan tests physical $\eta$ against Asimov truth $1$. Above the floor,
the continuous nominal ratio is sufficient for that pair. Below the floor it
uses the $\eta_{\min}$-versus-$1$ ratio instead. The Fisher table evaluates local
information at physical $\mu=\eta$; no change of the physical parameter is made.
This differs from the later width comparison, which holds the truth at $\mu_*=1$.
The frozen $\eta=1$ observable has zero shape information and one occupied bin,
while the extended information still includes the total rate.

Non-nested bin counts need not improve monotonically. Near $\eta=1$, finite
binning can obscure the continuous-ratio advantage. Compare with `analytic
unbinned`, separately from the learned closure references.'''
        elif source.startswith(("Keep the same estimator resolution", "Use the primary 36-bin estimator resolution")):
            source = r'''Use the primary 36-bin estimator resolution, explicitly passed as `N_BINS`.
The independent `BIN_COUNTS` sweep does not change this estimator setting.
The truth stays fixed while $\eta$ changes; below the floor the partition is
constant. At $\eta=1$ this is a rate-only measurement: multiple physical $\mu$ values
can have equal rates. A different fitted branch need not indicate optimizer
failure. Local widths do not describe global ambiguity; invalid widths are
reported rather than floored.

The next cells explicitly compare this floor with an otherwise identical
**no-floor control** ($\eta_{\mathrm{eff}}=\eta$), using the same 36 bins, quadrature events,
Asimov truth, $\eta$ grid, nuisance treatments and free-fit settings. They show
both binned variants and one analytical unbinned reference. The control writes
its own `reference_ratio_estimator_eta_no_floor` files, preserving the primary
floored estimator outputs. A low-eta zoom highlights the region affected by
the floor; the vertical line marks $\eta_{\min}$.

These are **local widths at the fixed Asimov truth**, not confidence-interval
coverage or a proof of improved global likelihood fits. The free-fit MLE panels
remain a separate diagnostic. Values at $\eta\geq\eta_{\min}$ should agree between
the two binned prescriptions because their partitions are then identical.'''
        elif source.startswith("### Read the comparisons separately"):
            source = source.replace(
                "At truth 1 the continuous nominal ratio preserves the fixed-NI pairwise test.",
                "At truth 1 and eta >= eta_min, the continuous nominal ratio preserves\n"
                "the fixed-NI pairwise test. Below eta_min this guarantee no longer applies\n"
                "to the physical eta-versus-1 comparison.")
        elif source.startswith("### Interpretation") and number == 9:
            source = '''### Interpretation

Compare the 36-bin direct and spline scans, including the exact floor plateau,
the transition just above eta_min, and the separate rate-only collapse at eta=1.
The floor only targets the low-eta compression problem diagnosed in notebook 12;
it does not imply full-family sufficiency or guarantee coverage. Notebook 11
uses direct yields and pseudo-experiments, independently of spline precision.'''
        if number == 8 and cell.cell_type == 'code':
            source = source.replace('ROOT / "results"', 'run.path("results")')
            source = source.replace('ROOT / "plots"', 'run.path("plots")')
        cell.source = source
    return cells


def build_reference_notebooks():
    for number, name in ((8, "08_reference_ratio_histograms.ipynb"),
                         (9, "09_reference_ratio_spline_templates.ipynb")):
        path = REPO / "notebooks" / name
        notebook = nbformat.read(path, as_version=4)
        notebook.cells = configure_reference_cells(notebook.cells, number)
        nbformat.write(notebook, path)
        print(path)


if __name__ == "__main__":
    build_reference_notebooks()
