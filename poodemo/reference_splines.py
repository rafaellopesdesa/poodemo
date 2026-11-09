"""Constrained reference-ratio template interpolation for notebook 09.

Eta selects a frozen partition, never the physical fit parameter. The exact
weighted empirical CDF uses the rescaled coordinate u.
Convex, degree-one interpolation in eta preserves coordinate monotonicity.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import logit

from . import pipeline
from .data import PROCESSES, save_json
from .inference import FitResult, component_coefficients, poisson_asimov_nll

STUDY_VERSION = 3
METHODS = ("ecdf_u",)
LABELS = {"direct": "Direct fills", "ecdf_u": "Exact weighted empirical CDF in u"}
VARIATIONS = ("nominal", "NI_down", "NI_up")


@dataclass(frozen=True)
class SplineStudySettings:
    initial_anchors: int = 81
    focus_anchors: int = 81
    focus_halfwidth: float = .1
    max_anchors: int = 4801
    max_rounds: int = 12
    max_bin_pull: float = .05
    max_deviance: float = .005
    q_tolerance: float = .01
    mle_tolerance: float = .02  # Diagnostic; a nearly flat likelihood can move its MLE.
    likelihood_refine_points: int = 32  # In addition to the displayed scan.
    validation_intervals: int = 96
    validation_fit_points: int = 8
    surface_points: int = 3
    surface_mu_points: int = 81
    surface_alpha_points: int = 41
    cdf_cache_size: int = 2  # Full event-coordinate CDFs; keep CPU RAM bounded.

    def __post_init__(self):
        for key in ("initial_anchors", "focus_anchors", "max_anchors", "max_rounds",
                    "likelihood_refine_points", "validation_intervals", "validation_fit_points", "surface_points",
                    "surface_mu_points", "surface_alpha_points", "cdf_cache_size"):
            value = getattr(self, key)
            minimum = 2 if key in ("initial_anchors", "focus_anchors", "max_anchors",
                                    "surface_mu_points", "surface_alpha_points") else 1
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        for key in ("focus_halfwidth", "max_bin_pull", "max_deviance", "q_tolerance", "mle_tolerance"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")


def rescaled_log_ratio(quad, eta):
    """log[p_eta/p_1]/(eta-1), with stable algebra and the exact score limit.

    I_eta-I_1 = (eta-1) [S + (SBI-S-B)/(sqrt(eta)+1)]. This avoids
    subtracting almost equal log densities, including within machine epsilon
    of eta=1. Densities and their total rates are normalized on the same bank.
    The caller applies the observable floor before calling this function.
    """
    eta = float(eta)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    s, sbi, b, ni = quad["nominal"]
    reference = sbi + ni
    derivative = s + (sbi - s - b) / (np.sqrt(eta) + 1.)
    local = derivative / reference
    average = (derivative @ quad["weights"]) / (reference @ quad["weights"])
    delta = eta - 1.
    if delta == 0:
        return local - average
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (np.log1p(delta * local) - np.log1p(delta * average)) / delta
    if np.any(~np.isfinite(result)):
        raise ValueError("The rescaled-coordinate study needs positive finite densities on the bank")
    return result


class TemplateBank:
    """Shared direct templates and a two-entry LRU of full-resolution u CDFs."""

    def __init__(self, run, quad, edges, cdf_cache_size=2):
        self.run, self.quad, self.edges = run, quad, np.asarray(edges, float)
        settings = pipeline.observable_configuration(run, "reference_ratio")
        self.floor = settings["reference_ratio_eta_floor"]
        self.power = settings["reference_ratio_power"]
        self.weighted = np.stack([quad["nominal"], quad["down"][0], quad["up"][0]]) * quad["weights"]
        self.totals = self.weighted.sum(axis=-1)
        self.cache_size = cdf_cache_size
        self.histograms, self.cdf_cache = {}, OrderedDict()
        self.central = np.searchsorted(self.edges, .5, side="right") - 1
        if not self.edges[self.central] < .5 < self.edges[self.central + 1]:
            raise ValueError("The equality collapse must lie strictly inside a bin")

    def exact(self, eta):
        eta = max(float(eta), self.floor)
        if eta not in self.histograms:
            z = pipeline.observable_values(self.run, self.quad, eta, "reference_ratio")
            self.histograms[eta] = pipeline._histogram_anchors(self.quad, z, self.edges)
        return self.histograms[eta]

    def counts(self, eta):
        # Independently refill observations; never manufacture data from a spline.
        z = pipeline.observable_values(self.run, self.quad, eta, "reference_ratio")
        return np.histogram(z, self.edges, weights=self.quad["weights"] * self.quad["truth"])[0]

    def _cdf(self, anchor):
        anchor = max(float(anchor), self.floor)
        key = anchor
        if key in self.cdf_cache:
            self.cdf_cache.move_to_end(key)
            return self.cdf_cache[key]
        # Discard before constructing another large CDF, bounding peak RAM.
        while len(self.cdf_cache) >= self.cache_size:
            self.cdf_cache.popitem(last=False)
        u = rescaled_log_ratio(self.quad, anchor)
        order = np.argsort(u, kind="stable")
        coordinates, starts = np.unique(u[order], return_index=True)
        flat = self.weighted.reshape(-1, len(u))
        # Identical nominal/varied S, SBI and B share one CDF. This also makes
        # their numerical support exactly identical across nuisance anchors.
        unique, mapping = [], []
        for row in flat:
            match = next((i for i, previous in enumerate(unique) if np.array_equal(row, previous)), None)
            if match is None:
                match = len(unique)
                unique.append(row)
            mapping.append(match)
        weights = np.asarray(unique)[:, order]
        grouped = np.add.reduceat(weights, starts, axis=1)
        cumulative = np.cumsum(grouped, axis=1)
        # Raw prefix sums include the empty prefix and every tied mass.
        record = (coordinates, np.column_stack([np.zeros(len(unique)), cumulative]), np.asarray(mapping))
        self.cdf_cache[key] = record
        return record

    def cdf_u_bins(self, anchor, eta):
        """Integrate the anchor ECDF over the query eta's ORIGINAL z bins."""
        if not np.isfinite(eta) or eta == 1.:
            raise ValueError("Query eta must be finite and different from 1; fill the collapse exactly")
        coordinates, cumulative, mapping = self._cdf(anchor)
        boundaries = logit(self.edges) / (self.power * (eta - 1.))
        reverse = eta < 1.
        if reverse:
            boundaries = boundaries[::-1]
        # [lo,hi) above eta=1, (lo,hi] below. Infinite endpoints retain all
        # finite coordinates, including events whose z rounds to zero or one.
        indices = np.searchsorted(coordinates, boundaries, side="right" if reverse else "left")
        bins = np.diff(cumulative[:, indices], axis=-1)[mapping].reshape(
            *self.totals.shape, len(boundaries) - 1)
        return bins[..., ::-1] if reverse else bins


class CumulativeTemplates:
    """Exact empirical u CDFs, with convex linear eta interpolation.

    Original z-bin boundaries move in u with the query eta. The representation
    is exact on its finite bank at eta anchors, up to arithmetic;
    interpolation between anchors still requires independent validation.
    """

    def __init__(self, bank, etas):
        self.bank, self.etas = bank, np.asarray(etas, float)
        if len(self.etas) < 2 or np.any(~np.isfinite(self.etas)) or np.any(np.diff(self.etas) <= 0):
            raise ValueError("Expected finite increasing eta anchors")
        self.totals = bank.totals

    def _one(self, eta):
        if not np.isfinite(eta) or eta < self.etas[0] or eta > self.etas[-1]:
            raise ValueError("eta is outside the spline anchor range; extrapolation is forbidden")
        if eta <= self.bank.floor or eta == 1.:
            return self.bank.exact(eta).copy()
        right = int(np.clip(np.searchsorted(self.etas, eta, side="right"), 1, len(self.etas) - 1))
        left, right = self.etas[right - 1], self.etas[right]
        fraction = (eta - left) / (right - left)
        low = self.bank.cdf_u_bins(left, eta)
        high = self.bank.cdf_u_bins(right, eta) if fraction != 0 else low
        return (1. - fraction) * low + fraction * high

    def __call__(self, eta):
        values = np.asarray(eta, float)
        if values.ndim == 0:
            return self._one(float(values))
        return np.stack([self._one(float(v)) for v in values.flat]).reshape(
            *values.shape, *self.totals.shape, len(self.bank.edges) - 1)


def initial_eta_grid(run, settings):
    floor = pipeline.observable_configuration(run, "reference_ratio")["reference_ratio_eta_floor"]
    high = float(run.config["mu_max"])
    if high <= 1 or floor >= high:
        raise ValueError("The reference comparison needs mu_max > 1 and floor < mu_max")
    grid = np.r_[0., floor, 1., np.linspace(floor, high, settings.initial_anchors),
                 np.linspace(max(floor, 1. - settings.focus_halfwidth),
                             min(high, 1. + settings.focus_halfwidth), settings.focus_anchors)]
    grid = np.unique(grid)
    # Independent linspaces can produce the same intended knot one ULP apart.
    # Preserve the exact special points and do not create degenerate intervals.
    for special in (0., floor, 1., high):
        grid[np.abs(grid - special) < 32 * np.finfo(float).eps * max(1., abs(special))] = special
    grid = np.unique(grid)
    grid = grid[np.r_[True, np.diff(grid) > 32 * np.finfo(float).eps]]
    if len(grid) > settings.max_anchors:
        raise ValueError("max_anchors is smaller than the initial grid")
    return grid


def make_interpolator(bank, etas, method):
    if method not in METHODS:
        raise ValueError(f"Unknown interpolation method: {method}")
    return CumulativeTemplates(bank, etas)


def template_metrics(exact, predicted, probe_mus):
    """No mean floors: the +1 only defines a diagnostic error scale in counts."""
    valid = np.all(np.isfinite(predicted)) and np.all(predicted >= 0)
    pull = float(np.max(np.abs(predicted - exact) / np.sqrt(exact + 1.))) if valid else np.inf
    deviation = 0.
    support_mismatch = 0
    for mu in probe_mus:
        coeff = component_coefficients(mu)
        for expected, approximate in zip(exact, predicted):
            data, model = coeff @ expected, coeff @ approximate
            if np.any(data < 0):
                raise ValueError("Direct physical template is negative on a validation probe")
            deviation = max(deviation, 2. * poisson_asimov_nll(model, data))
            support_mismatch += int(np.count_nonzero((data > 0) & (model <= 0)))
    return dict(max_bin_pull=pull, max_deviance=deviation, support_mismatch=support_mismatch)


def metric_pass(metrics, settings):
    return bool(metrics["max_bin_pull"] <= settings.max_bin_pull
                and metrics["max_deviance"] <= settings.max_deviance
                and metrics["support_mismatch"] == 0)


def likelihood_refinement_grid(run, grid, floor, count):
    """Declare optimization points before adaptation; golden points stay held out."""
    eligible = np.flatnonzero(np.asarray(grid)[1:] > floor)
    selected = eligible[np.unique(np.linspace(0, len(eligible)-1, min(count, len(eligible))).astype(int))]
    quarters = np.asarray(grid)[selected] + .25*np.diff(grid)[selected]
    points = np.unique(np.r_[pipeline._positive_study_grid(run), quarters])
    return points[(points > floor) & (points != 1.)]


def _refinement_anchor(low, high, optimization_points):
    # Do not make a likelihood optimization point an exact interpolation knot.
    for fraction in (.5, .43, .57):
        candidate = low + fraction*(high-low)
        if not np.any(np.abs(optimization_points-candidate) < 1e-13):
            return candidate
    raise FloatingPointError("Unable to separate refinement and fit points")


def adapt_templates(bank, grid, method, settings, probe_mus, progress=print, *, likelihood_points=None):
    """Refine on template midpoints and declared q checks, never held-out points."""
    etas, history, fit_checks = np.asarray(grid).copy(), [], []
    likelihood_points = np.asarray([] if likelihood_points is None else likelihood_points, float)
    direct_fits = {}
    for round_number in range(settings.max_rounds + 1):
        progress(f"{LABELS[method]}: round {round_number} ({len(etas)} anchors)")
        model = make_interpolator(bank, etas, method)
        additions, template_failures, likelihood_failures = {}, 0, 0
        worst_score = 0.
        def request(low, high, score):
            candidate = _refinement_anchor(low, high, likelihood_points)
            additions[candidate] = max(score, additions.get(candidate, 0.))
        for index, (low, high) in enumerate(zip(etas[:-1], etas[1:])):
            if high <= bank.floor:
                continue
            eta = (low + high) / 2
            metrics = template_metrics(bank.exact(eta), model(eta), probe_mus)
            score = max(metrics["max_bin_pull"] / settings.max_bin_pull,
                        metrics["max_deviance"] / settings.max_deviance)
            worst_score = max(worst_score, score)
            if not metric_pass(metrics, settings):
                template_failures += 1
                request(low, high, score)
            if (index+1) % 100 == 0:
                progress(f"  {index+1} midpoint checks complete")
        for index, eta in enumerate(likelihood_points):
            if eta not in direct_fits:
                direct = pipeline._histogram_model(bank.exact(eta), bank.counts(eta))
                direct_fits[eta] = {s: _fit(bank.run, direct, eta, s) for s in (False, True)}
            approximate = pipeline._histogram_model(model(eta), bank.counts(eta))
            for syst in (False, True):
                reference = direct_fits[eta][syst]
                fit = _fit(bank.run, approximate, eta, syst)
                valid = bool(reference['valid'] and fit['valid'])
                delta = fit['q']-reference['q'] if valid else np.nan
                score = abs(delta)/settings.q_tolerance if valid else np.inf
                worst_score = max(worst_score, score)
                passed = bool(valid and abs(delta) <= settings.q_tolerance)
                fit_checks.append(dict(method=method, round=round_number, eta=eta,
                    systematics=syst, q_direct=reference['q'], q_interpolated=fit['q'],
                    delta_q=delta, valid=valid, passed=passed,
                    message=fit['message'], direct_message=reference['message']))
                if not passed:
                    likelihood_failures += 1
                    right = int(np.clip(np.searchsorted(etas, eta, side='right'), 1, len(etas)-1))
                    request(etas[right-1], etas[right], score)
            if (index+1) % 20 == 0:
                progress(f"  {index+1}/{len(likelihood_points)} likelihood refinement points complete")
        history.append(dict(method=method, round=round_number, n_anchors=len(etas),
                            n_failed=template_failures, q_checks=2*len(likelihood_points),
                            q_failed=likelihood_failures, worst_score=worst_score))
        progress(f"  {template_failures} template and {likelihood_failures} likelihood checks failed")
        if not additions or round_number == settings.max_rounds or len(etas) >= settings.max_anchors:
            break
        candidates = sorted(additions, key=additions.get, reverse=True)
        etas = np.unique(np.r_[etas, candidates[:settings.max_anchors-len(etas)]])
    stop = "tolerances_met" if not additions else (
        "anchor_limit" if len(etas) >= settings.max_anchors else "round_limit")
    history[-1]["stop_reason"] = stop
    model.refinement_checks = pd.DataFrame(fit_checks)
    return model, pd.DataFrame(history)


def heldout_eta_grid(model, floor, count, excluded=()):
    """Select intervals after training, evaluate an unused irrational fraction."""
    low, high = model.etas[:-1], model.etas[1:]
    eligible = np.flatnonzero(high > floor)
    selected = eligible[np.unique(np.linspace(0, len(eligible) - 1, min(count, len(eligible))).astype(int))]
    # Reserve extra resolution where the observable contracts, on both sides.
    focus = eligible[(low[eligible] < 1.06) & (high[eligible] > .94)]
    if len(focus):
        selected = np.unique(np.r_[selected, focus[np.unique(np.linspace(
            0, len(focus) - 1, min(count, len(focus))).astype(int))], eligible[:4]])
    fraction = (3. - np.sqrt(5.)) / 2.
    points = low[selected] + fraction * (high[selected] - low[selected])
    return points[[not np.any(np.abs(np.asarray(excluded)-point) < 1e-13) for point in points]]


def _fit(run, model, mu, systematics):
    """Retain both fit objectives and nuisance MLEs, not only their difference."""
    try:
        options = dict(systematics=systematics, mu_bounds=tuple(run.config["mu_fit_bounds"]),
                       alpha_bounds=tuple(run.config["nuisance_bounds"]),
                       mu_starts=pipeline.mu_fit_starts(run))
        null = model.fit(mu_fixed=mu, **options)
        null_optimizer_success = bool(null.success)

        def certify(fit, candidate):
            # KL + squared auxiliary residual is nonnegative. At the exact
            # collapse, roundoff can make a line search report ABNORMAL at a
            # certified global minimum. Accept ONLY a feasible candidate
            # attaining that lower bound to 1e-10 in -2 log L, and record it.
            # This is not a generic override of optimizer failures or a forced q.
            if fit.success or not options['alpha_bounds'][0] <= 0 <= options['alpha_bounds'][1]:
                return fit, False
            if not options['mu_bounds'][0] <= candidate <= options['mu_bounds'][1]:
                return fit, False
            value = _safe_nll(model, candidate, 0.)
            if not 0. <= value <= 1e-10:
                return fit, False
            return FitResult(float(candidate), np.zeros(1), value, True,
                             'Certified nonnegative NLL bound; optimizer: ' + str(fit.message), fit.nfev), True

        null, null_certified = certify(null, mu)
        best = model.fit(initial_mu=mu, initial_alpha=null.alpha, **options)
        best_optimizer_success = bool(best.success)
        best, best_certified = certify(best, run.config['asimov_mu'])
        difference = null.nll - best.nll
        valid = bool(null.success and best.success and np.isfinite(null.nll)
                     and np.isfinite(best.nll) and difference >= -1e-6)
        bounds = np.asarray(run.config["mu_fit_bounds"])
        return dict(q=max(0., float(difference)) if valid else np.nan, valid=valid,
                    mu_hat=best.mu, alpha_hat=float(best.alpha[0]),
                    alpha_null=float(null.alpha[0]), nll_best=best.nll, nll_null=null.nll,
                    null_optimizer_success=null_optimizer_success, best_optimizer_success=best_optimizer_success,
                    null_certified=null_certified, best_certified=best_certified,
                    boundary=bool(np.min(np.abs(best.mu - bounds)) < 1e-5),
                    message=str(null.message) + "; " + str(best.message)
                    + ('; unrestricted objective exceeds conditional' if difference < -1e-6 else ''))
    except (ValueError, FloatingPointError, RuntimeError) as error:
        return dict(q=np.nan, valid=False, mu_hat=np.nan, alpha_hat=np.nan,
                    alpha_null=np.nan, nll_best=np.nan, nll_null=np.nan,
                    null_optimizer_success=False, best_optimizer_success=False,
                    null_certified=False, best_certified=False, boundary=False, message=str(error))


def _safe_nll(model, mu, alpha):
    try:
        return model.nll(float(mu), [float(alpha)])
    except (ValueError, FloatingPointError):
        return np.inf


def _fingerprint(bank):
    digest = hashlib.sha256()
    for value in (bank.quad["x"], bank.quad["weights"], bank.weighted, bank.quad["truth"]):
        data = np.ascontiguousarray(value)
        digest.update(str((data.shape, data.dtype.str)).encode())
        digest.update(memoryview(data).cast("B"))
    return digest.hexdigest()


def run_reference_spline_comparison(run, n_bins=36, *, settings=None, progress=print):
    """Run notebook 09's benchmark without changing notebook 08 or legacy splines.

    Artifacts live in reference_ratio_interpolation/ within the selected model
    evaluation. Saved grids and direct anchors are accompanied by the CDF recipe
    and quadrature fingerprint; u CDFs can be rebuilt from the existing bank.
    No method is automatically promoted to downstream inference.
    """
    settings = settings or SplineStudySettings()
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    fisher_path = run.path("results", "reference_ratio_binning_fisher.csv")
    if not fisher_path.exists():
        raise FileNotFoundError("Run notebook 08 with this same ensemble/filter evaluation first")
    fisher = pd.read_csv(fisher_path)
    pipeline._validate_reference_binning_configuration(run, "reference_ratio", fisher)
    if n_bins not in set(fisher.n_bins):
        raise ValueError("Include the requested bin count in notebook 08 first")
    unbinned = pipeline._saved_unbinned(run)  # Fail before costly interpolation if missing.
    quad = pipeline.prepare_quadrature(run)
    bank = TemplateBank(run, quad, pipeline.observable_bin_edges(n_bins, "reference_ratio"), settings.cdf_cache_size)
    grid = initial_eta_grid(run, settings)
    optimization_points = likelihood_refinement_grid(run, grid, bank.floor, settings.likelihood_refine_points)
    # Probe exact physical coefficients at both fit bounds, the rate turnover,
    # the Asimov truth, and intermediate strengths, at all three NI anchors.
    bounds = np.asarray(run.config["mu_fit_bounds"], float)
    probe_mus = np.unique(np.r_[np.linspace(*bounds, 5), run.config["asimov_mu"]])
    output = run.path("results", "reference_ratio_interpolation")
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(schema_version=3, settings=asdict(settings),
                    observable=pipeline.observable_configuration(run, "reference_ratio"),
                    bin_edges=bank.edges.tolist(), n_bins=n_bins,
                    quadrature_fingerprint=_fingerprint(bank),
                    probe_mus=probe_mus.tolist(), evaluation_root=str(run.path("results").parent),
                    cdf_recipes={"ecdf_u": "exact weighted empirical CDF in u; grouped ties and raw prefix sums"},
                    likelihood_refinement_points=optimization_points.tolist(),
                    eta_interpolation="convex linear; query eta maps original z edges into each anchor's u CDF",
                    anchor_archive_contents="Direct input histograms; the empirical CDF also requires the quadrature and recipe",
                    status="running", methods=list(METHODS))
    provenance = run.path("results", "reference_ratio_model_source.json")
    if provenance.exists():
        metadata["model_source"] = json.loads(provenance.read_text())
    save_json(output / "metadata.json", metadata)
    models, histories = {}, []
    for method in METHODS:
        model, history = adapt_templates(bank, grid, method, settings, probe_mus, progress,
                                         likelihood_points=optimization_points)
        models[method] = model
        histories.append(history)
        history.to_csv(output / f"{method}_refinement.csv", index=False)
        np.savez_compressed(output / f"{method}_anchors.npz", method=method, etas=model.etas, edges=bank.edges,
                            bin_yields=np.stack([bank.exact(e) for e in model.etas]), totals=bank.totals)

    validation, worst_bins, fit_candidates = [], [], []
    for method, model in models.items():
        heldout = heldout_eta_grid(model, bank.floor, settings.validation_intervals, excluded=optimization_points)
        # The exact empirical CDF should close at its anchors to roundoff.
        points = [(e, "heldout") for e in heldout]
        points += [(e, "anchor_coordinate_check") for e in model.etas]
        for point_number, (eta, role) in enumerate(sorted(points), start=1):
            exact, prediction = bank.exact(eta), model(eta)
            metrics = template_metrics(exact, prediction, probe_mus)
            row = dict(method=method, eta=eta, role=role, **metrics,
                       passed=metric_pass(metrics, settings))
            validation.append(row)
            delta = prediction - exact
            order = np.argsort(np.abs(delta).ravel())[-8:][::-1]
            for index in order:
                vi, pi, bi = np.unravel_index(index, exact.shape)
                worst_bins.append(dict(method=method, eta=eta, role=role, variation=VARIATIONS[vi],
                                       sample=PROCESSES[pi], bin=bi, direct=exact[vi, pi, bi],
                                       interpolated=prediction[vi, pi, bi], error=delta[vi, pi, bi]))
            if point_number % 100 == 0:
                progress(f"{LABELS[method]}: {point_number}/{len(points)} validation points complete")
        table = pd.DataFrame([r for r in validation if r["method"] == method and r["role"] == "heldout"])
        # These checks are never fed back into adaptation. Fit the worst ones
        # as well as evenly spaced reserved points, using identical data.
        fit_candidates.extend(table.nlargest(2, "max_deviance").eta.tolist())
        if len(heldout):
            fit_candidates.extend(heldout[np.unique(np.linspace(
                0, len(heldout) - 1, min(settings.validation_fit_points, len(heldout))).astype(int))])
        progress(f"{LABELS[method]}: independent validation complete")
    validation = pd.DataFrame(validation)
    validation.to_csv(output / "template_validation.csv", index=False)
    pd.DataFrame(worst_bins).to_csv(output / "worst_bins.csv", index=False)

    scan_grid = pipeline._positive_study_grid(run)
    fit_etas = np.unique(np.r_[scan_grid, fit_candidates])
    scans, fixed_checks, bin_errors = [], [], []
    alpha_probes = np.unique(np.r_[run.config['nuisance_bounds'], -1.5, -1., -.5, 0., .5, 1., 1.5])
    for point_number, eta in enumerate(fit_etas, start=1):
        exact, counts = bank.exact(eta), bank.counts(eta)
        direct = pipeline._histogram_model(exact, counts)
        for method in ("direct", *METHODS):
            prediction = exact if method == "direct" else models[method](eta)
            model = direct if method == "direct" else pipeline._histogram_model(prediction, counts)
            is_anchor = method != "direct" and bool(np.any(np.abs(models[method].etas - eta) < 1e-13))
            for systematic in (False, True):
                scans.append(dict(method=method, eta=eta, mu=eta, systematics=systematic,
                                  is_anchor=is_anchor, used_for_refinement=bool(np.any(np.abs(optimization_points-eta) < 1e-13)),
                                  on_scan=bool(np.any(np.abs(scan_grid - eta) < 1e-13)),
                                  **_fit(run, model, eta, systematic)))
            if method == "direct":
                continue
            # Compare raw objectives at identical parameters and data: fitting
            # away a template error must not disguise lack of closure.
            for mu in np.unique(np.r_[probe_mus, eta]):
                for alpha in alpha_probes:
                    a, b = _safe_nll(direct, mu, alpha), _safe_nll(model, mu, alpha)
                    fixed_checks.append(dict(method=method, eta=eta, mu=mu, alpha_NI=alpha,
                                             nll_direct=a, nll_interpolated=b,
                                             delta_nll=b-a if np.isfinite(a) and np.isfinite(b) else np.nan,
                                             finite_direct=bool(np.isfinite(a)), finite_interpolated=bool(np.isfinite(b))))
            at_truth = component_coefficients(run.config["asimov_mu"]) @ prediction[0]
            for bi, (data, predicted) in enumerate(zip(counts, at_truth)):
                bin_errors.append(dict(method=method, eta=eta, bin=bi, observed=data, expected=predicted,
                                       error=predicted-data, diagnostic_pull=(predicted-data)/np.sqrt(data+1.)))
        progress(f"Direct/interpolated fits complete at eta={eta:.7g}")
        if point_number % 10 == 0:
            pd.DataFrame(scans).to_csv(output / "scans.csv", index=False)
    scans = pd.DataFrame(scans)
    reference = scans.loc[scans.method == "direct"]
    comparison = scans.loc[scans.method != "direct"].merge(
        reference[["eta", "systematics", "q", "mu_hat", "valid", "nll_best", "alpha_hat", "boundary"]],
        on=["eta", "systematics"], suffixes=("", "_direct"))
    comparison["delta_q"] = comparison.q - comparison.q_direct
    comparison["delta_mu_hat"] = comparison.mu_hat - comparison.mu_hat_direct
    comparison["q_passed"] = (comparison.valid & comparison.valid_direct
                                & (comparison.delta_q.abs() <= settings.q_tolerance))
    comparison["mle_shift_flag"] = comparison.delta_mu_hat.abs() > settings.mle_tolerance

    # Select full 2D surfaces from the actual worst fit discrepancies. They use
    # one common saturated likelihood offset, never separately shifted minima.
    ranked = comparison.assign(rank=comparison.delta_q.abs().fillna(np.inf)).groupby(
        "eta")["rank"].max().nlargest(settings.surface_points).index
    surfaces = []
    base_mu_mesh = np.r_[np.linspace(*bounds, settings.surface_mu_points), run.config["asimov_mu"]]
    alpha_bounds = run.config['nuisance_bounds']
    base_alpha_mesh = np.r_[np.linspace(*alpha_bounds, settings.surface_alpha_points), 0.,
        np.linspace(max(alpha_bounds[0], -.15), min(alpha_bounds[1], .15), settings.surface_alpha_points)]
    for eta in sorted(ranked):
        counts = bank.counts(eta)
        fitted = scans.loc[(scans.eta == eta) & scans.systematics & scans.valid]
        # The full domain and a denser NI neighborhood share the actual fitted
        # points. A coarse regular mesh must not miss a narrow minimum.
        mu_mesh = np.unique(np.r_[base_mu_mesh, eta, fitted.mu_hat])
        alpha_mesh = np.unique(np.r_[base_alpha_mesh, fitted.alpha_hat, fitted.alpha_null])
        for method in ("direct", *METHODS):
            anchors = bank.exact(eta) if method == "direct" else models[method](eta)
            model = pipeline._histogram_model(anchors, counts)
            for alpha in alpha_mesh:
                for mu in mu_mesh:
                    surfaces.append(dict(method=method, eta=eta, mu=mu, alpha_NI=alpha,
                                         nll=_safe_nll(model, mu, alpha)))
        progress(f"Full mu/NI surfaces complete at eta={eta:.7g}")

    # Preserve the previous separation of nuisance-morph order from template
    # interpolation; this comparison uses direct histograms only.
    parent = pipeline._template_likelihood(quad, systematics=True)
    morph = []
    for eta in scan_grid[[0, len(scan_grid)//2, -1]]:
        z = pipeline.observable_values(run, quad, eta, "reference_ratio")
        before, after = parent.binned(z, bank.edges, "before", eta), parent.binned(z, bank.edges, "after", eta)
        for alpha in (-1., -.5, .5, 1., 1.5):
            a, b = before.intensity(eta, [alpha]), after.intensity(eta, [alpha])
            morph.append(dict(eta=eta, alpha_NI=alpha,
                              l1_relative_difference=float(np.abs(a-b).sum()/a.sum()),
                              nll_difference=after.nll(eta, [alpha])-before.nll(eta, [alpha])))
    summary = []
    fixed_frame = pd.DataFrame(fixed_checks)
    for method, model in models.items():
        v = validation.loc[validation.method == method]
        all_fits = comparison.loc[comparison.method == method]
        c = all_fits.loc[~all_fits.is_anchor]
        reserved_fits = c.loc[~c.used_for_refinement & ~c.on_scan]
        fixed = fixed_frame.loc[fixed_frame.method == method]
        domain_mismatches = int((fixed.finite_direct != fixed.finite_interpolated).sum())
        truth = fixed.loc[(fixed.mu == run.config["asimov_mu"]) & (fixed.alpha_NI == 0.)]
        truth_nll = float(truth.nll_interpolated.max())
        history = next(h for h in histories if h.method.iloc[0] == method)
        heldout = v.loc[v.role == "heldout"]
        anchors = v.loc[v.role == "anchor_coordinate_check"]
        summary.append(dict(method=method, n_anchors=len(model.etas),
                            stop_reason=history.stop_reason.iloc[-1],
                            heldout_checks=len(heldout), heldout_failures=int((~heldout.passed).sum()),
                            anchor_checks=len(anchors), anchor_failures=int((~anchors.passed).sum()),
                            fit_checks=len(all_fits), off_anchor_fits=len(c),
                            q_failures=int((~all_fits.q_passed).sum()),
                            heldout_fit_checks=len(reserved_fits),
                            heldout_q_failures=int((~reserved_fits.q_passed).sum()),
                            invalid_fits=int((~all_fits.valid | ~all_fits.valid_direct).sum()),
                            max_abs_delta_q=float(all_fits.delta_q.abs().max()),
                            mle_shift_flags=int(all_fits.mle_shift_flag.sum()),
                            domain_mismatches=domain_mismatches, max_truth_nll=truth_nll,
                            passed=bool(history.stop_reason.iloc[-1] == "tolerances_met"
                                        and v.passed.all() and len(c) and all_fits.q_passed.all()
                                        and domain_mismatches == 0 and truth_nll <= settings.max_deviance)))
    frames = dict(summary=pd.DataFrame(summary), validation=validation, scans=scans,
                  comparison=comparison, fixed_checks=fixed_frame,
                  bin_errors=pd.DataFrame(bin_errors), surfaces=pd.DataFrame(surfaces),
                  refinement=pd.concat(histories, ignore_index=True),
                  likelihood_refinement=pd.concat([m.refinement_checks for m in models.values()], ignore_index=True),
                  morph_comparison=pd.DataFrame(morph),
                  unbinned=unbinned)
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    metadata.update(status="complete", all_methods_passed=bool(frames["summary"].passed.all()))
    save_json(output / "metadata.json", metadata)
    progress("Closure summary:\n" + frames["summary"].to_string(index=False))
    return dict(**frames, models=models, bank=bank, output=output, metadata=metadata)
