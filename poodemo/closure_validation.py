"""Independent, high-statistics validation of the frozen density-ratio models.

The simulation bank, the Poisson experiment, and the workspace integration
sample are three different random streams. In particular, neither the learned
ratios nor their workspace normalizers are fitted on either validation sample.
An analytic comparator exposes finite workspace-integration error instead of
making its minimum stationary by using the same points on both sides.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .data import PROCESSES, SAMPLES, file_digest, save_array, save_json
from .inference import TemplateLikelihood, component_coefficients, exp_poly_factor


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _selected_sample(model, selector, threshold, sample, n, rng, directory, chunk):
    """Stream independent draws to a selected memmap; retain rejected counts."""
    directory = Path(directory)
    component, alpha = SAMPLES[sample]
    path = directory / f"{sample}.npy"
    parts = []
    try:
        for start in range(0, n, chunk):
            points = model.sample_component(component, min(chunk, n-start), rng,
                                            alpha_ni=alpha).astype(np.float32)
            accepted = selector.predict_proba(points)[:, 0] >= threshold
            part = directory / f"{sample}_{start}.partial.npy"
            save_array(part, points[accepted])
            parts.append(part)
        counts = [len(np.load(p, mmap_mode="r")) for p in parts]
        temporary = path.with_suffix(".partial.npy")
        output = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32",
                                          shape=(sum(counts), 3))
        offset = 0
        for part, count in zip(parts, counts):
            output[offset:offset+count] = np.load(part, mmap_mode="r")
            offset += count
        output.flush()
        del output
        temporary.replace(path)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)
    return path


def prepare_validation_bank(run, *, n_per_sample=2_000_000, seed=None, directory=None):
    """Generate/cache a new selected sample for each of the six positive sources.

    ``n_per_sample`` counts GENERATED events per source, before the frozen
    selection. Selected counts therefore fluctuate and acceptance is measurable.
    The stream namespace differs from every training/calibration/integration
    stream in ``data.generate_samples``. No run configuration is changed.
    """
    from .training import load_predictor
    n = _positive_integer(n_per_sample, "n_per_sample")
    seed = int(run.config["seed"] + 730_301 if seed is None else seed)
    directory = Path(directory or run.path("results", "density_validation",
                                          f"bank_{seed}_{n}"))
    directory.mkdir(parents=True, exist_ok=True)
    selection = json.loads(run.path("selected", "selection_state.json").read_text())
    selector_digest = file_digest(run.path("models", "preselection", "model.pt"))
    fingerprint = hashlib.sha256((selector_digest + repr(float(selection["threshold"]))).encode()).hexdigest()
    if fingerprint != selection["selection_fingerprint"]:
        raise ValueError("The selector checkpoint changed after preselection; cannot validate against stale selected samples.")
    provenance = {
        "schema": 1, "seed": seed, "stream_namespace": 730301,
        "n_generated_per_source": n, "physics": run.model.to_dict(),
        "selection": selection,
        "selector_sha256": selector_digest,
        "chunk_size": int(run.config["generation_chunk"]),
        "note": "Independent validation only; no weights or calibration are fitted on this bank.",
    }
    manifest = directory / "provenance.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError("Validation-bank provenance changed; choose another diagnostics directory.")
    save_json(manifest, provenance)
    selector = None
    samples = {}
    for index, sample in enumerate(SAMPLES):
        path = directory / f"{sample}.npy"
        if not path.exists():
            if selector is None:
                selector = load_predictor(run.path("models", "preselection"))
            rng = np.random.default_rng(np.random.SeedSequence([seed, 730301, index]))
            _selected_sample(run.model, selector, selection["threshold"], sample,
                             n, rng, directory, provenance["chunk_size"])
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
            raise ValueError(f"Validation source {sample} has fewer than two selected events or invalid shape.")
        samples[sample] = values
        print(f"Validation bank {sample}: {len(values):,} / {n:,} selected", flush=True)
    generated = {sample: n for sample in samples}
    pd.DataFrame([dict(sample=s, generated=n, selected=len(x), efficiency=len(x)/n)
                  for s, x in samples.items()]).to_csv(directory / "counts.csv", index=False)
    return dict(samples=samples, generated_counts=generated, directory=directory,
                provenance=provenance)


def expectation_convergence(values, sizes=None):
    """Prefix means, iid standard errors, and importance-weight ESS.

    Prefix points are correlated. Standard errors describe simulation noise
    conditional on the already-frozen network and calibration, not NN-training
    or calibration uncertainty. Nothing here rescales the supplied ratios.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("At least two finite, nonnegative ratio weights are required")
    if sizes is None:
        sizes = np.unique(np.r_[np.geomspace(min(100, len(values)), len(values), 14).astype(int), len(values)])
    sizes = np.asarray(sizes)
    if sizes.ndim != 1 or np.any(sizes != sizes.astype(int)) or np.any(sizes < 2) or np.any(sizes > len(values)):
        raise ValueError("Convergence sizes must be integer prefixes between 2 and the bank length")
    sizes = np.unique(sizes.astype(int))
    sums, sums2 = np.cumsum(values), np.cumsum(values**2)
    rows = []
    for n in sizes:
        total, second = sums[n-1], sums2[n-1]
        mean = total/n
        se = np.sqrt(max(0., (second-total**2/n)/(n-1))/n)
        rows.append(dict(n_events=int(n), expectation=mean, standard_error=se,
                         residual=mean-1., effective_sample_size=total**2/second if second > 0 else 0.,
                         max_weight_fraction=float(np.max(values[:n])/total) if total > 0 else np.nan))
    return pd.DataFrame(rows)


def ratio_expectation_closure(bank, predictors):
    """Evaluate raw and deployed mean-normalized ratios on fresh denominators."""
    from .pipeline import RATIO_TASKS
    frames = []
    for numerator, denominator in RATIO_TASKS:
        predictor = predictors[f"{numerator}_over_{denominator}"]
        points = bank["samples"][denominator]
        for stage, values in (("raw", predictor.predict_raw_ratio(points)),
                              ("normalized", predictor.predict_ratio(points))):
            frame = expectation_convergence(values)
            frame["ratio"] = f"{numerator}/{denominator}"
            frame["stage"] = stage
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@dataclass
class FrozenWorkspaceFields:
    """Reproduce the existing workspace normalization at new data coordinates."""
    run: object
    quadrature: dict
    predictors: dict

    def __post_init__(self):
        q = self.quadrature
        self.rates = np.asarray(q["nominal"]) @ q["weights"]
        self.rates_down = np.asarray(q["down"])[0] @ q["weights"]
        self.rates_up = np.asarray(q["up"])[0] @ q["weights"]
        self.normalizers = {}
        self.learned_quadrature = self.evaluate(q["x"], establish=True)

    def evaluate(self, x, *, establish=False):
        """Return nominal/down/up intensities using ONLY frozen q normalizers."""
        signal = self.run.model.component_density(x, "S")
        nominal = np.empty((4, len(x)), dtype=float)
        nominal[0] = signal
        for j, process in enumerate(PROCESSES[1:], 1):
            key = f"{process}_over_S"
            shape = signal/self.rates[0]*self.predictors[key].predict_ratio(x)
            if establish:
                self.normalizers[key] = float(shape @ self.quadrature["weights"])
            normalization = self.normalizers[key]
            if not np.isfinite(normalization) or normalization <= 0:
                raise ValueError(f"Invalid frozen workspace normalizer for {key}")
            nominal[j] = self.rates[j]*shape/normalization
        down, up = nominal.copy(), nominal.copy()
        for label, target, rates in (("down", down, self.rates_down), ("up", up, self.rates_up)):
            key = f"NI_{label}_over_NI"
            shape = nominal[3]/self.rates[3]*self.predictors[key].predict_ratio(x)
            if establish:
                self.normalizers[key] = float(shape @ self.quadrature["weights"])
            normalization = self.normalizers[key]
            if not np.isfinite(normalization) or normalization <= 0:
                raise ValueError(f"Invalid frozen workspace normalizer for {key}")
            target[3] = rates[3]*shape/normalization
        return nominal, down, up


class _PreparedMorph:
    """The existing exp-poly factor with array coefficients prepared once."""
    def __init__(self, lo, hi):
        self.lo, self.hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
        if (np.any(self.lo <= 0) or np.any(self.hi <= 0)
                or np.any(~np.isfinite(self.lo)) or np.any(~np.isfinite(self.hi))):
            raise ValueError("Morph anchors must be finite and positive")
        log_hi, log_lo = np.log(self.hi), np.log(self.lo)
        hi_d1, lo_d1 = self.hi*log_hi, -self.lo*log_lo
        hi_d2, lo_d2 = self.hi*log_hi**2, self.lo*log_lo**2
        s0, a0 = (self.hi+self.lo)/2, (self.hi-self.lo)/2
        s1, a1 = (hi_d1+lo_d1)/2, (hi_d1-lo_d1)/2
        s2, a2 = (hi_d2+lo_d2)/2, (hi_d2-lo_d2)/2
        self.coefficients = ((15*a0-7*s1+a2)/8, (-24+24*s0-9*a1+s2)/8,
                             (-5*a0+5*s1-a2)/4, (12-12*s0+7*a1-s2)/4,
                             (3*a0-3*s1+a2)/8, (-8+8*s0-5*a1+s2)/8)

    def __call__(self, alpha):
        with np.errstate(over="ignore", invalid="ignore"):
            if alpha > 1.:
                return np.power(self.hi, alpha)
            if alpha < -1.:
                return np.power(self.lo, -alpha)
            result = self.coefficients[-1].copy()
            for coefficient in self.coefficients[-2::-1]:
                result *= alpha
                result += coefficient
            result *= alpha
            result += 1.
            return result


class IndependentValidationLikelihood(TemplateLikelihood):
    """Extended log likelihood with separate integration and observation arrays.

    ``observation_weights`` may be independent quadrature weights times the
    analytic truth intensity, or one per event for a Poisson experiment.
    ``exposure`` scales the rate for the latter. The inherited multi-start fit
    covers both interference basins. The NI rate and shape morph matches the
    repository model and normalizes exclusively on the frozen workspace nodes.
    """
    def __init__(self, integration_fields, integration_weights, data_fields,
                 observation_weights, *, exposure=1., reference_intensity=None):
        nominal, down, up = [np.asarray(a, dtype=float) for a in integration_fields]
        super().__init__(nominal, down[None], up[None], integration_weights,
                         np.ones(nominal.shape[1]))
        self.data_nominal, self.data_down, self.data_up = [np.asarray(a, dtype=float) for a in data_fields]
        self.observation_weights = np.asarray(observation_weights, dtype=float)
        n = self.data_nominal.shape[1]
        if (self.data_nominal.shape != (4, n) or self.data_down.shape != (4, n)
                or self.data_up.shape != (4, n) or self.observation_weights.shape != (n,)):
            raise ValueError("Data fields must be (4,N), with N observation weights")
        if (not np.isfinite(exposure) or exposure <= 0 or n == 0
                or np.any(~np.isfinite(self.observation_weights)) or np.any(self.observation_weights < 0)):
            raise ValueError("Data weights and exposure must be finite and nonnegative, with positive exposure")
        for a in (self.data_nominal, self.data_down, self.data_up):
            if np.any(~np.isfinite(a)) or np.any(a <= 0):
                raise ValueError("Component data fields must be finite and positive")
        # Every variation changes NI only, as in the existing workspace.
        if not (np.array_equal(down[:3], nominal[:3]) and np.array_equal(up[:3], nominal[:3])):
            raise ValueError("Only the NI nuisance is supported")
        self.exposure = float(exposure)
        self.reference_rate = float(component_coefficients(1.) @ self.rates)
        reference = (component_coefficients(1.) @ self.data_nominal
                     if reference_intensity is None else np.asarray(reference_intensity, dtype=float))
        if reference.shape != (n,) or np.any(~np.isfinite(reference)) or np.any(reference <= 0):
            raise ValueError("The fixed log-likelihood reference must be positive")
        self.reference_intensity = reference
        self.log_reference = np.log(reference)
        self._data_ni_down_ratio = self.data_down[3]/self.data_nominal[3]*self.rates[3]/self.rates_down[0, 3]
        self._data_ni_up_ratio = self.data_up[3]/self.data_nominal[3]*self.rates[3]/self.rates_up[0, 3]
        self._q_ni_down_ratio = self.shapes_down[0, 3]/self.shapes[3]
        self._q_ni_up_ratio = self.shapes_up[0, 3]/self.shapes[3]
        self._ni_q_weights = self.shapes[3]*self.quadrature_weights
        self._q_morph = _PreparedMorph(self._q_ni_down_ratio, self._q_ni_up_ratio)
        self._data_morph = _PreparedMorph(self._data_ni_down_ratio, self._data_ni_up_ratio)
        self._last_alpha = None
        self._last_ni = None

    def _ni(self, alpha):
        if alpha == self._last_alpha:
            return self._last_ni
        if alpha == 0.:
            result = self.data_nominal[3], self.rates[3], self.nominal[3]
        else:
            factors = self._q_morph(alpha)
            data_factors = self._data_morph(alpha)
            rate_factor = float(exp_poly_factor(alpha, self.rates_down[0, 3]/self.rates[3],
                                               self.rates_up[0, 3]/self.rates[3]))
            normalization = float(factors @ self._ni_q_weights)
            if (np.any(factors <= 0) or np.any(data_factors <= 0) or not np.isfinite(normalization)
                    or normalization <= 0 or not np.isfinite(rate_factor) or rate_factor <= 0):
                raise ValueError("Invalid NI morph")
            result = (self.data_nominal[3]*data_factors*rate_factor/normalization,
                      self.rates[3]*rate_factor,
                      self.nominal[3]*factors*rate_factor/normalization)
        self._last_alpha, self._last_ni = alpha, result
        return result

    def nll(self, mu, alpha=None):
        alpha = self._alpha(alpha)
        try:
            coefficient = component_coefficients(mu)
            ni, ni_rate, q_ni = self._ni(float(alpha[0]))
            s, sbi, b, _ = self.data_nominal
            density = coefficient[0]*s + coefficient[1]*sbi + coefficient[2]*b + ni
            rate = float(coefficient[:3] @ self.rates[:3] + ni_rate)
            q_density = (coefficient[0]*self.nominal[0] + coefficient[1]*self.nominal[1]
                         + coefficient[2]*self.nominal[2] + q_ni)
            if (rate <= 0 or not np.isfinite(rate) or np.any(~np.isfinite(density)) or np.any(density <= 0)
                    or np.any(~np.isfinite(q_density)) or np.any(q_density <= 0)):
                return np.inf
            # Parameter-independent reference subtraction improves precision;
            # it does NOT make the analytic or learned minimum occur at one.
            delta = (density-self.reference_intensity)/self.reference_intensity
            near = np.abs(delta) < .5
            log_ratio = np.empty_like(density)
            log_ratio[near] = np.log1p(delta[near])
            log_ratio[~near] = np.log(density[~near])-self.log_reference[~near]
            value = 2*(self.exposure*(rate-self.reference_rate)
                       - np.sum(self.observation_weights*log_ratio))
            return float(value + alpha[0]**2)
        except (ValueError, FloatingPointError):
            return np.inf


def _analytic_fields(model, x):
    return (model.component_densities(x).T,
            model.component_densities(x, alpha_ni=-1.).T,
            model.component_densities(x, alpha_ni=1.).T)


def _expectation_bank(run, bank):
    names = ("S", "B", "NI")
    counts = np.array([bank["generated_counts"][name] for name in names])
    if np.any(counts != counts[0]):
        raise ValueError("Independent expectation quadrature requires equal generated stratum sizes")
    x = np.concatenate([bank["samples"][name] for name in names])
    proposal = sum(run.model.component_pdf(x, name) for name in names)/3.
    weights = 1./(3*counts[0]*proposal)
    truth = run.model.intensity(x, 1.)
    return x, weights*truth


def _poisson_experiment(run, bank, expected_generated):
    from .training import load_predictor
    expected_generated = _positive_integer(expected_generated, "poisson_generated_events")
    exposure = expected_generated/(run.model.component_yield("SBI") + run.model.component_yield("NI"))
    directory = bank["directory"] / f"poisson_{expected_generated}"
    directory.mkdir(exist_ok=True)
    seed = bank["provenance"]["seed"]
    provenance = dict(bank_provenance=bank["provenance"], seed=seed, stream_namespace=730302,
                      expected_generated=expected_generated, exposure=exposure,
                      generating_mu=1., generating_alpha_NI=0.)
    manifest = directory / "provenance.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError("Independent Poisson experiment provenance changed")
    save_json(manifest, provenance)
    counts, samples = {}, []
    selector = None
    for index, source in enumerate(("SBI", "NI")):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 730302, index]))
        n = int(rng.poisson(exposure*run.model.component_yield(source)))
        counts[source] = n
        path = directory / f"{source}.npy"
        if not path.exists():
            if selector is None:
                selector = load_predictor(run.path("models", "preselection"))
            _selected_sample(run.model, selector, bank["provenance"]["selection"]["threshold"],
                             source, n, rng, directory, run.config["generation_chunk"])
        samples.append(np.load(path, mmap_mode="r"))
    data = np.concatenate(samples)
    record = dict(exposure=exposure, n_generated=sum(counts.values()), n_selected=len(data),
                  generated_by_source=counts,
                  selected_by_source=dict(zip(("SBI", "NI"), map(len, samples))),
                  note="Independent Poisson data at mu=1, alpha_NI=0; not a model-generated Asimov sample.")
    save_json(directory / "counts.json", record)
    return data, exposure, record


def scan_independent_likelihood(likelihood, mu_values, *, mu_bounds=(.001, 2.),
                                alpha_bounds=(-3., 3.), profile=True):
    """Profile a fixed model, retaining failures as gaps and all fit diagnostics."""
    values = np.unique(np.r_[np.asarray(mu_values, dtype=float), 1.])
    if np.any(values < mu_bounds[0]) or np.any(values > mu_bounds[1]):
        raise ValueError("Scan values and truth mu=1 must be inside the fitting bounds")
    options = dict(mu_bounds=tuple(mu_bounds), alpha_bounds=tuple(alpha_bounds),
                   mu_starts=np.unique(np.r_[np.linspace(*mu_bounds, 9), 1.]).tolist())
    rows, fits = [], []
    for systematics in ((False, True) if profile else (False,)):
        best = likelihood.fit(systematics=systematics, **options)
        conditional = []
        warm = best.alpha if best.success else np.zeros(1)
        for mu in values:
            point = likelihood.fit(mu_fixed=mu, initial_alpha=warm,
                                   systematics=systematics, **options)
            if point.success:
                warm = point.alpha
            conditional.append(point)
        # Profile scans can discover a remote basin missed by a global start.
        # Refit from it before ever computing or reporting a test statistic.
        finite = [point for point in conditional if point.success and np.isfinite(point.nll)]
        if finite:
            lowest = min(finite, key=lambda point: point.nll)
            if not best.success or lowest.nll < best.nll - 1e-6:
                retry = likelihood.fit(initial_mu=lowest.mu, initial_alpha=lowest.alpha,
                                       systematics=systematics, **options)
                if retry.success and (not best.success or retry.nll < best.nll):
                    best = retry
        unresolved = bool(finite and min(p.nll for p in finite) < best.nll-1e-5)
        valid_global = bool(best.success and not unresolved)
        # A high-statistics minimum can be much narrower than the overview
        # grid. Curvature at fixed nuisance supplies a conservative small
        # scale; profiling can only broaden that local width.
        if valid_global:
            h = max(1e-5, 1e-3*(mu_bounds[1]-mu_bounds[0]))
            left, right = max(mu_bounds[0], best.mu-h), min(mu_bounds[1], best.mu+h)
            curvature = ((likelihood.nll(left, best.alpha)+likelihood.nll(right, best.alpha)-2*best.nll)/h**2
                         if left == best.mu-h and right == best.mu+h else np.nan)
            width = np.sqrt(2./curvature) if np.isfinite(curvature) and curvature > 0 else .03
            radius = min(.25*(mu_bounds[1]-mu_bounds[0]), max(5e-4, 4*width))
            refined = np.unique(np.r_[best.mu, np.linspace(max(mu_bounds[0], best.mu-radius),
                                                          min(mu_bounds[1], best.mu+radius), 31)])
            existing = set(values.tolist())
            warm = best.alpha
            for mu in refined:
                if mu in existing:
                    continue
                point = likelihood.fit(mu_fixed=mu, initial_alpha=warm,
                                       systematics=systematics, **options)
                if point.success:
                    warm = point.alpha
                conditional.append(point)
            if any(point.success and point.nll < best.nll-1e-5 for point in conditional):
                lowest = min((point for point in conditional if point.success), key=lambda point: point.nll)
                retry = likelihood.fit(initial_mu=lowest.mu, initial_alpha=lowest.alpha,
                                       systematics=systematics, **options)
                if retry.success and retry.nll < best.nll:
                    best = retry
                valid_global = bool(best.success and not any(
                    point.success and point.nll < best.nll-1e-5 for point in conditional))
        conditional.sort(key=lambda point: point.mu)
        at_one = next(point for point in conditional if point.mu == 1.)
        fits.append(dict(systematics=systematics, mu_hat=best.mu, alpha_NI_hat=best.alpha[0],
                         nll=best.nll, valid=valid_global, fit_message=best.message,
                         nll_at_generating_point=likelihood.nll(1., [0.]),
                         q_joint_at_truth=likelihood.nll(1., [0.])-best.nll if valid_global else np.nan,
                         q_mu1=at_one.nll-best.nll if valid_global and at_one.success else np.nan,
                         alpha_NI_mu1=at_one.alpha[0]))
        for point in conditional:
            valid = bool(valid_global and point.success and np.isfinite(point.nll))
            rows.append(dict(mu=point.mu, systematics=systematics,
                             q=point.nll-best.nll if valid else np.nan,
                             alpha_NI=point.alpha[0], valid=valid, nll=point.nll,
                             fit_message=point.message))
    return pd.DataFrame(rows), pd.DataFrame(fits)


def plot_expectation_convergence(frame, output):
    import matplotlib.pyplot as plt
    import mplhep as hep
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with plt.style.context(hep.style.ATLAS):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        for ax, (ratio, table) in zip(axes.flat, frame.groupby("ratio", sort=False)):
            for stage, color, label in (("raw", "C0", "Raw network"),
                                        ("normalized", "C1", "Mean-normalized")):
                group = table.loc[table.stage == stage]
                ax.errorbar(group.n_events, group.expectation, yerr=group.standard_error,
                            marker="o", markersize=4, color=color, label=label)
            ax.axhline(1., color="black", linestyle="--", linewidth=1)
            ax.set(xscale="log", xlabel="Selected denominator events", ylabel=r"$\widehat{E}_{\rm den}[r]$",
                   title=ratio)
            ax.legend(fontsize=11)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(.02, .95,
            "Fresh simulation bank\nFrozen networks and calibration\n\nBars: simulation standard error\nNested prefixes are correlated\n\nNo renormalization on this bank.\nHeavy tails can give small ESS;\ninspect the saved CSV table.",
            va="top", fontsize=13)
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"03_ratio_expectation_convergence.{suffix}", dpi=160)
    return fig


def plot_mle_closure(scans, fits, output, *, zoom=False):
    import matplotlib.pyplot as plt
    import mplhep as hep
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    experiments = list(scans.experiment.unique())
    modes = list(scans.systematics.unique())
    with plt.style.context(hep.style.ATLAS):
        fig, axes = plt.subplots(len(experiments), len(modes), figsize=(8*len(modes), 6*len(experiments)),
                                 squeeze=False, constrained_layout=True)
        for i, experiment in enumerate(experiments):
            for j, systematics in enumerate(modes):
                ax = axes[i, j]
                for model, color in (("analytic", "C0"), ("learned", "C1")):
                    group = scans.loc[(scans.experiment == experiment) & (scans.systematics == systematics)
                                      & (scans.model == model)]
                    fitted = fits.loc[(fits.experiment == experiment) & (fits.systematics == systematics)
                                      & (fits.model == model)].iloc[0]
                    label = f"{model.capitalize()}: " + (rf"$\hat\mu={fitted.mu_hat:.4f}$" if fitted.valid else "fit invalid")
                    ax.plot(group.mu, group.q, color=color, label=label)
                    if fitted.valid:
                        ax.plot(fitted.mu_hat, 0., marker="o", color=color)
                ax.axvline(1., color="black", linestyle="--", linewidth=1, label=r"Generated $\mu=1$")
                ax.set(xlabel=r"$\mu$", ylabel=r"$-2\log[L(\mu)/L(\hat\mu)]$")
                ax.set_title(experiment + ("; NI profiled" if systematics else "; NI fixed"), fontsize=17)
                ax.yaxis.label.set_fontsize(18)
                ax.set_ylim(bottom=0.)
                if zoom:
                    local_fits = fits.loc[(fits.experiment == experiment) & (fits.systematics == systematics)
                                          & fits.valid]
                    centers = np.r_[1., local_fits.mu_hat]
                    radius = max(.06, .15*np.ptp(centers))
                    ax.set_xlim(max(0., float(np.min(centers)-radius)), float(np.max(centers)+radius))
                    ax.set_ylim(-.03, 4.)
                ax.legend(fontsize=12)
        for suffix in ("pdf", "png"):
            stem = "03_independent_mle_closure" + ("_zoom" if zoom else "")
            fig.savefig(output / f"{stem}.{suffix}", dpi=160, bbox_inches="tight")
    return fig


def run_closure_validation(run, bank, *, predictors=None, scan_points=81,
                           poisson_generated_events=10_000_000, profile=True):
    """Run both nontrivial closure tests and save tables, provenance, and plots.

    The expected log likelihood retains the original physical exposure. The
    independent Poisson experiment increases luminosity, not the shape or the
    auxiliary constraint. Finite original integration and validation statistics
    can displace even the analytic MLE; compare analytic and learned curves.
    """
    from .pipeline import RATIO_TASKS, prepare_quadrature
    from .training import load_predictor
    scan_points = _positive_integer(scan_points, "scan_points")
    if scan_points < 3:
        raise ValueError("At least three scan points are required")
    if predictors is None:
        predictors = {f"{a}_over_{b}": load_predictor(run.path("models", "ratios", f"{a}_over_{b}"))
                      for a, b in RATIO_TASKS}
    output = Path(bank["directory"]) / "closure"
    output.mkdir(exist_ok=True)
    convergence = ratio_expectation_closure(bank, predictors)
    convergence.to_csv(output / "ratio_expectation_convergence.csv", index=False)
    figures = [plot_expectation_convergence(convergence, output)]
    q = prepare_quadrature(run)
    workspace = FrozenWorkspaceFields(run, q, predictors)
    analytic_q = (np.asarray(q["nominal"]), np.asarray(q["down"])[0], np.asarray(q["up"])[0])
    mu_bounds = tuple(run.config["mu_fit_bounds"])
    mus = np.unique(np.r_[np.linspace(*mu_bounds, scan_points), 1.])
    expectation_x, expectation_weights = _expectation_bank(run, bank)
    poisson_x, poisson_exposure, poisson_record = _poisson_experiment(run, bank, poisson_generated_events)
    records, fit_records = [], []
    for experiment, x, weights, exposure in (
            ("Independent expectation", expectation_x, expectation_weights, 1.),
            ("Independent Poisson sample", poisson_x, np.ones(len(poisson_x)), poisson_exposure)):
        analytic_data = _analytic_fields(run.model, x)
        truth = run.model.intensity(x, 1.)
        for model, integration_fields, data_fields in (
                ("analytic", analytic_q, analytic_data),
                ("learned", workspace.learned_quadrature, workspace.evaluate(x))):
            print(f"Closure scan: {experiment}, {model}; {len(x):,} selected points", flush=True)
            likelihood = IndependentValidationLikelihood(integration_fields, q["weights"], data_fields,
                weights, exposure=exposure, reference_intensity=truth)
            scans, fits = scan_independent_likelihood(likelihood, mus, mu_bounds=mu_bounds,
                alpha_bounds=run.config["nuisance_bounds"], profile=profile)
            for frame in (scans, fits):
                frame["experiment"], frame["model"] = experiment, model
                frame["exposure_multiplier"] = exposure
                frame["selected_points"] = len(x)
                frame["observed_event_mass"] = float(weights.sum())
            records.append(scans)
            fit_records.append(fits)
    scans, fits = pd.concat(records, ignore_index=True), pd.concat(fit_records, ignore_index=True)
    scans.to_csv(output / "independent_unbinned_scans.csv", index=False)
    fits.to_csv(output / "independent_unbinned_fits.csv", index=False)
    # Hash every persisted ratio checkpoint/calibrator without exposing contents.
    fingerprints = {}
    for a, b in RATIO_TASKS:
        directory = run.path("models", "ratios", f"{a}_over_{b}")
        fingerprints[f"{a}_over_{b}"] = {str(p.relative_to(directory)): file_digest(p)
            for p in sorted(directory.rglob("*")) if p.is_file()
            and p.name in ("model.pt", "ensemble.json", "calibrator.joblib")}
    save_json(output / "provenance.json", {
        "bank": bank["provenance"], "network_fingerprints": fingerprints,
        "workspace_quadrature": {name: file_digest(run.path("quadrature", name+".npy"))
                                 for name in ("x", "weights", "nominal", "down", "up")},
        "frozen_normalizers": workspace.normalizers,
        "nominal_selected_rates": dict(zip(PROCESSES, workspace.rates.tolist())),
        "expected_log_likelihood_points": len(expectation_x),
        "expected_event_mass": float(expectation_weights.sum()),
        "expectation_effective_sample_size": float(expectation_weights.sum()**2/np.sum(expectation_weights**2)),
        "poisson_experiment": poisson_record, "scan_points": scan_points,
        "mu_bounds": list(mu_bounds), "profile_NI": bool(profile),
        "normalization": "All rates and shape normalizers are frozen on the original workspace quadrature; no evaluation-bank normalization.",
        "interpretation": "Neither MLE is forced to one. Analytic displacement diagnoses finite original integration/data noise; additional learned displacement diagnoses density-model error. The Poisson curve also has physical sampling fluctuations.",
    })
    figures.append(plot_mle_closure(scans, fits, output))
    figures.append(plot_mle_closure(scans, fits, output, zoom=True))
    return dict(convergence=convergence, scans=scans, fits=fits, figures=figures,
                directory=output, poisson=poisson_record)
