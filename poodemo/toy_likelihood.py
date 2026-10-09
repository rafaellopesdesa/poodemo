"""Stat-only extended likelihoods for genuine (possibly weighted) toy data.

Unlike the Asimov likelihood, the rate integral is supplied separately from
observed events. All component arrays use the order S, SBI, B, NI. The fit
uses kappa=sqrt(mu), including the physical boundary mu=0, and searches all
minima resolved by its kappa grid rather than starting a single local fit.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def _fields(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 4:
        raise ValueError(f"{name} must have shape (4,N), order S,SBI,B,NI")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return values


def _polynomial(fields):
    return np.stack((fields[0], fields[1] - fields[0] - fields[2],
                     fields[2] + fields[3]))


def _negative_intervals(polynomial):
    """Merge open kappa intervals where any nonnegative-anchor model is <0.

    Precomputing these intervals makes checks of a large, independent
    validation bank inexpensive during every toy fit. Pointwise zeros are
    allowed here; zeros with positive observations are checked separately.
    """
    a, b, c = polynomial
    scale = np.maximum.reduce((a, np.abs(b), c))
    active = (b < 0) & (scale > 0)
    a, b, c = (row[active] / scale[active] for row in (a, b, c))
    disc = b * b - 4 * a * c
    quadratic = (a > 0) & (disc > 0)
    root = np.sqrt(disc[quadratic])
    denominator = -b[quadratic] + root
    starts = list(2 * c[quadratic] / denominator)
    ends = list(denominator / (2 * a[quadratic]))
    linear = a == 0
    starts.extend(c[linear] / -b[linear])
    ends.extend(np.full(np.count_nonzero(linear), np.inf))
    if not starts:
        return np.empty((0, 2))
    return _merge_intervals(np.column_stack((starts, ends)))


def _merge_intervals(intervals):
    if len(intervals) == 0:
        return np.empty((0, 2))
    starts, ends = np.asarray(intervals, dtype=float).T
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    running_end = np.maximum.accumulate(ends)
    groups = np.r_[0, np.flatnonzero(starts[1:] >= running_end[:-1]) + 1]
    last = np.r_[groups[1:] - 1, len(starts) - 1]
    return np.column_stack((starts[groups], running_end[last]))


class StatOnlyLikelihood:
    """Extended likelihood with independently normalized process intensities.

    ``fields[:, i]`` are component intensities at observed event ``i``;
    ``rates`` are their integrals over the selected region, obtained without
    using this toy's observed data. ``observation_weights`` can be integer
    multiplicities of unique bank points, fractional Asimov weights, or
    omitted for ordinary individual events. ``exposure`` scales model rates
    and intensities, but does not scale supplied observations.

    ``validity_fields`` optionally checks positivity on an independent bank,
    including locations absent from a particular toy. This is a bank-level
    diagnostic, not proof of positivity everywhere in continuous space.
    Negative predictions are never clipped. For bins use ``from_binned``.
    To reuse a large bank's positivity check across toys, construct an empty
    observation likelihood with that ``validity_fields`` once and pass its
    ``negative_intervals`` as ``validity_intervals`` to subsequent likelihoods.
    """

    def __init__(self, fields, rates, observation_weights=None, *, exposure=1.0,
                 validity_fields=None, validity_intervals=None):
        fields = _fields(fields, "fields")
        rates = np.asarray(rates, dtype=np.float64)
        if rates.shape != (4,) or np.any(~np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("rates must contain four finite nonnegative integrals")
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure must be finite and positive")
        weights = (np.ones(fields.shape[1]) if observation_weights is None
                   else np.asarray(observation_weights, dtype=np.float64))
        if weights.shape != (fields.shape[1],) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("observation_weights must have shape (N,) and be finite and nonnegative")
        self.exposure = float(exposure)
        self.rates = rates.copy()
        self.rate_polynomial = _polynomial(rates[:, None])[:, 0] * exposure
        polynomial = _polynomial(fields)
        # Empty bins have no logarithmic contribution, but their intensities
        # still enter positivity checks and the separately supplied rates.
        positive = weights > 0
        self.polynomial = polynomial[:, positive].copy()
        self.observation_weights = weights[positive].copy()
        checks = [polynomial, self.rate_polynomial[:, None]]
        if validity_fields is not None:
            checks.append(_polynomial(_fields(validity_fields, "validity_fields")))
        intervals = _negative_intervals(np.concatenate(checks, axis=1))
        if validity_intervals is not None:
            supplied = np.asarray(validity_intervals, dtype=float)
            if (supplied.ndim != 2 or supplied.shape[1] != 2
                    or np.any(~np.isfinite(supplied[:, 0]))
                    or np.any(np.isnan(supplied[:, 1]))
                    or np.any(supplied[:, 0] < 0)
                    or np.any(supplied[:, 1] <= supplied[:, 0])):
                raise ValueError("validity_intervals must have shape (K,2), with 0 <= start < end")
            intervals = _merge_intervals(np.concatenate((intervals, supplied)))
        self.negative_intervals = intervals
        self._reference_cache = {}

    @classmethod
    def from_binned(cls, component_bin_means, counts, *, exposure=1.0):
        """Independent Poisson-bin likelihood, including zero-count bins."""
        fields = _fields(component_bin_means, "component_bin_means")
        return cls(fields, fields.sum(axis=1), counts, exposure=exposure)

    def _valid_kappa(self, kappa):
        if not np.isfinite(kappa) or kappa < 0:
            return False
        if len(self.negative_intervals):
            index = np.searchsorted(self.negative_intervals[:, 0], kappa, side="left") - 1
            if index >= 0 and kappa < self.negative_intervals[index, 1]:
                return False
        return True

    def _intensity(self, kappa):
        a, b, c = self.polynomial
        return (a * kappa + b) * kappa + c

    def _rate(self, kappa):
        a, b, c = self.rate_polynomial
        return float((a * kappa + b) * kappa + c)

    def expected_count(self, mu):
        if not np.isfinite(mu) or mu < 0:
            raise ValueError("mu must be finite and nonnegative")
        return self._rate(np.sqrt(mu))

    def nll(self, mu, *, reference_mu=None):
        """Return -2 log L, or its stable difference relative to a hypothesis.

        Parameter-independent terms (including log(exposure) per event) are
        omitted. Invalid intensities return infinity, including zero intensity
        at an event or bin with a positive observation.
        """
        if not np.isfinite(mu) or mu < 0:
            return np.inf
        kappa = np.sqrt(mu)
        if not self._valid_kappa(kappa):
            return np.inf
        intensity = self._intensity(kappa)
        rate = self._rate(kappa)
        if rate < 0 or not np.isfinite(rate) or np.any(~np.isfinite(intensity)) or np.any(intensity <= 0):
            return np.inf
        if reference_mu is None:
            return float(2 * (rate - np.sum(self.observation_weights * np.log(intensity))))
        if not np.isfinite(reference_mu) or reference_mu < 0:
            raise ValueError("reference_mu must be finite and nonnegative")
        reference = np.sqrt(reference_mu)
        if not self._valid_kappa(reference):
            raise ValueError("reference hypothesis has negative model intensity")
        if reference_mu not in self._reference_cache:
            self._reference_cache[reference_mu] = self._intensity(reference)
        reference_intensity = self._reference_cache[reference_mu]
        if self._rate(reference) < 0 or np.any(reference_intensity <= 0):
            raise ValueError("reference hypothesis has zero intensity at positive observations")
        # Evaluate lambda(k)-lambda(k0) algebraically, without subtracting
        # two nearly equal intensities or two large absolute log likelihoods.
        delta_k = kappa - reference
        delta = delta_k * (self.polynomial[0] * (kappa + reference) + self.polynomial[1])
        delta_rate = delta_k * (self.rate_polynomial[0] * (kappa + reference) + self.rate_polynomial[1])
        relative = delta / reference_intensity
        close = np.abs(relative) < 0.5
        log_ratio = np.empty_like(relative)
        log_ratio[close] = np.log1p(relative[close])
        log_ratio[~close] = np.log(intensity[~close]) - np.log(reference_intensity[~close])
        return float(2 * (delta_rate - np.sum(self.observation_weights * log_ratio)))

    def _allowed_intervals(self, lower, upper):
        intervals = []
        cursor = lower
        for start, end in self.negative_intervals:
            if end <= cursor or start >= upper:
                continue
            if start >= cursor:
                intervals.append((cursor, min(start, upper)))
            cursor = max(cursor, end)
            if cursor >= upper:
                break
        if cursor <= upper:
            intervals.append((cursor, upper))
        return intervals

    def fit(self, *, mu_bounds=(0.0, 2.0), grid_size=65, reference_mu=None,
            extra_mu=(), kappa_tolerance=1e-10):
        """Search all resolved minima and boundaries in sqrt(mu).

        Each positivity-allowed interval gets a grid. Every discrete local
        minimum is refined with a bounded scalar fit; all interval boundaries
        and ``extra_mu`` points are candidates too. Increase ``grid_size`` to
        check numerical convergence for an unfamiliar model.
        """
        bounds = np.asarray(mu_bounds, dtype=float)
        if bounds.shape != (2,) or np.any(~np.isfinite(bounds)) or bounds[0] < 0 or bounds[1] <= bounds[0]:
            raise ValueError("mu_bounds must be increasing, finite and nonnegative")
        if isinstance(grid_size, bool) or int(grid_size) != grid_size or grid_size < 5:
            raise ValueError("grid_size must be an integer >=5")
        if not np.isfinite(kappa_tolerance) or kappa_tolerance <= 0:
            raise ValueError("kappa_tolerance must be finite and positive")
        lower, upper = np.sqrt(bounds)
        intervals = self._allowed_intervals(lower, upper)
        cache = {}
        failures = 0

        def objective(kappa):
            key = float(kappa)
            if key not in cache:
                cache[key] = self.nll(key * key, reference_mu=reference_mu)
            return cache[key]

        for lo, hi in intervals:
            if hi < lo:
                continue
            if hi == lo:
                objective(lo)
                continue
            # Include a point immediately inside each edge when an observed
            # intensity vanishes exactly on a positivity boundary.
            grid = np.unique(np.r_[np.linspace(lo, hi, int(grid_size)),
                                   np.nextafter(lo, hi), np.nextafter(hi, lo)])
            values = np.asarray([objective(kappa) for kappa in grid])
            for i in range(1, len(grid) - 1):
                if np.isfinite(values[i]) and values[i] <= values[i - 1] and values[i] <= values[i + 1]:
                    result = minimize_scalar(objective, bounds=(grid[i - 1], grid[i + 1]),
                                             method="bounded", options={"xatol": kappa_tolerance})
                    failures += int(not result.success)
                    objective(result.x)
        for mu in extra_mu:
            if np.isfinite(mu) and bounds[0] <= mu <= bounds[1]:
                objective(np.sqrt(mu))
        finite = [(value, kappa) for kappa, value in cache.items() if np.isfinite(value)]
        if not finite:
            return {"mu_hat": np.nan, "nll": np.inf, "valid": False,
                    "message": "No valid model hypothesis in the fit domain",
                    "at_lower": False, "at_upper": False,
                    "n_evaluations": len(cache), "optimizer_failures": failures}
        minimum, best = min(finite)
        boundary_tolerance = max(10 * kappa_tolerance, 1e-8 * (upper - lower))
        return {"mu_hat": float(best * best), "nll": float(minimum),
                "valid": failures == 0,
                "message": ("All resolved minima and boundaries searched" if failures == 0
                            else f"{failures} local minimizations did not converge"),
                "at_lower": bool(abs(best - lower) <= boundary_tolerance),
                "at_upper": bool(abs(best - upper) <= boundary_tolerance),
                "n_evaluations": len(cache), "optimizer_failures": failures}

    def test_statistic(self, mu_test, *, mu_bounds=(0.0, 2.0), grid_size=65,
                       kappa_tolerance=1e-10):
        """Two-sided q_mu=-2 log[L(mu)/L(mu_hat)] (no upper-limit truncation)."""
        if not np.isfinite(mu_test) or not mu_bounds[0] <= mu_test <= mu_bounds[1]:
            raise ValueError("mu_test must be finite and inside mu_bounds")
        if not np.isfinite(self.nll(mu_test)):
            return {"q": np.nan, "mu_hat": np.nan, "nll": np.inf, "valid": False,
                    "message": "Test hypothesis has invalid intensity or zero support for observed events",
                    "at_lower": False, "at_upper": False, "n_evaluations": 1,
                    "optimizer_failures": 0}
        result = self.fit(mu_bounds=mu_bounds, grid_size=grid_size,
                          reference_mu=mu_test, extra_mu=(mu_test,),
                          kappa_tolerance=kappa_tolerance)
        result["q"] = float(-result["nll"]) if result["valid"] else np.nan
        # The tested point is explicitly a candidate with relative NLL=0.
        # This max removes only signed zero, not negative-intensity predictions.
        if result["valid"]:
            result["q"] = max(0.0, result["q"])
        return result
