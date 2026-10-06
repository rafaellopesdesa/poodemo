"""Array likelihoods for the parameterized-observable demonstrations.

Component arrays have shape ``(4, n_points)`` in the order S, SBI, B, NI.
Nuisance anchor arrays have shape ``(n_nuisance, 4, n_points)``.  Their
entries are intensities, not normalized densities or classifier ratios.

The exp--poly coefficients below match nsbi-lhc-toolkit commit
fc09848fc6540fd32310faebbe9db6eea7ecd17b, ``_poly_interp`` and
``_exp_extrap`` in ``models/sbi_parametric_model.py``.  The toolkit morphs
rates and normalized shapes separately.  We additionally normalize every
morphed shape on the common quadrature, so its integral is exactly one.
This explicit normalization is also used by the toolkit adapter.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, minimize_scalar

COMPONENTS = ("S", "SBI", "B", "NI")


def component_coefficients(mu):
    """Physical coefficients, including the signed interference combination."""
    mu = float(mu)
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("mu must be finite and nonnegative")
    root = np.sqrt(mu)
    return np.array([mu - root, root, 1.0 - root, 1.0])


def exp_poly_factor(alpha, down_ratio, up_ratio):
    """Toolkit degree-six interpolation for |alpha|<=1, exponential outside.

    Return the multiplicative factor, rather than the toolkit helper's
    factor-minus-one.  Ratios must be strictly positive.  No clipping of
    the polynomial is performed: clipping would change the model.
    """
    a = float(alpha)
    lo, hi = np.broadcast_arrays(
        np.asarray(down_ratio, dtype=float), np.asarray(up_ratio, dtype=float)
    )
    if not np.isfinite(a) or np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        raise ValueError("interpolation inputs must be finite")
    if np.any(lo <= 0) or np.any(hi <= 0):
        raise ValueError("exp-poly anchor ratios must be strictly positive")
    with np.errstate(over="ignore", invalid="ignore"):
        if a > 1:
            return np.power(hi, a)
        if a < -1:
            return np.power(lo, -a)
        log_hi, log_lo = np.log(hi), np.log(lo)
        hi_d1, lo_d1 = hi * log_hi, -lo * log_lo
        hi_d2, lo_d2 = hi * log_hi**2, lo * log_lo**2
        s0, a0 = (hi + lo) / 2, (hi - lo) / 2
        s1, a1 = (hi_d1 + lo_d1) / 2, (hi_d1 - lo_d1) / 2
        s2, a2 = (hi_d2 + lo_d2) / 2, (hi_d2 - lo_d2) / 2
        c1 = (15 * a0 - 7 * s1 + a2) / 8
        c2 = (-24 + 24 * s0 - 9 * a1 + s2) / 8
        c3 = (-5 * a0 + 5 * s1 - a2) / 4
        c4 = (12 - 12 * s0 + 7 * a1 - s2) / 4
        c5 = (3 * a0 - 3 * s1 + a2) / 8
        c6 = (-8 + 8 * s0 - 5 * a1 + s2) / 8
        return 1 + a * (c1 + a * (c2 + a * (c3 + a * (c4 + a * (c5 + a * c6)))))


def exp_poly_interpolate(alpha, nominal, down, up):
    """Morph positive values; jointly empty entries remain exactly empty."""
    nominal, down, up = np.broadcast_arrays(
        np.asarray(nominal, dtype=float), np.asarray(down, dtype=float),
        np.asarray(up, dtype=float),
    )
    if np.any(nominal < 0) or np.any(down < 0) or np.any(up < 0):
        raise ValueError("component anchors must be nonnegative")
    active = nominal > 0
    if np.any((~active) & ((down > 0) | (up > 0))):
        raise ValueError("a zero nominal anchor cannot have nonzero variations")
    lo = np.divide(down, nominal, out=np.ones_like(nominal), where=active)
    hi = np.divide(up, nominal, out=np.ones_like(nominal), where=active)
    return nominal * exp_poly_factor(alpha, lo, hi)


def poisson_asimov_nll(expected, observed, weights=None):
    """Extended Asimov NLL relative to the saturated model.

    For bins, use expected yields, observed (possibly fractional) counts,
    and no weights.  For unbinned quadrature, use target/truth intensities
    and integration weights ``1/(N*q_proposal)``.  The same weights then
    determine the rate integral and the weighted data term.
    """
    expected, observed = np.broadcast_arrays(
        np.asarray(expected, dtype=float), np.asarray(observed, dtype=float)
    )
    if np.any(~np.isfinite(observed)) or np.any(observed < 0):
        raise ValueError("Asimov observations must be finite and nonnegative")
    if (np.any(~np.isfinite(expected)) or np.any(expected < 0)
            or np.any((expected == 0) & (observed > 0))):
        return np.inf
    # The ordinary x-y+y*log(y/x) formula catastrophically cancels near an
    # Asimov minimum.  In that region write x=y*(1+d), and evaluate
    # y*(d-log1p(d)); a short series resolves even very small displacements.
    terms = expected.copy()
    positive = observed > 0
    x, y = expected[positive], observed[positive]
    with np.errstate(over="ignore", invalid="ignore"):
        delta = (x - y) / y
    close = np.abs(delta) < 1e-3
    medium = (~close) & (np.abs(delta) < 0.5)
    far = ~(close | medium)
    contribution = np.empty_like(delta)
    d = delta[close]
    contribution[close] = y[close] * d**2 * (
        0.5 + d * (-1 / 3 + d * (0.25 + d * (-0.2 + d * (1 / 6 + d * (-1 / 7 + d / 8)))))
    )
    d = delta[medium]
    contribution[medium] = y[medium] * (d - np.log1p(d))
    # Separate logarithms also avoid rounding x/y to zero in a distant tail.
    contribution[far] = x[far] - y[far] + y[far] * (np.log(y[far]) - np.log(x[far]))
    terms[positive] = contribution
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("integration weights must be finite and nonnegative")
        terms = terms * weights
    return float(np.sum(terms))


def _bin_indices(scores, edges):
    scores, edges = np.asarray(scores, float), np.asarray(edges, float)
    if scores.ndim != 1 or edges.ndim != 1 or len(edges) < 2:
        raise ValueError("scores and edges must be one-dimensional")
    if np.any(np.diff(edges) <= 0) or np.any(~np.isfinite(scores)):
        raise ValueError("edges must increase and scores must be finite")
    if np.any(scores < edges[0]) or np.any(scores > edges[-1]):
        raise ValueError("bin edges must cover every score, including tails")
    index = np.searchsorted(edges, scores, side="right") - 1
    return np.minimum(index, len(edges) - 2)


def histogram_components(scores, components, quadrature_weights, edges):
    """Integrate component intensities into frozen score bins; output (4,B)."""
    index = _bin_indices(scores, edges)
    components, weights = np.asarray(components, float), np.asarray(quadrature_weights, float)
    if components.ndim != 2 or components.shape[1] != len(index) or weights.shape != (len(index),):
        raise ValueError("components must be (C,N) and weights (N,)")
    return np.stack([
        np.bincount(index, weights=weights * row, minlength=len(edges) - 1)
        for row in components
    ])


@dataclass
class FitResult:
    """Fit result; ``nll`` uses the toolkit convention -2 log L."""
    mu: float
    alpha: np.ndarray
    nll: float
    success: bool
    message: str
    nfev: int

    @property
    def parameters(self):
        return np.r_[self.mu, self.alpha]


class TemplateLikelihood:
    """Rate-and-shape exp-poly model on a common integration quadrature.

    ``truth_intensity`` is evaluated independently at the generating
    parameters.  For an Asimov sample of this morphing model, construct it
    with ``model.intensity(mu_true, alpha_true)`` and set ``aux_center`` to
    ``alpha_true``.  A truth sample from the analytic shifted-mean model
    instead tests morphing approximation; these are distinct studies.
    """

    def __init__(self, nominal, down, up, quadrature_weights, truth_intensity,
                 aux_center=None, aux_sigma=None, normalize_shapes=True):
        self.nominal = np.asarray(nominal, float).copy()
        self.down, self.up = np.asarray(down, float).copy(), np.asarray(up, float).copy()
        self.quadrature_weights = np.asarray(quadrature_weights, float).copy()
        self.truth_intensity = np.asarray(truth_intensity, float).copy()
        if self.nominal.ndim != 2 or self.nominal.shape[0] != 4:
            raise ValueError("nominal must have shape (4,N), order S,SBI,B,NI")
        if self.down.ndim != 3 or self.down.shape[1:] != self.nominal.shape or self.up.shape != self.down.shape:
            raise ValueError("down/up must have shape (K,4,N)")
        n = self.nominal.shape[1]
        if self.quadrature_weights.shape != (n,) or self.truth_intensity.shape != (n,):
            raise ValueError("quadrature weights and truth must have shape (N,)")
        for values in (self.nominal, self.down, self.up, self.quadrature_weights, self.truth_intensity):
            if np.any(~np.isfinite(values)) or np.any(values < 0):
                raise ValueError("anchors, integration weights, and truth must be finite and nonnegative")
        if np.sum(self.quadrature_weights) <= 0:
            raise ValueError("quadrature has no positive weight")
        self.n_nuisance = self.down.shape[0]
        self.aux_center = np.zeros(self.n_nuisance) if aux_center is None else np.asarray(aux_center, float).copy()
        self.aux_sigma = np.ones(self.n_nuisance) if aux_sigma is None else np.asarray(aux_sigma, float).copy()
        if self.aux_center.shape != (self.n_nuisance,) or self.aux_sigma.shape != (self.n_nuisance,):
            raise ValueError("auxiliary arrays must have shape (K,)")
        if np.any(self.aux_sigma <= 0) or np.any(~np.isfinite(self.aux_center)) or np.any(np.isnan(self.aux_sigma)):
            raise ValueError("auxiliary means must be finite and standard deviations positive")
        self.normalize_shapes = bool(normalize_shapes)
        self.rates = self.nominal @ self.quadrature_weights
        self.rates_down = self.down @ self.quadrature_weights
        self.rates_up = self.up @ self.quadrature_weights
        if np.any(self.rates <= 0) or np.any(self.rates_down <= 0) or np.any(self.rates_up <= 0):
            raise ValueError("all component anchor integrals must be positive")
        self.shapes = self.nominal / self.rates[:, None]
        self.shapes_down = self.down / self.rates_down[:, :, None]
        self.shapes_up = self.up / self.rates_up[:, :, None]
        self.asimov_weights = self.quadrature_weights * self.truth_intensity

    def _alpha(self, alpha):
        values = np.zeros(self.n_nuisance) if alpha is None else np.asarray(alpha, float)
        if values.shape != (self.n_nuisance,) or np.any(~np.isfinite(values)):
            raise ValueError("alpha must have shape (K,) and be finite")
        return values

    def components(self, alpha=None):
        alpha = self._alpha(alpha)
        rates, shapes = self.rates.copy(), self.shapes.copy()
        active = self.shapes > 0
        for k, value in enumerate(alpha):
            rates *= exp_poly_factor(value, self.rates_down[k] / self.rates,
                                     self.rates_up[k] / self.rates)
            if np.any((~active) & ((self.shapes_down[k] > 0) | (self.shapes_up[k] > 0))):
                raise ValueError("shape anchors have different support")
            lo = np.divide(self.shapes_down[k], self.shapes, out=np.ones_like(shapes), where=active)
            hi = np.divide(self.shapes_up[k], self.shapes, out=np.ones_like(shapes), where=active)
            shapes *= exp_poly_factor(value, lo, hi)
        if np.any(~np.isfinite(shapes)) or np.any(shapes < 0) or np.any(~np.isfinite(rates)) or np.any(rates <= 0):
            raise ValueError("exp-poly morph produced invalid component intensities")
        if self.normalize_shapes:
            integrals = shapes @ self.quadrature_weights
            if np.any(integrals <= 0):
                raise ValueError("morphed shape has no positive integral")
            shapes = shapes / integrals[:, None]
        return rates[:, None] * shapes

    def intensity(self, mu, alpha=None):
        return component_coefficients(mu) @ self.components(alpha)

    def expected_count(self, mu, alpha=None):
        return float(self.intensity(mu, alpha) @ self.quadrature_weights)

    def nll(self, mu, alpha=None):
        """Return -2 log L relative to the saturated Asimov model."""
        alpha = self._alpha(alpha)
        try:
            expected = self.intensity(mu, alpha)
        except (ValueError, FloatingPointError):
            return np.inf
        main = poisson_asimov_nll(expected, self.truth_intensity, self.quadrature_weights)
        constraint = np.sum(((alpha - self.aux_center) / self.aux_sigma)**2)
        return float(2 * main + constraint)

    def fit(self, mu_fixed=None, initial_mu=1.0, initial_alpha=None,
            mu_bounds=(0.0, 5.0), alpha_bounds=(-3.0, 3.0),
            mu_starts=None, maxiter=500, systematics=True):
        """Fit a frozen model, using multiple POI starts for interference scans.

        With ``systematics=False``, nuisance values stay at ``initial_alpha``
        (zero by default); otherwise all supplied nuisance parameters profile.
        """
        a0 = self._alpha(initial_alpha)
        bounds_alpha = [tuple(alpha_bounds)] * self.n_nuisance
        if mu_fixed is not None and (not systematics or self.n_nuisance == 0):
            value = self.nll(mu_fixed, a0)
            return FitResult(float(mu_fixed), a0.copy(), value, np.isfinite(value), "No free parameters", 1)
        if not systematics or self.n_nuisance == 0:
            return self._fit_scalar(a0, initial_mu, mu_bounds, mu_starts, maxiter)
        if mu_fixed is not None:
            component_coefficients(mu_fixed)
            starts = [a0]
            bounds = bounds_alpha
            unpack = lambda z: (float(mu_fixed), np.asarray(z))
        else:
            seeds = [initial_mu] if mu_starts is None else list(mu_starts)
            seeds = list(dict.fromkeys([initial_mu] + seeds))
            starts = [np.r_[np.clip(m, *mu_bounds), a0] for m in seeds]
            bounds = [tuple(mu_bounds)] + bounds_alpha
            unpack = lambda z: (float(z[0]), np.asarray(z[1:]))
        results = []
        for start in starts:
            def objective(z):
                value = self.nll(*unpack(z))
                return value if np.isfinite(value) else 1e100
            result = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7})
            if not result.success:
                fallback = minimize(objective, result.x, method="Powell", bounds=bounds,
                                    options={"maxiter": maxiter, "ftol": 1e-10, "xtol": 1e-7})
                if fallback.success or fallback.fun <= result.fun:
                    result = fallback
            results.append(result)
        converged = [result for result in results if result.success and result.fun < 1e99]
        best = min(converged or results, key=lambda result: result.fun)
        # A failed search with a materially better objective is unresolved,
        # rather than grounds to report a worse converged local minimum.
        unresolved = [result for result in results
                      if not result.success and result.fun < best.fun - 1e-6]
        if unresolved:
            best = min(unresolved, key=lambda result: result.fun)
        mu, alpha = unpack(best.x)
        value = self.nll(mu, alpha)
        return FitResult(mu, alpha.copy(), value, bool(best.success and np.isfinite(value)),
                         str(best.message), int(sum(result.nfev for result in results)))

    def _fit_scalar(self, alpha, initial_mu, mu_bounds, mu_starts, maxiter):
        """Bracket each grid-resolved minimum, including physical boundaries.

        Scalar Brent refinement avoids unreliable finite-difference gradients
        at exact Asimov minima.  Multiple brackets, rather than one bounded
        unimodal search, retain possible interference-induced minima.
        """
        low, high = map(float, mu_bounds)
        if not np.isfinite(low + high) or low < 0 or not low < high:
            raise ValueError("mu_bounds must be finite, increasing, and nonnegative")
        seeds = [] if mu_starts is None else list(mu_starts)
        grid = np.unique(np.r_[np.linspace(low, high, 65),
                               np.clip([initial_mu, *seeds], low, high)])
        cache = {}

        def objective(mu):
            mu = float(mu)
            if mu < low or mu > high:
                return np.inf
            if mu not in cache:
                cache[mu] = self.nll(mu, alpha)
            return cache[mu]

        values = np.array([objective(mu) for mu in grid])
        candidates, failures = [], []
        if np.isfinite(values[0]) and values[0] <= values[1]:
            candidates.append((grid[0], values[0], "Constrained scalar minimum at lower bound"))
        if np.isfinite(values[-1]) and values[-1] <= values[-2]:
            candidates.append((grid[-1], values[-1], "Constrained scalar minimum at upper bound"))
        for i in range(1, len(grid) - 1):
            if not np.isfinite(values[i]) or values[i] > min(values[i - 1], values[i + 1]):
                continue
            strict = values[i] < min(values[i - 1], values[i + 1])
            if strict and np.all(np.isfinite(values[i-1:i+2])):
                result = minimize_scalar(objective, method="brent", bracket=tuple(grid[i-1:i+2]),
                                         options={"xtol": 1e-10, "maxiter": maxiter})
            else:
                result = minimize_scalar(objective, method="bounded", bounds=(grid[i-1], grid[i+1]),
                                         options={"xatol": 1e-10, "maxiter": maxiter})
            if result.success and np.isfinite(result.fun):
                candidates.append((float(result.x), float(result.fun), "Converged bracketed scalar minimum"))
                # Flat plateaux, including an unidentified POI, may retain an
                # equally good sampled point; the successful bounded refinement
                # verifies this bracket rather than accepting a failed fit.
                if values[i] <= result.fun and abs(grid[i] - result.x) < 1e-7 * max(1.0, high-low):
                    candidates.append((grid[i], values[i], "Converged scalar minimum at sampled point"))
            else:
                failures.append((values[i], str(result.message)))
        if not candidates:
            finite = np.flatnonzero(np.isfinite(values))
            best_index = finite[np.argmin(values[finite])] if len(finite) else 0
            return FitResult(float(grid[best_index]), alpha.copy(), float(values[best_index]),
                             False, "No scalar bracket converged", len(cache))
        mu, value, message = min(candidates, key=lambda item: item[1])
        unresolved = [item for item in failures if item[0] < value - 1e-6]
        if unresolved:
            return FitResult(float(mu), alpha.copy(), float(value), False,
                             "A lower scalar bracket did not converge: " + unresolved[0][1], len(cache))
        return FitResult(float(mu), alpha.copy(), float(value), True, message, len(cache))

    def test_statistic(self, mu0, **fit_kwargs):
        conditional = self.fit(mu_fixed=mu0, **fit_kwargs)
        unrestricted_kwargs = dict(fit_kwargs)
        unrestricted_kwargs["initial_mu"] = mu0
        unrestricted_kwargs["initial_alpha"] = conditional.alpha
        unrestricted_kwargs.setdefault("mu_starts", [0.05, 1.0, 2.5])
        unrestricted = self.fit(**unrestricted_kwargs)
        if not conditional.success or not unrestricted.success:
            raise RuntimeError(
                "profile fit failed: conditional=" + conditional.message
                + "; unrestricted=" + unrestricted.message
            )
        difference = conditional.nll - unrestricted.nll
        if difference < -1e-6:
            raise RuntimeError("unrestricted fit is worse than conditional fit; inspect optimizer")
        return {"t": max(0.0, float(difference)), "null_fit": conditional, "best_fit": unrestricted}

    def binned(self, scores, edges, morph_order="after", eta=None):
        return FrozenHistogramLikelihood(self, scores, edges, morph_order=morph_order, eta=eta)


class FrozenHistogramLikelihood(TemplateLikelihood):
    """Freeze score/data assignments, then fit every candidate in those bins.

    ``morph_order='after'`` integrates the ±1 anchors and then morphs the
    bin templates. ``'before'`` morphs the unbinned fields first and
    integrates their result at every parameter evaluation.  The latter
    is a like-for-like compression of the parent model.  The operations
    generally do not commute; comparing them tests an additional model
    approximation, not solely the information lost by binning.
    """

    def __init__(self, parent, scores, edges, morph_order="after", eta=None):
        if morph_order not in ("before", "after"):
            raise ValueError("morph_order must be 'before' or 'after'")
        self.parent = parent
        self.scores, self.edges = np.asarray(scores, float).copy(), np.asarray(edges, float).copy()
        self.eta, self.morph_order = eta, morph_order
        self._index = _bin_indices(self.scores, self.edges)
        def integrate(values):
            return histogram_components(self.scores, values, parent.quadrature_weights, self.edges)
        nominal = integrate(parent.nominal)
        down = np.stack([integrate(anchor) for anchor in parent.down]) if parent.n_nuisance else np.empty((0,) + nominal.shape)
        up = np.stack([integrate(anchor) for anchor in parent.up]) if parent.n_nuisance else np.empty((0,) + nominal.shape)
        counts = np.bincount(self._index, weights=parent.asimov_weights, minlength=len(self.edges) - 1)
        super().__init__(nominal, down, up, np.ones(len(counts)), counts,
                         parent.aux_center, parent.aux_sigma, parent.normalize_shapes)
        self.data_counts = self.truth_intensity

    def components(self, alpha=None):
        if self.morph_order == "before":
            return histogram_components(self.scores, self.parent.components(alpha),
                                        self.parent.quadrature_weights, self.edges)
        return super().components(alpha)


class YieldFractionSpline:
    """Shape-preserving PCHIP in bin fractions, followed by normalization.

    Input shape is ``(n_eta, ..., n_bins)``.  Each process/variation has a
    fixed total independent of eta.  Interpolating nonnegative fractions
    directly preserves their range on each anchor interval and handles
    empty bins without a logarithm or positive floor.  Structural zeros
    (empty at every anchor) stay exactly zero.  Only negative roundoff is
    clipped; a materially negative interpolation raises an error.  The
    final normalization restores the sum of fractions to one, since
    independently interpolated bin fractions need not sum exactly to one.
    No eta extrapolation is allowed.

    This object interpolates EXPECTED templates only.  Observed/Asimov
    data counts must be refilled from events at each tested eta.
    """

    def __init__(self, etas, bin_yields, totals=None):
        self.etas = np.asarray(etas, float)
        self.bin_yields = np.asarray(bin_yields, float)
        if self.etas.ndim != 1 or len(self.etas) < 2 or np.any(np.diff(self.etas) <= 0):
            raise ValueError("eta anchors must strictly increase and include at least two points")
        if self.bin_yields.shape[0] != len(self.etas) or self.bin_yields.ndim < 2:
            raise ValueError("bin_yields must have shape (n_eta,...,n_bins)")
        if np.any(~np.isfinite(self.bin_yields)) or np.any(self.bin_yields < 0):
            raise ValueError("bin yields must be finite and nonnegative")
        sums = self.bin_yields.sum(axis=-1)
        self.totals = sums[0].copy() if totals is None else np.asarray(totals, float)
        if self.totals.shape != sums.shape[1:] or np.any(self.totals <= 0):
            raise ValueError("totals must be positive and match process/variation axes")
        if not np.allclose(sums, self.totals[None, ...], rtol=1e-8, atol=1e-10):
            raise ValueError("component totals must not change with the observable eta")
        fractions = self.bin_yields / self.totals[None, ..., None]
        self.active_bins = np.any(fractions > 0, axis=0)
        self._spline = PchipInterpolator(self.etas, fractions, axis=0, extrapolate=False)

    def fractions(self, eta):
        eta = np.asarray(eta, float)
        if np.any(~np.isfinite(eta)) or np.any(eta < self.etas[0]) or np.any(eta > self.etas[-1]):
            raise ValueError("eta is outside the spline anchor range; extrapolation is forbidden")
        fractions = self._spline(eta)
        tolerance = 64 * np.finfo(float).eps
        if np.any(~np.isfinite(fractions)) or np.any(fractions < -tolerance):
            raise FloatingPointError("fraction PCHIP violated nonnegativity beyond roundoff")
        fractions = np.where(self.active_bins, np.maximum(fractions, 0.0), 0.0)
        sums = fractions.sum(axis=-1, keepdims=True)
        if np.any(sums <= 0):
            raise FloatingPointError("interpolated fractions have no positive total")
        return fractions / sums

    def evaluate(self, eta):
        return self.fractions(eta) * self.totals[..., None]

    __call__ = evaluate
