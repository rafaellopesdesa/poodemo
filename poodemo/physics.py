"""Analytic three-dimensional signal/background interference model.

``S``, ``B`` and ``NI`` are Gaussian event intensities with configurable
integrals.  ``SBI = |A_S + A_B|**2`` is a *positive* process, while its
interference term ``I = SBI - S - B`` may be negative.  ``NI`` is incoherent.
All public ``*_pdf`` methods are normalized; ``*_density`` methods include
the expected event yield.  This distinction is important for reweighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


COMPONENTS = ("S", "SBI", "B", "NI")
FloatArray = NDArray[np.float64]


def _component_name(name: str) -> str:
    canonical = str(name).upper()
    if canonical not in COMPONENTS:
        raise ValueError(f"Unknown component {name!r}; choose from {COMPONENTS}.")
    return canonical


def _points(x: ArrayLike) -> FloatArray:
    points = np.asarray(x, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError("x must have shape (..., 3).")
    if not np.all(np.isfinite(points)):
        raise ValueError("x must contain finite coordinates.")
    return points


def _positive_mu(mu: float, *, derivative: bool = False) -> float:
    value = float(mu)
    if not np.isfinite(value) or value < 0 or (derivative and value == 0):
        condition = "strictly positive" if derivative else "nonnegative"
        raise ValueError(f"mu must be finite and {condition}.")
    return value


@dataclass(frozen=True)
class PhysicsModel:
    """Gaussian wavefunctions with coherent S/B and incoherent NI.

    The S amplitude has phase zero, and the B amplitude has relative phase
    ``arccos(phase_cos)``.  At the default phase, destructive interference
    cannot create exact zeros in ``SBI``.  ``exposure`` scales every yield.

    S and B have similar, distinct shapes so their destructive interference
    leaves a nearby second likelihood basin that shape information only partly
    lifts. The phase puts that basin near mu=0.5 for truth mu=1; the larger
    exposure makes the shallow intervening barrier visible without changing
    the number of generated Monte Carlo events.
    The sole nuisance displaces the NI Gaussian mean by 30% of the norm
    of its nominal mean, toward S in the paper benchmark. S, B and
    their coherent SBI sum remain fixed under this variation.
    """

    lambda_s: float = 100.0
    lambda_b: float = 1000.0
    lambda_ni: float = 10000.0
    exposure: float = 2000.0
    phase_cos: float = -0.273
    mean_s: tuple[float, ...] = (1.5, 1.1, 0.7)
    mean_b: tuple[float, ...] = (1.4762221822829635, 1.083015844487831, 0.6932063377951324)
    mean_ni: tuple[float, ...] = (-0.6, -0.4, 0.1)
    cov_s: tuple[tuple[float, ...], ...] = (
        (0.72, 0.18, 0.06), (0.18, 0.64, 0.10), (0.06, 0.10, 0.81)
    )
    cov_b: tuple[tuple[float, ...], ...] = (
        (0.72324, 0.18081, 0.06027), (0.18081, 0.64288, 0.10045), (0.06027, 0.10045, 0.813645)
    )
    cov_ni: tuple[tuple[float, ...], ...] = (
        (1.35, -0.15, 0.12), (-0.15, 1.10, 0.22), (0.12, 0.22, 1.25)
    )
    shift_fraction_ni: float = 0.3
    nuisance_seed: int = 314159
    direction_ni: tuple[float, ...] | None = (0.792593924, 0.566138517, 0.226455407)
    _precision: dict[str, FloatArray] = field(init=False, repr=False, compare=False)
    _log_norm: dict[str, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("lambda_s", "lambda_b", "lambda_ni"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not np.isfinite(self.exposure) or self.exposure <= 0:
            raise ValueError("exposure must be finite and positive.")
        if self.lambda_s + self.lambda_b <= 0:
            raise ValueError("S and B cannot both have zero yield.")
        if not np.isfinite(self.phase_cos) or not -1 <= self.phase_cos <= 1:
            raise ValueError("phase_cos must lie in [-1, 1].")
        if not np.isfinite(self.shift_fraction_ni) or self.shift_fraction_ni < 0:
            raise ValueError("shift_fraction_ni must be finite and nonnegative.")

        precision, log_norm = {}, {}
        for name in ("s", "b", "ni"):
            mean = np.asarray(getattr(self, f"mean_{name}"), dtype=float)
            covariance = np.asarray(getattr(self, f"cov_{name}"), dtype=float)
            if mean.shape != (3,) or not np.all(np.isfinite(mean)):
                raise ValueError(f"mean_{name} must contain three finite values.")
            if (covariance.shape != (3, 3)
                    or not np.all(np.isfinite(covariance))
                    or not np.allclose(covariance, covariance.T)):
                raise ValueError(f"cov_{name} must be a finite symmetric 3x3 matrix.")
            try:
                chol = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as error:
                raise ValueError(f"cov_{name} must be positive definite.") from error
            precision[name.upper()] = np.linalg.inv(covariance)
            log_norm[name.upper()] = -1.5 * np.log(2 * np.pi) - np.log(np.diag(chol)).sum()
            object.__setattr__(self, f"mean_{name}", tuple(mean.tolist()))
            object.__setattr__(self, f"cov_{name}", tuple(map(tuple, covariance.tolist())))
        object.__setattr__(self, "_precision", precision)
        object.__setattr__(self, "_log_norm", log_norm)

        # Preserve the NI direction from the original seeded model. The first
        # three draws are reserved so removing another nuisance does not change NI.
        rng = np.random.default_rng(self.nuisance_seed)
        generated = rng.normal(size=(2, 3))[1]
        direction = generated if self.direction_ni is None else np.asarray(self.direction_ni, dtype=float)
        if (direction.shape != (3,) or not np.all(np.isfinite(direction))
                or np.linalg.norm(direction) == 0):
            raise ValueError("direction_ni must be a nonzero finite 3-vector.")
        direction = direction / np.linalg.norm(direction)
        object.__setattr__(self, "direction_ni", tuple(direction.tolist()))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, reproducible model configuration."""
        result: dict[str, Any] = {}
        for item in fields(self):
            if not item.init:
                continue
            value = getattr(self, item.name)
            result[item.name] = np.asarray(value).tolist() if isinstance(value, tuple) else value
        return result

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "PhysicsModel":
        return cls(**dict(config))

    def mean(self, name: str, *, alpha_ni: float = 0.0) -> FloatArray:
        """Return a primitive Gaussian mean including its shape nuisance."""
        component = _component_name(name)
        if component == "SBI":
            raise ValueError("SBI is coherent and has no single Gaussian mean.")
        if not np.isfinite(alpha_ni):
            raise ValueError("alpha_ni must be finite.")
        key = component.lower()
        mean = np.asarray(getattr(self, f"mean_{key}"), dtype=float)
        if component == "NI":
            distance = self.shift_fraction_ni * np.linalg.norm(mean)
            mean = mean + alpha_ni * distance * np.asarray(self.direction_ni)
        return mean

    def _gaussian_logpdf(self, x: FloatArray, name: str, alpha_ni: float) -> FloatArray:
        delta = x - self.mean(name, alpha_ni=alpha_ni)
        quadratic = np.einsum("...i,ij,...j->...", delta, self._precision[name], delta)
        return self._log_norm[name] - 0.5 * quadratic

    def _primitive_log_densities(self, x: FloatArray, alpha_ni: float) -> dict[str, FloatArray]:
        result = {}
        for name in ("S", "B", "NI"):
            rate = self.exposure * getattr(self, f"lambda_{name.lower()}")
            log_rate = np.log(rate) if rate > 0 else -np.inf
            result[name] = log_rate + self._gaussian_logpdf(x, name, alpha_ni)
        return result

    def gaussian_overlap(self) -> float:
        """Exact integral of sqrt(f_S f_B): the Gaussian Bhattacharyya coefficient."""
        covariance_s, covariance_b = np.asarray(self.cov_s), np.asarray(self.cov_b)
        midpoint_covariance = 0.5 * (covariance_s + covariance_b)
        delta = self.mean("S") - self.mean("B")
        log_overlap = (
            0.25 * np.linalg.slogdet(covariance_s)[1]
            + 0.25 * np.linalg.slogdet(covariance_b)[1]
            - 0.5 * np.linalg.slogdet(midpoint_covariance)[1]
            - 0.125 * delta @ np.linalg.solve(midpoint_covariance, delta)
        )
        return float(np.exp(log_overlap))

    def component_yield(self, name: str, alpha_ni: float = 0.0) -> float:
        """Analytic full-space expected count of a positive process."""
        component = _component_name(name)
        if not np.isfinite(alpha_ni):
            raise ValueError("alpha_ni must be finite.")
        if component != "SBI":
            return float(self.exposure * getattr(self, f"lambda_{component.lower()}"))
        interference = 2 * self.phase_cos * np.sqrt(self.lambda_s * self.lambda_b) * self.gaussian_overlap()
        return float(self.exposure * (self.lambda_s + self.lambda_b + interference))

    def component_yields(self, alpha_ni: float = 0.0) -> dict[str, float]:
        return {name: self.component_yield(name, alpha_ni) for name in COMPONENTS}

    def component_densities(
        self, x: ArrayLike, alpha_ni: float = 0.0,
        order: Sequence[str] = COMPONENTS,
    ) -> FloatArray:
        """Return yield-weighted densities with component axis last.

        The default order is ``(S, SBI, B, NI)`` and the output shape is
        ``x.shape[:-1] + (4,)``.
        """
        points = _points(x)
        logs = self._primitive_log_densities(points, alpha_ni)
        densities = {name: np.exp(value) for name, value in logs.items()}
        interference = 2 * self.phase_cos * np.exp(0.5 * (logs["S"] + logs["B"]))
        densities["SBI"] = densities["S"] + densities["B"] + interference
        return np.stack([densities[_component_name(name)] for name in order], axis=-1)

    def component_density(self, x: ArrayLike, name: str, alpha_ni: float = 0.0) -> FloatArray:
        """Yield-weighted event density, as distinct from component_pdf."""
        return self.component_densities(x, alpha_ni, order=(name,))[..., 0]

    def component_pdf(self, x: ArrayLike, name: str, alpha_ni: float = 0.0) -> FloatArray:
        """Normalized density of one of the four positive sample sources."""
        component = _component_name(name)
        points = _points(x)
        if component != "SBI":
            return np.exp(self._gaussian_logpdf(points, component, alpha_ni))
        rate = self.component_yield(component, alpha_ni)
        if rate <= 0:
            raise ValueError("SBI has zero yield and therefore no normalized pdf.")
        return self.component_density(points, component, alpha_ni) / rate

    def amplitude(self, x: ArrayLike, name: str, alpha_ni: float = 0.0) -> NDArray[np.complex128]:
        """Complex primitive wavefunction; SBI is the coherent S+B sum."""
        component = _component_name(name)
        if component == "SBI":
            return self.amplitude(x, "S", alpha_ni) + self.amplitude(x, "B", alpha_ni)
        phase = np.arccos(self.phase_cos) if component == "B" else 0.0
        magnitude = np.sqrt(self.component_yield(component) * self.component_pdf(x, component, alpha_ni))
        return magnitude * np.exp(1j * phase)

    @staticmethod
    def coefficients(mu: float) -> FloatArray:
        """Coefficients multiplying positive processes in COMPONENTS order."""
        value = _positive_mu(mu)
        root = np.sqrt(value)
        return np.asarray((value - root, root, 1 - root, 1.0))

    @staticmethod
    def derivative_coefficients(mu: float) -> FloatArray:
        value = _positive_mu(mu, derivative=True)
        inverse = 0.5 / np.sqrt(value)
        return np.asarray((1 - inverse, inverse, -inverse, 0.0))

    def intensity(self, x: ArrayLike, mu: float, alpha_ni: float = 0.0) -> FloatArray:
        """Physical intensity mu*S + sqrt(mu)*I + B + NI."""
        value = _positive_mu(mu)
        logs = self._primitive_log_densities(_points(x), alpha_ni)
        interference = 2 * self.phase_cos * np.exp(0.5 * (logs["S"] + logs["B"]))
        return value * np.exp(logs["S"]) + np.sqrt(value) * interference + np.exp(logs["B"]) + np.exp(logs["NI"])

    def mu_derivative(self, x: ArrayLike, mu: float, alpha_ni: float = 0.0) -> FloatArray:
        """Derivative of physical intensity with respect to mu (mu > 0)."""
        value = _positive_mu(mu, derivative=True)
        logs = self._primitive_log_densities(_points(x), alpha_ni)
        return np.exp(logs["S"]) + self.phase_cos / np.sqrt(value) * np.exp(0.5 * (logs["S"] + logs["B"]))

    def _yield_vector(self, selected_yields: Mapping[str, float] | Sequence[float] | None, alpha_ni: float) -> FloatArray:
        if selected_yields is None:
            values = np.asarray([self.component_yield(name, alpha_ni) for name in COMPONENTS])
        elif isinstance(selected_yields, Mapping):
            canonical = {str(key).upper(): value for key, value in selected_yields.items()}
            values = np.asarray([canonical[name] for name in COMPONENTS], dtype=float)
        else:
            values = np.asarray(selected_yields, dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Selected yields must be four finite nonnegative values in S/SBI/B/NI order.")
        return values

    def total_yield(
        self, mu: float, alpha_ni: float = 0.0,
        selected_yields: Mapping[str, float] | Sequence[float] | None = None,
    ) -> float:
        result = float(self.coefficients(mu) @ self._yield_vector(selected_yields, alpha_ni))
        if result <= 0:
            raise ValueError("The physical total yield must be positive.")
        return result

    def yield_mu_derivative(
        self, mu: float, alpha_ni: float = 0.0,
        selected_yields: Mapping[str, float] | Sequence[float] | None = None,
    ) -> float:
        return float(self.derivative_coefficients(mu) @ self._yield_vector(selected_yields, alpha_ni))

    def pdf(self, x: ArrayLike, mu: float, alpha_ni: float = 0.0) -> FloatArray:
        """Normalized physical full-space density."""
        return self.intensity(x, mu, alpha_ni) / self.total_yield(mu, alpha_ni)

    def score(
        self, x: ArrayLike, eta: float,
        selected_yields: Mapping[str, float] | Sequence[float] | None = None,
        alpha_ni: float = 0.0,
    ) -> FloatArray:
        """Exact local *shape* score d_mu log p_mu at the fixed anchor eta.

        For a fixed preselection, supply its selected S/SBI/B/NI yields.
        The acceptance indicator is mu-independent, so only the
        normalization term changes.  Returned values are meaningful on the
        accepted region.  The event count separately carries rate information.

        Component densities are rescaled internally to avoid 0/0 in distant
        Gaussian tails.  eta=0 is excluded because sqrt(mu) is not regular there.
        """
        value = _positive_mu(eta, derivative=True)
        logs = self._primitive_log_densities(_points(x), alpha_ni)
        scale = np.maximum(np.maximum(logs["S"], logs["B"]), logs["NI"])
        s, b, ni = (np.exp(logs[name] - scale) for name in ("S", "B", "NI"))
        interference = 2 * self.phase_cos * np.exp(0.5 * (logs["S"] + logs["B"]) - scale)
        numerator = s + interference / (2 * np.sqrt(value))
        denominator = value * s + np.sqrt(value) * interference + b + ni
        shape_normalization = self.yield_mu_derivative(value, alpha_ni, selected_yields) / self.total_yield(value, alpha_ni, selected_yields)
        return numerator / denominator - shape_normalization

    def sample_component(
        self, name: str, n: int, rng: np.random.Generator | int | None = None,
        alpha_ni: float = 0.0,
    ) -> FloatArray:
        """Draw exact independent events from a normalized component density.

        SBI uses rejection sampling from the normalized S+B mixture.  The
        envelope is 2(S+B), valid for every allowed relative phase, and its
        efficiency is known analytically.  No approximate flow or MCMC is used.
        """
        component = _component_name(name)
        if int(n) != n or n < 0:
            raise ValueError("n must be a nonnegative integer.")
        n = int(n)
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        if n == 0:
            return np.empty((0, 3), dtype=float)
        if component != "SBI":
            return generator.multivariate_normal(
                self.mean(component, alpha_ni=alpha_ni),
                np.asarray(getattr(self, f"cov_{component.lower()}")), size=n,
            )
        yield_s, yield_b = self.component_yield("S"), self.component_yield("B")
        mixture_yield = yield_s + yield_b
        efficiency = self.component_yield("SBI", alpha_ni) / (2 * mixture_yield)
        if efficiency <= 0:
            raise ValueError("Cannot sample a zero-yield SBI component.")
        output = np.empty((n, 3), dtype=float)
        filled = 0
        while filled < n:
            batch = min(250_000, max(256, int(np.ceil(1.05 * (n - filled) / efficiency))))
            from_s = generator.random(batch) < yield_s / mixture_yield
            candidates = np.empty((batch, 3), dtype=float)
            candidates[from_s] = self.sample_component("S", int(from_s.sum()), generator, alpha_ni)
            candidates[~from_s] = self.sample_component("B", int((~from_s).sum()), generator, alpha_ni)
            logs = self._primitive_log_densities(candidates, alpha_ni)
            overlap_fraction = np.exp(0.5 * (logs["S"] + logs["B"]) - np.logaddexp(logs["S"], logs["B"]))
            acceptance = 0.5 + self.phase_cos * overlap_fraction
            accepted = candidates[generator.random(batch) < acceptance]
            take = min(len(accepted), n - filled)
            output[filled:filled + take] = accepted[:take]
            filled += take
        return output


# Friendly alias for interactive notebooks.
ToyModel = PhysicsModel
