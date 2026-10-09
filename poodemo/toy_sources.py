"""Statistical-only Poisson experiments for notebook 11.

Simulator experiments are fresh draws from the continuous physical model,
including its interference and the frozen preselection. Learned-model
experiments instead sample the *discrete quadrature approximation* to the
frozen learned intensity. They are not continuous neural generative samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .data import file_digest
from .physics import PhysicsModel


def _generator(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _parameters(mu, exposure):
    mu, exposure = float(mu), float(exposure)
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("mu must be finite and nonnegative")
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("exposure must be finite and positive")
    return mu, exposure


def load_frozen_selector(run):
    """Load notebook 02's selector and verify its saved selection fingerprint."""
    from .training import load_predictor

    selection = json.loads(run.path("selected", "selection_state.json").read_text())
    threshold = float(selection["threshold"])
    digest = file_digest(run.path("models", "preselection", "model.pt"))
    fingerprint = hashlib.sha256((digest + repr(threshold)).encode()).hexdigest()
    if not np.isfinite(threshold) or fingerprint != selection["selection_fingerprint"]:
        raise ValueError("The selector checkpoint or threshold changed after preselection")
    predictor = load_predictor(run.path("models", "preselection"))
    return predictor, threshold, dict(selection=selection, selector_sha256=digest)


def make_frozen_workspace(run, quadrature):
    """Reuse notebook 03's learned shapes at both bank and fresh coordinates.

    All normalization constants are established on the original quadrature,
    never on toy observations. This is the same evaluator used by notebook
    03's independent high-statistics closure diagnostics.
    """
    from .closure_validation import FrozenWorkspaceFields
    from .pipeline import RATIO_TASKS
    from .training import load_predictor

    predictors = {
        f"{num}_over_{den}": load_predictor(run.path("models", "ratios", f"{num}_over_{den}"))
        for num, den in RATIO_TASKS
    }
    return FrozenWorkspaceFields(run, quadrature, predictors)


def sample_simulator_poisson(run_or_model, mu, rng=None, *, exposure=1.,
                             selector=None, threshold=None, chunk_size=100_000):
    """Draw a fresh exact physical Poisson experiment at nominal nuisance.

    ``exposure`` multiplies the model's existing exposure. With a ``Run`` the
    saved selector is loaded unless one is supplied; with a ``PhysicsModel``
    and no selector the result covers full phase space. Supply a previously
    loaded selector/threshold when generating many experiments.

    NI is sampled directly. The coherent contribution is obtained by thinning
    a Poisson process of intensity ``k*(mu*S+B)``, where
    ``k = 1 + max(cos(phi), 0)``. The bound follows from
    ``2*sqrt(mu*S*B) <= mu*S+B``. No selected-rate integral or finite event
    pool is used, so both the count and coordinates are genuine simulator
    fluctuations, including at mu=0.
    """
    mu, exposure = _parameters(mu, exposure)
    if isinstance(chunk_size, (bool, np.bool_)) or int(chunk_size) != chunk_size or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)
    if isinstance(run_or_model, PhysicsModel):
        model = run_or_model
    else:
        model = run_or_model.model
        if selector is None:
            selector, threshold, _ = load_frozen_selector(run_or_model)
    if selector is not None and (threshold is None or not np.isfinite(threshold)):
        raise ValueError("A finite threshold is required with a selector")
    generator = _generator(rng)
    envelope = 1. + max(model.phase_cos, 0.)
    parts = []
    for process, multiplier in (("S", envelope*mu), ("B", envelope), ("NI", 1.)):
        count = int(generator.poisson(exposure*multiplier*model.component_yield(process)))
        for start in range(0, count, chunk_size):
            points = model.sample_component(process, min(chunk_size, count-start), generator)
            if process != "NI" and mu > 0:
                # Stable even in far Gaussian tails: log-space cancellation
                # avoids dividing two underflowing densities.
                logs = model._primitive_log_densities(points, 0.)
                log_s, log_b = np.log(mu) + logs["S"], logs["B"]
                overlap = np.exp(.5*(log_s+log_b) - np.logaddexp(log_s, log_b))
                probability = (1. + 2.*model.phase_cos*overlap)/envelope
                if np.any(probability < -1e-12) or np.any(probability > 1.+1e-12):
                    raise ValueError("Invalid coherent Poisson-thinning probability")
                points = points[generator.random(len(points)) < np.clip(probability, 0., 1.)]
            elif process != "NI" and envelope != 1.:
                points = points[generator.random(len(points)) < 1./envelope]
            if selector is not None and len(points):
                accepted = selector.predict_proba(points)[:, 0] >= threshold
                points = points[accepted]
            if len(points):
                parts.append(points)
    result = np.concatenate(parts) if parts else np.empty((0, 3), dtype=float)
    # Conditional on N, also expose the ordinary iid ordering to callers.
    generator.shuffle(result)
    return result


@dataclass
class QuadraturePoissonSource:
    """Poisson-resample a frozen finite-bank approximation to an intensity.

    ``nominal_fields`` has shape (4, N), in S/SBI/B/NI order, and should be
    the frozen learned fields for learned-model toys. Weights and fields are
    used exactly as supplied. Shape probabilities are normalized only for
    drawing coordinates; the unnormalized sum determines the Poisson count.
    """
    quadrature: dict
    nominal_fields: np.ndarray
    mu: float
    exposure: float = 1.

    def __post_init__(self):
        self.mu, self.exposure = _parameters(self.mu, self.exposure)
        weights = np.asarray(self.quadrature["weights"], dtype=float)
        fields = np.asarray(self.nominal_fields, dtype=float)
        if weights.ndim != 1 or len(weights) == 0 or fields.shape != (4, len(weights)):
            raise ValueError("Expected N quadrature weights and nominal fields of shape (4, N)")
        if (np.any(~np.isfinite(weights)) or np.any(weights < 0)
                or np.any(~np.isfinite(fields)) or np.any(fields < 0)):
            raise ValueError("Quadrature weights and component fields must be finite and nonnegative")
        intensity = PhysicsModel.coefficients(self.mu) @ fields
        if np.any(~np.isfinite(intensity)) or np.any(intensity < 0):
            raise ValueError("Learned physical intensity is negative or nonfinite on the toy bank")
        self.masses = self.exposure*weights*intensity
        self.expected_count = float(self.masses.sum())
        if not np.isfinite(self.expected_count):
            raise ValueError("Toy expected count must be finite")
        self.probabilities = (self.masses/self.expected_count if self.expected_count > 0
                              else np.zeros_like(self.masses))
        self._cdf = np.cumsum(self.probabilities)
        if self.expected_count > 0:
            self._cdf[-1] = 1.
        sum_squares = float(np.sum(self.probabilities**2))
        self.diagnostics = dict(
            source="finite_quadrature_poisson_resampling",
            continuous_generator=False, mu=self.mu, exposure=self.exposure,
            expected_count=self.expected_count, bank_size=int(len(weights)),
            positive_mass_nodes=int(np.count_nonzero(self.masses)),
            effective_bank_size=1./sum_squares if sum_squares > 0 else 0.,
            max_node_probability=float(np.max(self.probabilities)),
            expected_unique_nodes=float(np.sum(-np.expm1(-self.masses))),
            note="Independent Poisson experiments conditional on this frozen finite bank; repeated coordinates are expected.",
        )

    def draw_indices(self, rng=None):
        """Draw Poisson N, then N iid bank indices; duplicate nodes are valid."""
        generator = _generator(rng)
        count = int(generator.poisson(self.expected_count))
        return np.searchsorted(self._cdf, generator.random(count), side="right")

    def draw_counts(self, rng=None):
        """Return unique bank indices and multiplicities for a weighted fit."""
        return np.unique(self.draw_indices(rng), return_counts=True)
