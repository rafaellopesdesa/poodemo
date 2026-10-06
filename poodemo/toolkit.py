"""Array-backed adapter to the pinned nsbi-lhc-toolkit workspace and inference.

The upstream model currently multiplies only linear norm factors. This adapter
adds the interference coefficients, a frozen-quadrature normalization of morphed
PDFs, and explicit physical-domain checks. It calls the toolkit's own strategy-5
interpolator, workspace serializer, JAX model interface, and Minuit inference
interface. It does not emulate a toolkit neural network or silently replace a
trained result by an analytical result.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
from typing import Sequence

import numpy as np

TOOLKIT_COMMIT = "fc09848fc6540fd32310faebbe9db6eea7ecd17b"
PROCESSES = ("S", "SBI", "B", "NI")
NUISANCES = ("alpha_NI",)


@dataclass
class ToolkitFit:
    """A minimum of -2 log L (up to a parameter-independent constant)."""

    parameters: np.ndarray
    nll: float
    valid: bool
    edm: float


@lru_cache(maxsize=1)
def _toolkit_classes():
    """Import the optional Colab dependencies only when this adapter is used."""
    try:
        import jax
        import jax.numpy as jnp
        from iminuit import Minuit
        from nsbi_common_utils.workspace_builder import WorkspaceBuilder
        from nsbi_common_utils.models.sbi_parametric_model import (
            sbi_parametric_model, _calculate_combined_var,
        )
        from nsbi_common_utils.inference import inference
    except ImportError as exc:
        raise ImportError(
            "Install the pinned nsbi-lhc-toolkit and its inference dependencies "
            "with the notebook setup cell before building a toolkit workspace."
        ) from exc

    jax.config.update("jax_enable_x64", True)

    class ArrayWorkspaceBuilder(WorkspaceBuilder):
        """Feed evaluated arrays to the upstream builder without a ROOT roundtrip.

        ROOT loading is the only part bypassed. The actual upstream build(),
        measurements(), and dump_workspace() methods generate the workspace.
        """

        def __init__(self, channel, parameter_bounds):
            self._channel = channel
            self.ParametersToFit = ["mu", *NUISANCES]
            self.config_dict = {
                "General": {"Measurement": {"Name": "poodemo", "POI": "mu"}},
                "NormFactors": [{"Name": "mu", "Nominal": 1.0,
                                 "Bounds": list(parameter_bounds[0])}],
                "Systematics": [
                    {"Name": name, "Nominal": 0.0, "Bounds": list(bounds)}
                    for name, bounds in zip(NUISANCES, parameter_bounds[1:])
                ],
            }

        def channels(self):
            return [self._channel]

    class InterferenceModel(sbi_parametric_model):
        """The toolkit model with coherent-amplitude POI coefficients."""

        def _calculate_norm_variations(self, param_vec):
            mu = param_vec[0]
            root_mu = jnp.sqrt(jnp.maximum(mu, 0.0))
            return {"S": mu - root_mu, "SBI": root_mu,
                    "B": 1.0 - root_mu, "NI": 1.0}

        def _build_stacked_data(self):
            super()._build_stacked_data()
            info = self.workspace["poodemo"]
            self._model_data["integration_weights"] = jnp.asarray(
                np.load(info["integration_weights"])
            )
            self._model_data["truth_ratio"] = jnp.asarray(
                np.load(info["truth_ratio"])
            )
            self._model_data["bounds"] = jnp.asarray(info["parameter_bounds"])
            # The observed rate must come from the specified Asimov truth,
            # not from coefficients evaluated at the toolkit's initial values.
            self._model_data["expected_rate"] = jnp.asarray(
                [float(np.sum(np.asarray(self.weight_arrays_unbinned)))]
            )
            self.expected_rate_unbinned = self._model_data["expected_rate"]

        def _build_jit_functions(self):
            batched_variation = jax.vmap(_calculate_combined_var,
                                          in_axes=(None, 0, 0))

            def components(param_vec, data):
                alpha = param_vec[1:]
                shape_mod = batched_variation(
                    alpha, data["var_up_unbinned"], data["var_dn_unbinned"]
                )
                rate_mod = batched_variation(
                    alpha, data["tot_up_unbinned"], data["tot_dn_unbinned"]
                )
                unnormalized = data["ratios"] * shape_mod
                norm = jnp.sum(
                    unnormalized * data["integration_weights"][None, :], axis=1
                )
                valid = (jnp.all(shape_mod > 0) & jnp.all(rate_mod > 0)
                         & jnp.all(norm > 0))
                # Guards only make the rejected branch differentiable. They
                # never turn an invalid prediction into an accepted likelihood.
                safe_norm = jnp.where(norm > 0, norm, 1.0)
                ratios = unnormalized / safe_norm[:, None]
                yields = data["unbinned_total"] * rate_mod
                return ratios * yields, yields[:, 0], valid

            def objective(param_vec, data):
                mu = param_vec[0]
                root_mu = jnp.sqrt(jnp.maximum(mu, 1.e-300))
                coefficient = jnp.array([mu - root_mu, root_mu,
                                         1.0 - root_mu, 1.0])
                fields, yields, valid = components(param_vec, data)
                rate = jnp.dot(coefficient, yields)
                intensity_ratio = jnp.sum(coefficient[:, None] * fields, axis=0)
                valid = (valid & (mu >= 0) & (rate > 0)
                         & jnp.all(intensity_ratio > 0)
                         & jnp.all(jnp.isfinite(intensity_ratio))
                         & jnp.all(param_vec >= data["bounds"][:, 0])
                         & jnp.all(param_vec <= data["bounds"][:, 1]))
                safe_ratio = jnp.where(intensity_ratio > 0, intensity_ratio, 1.0)
                observed_rate = jnp.sum(data["weights"])
                # Stable Asimov-relative form: no large parameter-independent
                # Poisson/log-reference constants need to be subtracted later.
                value = (2.0 * (rate - observed_rate
                         - jnp.sum(data["weights"] * jnp.log(
                             safe_ratio / data["truth_ratio"])))
                         # Standard-normal auxiliary measurement at zero:
                         # -2 log[G(alpha_NI;0,1)/G(0;0,1)] = alpha_NI**2.
                         + jnp.sum(param_vec[1:] ** 2))
                return jnp.where(valid & jnp.isfinite(value), value, jnp.inf)

            self._jit_components = jax.jit(components)
            return (jax.jit(objective),
                    jax.jit(jax.value_and_grad(objective, argnums=0)))

        def component_intensity_ratios(self, alpha=(0.0,)):
            """Return lambda_j(alpha) p_j(x;alpha) / p_S(x;0)."""
            fields, _, valid = self._jit_components(
                jnp.asarray([1.0, *alpha]), self._model_data
            )
            if not bool(valid):
                raise ValueError("The exp-poly component morph is not positive.")
            return np.asarray(fields)

    class BoundedInference(inference):
        """Toolkit inference extended with bounds and explicit fit diagnostics.

        The pinned upstream fitter does not read workspace parameter bounds and
        subtracts the minimum sampled scan value. Here every scan subtracts the
        actual fitted global minimum, and failed points are reported as NaN.
        """

        def __init__(self, model):
            names, initial = model.get_model_parameters()
            super().__init__(model.model, initial, names,
                             model.num_unconstrained_param, model.model_grad)
            self.model_object = model
            self.bounds = model.workspace["poodemo"]["parameter_bounds"]
            self.last_fit = None
            self.last_scan = None

        def _minuit(self, values, freeze_params=(), mu_fixed=None, strategy=1):
            m = Minuit(self.model_nll, np.asarray(values, dtype=float),
                       grad=self.model_grad, name=tuple(self.list_parameters))
            m.errordef = Minuit.LEAST_SQUARES
            m.strategy = strategy
            m.tol = 0.01
            for name, bounds in zip(self.list_parameters, self.bounds):
                m.limits[name] = tuple(bounds)
            for name in freeze_params:
                m.fixed[name] = True
            if mu_fixed is not None:
                m.values["mu"] = float(mu_fixed)
                m.fixed["mu"] = True
            return m

        def _run_start(self, values, frozen, mu_fixed, strategy):
            m = self._minuit(values, frozen, mu_fixed, strategy)
            if all(m.fixed):
                value = float(self.model_nll(np.asarray(m.values)))
                return ToolkitFit(np.asarray(m.values), value,
                                  bool(np.isfinite(value)), 0.0)
            m.migrad()
            if not m.valid:
                m.simplex()
                m.migrad()
            return ToolkitFit(np.asarray(m.values).copy(), float(m.fval),
                              bool(m.valid and np.isfinite(m.fval)),
                              float(m.fmin.edm))

        def fit(self, mu_fixed=None, systematics=True, initial=None, strategy=1,
                freeze_params=()):
            values = np.asarray(self.initial_values if initial is None else initial,
                                dtype=float).copy()
            frozen = tuple(freeze_params) if systematics else tuple(set(
                (*freeze_params, *NUISANCES)))
            if not systematics:
                values[1:] = 0.0
            starts = [values]
            if mu_fixed is None and "mu" not in frozen:
                # Interference likelihoods can have separated local maxima.
                # Multi-start fits prevent silently subtracting a local minimum.
                for mu in (.05, .3, 1.0, 2.5, 3.8):
                    if self.bounds[0][0] <= mu <= self.bounds[0][1]:
                        trial = values.copy()
                        trial[0] = mu
                        trial[1:] = 0.0
                        if not any(np.array_equal(trial, x) for x in starts):
                            starts.append(trial)
            results = [self._run_start(v, frozen, mu_fixed, strategy) for v in starts]
            valid = [r for r in results if r.valid]
            if mu_fixed is not None and not valid:
                # Retry a failed warm-start profile at the global-fit nuisance
                # values and at the nominal nuisance point before marking NaN.
                retries = [np.asarray(self.initial_values).copy()]
                if self.pulls_global_fit is not None:
                    retries.append(np.asarray(self.pulls_global_fit).copy())
                for retry in retries:
                    if not systematics:
                        retry[1:] = 0.0
                    result = self._run_start(retry, frozen, mu_fixed, 2)
                    results.append(result)
                    if result.valid:
                        valid.append(result)
            return min(valid or results,
                       key=lambda r: r.nll if np.isfinite(r.nll) else np.inf)

        def perform_fit(self, fit_strategy=1, freeze_params=()):
            """Compatible upstream entry point; return the result as well."""
            result = self.fit(strategy=fit_strategy, freeze_params=freeze_params)
            self.pulls_global_fit = result.parameters
            self.last_fit = result
            return result

        def scan(self, mu_values, systematics=True, strategy=1):
            """Profile at supplied mu points, returning values and diagnostics."""
            global_fit = self.fit(systematics=systematics, strategy=strategy)
            if not global_fit.valid:
                raise RuntimeError(f"Toolkit global fit failed: {global_fit}")
            self.pulls_global_fit = global_fit.parameters.copy()
            mu_values = np.asarray(mu_values, dtype=float)
            results = []
            warm = global_fit.parameters.copy()
            for mu in mu_values:
                if not self.bounds[0][0] <= mu <= self.bounds[0][1]:
                    raise ValueError(f"mu={mu} lies outside workspace bounds")
                result = self.fit(mu_fixed=mu, systematics=systematics, initial=warm,
                                  strategy=strategy)
                if result.valid:
                    warm = result.parameters.copy()
                results.append(result)
            test_statistic = np.array([
                r.nll - global_fit.nll if r.valid else np.nan for r in results
            ])
            if np.any(test_statistic < -1.e-4):
                raise RuntimeError("A profile point lies below the global fit; "
                                   "repeat the global fit with additional starts.")
            self.pulls_global_fit = global_fit.parameters
            self.last_fit = global_fit
            self.last_scan = {
                "mu": mu_values, "t_mu": test_statistic,
                "parameters": np.array([r.parameters for r in results]),
                "valid": np.array([r.valid for r in results]),
                "edm": np.array([r.edm for r in results]),
            }
            return self.last_scan

        def perform_profile_scan(self, parameter_name="mu", bound_range=(.001, 4.0),
                                 fit_strategy=1, freeze_params=(), doStatOnly=False,
                                 isConstrainedNP=False, size=100):
            """Bound-safe upstream-style entry point for the POI scan.

            Here "stat only" fixes nuisances at their nominal zero values,
            appropriate for comparing the two explicitly specified models.
            ``scan`` additionally returns the per-point validity diagnostics.
            """
            if parameter_name != "mu" or isConstrainedNP:
                raise ValueError("This demo adapter profiles the POI mu only")
            if freeze_params and set(freeze_params) != set(NUISANCES):
                raise ValueError("Freeze alpha_NI or leave it free for this scan")
            grid = np.linspace(*bound_range, int(size))
            main = self.scan(grid, systematics=not bool(freeze_params), strategy=fit_strategy)
            if doStatOnly:
                stat = self.scan(grid, systematics=False, strategy=fit_strategy)
                return main["mu"], main["t_mu"], stat["mu"], stat["t_mu"]
            return main["mu"], main["t_mu"]

    return ArrayWorkspaceBuilder, InterferenceModel, BoundedInference


def build_workspace(root: str | Path, nominal, down, up, quadrature_weights,
                    truth_intensity, *, parameter_bounds=((1.e-3, 4.0),
                    (-3.0, 3.0))) -> dict:
    """Build/write an actual toolkit workspace from common quadrature arrays.

    ``nominal`` is (4,N), in S,SBI,B,NI order. ``down``/``up`` are (1,4,N),
    with the single alpha_NI variation, including nominal fields for unaffected
    components. All entries are *absolute intensities*, not normalized PDFs.
    ``quadrature_weights`` integrate dx at the common proposal points; the
    observed Asimov weights are quadrature_weights * truth_intensity.

    The density-ratio reference is the accepted, nominal signal PDF even though
    the points used for numerical integration can follow a broader mixture.
    The returned dictionary contains absolute array paths under ``root``. Build
    it again if moving the workspace to a different mounted Drive location.
    """
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    nominal = np.asarray(nominal, dtype=np.float64)
    down, up = (np.asarray(a, dtype=np.float64) for a in (down, up))
    weights = np.asarray(quadrature_weights, dtype=np.float64)
    truth = np.asarray(truth_intensity, dtype=np.float64)
    n = len(weights)
    if (nominal.shape != (4, n) or down.shape != (1, 4, n)
            or up.shape != (1, 4, n) or truth.shape != (n,)):
        raise ValueError("Expected nominal(4,N), down/up(1,4,N), weights/truth(N)")
    for name, a in [("nominal", nominal), ("down", down), ("up", up),
                    ("quadrature_weights", weights), ("truth_intensity", truth)]:
        if not np.all(np.isfinite(a)) or np.any(a <= 0):
            raise ValueError(f"{name} must contain positive finite values")
    bounds = np.asarray(parameter_bounds, dtype=float)
    if (bounds.shape != (2, 2) or np.any(bounds[:, 0] >= bounds[:, 1])
            or bounds[0, 0] <= 0):
        raise ValueError("Use finite ordered bounds with a strictly positive mu floor")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("All workspace bounds must be finite")

    yields = nominal @ weights
    yields_down, yields_up = down @ weights, up @ weights
    reference = nominal[0] / yields[0]
    ratios = nominal / yields[:, None] / reference[None, :]
    shape_down = down / nominal[None, :, :] * yields[None, :, None] / yields_down[:, :, None]
    shape_up = up / nominal[None, :, :] * yields[None, :, None] / yields_up[:, :, None]

    def save(name, array):
        path = root / f"{name}.npy"
        np.save(path, np.asarray(array, dtype=np.float64))
        return str(path)

    samples = []
    for j, process in enumerate(PROCESSES):
        modifiers = [{"name": "mu", "type": "normfactor", "data": None}]
        for k, nuisance in enumerate(NUISANCES):
            modifiers.append({
                "name": nuisance, "type": "normplusshape", "data": {
                    "hi_data": [float(yields_up[k, j] / yields[j])],
                    "lo_data": [float(yields_down[k, j] / yields[j])],
                    "hi_ratio": save(f"{process}_{nuisance}_up", shape_up[k, j]),
                    "lo_ratio": save(f"{process}_{nuisance}_down", shape_down[k, j]),
                },
            })
        samples.append({"name": process, "data": [float(yields[j])],
                        "ratios": save(f"{process}_ratio", ratios[j]),
                        "modifiers": modifiers})
    channel = {"name": "selected", "type": "unbinned", "samples": samples,
               "weights": save("asimov_weights", weights * truth)}
    Builder, _, _ = _toolkit_classes()
    builder = Builder(channel, bounds)
    workspace = builder.build()
    workspace["poodemo"] = {
        "toolkit_commit": TOOLKIT_COMMIT,
        "coefficient_order": list(PROCESSES),
        "reference": "accepted nominal signal PDF",
        "normalized_morphs": True,
        "integration_weights": save("reference_integration_weights", weights * reference),
        "truth_ratio": save("truth_ratio", truth / reference),
        "parameter_bounds": bounds.tolist(),
    }
    builder.dump_workspace(workspace, str(root / "workspace.json"))
    return workspace


def load_model(workspace: dict | str | Path):
    """Instantiate the JAX subclass of the toolkit's sbi_parametric_model."""
    if not isinstance(workspace, dict):
        with open(workspace) as handle:
            workspace = json.load(handle)
    _, Model, _ = _toolkit_classes()
    return Model(workspace, "poodemo")


def make_fitter(model):
    """Return the bounded subclass of the toolkit's actual inference engine."""
    _, _, Fitter = _toolkit_classes()
    return Fitter(model)


def profile_scans(workspace, mu_values: Sequence[float]):
    """Convenience entry point used by notebook 3 (with and without nuisances)."""
    model = load_model(workspace)
    fitter = make_fitter(model)
    return {
        "stat_only": fitter.scan(mu_values, systematics=False),
        "profiled": fitter.scan(mu_values, systematics=True),
    }
