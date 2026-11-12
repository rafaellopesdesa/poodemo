"""Learned coordinates and Monte Carlo templates, without density-oracle inputs.

The simulator supplies events and total generated yields. Only the analytical
unbinned comparator may call its density evaluator. MC normalization, NN
normalization and bin assignment are frozen before fitting or generating toys.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit

from .data import Run, PROCESSES, SAMPLES, SPLITS, sample_size, selected_split, file_digest, save_json
from .inference import (TemplateLikelihood, component_coefficients, exp_poly_factor,
                        histogram_components, _bin_indices)
from .mc_yields import YieldPolicy, yield_policy, yield_diagnostics

VERSION = 1


class PracticalRun(Run):
    """Keep the new study separate from notebook 03 and historical outputs."""
    def __init__(self, base_run):
        super().__init__(base_run.root, {**base_run.config, "practical_inference": True}, base_run.model)
        self.base_run = base_run
        self.evaluation_root = base_run.path("results").parent / "practical_v1"
        for folder in ("results", "plots", "workspaces"):
            (self.evaluation_root / folder).mkdir(parents=True, exist_ok=True)

    def path(self, *parts):
        if parts[:2] in (("results", "preselection.json"), ("results", "density_validation")):
            return self.base_run.path(*parts)
        if parts and parts[0] in ("results", "plots", "workspaces"):
            return self.evaluation_root.joinpath(*parts)
        return self.base_run.path(*parts)


def use_practical_run(run):
    return run if isinstance(run, PracticalRun) else PracticalRun(run)


class LearnedMap:
    """Component intensities relative to the common S PDF, which cancels.

    Predictors retain their deployed member filtering/calibration. Their final
    MC mean normalizers are measured once on the S integration sample; no
    normalizer depends on eta, the fit parameters, or observed/toy events.
    """
    def __init__(self, run, rates, normalizers):
        self.run = run
        self.rates = np.asarray(rates, float)
        self.normalizers = np.asarray(normalizers, float)
        self.predictors = {}

    def raw_ratios(self, x):
        from .training import load_predictor
        result = np.ones((4, len(x)), float)
        for j, process in enumerate(PROCESSES[1:], 1):
            if process not in self.predictors:
                self.predictors[process] = load_predictor(self.run.path("models", "ratios", f"{process}_over_S"))
            result[j] = self.predictors[process].predict_ratio(x)
        if np.any(~np.isfinite(result)) or np.any(result <= 0):
            raise ValueError("Filtered learned ratios must be positive and finite; no clipping is applied")
        return result

    def evaluate(self, x):
        return self.rates[:, None] * self.raw_ratios(x) / self.normalizers[:, None]

    def coordinate(self, fields, eta, power=100., floor=0.):
        from .toy_study import reference_observable
        return reference_observable(fields, self.rates, eta, power, floor)

    def u(self, fields, eta):
        """Stable learned log-ratio/(eta-1), including its exact score limit."""
        eta = float(eta)
        if not np.isfinite(eta) or eta < 0:
            raise ValueError("eta must be nonnegative and finite")
        s, sbi, b, ni = fields
        derivative = s + (sbi-s-b)/(np.sqrt(eta)+1.)
        local = derivative/(sbi+ni)
        s, sbi, b, ni = self.rates
        average = (s+(sbi-s-b)/(np.sqrt(eta)+1.))/(sbi+ni)
        delta = eta-1.
        with np.errstate(divide="ignore", invalid="ignore"):
            result = local-average if delta == 0 else (np.log1p(delta*local)-np.log1p(delta*average))/delta
        if np.any(~np.isfinite(result)):
            raise ValueError("Learned physical mixture is not positive on this bank; inspect NN closure (no floor applied)")
        return result


def prepare_simulation_bank(run):
    """Use existing held-out selected events, with yield/number-generated weights.

    The cap is on selected events per source; all selected events determine the
    acceptance. A deterministic prefix of the iid sample estimates its shape.
    This is MC, not density-weighted analytical quadrature. Dense rows are a
    compatibility representation of six independently labelled event samples.
    """
    cap = run.config.get("simulation_per_sample", run.config.get("quadrature_per_process", 250_000))
    if isinstance(cap, (bool, np.bool_)) or not isinstance(cap, (int, np.integer)):
        raise ValueError("simulation_per_sample must be an integer")
    cached = getattr(run, "_simulation_bank", None)
    if cached is not None and cached.get("cap") == cap:
        return {**cached, "yield_policy": yield_policy(run.config),
                "truth": component_coefficients(run.config["asimov_mu"]) @ cached["nominal"]}
    if cap < 2:
        raise ValueError("simulation_per_sample must be at least two")
    selection = json.loads(run.path("selected", "selection_state.json").read_text())
    owner = run
    while not hasattr(owner, "filter_manifest") and hasattr(owner, "base_run"):
        owner = owner.base_run
    if not hasattr(owner, "filter_manifest"):
        raise ValueError("Select filtered ensembles before preparing a practical study")
    provenance = dict(version=VERSION, cap=cap, selection=selection,
                      generated_sample_sizes={s: sample_size(run, s) for s in SAMPLES},
                      filter_identity=owner.filter_manifest["identity"], physics=run.model.to_dict(),
                      sample_hashes={s: file_digest(run.path("selected", f"{s}_integration.npy")) for s in SAMPLES})
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    directory = run.path("results", "simulation_bank", fingerprint[:20])
    directory.mkdir(parents=True, exist_ok=True)
    samples, rates, records, offsets = [], [], [], {}
    offset = 0
    for sample, (process, alpha) in SAMPLES.items():
        source = selected_split(run, sample, "integration")
        n = min(len(source), cap)
        if n < 2:
            raise ValueError(f"Too few selected integration events for {sample}")
        a, b = SPLITS["integration"]
        generated = int(b*sample_size(run, sample))-int(a*sample_size(run, sample))
        if len(source) > generated:
            raise ValueError(f"Selected count exceeds generated integration count for {sample}")
        rate = run.model.component_yield(process, alpha_ni=alpha)*len(source)/generated
        samples.append(np.asarray(source[:n]))
        offsets[sample] = slice(offset, offset+n)
        offset += n
        rates.append(rate)
        records.append(dict(sample=sample, generated=generated, selected=len(source), used=n,
                            selected_yield=rate, event_weight=rate/n, effective_sample_size=n))
    x = np.concatenate(samples)
    learned_map = LearnedMap(run, rates[:4], np.ones(4))
    fields_path = directory / "learned_fields.npy"
    norm_path = directory / "normalizers.json"
    if fields_path.exists() and norm_path.exists():
        fields = np.load(fields_path, mmap_mode="r")
        learned_map.normalizers = np.asarray(json.loads(norm_path.read_text()))
        if fields.shape != (4, len(x)):
            raise ValueError("Cached learned fields have an incompatible shape")
    else:
        raw = learned_map.raw_ratios(x)
        normalizers = raw[:, offsets["S"]].mean(axis=1)
        if np.any(~np.isfinite(normalizers)) or np.any(normalizers <= 0):
            raise ValueError("Invalid frozen Monte Carlo ratio normalization")
        learned_map.normalizers = normalizers
        fields = learned_map.rates[:, None]*raw/normalizers[:, None]
        from .data import save_array
        save_array(fields_path, fields)
        save_json(norm_path, normalizers.tolist())
    nominal = np.zeros((4, len(x)))
    for j, sample in enumerate(PROCESSES):
        nominal[j, offsets[sample]] = records[j]["event_weight"]
    down, up = nominal.copy(), nominal.copy()
    for sample, target in (("NI_up", up), ("NI_down", down)):
        target[3] = 0.
        target[3, offsets[sample]] = next(r["event_weight"] for r in records if r["sample"] == sample)
    quad = dict(x=x, nominal=nominal, down=down[None], up=up[None], weights=np.ones(len(x)),
                truth=component_coefficients(run.config["asimov_mu"]) @ nominal,
                learned_fields=fields, learned_map=learned_map, sample_slices=offsets,
                practical=True, fingerprint=fingerprint, cap=cap, yield_policy=yield_policy(run.config))
    save_json(directory / "provenance.json", provenance)
    pd.DataFrame(records).to_csv(directory / "sample_counts.csv", index=False)
    print("MC template bank:", directory, "| selected events:", len(x), flush=True)
    run._simulation_bank = quad
    return quad


def prepare_binning_bank(run):
    from .pipeline import prepare_quadrature
    return prepare_simulation_bank(run) if run.config.get("practical_inference") else prepare_quadrature(run)


class SimulationHistogram(TemplateLikelihood):
    """Poisson templates with explicit treatment of finite-MC support changes.

    Positive NI bins retain the existing exp-poly rate/shape model. A bin with
    a zero at any anchor uses nonnegative linear shape interpolation between
    anchors, holding its boundary shape beyond +/-1. This introduces neither
    pseudo-events nor a yield floor. Rates still use exp-poly extrapolation.
    Sparse-support bins are counted in diagnostics; this is an approximation
    to validate with more MC, not an analytical density oracle.
    An optional bounded-zeroing policy acts separately on the final total mean.
    """
    def __init__(self, *args, yield_policy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.yield_policy = yield_policy or YieldPolicy()
        self.zeroing_audit = dict(evaluations=0, zeroed_evaluations=0,
                                 rejected_evaluations=0, max_zeroed_total=0.,
                                 max_negative_bin=0., max_negative_total=0.)

    def raw_intensity(self, mu, alpha=None):
        return super().intensity(mu, alpha)

    def intensity(self, mu, alpha=None):
        # Component templates and interference are never clipped. The same
        # corrected total enters both the Poisson rate and logarithmic term.
        expected, report = self.yield_policy.apply(self.raw_intensity(mu, alpha))
        audit = self.zeroing_audit
        audit['evaluations'] += 1
        audit['zeroed_evaluations'] += int(report['zeroed_bins'] > 0)
        audit['rejected_evaluations'] += int(report['rejected_negative'])
        for key, value in [('max_zeroed_total', report['zeroed_total']),
                           ('max_negative_bin', report['largest_negative']),
                           ('max_negative_total', report['negative_total'])]:
            audit[key] = max(audit[key], value)
        return expected

    @property
    def incomplete_support_bins(self):
        if not self.n_nuisance:
            return 0
        anchors = np.stack([self.nominal[3], self.down[0, 3], self.up[0, 3]])
        return int(np.count_nonzero((anchors.min(axis=0) == 0) & (anchors.max(axis=0) > 0)))

    def components(self, alpha=None):
        alpha = self._alpha(alpha)
        rates, shapes = self.rates.copy(), self.shapes.copy()
        for k, value in enumerate(alpha):
            rates *= exp_poly_factor(value, self.rates_down[k]/self.rates, self.rates_up[k]/self.rates)
            lo, hi = self.shapes_down[k], self.shapes_up[k]
            common = (self.shapes > 0) & (lo > 0) & (hi > 0)
            varied = self.shapes.copy()
            varied[common] *= exp_poly_factor(value, lo[common]/self.shapes[common], hi[common]/self.shapes[common])
            fraction = min(abs(value), 1.)
            target = hi if value >= 0 else lo
            varied[~common] = (1-fraction)*self.shapes[~common] + fraction*target[~common]
            shapes = varied
        integrals = shapes @ self.quadrature_weights
        if np.any(~np.isfinite(shapes)) or np.any(shapes < 0) or np.any(integrals <= 0):
            raise ValueError("Invalid Monte Carlo nuisance morph")
        return rates[:, None]*shapes/integrals[:, None]


class SimulationParent:
    """Histogram factory only: independent MC sources are not a density PDF."""
    def __init__(self, quad, systematics=True):
        self.quad, self.systematics = quad, systematics
        self.nominal = quad["nominal"]
        self.quadrature_weights = quad["weights"]
        self.asimov_weights = quad["weights"]*quad["truth"]
        self.n_nuisance = int(systematics)

    def binned(self, scores, edges, morph_order="after", eta=None):
        if morph_order != "after":
            raise ValueError("MC templates morph after histogramming; no analytical pointwise morph is available")
        from .pipeline import _histogram_anchors
        anchors = _histogram_anchors(self.quad, scores, edges)
        counts = np.histogram(scores, edges, weights=self.asimov_weights)[0]
        down = anchors[1:2] if self.systematics else np.empty((0, 4, len(counts)))
        up = anchors[2:3] if self.systematics else np.empty((0, 4, len(counts)))
        model = SimulationHistogram(anchors[0], down, up, np.ones(len(counts)), counts,
                                    yield_policy=self.quad.get('yield_policy'))
        model.parent, model.data_counts, model.eta = self, counts, eta
        model._index = _bin_indices(scores, edges)
        return model


def refresh_analytic_reference(run):
    """The only oracle calculation in the practical studies: unbinned reference."""
    from . import pipeline
    quad = pipeline.prepare_quadrature(run)
    rows = []
    for syst in (False, True):
        model = pipeline._template_likelihood(quad, systematics=syst)
        for mu in pipeline._positive_study_grid(run):
            rows.append(pipeline._record_scan_result(pipeline._fit_test(run, model, mu, syst),
                mu=mu, model="analytic unbinned", systematics=syst, asimov_mu=run.config["asimov_mu"]))
    frame = pd.DataFrame(rows)
    frame.to_csv(run.path("results", "unbinned_scans.csv"), index=False)
    return frame


def local_information(model, mu):
    """Retain invalid MC Fisher points as diagnostics instead of aborting a scan."""
    from .diagnostics import local_asimov_information
    try:
        if isinstance(model, SimulationHistogram):
            raw = model.raw_intensity(mu)
            if np.any(raw < 0):
                raise ValueError('A clipped intensity is not a regular Fisher point')
        result = local_asimov_information(model, mu)
    except ValueError:
        if not isinstance(model, SimulationHistogram):
            raise
        n = 1+model.n_nuisance
        result = {key: np.full((n, n), np.nan) for key in
                  ("matrix", "data_matrix", "rate_matrix", "shape_matrix", "auxiliary_matrix")}
        result.update(expected_events=float(model.expected_count(mu)), nuisance_derivative_relative_error=np.nan)
    if isinstance(model, SimulationHistogram) and model.incomplete_support_bins:
        # The sparse-bin shape convention can have a kink at alpha=0.
        # A central derivative is not a regular profiled Fisher information.
        result["matrix"][1:, :] = np.nan
        result["matrix"][:, 1:] = np.nan
    return result


def fit_direct_asimov(run, model, mu, systematics):
    """Certify the feasible zero-NLL truth; retain invalid conditional fits.

    Used only for direct MC Asimov templates and the analytical unbinned
    benchmark, never for empirical-CDF interpolated templates or event toys.
    A certified truth is one representative unrestricted MLE, not a claim of
    uniqueness in an interference model.
    """
    from .reference_floor import FloorStudySettings, profile_nuisance
    truth = float(run.config['asimov_mu'])
    bounds, alpha_bounds = run.config['mu_fit_bounds'], run.config['nuisance_bounds']
    alpha = np.zeros(model.n_nuisance)
    nll_truth = model.nll(truth, alpha)
    certified = (bounds[0] <= truth <= bounds[1] and alpha_bounds[0] <= 0 <= alpha_bounds[1]
                 and 0. <= nll_truth <= 1e-10)
    if systematics and model.n_nuisance:
        fit = profile_nuisance(lambda a: model.nll(mu, [a]), alpha_bounds, FloorStudySettings())
        conditional, nuisance, valid, message = fit['nll'], fit['alpha'], fit['valid'], fit['message']
    else:
        conditional, nuisance = model.nll(mu, alpha), 0.
        valid, message = np.isfinite(conditional), 'NI fixed at zero'
    valid = bool(valid and certified and np.isfinite(conditional) and conditional-nll_truth >= -1e-8)
    audit = yield_diagnostics(model, mu, [nuisance] if model.n_nuisance else []) if np.isfinite(nuisance) else {}
    return dict(q=max(0., float(conditional-nll_truth)) if valid else np.nan, valid=valid, **audit,
                mu_hat=truth if certified else np.nan, alpha_NI=nuisance,
                fit_message=message+'; unrestricted truth attains nonnegative NLL bound' if certified
                            else 'Direct Asimov truth could not be certified', truth_certified=bool(certified))


def reuse_filtered_ensembles(run, *, policy=None, device='auto'):
    """Verify and reuse notebook 03's frozen selection, without oracle screening.

    The screening implementation and all member lists remain unchanged. This
    path only checks hashes and metadata; it never prepares analytical fields,
    re-screens members, or changes their deployed calibration.
    """
    from .member_filter import (MemberFilterPolicy, FilteredEvaluationRun,
                                unfiltered_run, _inventory, _digest)
    policy = MemberFilterPolicy() if policy is None else policy
    if not isinstance(policy, MemberFilterPolicy):
        raise TypeError('policy must be MemberFilterPolicy')
    run = unfiltered_run(run)
    message = ('No matching completed notebook-03 filter is available. Run notebook 03 with '
               'this bundle and ENSEMBLE_FILTER_POLICY first; practical notebooks do not '
               're-screen with analytical density weights.')
    try:
        _, inputs = _inventory(run, policy)
        inputs['device'] = device
        identity = _digest(inputs)
        base = getattr(run, 'evaluation_root', run.root/'evaluations')
        root = base/'filtered'/identity[:16]
        saved = json.loads((root/'results/ensemble_filter/complete.json').read_text())
        manifest = json.loads((root/'results/ensemble_filter/manifest.json').read_text())
        valid = (saved.get('identity') == identity and manifest.get('identity') == identity
                 and manifest.get('status') == 'passed' and saved.get('artifacts')
                 and all((root/p).is_file() and file_digest(root/p) == sha
                         for p, sha in saved['artifacts'].items()))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(message) from error
    if not valid:
        raise RuntimeError(message)
    print('Reusing frozen notebook-03 member selection:', root, flush=True)
    return FilteredEvaluationRun(run, root, manifest)
