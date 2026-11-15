"""Signal-reference coordinates for the parallel 04--07 studies.

Only learned component ratios and frozen selected rates enter this map.
The implicit selected S density cancels; no analytical PDF is evaluated.
"""
import numpy as np
from scipy.special import expit

from .inference import component_coefficients


def reference_denominator(config):
    value = config.get('reference_ratio_denominator', 'physical_1')
    if value not in ('physical_1', 'S'):
        raise ValueError('reference_ratio_denominator must be physical_1 or S')
    return value


def reference_metadata(config):
    """Leave published physical-reference cache fingerprints unchanged."""
    return ({'reference_ratio_denominator': 'S'}
            if reference_denominator(config) == 'S' else {})


def signal_log_ratio(fields, rates, eta):
    """log[p_selected(x; eta)/p_selected_S(x)], with frozen normalization."""
    fields, rates = np.asarray(fields, float), np.asarray(rates, float)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError('eta must be finite and nonnegative')
    if fields.ndim != 2 or fields.shape[0] != 4 or rates.shape != (4,):
        raise ValueError('fields and frozen rates must have shapes (4,N) and (4,)')
    coefficients = component_coefficients(eta)
    density, total = coefficients @ fields, float(coefficients @ rates)
    signal, signal_total = fields[0], float(rates[0])
    if (np.any(~np.isfinite(density)) or np.any(density < 0)
            or np.any(~np.isfinite(signal)) or np.any(signal <= 0)
            or not np.isfinite(total) or total <= 0
            or not np.isfinite(signal_total) or signal_total <= 0):
        raise ValueError('Signal reference requires nonnegative physical density and positive selected rates/S density')
    with np.errstate(divide='ignore'):
        return np.log(density) - np.log(total) - np.log(signal) + np.log(signal_total)


def learned_observable(fields, rates, eta, power, floor=0., *, denominator='physical_1'):
    reference_denominator({'reference_ratio_denominator': denominator})
    if denominator == 'physical_1':
        from .toy_study import reference_observable
        return reference_observable(fields, rates, eta, power, floor)
    from .pipeline import reference_ratio_eta
    if not np.isfinite(power) or power <= 0:
        raise ValueError('power must be finite and positive')
    effective_eta = reference_ratio_eta(eta, eta_floor=floor)
    return expit(power * signal_log_ratio(fields, rates, effective_eta))


def publish_signal_baseline(base_run, study_run, samples, templates, n_bins):
    from .colab_reference import publish_08_baseline
    if reference_denominator(study_run.config) != 'S':
        raise ValueError('Notebook 04 must use the selected S reference')
    return publish_08_baseline(base_run, study_run, samples, templates, n_bins, baseline='04')


def load_signal_baseline(base_run, n_bins, **kwargs):
    from .colab_reference import load_08_baseline
    if reference_denominator(base_run.config) != 'S':
        raise ValueError('Notebooks 05--07 must use the selected S reference')
    return load_08_baseline(base_run, n_bins, baseline='04', **kwargs)
