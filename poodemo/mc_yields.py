"""Explicit, bounded zeroing of tiny negative *total* MC bin predictions."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class YieldPolicy:
    max_bin: float = 0.
    max_total: float = 0.

    def __post_init__(self):
        if not np.isfinite([self.max_bin, self.max_total]).all() or min(self.max_bin, self.max_total) < 0:
            raise ValueError('MC zeroing limits must be finite nonnegative expected event counts')

    def apply(self, raw):
        raw = np.asarray(raw, dtype=float)
        finite = bool(np.isfinite(raw).all())
        correction = np.maximum(-raw, 0.)
        largest = float(correction.max(initial=0.)) if finite else np.inf
        total = float(correction.sum()) if finite else np.inf
        negative = int(np.count_nonzero(raw < 0))
        accepted = bool(finite and largest <= self.max_bin and total <= self.max_total)
        report = dict(raw_minimum=float(raw.min()) if finite else np.nan,
                      negative_bins=negative, largest_negative=largest,
                      negative_total=total, zeroing_accepted=accepted,
                      zeroed_bins=negative if accepted else 0,
                      zeroed_total=total if accepted else 0.,
                      rejected_negative=bool(not accepted))
        # An excessive correction rejects the whole parameter point. Leaving
        # its raw negative/nonfinite values makes the Poisson likelihood fail.
        return (np.maximum(raw, 0.) if accepted else raw.copy()), report

    def metadata(self):
        return dict(mc_yield_policy='bounded_zero_v1',
                    mc_zero_max_bin=float(self.max_bin), mc_zero_max_total=float(self.max_total))


def yield_policy(config):
    return YieldPolicy(float(config.get('mc_zero_max_bin', 0.)),
                       float(config.get('mc_zero_max_total', 0.)))


def yield_diagnostics(model, mu, alpha=None):
    """Stateless audit at the reported fit point, separate from optimizer probes."""
    if not hasattr(model, 'yield_policy'):
        return {}
    raw = model.raw_intensity(mu, alpha)
    _, report = model.yield_policy.apply(raw)
    return report
