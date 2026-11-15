"""A dense bin-count sweep at one frozen signal-reference coordinate.

Use cached predictions only. The analytical density enters solely through the
unbinned information reference. Undefined finite-MC Fisher points remain NaN.
"""
import numpy as np
import pandas as pd

from .colab_mc import _integer, fill_colab_histograms
from .data import save_json
from .inference import component_coefficients
from .pipeline import observable_configuration


def poisson_information(components, mu, weights=None):
    """Stat-only extended-Poisson information and its own rate/shape split."""
    fields = np.asarray(components, float)
    if fields.ndim != 2 or fields.shape[0] != 4 or not np.isfinite(mu) or mu <= 0:
        raise ValueError('Expected (4,N) components and a strictly positive finite mu')
    weights = np.ones(fields.shape[1]) if weights is None else np.asarray(weights, float)
    if (weights.shape != (fields.shape[1],) or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError('Integration weights must be finite and nonnegative')
    inverse = .5/np.sqrt(mu)
    means = component_coefficients(mu) @ fields
    derivative = np.array([1-inverse, inverse, -inverse, 0.]) @ fields
    active = weights > 0
    nonfinite = (~np.isfinite(means) | ~np.isfinite(derivative)) & active
    negative = (means < 0) & active
    nonregular = (means == 0) & (derivative != 0) & active
    rate, drate = float(weights @ means), float(weights @ derivative)
    rate_valid = np.isfinite([rate, drate]).all() and rate > 0
    valid = bool(rate_valid and not np.any(nonfinite | negative | nonregular))
    information = (float(np.sum(weights*np.divide(derivative**2, means,
                   out=np.zeros_like(means), where=means > 0))) if valid else np.nan)
    rate_information = drate**2/rate if rate_valid else np.nan
    return dict(information_valid=valid, information=information,
        rate_information=rate_information, shape_information=information-rate_information,
        rate=rate, rate_derivative=drate, occupied_bins=int(np.count_nonzero(active & (means > 0))),
        empty_bins=int(np.count_nonzero(active & (means == 0))),
        negative_bins=int(negative.sum()), nonregular_bins=int(nonregular.sum()),
        nonfinite_bins=int(nonfinite.sum())), means, derivative


def run_signal_information_study(run, samples, *, bin_counts, eta=1., prefixes=None, progress=print):
    """Add resolutions at one eta without repeating the full likelihood scan."""
    from .colab_discriminators import cached_analytic_quadrature
    if samples['target'] > 10_000_000:
        raise ValueError('The signal information study may not exceed 10M events per source')
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError('Fisher information requires positive finite eta')
    bins = sorted({_integer(n, 'n_bins', 2) for n in bin_counts})
    if not bins:
        raise ValueError('Specify at least one bin count')
    prefixes = [samples['target']] if prefixes is None else prefixes
    bank = fill_colab_histograms(run, samples, etas=[eta], bin_counts=bins,
                                 prefixes=prefixes, progress=progress)
    quad = cached_analytic_quadrature(run)
    reference, _, _ = poisson_information(quad['nominal'], eta, quad['weights'])
    if not reference['information_valid'] or reference['information'] <= 0:
        raise ValueError('The analytical unbinned information reference is invalid')
    del quad
    rows, details = [], []
    for size in bank.prefixes:
        for n in bins:
            info, means, derivative = poisson_information(bank.anchors(eta,n,size)[0], eta)
            shape = reference['shape_information']
            rows.append(dict(eta=float(eta), n_bins=n, selected_per_source=size,
                information_fraction=info['information']/reference['information'],
                shape_information_fraction=info['shape_information']/shape
                    if shape > 1e-12*reference['information'] else np.nan,
                full_information=reference['information'],
                full_rate_information=reference['rate_information'], **info))
            raw = bank.raw_counts(eta,n,size)
            for index in range(n):
                details.append(dict(eta=float(eta), n_bins=n, selected_per_source=size, bin=index,
                    raw_mean=means[index], derivative=derivative[index],
                    nonregular=bool(means[index]==0 and derivative[index]!=0),
                    **{f'count_{name}':int(raw[j,index])
                       for j,name in enumerate(('S','SBI','B','NI','NI_up','NI_down'))}))
    frame, detail = pd.DataFrame(rows), pd.DataFrame(details)
    output = run.path('results','signal_information_binning')
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'information.csv',index=False)
    detail.to_csv(output/'bin_diagnostics.csv',index=False)
    save_json(output/'configuration.json', dict(version=1, eta=float(eta), bin_counts=bins,
        prefixes=bank.prefixes, sample_fingerprint=samples['fingerprint'],
        observable=observable_configuration(run,'reference_ratio'), reference=reference,
        histogram_directory=str(bank.directory),
        definition='NI fixed; raw Poisson means; each model subtracts its own rate information',
        missing_points='Nonfinite/negative means, or zero mean with nonzero derivative; never floored'))
    return dict(information=frame, diagnostics=detail, output=output)


def plot_signal_information(study, *, selected_per_source, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    frame = study['information']
    frame = frame.loc[frame.selected_per_source.eq(selected_per_source)].sort_values('n_bins')
    if frame.empty:
        raise ValueError('The requested MC prefix is absent')
    fig, ax = plt.subplots(figsize=(9,5.2))
    for column,label in [('information_fraction','Total information'),
                         ('shape_information_fraction','Shape information')]:
        ax.plot(frame.n_bins,frame[column].where(frame.information_valid),'o-',label=label)
    bad = frame.loc[~frame.information_valid]
    if len(bad):
        ax.plot(bad.n_bins,np.full(len(bad),.025),'rx',ls='none',
                transform=ax.get_xaxis_transform(),label='Undefined Fisher: MC support')
    ax.set_xscale('log')
    ax.set_xticks(frame.n_bins)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='x',labelrotation=55,labelsize=8)
    ax.set(xlabel='Number of bins',ylabel='Retained Fisher information fraction',
        title=rf'Information retained at $\eta=\mu={frame.eta.iloc[0]:g}$; NI fixed')
    ax.text(.98,.97,f'{int(frame.information_valid.sum())}/{len(frame)} valid resolutions; '
            f'{selected_per_source:,} selected events/source',transform=ax.transAxes,fontsize=9,va='top',ha='right')
    ax.legend(loc='best')
    fig.tight_layout()
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):
        fig.savefig(output/f'04_binning_diagnostics.{suffix}',dpi=160,bbox_inches='tight')
    return fig
