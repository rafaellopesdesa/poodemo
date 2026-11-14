"""Descriptive own-model distributions; failed toys are reported explicitly."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .own_model_toys import CASES


def plot_own_model_toys(study, output=None, *, labels=None, styles=None,
                        filename_prefix='own_model_q', title_note=None):
    labels = CASES if labels is None else labels
    styles = {} if styles is None else styles
    figures = []
    for mu, rows in study['results'].groupby('mu_test'):
        rows = rows.loc[rows.case.isin(labels)]
        valid = rows.loc[rows.valid & np.isfinite(rows.q)]
        maximum = max(1., valid.q.max()) if len(valid) else 1.
        bins = np.linspace(0., maximum*1.01, 51)
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        notes = []
        for color, (case, label) in zip(plt.rcParams['axes.prop_cycle'].by_key()['color'], labels.items()):
            style = {'color': color, 'lw': 1.6, **styles.get(case, {})}
            all_rows = rows.loc[rows.case == case]
            q = np.sort(valid.loc[valid.case == case, 'q'].to_numpy())
            notes.append(f'{case}: {len(q)}/{len(all_rows)} valid')
            if not len(q):
                continue
            axes[0].hist(q, bins=bins, density=True, histtype='step', log=True, label=label, **style)
            # Include all ties at zero in the initial CDF jump.
            values, counts = np.unique(q, return_counts=True)
            cdf = np.cumsum(counts)/len(q)
            axes[1].step(np.r_[-.01, values, maximum*1.01], np.r_[0., cdf, 1.],
                         where='post', label=label, **style)
            tail = 1.-cdf
            # Keep the final zero: the last positive plateau extends all
            # the way to the largest observation, before dropping off-scale.
            axes[2].step(np.r_[0., values], np.r_[np.mean(q > 0), tail],
                         where='post', label=label, **style)
        axes[0].set(ylabel='Probability density (including the zero bin)', yscale='log')
        axes[1].set(ylabel=r'Empirical $P(q\leq t)$', ylim=(0, 1.02))
        axes[2].set(ylabel=r'Empirical $P(q>t)$', yscale='log', ylim=(.2/max(1, rows.groupby('case').size().max()), 1.05))
        for ax in axes:
            ax.set(xlabel=r'Test statistic $q_{\mu_0}$', xlim=(0, maximum*1.01))
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        note = r'$\eta=\mu_0$ fixed in the 36-bin fit' if title_note is None else title_note
        fig.suptitle(fr'Own-model stat-only toys, $\mu_0={mu:g}$; ' + note)
        status = '; '.join(notes) if len(notes) <= 4 else '; '.join(notes[:3])+'\n'+'; '.join(notes[3:])
        fig.text(.5, .01, status+' — curves conditional on valid fits', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, .95))
        name = f'{filename_prefix}_mu_{mu:g}'
        if output is not None:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            for extension in ('png', 'pdf'):
                fig.savefig(directory/f'{name}.{extension}', dpi=160, bbox_inches='tight')
        figures.append((name, fig))
    return figures


def plot_moving_resolution(study, output=None):
    """Four resolutions and both unbinned references; log q and MLE histograms."""
    labels = {f'moving{n}': f'Moving observable: {n} bins' for n in study['bin_counts']}
    labels.update(analytic=CASES['analytic'], learned=CASES['learned'])
    styles = {'analytic': {'color': 'black', 'ls': '--', 'lw': 1.3},
              'learned': {'color': '.45', 'ls': ':', 'lw': 1.8}}
    figures = plot_own_model_toys(study, output, labels=labels, styles=styles,
        filename_prefix='moving_resolution_q', title_note=r'$\eta=\mu_0$ fixed in each 1D partition')
    for mu, rows in study['results'].groupby('mu_test'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        valid = rows.loc[rows.valid & np.isfinite(rows.q) & rows.case.isin(labels)]
        upper = max(3., valid.mu_hat.max()) if len(valid) else 3.
        edges = np.linspace(0., upper, 121)
        for color, (case, label) in zip(plt.rcParams['axes.prop_cycle'].by_key()['color'], labels.items()):
            values = np.sort(valid.loc[valid.case == case, 'mu_hat'].to_numpy())
            if not len(values):
                continue
            style = {'color': color, 'lw': 1.6, **styles.get(case, {})}
            axes[0].hist(values, bins=edges, density=True, log=True, histtype='step', label=label, **style)
            x, count = np.unique(values, return_counts=True)
            axes[1].step(np.r_[-.01, x, upper], np.r_[0., np.cumsum(count)/len(values), 1.],
                         where='post', label=label, **style)
        axes[0].set(ylabel='Probability density', yscale='log')
        axes[1].set(ylabel=r'Empirical $P(\widehat\mu\leq m)$', ylim=(0, 1.02))
        for ax in axes:
            ax.axvline(.5, color='.7', ls='--', lw=1.)
            ax.set(xlabel=r'Global best fit $\widehat\mu$', xlim=(0, upper))
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(fr'Best-fit branches at $\mu_0={mu:g}$; dashed line: diagnostic split at $\widehat\mu=0.5$')
        fig.tight_layout()
        name = f'moving_resolution_mu_hat_mu_{mu:g}'
        if output is not None:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            for extension in ('png', 'pdf'):
                fig.savefig(directory/f'{name}.{extension}', dpi=160, bbox_inches='tight')
        figures.append((name, fig))
    return figures
