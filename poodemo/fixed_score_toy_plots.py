"""Logarithmic score-toy distributions and global best-fit branches."""
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from .own_model_toy_plots import plot_own_model_toys


def plot_fixed_score_toys(study, output):
    labels = study['labels']
    styles = {'analytic': {'color': 'black', 'ls': '--', 'lw': 1.4},
              'learned': {'color': '#CC79A7', 'ls': '-.', 'lw': 1.6}}
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    sizes = {'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 13,
             'ytick.labelsize': 13, 'figure.titlesize': 17}
    with plt.style.context(hep.style.ATLAS), plt.rc_context(sizes):
        figures = plot_own_model_toys(study, labels=labels, styles=styles,
            filename_prefix='fixed_score_q', title_note=r'score anchor $\eta=1$ fixed in every fit')
        for _, fig in figures:
            fig.axes[0].set_ylabel('Probability density')
            fig.axes[0].legend(fontsize=10)
            fig.tight_layout(rect=(0, .06, 1, .94))
        for mu, rows in study['results'].groupby('mu_test'):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            valid = rows.loc[rows.valid & np.isfinite(rows.q)]
            upper = study['mu_bounds'][1]
            edges = np.linspace(0., upper, 121)
            notes = []
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for color, (case, label) in zip(colors, labels.items()):
                values = np.sort(valid.loc[valid.case == case, 'mu_hat'].to_numpy())
                notes.append(f'{case}: {len(values)}/{int(rows.case.eq(case).sum())} valid')
                if not len(values):
                    continue
                style = {'color': color, 'lw': 1.6, **styles.get(case, {})}
                axes[0].hist(values, bins=edges, density=True, log=True, histtype='step', label=label, **style)
                x, counts = np.unique(values, return_counts=True)
                axes[1].step(np.r_[-.01, x, upper], np.r_[0., np.cumsum(counts)/len(values), 1.],
                             where='post', **style)
            axes[0].set(ylabel='Probability density', yscale='log')
            axes[1].set(ylabel=r'Empirical $P(\widehat\mu\leq m)$', ylim=(0., 1.02))
            for ax in axes:
                ax.axvline(.5, color='.6', ls='--', lw=1.)
                ax.set(xlabel=r'Global best fit $\widehat\mu$', xlim=(0., upper))
                ax.grid(alpha=.2)
            axes[0].legend(fontsize=10)
            fig.suptitle(fr'Fixed-score toys at $\mu_0={mu:g}$; diagnostic branch split at $\widehat\mu=0.5$')
            fig.text(.5, .01, '; '.join(notes)+' — conditional on valid fits', ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .06, 1, .94))
            name = f'fixed_score_mu_hat_mu_{mu:g}'
            figures.append((name, fig))
        for name, fig in figures:
            for extension in ('png', 'pdf'):
                fig.savefig(directory/f'{name}.{extension}', dpi=160, bbox_inches='tight')
    return figures
