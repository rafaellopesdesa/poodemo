"""Descriptive own-model distributions; failed toys are reported explicitly."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .own_model_toys import CASES


def plot_own_model_toys(study, output=None):
    figures = []
    for mu, rows in study['results'].groupby('mu_test'):
        valid = rows.loc[rows.valid & np.isfinite(rows.q)]
        maximum = max(1., valid.q.max()) if len(valid) else 1.
        bins = np.linspace(0., maximum*1.01, 51)
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        notes = []
        for color, (case, label) in zip(plt.rcParams['axes.prop_cycle'].by_key()['color'], CASES.items()):
            all_rows = rows.loc[rows.case == case]
            q = np.sort(valid.loc[valid.case == case, 'q'].to_numpy())
            notes.append(f'{case}: {len(q)}/{len(all_rows)} valid')
            if not len(q):
                continue
            axes[0].hist(q, bins=bins, density=True, histtype='step', lw=1.6, color=color, label=label)
            # Include all ties at zero in the initial CDF jump.
            values, counts = np.unique(q, return_counts=True)
            cdf = np.cumsum(counts)/len(q)
            axes[1].step(np.r_[-.01, values, maximum*1.01], np.r_[0., cdf, 1.],
                         where='post', color=color, label=label)
            tail = 1.-cdf
            # Keep the final zero: the last positive plateau extends all
            # the way to the largest observation, before dropping off-scale.
            axes[2].step(np.r_[0., values], np.r_[np.mean(q > 0), tail],
                         where='post', color=color, label=label)
        axes[0].set_ylabel('Probability density (including the zero bin)')
        axes[1].set(ylabel=r'Empirical $P(q\leq t)$', ylim=(0, 1.02))
        axes[2].set(ylabel=r'Empirical $P(q>t)$', yscale='log', ylim=(1e-4, 1.05))
        for ax in axes:
            ax.set(xlabel=r'Test statistic $q_{\mu_0}$', xlim=(0, maximum*1.01))
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(fr'Own-model stat-only toys, $\mu_0={mu:g}$; $\eta=\mu_0$ fixed in the 36-bin fit')
        fig.text(.5, .01, '; '.join(notes)+' — curves conditional on valid fits', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, .95))
        name = f'own_model_q_mu_{mu:g}'
        if output is not None:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            for extension in ('png', 'pdf'):
                fig.savefig(directory/f'{name}.{extension}', dpi=160, bbox_inches='tight')
        figures.append((name, fig))
    return figures
