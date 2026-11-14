"""Per-bin MC precision displays; no adaptive binning or sample generation."""
from pathlib import Path

import numpy as np
import pandas as pd

from .colab_discriminators import precision_table
from .data import SAMPLES


def save_precision_tables(study, mus=(.02, .054, .122, 1.)):
    """Write all fixed cells, all six sources and every moving scan partition."""
    samples, output = study['samples'], study['output']/'mc_precision'
    output.mkdir(parents=True, exist_ok=True)
    mus = np.unique(np.asarray(mus, float))
    if not len(mus) or np.any(~np.isfinite(mus)) or np.any(mus < 0):
        raise ValueError('Precision probes must be finite nonnegative strengths')
    summary = []

    def record(frame, **keys):
        positive = frame.raw_yield > 0
        for key, value in keys.items():
            frame[key] = value
        summary.append(dict(**keys, cells=len(frame), positive_cells=int(positive.sum()),
            no_sampled_support=int((frame.status == 'no_sampled_support').sum()),
            nonpositive_cells=int((frame.status == 'nonpositive').sum()),
            cells_above_10pct=int((frame.relative_mc_sigma > .1).sum()),
            cells_above_30pct=int((frame.relative_mc_sigma > .3).sum()),
            cells_above_100pct=int((frame.relative_mc_sigma > 1.).sum()),
            maximum_relative_mc_sigma=frame.relative_mc_sigma.max(),
            positive_yield_fraction_above_10pct=frame.loc[frame.relative_mc_sigma > .1, 'raw_yield'].sum()
                /frame.loc[positive, 'raw_yield'].sum() if positive.any() else np.nan))
        return frame

    for dimension, counts in study['fixed_counts'].items():
        frames = [record(precision_table(counts, samples['rates'], samples['target'], mu=mu),
                         dimension=dimension, source='total', mu=mu) for mu in mus]
        pd.concat(frames, ignore_index=True).to_csv(output/f'fixed_{dimension}d_total.csv.gz', index=False)
        frames = []
        for source in SAMPLES:
            frame = precision_table(counts, samples['rates'], samples['target'], source=source)
            # Retain the relevant source count; the shared raw archive holds all sources.
            frame = frame.drop(columns=['n_'+s for s in SAMPLES if s != source]).rename(columns={'n_'+source: 'mc_count'})
            frames.append(record(frame, dimension=dimension, source=source, mu=np.nan))
        pd.concat(frames, ignore_index=True).to_csv(output/f'fixed_{dimension}d_sources.csv.gz', index=False)
    frames = []
    moving = study['resolution_bank']
    for n in study['config']['recap_bin_counts']:
        for mu in study['config']['scan_grid']:
            eta = min((e for e, bins in moving.lookup if bins == n), key=lambda e: abs(e-mu))
            frame = precision_table(moving.raw_counts(eta, n), samples['rates'], samples['target'], mu=mu)
            frames.append(record(frame, dimension=1, source='moving_total', mu=mu, n_bins=n))
    pd.concat(frames, ignore_index=True).to_csv(output/'moving_total.csv.gz', index=False)
    pd.DataFrame(summary).to_csv(output/'summary.csv', index=False)
    study['precision_mus'] = mus
    study['precision_output'] = output
    study['precision_summary'] = pd.DataFrame(summary)
    return study['precision_summary']


def plot_mc_precision(study, *, source_3d=(), output=None):
    """Yield figures one at a time, including every 3D cell in labelled slices.

    Total-yield plots use raw nominal-NI predictions at the declared strengths.
    Source plots include independent NI down/nominal/up anchor statistics.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    n, edges = study['config']['n_bins'], np.asarray(study['config']['bin_edges'])
    samples, mus = study['samples'], study['precision_mus']
    centers = (edges[:-1]+edges[1:])/2
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#dddddd')
    cmap.set_over('#a50f15')
    norm = LogNorm(.01, 10.)
    legend = [Patch(color='#dddddd', label='No sampled support: uncertainty unknown'),
              Patch(color='#b15aab', label='Nonpositive raw yield: relative uncertainty undefined')]

    def table(d, mu=None, source=None):
        return precision_table(study['fixed_counts'][d], samples['rates'], samples['target'], mu, source)

    def draw(ax, frame, shape, xedges, yedges):
        values = frame.relative_mc_sigma.to_numpy().reshape(shape)
        # True zero variances (all events in one cell) use the under-range color.
        positive = frame.status.to_numpy().reshape(shape) == 'positive'
        colors = np.where(positive, np.maximum(values, 1e-12), np.nan)
        mesh = ax.pcolormesh(xedges, yedges, colors.T, cmap=cmap, norm=norm, shading='flat', rasterized=True)
        invalid = frame.status.to_numpy().reshape(shape) == 'nonpositive'
        if invalid.any():
            ax.pcolormesh(xedges, yedges, np.ma.masked_where(~invalid.T, np.ones(invalid.T.shape)),
                          cmap=ListedColormap(['#b15aab']), vmin=0, vmax=1, rasterized=True)
        return mesh

    def colorbar(fig, mesh, axes):
        cb = fig.colorbar(mesh, ax=axes, extend='both', shrink=.8)
        cb.set_ticks([.01, .1, 1., 10.], labels=['1%', '10%', '100%', '1000%'])
        cb.set_label(r'Relative MC uncertainty $\sigma_{\mathrm{MC}}/\nu_{\mathrm{raw}}$')

    def finish(name, fig):
        if output is not None:
            Path(output).mkdir(parents=True, exist_ok=True)
            for ext in ('png', 'pdf'):
                fig.savefig(Path(output)/f'10_mc_precision_{name}.{ext}', dpi=140, bbox_inches='tight')
        return name, fig

    # The same cells at low mu, both secondary-minimum regions and truth.
    fig, axes = plt.subplots(2, len(mus), figsize=(4.4*len(mus), 7.4), squeeze=False, layout='constrained')
    for i, mu in enumerate(mus):
        frame = table(1, mu)
        ax = axes[0, i]
        ax.plot(centers, frame.relative_mc_sigma, '.-', markersize=3)
        for status, color in [('no_sampled_support', '#999999'), ('nonpositive', '#b15aab')]:
            bad = frame.status == status
            ax.plot(centers[bad], np.full(bad.sum(), .02), 's', color=color, markersize=3,
                    transform=ax.get_xaxis_transform())
        ax.axhline(.1, color='gray', linestyle='--', linewidth=1)
        ax.set_yscale('symlog', linthresh=.01)
        ax.set(ylim=(0, None), xlabel=r'$z_{NI}$', ylabel=r'$\sigma_{MC}/\nu_{raw}$',
               title=rf'1D, $\mu={mu:g}$')
        ax.yaxis.set_major_formatter(PercentFormatter(1.))
        mesh = draw(axes[1, i], table(2, mu), (n, n), edges, edges)
        axes[1, i].set(xlabel=r'$z_{NI}$', ylabel=r'$z_{SBI}$', title=rf'2D, $\mu={mu:g}$')
    colorbar(fig, mesh, axes[1].tolist())
    fig.legend(handles=legend, loc='outside lower center', ncol=2, fontsize=9)
    fig.suptitle('Raw total-yield MC precision, nominal NI; fixed binning and sample size')
    yield finish('fixed_totals', fig)

    def atlas(mu=None, source=None):
        frame = table(3, mu, source)
        columns = min(6, n)
        rows = int(np.ceil(n/columns))
        fig, axes = plt.subplots(rows, columns, figsize=(3.1*columns, 2.7*rows),
                                 squeeze=False, layout='constrained')
        # C-order flattening is (NI, SBI, B); a slice preserves all 36x36 cells.
        index = np.arange(n**3).reshape((n, n, n))
        for b, ax in enumerate(axes.flat):
            if b >= n:
                ax.set_visible(False)
                continue
            mesh = draw(ax, frame.iloc[index[:, :, b].ravel()], (n, n), edges, edges)
            ax.set(title=rf'$z_B\in[{edges[b]:.3f},{edges[b+1]:.3f}]$',
                   xlabel=r'$z_{NI}$', ylabel=r'$z_{SBI}$')
        colorbar(fig, mesh, list(axes.flat))
        fig.legend(handles=legend, loc='outside lower center', ncol=2, fontsize=9)
        label = rf'total yield, $\mu={mu:g}$, nominal NI' if source is None else f'{source} source'
        fig.suptitle('3D MC precision: '+label+'; every cell shown, no projections or merging')
        return fig

    for mu in mus:
        yield finish(f'3d_mu_{mu:g}', atlas(mu=mu))

    # Source uncertainties are particularly useful for NI morph support changes.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout='constrained')
    for ax, source in zip(axes.flat, SAMPLES):
        mesh = draw(ax, table(2, source=source), (n, n), edges, edges)
        ax.set(title=source, xlabel=r'$z_{NI}$', ylabel=r'$z_{SBI}$')
    colorbar(fig, mesh, list(axes.flat))
    fig.legend(handles=legend[:1], loc='outside lower center', fontsize=9)
    fig.suptitle('2D per-source shape MC precision; frozen selected yields')
    yield finish('2d_sources', fig)
    for source in source_3d:
        if source not in SAMPLES:
            raise ValueError('Unknown source requested for 3D precision plot')
        yield finish('3d_source_'+source, atlas(source=source))

    # A row is a separate partition frozen at eta=mu, not a moving fit bin.
    frame = pd.read_csv(study['precision_output']/'moving_total.csv.gz')
    grid = np.asarray(study['config']['scan_grid'])
    boundaries = np.r_[grid[0]-(grid[1]-grid[0])/2, (grid[:-1]+grid[1:])/2,
                       grid[-1]+(grid[-1]-grid[-2])/2] if len(grid) > 1 else np.r_[grid-.01, grid+.01]
    from .pipeline import observable_bin_edges
    bin_counts = study['config']['recap_bin_counts']
    fig, axes = plt.subplots(1, len(bin_counts), figsize=(5*len(bin_counts), 5.8), squeeze=False, layout='constrained')
    for ax, bins in zip(axes[0], bin_counts):
        mesh = draw(ax, frame[frame.n_bins == bins], (len(grid), bins), boundaries,
                    observable_bin_edges(bins, 'reference_ratio'))
        ax.set(xlabel=r'Frozen anchor $\eta=\mu_0$', ylabel=r'Moving-observable bin in $z_\eta$',
               title=f'{bins} bins')
    colorbar(fig, mesh, list(axes[0]))
    fig.suptitle('Moving-observable recap: raw total-yield MC precision at each test')
    fig.legend(handles=legend, loc='outside lower center', fontsize=9)
    yield finish('moving_total', fig)


def plot_resolution_recap(study, output=None):
    """Direct 36/72/108/144-bin scans and local information with both references."""
    import matplotlib.pyplot as plt
    from .discriminator_unbinned import LEARNED_LABEL
    bins = study['config']['recap_bin_counts']
    styles = {'analytic unbinned': ('black', '--', 'Analytical unbinned'),
              LEARNED_LABEL: ('C4', ':', 'Learned unbinned (03 physical Asimov)')}
    styles.update({f'moving reference ratio ({n} bins)': (f'C{i}', '-', f'{n} bins') for i, n in enumerate(bins)})
    for name, xlim in [('recap_scans', None), ('recap_secondary', (.02, .3)), ('recap_information', None)]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
        is_information = name == 'recap_information'
        frame = study['resolution_information' if is_information else 'resolution_scans']
        for ax, syst in zip(axes, (False, True)):
            for label, (color, ls, display) in styles.items():
                rows = frame[(frame.systematics == syst)&(frame.model == label)]
                if is_information:
                    if label in ('analytic unbinned', LEARNED_LABEL):
                        ax.axhline(rows.information_fraction.iloc[0], color=color, ls=ls, label=display)
                    else:
                        rows = rows.sort_values('eta')
                        ax.plot(rows.eta, rows.information_fraction.where(rows.width_valid), color=color, ls=ls, label=display)
                else:
                    rows = rows.sort_values('mu')
                    ax.plot(rows.mu, rows.q.where(rows.valid), color=color, ls=ls, label=display)
                    bad = rows.loc[~rows.valid]
                    if len(bad):
                        marker, = ax.plot(bad.mu, np.zeros(len(bad)), 'x', color=color,
                                          transform=ax.get_xaxis_transform(), clip_on=False)
                        marker.set_in_layout(False)
            ax.set(xlabel=r'Frozen $\eta$' if is_information else r'Tested $\mu_0=\eta$',
                   ylabel='Local information / curvature fraction' if is_information else r'$q(\mu_0)$',
                   title='NI profiled' if syst else 'NI fixed', ylim=(0, None))
            if xlim:
                ax.set_xlim(*xlim)
            ax.legend(fontsize=10)
        fig.suptitle('Moving learned observable: same MC bank, direct fills, no floor')
        if output is not None:
            Path(output).mkdir(parents=True, exist_ok=True)
            for ext in ('png', 'pdf'):
                fig.savefig(Path(output)/f'10_{name}.{ext}', dpi=160, bbox_inches='tight')
        yield name, fig
