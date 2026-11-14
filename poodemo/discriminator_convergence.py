"""Cross bin resolution with existing MC prefixes, without generating events."""
from pathlib import Path

import numpy as np
import pandas as pd

from .colab_discriminators import (MAX_SELECTED, FIXED_LABELS, fixed_counts,
                                  source_moments, _histogram, _occupancy)
from .colab_mc import _digest, _integer
from .data import save_json
from .discriminator_unbinned import LEARNED_LABEL
from .inference import component_coefficients
from .mc_yields import yield_policy
from .practical import fit_direct_asimov
from .toy_integration_diagnostics import _atomic_npz

VERSION = 1


def merge_cells(counts, edges, n_bins):
    """Merge adjacent cells on every axis; never regenerate shifted edges."""
    counts, edges = np.asarray(counts), np.asarray(edges)
    fine = len(edges)-1
    n_bins = _integer(n_bins, 'coarse bins', 2)
    if counts.shape != (6, fine, fine, fine) or fine % n_bins:
        raise ValueError('Coarse bins must divide the existing cubic grid exactly')
    factor = fine//n_bins
    merged = counts.reshape(6, n_bins, factor, n_bins, factor, n_bins, factor).sum(axis=(2, 4, 6))
    return merged, edges[::factor].copy()


def contrast_table(counts, rates, target, mu, truth=1.):
    """Nominal-NI contrast precision, including shared-template correlations.

    The second-order bias is a plug-in diagnostic on strictly positive bins.
    It is not reliable at support boundaries or a correction to the likelihood.
    """
    flat = np.asarray(counts).reshape(6, -1)
    means, variances = source_moments(flat, rates, target)
    c, c0 = component_coefficients(mu), component_coefficients(truth)
    x, y = c @ means[:4], c0 @ means[:4]
    delta = x-y
    sigma = np.sqrt((c-c0)**2 @ variances[:4])
    relative = np.divide(sigma, np.abs(delta), out=np.full_like(sigma, np.nan), where=delta != 0)
    relative[(delta == 0) & (sigma > 0)] = np.inf
    regular = (x > 0) & (y > 0)
    empty = flat[:4].sum(axis=0) == 0
    status = np.where(regular, 'regular', np.where(empty, 'no_sampled_support',
        np.where((x < 0) | ((x == 0) & (y > 0)), 'nonpositive_prediction', 'support_boundary')))
    q, bias = np.full_like(x, np.nan), np.full_like(x, np.nan)
    d = delta[regular]/y[regular]
    small = np.abs(d) < 1e-3
    deviance = np.empty_like(d)
    t = d[small]
    deviance[small] = t*t*(.5+t*(-1/3+t*(.25+t*(-.2+t/6))))
    deviance[~small] = d[~small]-np.log1p(d[~small])
    q[regular] = 2*y[regular]*deviance
    q[(y == 0) & (x >= 0)] = 2*x[(y == 0) & (x >= 0)]
    # Half the Hessian of the Poisson deviance, contracted with independent
    # source variances. x and y use the SAME templates, so they are correlated.
    bias[regular] = y[regular]*np.sum(variances[:4, regular]
        *(c[:, None]/x[regular]-c0[:, None]/y[regular])**2, axis=0)
    result = pd.DataFrame(dict(bin=np.arange(len(x)), raw_test_yield=x, raw_truth_yield=y,
        delta_yield=delta, mc_sigma_delta=sigma, relative_mc_sigma_delta=relative,
        raw_q_contribution=q, mc_bias_second_order=bias, status=status))
    return result


def support_audit(model, mu, alpha):
    """A tiny permitted zeroing can still give infinite Poisson deviance."""
    expected, _ = model.yield_policy.apply(model.raw_intensity(mu, alpha))
    bad = (expected <= 0) & (model.truth_intensity > 0)
    return dict(support_mismatch_bins=int(bad.sum()),
                support_mismatch_asimov=float(model.truth_intensity[bad].sum()))


def run_convergence_study(run, study, *, bin_counts=(9, 18, 36),
                          prefixes=(2_500_000, 5_000_000, 10_000_000),
                          probes=(.02, .054, .122, .344, 1.4), progress=print):
    """A fixed factorial design; every prefix retains the full-bank rates/map."""
    samples = study['samples']
    fine = _integer(study['config']['n_bins'], 'fine bins', 2)
    bins = sorted(set(_integer(n, 'coarse bins', 2) for n in bin_counts))
    sizes = sorted(set(_integer(n, 'prefix', 2) for n in prefixes))
    if not bins or bins[-1] != fine or any(b % a for a, b in zip(bins, bins[1:])):
        raise ValueError('Use a nested divisor chain ending at the existing fine grid')
    if not sizes or sizes[-1] != samples['target'] or sizes[-1] > MAX_SELECTED:
        raise ValueError('Prefixes must end at the completed bank target, at most 10M; no generation')
    grid = np.asarray(study['config']['scan_grid'])
    probes = np.asarray(probes, float)
    if not len(probes) or np.any(~np.isfinite(probes)) or np.any(probes < grid.min()) or np.any(probes > grid.max()):
        raise ValueError('Choose finite diagnostic strengths inside the saved scan range')
    mapping = pd.DataFrame(dict(requested_mu=probes,
        mu=[grid[np.argmin(np.abs(grid-p))] for p in probes]))
    actual_probes = np.sort(mapping.mu.unique())
    policy, truth = yield_policy(run.config), float(run.config['asimov_mu'])
    if policy.metadata() != study['config']['yield_policy'] or truth != study['config']['asimov_mu']:
        raise ValueError('Use the same Asimov truth and yield policy as the main study')
    config = dict(version=VERSION, source_study_sha256=_digest(study['config']),
        sample_fingerprint=samples['fingerprint'], bin_counts=bins, prefixes=sizes,
        fine_bins=fine, scan_grid=grid.tolist(), probes=mapping.to_dict('records'),
        powers=study['config']['powers'], rates=samples['rates'].tolist(),
        normalizers=samples['normalizers'].tolist(), yield_policy=policy.metadata(),
        asimov_mu=truth, nested_cells=True, frozen_rates_and_map=True,
        asimov_source='same templates at each prefix', correlated_prefixes=True,
        no_generation=True, max_selected_per_source=MAX_SELECTED,
        automatic_bias_correction=False, inverse_N_fit=False,
        contrast_diagnostic='nominal NI only; excludes rate and network uncertainty')
    identity = _digest(config)
    output = study['output']/'3d_convergence_v1'/identity[:20]
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'config.json', config)
    mapping.to_csv(output/'probe_mapping.csv', index=False)
    references = study['scans'][study['scans'].model.isin(['analytic unbinned', LEARNED_LABEL])].copy()
    for label in ['analytic unbinned', LEARNED_LABEL]:
        for syst in (False, True):
            rows = references[(references.model == label) & (references.systematics == syst)]
            if len(rows) != len(grid) or rows.mu.duplicated().any() or not np.array_equal(np.sort(rows.mu), grid):
                raise ValueError('Missing or duplicated unbinned comparison')
    scans, occupancy, summaries, edges_by_bins = [], [], [], {}
    for size in sizes:
        # Reuse the existing full-bank checkpoint. Smaller prefixes have their
        # own resumable checkpoints and read cached ratios, never networks.
        subset = {**samples, 'target':size}
        filled = fixed_counts(subset, fine, study['config']['powers'], progress)
        if size == samples['target'] and not np.array_equal(filled['counts'], study['fixed_counts'][3]):
            raise ValueError('The full-prefix counts differ from the main notebook 10 study')
        _atomic_npz(output/f'source_counts_{size}.npz', counts=filled['counts'],
            edges=filled['edges'], rates=samples['rates'], target=np.array(size))
        for n in bins:
            counts, edges = merge_cells(filled['counts'], filled['edges'], n)
            if not np.all(counts.sum(axis=(1, 2, 3)) == size):
                raise ValueError('Merging changed the source event totals')
            edges_by_bins[str(n)] = edges.tolist()
            keys = dict(n_bins=n, selected_per_source=size, total_cells=n**3)
            occupancy.append({**_occupancy(counts, subset, truth, f'3D {n} cubed'), **keys})
            for syst in (False, True):
                model = _histogram(counts, subset, truth, policy, syst)
                rows = []
                for mu in grid:
                    fit = fit_direct_asimov(run, model, mu, syst)
                    nominal = support_audit(model, mu, np.zeros(model.n_nuisance))
                    if not fit['valid'] and not syst and nominal['support_mismatch_bins']:
                        fit['fit_message'] += '; nonpositive prediction with positive Asimov yield'
                    rows.append(dict(mu=mu, systematics=syst, **keys, **fit,
                                     **{'nominal_'+k:v for k, v in nominal.items()}))
                scans.extend(rows)
                progress(f'3D {n} x {n} x {n}, {size:,}/source, NI profiled={syst}: '
                         f'{sum(row["valid"] for row in rows)}/{len(rows)} valid')
            contrasts = []
            for mu in actual_probes:
                frame = contrast_table(counts, samples['rates'], size, mu, truth)
                frame['mu'] = mu
                contrasts.append(frame)
                good = frame.status == 'regular'
                noise = good & (frame.mc_sigma_delta >= frame.delta_yield.abs())
                total_q = frame.loc[good, 'raw_q_contribution'].sum()
                summaries.append(dict(**keys, mu=mu, systematics=False,
                    regular_cells=int(good.sum()), no_sampled_support=int((frame.status == 'no_sampled_support').sum()),
                    support_boundary_cells=int((frame.status == 'support_boundary').sum()),
                    nonpositive_cells=int((frame.status == 'nonpositive_prediction').sum()),
                    noise_dominated_regular_cells=int(noise.sum()), q_regular_sum=total_q,
                    q_fraction_noise_dominated=frame.loc[noise, 'raw_q_contribution'].sum()/total_q if total_q else np.nan,
                    mc_bias_second_order_regular=frame.loc[good, 'mc_bias_second_order'].sum() if good.any() else np.nan))
            pd.concat(contrasts, ignore_index=True).to_csv(output/f'contrast_{size}_{n}.csv.gz', index=False)
    frame = pd.DataFrame(scans)
    for label, column in [('analytic unbinned', 'q_analytic'), (LEARNED_LABEL, 'q_learned')]:
        ref = references[references.model == label][['mu', 'systematics', 'q', 'valid']].copy()
        ref['q'] = ref.q.where(ref.valid)
        frame = frame.merge(ref.drop(columns='valid').rename(columns={'q':column}),
                            on=['mu', 'systematics'], validate='many_to_one')
        frame['delta_'+column] = frame.q-frame[column]
    baseline = frame[(frame.n_bins == fine) & (frame.selected_per_source == samples['target'])]
    previous = study['scans'][study['scans'].model == FIXED_LABELS[-1]][['mu', 'systematics', 'q', 'valid']]
    checks = baseline[['mu', 'systematics', 'q', 'valid']].merge(previous,
        on=['mu', 'systematics'], suffixes=('_study', '_main'), validate='one_to_one')
    checks['delta_q'] = checks.q_study-checks.q_main
    checks['passed'] = (checks.valid_study == checks.valid_main) & (~checks.valid_main | (checks.delta_q.abs() <= 1e-7))
    tables = dict(scans=frame, references=references, occupancy=pd.DataFrame(occupancy),
                  contrast_summary=pd.DataFrame(summaries), baseline_checks=checks)
    for name, data in tables.items():
        data.to_csv(output/(name+'.csv'), index=False)
    save_json(output/'bin_edges.json', edges_by_bins)
    if len(checks) != 2*len(grid) or not checks.passed.all():
        raise ValueError('The full-prefix fine-grid scan does not reproduce the main study; inspect baseline_checks.csv')
    return dict(**tables, config=config, output=output, identity=identity, probe_mapping=mapping)


def plot_convergence_study(result, output=None):
    """Yield comparisons with both unbinned references; no bias subtraction."""
    import matplotlib.pyplot as plt
    bins, sizes = result['config']['bin_counts'], result['config']['prefixes']
    scans, refs = result['scans'], result['references']
    styles = [('analytic unbinned', 'black', '--', 'Analytical unbinned'),
              (LEARNED_LABEL, 'C4', ':', 'Learned unbinned (03 physical Asimov)')]

    def finish(name, fig):
        if output is not None:
            Path(output).mkdir(parents=True, exist_ok=True)
            for ext in ('png', 'pdf'):
                fig.savefig(Path(output)/f'10_3d_{name}.{ext}', dpi=150, bbox_inches='tight')
        return name, fig

    def curve(ax, rows, *, x='mu', y='q', color, label, ls='-', marker=None):
        rows = rows.sort_values(x)
        ax.plot(rows[x], rows[y].where(rows.valid), color=color, label=label, ls=ls,
                lw=1.7, marker=marker, markersize=4)
        bad = rows.loc[~rows.valid]
        if len(bad):
            marker, = ax.plot(bad[x], np.full(len(bad), .02), 'x', color=color,
                             transform=ax.get_xaxis_transform(), clip_on=True)
            marker.set_in_layout(False)
        if 'nominal_support_mismatch_bins' in rows:
            flagged = rows[rows.valid & (rows.nominal_support_mismatch_bins > 0)]
            if len(flagged):
                ax.plot(flagged[x], flagged[y], 'D', color=color, markersize=5, fillstyle='none')

    def references(ax, syst, difference=False):
        for label, color, ls, display in styles:
            rows = refs[(refs.systematics == syst) & (refs.model == label)].copy()
            if difference:
                oracle = refs[(refs.systematics == syst) & (refs.model == 'analytic unbinned')].set_index('mu')
                rows['q'] = rows.q-rows.mu.map(oracle.q.where(oracle.valid))
            curve(ax, rows, color=color, label=display, ls=ls)

    for name, grouping, zoom, difference in [('by_sample_size', 'size', False, False),
            ('by_binning', 'bins', False, False), ('secondary', 'size', True, False),
            ('difference', 'size', False, True)]:
        groups = sizes if grouping == 'size' else bins
        fig, axes = plt.subplots(2, len(groups), figsize=(6*len(groups), 8.5), squeeze=False, layout='constrained')
        for row, syst in enumerate((False, True)):
            for ax, group in zip(axes[row], groups):
                references(ax, syst, difference)
                for i, other in enumerate(bins if grouping == 'size' else sizes):
                    n, size = (other, group) if grouping == 'size' else (group, other)
                    data = scans[(scans.systematics == syst) & (scans.n_bins == n) & (scans.selected_per_source == size)]
                    label = rf'${n}^3$ cells' if grouping == 'size' else f'{size/1e6:g}M / source'
                    curve(ax, data, y='delta_q_analytic' if difference else 'q', color=f'C{i}', label=label)
                title = f'{group/1e6:g}M events / source' if grouping == 'size' else rf'${group}^3$ cells'
                ax.set(title=title+(' — NI profiled' if syst else ' — NI fixed'), xlabel=r'Tested $\mu$',
                       ylabel=r'$q-q_{\mathrm{analytical}}$' if difference else r'$q(\mu)$')
                if zoom:
                    ax.set_xlim(.02, .3)
                    # Set the vertical range from this window, not the high-q endpoint.
                    window = scans[(scans.systematics == syst) & (scans.selected_per_source == group)
                                   & scans.mu.between(.02, .3) & scans.valid].q
                    ref_window = refs[(refs.systematics == syst) & refs.mu.between(.02, .3) & refs.valid].q
                    ymax = pd.concat([window, ref_window]).max()
                    ax.set_ylim(0, 1.08*ymax if np.isfinite(ymax) and ymax > 0 else 1.)
                elif not difference:
                    ax.set_ylim(0, None)
                ax.legend(fontsize=8)
        fig.suptitle('Nested 3D bins and existing MC prefixes; frozen learned coordinates and rates'
                     '\nx: invalid fit; hollow diamond: valid fit with a nominal-NI support mismatch')
        yield finish(name, fig)

    probes = sorted(result['probe_mapping'].mu.unique())
    fig, axes = plt.subplots(2, len(probes), figsize=(5*len(probes), 8), squeeze=False, layout='constrained')
    for row, syst in enumerate((False, True)):
        for ax, mu in zip(axes[row], probes):
            for label, color, ls, display in styles:
                ref = refs[(refs.systematics == syst) & (refs.model == label) & (refs.mu == mu)].iloc[0]
                if ref.valid:
                    ax.axhline(ref.q, color=color, ls=ls, label=display)
            for i, n in enumerate(bins):
                data = scans[(scans.systematics == syst) & (scans.n_bins == n) & (scans.mu == mu)].copy()
                data['inverse_n'] = sizes[-1]/data.selected_per_source
                curve(ax, data, x='inverse_n', color=f'C{i}', label=rf'${n}^3$ cells', marker='o')
            ax.set(title=rf'$\mu={mu:.4g}$'+(' — NI profiled' if syst else ' — NI fixed'),
                   xlabel=r'$N_{\max}/N_{MC}$', ylabel=r'$q(\mu)$', ylim=(0, None))
            ax.set_xticks([sizes[-1]/s for s in sizes])
            ax.legend(fontsize=7)
    fig.suptitle('Sample-size dependence at saved scan points; correlated prefixes, no extrapolation'
                 '\nx: invalid fit; hollow diamond: valid fit with a nominal-NI support mismatch')
    yield finish('inverse_n', fig)

    # Compact per-cell contrast diagnostics near the secondary region. Every
    # cell/probe is retained in the CSVs, including undefined relative errors.
    probe = min(probes, key=lambda mu:abs(mu-.122))
    fig, axes = plt.subplots(len(bins), len(sizes), figsize=(5*len(sizes), 3.8*len(bins)),
                             squeeze=False, layout='constrained')
    for row, n in enumerate(bins):
        for ax, size in zip(axes[row], sizes):
            data = pd.read_csv(result['output']/f'contrast_{size}_{n}.csv.gz')
            data = data[np.isclose(data.mu, probe, atol=1e-12, rtol=0) & (data.status == 'regular')]
            positive = (data.delta_yield.abs() > 0) & (data.mc_sigma_delta > 0)
            values = data.loc[positive].copy()
            x = values.delta_yield.abs()/values.raw_truth_yield
            y = values.mc_sigma_delta/values.raw_truth_yield
            ax.scatter(x, y, s=6, alpha=.45, rasterized=True)
            finite = np.r_[x, y]
            low, high = (finite.min()*.7, finite.max()*1.4) if len(finite) else (1e-5, 1.)
            ax.plot([low, high], [low, high], '--', color='black', lw=1, label=r'$\sigma_{\Delta}=|\Delta\nu|$')
            ax.set(xscale='log', yscale='log', xlim=(low, high), ylim=(low, high),
                   xlabel=r'$|\Delta\nu_b|/\nu_b(1)$', ylabel=r'$\sigma_{\Delta,b}/\nu_b(1)$',
                   title=rf'${n}^3$ cells, {size/1e6:g}M / source')
            ax.legend(fontsize=8)
    fig.suptitle(rf'Nominal-NI contrast precision at $\mu={probe:.4g}$; points above the line are MC-noise dominated'
                '\nPositive yields and nonzero contrast/error only; all other cells are recorded in the tables')
    yield finish('contrast_precision', fig)
