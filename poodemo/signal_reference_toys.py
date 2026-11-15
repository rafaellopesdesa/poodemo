"""Signal-reference preparation; reuse the unchanged own-model toy engines."""
from pathlib import Path
import numpy as np
from .colab_mc import _integer
from .data import file_digest
from .mc_yields import yield_policy
from .own_model_toys import VERSION, PoissonTemplate, load_signal_fields
from .toy_likelihood import StatOnlyLikelihood
from .toy_study import _array_digest
from .signal_reference import reference_denominator


def prepare_signal_model_toys(run, samples, *, mu_values=(0., 1.4), exposure=1., progress=print):
    """Prepare the four frozen models, reusing 04/06 and the 03 reference."""
    from .colab_discriminators import cached_analytic_quadrature, fixed_counts
    from .colab_mc import fill_colab_histograms
    from .discriminator_convergence import merge_cells
    from .toy_sources import load_frozen_selector, sample_simulator_poisson

    mus = tuple(float(mu) for mu in mu_values)
    if (not mus or len(set(mus)) != len(mus) or any(not np.isfinite(mu) or mu < 0 for mu in mus)):
        raise ValueError('Use unique finite nonnegative hypotheses')
    if (reference_denominator(run.config) != 'S' or run.config.get('reference_ratio_power') != 1.
            or run.config.get('reference_ratio_eta_floor', 0.) != 0.):
        raise ValueError('Notebook 07 requires the S reference, a=1 and no eta floor')
    _integer(samples['target'], 'selected target', 2)
    if samples['target'] > 10_000_000:
        raise ValueError('Notebook 07 reference banks may not exceed 10M per source')
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError('exposure must be finite and positive')
    policy = yield_policy(run.config)
    # Direct fills include eta=0 exactly; 05 interpolation is unnecessary.
    moving = fill_colab_histograms(run, samples, etas=mus, bin_counts=[36],
                                   prefixes=[samples['target']], progress=progress)
    fine = fixed_counts(samples, n_bins=36, progress=progress)
    merged, edges = merge_cells(fine['counts'], fine['edges'], 9)
    fixed_fields = merged[:4].reshape(4, -1)*samples['rates'][:4, None]/samples['target']
    templates = {'fixed9': PoissonTemplate(fixed_fields, policy=policy, exposure=exposure)}
    templates['moving36'] = {mu: PoissonTemplate(moving.anchors(mu, 36)[0], policy=policy,
                                               exposure=exposure) for mu in mus}
    progress('Preparing the learned finite S-bank model from cached ratios')
    fields, normalizers = load_signal_fields(samples)
    rates = np.asarray(samples['rates'][:4], dtype=float)
    guard = StatOnlyLikelihood(np.empty((4, 0)), rates, exposure=exposure, validity_fields=fields)
    quad = cached_analytic_quadrature(run)
    analytic_rates = np.asarray(quad['nominal']) @ quad['weights']
    selector, threshold, selection = load_frozen_selector(run)

    def analytic_draw(mu, rng):
        points = sample_simulator_poisson(run, mu, rng, exposure=exposure,
                                         selector=selector, threshold=threshold)
        return StatOnlyLikelihood(run.model.component_densities(points).T, analytic_rates,
                                  exposure=exposure), len(points)

    configuration = dict(version=VERSION, experiment='four own-model stat-only null ensembles',
        mu_values=sorted(mus), exposure=float(exposure), sample_fingerprint=samples['fingerprint'],
        selected_per_source=int(samples['target']), rates=rates.tolist(),
        observable_normalizers=samples['normalizers'].tolist(), learned_bank_normalizers=normalizers.tolist(),
        learned_fields_sha256=_array_digest(fields), fixed_counts_sha256=_array_digest(merged),
        fixed_edges=edges.tolist(), fixed_powers=fine['powers'],
        moving_fields_sha256={str(mu): _array_digest(templates['moving36'][mu].fields) for mu in sorted(mus)},
        moving_bins=36, reference_power=1., reference_ratio_denominator="S", eta_floor=0., yield_policy=policy.metadata(),
        analytic_rates=analytic_rates.tolist(), selection=selection, physics=run.model.to_dict(),
        backend_sha256={name: file_digest(Path(__file__).with_name(name+'.py'))
                        for name in ('own_model_toys', 'toy_likelihood', 'toy_sources', 'mc_yields', 'physics',
                                     'signal_reference', 'signal_reference_toys', 'colab_mc')},
        analytic_normalization='Original selected analytical quadrature: finite numerical integration error',
        learned_normalization='Each ratio normalized once on the frozen S bank; preserves selected component rates')
    del quad
    return dict(configuration=configuration, templates=templates, learned_fields=fields,
                learned_rates=rates, learned_intervals=guard.negative_intervals,
                analytic_rates=analytic_rates, analytic_draw=analytic_draw, exposure=float(exposure),
                mu_values=mus)

