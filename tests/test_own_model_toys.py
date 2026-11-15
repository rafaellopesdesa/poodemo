"""Own-model consistency, bounded Poisson means and deterministic resumption."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from poodemo.colab_mc import _array
from poodemo.mc_yields import YieldPolicy
from poodemo.own_model_toys import (CASES, PoissonTemplate, load_signal_fields,
    prepare_own_model_toys, run_own_model_toys, _source)
from poodemo.toy_likelihood import StatOnlyLikelihood


@pytest.fixture
def prepared():
    fields = np.array([[2., 1.], [4., 5.], [3., 4.], [2., 1.]])
    rates = fields.sum(axis=1)
    templates = {mu: PoissonTemplate(fields) for mu in (0., 1.4)}
    fixed = PoissonTemplate(fields)
    def draw(mu, rng):
        counts = rng.poisson(fixed.means(mu)[0])
        return StatOnlyLikelihood(fields, rates, counts), int(counts.sum())
    return dict(configuration={'version': 1, 'fixture': 'two-cell own-model'},
        mu_values=(0., 1.4), exposure=1., learned_fields=2*fields,
        learned_rates=rates, learned_intervals=np.empty((0, 2)),
        analytic_rates=rates, analytic_draw=draw,
        templates=dict(moving36=templates, fixed9=fixed))


def test_zeroing_is_the_same_rate_and_likelihood_and_preserves_support():
    # First bin is slightly negative at kappa=.5, second stays positive.
    fields = np.array([[1., 0.], [.2, 1.], [.1, 1.], [.095, 1.]])
    template = PoissonTemplate(fields)
    means, report = template.means(.25)
    np.testing.assert_allclose(means, [0., 2.])
    assert report['zeroed_total'] == pytest.approx(.005)
    fit = template.likelihood([0, 3])
    assert fit.expected_count(.25) == 2.
    assert fit.nll(.25) == pytest.approx(2*(2-3*np.log(2)))
    for mu in (0., .25, 1.4):
        assert fit.nll(mu, reference_mu=.25) == pytest.approx(fit.nll(mu)-fit.nll(.25), abs=1e-12)
    assert np.isinf(template.likelihood([1, 3]).nll(.25))
    # A correction over either threshold is forbidden, including empty bins.
    assert np.isinf(PoissonTemplate(fields, policy=YieldPolicy(.001, .05)).likelihood([0, 0]).nll(.25))
    assert np.isinf(PoissonTemplate(np.tile(fields[:, :1], (1, 11))).likelihood(np.zeros(11)).nll(.25))


def test_policy_intervals_match_direct_checks_including_total_budget():
    rng = np.random.default_rng(81)
    fields = rng.uniform(.01, 1., (4, 50))
    fields[1] *= .15
    fields[3] *= .02
    for policy in (YieldPolicy(), YieldPolicy(.01, .05), YieldPolicy(.5, .7)):
        template = PoissonTemplate(fields, policy=policy)
        fit = template.likelihood(np.zeros(50))
        for k in np.linspace(0, 4, 3001):
            assert fit._valid_kappa(k) == template.means(k*k)[1]['zeroing_accepted']


def test_binned_fit_matches_strict_likelihood_when_no_correction():
    fields = np.array([[40., 1.], [0., 1.9], [40., 1.], [10., .1]])
    expected = StatOnlyLikelihood.from_binned(fields, [20, 3]).test_statistic(0, mu_bounds=(0, 4))
    found = PoissonTemplate(fields).likelihood([20, 3]).test_statistic(0, mu_bounds=(0, 4))
    assert expected['mu_hat'] > 1  # Global second branch, not nearest minimum.
    assert found['valid']
    assert found['q'] == pytest.approx(expected['q'], abs=1e-10)
    assert found['mu_hat'] == pytest.approx(expected['mu_hat'], abs=1e-7)


def test_signal_bank_renormalized_once_and_no_enlargement(tmp_path):
    ratio = np.array([[1., 1., 1.], [1., 3., 2.], [.1, .3, .2], [4., 6., 2.]])
    ratios = {}
    for j, name in enumerate(('SBI', 'B', 'NI'), 1):
        path = tmp_path/f'{name}.npy'
        ratios[name] = dict(file=path.name, sha256=_array(path, ratio[j]))
    samples = dict(target=3, directory=tmp_path, manifests={'S': {'chunks': [dict(count=3, ratios=ratios)]}},
                   rates=np.array([2., 9., 7., 3., 4., 2.]))
    fields, norms = load_signal_fields(samples)
    np.testing.assert_allclose(norms, ratio.mean(axis=1))
    np.testing.assert_allclose(fields.mean(axis=1), samples['rates'][:4], rtol=1e-15)
    samples['target'] = 4
    with pytest.raises(ValueError, match='Incomplete'): load_signal_fields(samples)
    samples['target'] = 10_000_001
    with pytest.raises(ValueError, match='10M'): load_signal_fields(samples)


def test_learned_counts_and_fitted_rates_agree(prepared):
    for mu in (0., 1.4):
        draw, audit = _source(prepared, 'learned', mu)
        rng = np.random.default_rng(7)
        totals = []
        for _ in range(1000):
            likelihood, n = draw(rng)
            assert likelihood.expected_count(mu) == pytest.approx(audit['expected_count'], abs=1e-12)
            assert likelihood.observation_weights.sum() == n
            assert np.all(likelihood.observation_weights == np.floor(likelihood.observation_weights))
            totals.append(n)
        assert np.mean(totals) == pytest.approx(audit['expected_count'], abs=5*np.sqrt(audit['expected_count']/len(totals)))
        assert audit['bank_size'] == 2


def normalized(frame):
    return frame.drop(columns='seconds').sort_values(['mu_test', 'case', 'toy_id']).reset_index(drop=True)


def test_four_cases_resume_extension_and_order_independence(prepared, tmp_path):
    kwargs = dict(grid_size=17, progress=lambda _: None)
    first = run_own_model_toys(prepared, tmp_path/'extend', n_toys=2, **kwargs)
    assert len(first['results']) == 16
    extended = run_own_model_toys(prepared, tmp_path/'extend', n_toys=4, **kwargs)
    assert len(extended['results']) == 32
    assert extended['results'].valid.all()
    assert set(extended['results'].case) == set(CASES)
    prepared['mu_values'] = (1.4, 0.)
    direct = run_own_model_toys(prepared, tmp_path/'direct', n_toys=4, **kwargs)
    pd.testing.assert_frame_equal(normalized(extended['results']), normalized(direct['results']), check_dtype=False)
    resumed = run_own_model_toys(prepared, tmp_path/'extend', n_toys=2, **kwargs)
    pd.testing.assert_frame_equal(normalized(first['results']), normalized(resumed['results']), check_dtype=False)
    # Reducing the requested output never deletes the longer checkpoint.
    assert len(pd.read_csv(next((tmp_path/'extend').glob('toys_*.csv')))) == 4
    prepared['configuration']['changed_normalization'] = True
    with pytest.raises(ValueError, match='changed'):
        run_own_model_toys(prepared, tmp_path/'extend', n_toys=4, **kwargs)


def test_failure_is_recorded_without_redraw_and_interruption_is_saved(prepared, tmp_path):
    calls = []
    original = prepared['analytic_draw']
    def draw(mu, rng):
        calls.append(mu)
        if len(calls) == 2:
            raise RuntimeError('deliberate failed experiment')
        return original(mu, rng)
    prepared['analytic_draw'] = draw
    def interrupt(message):
        if '2/3;' in message:
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_own_model_toys(prepared, tmp_path, n_toys=3, grid_size=9,
                           checkpoint_every=5, progress_every=2, progress=interrupt)
    saved = pd.read_csv(next(tmp_path.glob('toys_*.csv')))
    assert len(saved) == 2
    assert not saved.iloc[1].valid
    result = run_own_model_toys(prepared, tmp_path, n_toys=3, grid_size=9, progress=lambda _: None)
    assert len(calls) == 6  # Exactly three per analytic hypothesis, failed ID retained.
    assert result['summary'].failed.sum() == 1
    assert result['results'].loc[~result['results'].valid, 'message'].str.contains('deliberate').all()


def test_invalid_learned_source_does_not_get_clipped_or_replace_other_cases(prepared, tmp_path):
    prepared['learned_fields'][1] = 0.
    prepared['learned_fields'][0] *= 100.
    prepared['learned_rates'] = prepared['learned_fields'].mean(axis=1)
    prepared['mu_values'] = (.25,)
    prepared['templates']['moving36'][.25] = prepared['templates']['moving36'][0.]
    result = run_own_model_toys(prepared, tmp_path, n_toys=2, grid_size=9, progress=lambda _: None)
    rows = result['results']
    assert not rows.loc[rows.case == 'learned'].valid.any()
    assert rows.loc[rows.case != 'learned'].valid.all()
    assert len(rows) == 8


def test_plot_preserves_all_four_models_and_zero_mass(prepared, tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from poodemo.own_model_toy_plots import plot_own_model_toys
    result = run_own_model_toys(prepared, tmp_path/'study', n_toys=4, grid_size=9, progress=lambda _: None)
    figures = plot_own_model_toys(result, tmp_path/'plots')
    assert len(figures) == 2
    for name, fig in figures:
        assert fig.axes[0].get_yscale() == 'log'
        assert len(fig.axes[1].lines) == 4
        for line in fig.axes[2].lines:
            assert line.get_ydata()[-1] == 0
            assert line.get_xdata()[-1] >= line.get_xdata()[-2]
        assert (tmp_path/'plots'/f'{name}.png').is_file()
        plt.close(fig)


def test_notebook_has_explicit_production_budget_and_compiles(monkeypatch):
    import ast
    from pathlib import Path
    import nbformat
    nb = nbformat.read(Path(__file__).resolve().parents[1]/'notebooks/11_test_statistic_toys.ipynb', as_version=4)
    # Colab may attach nonstandard metadata to stream outputs. Validate the
    # runnable source without rewriting the user's committed results.
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []; cell.execution_count = None
    nbformat.validate(nb)
    code = {cell.id: cell.source for cell in nb.cells if cell.cell_type == 'code'}
    assert 'N_TOYS = 10_000' in code['settings']
    assert 'MOVING_BIN_COUNTS = (36, 72, 108, 144)' in code['settings']
    assert 'MU_VALUES = (0.0, 1.4)' in code['settings']
    assert '250_000' not in code['model-source']
    assert 'run_toy_study(' not in '\n'.join(code.values())
    for cell in nb.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    # Committed Colab results are valid; only newly generated cells must be clean.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from build_colab_11 import toy_cells
    for cell in toy_cells():
        if cell.cell_type == 'code':
            assert cell.outputs == [] and cell.execution_count is None


@pytest.mark.parametrize("signal_reference", [False, True])
def test_preparation_reuses_cached_ratios_and_merges_the_original_grid(tmp_path, monkeypatch, signal_reference):
    from poodemo import colab_discriminators as cd, colab_mc as cm, toy_sources
    target = 20
    rng = np.random.default_rng(9)
    rates = np.array([2., 9., 7., 3., 4., 2.])
    manifests = {}
    for source in ('S', 'SBI', 'B', 'NI', 'NI_up', 'NI_down'):
        ratio_items = {}
        for name in ('SBI', 'B', 'NI'):
            path = tmp_path/f'{source}_{name}.npy'
            ratio_items[name] = dict(file=path.name, sha256=_array(path, np.exp(rng.normal(0, .01, target))))
        manifests[source] = dict(chunks=[dict(count=target, ratios=ratio_items)])
    samples = dict(directory=tmp_path, target=target, manifests=manifests,
        fingerprint='fixed-fixture', rates=rates, normalizers=np.ones(4))
    run = SimpleNamespace(config=dict(reference_ratio_power=100., reference_ratio_eta_floor=0.,
        mc_zero_max_bin=.01, mc_zero_max_total=.05), model=SimpleNamespace(to_dict=lambda: {'model': 'test'}))
    quad = dict(nominal=np.ones((4, 3)), weights=np.ones(3))
    monkeypatch.setattr(cd, 'cached_analytic_quadrature', lambda run: quad)
    monkeypatch.setattr(toy_sources, 'load_frozen_selector', lambda run: (object(), .5, {'frozen': True}))
    prepare_models = prepare_own_model_toys
    if signal_reference:
        from poodemo.signal_reference_toys import prepare_signal_model_toys
        prepare_models = prepare_signal_model_toys
        run.config.update(reference_ratio_denominator='S', reference_ratio_power=2.)
    result = prepare_models(run, samples, progress=lambda _: None)
    assert result['configuration'].get('reference_ratio_denominator', 'physical_1') == ('S' if signal_reference else 'physical_1')
    assert result['templates']['fixed9'].fields.shape == (4, 729)
    np.testing.assert_allclose(result['templates']['fixed9'].fields.sum(axis=1), rates[:4])
    np.testing.assert_allclose(result['learned_fields'].mean(axis=1), rates[:4])
    fine = cd.fixed_counts(samples, progress=lambda _: None)
    np.testing.assert_array_equal(result['configuration']['fixed_edges'], fine['edges'][::4])
    for mu in (0., 1.4):
        assert result['templates']['moving36'][mu].fields.shape == (4, 36)
    # All source data and NN modules are unavailable; cached ratios suffice.
    again = prepare_models(run, samples, progress=lambda _: None)
    assert again['configuration'] == result['configuration']
    assert len(list(tmp_path.glob('*.npy'))) == 18
    from poodemo.own_model_resolution import prepare_moving_resolution
    resolution = prepare_moving_resolution(run, samples, result, progress=lambda _: None)
    assert resolution['templates'][36] is result['templates']['moving36']
    for n in (72, 108, 144):
        for mu in (0., 1.4):
            fields = resolution['templates'][n][mu].fields
            assert fields.shape == (4, n)
            np.testing.assert_allclose(fields.sum(axis=1), rates[:4])
    assert len(list(tmp_path.glob('*.npy'))) == 18
