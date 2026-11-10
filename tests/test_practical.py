"""No density oracle in learned maps/MC templates; bounded 08--11 regressions."""
from types import SimpleNamespace
import json
import sys

import numpy as np
import pandas as pd
import pytest

from poodemo import pipeline
from poodemo.data import Run, PROCESSES, SAMPLES, save_array, save_json
from poodemo.practical import (LearnedMap, SimulationHistogram, prepare_simulation_bank,
                              use_practical_run, refresh_analytic_reference)
from poodemo.inference import component_coefficients


@pytest.fixture
def practical_run(tmp_path, monkeypatch):
    class Simulator:
        yields = dict(S=2., SBI=8., B=4., NI=10.)
        def component_yield(self, process, alpha_ni=0.):
            return self.yields[process]*np.exp(.1*alpha_ni)
        def to_dict(self):
            return dict(test="simulator samples and generated yields only")
        def component_densities(self, x):
            pytest.fail("An analytical PDF entered the practical map or MC templates")
        component_density = component_densities
    base = Run(tmp_path, dict(seed=19, n_per_sample=1000, ni_sample_multiplier=1,
        quadrature_per_process=80, asimov_mu=1., mu_min=.2, mu_max=1.4, mu_points=3,
        mu_fit_bounds=[0., 2.], nuisance_bounds=[-2., 2.], bin_counts=[4],
        reference_ratio_power=10., reference_ratio_eta_floor=0.), Simulator())
    base.filter_manifest = dict(identity="filtered-model-identity")
    save_json(base.path("selected", "selection_state.json"), {"selection_fingerprint": "fixture"})
    x = np.column_stack([np.linspace(-1, 1, 120), np.zeros((120, 2))]).astype('float32')
    for sample in SAMPLES:
        save_array(base.path("selected", f"{sample}_integration.npy"), x)
    calls = []
    def load_predictor(path):
        calls.append(str(path))
        slope = {"SBI_over_S": .15, "B_over_S": -.1, "NI_over_S": .2}[path.name]
        return SimpleNamespace(predict_ratio=lambda points: np.exp(slope*points[:, 0]))
    monkeypatch.setitem(sys.modules, 'poodemo.training', SimpleNamespace(load_predictor=load_predictor))
    return use_practical_run(base), calls


def test_mc_weights_no_oracle_cache_and_frozen_normalizers(practical_run):
    run, calls = practical_run
    q = prepare_simulation_bank(run)
    expected_rates = np.array([2., 8., 4., 10.])*.8
    np.testing.assert_allclose(q['nominal'].sum(axis=1), expected_rates)
    np.testing.assert_allclose(q['down'][0].sum(axis=1), expected_rates*[1, 1, 1, np.exp(-.1)])
    np.testing.assert_allclose(q['up'][0].sum(axis=1), expected_rates*[1, 1, 1, np.exp(.1)])
    assert len(calls) == 3 and q['x'].shape == (480, 3)
    frozen = q['learned_map']
    original = frozen.normalizers.copy()
    coords = q['x'][[10, 20, 40]]
    fields = frozen.evaluate(coords)
    z = frozen.coordinate(fields, .3)
    np.testing.assert_array_equal(frozen.coordinate(frozen.evaluate(coords[[1, 1, 0]]), .3), z[[1, 1, 0]])
    np.testing.assert_array_equal(original, frozen.normalizers)
    np.testing.assert_array_equal(frozen.coordinate(fields, 1.), .5)
    # Fresh views reuse hashed predictions, even if prediction becomes unavailable.
    other = use_practical_run(run.base_run)
    q2 = prepare_simulation_bank(other)
    assert len(calls) == 3
    np.testing.assert_array_equal(q2['learned_fields'], q['learned_fields'])
    assert run.with_config(reference_ratio_power=23.).path('models', 'ratios') == run.path('models', 'ratios')
    assert run.path('results') != run.base_run.path('results')


def test_ecdf_matches_direct_mc_fills_and_exact_collapse(practical_run):
    from poodemo.reference_splines import TemplateBank, CumulativeTemplates, rescaled_log_ratio
    run, _ = practical_run
    q = prepare_simulation_bank(run)
    bank = TemplateBank(run, q, pipeline.observable_bin_edges(12, 'reference_ratio'))
    anchors = np.array([0., .2, .8, .999, 1., 1.001, 1.4, 2.])
    model = CumulativeTemplates(bank, anchors)
    for eta in anchors:
        np.testing.assert_allclose(model(eta), bank.exact(eta), atol=2e-14)
        np.testing.assert_allclose(model(eta).sum(axis=-1), bank.totals, atol=2e-14)
    np.testing.assert_allclose(rescaled_log_ratio(q, np.nextafter(1., 2.)), rescaled_log_ratio(q, 1.), atol=1e-14)
    assert np.count_nonzero(bank.exact(1.).sum(axis=(0, 1))) == 1
    assert bank.model(bank.exact(.2), bank.counts(.2)).nll(1., [0.]) == pytest.approx(0., abs=1e-15)
    # The map and templates are separate: changing NN fields changes assignments,
    # but never changes the component rates or creates analytical weights.
    before = q['nominal'].copy()
    changed = {**q, 'learned_fields': q['learned_fields'][:, ::-1]}
    assert not np.array_equal(pipeline.observable_values(run, q, .2, 'reference_ratio'),
                              pipeline.observable_values(run, changed, .2, 'reference_ratio'))
    np.testing.assert_array_equal(before, q['nominal'])


def test_sparse_ni_support_has_no_floor_and_invalid_physics_stays_invalid():
    nominal = np.array([[1., 0., 0.], [2., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    down, up = nominal.copy(), nominal.copy()
    down[3] = [0., 1., 0.]
    up[3] = [1., 1., 0.]
    counts = component_coefficients(1.) @ nominal
    model = SimulationHistogram(nominal, down[None], up[None], np.ones(3), counts)
    assert model.incomplete_support_bins == 2
    for alpha, anchor in ((-1., down), (0., nominal), (1., up)):
        np.testing.assert_allclose(model.components([alpha]), anchor)
    for alpha in (-3., -.4, .4, 3.):
        assert np.all(model.components([alpha]) >= 0)
        assert np.all(model.components([alpha])[:, 2] == 0)
    assert model.nll(1., [0.]) == 0.
    from poodemo.practical import local_information
    assert np.isnan(local_information(model, 1.)['matrix'][1, 1])
    bad = nominal.copy()
    bad[0, 0] = 100.
    invalid = SimulationHistogram(bad, np.empty((0, 4, 3)), np.empty((0, 4, 3)), np.ones(3), counts)
    assert np.isinf(invalid.nll(.25))


def test_small_end_to_end_08_09_10(practical_run, monkeypatch):
    from poodemo.reference_splines import SplineStudySettings, run_reference_spline_comparison
    from poodemo.reference_floor import FloorStudySettings, run_reference_floor_study, plot_reference_floor_study
    from poodemo.discriminator_study import run_discriminator_study, plot_discriminator_study
    from poodemo.diagnostics import run_estimator_study
    run, _ = practical_run
    q = prepare_simulation_bank(run)
    n = 40
    fields = np.broadcast_to(q['nominal'].sum(axis=1)[:, None]/n, (4, n)).copy()
    aq = dict(x=np.zeros((n, 3)), weights=np.ones(n), nominal=fields,
              down=fields[None].copy(), up=fields[None].copy(), truth=fields[1]+fields[3])
    aq['down'][0, 3] *= np.exp(-.1)
    aq['up'][0, 3] *= np.exp(.1)
    monkeypatch.setattr(pipeline, 'prepare_quadrature', lambda run: aq)
    refresh_analytic_reference(run)
    binned = pipeline.run_binning_study(run, observable='reference_ratio')
    assert binned['scans'].valid.all()
    estimated = run_estimator_study(run, observable='reference_ratio', n_bins=4)
    assert estimated.fit_valid.all()
    study = run_reference_spline_comparison(run, n_bins=4, progress=lambda _: None,
        settings=SplineStudySettings(initial_anchors=3, focus_anchors=3, max_anchors=15,
            max_rounds=1, likelihood_refine_points=1, validation_intervals=2, validation_fit_points=1,
            surface_points=1, surface_mu_points=3, surface_alpha_points=3))
    assert study['metadata']['template_source'] == 'simulation'
    assert study['morph_comparison'].empty
    assert set(study['unbinned'].model) == {'analytic unbinned'}
    floor = run_reference_floor_study(run, n_bins=4, quadrature=q,
        settings=FloorStudySettings(resolution_multiplier=2, alpha_grid_points=5), progress=lambda _: None)
    assert set(floor['scans'].morph_order) == {'after'}
    assert floor['scans'].valid.all()
    assert floor['morph_comparison'].empty
    figures = plot_reference_floor_study(run, floor)
    discriminators = run_discriminator_study(run, n_bins=4, reference_power=10., ratio_source='learned', scan_grid=[.3, 1., 1.4])
    assert discriminators['config']['likelihood_source'] == 'simulation'
    assert discriminators['config']['asimov_source'] == 'same MC templates'
    assert discriminators['scans'].valid.all()
    assert 'incomplete_NI_support_bins' in discriminators['occupancy']
    figures += list(plot_discriminator_study(discriminators).values())
    import matplotlib.pyplot as plt
    for fig in figures:
        plt.close(fig)


def test_practical_toys_use_learned_map_and_mc_templates(practical_run, monkeypatch):
    from poodemo import toy_study, toy_sources
    run, _ = practical_run
    q = prepare_simulation_bank(run)
    rates = q['nominal'].sum(axis=1)
    aq = dict(x=q['x'][:80], weights=np.full(80, 1./80), nominal=np.broadcast_to(rates[:, None], (4, 80)))
    monkeypatch.setattr(toy_study, 'prepare_quadrature', lambda _: aq)
    monkeypatch.setattr(toy_sources, 'load_frozen_selector', lambda _: (None, None, {'fixture': True}))
    monkeypatch.setattr(toy_sources, 'make_frozen_workspace', lambda *a: pytest.fail('Legacy oracle normalization used'))
    monkeypatch.setattr(run.model, 'component_densities', lambda x: np.broadcast_to(rates, (len(x), 4)))
    def simulate(run, mu, rng, exposure, **kwargs):
        n = rng.poisson(exposure*(component_coefficients(mu)@rates))
        return q['x'][rng.integers(0, 80, n)]
    monkeypatch.setattr(toy_sources, 'sample_simulator_poisson', simulate)
    first = toy_study.run_toy_study(run, n_toys=2, mu_values=(0., 1.4), n_bins=4,
        reference_power=10., grid_size=9, output_tag='practical_test', progress_every=1)
    assert len(first['results']) == 20
    assert first['results'].valid.all()
    config = first['metadata']['configuration']
    assert config['version'] == 3 and config['template_method'] == 'direct_simulation'
    second = toy_study.run_toy_study(run, n_toys=3, mu_values=(0., 1.4), n_bins=4,
        reference_power=10., grid_size=9, output_tag='practical_test', progress_every=1)
    pd.testing.assert_frame_equal(first['results'], second['results'].loc[second['results'].toy_id < 2].reset_index(drop=True), check_dtype=False)


def test_reuse_filter_does_not_prepare_analytical_quadrature(tmp_path, monkeypatch):
    from poodemo import member_filter
    from poodemo.practical import reuse_filtered_ensembles
    from poodemo.data import file_digest
    run = Run(tmp_path, {}, None)
    inputs = {'models': 'checksummed', 'policy': 'checksummed'}
    monkeypatch.setattr(member_filter, '_inventory', lambda *args: ({}, dict(inputs)))
    monkeypatch.setattr(member_filter, 'prepare_quadrature', lambda *args: pytest.fail('Oracle screening was invoked'))
    identity = member_filter._digest({**inputs, 'device': 'auto'})
    directory = tmp_path/'evaluations'/'filtered'/identity[:16]
    artifact = directory/'models'/'retained.json'
    save_json(artifact, {'retained': [1, 3]})
    save_json(directory/'results/ensemble_filter/complete.json',
              {'identity': identity, 'artifacts': {'models/retained.json': file_digest(artifact)}})
    save_json(directory/'results/ensemble_filter/manifest.json', {'identity': identity, 'status': 'passed'})
    selected = reuse_filtered_ensembles(run)
    assert selected.filter_manifest['identity'] == identity
    save_json(artifact, {'retained': [1, 2, 3]})
    with pytest.raises(RuntimeError, match='matching completed notebook-03 filter'):
        reuse_filtered_ensembles(run)
