"""Direct Asimov profile and matched compression controls for notebook 09."""
import json

import numpy as np
import pandas as pd
import pytest

from test_ratio_observable import selected_quadrature
from poodemo import pipeline
from poodemo.reference_floor import (
    FloorStudySettings, direct_asimov_fits, profile_nuisance,
    run_reference_floor_study, plot_reference_floor_study)
from poodemo.reference_splines import _fit


def test_profile_search_resolves_multiple_basins_and_endpoints():
    settings = FloorStudySettings(alpha_grid_points=25)
    objective = lambda a: ((a+.6)*(a-1.4))**2+.03*(a-1.4)**2
    result = profile_nuisance(objective, [-3., 3.], settings)
    assert result['valid'] and result['basins'] >= 2
    assert result['alpha'] == pytest.approx(1.4, abs=1e-6)
    assert result['nll'] < 1e-12
    boundary = profile_nuisance(lambda a: (a-4.)**2, [-3., 3.], settings)
    assert boundary['valid'] and boundary['boundary'] and boundary['alpha'] == 3.
    failed = profile_nuisance(lambda a: np.inf, [-3., 3.], settings)
    assert not failed['valid'] and np.isnan(failed['nll'])


def test_profile_failure_is_reported_even_if_another_basin_succeeds(monkeypatch):
    from types import SimpleNamespace
    from poodemo import reference_floor
    monkeypatch.setattr(reference_floor, 'minimize_scalar', lambda *args, **kwargs:
                        SimpleNamespace(success=False, fun=0., x=0.))
    result = profile_nuisance(lambda a: a*a, [-3., 3.], FloorStudySettings())
    assert result['failed_searches'] > 0 and not result['valid']


def test_direct_certificate_and_profile_agree_with_full_fit(selected_quadrature):
    run, quad = selected_quadrature
    run = run.with_config(reference_ratio_eta_floor=.2, reference_ratio_power=100.,
                          asimov_mu=1., mu_fit_bounds=[.001, 2.], nuisance_bounds=[-3., 3.])
    eta = .105
    z = pipeline.observable_values(run, quad, eta, 'reference_ratio')
    model = pipeline._template_likelihood(quad, systematics=True).binned(
        z, pipeline.observable_bin_edges(12, 'reference_ratio'), 'before', eta)
    results = direct_asimov_fits(run, model, eta, FloorStudySettings())
    for row in results:
        reference = _fit(run, model, eta, row['systematics'])
        assert row['valid'] and row['truth_certified'] and reference['valid']
        assert row['q'] == pytest.approx(reference['q'], abs=1e-5)
    assert results[1]['q'] <= results[0]['q']+1e-10
    # A mismatched Asimov truth cannot acquire a certificate or a valid q.
    wrong = direct_asimov_fits(run.with_config(asimov_mu=.5), model, eta, FloorStudySettings())
    assert not any(row['truth_certified'] or row['valid'] for row in wrong)


def test_complete_direct_controls_preserve_run_and_match_above_floor(tmp_path, selected_quadrature, monkeypatch):
    original, quad = selected_quadrature
    run = original.with_config(reference_ratio_eta_floor=.2, reference_ratio_power=100.,
        asimov_mu=1., mu_fit_bounds=[.001, 2.], nuisance_bounds=[-3., 3.],
        mu_min=.1, mu_max=.3, mu_points=3)
    run.path('results').mkdir(parents=True, exist_ok=True)
    reference = pd.DataFrame([dict(mu=np.nextafter(mu, np.inf), systematics=s, model='analytic unbinned', q=0., valid=True)
                             for mu in pipeline._positive_study_grid(run) for s in (False, True)])
    reference.to_csv(run.path('results', 'unbinned_scans.csv'), index=False)
    sentinel = run.path('results', 'reference_ratio_binning_config.json')
    sentinel.write_text('baseline untouched')
    result = run_reference_floor_study(run, 4, quadrature=quad,
        settings=FloorStudySettings(resolution_multiplier=2, alpha_grid_points=7), progress=lambda _: None)
    assert run.config['reference_ratio_eta_floor'] == .2
    assert sentinel.read_text() == 'baseline untouched'
    assert result['metadata']['all_fits_valid']
    assert len(result['scans']) == len(pipeline._positive_study_grid(run))*2*2*2*2
    floor = result['floor_comparison']
    np.testing.assert_array_equal(floor.loc[floor.eta >= .2, 'delta_q_no_floor_minus_floor'], 0.)
    morph = result['morph_comparison']
    np.testing.assert_allclose(morph.loc[~morph.systematics, 'delta_q_after_minus_before'], 0., atol=1e-9)
    assert np.max(abs(floor.loc[floor.eta < .2, 'delta_q_no_floor_minus_floor'])) > 1e-5
    scans = result['scans']
    fixed = scans.loc[~scans.systematics].set_index(['eta', 'floor', 'n_bins', 'morph_order'])
    profiled = scans.loc[scans.systematics].set_index(['eta', 'floor', 'n_bins', 'morph_order'])
    assert np.all(profiled.q <= fixed.q+1e-8)
    assert result['reference_comparison'].q_unbinned.notna().all()
    for name in ('scans', 'summary', 'floor_comparison', 'morph_comparison', 'reference_comparison'):
        assert len(pd.read_csv(result['output']/f'{name}.csv')) == len(result[name])
    assert json.loads((result['output']/'metadata.json').read_text())['status'] == 'complete'
    import matplotlib
    matplotlib.use('Agg')
    # Colab enables NumExpr; the `floor` column must not be sent to query().
    original_query = pd.DataFrame.query
    def checked_query(frame, expression, *args, **kwargs):
        assert 'floor' not in expression
        kwargs['level'] = kwargs.get('level', 0) + 1
        return original_query(frame, expression, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, 'query', checked_query)
    figures = plot_reference_floor_study(run, result)
    assert len(figures) == 3
    assert len(list(run.path('plots', '09_floor_study').glob('*.png'))) == 3
    import matplotlib.pyplot as plt
    plt.close('all')
    # Execute the notebook recovery cell with no in-memory study. It must read
    # the saved evaluation and plot without calling the expensive fit workflow.
    from pathlib import Path
    from poodemo import reference_floor
    notebook = json.loads((Path(__file__).resolve().parents[1] /
        'notebooks/09_reference_ratio_spline_templates.ipynb').read_text())
    cell = next(c for c in notebook['cells'] if c['id'] == 'floor-plots')
    seen = []
    def check_reloaded(selected_run, study, **kwargs):
        assert selected_run is run
        pd.testing.assert_frame_equal(study['scans'], result['scans'], check_dtype=False)
        assert study['metadata']['status'] == 'complete'
        assert not pd.get_option('compute.use_numexpr')
        seen.append(study['output'])
        return []
    monkeypatch.setattr(reference_floor, 'plot_reference_floor_study', check_reloaded)
    option = pd.get_option('compute.use_numexpr')
    ns = dict(run=run, SCAN_XLIM=(.02, 1.15), SCAN_YLIM=(0., 8.),
              SECONDARY_XLIM=(.02, .3), display=lambda _: None, plt=plt)
    exec(''.join(cell['source']), ns)
    assert seen == [result['output']]
    assert pd.get_option('compute.use_numexpr') == option
