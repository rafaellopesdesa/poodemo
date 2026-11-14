"""Binning extension preserves original toys, independent streams and global fits."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poodemo.own_model_toys import PoissonTemplate, run_own_model_toys
from poodemo.own_model_resolution import (prepare_moving_resolution, run_moving_resolution,
                                        BIN_COUNTS, STREAMS)
from test_own_model_toys import prepared


@pytest.fixture
def resolution(prepared):
    templates = {36: prepared['templates']['moving36']}
    for n in BIN_COUNTS[1:]:
        fields = np.repeat(prepared['templates']['moving36'][0.].fields, n//2, axis=1)/(n//2)
        templates[n] = {mu: PoissonTemplate(fields) for mu in prepared['mu_values']}
    return dict(baseline=prepared, bin_counts=BIN_COUNTS, templates=templates)


def normalized(frame):
    return frame.drop(columns='seconds').sort_values(['mu_test', 'case', 'toy_id']).reset_index(drop=True)


def test_seven_cases_extend_and_reuse_baseline_rows(resolution, tmp_path):
    kwargs = dict(grid_size=9, progress=lambda _: None)
    baseline = run_own_model_toys(resolution['baseline'], tmp_path/'baseline', n_toys=2, **kwargs)
    first = run_moving_resolution(resolution, baseline, tmp_path/'resolution', n_toys=2, **kwargs)
    assert len(first['results']) == 7*2*2
    assert set(first['results'].case) == {'analytic', 'learned', 'fixed9', 'moving36', 'moving72', 'moving108', 'moving144'}
    assert not first['results'].duplicated(['mu_test', 'case', 'toy_id']).any()
    kept = first['results'].loc[first['results'].case.isin(baseline['results'].case.unique())]
    pd.testing.assert_frame_equal(normalized(kept), normalized(baseline['results']), check_dtype=False)
    assert first['summary'].attempted.eq(2).all()
    assert first['results'].valid.all()
    assert 'fraction_other_branch' in first['summary']
    baseline4 = run_own_model_toys(resolution['baseline'], tmp_path/'baseline', n_toys=4, **kwargs)
    extended = run_moving_resolution(resolution, baseline4, tmp_path/'resolution', n_toys=4, **kwargs)
    pd.testing.assert_frame_equal(normalized(first['results']),
        normalized(extended['results'].loc[extended['results'].toy_id < 2]), check_dtype=False)
    assert len(extended['results']) == 56
    # Reorder/subset the requested binnings; previously saved draws stay identical.
    resolution['bin_counts'] = (144, 36, 72)
    direct = run_moving_resolution(resolution, baseline4, tmp_path/'direct', n_toys=4, **kwargs)
    expected = extended['results'].loc[extended['results'].case != 'moving108']
    pd.testing.assert_frame_equal(normalized(direct['results']), normalized(expected), check_dtype=False)
    assert len(set(STREAMS.values()) | {0, 1, 2, 3}) == 7


def test_incompatible_or_incomplete_baseline_rejected(resolution, tmp_path):
    kwargs = dict(grid_size=9, progress=lambda _: None)
    baseline = run_own_model_toys(resolution['baseline'], tmp_path/'baseline', n_toys=2, **kwargs)
    with pytest.raises(ValueError, match='Extend the four'):
        run_moving_resolution(resolution, baseline, tmp_path/'resolution', n_toys=3, **kwargs)
    with pytest.raises(ValueError, match='settings differ'):
        run_moving_resolution(resolution, baseline, tmp_path/'resolution', n_toys=2, seed=5, **kwargs)
    baseline['results'] = pd.concat([baseline['results'], baseline['results'].iloc[:1]])
    with pytest.raises(ValueError, match='Duplicate'):
        run_moving_resolution(resolution, baseline, tmp_path/'resolution', n_toys=2, **kwargs)


def test_checkpoint_rejects_changed_template_but_allows_budget_change(resolution, tmp_path):
    kwargs = dict(grid_size=9, progress=lambda _: None)
    baseline = run_own_model_toys(resolution['baseline'], tmp_path/'baseline', n_toys=2, **kwargs)
    run_moving_resolution(resolution, baseline, tmp_path/'resolution', n_toys=2, **kwargs)
    resolution['templates'][72][0.].fields[0, 0] += .0001
    with pytest.raises(ValueError, match='settings/templates changed'):
        run_moving_resolution(resolution, baseline, tmp_path/'resolution', n_toys=2, **kwargs)


def test_interruption_saves_pending_rows_and_resumes(resolution, tmp_path):
    kwargs = dict(grid_size=9, progress=lambda _: None)
    baseline = run_own_model_toys(resolution['baseline'], tmp_path/'baseline', n_toys=3, **kwargs)
    def interrupt(message):
        if '2/3;' in message:
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_moving_resolution(resolution, baseline, tmp_path/'resolution', n_toys=3, grid_size=9,
            checkpoint_every=100, progress_every=2, progress=interrupt)
    path = next((tmp_path/'resolution'/'bins_72').glob('toys_*.csv'))
    saved = pd.read_csv(path)
    assert len(saved) == 2
    resumed = run_moving_resolution(resolution, baseline, tmp_path/'resolution', n_toys=3, **kwargs)
    same = resumed['results'].loc[(resumed['results'].case == 'moving72') &
        (resumed['results'].mu_test == 0) & (resumed['results'].toy_id < 2)]
    pd.testing.assert_frame_equal(normalized(saved), normalized(same), check_dtype=False)


def test_invalid_fine_generator_retains_failures(resolution, tmp_path):
    kwargs = dict(grid_size=9, progress=lambda _: None)
    baseline = run_own_model_toys(resolution['baseline'], tmp_path/'baseline', n_toys=2, **kwargs)
    resolution['templates'][72][1.4] = PoissonTemplate(np.array([[1.], [0.], [10.], [0.]]))
    study = run_moving_resolution(resolution, baseline, tmp_path/'resolution', n_toys=2, **kwargs)
    invalid = study['results'].loc[(study['results'].case == 'moving72') & (study['results'].mu_test == 1.4)]
    assert len(invalid) == 2 and not invalid.valid.any()
    assert invalid.message.str.contains('Invalid binned generator').all()
    assert study['summary'].failed.sum() == 2
    assert len(study['results']) == 28


def test_resolution_plots_are_logarithmic_with_both_references(resolution, tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from poodemo.own_model_toy_plots import plot_moving_resolution
    kwargs = dict(grid_size=9, progress=lambda _: None)
    baseline = run_own_model_toys(resolution['baseline'], tmp_path/'baseline', n_toys=4, **kwargs)
    study = run_moving_resolution(resolution, baseline, tmp_path/'resolution', n_toys=4, **kwargs)
    plots = plot_moving_resolution(study, tmp_path/'plots')
    assert len(plots) == 4
    for name, fig in plots:
        assert fig.axes[0].get_yscale() == 'log'
        assert fig.axes[1].get_yscale() == 'linear'
        legend = [text.get_text() for text in fig.axes[0].get_legend().texts]
        assert len(legend) == 6
        assert 'Analytical unbinned' in legend
        assert 'Learned unbinned (S bank)' in legend
        assert (tmp_path/'plots'/f'{name}.pdf').is_file()
        if '_q_' in name:
            assert fig.axes[2].get_yscale() == 'log'
            assert fig.axes[0].get_xlim()[0] == 0  # Zero mass retained, no log-x offset.
        plt.close(fig)


def test_prepare_reuses_36_and_fills_only_cached_extra_resolutions(monkeypatch):
    from types import SimpleNamespace
    from poodemo import own_model_resolution as module
    from poodemo.mc_yields import YieldPolicy
    policy = YieldPolicy(.01, .05)
    rates = np.array([2., 9., 7., 3., 4., 2.])
    config = dict(selected_per_source=10_000_000, sample_fingerprint='frozen', reference_power=100.,
        eta_floor=0., yield_policy=policy.metadata(), rates=rates[:4].tolist(), observable_normalizers=[1.]*4)
    base36 = {mu: PoissonTemplate(np.ones((4, 36))) for mu in (0., 1.4)}
    baseline = dict(configuration=config, exposure=1., mu_values=(0., 1.4), templates={'moving36': base36})
    run = SimpleNamespace(config=dict(reference_ratio_power=100., reference_ratio_eta_floor=0.,
        mc_zero_max_bin=.01, mc_zero_max_total=.05))
    samples = dict(target=10_000_000, fingerprint='frozen', rates=rates, normalizers=np.ones(4))
    calls = []
    def fill(run, samples, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(anchors=lambda mu, n: np.ones((3, 4, n)))
    monkeypatch.setattr(module, 'fill_colab_histograms', fill)
    result = prepare_moving_resolution(run, samples, baseline, progress=lambda _: None)
    assert result['templates'][36] is base36
    assert calls[0]['bin_counts'] == [72, 108, 144]
    assert calls[0]['prefixes'] == [10_000_000]
    assert set(result['templates']) == set(BIN_COUNTS)
    samples['target'] += 1
    with pytest.raises(ValueError, match='same <=10M'):
        prepare_moving_resolution(run, samples, baseline)


def test_current_builder_matches_notebook(monkeypatch):
    import nbformat
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    from build_colab_11 import toy_cells
    actual = nbformat.read(root/'notebooks/11_test_statistic_toys.ipynb', as_version=4)
    assert [cell.source for cell in toy_cells()] == [cell.source for cell in actual.cells]
    settings = next(cell.source for cell in actual.cells if cell.id == 'settings')
    assert 'N_TOYS = 10_000' in settings
    assert 'CHECKPOINT_EVERY = 100' in settings
    assert 'MOVING_BIN_COUNTS = (36, 72, 108, 144)' in settings


def test_published_v1_backend_hashes_still_resume_completed_toys():
    # Hashes recorded in the completed November 14 four-case campaign.
    from poodemo.data import file_digest
    root = Path(__file__).resolve().parents[1]/"poodemo"
    published = {'own_model_toys': '7a185085e0e117c196c58f8f933c7c83c758add048f265241c4a0d3a4aa5ad1e', 'toy_likelihood': '18445d804f0fa8315192006bac4a2b76d5f182e376c49a7d62fc7a4fff2b8bd5', 'toy_sources': '3eec1834d5ce577cc00a23e86e55165495bb87aa1e25de63f068d76eb75e3166', 'mc_yields': '65d22f09a714971a8bf61e1bffc96df18f034e61b6a21dd69a589eb782fa76f3', 'physics': 'f13359c8903672b0615684beb19efedfedd9a6eeb22ae04e04b3d100f9ffd5c1'}
    for name, digest in published.items():
        assert file_digest(root/(name+".py")) == digest
