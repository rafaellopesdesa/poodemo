"""Own-model score toys, reference reuse, deterministic resumption and notebook cells."""
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from poodemo.data import file_digest
from poodemo.fixed_score_toys import (prepare_fixed_score_toys, run_fixed_score_toys,
                                     REFERENCE_BACKENDS)
from poodemo.own_model_toys import PoissonTemplate, run_own_model_toys
from poodemo.toy_likelihood import StatOnlyLikelihood
from test_colab_mc import setup, prepare


@pytest.fixture
def prepared():
    fields = np.array([[2., 1.], [4., 5.], [3., 4.], [2., 1.]])
    rates = fields.sum(axis=1)
    template = PoissonTemplate(fields)
    def draw(mu, rng):
        counts = rng.poisson(template.means(mu)[0])
        return StatOnlyLikelihood(fields, rates, counts), int(counts.sum())
    root = Path(__file__).resolve().parents[1]/'poodemo'
    hashes = {name: file_digest(root/(name+'.py')) for name in REFERENCE_BACKENDS}
    return dict(configuration=dict(version=1, reference=dict(exposure=1., rates=rates.tolist(),
                    sample_fingerprint='fixture', analytic_rates=rates.tolist()), reference_backend_sha256=hashes),
        templates={'score36': template, 'score144': PoissonTemplate(fields)},
        labels={'score36': 'Fixed score: 36 bins', 'score144': 'Fixed score: 144 bins',
                'analytic': 'Analytical unbinned', 'learned': 'Learned unbinned (S bank)'},
        streams={'score36': 130036, 'score144': 130144, 'analytic': 0, 'learned': 1},
        mu_values=(0., 1.4), exposure=1., learned_fields=2*fields, learned_rates=rates,
        learned_intervals=np.empty((0, 2)), analytic_rates=rates, analytic_draw=draw)


def legacy(prepared):
    return dict(prepared, configuration=dict(version=1, **prepared['configuration']['reference'],
        backend_sha256=prepared['configuration']['reference_backend_sha256']),
        templates={'moving36': {mu: prepared['templates']['score36'] for mu in prepared['mu_values']},
                   'fixed9': prepared['templates']['score144']})


def normalized(frame):
    return frame.drop(columns=['seconds', 'origin']).sort_values(['mu_test','case','toy_id']).reset_index(drop=True)


def test_prepare_cell_reuses_exact_12_partition_and_never_generates(setup, monkeypatch):
    from poodemo import colab_discriminators
    from poodemo.fixed_score import fill_fixed_score
    run, calls = setup
    run = run.with_config(reference_ratio_denominator='S', reference_ratio_power=2.)
    samples = prepare(run)
    bank = fill_fixed_score(run, samples, bin_counts=(4, 8), progress=lambda _: None)
    before = calls.copy()
    monkeypatch.setattr(colab_discriminators, 'cached_analytic_quadrature',
                        lambda _: dict(nominal=np.ones((4, 3)), weights=np.ones(3)))
    nb = nbformat.read(Path(__file__).resolve().parents[1]/'notebooks/13_fixed_score_toys.ipynb', as_version=4)
    env = dict(run=run, samples=samples, MU_VALUES=(0., 1.4), BIN_COUNTS=(4, 8),
               SCORE_SCALE=.04, EXPOSURE=1., display=lambda *_: None, pd=pd)
    exec(next(c.source for c in nb.cells if c.id=='prepare'), env)
    result = env['prepared']
    assert calls == before
    assert result['bank']['identity'] == bank['identity']
    for n in (4, 8):
        fields = result['templates'][f'score{n}'].fields
        np.testing.assert_array_equal(fields, bank['counts'][n][:4]*samples['rates'][:4,None]/samples['target'])
        np.testing.assert_allclose(fields.sum(axis=1), samples['rates'][:4])
        means, _ = result['templates'][f'score{n}'].means(0.)
        np.testing.assert_allclose(means, fields[2]+fields[3])
    np.testing.assert_allclose(result['learned_fields'].mean(axis=1), samples['rates'][:4])
    from poodemo.signal_reference_toys import prepare_signal_model_toys
    previous = prepare_signal_model_toys(run, samples, progress=lambda _:None)
    for key, value in result['configuration']['reference'].items():
        assert previous['configuration'][key]==value, key
    for name, digest in result['configuration']['reference_backend_sha256'].items():
        assert previous['configuration']['backend_sha256'][name]==digest, name
    assert calls == before
    with pytest.raises(ValueError, match='10M'):
        prepare_fixed_score_toys(run, {**samples, 'target':10_000_001})


def test_resume_extend_and_independent_stable_streams(prepared, tmp_path):
    kwargs = dict(grid_size=17, progress=lambda _: None)
    first = run_fixed_score_toys(prepared, tmp_path/'extend', n_toys=2, **kwargs)
    result = run_fixed_score_toys(prepared, tmp_path/'extend', n_toys=4, **kwargs)
    assert len(result['results']) == 32 and result['results'].valid.all()
    prepared['mu_values'] = (1.4, 0.)
    prepared['labels'] = dict(reversed(list(prepared['labels'].items())))
    direct = run_fixed_score_toys(prepared, tmp_path/'direct', n_toys=4, **kwargs)
    pd.testing.assert_frame_equal(normalized(result['results']), normalized(direct['results']), check_dtype=False)
    reduced = run_fixed_score_toys(prepared, tmp_path/'extend', n_toys=2, **kwargs)
    pd.testing.assert_frame_equal(normalized(first['results']), normalized(reduced['results']), check_dtype=False)
    assert len(pd.read_csv(next((tmp_path/'extend').glob('toys_*.csv')))) == 4
    prepared['configuration']['score_scale'] = .1
    with pytest.raises(ValueError, match='OUTPUT_TAG'):
        run_fixed_score_toys(prepared, tmp_path/'extend', n_toys=4, **kwargs)


def test_matching_07_references_reused_including_failures_without_writes(prepared, tmp_path, monkeypatch):
    import poodemo.fixed_score_toys as module
    calls = []
    original = prepared['analytic_draw']
    def occasional_failure(mu, rng):
        calls.append(mu)
        if len(calls) == 1:
            raise RuntimeError('retained reference failure')
        return original(mu, rng)
    prepared['analytic_draw'] = occasional_failure
    old = tmp_path/'07'
    reference = run_own_model_toys(legacy(prepared), old, n_toys=3, grid_size=17, progress=lambda _:None)
    hashes = {p.name:file_digest(p) for p in old.iterdir() if p.is_file()}
    source = module._source
    def only_score(prepared, case, mu):
        assert case.startswith('score'), 'A completed reference was regenerated'
        return source(prepared, case, mu)
    monkeypatch.setattr(module, '_source', only_score)
    result = run_fixed_score_toys(prepared, tmp_path/'13', reference_directory=old,
                                   n_toys=3, grid_size=17, progress=lambda _:None)
    reused = result['results'].loc[result['results'].case.isin(['analytic','learned'])]
    expected = reference['results'].loc[reference['results'].case.isin(['analytic','learned'])]
    columns = ['mu_test','case','toy_id','q','mu_hat','valid','n_observed']
    pd.testing.assert_frame_equal(reused[columns].reset_index(drop=True), expected[columns].reset_index(drop=True), check_dtype=False)
    assert reused.origin.eq('07').all()
    assert len(calls)==6 and result['summary'].failed.sum()==1
    assert result['reference_reuse']['available_rows']==12
    assert hashes=={p.name:file_digest(p) for p in old.iterdir() if p.is_file()}


def test_partial_reference_extension_matches_fresh_run_and_rejects_mismatch(prepared, tmp_path):
    old = tmp_path/'07'
    run_own_model_toys(legacy(prepared), old, n_toys=1, grid_size=9, progress=lambda _:None)
    kwargs = dict(n_toys=3, grid_size=9, progress=lambda _:None)
    mixed = run_fixed_score_toys(prepared, tmp_path/'13', reference_directory=old, **kwargs)
    fresh = run_fixed_score_toys(prepared, tmp_path/'fresh', **kwargs)
    pd.testing.assert_frame_equal(normalized(mixed['results']), normalized(fresh['results']), check_dtype=False)
    assert mixed['summary'].reused_from_07.sum()==4
    prepared['configuration']['reference']['sample_fingerprint']='changed'
    with pytest.raises(ValueError, match='sample_fingerprint'):
        run_fixed_score_toys(prepared, tmp_path/'mismatch', reference_directory=old, **kwargs)
    assert not (tmp_path/'mismatch').exists()
    with pytest.raises(ValueError, match='separate'):
        run_fixed_score_toys(prepared, old, reference_directory=old, **kwargs)


def test_interrupt_saves_completed_ids_and_no_redraw(prepared, tmp_path):
    def interrupt(message):
        if '2/3;' in message:
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_fixed_score_toys(prepared, tmp_path, n_toys=3, grid_size=9,
            checkpoint_every=100, progress_every=2, progress=interrupt)
    file = next(tmp_path.glob('toys_*.csv'))
    before = pd.read_csv(file)
    assert len(before)==2
    result = run_fixed_score_toys(prepared, tmp_path, n_toys=3, grid_size=9, progress=lambda _:None)
    pd.testing.assert_frame_equal(before, pd.read_csv(file).iloc[:2].reset_index(drop=True), check_dtype=False)
    assert len(result['results'])==24


def test_zero_support_null_is_valid_and_bad_generator_is_recorded(prepared, tmp_path):
    # Bin 0 has exact zero at truth 0, but positive Asimov content at truth 1.
    prepared['templates']['score36'] = PoissonTemplate(np.array([[0.,1.], [2.,4.], [0.,3.], [0.,2.]]))
    result = run_fixed_score_toys(prepared, tmp_path/'zero', n_toys=3, grid_size=17, progress=lambda _:None)
    rows = result['results'].query('case=="score36" and mu_test==0')
    assert rows.valid.all()
    audit = result['sources'].query('case=="score36" and mu_test==0').iloc[0]
    assert audit.zero_mean_bins_with_mu1_support==1 and audit.mu1_yield_in_zero_mean_bins==2.
    prepared['mu_values'] = (.25,)
    prepared['templates']['score36'] = PoissonTemplate(np.array([[100.], [0.], [0.], [1.]]))
    bad = run_fixed_score_toys(prepared, tmp_path/'bad', n_toys=2, grid_size=9, progress=lambda _:None)
    assert not bad['results'].query('case=="score36"').valid.any()
    assert bad['results'].query('case!="score36"').valid.all()


def test_actual_run_and_plot_cells(prepared, tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(root/'notebooks/13_fixed_score_toys.ipynb', as_version=4)
    cells = {c.id:c.source for c in nb.cells}
    env = dict(prepared=prepared, run_fixed_score_toys=run_fixed_score_toys,
        RESULT_DIR=tmp_path/'study', FIGURE_DIR=tmp_path/'plots', REFERENCE_DIR=None,
        N_TOYS=3, SEED=110923, MU_BOUNDS=(0.,3.), FIT_GRID_SIZE=17,
        CHECKPOINT_EVERY=2, PROGRESS_EVERY=100, display=lambda *_:None, plt=plt)
    exec(cells['run-toys'], env)
    exec(cells['plots'], env)
    assert len(list(env['FIGURE_DIR'].glob('*.png')))==4
    assert len(list(env['FIGURE_DIR'].glob('*.pdf')))==4
    from poodemo.fixed_score_toy_plots import plot_fixed_score_toys
    for name, fig in plot_fixed_score_toys(env['study'], tmp_path/'inspect'):
        assert fig.axes[0].get_yscale()=='log'
        assert len(fig.axes[1].lines) in (4,5)  # CDFs and optional branch split.
        if 'score_q' in name:
            assert fig.axes[2].get_yscale()=='log'
            assert all(line.get_ydata()[-1]==0 for line in fig.axes[2].lines)
        plt.close(fig)


def test_new_notebook_is_clean_reproducible_and_has_production_budget(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    from build_fixed_score_toys_notebook import score_toy_cells, NAME
    nb = nbformat.read(root/'notebooks'/NAME, as_version=4)
    nbformat.validate(nb)
    assert [c.source for c in nb.cells]==[c.source for c in score_toy_cells()]
    cells = {c.id:c.source for c in nb.cells}
    assert 'N_TOYS = 10_000' in cells['settings']
    assert 'MU_VALUES = (0.0, 1.4)' in cells['settings']
    assert 'BIN_COUNTS = (36, 144)' in cells['settings']
    assert '250_000' not in cells['model-source']
    for cell in nb.cells:
        if cell.cell_type=='code':
            compile(cell.source, NAME+':'+cell.id, 'exec')
            assert cell.outputs==[] and cell.execution_count is None
        else:
            assert r'\(' not in cell.source and r'\[' not in cell.source
