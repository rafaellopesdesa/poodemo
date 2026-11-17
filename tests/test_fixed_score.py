"""Score convention, frozen bins, full physical yields and Colab entry points."""
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, xlogy

from poodemo.fixed_score import shape_score_at_one, fill_fixed_score, run_fixed_score_study
from poodemo.inference import component_coefficients
from poodemo.pipeline import _positive_study_grid
from test_colab_mc import setup,prepare


def test_score_is_derivative_of_normalized_density_and_uses_selected_rates():
    rng=np.random.default_rng(32)
    fields=np.exp(rng.normal(size=(4,100)))
    rates=np.array([3.,7.,8.,19.])
    def log_density(mu):
        c=component_coefficients(mu)
        return np.log(c@fields)-np.log(c@rates)
    h=1e-5
    expected=(log_density(1+h)-log_density(1-h))/(2*h)
    score=shape_score_at_one(fields,rates)
    np.testing.assert_allclose(score,expected,rtol=1e-7,atol=1e-9)
    literal=(fields[0]+fields[1]-fields[2])/(fields[1]+fields[3])
    center=.5*(rates[0]+rates[1]-rates[2])/(rates[1]+rates[3])
    np.testing.assert_allclose(score,.5*literal-center)
    np.testing.assert_allclose(shape_score_at_one(fields*np.exp(rng.normal(size=100)),rates),score)
    np.testing.assert_allclose(shape_score_at_one(fields*1e200,rates*1e200),score)
    with pytest.raises(ValueError):shape_score_at_one(fields,np.ones(6))
    with pytest.raises(ValueError):shape_score_at_one(np.zeros((4,2)),rates)


def test_fixed_partition_streaming_totals_variations_and_cache(setup,monkeypatch):
    from poodemo.colab_reference import ColabTemplateBank
    run,calls=setup
    run=run.with_config(reference_ratio_denominator='S',reference_ratio_power=2.)
    samples=prepare(run);before=calls.copy()
    bank=fill_fixed_score(run,samples,bin_counts=(4,8),progress=lambda _:None)
    stream=ColabTemplateBank(run,samples)
    for i,state in enumerate(samples['manifests'].values()):
        score=np.concatenate([shape_score_at_one(f,samples['rates'][:4]) for _,f in stream._fields(state)])
        for n in (4,8):
            expected=np.histogram(expit(score/.04),bank['edges'][n])[0]
            np.testing.assert_array_equal(expected,bank['counts'][n][i])
            assert bank['counts'][n][i].sum()==samples['target']
    assert calls==before
    with monkeypatch.context() as m:
        m.setattr(ColabTemplateBank,'_fields',lambda *_:pytest.fail('completed counts should be reused'))
        resumed=fill_fixed_score(run,samples,bin_counts=(4,8),progress=lambda _:None)
        assert resumed['identity']==bank['identity']
    changed=fill_fixed_score(run,samples,bin_counts=(4,8),score_scale=.08,progress=lambda _:None)
    assert changed['identity']!=bank['identity']
    path=bank['directory']/'counts.npz'
    with np.load(path) as saved: arrays={key:saved[key] for key in saved.files}
    arrays['counts_4'][0,0]+=1
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='content mismatch'):
        fill_fixed_score(run,samples,bin_counts=(4,8),progress=lambda _:None)
    with pytest.raises(ValueError,match='10M'):
        fill_fixed_score(run,{**samples,'target':10_000_001})


def test_actual_notebook_study_and_plots_use_frozen_full_likelihood(setup,tmp_path,monkeypatch):
    run,calls=setup
    run=run.with_config(reference_ratio_denominator='S',reference_ratio_power=2.,
        mc_zero_max_bin=.01,mc_zero_max_total=.05)
    samples=prepare(run);before=calls.copy()
    refs=pd.DataFrame([dict(model=model,mu=mu,systematics=syst,q=(mu-1.)**2,valid=True)
        for model in ('analytic unbinned','learned unbinned (analytic Asimov)')
        for mu in _positive_study_grid(run) for syst in (False,True)])
    root=Path(__file__).resolve().parents[1]
    notebook=nbformat.read(root/'notebooks/12_fixed_score_histograms.ipynb',as_version=4)
    cells={c.id:c.source for c in notebook.cells}
    import matplotlib.pyplot as plt
    env=dict(run=run,samples=samples,references=refs,reference_provenance={'test':'synthetic references'},
             BIN_COUNTS=(4,8),SCORE_SCALE=.04,display=lambda *_:None,plt=plt)
    exec(cells['study'],env)
    study=env['study']
    assert calls==before  # Simulator and predictor boundaries were never crossed.
    assert study['scans'].eta.eq(1.).all()
    assert set(study['scans'].systematics)=={False,True}
    assert 0. in set(study['scans'].mu)
    for n,model in study['models'].items():
        counts=study['bank']['counts'][n]
        raw_yields=counts*samples['rates'][:,None]/samples['target']
        np.testing.assert_allclose(model.nominal,raw_yields[:4])
        np.testing.assert_allclose(model.nominal.sum(axis=1),samples['rates'][:4])
        # At zero the exact physical mean is B+NI, not the tangent at mu=1.
        np.testing.assert_allclose(model.raw_intensity(0.,[0.]),model.nominal[2]+model.nominal[3])
        np.testing.assert_allclose(model.raw_intensity(1.,[-1.]),raw_yields[1]+raw_yields[5])
        np.testing.assert_allclose(model.raw_intensity(1.,[1.]),raw_yields[1]+raw_yields[4])
        rows=study['scans'].query('n_bins==@n')
        for row in rows.loc[~rows.systematics & rows.valid].itertuples():
            means=model.intensity(row.mu,[0.]);data=model.asimov_weights
            positive=data>0
            expected=2*(means.sum()-data.sum()+np.sum(xlogy(data[positive],data[positive]/means[positive])))
            np.testing.assert_allclose(row.q,expected,atol=1e-10)
        merged=rows.loc[rows.systematics].merge(rows.loc[~rows.systematics],on='mu',suffixes=('_profile','_fixed'))
        valid=merged.valid_profile & merged.valid_fixed
        assert (merged.loc[valid,'q_profile']<=merged.loc[valid,'q_fixed']+1e-8).all()
    assert study['scans'].loc[study['scans'].mu.eq(1.),'q'].abs().max()<1e-9
    exec(cells['plots'],env)
    assert len(list(env['FIGURE_DIR'].glob('*.png')))==4
    assert len(list(env['FIGURE_DIR'].glob('*.pdf')))==4
    assert (study['output']/'metadata.json').is_file()
    assert len(pd.read_csv(study['output']/'unbinned.csv'))==len(refs)


def test_notebook_matches_builder_and_math_uses_dollar_delimiters(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    from build_fixed_score_notebook import score_cells,NAME
    notebook=nbformat.read(root/'notebooks'/NAME,as_version=4)
    # Validate runnable sources without rewriting committed Colab outputs.
    for cell in notebook.cells:
        if cell.cell_type=='code':
            cell.outputs=[];cell.execution_count=None
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells]==[c.source for c in score_cells()]
    for cell in notebook.cells:
        if cell.cell_type=='code':
            compile(cell.source,NAME+':'+cell.id,'exec')
            assert cell.execution_count is None and cell.outputs==[]
        else:
            assert r'\(' not in cell.source and r'\[' not in cell.source
