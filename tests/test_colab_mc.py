"""Fresh selected streams, interruption recovery and compact inference in 08."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

from poodemo.data import Run, SAMPLES, save_array
from poodemo.practical import use_practical_run
from poodemo.colab_mc import (prepare_colab_samples, fill_colab_histograms,
    template_diagnostics, convergence_scans, _stage_predictor)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    calls=dict(generation=0,prediction=0,fail_generation=None,fail_prediction=None)
    class Simulator:
        def to_dict(self): return {'test':'independent generator'}
        def component_yield(self,process,alpha_ni=0.):
            return dict(S=2.,SBI=8.,B=4.,NI=10.)[process]*np.exp(.1*alpha_ni)
        def sample_component(self,process,n,rng,alpha_ni=0.):
            calls['generation']+=1
            if calls['generation']==calls['fail_generation']: raise RuntimeError('interrupt generation')
            return rng.normal(loc=.1*alpha_ni,size=(n,3))
        def component_densities(self,*args,**kwargs): pytest.fail('density oracle entered templates')
    run=Run(tmp_path,dict(seed=15,n_per_sample=1000,ni_sample_multiplier=1,
        asimov_mu=1.,mu_min=.2,mu_max=1.4,mu_points=3,mu_fit_bounds=[.001,2.],
        nuisance_bounds=[-2.,2.],reference_ratio_power=10.,reference_ratio_eta_floor=0.,bin_counts=[4,8]),Simulator())
    run.filter_manifest={'identity':'fixed'}
    x=np.random.default_rng(99).normal(size=(100,3)).astype('float32')
    for sample in SAMPLES: save_array(run.path('selected',f'{sample}_integration.npy'),x)
    def load_predictor(path):
        slope={'SBI_over_S':.1,'B_over_S':-.1,'NI_over_S':.2}[Path(path).name]
        def predict(x,batch_size):
            assert batch_size==3
            calls['prediction']+=1
            if calls['prediction']==calls['fail_prediction']: raise RuntimeError('interrupt prediction')
            return np.exp(slope*x[:,0])
        return SimpleNamespace(predict_ratio=predict)
    monkeypatch.setitem(sys.modules,'poodemo.training',SimpleNamespace(load_predictor=load_predictor))
    from poodemo import toy_sources
    def select(x,batch_size):
        assert batch_size==3
        return np.column_stack([x[:,0]>=0,x[:,0]<0])
    monkeypatch.setattr(toy_sources,'load_frozen_selector',lambda run:(SimpleNamespace(predict_proba=select),.5,{'fixed':True}))
    return use_practical_run(run),calls


def prepare(run,**kwargs):
    return prepare_colab_samples(run,selected_per_source=kwargs.pop('target',13),
        seed=111,selected_chunk=5,generation_chunk=7,inference_batch=3,
        normalization_events=11,progress=lambda _:None,**kwargs)


def test_target_is_after_selection_and_frozen_rates_do_not_scale_with_mc(setup):
    run,calls=setup
    small=prepare(run,target=7)
    large=prepare(run,target=13)
    assert small['fingerprint']==large['fingerprint']
    assert (large['sample_counts'].selected_used==13).all()
    assert (large['sample_counts'].selected_cached==15).all()
    assert (large['sample_counts'].generated>15).all()
    np.testing.assert_array_equal(small['rates'],large['rates'])
    np.testing.assert_array_equal(small['normalizers'],large['normalizers'])
    for sample,state in large['manifests'].items():
        for chunk in state['chunks']:
            x=np.load(large['directory']/chunk['file'])
            assert x.shape==(5,3) and np.all(x[:,0]>=0)
            assert len(chunk['ratios'])==3
        assert state['chunks'][:2]==small['manifests'][sample]['chunks'][:2]
    before=calls.copy()
    again=prepare(run)
    assert calls==before
    assert again['fingerprint']==large['fingerprint']


@pytest.mark.parametrize('stage',['generation','prediction'])
def test_interrupted_stream_is_identical_to_uninterrupted(setup,stage):
    run,calls=setup
    calls['fail_'+stage]=5
    with pytest.raises(RuntimeError,match='interrupt'): prepare(run)
    calls['fail_'+stage]=None
    resumed=prepare(run)
    # Rebuild in another result root with identical original inputs and seed.
    alternate=run.with_config()
    alternate.evaluation_root=run.evaluation_root/'separate'
    direct=prepare(alternate)
    assert resumed['fingerprint']==direct['fingerprint']
    for sample in SAMPLES:
        assert resumed['manifests'][sample]==direct['manifests'][sample]


def test_modified_cached_events_are_rejected(setup):
    run,_=setup
    sample=prepare(run)
    path=sample['directory']/sample['manifests']['S']['chunks'][0]['file']
    with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed or is missing'):prepare(run)


def test_streamed_histograms_match_direct_prefixes_and_resume(setup):
    run,calls=setup;samples=prepare(run)
    kwargs=dict(etas=[0.,.2,.99,1.,1.01,1.4],bin_counts=[4,8],prefixes=[4,7,13])
    def interrupt(_):raise RuntimeError('interrupt histogram')
    with pytest.raises(RuntimeError,match='interrupt histogram'):
        fill_colab_histograms(run,samples,progress=interrupt,**kwargs)
    bank=fill_colab_histograms(run,samples,progress=lambda _:None,**kwargs)
    from poodemo.toy_study import reference_observable
    from poodemo.pipeline import observable_bin_edges
    for si,(sample,state) in enumerate(samples['manifests'].items()):
        fields=np.vstack([np.ones(15)]+[np.concatenate([np.load(samples['directory']/c['ratios'][p]['file'])
                                      for c in state['chunks']]) for p in ('SBI','B','NI')])
        fields*=samples['rates'][:4,None]/samples['normalizers'][:,None]
        for eta in kwargs['etas']:
            z=reference_observable(fields,samples['rates'][:4],eta,10.)
            for n in kwargs['bin_counts']:
                for size in kwargs['prefixes']:
                    np.testing.assert_array_equal(bank.raw_counts(eta,n,size)[si],
                        np.histogram(z[:size],observable_bin_edges(n,'reference_ratio'))[0])
    for size in kwargs['prefixes']:
        a=bank.anchors(1.,8,size)
        np.testing.assert_allclose(a.sum(axis=-1), samples['rates'][[[0,1,2,3],[0,1,2,5],[0,1,2,4]]], atol=1e-14)
        np.testing.assert_allclose(a[0].sum(axis=-1),samples['rates'][:4],atol=1e-14)
        np.testing.assert_allclose(a[1,3].sum(),samples['rates'][5],atol=1e-14)
        np.testing.assert_allclose(a[2,3].sum(),samples['rates'][4],atol=1e-14)
        assert np.count_nonzero(a.sum(axis=(0,1)))==1
    table=bank.collapse_table([.99,1.,1.01],8)
    assert table.loc[table.eta.eq(1.),'z_min'].item()==.5
    assert table.loc[table.eta.eq(1.),'z_max'].item()==.5
    before=calls.copy();fill_colab_histograms(run,samples,progress=lambda _:None,**kwargs);assert calls==before


def test_compact_bank_drives_existing_08_fits_without_dense_sample_allocation(setup,monkeypatch):
    run,_=setup;samples=prepare(run)
    from poodemo import pipeline,practical
    from poodemo.diagnostics import run_estimator_study
    etas=pipeline._positive_study_grid(run)
    bank=fill_colab_histograms(run,samples,etas=np.r_[etas,.3],bin_counts=[4,8],prefixes=[7,13],progress=lambda _:None)
    study=bank.study_run(run)
    rates=samples['rates'][:4];nom=np.broadcast_to(rates[:,None],(4,10)).copy()
    aq=dict(nominal=nom,down=nom[None].copy(),up=nom[None].copy(),weights=np.full(10,.1),truth=nom[1]+nom[3])
    aq['down'][0,3]*=np.exp(-.1);aq['up'][0,3]*=np.exp(.1)
    monkeypatch.setattr(pipeline,'prepare_quadrature',lambda _:aq)
    monkeypatch.setattr(practical,'prepare_binning_bank',lambda _:pytest.fail('dense bank requested'))
    practical.refresh_analytic_reference(study)
    result=pipeline.run_binning_study(study,observable='reference_ratio',template_bank=bank)
    assert result['scans'].template_source.eq('generated_colab_mc_v1').all()
    estimator=run_estimator_study(study,observable='reference_ratio',n_bins=4,template_bank=bank)
    assert len(estimator)==4*len(etas)
    floor=run_estimator_study(study.with_config(reference_ratio_eta_floor=.3),observable='reference_ratio',n_bins=4,template_bank=bank,output_tag='floor')
    assert floor.eta_effective.min()==.3
    diagnostic=template_diagnostics(study,bank,etas=etas,n_bins=4)
    assert len(diagnostic)==len(etas)*4*2
    scans=convergence_scans(study,bank,etas=etas,n_bins=4)
    assert set(scans.selected_per_source)=={7,13}
    assert study.path('models','ratios')==run.path('models','ratios')
    assert study.path('results')!=run.path('results')


def test_staging_preserves_filtered_member_selection_and_calibration(tmp_path):
    source=tmp_path/'source'; member=tmp_path/'original';member.mkdir()
    (member/'model.pt').write_bytes(b'model');(member/'calibrator.joblib').write_bytes(b'calibration')
    source.mkdir();(source/'ensemble.json').write_text(json.dumps(dict(members=['../original'],labels=['S','B'],streaming=True,log_normalization=.23)))
    target=tmp_path/'cache';_stage_predictor(source,target)
    saved=json.loads((target/'ensemble.json').read_text())
    assert saved['log_normalization']==.23
    assert saved['members']==['members/0000']
    assert (target/saved['members'][0]/'model.pt').read_bytes()==b'model'
    assert (target/saved['members'][0]/'calibrator.joblib').read_bytes()==b'calibration'


def test_notebook_08_analysis_cells_run_on_small_fresh_campaign(setup,monkeypatch):
    """Execute the delivered analysis cells, with only sizes and ML IO reduced."""
    run,_=setup
    import ast
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from poodemo import pipeline
    import nbformat
    repo=Path(__file__).resolve().parents[1]
    notebook=nbformat.read(repo/'notebooks/08_reference_ratio_histograms.ipynb',as_version=4)
    nominal=np.broadcast_to(np.array([2.,8.,4.,10.])[:,None]*(2/3),(4,10)).copy()
    aq=dict(nominal=nominal,down=nominal[None].copy(),up=nominal[None].copy(),
            weights=np.full(10,.1),truth=nominal[1]+nominal[3])
    aq['down'][0,3]*=np.exp(-.1);aq['up'][0,3]*=np.exp(.1)
    monkeypatch.setattr(pipeline,'prepare_quadrature',lambda _:aq)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(Tensor=type('Tensor',(),{}),
        cuda=SimpleNamespace(is_available=lambda:False)))
    namespace=dict(run=run,np=np,pd=pd,plt=plt,Path=Path,IN_COLAB=False,N_BINS=8,
                   REFERENCE_RATIO_A=10.,REFERENCE_RATIO_ETA_MIN=0.,display=lambda *args:None)
    replacements={'SELECTED_EVENTS_PER_SOURCE':'13','MC_PREFIXES':'[4, 7, SELECTED_EVENTS_PER_SOURCE]',
                  'SELECTED_CHUNK':'5','GENERATION_CHUNK':'7','INFERENCE_BATCH':'3',
                  'NORMALIZATION_EVENTS':'11','BIN_COUNTS':'[4, 8]'}
    started=False
    for cell in notebook.cells:
        if cell.id=='colab-sampling-setup':started=True
        if not started or cell.cell_type!='code':continue
        source=cell.source
        if cell.id=='colab-sampling-setup':
            lines=source.splitlines()
            for node in reversed(ast.parse(source).body):
                if isinstance(node,ast.Assign) and len(node.targets)==1 and getattr(node.targets[0],'id','') in replacements:
                    name=node.targets[0].id;lines[node.lineno-1:node.end_lineno]=[name+' = '+replacements[name]]
            source='\n'.join(lines)
        exec(compile(source,cell.id,'exec'),namespace)
    assert namespace['templates'].target==13
    assert len(namespace['mc_scans'])==3*2*len(namespace['scan_grid'])
    assert namespace['run'].path('plots','08_mc_convergence.png').exists()
    plt.close('all')


def test_colab_builder_is_idempotent_and_preserves_controls(monkeypatch):
    import nbformat
    repo=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'scripts'))
    from build_colab_08 import configure_colab_08
    nb=nbformat.read(repo/'notebooks/08_reference_ratio_histograms.ipynb',as_version=4)
    first=configure_colab_08(nb.cells)
    second=configure_colab_08(first)
    assert [(c.id,c.source) for c in first]==[(c.id,c.source) for c in second]
    cell=next(c for c in second if c.id=='colab-sampling-setup')
    cell.source=cell.source.replace('SELECTED_EVENTS_PER_SOURCE = 10_000_000','SELECTED_EVENTS_PER_SOURCE = 20_000_000')
    third=configure_colab_08(second)
    assert 'SELECTED_EVENTS_PER_SOURCE = 20_000_000' in next(c.source for c in third if c.id=='colab-sampling-setup')


def test_diagnostics_identify_signed_empty_truth_bin_without_clipping(setup):
    from poodemo.colab_mc import HistogramBank
    run,_=setup
    directory=run.path('results','handmade');directory.mkdir(parents=True)
    # One B event in bin zero, all other component events in bin one.
    counts=np.zeros((1,1,6,4),dtype=np.int64)
    counts[0,0,:,1]=100
    counts[0,0,2,:2]=[1,99]
    samples=dict(rates=np.array([2.,8.,4.,10.,11.,9.]))
    bank=HistogramBank(samples,[dict(eta=1.4,n_bins=4)], [100],counts,np.array([[0.,1.]]),directory)
    study=bank.study_run(run)
    frame=template_diagnostics(study,bank,etas=[1.4],n_bins=4)
    first=frame.iloc[0]
    assert first.negative and first.nonregular_mu_at_truth
    assert first.truth_mean==0.
    assert first.minimum_expected==pytest.approx((1-np.sqrt(2))*.04)
    assert np.isinf(bank.model(1.4,4).nll(1.4,[0.]))


def test_changed_histogram_checkpoint_is_rejected(setup):
    run,_=setup;samples=prepare(run)
    kwargs=dict(etas=[.2,1.],bin_counts=[4],prefixes=[7,13],progress=lambda _:None)
    bank=fill_colab_histograms(run,samples,**kwargs)
    path=bank.directory/'counts.npz'
    with np.load(path) as saved:payload={k:saved[k] for k in saved.files}
    payload['counts'][0,0,0,0]+=1
    with path.open('wb') as f:np.savez(f,**payload)
    with pytest.raises(ValueError,match='payload changed'):fill_colab_histograms(run,samples,**kwargs)
