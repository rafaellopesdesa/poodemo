"""Selected-S normalization, cache isolation, streamed CDFs and notebook parity."""
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest
from scipy.special import expit

from poodemo.inference import component_coefficients
from poodemo.signal_reference import (signal_log_ratio, learned_observable,
    load_signal_baseline, publish_signal_baseline)
from test_colab_mc import setup, prepare


def test_normalized_signal_ratio_without_eta_one_collapse():
    fields = np.array([[2., 2., 2., 2.], [3., 8., 2., 5.],
                       [1., 4., 5., 2.], [2., 1., 3., 6.]])
    rates = fields.mean(axis=1)
    for eta in (0., .01, .99, 1., 1.01, 1.4):
        mixture = component_coefficients(eta) @ fields
        ratio = (mixture/(component_coefficients(eta) @ rates))/(fields[0]/rates[0])
        np.testing.assert_allclose(np.exp(signal_log_ratio(fields, rates, eta)), ratio)
        np.testing.assert_allclose(learned_observable(fields, rates, eta, 1., denominator='S'), ratio/(1+ratio))
        # Changing luminosity or the common per-event reference measure cancels.
        np.testing.assert_allclose(signal_log_ratio(10*fields,10*rates,eta), np.log(ratio))
        np.testing.assert_allclose(signal_log_ratio(fields*np.arange(1,5),rates,eta), np.log(ratio))
    assert np.ptp(learned_observable(fields,rates,1.,1.,denominator='S')) > .1
    np.testing.assert_array_equal(learned_observable(fields,rates,1.,1.), np.full(4,.5))
    np.testing.assert_array_equal(learned_observable(fields,rates,0.,1.,.2,denominator='S'),
                                  learned_observable(fields,rates,.2,1.,denominator='S'))
    with pytest.raises(ValueError): learned_observable(fields,rates,1.,1.,denominator='unknown')
    with pytest.raises(ValueError): learned_observable(fields,rates,1.,0.,denominator='S')
    bad = fields.copy(); bad[1] = -100
    with pytest.raises(ValueError): signal_log_ratio(bad,rates,1.)


def test_shared_events_separate_histograms_and_signal_ecdf(setup,tmp_path):
    from poodemo.colab_mc import fill_colab_histograms
    from poodemo.colab_reference import ColabTemplateBank, publish_08_baseline, load_08_baseline
    from poodemo.reference_splines import CumulativeTemplates, _fingerprint
    base,calls = setup
    base = base.with_config(reference_ratio_power=1.,mc_zero_max_bin=.01,mc_zero_max_total=.05)
    samples = prepare(base)
    before = calls.copy()
    signal = base.with_config(reference_ratio_denominator='S')
    reused = prepare(signal)
    assert reused['fingerprint']==samples['fingerprint'] and calls==before
    etas=[0.,.2,.99,1.,1.01,1.4]
    kwargs=dict(etas=etas,bin_counts=[36],prefixes=[samples['target']],progress=lambda _:None)
    old = fill_colab_histograms(base,samples,**kwargs)
    new = fill_colab_histograms(signal,samples,**kwargs)
    assert old.directory != new.directory  # Even at identical powers!
    old_marker=publish_08_baseline(base,old.study_run(base),samples,old,36)
    old_bytes=old_marker.read_bytes()
    marker=publish_signal_baseline(signal,new.study_run(signal),samples,new,36)
    assert marker.name=='colab_04_baseline.json' and old_marker.read_bytes()==old_bytes
    loaded,cached=load_signal_baseline(signal,36,local_cache=tmp_path/'stage',progress=lambda _:None)
    assert loaded.evaluation_root==new.study_run(signal).evaluation_root
    assert calls==before  # No extra simulation or network inference.
    with pytest.raises(ValueError,match='differs'): load_08_baseline(signal,36,progress=lambda _:None)
    with pytest.raises(ValueError): load_signal_baseline(base,36,progress=lambda _:None)
    from poodemo.colab_discriminators import completed_histograms
    with pytest.raises(ValueError,match='does not match'):
        completed_histograms(base,cached)
    np.testing.assert_array_equal(completed_histograms(loaded,cached).counts,new.counts)
    cdf=ColabTemplateBank(loaded,cached,36,2)
    interpolation=CumulativeTemplates(cdf,etas)
    for eta in etas:
        np.testing.assert_allclose(cdf.exact(eta),new.anchors(eta,36),atol=1e-14)
        np.testing.assert_allclose(cdf.cdf_u_bins(eta,eta),cdf.exact(eta),atol=1e-14)
        np.testing.assert_allclose(interpolation(eta),cdf.exact(eta),atol=1e-14)
        np.testing.assert_allclose(cdf.exact(eta).sum(axis=-1),cdf.totals,atol=1e-14)
    # With log(R_S) there is no eta-1 transport or reversal.
    np.testing.assert_allclose(interpolation(.1),.5*(cdf.exact(0)+cdf.exact(.2)),atol=1e-14)
    assert len(cdf.cdf_cache)<=2
    old_cdf=ColabTemplateBank(old.study_run(base),samples,36,2)
    assert _fingerprint(cdf)!=_fingerprint(old_cdf)
    assert loaded.config['reference_ratio_denominator']=='S'
    assert json.loads((new.directory/'configuration.json').read_text())['reference_ratio_denominator']=='S'


def test_zero_physical_density_is_retained_in_first_cdf_bin():
    from types import SimpleNamespace
    from poodemo.colab_reference import ColabTemplateBank
    fields=np.array([[1.,1.,1.],[2.,3.,4.],[0.,1.,2.],[0.,1.,2.]])
    rates=np.ones(4)
    u=signal_log_ratio(fields,rates,0.)
    assert np.isneginf(u[0])
    # Exercise endpoint/tie handling, including -infinity at a true zero.
    bank=ColabTemplateBank(SimpleNamespace(config={'reference_ratio_power':1.,'reference_ratio_eta_floor':0.,
                           'reference_ratio_denominator':'S'}),
        dict(rates=np.ones(6),normalizers=np.ones(4),target=3,fingerprint='zero-endpoint'),n_bins=36)
    bank._cdf=lambda _: [np.sort(u)]*6
    direct=np.histogram(expit(u),bank.edges)[0]
    np.testing.assert_allclose(bank.cdf_u_bins(0.,1.),bank._anchors(np.tile(direct,(6,1))))


def test_parallel_notebooks_are_clean_and_match_builder(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    from build_signal_reference_notebooks import NAMES,signal_cells
    for number,name in NAMES.items():
        nb=nbformat.read(root/'notebooks'/name,as_version=4)
        old=nbformat.read(next((root/'notebooks').glob(f'{number+4:02}_*.ipynb')),as_version=4)
        nbformat.validate(nb)
        assert [c.source for c in nb.cells]==[c.source for c in signal_cells(number)]
        assert {c.id for c in nb.cells}=={c.id for c in old.cells}|{'signal-parallel'}
        code={c.id:c.source for c in nb.cells if c.cell_type=='code'}
        for c in nb.cells:
            if c.cell_type=='code':
                compile(c.source,name+':'+c.id,'exec')
                assert c.execution_count is None and c.outputs==[]
            else:
                assert r'\(' not in c.source and r'\[' not in c.source
        controls=code.get('settings',code.get('cell-003'))
        assert 'reference_ratio_denominator="S"' in controls
        assert 'REFERENCE_RATIO_A = 2.0' in controls
        assert 'REFERENCE_RATIO_ETA_MIN = 0.0' in controls
        if number>4: assert 'load_signal_baseline' in '\n'.join(code.values())
        if number==7:
            for setting in ('N_TOYS = 10_000','MU_VALUES = (0.0, 1.4)',
                            'MOVING_BIN_COUNTS = (36, 72, 108, 144)','FIT_GRID_SIZE = 129'):
                assert setting in controls
            assert 'prepare_signal_model_toys' in code['prepare']
        if number==4:
            assert 'MC_SEED = 20261111' in code['colab-sampling-setup']
            assert 'POWER_VALUES = [1.0, 2.0, 3.0]' in code['mc-review-power']
