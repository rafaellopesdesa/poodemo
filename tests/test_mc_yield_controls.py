"""Likelihood semantics, strict rejection, streamed CDF identity and bank reuse."""
import json
import numpy as np
import pandas as pd
import pytest

from poodemo.mc_yields import YieldPolicy, yield_diagnostics
from poodemo.practical import SimulationHistogram, local_information
from test_colab_mc import setup, prepare


def histogram(background=.004, observed=None):
    nominal=np.array([[1.,0.],[2.,0.],[1.,background],[1.,0.]])
    return SimulationHistogram(nominal,nominal[None],nominal[None],np.ones(2),
        nominal[1]+nominal[3] if observed is None else observed,yield_policy=YieldPolicy(.01,.05))


def test_zeroing_applies_to_total_with_exact_poisson_zero_semantics():
    model=histogram();nominal=model.nominal.copy()
    raw=model.raw_intensity(1.2,[0.]);expected=model.intensity(1.2,[0.])
    assert raw[1]<0 and expected[1]==0 and expected[0]==raw[0]
    np.testing.assert_array_equal(model.nominal,nominal)
    assert np.isfinite(model.nll(1.2,[0.]))
    assert model.expected_count(1.2,[0.])==expected.sum()
    assert yield_diagnostics(model,1.2,[0.])['zeroed_bins']==1
    assert np.isinf(histogram(observed=np.array([3.,.001])).nll(1.2,[0.]))
    # Negative interference coefficients are retained; the corrected total
    # must not change already-valid predictions at mu<1.
    np.testing.assert_array_equal(model.intensity(.5,[0.]),model.raw_intensity(.5,[0.]))


def test_policy_rejects_large_single_aggregate_and_nonfinite_predictions():
    p=YieldPolicy(.01,.05)
    for raw in [np.array([-.011,10.]),np.full(6,-.009),np.array([np.nan,1.]),np.array([np.inf,1.])]:
        corrected,audit=p.apply(raw)
        assert audit['rejected_negative'] and audit['zeroed_total']==0
        np.testing.assert_array_equal(corrected,raw)
    assert np.isinf(histogram(background=4.).nll(1.2,[0.]))
    assert YieldPolicy().apply(np.array([-.0001]))[1]['rejected_negative']
    for limits in [(-1.,1.),(1.,np.inf)]:
        with pytest.raises(ValueError):YieldPolicy(*limits)


def test_zeroed_scan_does_not_certify_nonregular_fisher():
    model=histogram()
    assert model.nll(1.,[0.])==0
    assert np.isfinite(model.nll(1.2,[0.]))
    assert np.isnan(local_information(model,1.)['matrix']).all()
    assert np.isnan(local_information(model,1.2)['matrix']).all()


def test_raw_interpolation_errors_remain_visible_after_zeroing():
    from poodemo.reference_splines import template_metrics
    exact=np.repeat(histogram().nominal[None],3,axis=0)
    pred=exact.copy();pred[:,2,1]*=1.1
    report=template_metrics(exact,pred,[1.2],YieldPolicy(.01,.05))
    assert report['direct_invalid_bins']==0
    assert report['direct_zeroed_bins']==report['interpolated_zeroed_bins']==3
    assert report['raw_mixture_max_abs_error']>0
    assert report['max_deviance']==0
    assert report['max_bin_pull']>0


def test_completed_08_reuse_streamed_cdf_and_policy_match(setup,tmp_path,monkeypatch):
    from poodemo.colab_mc import fill_colab_histograms
    from poodemo.colab_reference import publish_08_baseline,load_08_baseline,ColabTemplateBank
    from poodemo.reference_splines import CumulativeTemplates,run_reference_spline_comparison,SplineStudySettings
    from poodemo import pipeline
    base,calls=setup
    base=base.with_config(mc_zero_max_bin=.01,mc_zero_max_total=.05)
    samples=prepare(base)
    etas=[0.,.2,.99,1.,1.01,1.4]
    bank=fill_colab_histograms(base,samples,etas=etas,bin_counts=[8],prefixes=[13],progress=lambda _:None)
    study=bank.study_run(base)
    marker=publish_08_baseline(base,study,samples,bank,8)
    assert json.loads(marker.read_text())['selected_per_source']==13
    before=calls.copy()
    loaded,cached=load_08_baseline(base,8,local_cache=tmp_path/'stage',progress=lambda _:None)
    assert calls==before and loaded.evaluation_root==study.evaluation_root
    cdf=ColabTemplateBank(loaded,cached,8,2);interpolator=CumulativeTemplates(cdf,etas)
    for eta in etas:
        np.testing.assert_allclose(cdf.exact(eta),bank.anchors(eta,8),atol=1e-14)
        np.testing.assert_allclose(interpolator(eta),bank.anchors(eta,8),atol=1e-14)
        np.testing.assert_allclose(cdf.exact(eta).sum(axis=-1),cdf.totals,atol=1e-14)
        assert cdf.model(cdf.exact(eta),cdf.counts(eta)).yield_policy==YieldPolicy(.01,.05)
    # Off-anchor transported coordinates and ties use the same bin convention.
    from scipy.special import expit
    for anchor,eta in [(.2,.3),(1.,.98),(1.,1.02),(1.4,1.3)]:
        raw=[]
        for state in cached['manifests'].values():
            zs=[expit(cdf.power*(eta-1)*cdf.map.u(fields,anchor)) for _,fields in cdf._fields(state)]
            raw.append(np.histogram(np.concatenate(zs),cdf.edges)[0])
        np.testing.assert_allclose(cdf.cdf_u_bins(anchor,eta),cdf._anchors(raw),atol=1e-14)
    assert len(cdf.cdf_cache)<=2
    # Real 08 -> streamed 09 fits, with analytical comparison stubbed at the
    # simulator boundary only. No dense event/component bank may be requested.
    from poodemo.practical import refresh_analytic_reference
    from poodemo.reference_floor import run_reference_floor_study,FloorStudySettings
    from poodemo import practical
    nominal=np.broadcast_to(samples['rates'][:4,None],(4,10)).copy()
    aq=dict(nominal=nominal,down=nominal[None].copy(),up=nominal[None].copy(),
            weights=np.full(10,.1),truth=nominal[1]+nominal[3])
    aq['down'][0,3]*=np.exp(-.1);aq['up'][0,3]*=np.exp(.1)
    monkeypatch.setattr(pipeline,'prepare_quadrature',lambda _:aq)
    monkeypatch.setattr(practical,'prepare_binning_bank',lambda _:pytest.fail('dense bank requested'))
    loaded=loaded.with_config(bin_counts=[8])
    refresh_analytic_reference(loaded)
    # The scan also needs the middle configured point.
    complete=fill_colab_histograms(base,samples,etas=pipeline._positive_study_grid(loaded),
                                   bin_counts=[8],prefixes=[13],progress=lambda _:None)
    pipeline.run_binning_study(loaded,observable='reference_ratio',template_bank=complete)
    result=run_reference_spline_comparison(loaded,n_bins=8,template_bank=cdf,progress=lambda _:None,
        settings=SplineStudySettings(initial_anchors=3,focus_anchors=3,max_anchors=15,max_rounds=1,
            likelihood_refine_points=1,validation_intervals=2,validation_fit_points=1,
            surface_points=1,surface_mu_points=3,surface_alpha_points=3))
    assert result['metadata']['yield_policy']['mc_zero_max_bin']==.01
    floor=run_reference_floor_study(loaded,n_bins=8,template_bank=cdf,progress=lambda _:None,
        settings=FloorStudySettings(resolution_multiplier=2,alpha_grid_points=5))
    assert set(floor['scans'].n_bins)=={8,16}
    with pytest.raises(ValueError,match='differ'):
        load_08_baseline(base.with_config(reference_ratio_power=99.),8)
    # Changes to staged data are repaired from a verified source; changes to
    # the source too must fail rather than silently redefine the bank.
    item=cached['manifests']['S']['chunks'][0]['ratios']['SBI']
    (cached['ratio_directory']/item['file']).write_bytes(b'changed')
    load_08_baseline(base,8,local_cache=tmp_path/'stage',progress=lambda _:None)
    (samples['directory']/item['file']).write_bytes(b'changed')
    (cached['ratio_directory']/item['file']).unlink()
    with pytest.raises(ValueError,match='cache changed'):
        load_08_baseline(base,8,local_cache=tmp_path/'stage',progress=lambda _:None)


def test_power_controls_resume_without_prediction(setup):
    from poodemo.power_study import run_power_study
    base,calls=setup;samples=prepare(base);before=calls.copy()
    etas=[0.,.2,1.]
    pd.DataFrame([dict(mu=e,systematics=s,q=0.,valid=True) for e in etas[1:] for s in [False,True]]).to_csv(
        base.path('results','unbinned_scans.csv'),index=False)
    kwargs=dict(powers=[10.,20.],etas=etas,n_bins=4,progress=lambda _:None)
    a=run_power_study(base,samples,**kwargs);b=run_power_study(base,samples,**kwargs)
    pd.testing.assert_frame_equal(a,b)
    assert calls==before
    assert a.reference_end_bin_fraction.between(0,1).all()
    assert a.reference_numerical_saturation_fraction.between(0,1).all()
    assert a[a.eta.eq(1)].reference_end_bin_fraction.eq(0).all()
    assert a[a.eta.eq(0)].distribution_only.all()
