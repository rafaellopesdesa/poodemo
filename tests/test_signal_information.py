"""Dense information sweep and notebook-03 overlays without extra simulation."""
from pathlib import Path
import json

import nbformat
import numpy as np
import pandas as pd
import pytest

from poodemo.signal_information import (poisson_information, run_signal_information_study,
                                        plot_signal_information)
from test_colab_mc import setup, prepare


def test_poisson_rate_shape_split_and_nonregular_means():
    fields=np.array([[2.,1.],[5.,4.],[3.,2.],[1.,3.]])
    info,means,derivative=poisson_information(fields,1.)
    np.testing.assert_array_equal(means,[6.,7.])
    np.testing.assert_array_equal(derivative,[2.,1.5])
    assert info['information']==pytest.approx(4/6+2.25/7)
    assert info['rate_information']==pytest.approx(3.5**2/13)
    assert info['shape_information']==pytest.approx(info['information']-info['rate_information'])
    # An entirely empty bin contributes zero; a B-only MC bin is nonregular at 1.
    empty=np.column_stack([fields,np.zeros(4)])
    assert poisson_information(empty,1.)[0]['information']==info['information']
    empty[2,-1]=.001
    bad=poisson_information(empty,1.)[0]
    assert bad['nonregular_bins']==1 and not bad['information_valid']
    assert np.isnan(bad['information']) and np.isnan(bad['shape_information'])
    assert poisson_information(empty,1.1)[0]['negative_bins']==1
    # Shape information uses this model's rate term, not another model's rate.
    constant_score=np.array([[1.,2.],[2.,4.],[1.,2.],[2.,4.]])
    assert poisson_information(constant_score,1.)[0]['shape_information']==pytest.approx(0.,abs=1e-14)
    weighted=poisson_information(fields,1.,weights=np.array([2.,3.]))[0]
    assert weighted['information']==pytest.approx(2*4/6+3*2.25/7)
    with pytest.raises(ValueError):poisson_information(fields,0.)


def test_information_sweep_reuses_predictions_and_matches_direct_bins(setup,tmp_path,monkeypatch):
    from poodemo import colab_discriminators
    base,calls=setup
    base=base.with_config(reference_ratio_denominator='S',reference_ratio_power=2.)
    samples=prepare(base)
    before=calls.copy()
    fields=np.array([[2.,1.],[5.,4.],[3.,2.],[1.,3.]])
    monkeypatch.setattr(colab_discriminators,'cached_analytic_quadrature',
                        lambda _:dict(nominal=fields,weights=np.ones(2)))
    study=run_signal_information_study(base,samples,bin_counts=[4,6,8,12,36],
                                      prefixes=[5,13],progress=lambda _:None)
    assert calls==before
    assert len(study['information'])==10
    assert set(study['information'].n_bins)=={4,6,8,12,36}
    assert set(study['information'].eta)=={1.}
    grouped=study['diagnostics'].groupby(['n_bins','selected_per_source'])
    for (n,size),bins in grouped:
        raw=bins[['count_S','count_SBI','count_B','count_NI']].to_numpy().T
        np.testing.assert_array_equal(raw.sum(axis=1),np.full(4,size))
        expected,_,_=poisson_information(raw*samples['rates'][:4,None]/size,1.)
        row=study['information'].query('n_bins == @n and selected_per_source == @size').iloc[0]
        assert row.information_valid==expected['information_valid']
        np.testing.assert_allclose(row.information,expected['information'],equal_nan=True)
    config=json.loads((study['output']/'configuration.json').read_text())
    assert config['observable']['reference_ratio_power']==2.
    assert config['observable']['reference_ratio_denominator']=='S'
    assert config['prefixes']==[5,13]
    repeated=run_signal_information_study(base,samples,bin_counts=[4,6,8,12,36],
                                         prefixes=[5,13],progress=lambda _:None)
    pd.testing.assert_frame_equal(study['information'],repeated['information'])
    assert calls==before
    with pytest.raises(ValueError,match='10M'):
        run_signal_information_study(base,{**samples,'target':10_000_001},bin_counts=[4])


def test_plot_shows_missing_points_without_fabricating_information(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(dict(n_bins=[4,8,12,36],eta=1.,selected_per_source=10_000_000,
        information_fraction=[.9,np.nan,.98,np.nan],shape_information_fraction=[.8,np.nan,.97,np.nan],
        information_valid=[True,False,True,False]))
    fig=plot_signal_information(dict(information=frame),selected_per_source=10_000_000,output=tmp_path)
    ax=fig.axes[0]
    assert ax.get_xscale()=='log'
    assert np.isnan(ax.lines[0].get_ydata()[[1,3]]).all()
    np.testing.assert_array_equal(ax.lines[2].get_xdata(),[8,36])
    assert 'Undefined Fisher' in ax.lines[2].get_label()
    assert (tmp_path/'04_binning_diagnostics.png').is_file()
    plt.close(fig)


def test_actual_04_likelihood_plots_include_both_unbinned_references(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/'notebooks/04_signal_ratio_histograms.ipynb'
    cells={c.id:c.source for c in nbformat.read(path,4).cells}
    grid=np.array([.02,.12,.3,1.,1.4])
    def rows(name,q,systematics=False):
        return pd.DataFrame(dict(mu=grid,q=q,model=name,systematics=systematics,valid=True))
    analytic=rows('analytic unbinned',[5.,2.,3.,0.,1.])
    binned=rows('direct reference_ratio histogram',[4.,1.5,2.5,0.,.9]).assign(n_bins=36)
    learned=pd.concat([rows('learned unbinned (analytic Asimov)',[4.5,1.8,2.8,0.,.95],s)
                       for s in (False,True)],ignore_index=True)
    run=SimpleNamespace(path=lambda *parts:tmp_path.joinpath(*parts))
    (tmp_path/'results').mkdir()
    namespace=dict(run=run,np=np,pd=pd,plt=plt,binning={'scans':pd.concat([binned,analytic])},
                   learned_unbinned=learned,display=lambda *a:None)
    exec(cells['cell-012'],namespace)
    exec(cells['cell-013'],namespace)
    for number in plt.get_fignums():
        ax=plt.figure(number).axes[0]
        labels=[line.get_label() for line in ax.lines]
        assert any('learned unbinned' in label for label in labels)
        assert any('analytic unbinned' in label for label in labels)
    assert (tmp_path/'plots/04_binning_scans_secondary.pdf').is_file()
    assert set(pd.read_csv(tmp_path/'results/signal_likelihood_comparison.csv').model)=={
        'analytic unbinned','learned unbinned (analytic Asimov)','direct reference_ratio histogram'}
    plt.close('all')
    assert 'learned_reference(run, scan_grid)' in cells['cell-010']
    assert 'learned_unbinned.loc[' in cells['colab-convergence']
    assert 'learned_unbinned.loc[' in cells['mc-review-power']
    # The denser plot does not enlarge the full likelihood-scan sweep or event bank.
    assert 'BIN_COUNTS = [4, 12, 24, 36, 60]' in cells['colab-sampling-setup']
    assert 'INFORMATION_BIN_COUNTS = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 48, 60, 72, 96, 108, 144]' in cells['colab-sampling-setup']
