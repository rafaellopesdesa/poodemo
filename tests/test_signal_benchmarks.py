"""Reuse the exact 04 references without confusing them with interpolation closure."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from poodemo.data import file_digest, save_json
from poodemo.discriminator_unbinned import LEARNED_LABEL
from poodemo.pipeline import _positive_study_grid
from poodemo.signal_benchmarks import load_signal_benchmarks, add_signal_benchmarks


@pytest.fixture
def saved(tmp_path):
    def path(root):
        def make(*parts):
            value = tmp_path.joinpath(root, *parts)
            value.parent.mkdir(parents=True, exist_ok=True)
            return value
        return make
    owner = SimpleNamespace(path=path('03'), filter_manifest={'identity':'filtered'})
    run = SimpleNamespace(path=path('04'), base_run=owner,
        config=dict(asimov_mu=1., mu_min=.2, mu_max=1.4, mu_points=3))
    rows = [dict(mu=mu, systematics=syst, model=LEARNED_LABEL,
                 q=(mu-1)**2, valid=not (syst and mu < .3), asimov_mu=1.)
            for syst in (False,True) for mu in _positive_study_grid(run)]
    frame = pd.DataFrame(rows)
    frame.loc[~frame.valid,'q'] = np.nan
    frame.to_csv(owner.path('results','unbinned_scans.csv'), index=False)
    owner.path('results','unbinned_fits.csv').write_text('saved fits')
    artifact = owner.path('workspaces','learned','weights.npy')
    np.save(artifact,np.array([1.,2.]))
    provenance = dict(model=LEARNED_LABEL,filter_identity='filtered',
        scan_sha256=file_digest(owner.path('results','unbinned_scans.csv')),
        fit_sha256=file_digest(owner.path('results','unbinned_fits.csv')),
        workspace_artifacts=[dict(file=artifact.name,sha256=file_digest(artifact))])
    save_json(run.path('results','signal_learned_unbinned_reference.json'),provenance)
    frame.to_csv(run.path('results','signal_learned_unbinned_reference.csv'),index=False)
    frame.assign(model='analytic unbinned',valid=True,q=(frame.mu-1)**2).to_csv(
        run.path('results','unbinned_scans.csv'),index=False)
    return run,owner


def test_reuses_both_references_and_preserves_fit_validation(saved):
    run,owner = saved
    curves,provenance = load_signal_benchmarks(run)
    assert set(curves.model)=={'analytic unbinned',LEARNED_LABEL}
    assert (~curves.valid).sum()==1
    scans=curves.loc[curves.model.eq('analytic unbinned')].drop(columns='model').assign(method='direct',on_scan=True)
    scans['mu']+=1e-16  # Match CSV roundoff without interpolating the reference.
    scans=pd.concat([scans,scans.iloc[[0]].assign(mu=.456)],ignore_index=True)
    output=run.path('results','interpolation');output.mkdir()
    summary=pd.DataFrame({'passed':[False]})
    study=dict(scans=scans,metadata={'status':'complete'},output=output,summary=summary)
    source_bytes=run.path('results','unbinned_scans.csv').read_bytes()
    add_signal_benchmarks(study,curves,provenance)
    assert study['summary'] is summary and not summary.passed.iloc[0]
    assert study['scans'] is scans
    comparison=study['benchmark_comparison']
    assert comparison.loc[comparison.mu.eq(.456),'q_reference'].isna().all()
    assert comparison.loc[comparison.reference_valid.eq(False),'delta_q_to_reference'].isna().all()
    assert len(pd.read_csv(output/'unbinned.csv'))==len(curves)
    assert json.loads((output/'metadata.json').read_text())['unbinned_benchmarks']==provenance
    assert run.path('results','unbinned_scans.csv').read_bytes()==source_bytes


@pytest.mark.parametrize('change', ['scan','workspace','snapshot','truth','filter'])
def test_refuses_mixed_or_modified_reference_sources(saved,change):
    run,owner=saved
    if change=='scan':
        with owner.path('results','unbinned_scans.csv').open('a') as file: file.write('\n')
    elif change=='workspace':
        np.save(owner.path('workspaces','learned','weights.npy'),[2.,3.])
    elif change in ('snapshot','truth'):
        path=run.path('results','signal_learned_unbinned_reference.csv')
        frame=pd.read_csv(path)
        frame.loc[0,'q' if change=='snapshot' else 'asimov_mu']=2.
        frame.to_csv(path,index=False)
    else:
        owner.filter_manifest={'identity':'different'}
    with pytest.raises(ValueError): load_signal_benchmarks(run)
