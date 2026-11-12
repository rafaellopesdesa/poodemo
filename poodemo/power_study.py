"""Paired power controls from cached NN predictions, without new MC or training."""
import numpy as np
import pandas as pd

from .colab_mc import _digest, _read_array, _count_digest
from .data import PROCESSES, save_json
from .inference import _bin_indices, component_coefficients
from .mc_yields import yield_policy
from .pipeline import observable_bin_edges
from .practical import SimulationHistogram, fit_direct_asimov
from .toy_integration_diagnostics import _atomic_npz
from .toy_study import reference_observable


def run_power_study(run, samples, *, powers=(50.,100.,150.), etas=(0.,.02,.12,.2), n_bins=36, progress=print):
    powers, etas = np.unique(np.asarray(powers,float)), np.unique(np.asarray(etas,float))
    if not len(powers) or np.any(powers<=0) or not np.isfinite(powers).all():
        raise ValueError('Powers must be positive finite values')
    if not len(etas) or np.any(etas<0) or not np.isfinite(etas).all():
        raise ValueError('Eta values must be finite and nonnegative')
    edges = observable_bin_edges(n_bins,'reference_ratio')
    metadata = dict(version=1, sample_fingerprint=samples['fingerprint'], target=samples['target'],
                    powers=powers.tolist(), etas=etas.tolist(), edges=edges.tolist(),
                    rates=samples['rates'].tolist(), normalizers=samples['normalizers'].tolist())
    fingerprint = _digest(metadata)
    root = samples['directory']/'power_checks'/fingerprint[:20]
    root.mkdir(parents=True,exist_ok=True)
    save_json(root/'configuration.json',metadata)
    path=root/'counts.npz'
    shape=(len(powers),len(etas),6,n_bins)
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            counts, completed, saturated = z['counts'],z['completed'],z['saturated']
            if str(z['fingerprint']) != fingerprint or str(z['payload_sha256']) != _count_digest(counts,completed,saturated):
                raise ValueError('Power-control checkpoint changed')
        if counts.shape!=shape or completed.shape!=(6,) or saturated.shape!=shape[:-1]+(2,):
            raise ValueError('Malformed power checkpoint')
    else:
        counts=np.zeros(shape,dtype=np.int64);completed=np.zeros(6,dtype=np.int64)
        saturated=np.zeros(shape[:-1]+(2,),dtype=np.int64)
    for si,(source,state) in enumerate(samples['manifests'].items()):
        start=0
        for ci,chunk in enumerate(state['chunks']):
            take=min(chunk['count'],samples['target']-start)
            if take<=0: break
            if ci<completed[si]:
                start+=take;continue
            ratios=np.vstack([np.ones(take)]+[_read_array(samples['directory'],chunk['ratios'][p])[:take] for p in PROCESSES[1:]])
            fields=samples['rates'][:4,None]*ratios/samples['normalizers'][:,None]
            for ei,eta in enumerate(etas):
                for pi,power in enumerate(powers):
                    z=reference_observable(fields,samples['rates'][:4],eta,power,0.)
                    counts[pi,ei,si]+=np.bincount(_bin_indices(z,edges),minlength=n_bins)
                    saturated[pi,ei,si]+=[np.count_nonzero(z==0),np.count_nonzero(z==1)]
            completed[si]=ci+1;start+=take
            _atomic_npz(path,counts=counts,completed=completed,saturated=saturated,
                        fingerprint=np.array(fingerprint),payload_sha256=np.array(_count_digest(counts,completed,saturated)))
            progress(f'Power controls: {source} {start:,}/{samples["target"]:,} cached')
    if not np.all(counts.sum(axis=-1)==samples['target']):
        raise ValueError('Power-control totals differ from the selected target')
    rows=[]
    for pi,power in enumerate(powers):
        for ei,eta in enumerate(etas):
            y=counts[pi,ei]*samples['rates'][:,None]/samples['target']
            nominal=y[:4];down=nominal.copy();up=nominal.copy();down[3]=y[5];up[3]=y[4]
            truth=component_coefficients(run.config['asimov_mu'])@nominal
            # Numerical 0/1 saturation is distinct from finite end-bin occupancy.
            sat=saturated[pi,ei]*samples['rates'][:,None]/samples['target']
            sat_truth=component_coefficients(run.config['asimov_mu'])@sat[:4]
            for syst in (False,True):
                model=SimulationHistogram(nominal,down[None],up[None],np.ones(n_bins),truth,
                                          yield_policy=yield_policy(run.config))
                fit=fit_direct_asimov(run,model,float(eta),syst) if eta>0 else dict(q=np.nan,valid=False)
                rows.append(dict(power=power,eta=eta,systematics=syst,n_bins=n_bins,
                    selected_per_source=samples['target'],distribution_only=bool(eta==0),
                    reference_end_bin_fraction=float(truth[[0,-1]].sum()/truth.sum()),
                    reference_numerical_saturation_fraction=float(sat_truth.sum()/truth.sum()),**fit))
    frame=pd.DataFrame(rows)
    ref=pd.read_csv(run.path('results','unbinned_scans.csv'))
    frame['q_analytic']=np.nan
    for i,row in frame.iterrows():
        near=ref[ref.systematics.eq(row.systematics)&np.isclose(ref.mu,row.eta,rtol=0,atol=1e-12)&ref.valid]
        if len(near)==1:frame.loc[i,'q_analytic']=near.q.iloc[0]
    frame['delta_q_to_analytic']=frame.q-frame.q_analytic
    frame.to_csv(run.path('results','reference_ratio_power_controls.csv'),index=False)
    save_json(run.path('results','reference_ratio_power_controls.json'),
              {**metadata,'yield_policy':yield_policy(run.config).metadata(),
               'note':'Paired finite-MC controls, not an optimization on the analytical reference; eta=0 is distribution-only'})
    return frame
