"""Reuse notebook 03's learned physical-Asimov benchmark without NN evaluation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import file_digest
from .diagnostics import local_width

LEARNED_LABEL = 'learned unbinned (analytic Asimov)'


def objective_curvature(objective, truth, step=1e-3):
    """Half the Hessian of -2 log L at the common truth, with a step check.

    For a misspecified learned model this is local objective curvature, not
    a Fisher/sandwich coverage assertion and not a fitted confidence interval.
    """
    point = np.array([truth, 0.])
    h = np.array([min(step, truth/4), step])
    def hessian(steps):
        center = float(objective(point))
        result = np.empty((2, 2))
        for i in range(2):
            shift = np.eye(2)[i]*steps[i]
            result[i, i] = (float(objective(point+shift))-2*center+float(objective(point-shift)))/steps[i]**2
        a, b = np.eye(2)*steps[:, None]
        result[0, 1] = result[1, 0] = (float(objective(point+a+b))-float(objective(point+a-b))
            -float(objective(point-a+b))+float(objective(point-a-b)))/(4*steps[0]*steps[1])
        return result/2
    coarse, fine = hessian(h), hessian(h/2)
    relative_error = float(np.linalg.norm(fine-coarse)/max(np.linalg.norm(fine), 1e-30))
    valid = np.isfinite(fine).all() and relative_error < 1e-3
    return ((4*fine-coarse)/3 if valid else np.full((2, 2), np.nan)), relative_error


def _load_objective(workspace_path):
    from .toolkit import load_model
    # Saved workspaces contain absolute paths. Rebase only their own .npy
    # artifacts so a different mounted run directory is supported.
    root = workspace_path.parent
    files = []
    def rebase(value):
        if isinstance(value, dict):
            return {key:rebase(item) for key,item in value.items()}
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, str) and value.endswith('.npy'):
            path = root/Path(value).name
            if not path.is_file():
                raise FileNotFoundError('Missing completed notebook 03 workspace array: '+str(path))
            files.append(dict(file=path.name, sha256=file_digest(path)))
            return str(path)
        return value
    workspace = rebase(json.loads(workspace_path.read_text()))
    model = load_model(workspace)
    return model.model, files


def learned_reference(run, grid):
    owner = run
    while not hasattr(owner, 'filter_manifest') and hasattr(owner, 'base_run'):
        owner = owner.base_run
    if not hasattr(owner, 'filter_manifest'):
        raise ValueError('Select the completed filtered notebook 03 ensemble first')
    path = owner.path('results', 'unbinned_scans.csv')
    fits_path = owner.path('results', 'unbinned_fits.csv')
    saved = pd.read_csv(path)
    saved = saved[(saved.model == LEARNED_LABEL) & np.isclose(saved.asimov_mu, run.config['asimov_mu'], atol=1e-12, rtol=0)]
    rows = []
    for syst in (False, True):
        for mu in grid:
            match = saved[(saved.systematics == syst) & np.isclose(saved.mu, mu, atol=1e-12, rtol=0)]
            if len(match) != 1:
                raise ValueError('The matching notebook 03 learned unbinned scan is missing/incomplete')
            record = match.iloc[0].to_dict()
            record.update(mu=mu, baseline_source='completed filtered notebook 03')
            rows.append(record)
    objective, artifacts = _load_objective(owner.path('workspaces', 'learned', 'workspace.json'))
    truth = float(run.config['asimov_mu'])
    fits = pd.read_csv(fits_path)
    fits = fits[fits.model == LEARNED_LABEL]
    current = float(objective(np.array([truth, 0.])))
    if len(fits) != 2 or not np.allclose(fits.nll_at_asimov, current, rtol=1e-8, atol=1e-6):
        raise ValueError('Saved learned workspace and notebook 03 fits disagree at the Asimov truth')
    curvature, error = objective_curvature(objective, truth)
    widths = {syst:dict(local_width(curvature if syst else curvature[:1, :1], systematics=syst),
                        curvature_step_relative_error=error) for syst in (False, True)}
    provenance = dict(model=LEARNED_LABEL, asimov_source='physical analytical Asimov from notebook 03',
        filter_identity=owner.filter_manifest['identity'], scan_file=str(path), scan_sha256=file_digest(path),
        fit_sha256=file_digest(fits_path), workspace_artifacts=artifacts,
        normalization='Notebook 03 frozen quadrature normalization; not the 10M histogram normalization',
        information='Half Hessian of the physical-Asimov objective at mu=truth, alpha=0; not coverage')
    return pd.DataFrame(rows), widths, provenance
