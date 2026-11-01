"""Import immutable Unity inference bundles without moving event samples."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .data import Run, file_digest, save_json
from .pipeline import RATIO_TASKS


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'bundle.json').read_text())
    if manifest['schema'] != 1:
        raise ValueError('Unsupported ensemble bundle schema')
    for name, expected in manifest['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f'Missing or invalid bundle file: {name}')
        if file_digest(path) != expected:
            raise ValueError(f'Bundle checksum mismatch: {name}; ensure Git LFS files are downloaded')
    for num, den in RATIO_TASKS:
        task = directory / 'ratios' / f'{num}_over_{den}'
        info = json.loads((task / 'ensemble.json').read_text())
        if (len(info['members']) != manifest['members_per_task'] or
                len(set(info['members'])) != len(info['members']) or info['labels'] != [den, num]):
            raise ValueError(f'Invalid ensemble membership: {task}')
        for member in info['members']:
            model_path = (task / member / 'model.pt').resolve()
            if (not model_path.is_relative_to(task) or
                    str(model_path.relative_to(directory)) not in manifest['files']):
                raise ValueError('Ensemble refers to an unverified model')
    return manifest


def fetch_bundle(name, ref, cache, repository='https://github.com/rafaellopesdesa/poodemo.git'):
    """Sparse Git LFS checkout pinned to the publication commit printed on Unity."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Invalid bundle name')
    if not re.fullmatch(r'[0-9a-f]{40}', ref):
        raise ValueError('Set UNITY_REF to the full commit SHA printed by unity.publish')
    checkout = Path(cache).expanduser().resolve() / ref
    relative = f'unity/ensembles/{name}'
    directory = checkout / relative
    if directory.exists() and (checkout / '.download-complete').exists():
        verify_bundle(directory)
        return directory
    subprocess.run(['git', 'lfs', 'version'], check=True, capture_output=True)
    checkout.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1'}

    def git(*args):
        return subprocess.run(['git', '-C', str(checkout), *args], check=True, env=env)

    if not (checkout / '.git').exists():
        git('init')
        git('remote', 'add', 'origin', repository)
    git('lfs', 'install', '--local')
    git('fetch', '--depth=1', 'origin', ref)
    git('sparse-checkout', 'init', '--cone')
    git('sparse-checkout', 'set', relative)
    git('checkout', '--detach', ref)
    git('lfs', 'pull', 'origin', '--include', relative + '/**', '--exclude', '')
    verify_bundle(directory)
    (checkout / '.download-complete').write_text(ref + '\n')
    return directory


class BundleEvaluationRun(Run):
    """Read Drive events/selection/quadrature, use imported models, isolate results."""
    def __init__(self, base_run, bundle, manifest):
        super().__init__(base_run.root, base_run.config, base_run.model)
        self.base_run = base_run
        self.bundle = Path(bundle).resolve()
        self.bundle_manifest = manifest
        self.evaluation_root = base_run.path('evaluations', manifest['name'], manifest['campaign_id'][:16])
        for folder in ('results', 'plots', 'workspaces'):
            (self.evaluation_root / folder).mkdir(parents=True, exist_ok=True)
        save_json(self.evaluation_root / 'bundle.json', manifest)

    def path(self, *parts):
        if parts[:2] == ('models', 'ratios'):
            return self.bundle.joinpath('ratios', *parts[2:])
        # These are frozen input metadata / independent cached event banks.
        if parts[:2] in (('results', 'preselection.json'), ('results', 'density_validation')):
            return self.base_run.path(*parts)
        if parts and parts[0] in ('results', 'plots', 'workspaces'):
            return self.evaluation_root.joinpath(*parts)
        return self.base_run.path(*parts)


def use_bundle(run, directory):
    if isinstance(run, BundleEvaluationRun):
        run = run.base_run
    manifest = verify_bundle(directory)
    identity = manifest['identity']
    state = json.loads(run.path('selected', 'selection_state.json').read_text())
    model_sha = file_digest(run.path('models', 'preselection', 'model.pt'))
    threshold = float(json.loads(run.path('results', 'preselection.json').read_text())['threshold'])
    fingerprint = hashlib.sha256((model_sha + repr(threshold)).encode()).hexdigest()
    if (identity['physics'] != run.model.to_dict() or identity['selection'] != state or
            identity['selector_sha256'] != model_sha or state['selection_fingerprint'] != fingerprint):
        raise ValueError('Unity bundle and Drive run have different physics, selector, or threshold')
    return BundleEvaluationRun(run, directory, manifest)


def validate_bundle_ratios(run, samples, *, on_validation=None, validation_options=None,
                          validate_members=False):
    """Evaluate imported models. Never train, recalibrate, or renormalize them."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from .ratio_validation import plot_ratio_task
    from .training import load_predictor, Predictor
    records = []
    for num, den in RATIO_TASKS:
        task = f'{num}_over_{den}'
        predictor = load_predictor(run.path('models', 'ratios', task))
        paths = predictor.member_paths if validate_members else []
        for index in range(len(paths) + 1):
            is_member = index < len(paths)
            current = Predictor.load(paths[index]) if is_member else predictor
            stage = f'member_{index+1:03d}' if is_member else 'ensemble'
            report = plot_ratio_task(num, den, samples, current,
                run.path('plots', '03_validation', task, stage),
                results=run.path('results', '03_validation', task, stage),
                stages=('raw',) if is_member else ('raw', 'normalized'),
                label=stage, **(validation_options or {}))
            report.update(task=task, ratio=f'{num}/{den}', stage=stage,
                          member_index=index if is_member else None,
                          total_members=len(predictor.member_paths))
            if on_validation:
                on_validation(report)
            for figure in report['figures'].values():
                plt.close(figure)
            del current
        record = json.loads((predictor.output_dir / 'normalization.json').read_text())
        record.pop('member_sha256', None)
        records.append({'ratio': f'{num}/{den}', 'members': len(predictor.member_paths), **record})
        del predictor
    frame = pd.DataFrame(records)
    frame.to_csv(run.path('results', 'ratio_diagnostics.csv'), index=False)
    return frame
