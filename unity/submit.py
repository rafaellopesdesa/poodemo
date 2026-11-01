"""Submit independent Slurm arrays; run from the project checkout."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

from unity.workflow import REPO, campaign

PROJECT = Path('/project/pi_rclsa_umass_edu/rclsa_umass_edu')
WORK = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu')


def commands(root, python, stage, concurrency, account, gpu_partition, cpu_partition,
             time, constraint, indices=None, dependency=None):
    m = campaign(root)
    total = 5 * m['config']['members']
    if indices and (stage == 'all' or not re.fullmatch(r'[0-9,\-]+', indices)):
        raise ValueError('Use --indices only with a single stage, e.g. 2,4,10-15')
    for current in (['generate', 'train', 'normalize'] if stage == 'all' else [stage]):
        cpu = current == 'generate'
        count = 6 if cpu else (total if current == 'train' else 5)
        array = indices or f'0-{count-1}'
        cmd = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
               '--mem=16G', '--requeue', f'--time={time}',
               f'--partition={cpu_partition if cpu else gpu_partition}',
               f'--array={array}%{min(concurrency, count)}', f'--chdir={root}',
               f'--job-name=poo-{current}', f'--output={root}/logs/{current}_%A_%a.log']
        if account:
            cmd.append(f'--account={account}')
        if not cpu:
            cmd.append('--gpus=1')
            if constraint:
                cmd.append(f'--constraint={constraint}')
        if dependency:
            cmd.append(f'--dependency=afterok:{dependency}')
        cmd += [str(REPO / 'unity/job.sh'), str(python), current, str(root)]
        yield current, cmd
        # Caller replaces this placeholder with the returned array job ID.
        dependency = f'<{current}-job-id>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=WORK / 'poodemo/paper-distinct-s-v4-unity100')
    p.add_argument('--python', type=Path, default=PROJECT / 'poodemo-env/bin/python')
    p.add_argument('--stage', choices=('all', 'generate', 'train', 'normalize'), default='all')
    p.add_argument('--concurrency', type=int, default=20)
    p.add_argument('--account', default='pi_rclsa_umass_edu')
    p.add_argument('--gpu-partition', default='gpu')
    p.add_argument('--cpu-partition', default='cpu')
    p.add_argument('--time', default='1-00:00:00')
    p.add_argument('--constraint', default='')
    p.add_argument('--indices')
    p.add_argument('--dependency')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.concurrency < 1:
        p.error('--concurrency must be positive')
    root = a.root.resolve()
    (root / 'logs').mkdir(exist_ok=True)
    jobs = {}
    previous = a.dependency
    for stage, cmd in commands(root, a.python.resolve(), a.stage, a.concurrency, a.account,
            a.gpu_partition, a.cpu_partition, a.time, a.constraint, a.indices, a.dependency):
        cmd = [f'--dependency=afterok:{previous}' if '<' in arg and arg.startswith('--dependency=')
               else arg for arg in cmd]
        print(shlex.join(cmd), flush=True)
        if a.dry_run:
            previous = f'<{stage}-job-id>'
            continue
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        job = result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError(f'Unexpected sbatch reply: {result.stdout}')
        jobs[stage] = previous = job
        print(f'{stage}: {job}', flush=True)
        # Persist each successful submission even if a later sbatch fails.
        (root / f'submission_{next(iter(jobs.values()))}.json').write_text(json.dumps(jobs, indent=2) + '\n')


if __name__ == '__main__':
    main()
