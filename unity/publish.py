"""Collect models into an isolated Git LFS branch, commit, and optionally push."""
import argparse
from pathlib import Path
import subprocess

from unity.workflow import REPO, campaign, collect, safe_name


def git(repo, *args, capture=False):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          capture_output=capture, text=True)


def publish(root, checkout, *, push=False, remote='origin'):
    root, checkout = Path(root).resolve(), Path(checkout).resolve()
    m = campaign(root)
    name = safe_name(m['config']['name'])
    branch = f'unity-models/{name}'
    git(REPO, 'lfs', 'version')
    if git(REPO, 'status', '--porcelain', capture=True).stdout:
        raise ValueError('Commit source changes before publishing models')
    if checkout.exists():
        raise FileExistsError('Use a fresh publication checkout; an existing bundle is never overwritten')
    git(REPO, 'worktree', 'add', '-b', branch, str(checkout), 'HEAD')
    git(checkout, 'lfs', 'install', '--local')
    relative = Path('unity/ensembles') / name
    collect(root, checkout / relative)
    git(checkout, 'add', '--', str(relative))
    files = git(checkout, 'lfs', 'ls-files', '--name-only', capture=True).stdout.splitlines()
    expected = {str(p.relative_to(checkout)) for p in (checkout / relative).rglob('model.pt')}
    if not expected.issubset(files):
        raise RuntimeError('Inference checkpoints were not staged as Git LFS objects')
    git(checkout, 'diff', '--cached', '--stat')
    git(checkout, 'commit', '-m', f'Add {name}: five {m["config"]["members"]}-member inference ensembles')
    sha = git(checkout, 'rev-parse', 'HEAD', capture=True).stdout.strip()
    if push:
        # The git-lfs pre-push hook uploads objects before the branch ref moves.
        git(checkout, 'push', '-u', remote, branch)
    print(f'Bundle: {relative}\nCommit: {sha}\nBranch: {branch}\nCheckout: {checkout}')
    print(f'Notebook: UNITY_BUNDLE = "{name}"; UNITY_REF = "{sha}"')
    if not push:
        print(f'Publish with: git -C {checkout} push -u {remote} {branch}')
    return sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--push', action='store_true')
    p.add_argument('--remote', default='origin')
    a = p.parse_args()
    publish(a.root, a.checkout, push=a.push, remote=a.remote)


if __name__ == '__main__':
    main()
