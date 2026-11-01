# Unity training for the notebook 03 ensembles

This workflow freezes the `paper-distinct-s-v4` generator and preselection from
Google Drive, generates new selected events on Unity, trains five independent
100-member ensembles, and publishes **inference models only** to GitHub. Colab
computes event weights again from those models on the existing Drive samples.
No generated events, per-event weights, optimizer states, or Lightning checkpoints
are returned to GitHub.

The frozen inputs are already in `unity/source/`: the analytic generator is
specified by `run.json`, and the saved selector and threshold are copied from
Drive. `provenance.json` records file IDs and SHA-256 hashes. The generation code
is the repository's `PhysicsModel`; there is no additional learned generator.
The original selector is reused, never retrained or retuned.

## Directories

- Checkout: `/project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo`
- Environment: `/project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo-env`
- Submission, generated banks, checkpoints and execution:
  `/work/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo/paper-distinct-s-v4-unity100`
- Publication checkout (created separately by the collector):
  `/project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo-publish-unity100`

Keep the source checkout unchanged during a campaign. Each job verifies its code
and frozen input identity. Commit source changes before initializing a production
campaign. Changed settings or code require a new campaign directory. Copy any
long-term training checkpoints you wish to retain out of `/work` according to the
cluster's retention policy; the scripts do not delete training outputs.

## One-time installation on Unity

```bash
cd /project/pi_rclsa_umass_edu/rclsa_umass_edu
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/rafaellopesdesa/poodemo.git
cd poodemo
module load conda/latest
conda create -y -c conda-forge --prefix /project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo-env python=3.11 pip git-lfs
conda activate /project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo-env
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r unity/requirements.txt -e '.[training]'
git lfs install --local
```

If the checkout already exists, update it instead of cloning it again. Use a CUDA
wheel compatible with the selected Unity GPU/driver. PyTorch, Lightning and the
NSBI toolkit revision are pinned in `unity/requirements.txt`; the remaining
package ranges are shared with the notebooks. Python must import `unity` and
`poodemo` from this checkout when invoked in `/work` (the editable install does
this). Check GPU availability inside an allocated GPU job, not on the login node.

## Campaign and resource configuration

`unity/config.json` defines the production experiment:

| Source | Selected training | Selected validation | Selected normalization |
| --- | ---: | ---: | ---: |
| S, SBI, B, NI, NI_up, NI_down (each) | 5,000,000 | 500,000 | 1,000,000 |

Thus there are 6.5 million selected events per source. Rejection sampling continues
until the requested accepted count is reached, using independent streams for
every source and split, in a separate namespace from notebook samples. No raw
rejected events are retained. These fixed-accepted-count banks are for training;
their acceptance fractions are **not** used to set physical yields. Notebook
quadrature still determines the rates in the existing inference construction.

The five tasks are SBI/S, B/S, NI/S, NI_up/NI and NI_down/NI. Each has 100 members
with independent initialization and minibatch seeds. They share the same training
bank; this averages optimization variability, not independent MC statistical
uncertainty. Each epoch draws five million events **with replacement per class**
from the entire bank, using the existing balanced sampler. Validation uses all
500,000 events per class. Architecture, BCE objective, NAdam, float32 precision,
100 epochs and the learning-rate schedule match notebook 03. The best validation
checkpoint is retained, with no early stopping.

There are 500 one-GPU jobs. The submission script defaults to `gpu`, 4 CPUs, 16 GB
host memory, 24 hours per job and a maximum of 20 simultaneous training jobs.
Generation is a six-element CPU array. A dependent five-element GPU array computes
one mean-normalization scalar per ensemble on the reserved Unity normalization
bank; it saves no event weights and fits no calibration map.

The 500 inference models contain roughly 4.2 GB of parameters. Training checkpoints
also contain optimizer state and need substantially more space (roughly 30 GB for
models plus best/last checkpoints, before filesystem overhead). Only the inference
files enter Git LFS. GitHub LFS storage/download usage applies to the repository
owner; repeated Colab downloads can be avoided with a persistent cache.

Run a pilot member to measure actual runtime before releasing the full array.
A full production job can be expensive: 100 epochs means roughly one million
optimizer steps per member. Increase `--time`, use `--qos=long` via a manually
adapted sbatch command if appropriate, or resubmit interrupted jobs to continue
from their last completed epoch. No production timing is assumed here.

## Initialize and submit

```bash
conda activate /project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo-env
export POODEMO_UNITY_RUN=/work/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo/paper-distinct-s-v4-unity100
mkdir -p "$POODEMO_UNITY_RUN"
cd "$POODEMO_UNITY_RUN"
python -m unity.workflow init --root "$POODEMO_UNITY_RUN"
python -m unity.submit --root "$POODEMO_UNITY_RUN" --dry-run
```

For a pilot, submit generation, then one member using the printed generation job ID:

```bash
python -m unity.submit --root "$POODEMO_UNITY_RUN" --stage generate
python -m unity.submit --root "$POODEMO_UNITY_RUN" --stage train --indices 0 --dependency GENERATION_JOB_ID
```

After the pilot, submit the full chain. Generation validates and reuses completed
banks, and the completed pilot model is reused. The final normalization array
waits for all training members to succeed:

```bash
python -m unity.submit --root "$POODEMO_UNITY_RUN" --concurrency 20
```

Override `--account`, `--gpu-partition`, `--cpu-partition`, `--constraint`,
`--time` and `--python` if needed. The default account is `pi_rclsa_umass_edu`.
The emitted `sbatch` commands are also available for manual adaptation. Commands
are submitted from the work directory and every job uses `--chdir` there.
The job scripts use `srun` and the absolute environment Python path.

Use `squeue -u "$USER"` and `sacct -j JOB_ID` to inspect progress. Logs are
`$POODEMO_UNITY_RUN/logs/{generate,train,normalize}_<job>_<index>.log`.
Array index is `task_index * 100 + member_index`, with tasks in the order above.

Jobs save `last.ckpt` after each epoch and resume optimizer, scheduler and epoch
state. The minibatch stream resumes at the correct epoch. A killed partial epoch
is repeated. Interrupted generation restarts only the incomplete split from its
original seed. Duplicate jobs for the same sample/member are rejected by a file
lock. Never delete completed receipts to force reuse with changed inputs.

For failed members, resubmit only their indices and then submit normalization
with a dependency on that retry array. Cancel any old normalization array left
waiting on a failed dependency first:

```bash
python -m unity.submit --root "$POODEMO_UNITY_RUN" --stage train --indices 4,109,302
python -m unity.submit --root "$POODEMO_UNITY_RUN" --stage normalize --dependency RETRY_JOB_ID
```

## Collect, commit, and push inference models

Configure your usual GitHub authentication and Git author identity on Unity.
Do this publication step from the login node after all jobs finish. It creates
an isolated worktree/branch, validates every required member and normalization
receipt, copies only a strict list of inference/metadata files, checks Git LFS
tracking, and commits the result. It does not alter `main`.

```bash
python -m unity.publish \
  --root "$POODEMO_UNITY_RUN" \
  --checkout /project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo-publish-unity100 \
  --push
```

Omit `--push` to create the local commit first. The branch is
`unity-models/paper-distinct-s-v4-unity100`. No partial ensemble is published;
a missing, corrupt, mismatched or unfinished member stops collection. A bundle
cannot overwrite an existing publication directory. After a failed upload, rerun
`git push` from the publication checkout; do not collect again. The commit SHA
printed by the script is the immutable identifier to use in Colab.

## Retest in notebook 03

Notebook 03 now has an optional Unity model-source cell. Until a production bundle
exists its default remains `RATIO_SOURCE = "drive"`.

```python
RATIO_SOURCE = "unity"
UNITY_BUNDLE = "paper-distinct-s-v4-unity100"
UNITY_REF = "FULL_40_CHARACTER_PUBLICATION_COMMIT_SHA"
```

Run the source cell and remaining notebook cells. The source helper uses a sparse
Git LFS checkout, checks every file hash, and confirms the generator, selector and
threshold match the Drive run. It never calls ratio training in Unity mode.
Inference loads one network at a time and averages **ratios**, not logits, without
allocating a members-by-events matrix. The saved scalar normalization is applied
without refitting it on the plotted samples. Per-event weights are calculated
in Colab and are not fetched from Unity.

Original selected events, diagnostic banks, independent Poisson draws and workspace
quadrature are reused. New outputs go under the original Drive run's
`evaluations/<bundle>/<campaign-id>/`, so the three-member results remain
available. Set `VALIDATE_UNITY_MEMBERS = True` to also plot all 500 individual
networks; otherwise only the five full ensembles are plotted. Reweighting,
calibration, independent closure and the usual three Asimov comparisons all use
the imported models. Existing ratio/residual axis limits are retained.

Ensembling may reduce fluctuation but cannot remove bias shared by the members.
The unchanged notebook evaluation statistics and quadrature can still limit
closure. Compare learned and analytic fits on the same events when assessing
improvement.

## Tests

```bash
python -m pytest -q tests/test_unity_workflow.py tests/test_training.py tests/test_ratio_training_sequence.py
```

The tiny real-training test uses the frozen selector, six small independent
banks, two members per task, normalization, collection, checksum verification,
streamed reload and a Drive evaluation view. It uses CPU and does not run the
production campaign or notebook likelihood scans.

Unity resource conventions:
https://docs.unity.rc.umass.edu/documentation/tools/gpus/
https://docs.unity.rc.umass.edu/documentation/jobs/slurm/

Git LFS:
https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage
