# Unity training for the notebook 03 ensembles

This workflow freezes the `paper-distinct-s-v4` generator and preselection from
Google Drive, generates new selected events on Unity, trains five independent
100-member ensembles, and collects **inference models only** for transfer to Google Drive. Colab
computes event weights again from those models on the existing Drive samples.
No generated events, per-event weights, optimizer states, or Lightning checkpoints
are included in the transfer.

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
- Collected model directory for SCP (created by `collect`):
  `/project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo-models/paper-distinct-s-v4-unity100`
- Google Drive destination: `My Drive/poodemo/unity_models/paper-distinct-s-v4-unity100/`

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
conda create -y -c conda-forge --prefix /project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo-env python=3.11 pip
conda activate /project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo-env
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r unity/requirements.txt -e '.[training]'
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
models plus best/last checkpoints, before filesystem overhead). Only inference
models and small metadata files enter the collected directory transferred to Drive.
The notebook loads them directly from the mounted Drive directory.

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

## Collect the inference models for SCP

After **all training and normalization jobs finish**, run the existing collector
with the same source checkout used for training. This does not require a GitHub
commit or upload. It verifies every member and normalization receipt and copies
only inference models and small metadata files; samples and optimizer checkpoints
remain in the work directory.

```bash
python -m unity.workflow collect \
  --root /work/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo/paper-distinct-s-v4-unity100 \
  --destination /project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo-models/paper-distinct-s-v4-unity100
```

The collector refuses missing, mismatched or unfinished members and does not
overwrite an existing destination. The entire collected directory is about
4.2 GB with the default architecture. Keep its layout unchanged during transfer:

| Relative path inside `paper-distinct-s-v4-unity100/` | Contents |
| --- | --- |
| `bundle.json` | File checksums and model/selection identity |
| `campaign.json` | Frozen training configuration and provenance |
| `ratios/<task>/ensemble.json` | Member list and ensemble normalization |
| `ratios/<task>/normalization.json` | Normalization metadata |
| `ratios/<task>/member_000/` through `member_099/` | `model.pt` and member metadata |

The five `<task>` names are `SBI_over_S`, `B_over_S`, `NI_over_S`,
`NI_up_over_NI` and `NI_down_over_NI`. Copy all files produced by the collector.

From your computer, create the `poodemo/unity_models` folder inside your locally
synced Google Drive, then use SCP to copy the collected campaign folder into it.
Replace the SSH alias and local Drive path in this example:

```bash
scp -r YOUR_UNITY_SSH_ALIAS:/project/pi_rclsa_umass_edu/rclsa_umass_edu/poodemo-models/paper-distinct-s-v4-unity100 \
  /path/to/GoogleDrive/MyDrive/poodemo/unity_models/
```

Alternatively, SCP to a local staging folder and upload that campaign folder to
Drive. Google Drive itself is not an SCP endpoint. Wait for synchronization/upload
to finish before running notebook 03. Avoid adding an extra enclosing directory:
`bundle.json` must be directly inside the campaign folder at the destination.

## Retest in notebook 03

Notebook 03 now defaults to loading the transferred Unity models:

```python
RATIO_SOURCE = "unity"
UNITY_MODEL_DIR = Path(
    "/content/drive/MyDrive/poodemo/unity_models/paper-distinct-s-v4-unity100"
)
```

Change `UNITY_MODEL_DIR` in the notebook (or set `POODEMO_UNITY_MODEL_DIR`) for a
different location. On a local Jupyter installation, set it to the local path of
the same collected directory. No Git LFS installation, GitHub model branch or
commit SHA is needed. The previous Git LFS publication/download helpers remain
available for optional use, but the notebook does not call them.

Run the source cell and remaining notebook cells. The loader checks every file
hash and confirms the generator, selector and threshold match the Drive run.
A missing or incomplete transfer fails explicitly; it never falls back to training.
Inference loads one network at a time and averages **ratios**, not logits.
Per-event weights are calculated in Colab and are not fetched from Unity.

### Member screening before validation and fits

Notebook 03 screens **all five ensembles**, including SBI/S, B/S, NI/S,
NI_up/NI and NI_down/NI. It rejects whole member trainings; it does not clip
individual event weights, fit a calibration map, select on fitted parameters,
or use analytic target ratios to decide membership. The original checkpoints
and transferred bundle stay intact. The original Drive workflow also uses this
screening after its members have been trained/reused.

Each member is evaluated on the **full frozen integration bank**, including its
sparsely populated tails, and the reserved Drive denominator calibration split.
Nominal tasks use the signal integration measure. Systematic tasks use the
**retained learned NI shape** on those same nodes, so screening checks the actual
product of ratios used by the workspace. The diagnostic and independent Poisson
banks never determine membership or scalar normalization. Because integration
nodes now participate in member selection, they are not an independent test of
that selection; the held-out diagnostics remain essential.

The defaults in `MemberFilterPolicy` are deliberately loose numerical-quality
cuts, not statistical confidence levels:

| Check | Default rejection condition |
| --- | --- |
| Numeric output | Nonfinite prediction, or log ratio outside the inference range $[-700,700]$ |
| Calibration mean | Raw deployed member mean outside $[1/5,5]$ on reserved calibration events |
| Integration consistency | Integration mean divided by calibration mean outside $[1/5,5]$ |
| Single-point concentration | More than 1% of normalized integration or calibration mass at one point |
| Effective sample size | Less than 0.1% of the bank size on integration or calibration events |
| Peer disagreement | Log-ratio departure exceeds both $\log(100)$ and 8 robust standard deviations, on more than 0.1% of member or peer integration mass |
| Extreme pointwise disagreement | Departure exceeds both $\log(10^6)$ and 8 robust standard deviations at any integration point |

Peer comparison uses the pointwise median and $1.4826\,\mathrm{MAD}$ among
members that pass the preceding checks. Members are centered using their reserved
calibration means **for this comparison only**. Inference still averages the
original retained ratios arithmetically and applies one ensemble scalar. A
negligible tail discrepancy alone is not sufficient for ordinary peer rejection.
The separate six-orders-of-magnitude criterion catches extreme extrapolation
even at very low-weight nodes, because the nuisance polynomial must remain
physical at those nodes too.

A strict majority must survive, with at least two retained members when the
original ensemble has two or more. Widespread failures stop the audit with an
explicit report; the code does not keep pruning until a fit succeeds. The retained
ensemble must also pass the integral/concentration checks. Finally, the NI
interpolation is checked over a grid in $[-1,1]$, including tiny displacements
around zero. A remaining invalid morph stops evaluation. This grid is a preflight
check, not a proof between grid points; the likelihood's domain checks remain.

Configure thresholds in the notebook source cell, for example:

```python
from poodemo.member_filter import MemberFilterPolicy
ENSEMBLE_FILTER_POLICY = MemberFilterPolicy(
    max_integral_factor=5.0,
    max_point_mass=0.01,
    min_effective_fraction=0.001,
    peer_factor=100.0,
    catastrophic_peer_factor=1.e6,
    peer_mad_scale=8.0,
    peer_outlier_mass=0.001,
    min_retained_fraction=0.5,
)
```

Do not tune these thresholds to obtain a preferred likelihood result. A shared
bias among members may pass this filter, and an otherwise legitimate but poorly
sampled tail can fail the numerical-quality checks. Inspect reported failures
and independent closure before changing a policy.

After selection, the retained ensemble's mean-normalization scalar is recomputed
on the original **Drive calibration split** (`selected/S_calibration.npy` or
`selected/NI_calibration.npy`), since the Unity scalar belonged to a different
membership. No Unity calibration events need to be transferred. The usual frozen
workspace quadrature normalization is subsequently applied as before.

The first audit adds inference over both banks for every member. It uses a
temporary disk-backed log-prediction matrix in the Colab runtime, approximately
360 MB for 100 members and 451,000 integration points, and removes it after each
task. No large prediction matrix is written to Drive. Complete audits are reused
when checkpoint/calibrator hashes, integration and calibration files, policy,
and implementation match. A changed input creates a new filter identity.

For Unity, outputs are under:

```text
runs/paper-distinct-s-v4/evaluations/<bundle>/<campaign-id>/filtered/<filter-id>/
```

The notebook prints this path and the retained/rejected counts. In that directory:

- `results/ensemble_filter/members.csv`: every member, metrics, decision, and reasons.
- `results/ensemble_filter/summary.csv`: counts and retained-ensemble metrics for all tasks.
- `results/ensemble_filter/manifest.json`: source hashes, policy, calibration provenance,
  original member paths, retained/rejected lists, and nuisance preflight result.
- `models/ratios/<task>/ensemble.json`: relative references to the retained original
  checkpoints, with the new normalization. These are views, not copied model files.

All subsequent notebook-03 reweighting, calibration, architecture, independent
closure, and Asimov fits read these same filtered manifests. Set
`VALIDATE_UNITY_MEMBERS = True` to plot each retained member as well. Existing
ratio/residual axis limits are retained. Previous unfiltered outputs are isolated
from this run, and stale notebook outputs were cleared.

Use a **fresh Colab runtime** (or refresh `/content/poodemo` and restart it) so the
new Python module is loaded, then run notebook 03 from the top. Existing Drive
samples and models are reused. For the original three-member workflow, explicitly
set `RATIO_SOURCE = "drive"`; derived outputs live under
`runs/<run>/evaluations/filtered/<filter-id>/`.

This update leaves all files hashed by the Unity campaign unchanged. Existing
training/generation jobs and the collector need no update or retraining.

## Tests

```bash
python -m pytest -q tests/test_member_filter.py tests/test_notebook_unity_directory.py tests/test_closure_validation.py
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
