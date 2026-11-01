"""Frozen inputs -> selected iid banks -> independent members -> inference bundle.

Run `python -m unity.workflow --help`. All large/intermediate files stay in --root.
The six sources and five ratio orientations are imported from the notebook code.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np

from poodemo.data import SAMPLES, SAMPLE_STREAMS, file_digest, save_json
from poodemo.physics import PhysicsModel
from poodemo.pipeline import RATIO_TASKS
from poodemo.training import TrainingConfig, fit_density_ratio, load_predictor

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "unity/source"
SPLIT_STREAMS = {"train": 0, "validation": 1, "calibration": 2}
CODE_FILES = ("poodemo/physics.py", "poodemo/training.py", "poodemo/ratio_network.py",
              "poodemo/data.py", "poodemo/pipeline.py", "unity/workflow.py")


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Use a simple name containing letters, numbers, dots, dashes or underscores")
    return value


def source_identity(source):
    source = Path(source)
    physics = read(source / "run.json")["physics"]
    state = read(source / "selection_state.json")
    model_sha = file_digest(source / "preselection/model.pt")
    threshold = float(read(source / "preselection.json")["threshold"])
    fingerprint = hashlib.sha256((model_sha + repr(threshold)).encode()).hexdigest()
    if state != {"selection_fingerprint": fingerprint, "threshold": threshold}:
        raise ValueError("Frozen selector checkpoint/threshold do not match selection_state.json")
    return {"physics": physics, "selection": state, "selector_sha256": model_sha}


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process owns {path}") from None
        yield


def initialize(root, config, source=SOURCE):
    root, source = Path(root).resolve(), Path(source).resolve()
    cfg = read(config) if not isinstance(config, dict) else config
    safe_name(cfg["name"])
    for key in ("members", "seed", "generation_chunk", "generation_max_proposals", "prediction_batch_size"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if set(cfg["selected_counts"]) != set(SPLIT_STREAMS):
        raise ValueError("Provide independent train, validation, calibration counts")
    if any(type(n) is not int or n < 2 for n in cfg["selected_counts"].values()):
        raise ValueError("Each selected split needs at least two events")
    train_cfg = TrainingConfig(**cfg["training"])
    if train_cfg.ensemble_size != 1:
        raise ValueError("Each Slurm task trains one member; use the top-level members setting")
    identity = source_identity(source)
    code = {p: file_digest(REPO / p) for p in CODE_FILES}
    manifest = {"schema": 1, "config": cfg, "identity": identity, "code": code,
                "tasks": [list(t) for t in RATIO_TASKS], "samples": list(SAMPLES)}
    manifest["campaign_id"] = digest(manifest)
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / ".init.lock"):
        path = root / "campaign.json"
        if path.exists():
            if read(path) != manifest:
                raise ValueError("Campaign/source/code changed: choose a new work directory")
            return manifest
        if (root / "source").exists():
            raise ValueError("Incomplete initialization: choose a fresh work directory")
        shutil.copytree(source, root / "source")
        for folder in ("selected", "members", "normalization", "logs"):
            (root / folder).mkdir(exist_ok=True)
        save_json(path, manifest)
    return manifest


def campaign(root):
    root = Path(root)
    m = read(root / "campaign.json")
    body = {k: v for k, v in m.items() if k != "campaign_id"}
    if digest(body) != m["campaign_id"] or source_identity(root / "source") != m["identity"]:
        raise ValueError("Campaign inputs were modified")
    if any(file_digest(REPO / p) != sha for p, sha in m["code"].items()):
        raise ValueError("Training code changed since initialization; use the original checkout or a new campaign")
    return m


def selected_path(root, sample, split):
    return Path(root) / "selected" / f"{sample}_{split}.npy"


def checked_array(root, sample, split, m):
    path = selected_path(root, sample, split)
    receipt = read(path.with_suffix(".json"))
    expected_count = m["config"]["selected_counts"][split]
    if (receipt["campaign_id"] != m["campaign_id"] or receipt["sha256"] != file_digest(path)
            or receipt["sample"] != sample or receipt["split"] != split):
        raise ValueError(f"Selected sample provenance mismatch: {path}")
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.shape != (expected_count, 3) or x.dtype != np.float32:
        raise ValueError(f"Unexpected selected array: {path}")
    return x


def generate(root, sample, selector=None):
    root = Path(root)
    m = campaign(root)
    cfg = m["config"]
    component, alpha = SAMPLES[sample]
    model = PhysicsModel.from_dict(m["identity"]["physics"])
    selector = selector or load_predictor(root / "source/preselection", device="cpu")
    threshold = m["identity"]["selection"]["threshold"]
    with lock(root / "selected" / f".{sample}.lock"):
        for split, count in cfg["selected_counts"].items():
            path = selected_path(root, sample, split)
            if path.exists() and path.with_suffix(".json").exists():
                checked_array(root, sample, split, m)
                print(f"Reusing {sample}/{split}", flush=True)
                continue
            # A dedicated namespace and split streams never reuse notebook draws.
            seed = [cfg["seed"], 864209, SAMPLE_STREAMS[sample], SPLIT_STREAMS[split]]
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            temporary = path.with_suffix(".partial.npy")
            out = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(count, 3))
            stored = proposed = accepted = 0
            while stored < count:
                size = min(cfg["generation_chunk"], cfg["generation_max_proposals"] - proposed)
                if size <= 0:
                    raise RuntimeError(f"Proposal limit reached for {sample}/{split}; increase it in a new campaign")
                block = model.sample_component(component, size, rng, alpha_ni=alpha).astype(np.float32)
                keep = selector.predict_proba(block, batch_size=cfg["prediction_batch_size"])[:, 0] >= threshold
                chosen = block[keep]
                n = min(len(chosen), count - stored)
                out[stored:stored+n] = chosen[:n]
                stored += n
                proposed += size
                accepted += len(chosen)
            out.flush()
            del out
            os.replace(temporary, path)
            save_json(path.with_suffix(".json"), {"campaign_id": m["campaign_id"], "sample": sample,
                      "split": split, "selected": count, "proposed": proposed, "accepted_before_truncation": accepted,
                      "rng_seed_sequence": seed, "sha256": file_digest(path),
                      "note": "Fixed accepted count: do not use this stopping experiment to set physical yields"})
            print(f"{sample}/{split}: {count:,} selected, {proposed:,} proposals", flush=True)


def task_member(index, members):
    if not 0 <= index < len(RATIO_TASKS) * members:
        raise ValueError("Array index outside the campaign")
    task_index, member = divmod(index, members)
    return task_index, member, RATIO_TASKS[task_index]


def member_directory(root, num, den, member):
    return Path(root) / "members" / f"{num}_over_{den}" / f"member_{member:03d}"


def train(root, index):
    root = Path(root)
    m = campaign(root)
    cfg = m["config"]
    task_index, member, (num, den) = task_member(index, cfg["members"])
    directory = member_directory(root, num, den, member)
    seed = cfg["seed"] + 103 * task_index + 100003 * member
    config = replace(TrainingConfig(**cfg["training"]), seed=seed, ensemble_size=1)
    with lock(directory / ".train.lock"):
        expected = {"campaign_id": m["campaign_id"], "task_index": task_index, "member": member, "seed": seed}
        receipt = directory / "member.json"
        if receipt.exists() and read(receipt) != expected:
            raise ValueError("Member provenance mismatch")
        save_json(receipt, expected)
        arrays = {(p, split): checked_array(root, p, split, m)
                  for p in (num, den) for split in ("train", "validation")}
        # _fit resumes optimizer/scheduler/epoch state from last.ckpt and retains
        # the best-validation checkpoint after all scheduled epochs.
        predictor = fit_density_ratio(arrays[num, "train"], arrays[den, "train"],
            arrays[num, "validation"], arrays[den, "validation"], directory, config,
            numerator_label=num, denominator_label=den)
        if predictor.calibration is not None or predictor.log_normalization != 0.:
            raise ValueError("Unity members must remain raw; normalize only the assembled ensemble")
        metadata = read(directory / "training.json")
        if not metadata["complete"] or not np.isfinite(metadata["best_validation_loss"]):
            raise ValueError("Training did not produce a complete finite model")
        save_json(directory / "complete.json", {**expected, "model_sha256": file_digest(directory / "model.pt")})


def checked_member(root, m, task_index, member):
    num, den = RATIO_TASKS[task_index]
    directory = member_directory(root, num, den, member)
    receipt = read(directory / "complete.json")
    seed = m["config"]["seed"] + 103 * task_index + 100003 * member
    expected = {"campaign_id": m["campaign_id"], "task_index": task_index, "member": member,
                "seed": seed, "model_sha256": file_digest(directory / "model.pt")}
    training = read(directory / "training.json")
    if receipt != expected or not training["complete"] or training["labels"] != [den, num]:
        raise ValueError(f"Incomplete/mismatched member: {directory}")
    return directory


def normalize(root, task_index, device="cpu"):
    """One held-out scalar for the arithmetic ensemble; no per-event outputs."""
    from scipy.special import logsumexp
    root = Path(root)
    m = campaign(root)
    num, den = RATIO_TASKS[task_index]
    out = root / "normalization" / f"{num}_over_{den}.json"
    with lock(out.with_suffix(".lock")):
        x = checked_array(root, den, "calibration", m)
        total = np.full(len(x), -np.inf, dtype=np.float64)
        hashes = []
        for member in range(m["config"]["members"]):
            directory = checked_member(root, m, task_index, member)
            predictor = load_predictor(directory, device=device)
            np.logaddexp(total, predictor.predict_raw_log_ratio(x, m["config"]["prediction_batch_size"]), out=total)
            hashes.append(file_digest(directory / "model.pt"))
            del predictor
        total -= np.log(len(hashes))
        value = float(logsumexp(total) - np.log(len(x)))
        ess = float(np.exp(2 * logsumexp(total) - logsumexp(2 * total)))
        save_json(out, {"campaign_id": m["campaign_id"], "member_sha256": hashes,
                  "log_normalization": value, "n_calibration": len(x), "effective_sample_size": ess,
                  "relative_standard_error": float(np.sqrt(max(len(x) / ess - 1, 0) / (len(x) - 1))),
                  "note": "Independent Unity calibration bank; scalar mean normalization only, no fitted calibration map"})


def collect(root, destination):
    """Export only inference checkpoints and small metadata; never copy event arrays."""
    root, destination = Path(root), Path(destination)
    m = campaign(root)
    if destination.exists():
        raise FileExistsError(f"Choose a new bundle destination: {destination}")
    # Check the entire campaign before copying any model.
    tasks = {}
    for ti, (num, den) in enumerate(RATIO_TASKS):
        task = f"{num}_over_{den}"
        directories = [checked_member(root, m, ti, i) for i in range(m["config"]["members"])]
        norm = read(root / "normalization" / f"{task}.json")
        if norm["campaign_id"] != m["campaign_id"] or norm["member_sha256"] != [file_digest(p / "model.pt") for p in directories]:
            raise ValueError(f"Stale ensemble normalization for {task}")
        tasks[task] = (num, den, directories, norm)
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise FileExistsError(f"Remove incomplete staging directory before retrying: {staging}")
    staging.mkdir(parents=True)
    for task, (num, den, directories, norm) in tasks.items():
        target = staging / "ratios" / task
        for directory in directories:
            member = target / directory.name
            member.mkdir(parents=True)
            for name in ("model.pt", "training.json", "history.json", "member.json", "complete.json"):
                shutil.copy2(directory / name, member / name)
        save_json(target / "ensemble.json", {"members": [p.name for p in directories],
                  "labels": [den, num], "log_normalization": norm["log_normalization"], "streaming": True})
        save_json(target / "normalization.json", norm)
    save_json(staging / "campaign.json", m)
    hashes = {str(p.relative_to(staging)): file_digest(p) for p in sorted(staging.rglob("*")) if p.is_file()}
    save_json(staging / "bundle.json", {"schema": 1, "name": m["config"]["name"],
              "campaign_id": m["campaign_id"], "identity": m["identity"],
              "members_per_task": m["config"]["members"], "files": hashes})
    staging.rename(destination)
    print(f"Collected inference models to {destination}; no samples or event weights exported", flush=True)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("init", "generate", "train", "normalize", "collect"):
        s = sub.add_parser(command)
        s.add_argument("--root", type=Path, required=True)
        if command == "init":
            s.add_argument("--config", type=Path, default=REPO / "unity/config.json")
            s.add_argument("--source", type=Path, default=SOURCE)
        elif command == "generate":
            s.add_argument("--sample", choices=SAMPLES)
            s.add_argument("--index", type=int)
        elif command in ("train", "normalize"):
            s.add_argument("--index", type=int, required=True)
            if command == "normalize":
                s.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
        else:
            s.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    if a.command == "init":
        print(initialize(a.root, a.config, a.source)["campaign_id"])
    elif a.command == "generate":
        if (a.sample is None) == (a.index is None):
            p.error("Supply exactly one of --sample or --index")
        generate(a.root, a.sample if a.sample is not None else list(SAMPLES)[a.index])
    elif a.command == "train":
        train(a.root, a.index)
    elif a.command == "normalize":
        normalize(a.root, a.index, a.device)
    else:
        collect(a.root, a.destination)


if __name__ == "__main__":
    main()
