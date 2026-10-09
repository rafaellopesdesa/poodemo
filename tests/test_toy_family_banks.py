"""Nested full-family grids on an authenticated exact replay of existing banks."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_toy_convergence_banks import _context, _original, _hashes, GuardedModel
from poodemo.inference import component_coefficients
from poodemo.toy_convergence_banks import prepare_convergence_banks
from poodemo.toy_family_banks import (
    _pilot_edges, family_coordinates, family_bin_indices, prepare_family_banks,
)


def _prepared(path):
    context = _context(path)
    _original(context)
    prepare_convergence_banks(context, extra_multiples=(10,))
    context.pop("workspace")  # Replay must never need density-ratio inference.
    return context


def test_raw_coordinates_preserve_family_intensity_ratios_and_bins_cover_tails():
    rng = np.random.default_rng(881)
    fields = rng.uniform(.1, 8., size=(4, 100))
    # Set coherent interference to respect positivity for all kappa.
    fields[1] = fields[0] + fields[2] - .6 * np.sqrt(fields[0] * fields[2])
    u, v = family_coordinates(fields)
    denominator = fields[2] + fields[3]
    for mu in (0., .03, 1., 1.4, 3.):
        np.testing.assert_allclose(component_coefficients(mu) @ fields,
            denominator * (1. + np.sqrt(mu) * u + mu * v), rtol=1e-13)
    edges = (np.array([-np.inf, -.1, np.inf]), np.array([-np.inf, 1., np.inf]))
    indices = family_bin_indices(fields, edges)
    assert indices.min() >= 0 and indices.max() < 4
    np.testing.assert_array_equal(indices, (u >= -.1) * 2 + (v >= 1.))
    assert family_bin_indices(np.empty((4, 0)), edges).size == 0
    with pytest.raises(ValueError, match="positive B\\+NI"):
        family_coordinates(np.zeros((4, 1)))
    with pytest.raises(ValueError, match="finite"):
        family_coordinates(np.full((4, 1), np.nan))
    with pytest.raises(ValueError, match="infinite exterior"):
        family_bin_indices(fields, ([-1., 0., 1.], [-1., 0., 1.]))


def test_quantiles_are_frozen_nested_and_ties_are_explicit(tmp_path):
    context = _context(tmp_path)
    edges, digest = _pilot_edges(context, (12, 36))
    assert len(digest) == 64
    for axis in (0, 1):
        np.testing.assert_array_equal(edges[12][axis], edges[36][axis][::3])
        assert np.all(np.diff(edges[36][axis]) > 0)
    fields = context["quad"]["nominal"]
    coarse = family_bin_indices(fields, edges[12])
    fine = family_bin_indices(fields, edges[36])
    np.testing.assert_array_equal(coarse, (fine // 36 // 3) * 12 + ((fine % 36) // 3))
    context["quad"] = dict(nominal=np.ones((4, 100)), weights=np.ones(100))
    with pytest.raises(ValueError, match="tied boundaries"):
        _pilot_edges(context, (12, 36))
    with pytest.raises(ValueError, match="divide"):
        _pilot_edges(context, (12, 35))


def test_exact_replay_matches_all_prefixes_and_preserves_sources(tmp_path):
    context = _prepared(tmp_path)
    original_hashes = _hashes(tmp_path / "integration")
    convergence_hashes = _hashes(tmp_path / "convergence/bank_convergence/integration")
    study = prepare_family_banks(context)
    assert set(study["banks"]) == {(r, n) for r in (0, 1) for n in (400, 800)}
    assert set(study["convergence_banks"]) == set(study["banks"])
    assert len(study["closure"]) == 2 * 4 * 3
    assert study["closure"].moments_reproduced.all()
    assert study["closure"].rng_reproduced.all()
    assert study["closure"].selected_reproduced.all()
    assert set(study["closure"].generated_per_source) == {80, 160, 400, 800}
    for key, bank in study["banks"].items():
        source = study["convergence_banks"][key]
        np.testing.assert_array_equal(bank["rates"], source["rates"])
        assert bank["metadata"]["convergence_bank_fingerprint"] == source["metadata"]["fingerprint"]
        for n, prediction in bank["templates"].items():
            np.testing.assert_allclose(prediction.sum(axis=1), bank["rates"], rtol=1e-13)
            assert bank["occupancy"][n].sum() == source["metadata"]["selected"]
            assert bank["component_covariances"][n].shape == (4, 4, n * n)
            # Empty cells remain explicit zero support, without pseudocounts.
            empty = bank["occupancy"][n] == 0
            assert np.any(empty)
            np.testing.assert_array_equal(prediction[:, empty], 0.)
        np.testing.assert_allclose(bank["templates"][36].reshape(4, 12, 3, 12, 3).sum(axis=(2, 4)).reshape(4, 144),
                                   bank["templates"][12], rtol=1e-13, atol=1e-15)
    support = study["support"]
    assert support.empty_mc.any()
    assert support.loc[support.empty_mc, "relative_mc_se"].isna().all()
    assert support.loc[~support.empty_mc, "mc_se"].ge(0.).all()
    assert _hashes(tmp_path / "integration") == original_hashes
    assert _hashes(tmp_path / "convergence/bank_convergence/integration") == convergence_hashes
    context["run"].model = GuardedModel(context["run"].model, fail_after=0)
    repeated = prepare_family_banks(context)
    assert context["run"].model.calls == 0
    for key in study["banks"]:
        for n in (12, 36):
            np.testing.assert_array_equal(study["banks"][key]["templates"][n], repeated["banks"][key]["templates"][n])
    pd.testing.assert_frame_equal(study["support"], repeated["support"])


def test_interrupted_replay_resumes_the_original_chunk_schedule(tmp_path):
    context = _prepared(tmp_path / "resume")
    model = context["run"].model
    context["run"].model = GuardedModel(model, fail_after=5)
    with pytest.raises(RuntimeError, match="Injected"):
        prepare_family_banks(context)
    work = context["output_dir"] / "convergence/bank_convergence/family/family_control/integration/replica_0_S_work.npz"
    with np.load(work) as saved:
        # Prefix80 takes73+7, prefix160 takes73+7; thenone73chunk.
        assert int(saved["completed"]) == 233
    context["run"].model = GuardedModel(model)
    resumed = prepare_family_banks(context)
    fresh = _prepared(tmp_path / "fresh")
    uninterrupted = prepare_family_banks(fresh)
    for key in resumed["banks"]:
        for n in (12, 36):
            for field in ("templates", "component_covariances", "occupancy"):
                np.testing.assert_array_equal(resumed["banks"][key][field][n], uninterrupted["banks"][key][field][n])


def test_frozen_inputs_and_saved_payload_changes_are_rejected(tmp_path):
    context = _prepared(tmp_path)
    prepare_family_banks(context)
    with pytest.raises(ValueError, match="family_tag"):
        prepare_family_banks(context, axis_bins=(12,))
    with pytest.raises(ValueError, match="family_tag"):
        prepare_family_banks(context, family_tag="../bad")
    with pytest.raises(ValueError, match="largest_prefixes"):
        prepare_family_banks(context, largest_prefixes=5)
    context["quad"]["weights"] = context["quad"]["weights"] * 1.0001
    with pytest.raises(ValueError, match="saved inputs changed"):
        prepare_family_banks(context)
    context["quad"]["weights"] /= 1.0001
    # Float roundtrip in pilotweights may not be bitexact; use a separate tag.
    prepare_family_banks(context, family_tag="integrity")
    path = tmp_path / "convergence/bank_convergence/family/integrity/integration/replica_0_S_80.npz"
    with np.load(path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays["family_sum_12"][0, 0] += 1.
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="payload changed"):
        prepare_family_banks(context, family_tag="integrity")


def test_selector_drift_fails_exact_bank_reproduction(tmp_path):
    context = _prepared(tmp_path)
    context["selector"] = None
    with pytest.raises(ValueError, match="Exact-bank replay failed"):
        prepare_family_banks(context)


def test_cell_mc_errors_use_fixed_generated_strata_and_component_covariance(tmp_path):
    context = _prepared(tmp_path)
    study = prepare_family_banks(context)
    bank = study["banks"][0, 800]
    mu, n_axis, exposure = 1.4, 12, context["configuration"]["exposure_multiplier"]
    coefficients = component_coefficients(mu)
    expected_variance = np.zeros(n_axis**2)
    expected_prediction = np.zeros(n_axis**2)
    # Independent direct calculation: materialize one physical contribution
    # per GENERATED point, including zeros for selection and cell rejection.
    for source_id, source in enumerate(("S", "B", "NI")):
        rng = np.random.default_rng(np.random.SeedSequence([82721, 120524, 0, source_id]))
        draws, count = [], 0
        for size in (80, 160, 400, 800):
            while count < size:
                take = min(73, size - count)
                draws.append(context["run"].model.sample_component(source, take, rng))
                count += take
        x = np.concatenate(draws)
        selected = context["selector"].predict_proba(x)[:, 0] >= context["threshold"]
        selected_rows = np.flatnonzero(selected)
        fields = context["run"].model.component_densities(x[selected]).T
        mixture_sum = sum(context["run"].model.component_pdf(x[selected], p) for p in ("S", "B", "NI"))
        physical_weights = exposure * (coefficients @ fields) / mixture_sum
        cells = family_bin_indices(fields, bank["edges"][n_axis])
        for cell in range(n_axis**2):
            contributions = np.zeros(800)
            mask = cells == cell
            contributions[selected_rows[mask]] = physical_weights[mask]
            expected_prediction[cell] += contributions.mean()
            expected_variance[cell] += contributions.var(ddof=1) / 800
    reported = study["support"].query("replica == 0 and generated_per_source == 800 and n_axis == 12 and mu == 1.4")
    np.testing.assert_allclose(reported.predicted_yield, expected_prediction, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(reported.mc_se**2, expected_variance, rtol=1e-11, atol=1e-14)
