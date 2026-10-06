#!/usr/bin/env python3
"""Validate the analytic interference benchmark before expensive training.

This uses the ideal balanced S/B/NI classifier, not a trained network. The
default cut was calibrated separately on 32,768 Sobol S/NI draws using
scramble seeds 100/102. Use --seed 1011 --power 16 for independent integration
with that cut frozen. All nuisance profiles use the notebooks' normalized
rate/shape exp-poly TemplateLikelihood, including the unit Gaussian penalty.

Example, from the repository root after installing poodemo:
    python scripts/validate_benchmark.py --output benchmark_validation.json

The local widths are expected-Fisher diagnostics. This calculation does not
measure finite-sample confidence-set coverage.
"""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, ndtri
from scipy.stats import qmc

from poodemo.inference import TemplateLikelihood
from poodemo.physics import PhysicsModel


def quadrature(physics, power, seed, threshold):
    blocks = []
    for offset, name in enumerate(("S", "B", "NI")):
        uniform = qmc.Sobol(3, scramble=True, seed=seed + offset).random_base2(power)
        normal = ndtri(uniform)
        covariance = np.asarray(getattr(physics, "cov_" + name.lower()))
        blocks.append(normal @ np.linalg.cholesky(covariance).T + physics.mean(name))
    x = np.concatenate(blocks)
    pdfs = np.stack([physics.component_pdf(x, name) for name in ("S", "B", "NI")])
    weights = 1 / (len(x) * pdfs.mean(axis=0))
    score = pdfs[0] / pdfs.sum(axis=0)
    keep = score > threshold
    x, weights = x[keep], weights[keep]
    nominal = physics.component_densities(x).T
    down = physics.component_densities(x, alpha_ni=-1).T[None]
    up = physics.component_densities(x, alpha_ni=1).T[None]
    model = TemplateLikelihood(nominal, down, up, weights, nominal[1] + nominal[3])
    return model


def extrema(mus, values, objective):
    extrema = {"minima": [], "maxima": []}
    for kind, sign in (("minima", 1), ("maxima", -1)):
        oriented = sign * values
        for index in range(1, len(mus) - 1):
            if oriented[index] < min(oriented[index - 1], oriented[index + 1]):
                fit = minimize_scalar(lambda mu: sign * objective(mu), method="bounded",
                                      bounds=(mus[index - 1], mus[index + 1]),
                                      options={"xatol": 1e-7})
                if not fit.success:
                    raise RuntimeError("Could not refine an interference extremum.")
                extrema[kind].append({"mu": float(fit.x), "q": float(sign * fit.fun)})
    return extrema


def local_widths(model):
    signal, sbi, background, ni = model.nominal
    derivative_mu = signal + (sbi - signal - background) / 2
    step = 1e-4
    derivative_alpha = (model.intensity(1, [step]) - model.intensity(1, [-step])) / (2 * step)
    weighted = model.quadrature_weights / (sbi + ni)
    f_mm = np.sum(weighted * derivative_mu**2)
    f_ma = np.sum(weighted * derivative_mu * derivative_alpha)
    f_aa = 1 + np.sum(weighted * derivative_alpha**2)
    fixed = 1 / np.sqrt(f_mm)
    profiled = 1 / np.sqrt(f_mm - f_ma**2 / f_aa)
    return {"sigma_mu_fixed": float(fixed), "sigma_mu_profiled": float(profiled),
            "profiled_over_fixed": float(profiled / fixed)}


def score_resolution(model, mus, scale):
    signal, sbi, background, ni = model.nominal
    interference = sbi - signal - background
    weights = model.quadrature_weights
    retained = {count: [] for count in (64, 128, 256, 512)}
    summary = []
    inspect_etas = (.1, 1., 3.5, 7.2, 10., 12.)
    for eta in np.unique(np.r_[mus, inspect_etas]):
        density = eta * signal + np.sqrt(eta) * interference + background + ni
        derivative = signal + interference / (2 * np.sqrt(eta))
        rate = density @ weights
        score = derivative / density - (derivative @ weights) / rate
        full_information = np.sum(weights * derivative**2 / density)
        bin_index = np.minimum((512 * expit(score / scale)).astype(int), 511)
        for count in retained:
            index = bin_index // (512 // count)
            means = np.bincount(index, weights=weights * density, minlength=count)
            gradient = np.bincount(index, weights=weights * derivative, minlength=count)
            information = np.sum(np.divide(gradient**2, means,
                                           out=np.zeros(count), where=means > 0))
            retained[count].append(float(information / full_information))
        if eta in inspect_etas:
            order = np.argsort(score)
            probability = np.cumsum(weights[order] * density[order]) / rate
            quantiles = np.interp([.01, .5, .99], probability, score[order])
            summary.append({"eta": float(eta),
                            "rms": float(np.sqrt(np.sum(weights * density * score**2) / rate)),
                            "quantiles_01_50_99": quantiles.tolist()})
    return {"score_scale": scale, "summary": summary,
            "worst_extended_fisher_retention": {str(k): min(v) for k, v in retained.items()}}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=int, default=15, help="Sobol draws per process are 2**power.")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=.30816816816816817)
    parser.add_argument("--mu-max", type=float, default=12.)
    parser.add_argument("--scan-points", type=int, default=81)
    parser.add_argument("--score-scale", type=float, default=.075)
    parser.add_argument("--output", type=Path, default=Path("benchmark_validation.json"))
    args = parser.parse_args()
    if not 4 <= args.power <= 20 or args.scan_points < 9 or args.mu_max <= 8:
        parser.error("Use power in [4,20], at least 9 scan points, and mu-max above 8.")
    if not 0 < args.threshold < 1 or args.score_scale <= 0:
        parser.error("threshold must be between 0 and 1; score-scale must be positive.")
    physics = PhysicsModel()
    model = quadrature(physics, args.power, args.seed, args.threshold)
    mus = np.unique(np.r_[np.linspace(.1, args.mu_max, args.scan_points), 1.])
    cache = {}

    def profile(mu):
        key = float(mu)
        if key not in cache:
            fit = model.fit(mu_fixed=key)
            if not fit.success:
                raise RuntimeError(f"NI profile failed at mu={mu}: {fit.message}")
            cache[key] = {"q": float(fit.nll), "alpha_NI": float(fit.alpha[0])}
        return cache[key]["q"]

    fixed = lambda mu: model.nll(float(mu), [0.])
    fixed_values = np.array([fixed(mu) for mu in mus])
    profile_values = []
    for number, mu in enumerate(mus, 1):
        profile_values.append(profile(mu))
        if number % 10 == 0 or number == len(mus):
            print(f"Profiled {number}/{len(mus)} oracle scan points", flush=True)
    rates = model.rates
    report = {
        "scope": "Analytic oracle selector and quadrature; no trained-network validation or coverage study.",
        "physics": physics.to_dict(),
        "quadrature": {"draws_per_process": 2**args.power, "scramble_seeds": list(range(args.seed, args.seed + 3)),
                       "selected_nodes": len(model.quadrature_weights)},
        "preselection": {"balanced_classifier": "fS/(fS+fB+fNI)", "frozen_threshold": args.threshold,
                         "selected_rates_S_SBI_B_NI": rates.tolist(), "S_over_NI": float(rates[0] / rates[3]),
                         "NI_down_nominal_up": [float(model.rates_down[0, 3]), float(rates[3]),
                                               float(model.rates_up[0, 3])]},
        "fixed": extrema(mus, fixed_values, fixed),
        "profiled": extrema(mus, np.asarray(profile_values), profile),
        "local_fisher_widths": local_widths(model),
        "score_resolution": score_resolution(model, mus, args.score_scale),
        "scan": [{"mu": float(mu), "q_fixed": float(value), "q_profiled": cache[float(mu)]["q"],
                  "alpha_NI": cache[float(mu)]["alpha_NI"]} for mu, value in zip(mus, fixed_values)],
    }
    for minimum in report["profiled"]["minima"]:
        profile(minimum["mu"])
        minimum["alpha_NI"] = cache[minimum["mu"]]["alpha_NI"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"fixed": report["fixed"], "profiled": report["profiled"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
