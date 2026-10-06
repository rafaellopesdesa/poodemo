"""Exercise every scientific stage with a small independent-data run.

Run from a checkout after installing requirements-colab.txt and the package:
    python scripts/smoke.py --root /tmp/poodemo-smoke
This is a workflow check, not production neural-network validation.
"""
import argparse
from poodemo.pipeline import (create_run, generate_data, run_preselection,
                              train_ratios, run_unbinned, run_binning_study,
                              run_spline_study)
from poodemo.diagnostics import run_estimator_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    run = create_run(args.root, "smoke")
    generate_data(run)
    result = run_preselection(run)
    print("Selected S/NI:", result["achieved_ratio"])
    train_ratios(run)
    unbinned = run_unbinned(run)
    print(unbinned["diagnostics"].to_string(index=False))
    bins = run_binning_study(run)
    print(bins["fidelity"].groupby("n_bins").information_fraction.min())
    estimators = run_estimator_study(run)
    print(estimators.groupby(["model", "systematics"])[["sigma_mu", "mu_hat"]].agg(["min", "max"]))
    spline = run_spline_study(run)
    print("Spline bins:", spline["n_bins"])
    print(spline["validation"].groupby("variation").max_fraction_error.max())
    print("Workflow complete. Inspect ratio closure and fit validity before drawing physics conclusions.")


if __name__ == "__main__":
    main()
