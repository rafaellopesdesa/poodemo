"""Cross-check the actual toolkit adapter against direct intensity calculations."""
import importlib.util
import numpy as np
import pytest

REQUIRED = ("nsbi_common_utils", "jax", "iminuit", "uproot", "awkward")
pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in REQUIRED),
    reason="Optional pinned toolkit/inference dependencies are not installed",
)


def arrays():
    x = np.linspace(-2.5, 2.5, 51)
    w = np.full(x.size, x[1] - x[0])
    signal = 3 * np.exp(-0.5 * ((x - .5) / .65) ** 2)
    background = 12 * np.exp(-0.5 * ((x + .2) / .9) ** 2)
    ni = 50 * np.exp(-0.5 * ((x + .7) / 1.1) ** 2)
    sbi = signal + background - .6 * np.sqrt(signal * background)
    nominal = np.stack([signal, sbi, background, ni])
    down = nominal[None].copy()
    up = down.copy()
    for out, sign in [(down, -1), (up, 1)]:
        out[0, 3] = ni * np.exp(sign * .12 * x)
    return nominal, down, up, w, nominal[1] + nominal[3]


def model(tmp_path):
    from poodemo.toolkit import build_workspace, load_model
    return load_model(build_workspace(tmp_path, *arrays()))


def test_asimov_stationarity_and_interference(tmp_path):
    m = model(tmp_path)
    nominal, _, _, w, truth = arrays()
    assert tuple(m.list_parameters) == ("mu", "alpha_NI")
    assert float(m.model([1., 0.])) == pytest.approx(0., abs=1.e-10)
    assert m.model_grad([1., 0.]) == pytest.approx(np.zeros(2), abs=1.e-9)
    for mu in [.2, .7, 1.6, 3.0]:
        c = np.array([mu - np.sqrt(mu), np.sqrt(mu), 1-np.sqrt(mu), 1.])
        intensity = c @ nominal
        direct = 2 * np.sum(w * (intensity - truth - truth * np.log(intensity/truth)))
        assert float(m.model([mu, 0.])) == pytest.approx(direct, abs=1.e-9)


def test_up_down_anchors_and_shape_normalization(tmp_path):
    m = model(tmp_path)
    nominal, down, up, w, _ = arrays()
    reference = nominal[0] / np.dot(w, nominal[0])
    for k in range(1):
        for sign, fields in [(-1, down), (1, up)]:
            alpha = np.zeros(1)
            alpha[k] = sign
            restored = m.component_intensity_ratios(alpha) * reference
            np.testing.assert_allclose(restored, fields[k], rtol=1.e-12)
    # The total rates follow the toolkit rate interpolator; shapes integrate to
    # those rates even at an intermediate nuisance point.
    from nsbi_common_utils.models.sbi_parametric_model import _calculate_combined_var
    import jax.numpy as jnp
    alpha = np.array([.37])
    reconstructed = m.component_intensity_ratios(alpha) * reference
    yields = nominal @ w
    for j in range(4):
        rate = yields[j] * float(_calculate_combined_var(
            jnp.asarray(alpha), jnp.asarray((up @ w)[:, j, None] / yields[j]),
            jnp.asarray((down @ w)[:, j, None] / yields[j]))[0])
        assert np.dot(reconstructed[j], w) == pytest.approx(rate, rel=1.e-12)


def test_invalid_domain_rejected_and_profile_fit(tmp_path):
    from poodemo.toolkit import make_fitter
    m = model(tmp_path)
    assert np.isinf(m.model([-1., 0.]))
    assert np.isinf(m.model([1., 4.]))
    fitter = make_fitter(m)
    fit = fitter.fit()
    assert fit.valid
    np.testing.assert_allclose(fit.parameters, [1., 0.], atol=2.e-4)
    scan = fitter.scan([.7, 1., 1.3], systematics=False)
    assert np.all(scan["valid"])
    assert scan["t_mu"][1] == pytest.approx(0., abs=1.e-8)
    assert np.all(scan["t_mu"] >= -1.e-8)


def test_matches_numpy_template_likelihood(tmp_path):
    from poodemo.inference import TemplateLikelihood
    m = model(tmp_path)
    reference = TemplateLikelihood(*arrays())
    for mu, alpha in [(1.0, [-.2]), (.4, [.4]), (1.9, [-1.2])]:
        assert float(m.model([mu, *alpha])) == pytest.approx(
            reference.nll(mu, alpha), rel=2.e-9, abs=2.e-8
        )


@pytest.mark.parametrize("exposure", [2_000., 1_000_000.])
def test_high_exposure_near_truth_matches_stable_numpy_and_has_finite_gradients(tmp_path, exposure):
    """Resolve tiny local displacements without subtracting large total rates."""
    from poodemo.inference import TemplateLikelihood
    from poodemo.toolkit import build_workspace, load_model
    nominal, down, up, weights, truth = arrays()
    fields = [exposure * values for values in (nominal, down, up)]
    m = load_model(build_workspace(tmp_path, *fields, weights, exposure * truth))
    reference = TemplateLikelihood(*fields, weights, exposure * truth)

    assert float(m.model([1., 0.])) < 1e-18
    np.testing.assert_allclose(m.model_grad([1., 0.]), [0., 0.],
                               atol=5e-8 * max(1., exposure / 2000.), rtol=0.)
    for mu, alpha in [(1. + 1e-7, -2e-7), (1. - 1e-6, 3e-6), (.7, .2)]:
        expected = reference.nll(mu, [alpha])
        actual = float(m.model([mu, alpha]))
        assert actual > 0
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-18)
        assert np.all(np.isfinite(m.model_grad([mu, alpha])))


def test_standard_normal_ni_auxiliary_is_exactly_quadratic(tmp_path):
    """Remove all template effects to isolate the Gaussian constraint term.

    A standard-normal auxiliary measurement with its Asimov observation at zero
    contributes alpha_NI**2 to -2 log L, or alpha_NI**2/2 to ordinary NLL.
    Test away from +/-1 too so this checks interpolation and extrapolation
    points without confusing a template deformation with the auxiliary term.
    """
    from poodemo.toolkit import build_workspace, load_model
    nominal, _, _, w, truth = arrays()
    neutral = nominal[None].copy()
    m = load_model(build_workspace(tmp_path, nominal, neutral, neutral, w, truth))
    assert tuple(m.list_parameters) == ("mu", "alpha_NI")
    for mu in (.7, 1., 1.5):
        baseline = float(m.model([mu, 0.]))
        for alpha in (-2.2, -.4, 0., .7, 1.8):
            np.testing.assert_allclose(
                float(m.model([mu, alpha])) - baseline, alpha**2,
                rtol=1.e-12, atol=1.e-10,
            )
            assert m.model_grad([mu, alpha])[1] == pytest.approx(2*alpha, abs=1.e-10)
