"""Analytic normalization, physical coherence, derivatives, and exact sampling."""

import json
import unittest

import numpy as np

from poodemo.physics import COMPONENTS, PhysicsModel


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.model = PhysicsModel()
        self.x = np.random.default_rng(90210).normal(size=(200, 3))

    def test_identical_gaussian_overlap_and_sbi_yield(self):
        identity = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        model = PhysicsModel(mean_s=(1., 1., 1.), mean_b=(1., 1., 1.), cov_s=identity, cov_b=identity)
        self.assertAlmostEqual(model.gaussian_overlap(), 1., places=14)
        expected = 1100 + 2 * model.phase_cos * np.sqrt(100 * 1000)
        self.assertAlmostEqual(model.component_yield("SBI"), expected, places=11)

    def test_coherent_amplitudes_and_positive_intensity(self):
        for alpha in (-1., 0., 1.):
            sbi = self.model.component_density(self.x, "SBI", alpha_b=alpha)
            np.testing.assert_allclose(sbi, np.abs(self.model.amplitude(self.x, "SBI", alpha_b=alpha))**2, rtol=2e-14, atol=1e-14)
            self.assertTrue(np.all(sbi > 0))
            for mu in (0., .01, .4, 1., 4.):
                direct = self.model.intensity(self.x, mu, alpha_b=alpha)
                coherent = np.abs(np.sqrt(mu) * self.model.amplitude(self.x, "S") + self.model.amplitude(self.x, "B", alpha_b=alpha))**2
                coherent += self.model.component_density(self.x, "NI")
                np.testing.assert_allclose(direct, coherent, rtol=2e-14, atol=1e-14)
                reconstructed = self.model.component_densities(self.x, alpha_b=alpha) @ self.model.coefficients(mu)
                np.testing.assert_allclose(direct, reconstructed, rtol=3e-14, atol=2e-14)
                self.assertTrue(np.all(direct > 0))

    def test_nuisance_displacements_and_coherent_propagation(self):
        for component in ("B", "NI"):
            keyword = "alpha_b" if component == "B" else "alpha_ni"
            nominal = self.model.mean(component)
            up = self.model.mean(component, **{keyword: 1.})
            down = self.model.mean(component, **{keyword: -1.})
            np.testing.assert_allclose(up - nominal, nominal - down, atol=1e-15)
            self.assertAlmostEqual(np.linalg.norm(up - nominal), .1 * np.linalg.norm(nominal), places=14)
            self.assertEqual(self.model.component_yield(component, **{keyword: 1.}), self.model.component_yield(component))
        np.testing.assert_array_equal(self.model.component_density(self.x, "NI", alpha_b=1.), self.model.component_density(self.x, "NI"))
        np.testing.assert_array_equal(self.model.component_density(self.x, "SBI", alpha_ni=1.), self.model.component_density(self.x, "SBI"))
        self.assertNotEqual(self.model.component_yield("SBI", alpha_b=1.), self.model.component_yield("SBI", alpha_b=-1.))

    def test_intensity_rate_and_shape_score_derivatives(self):
        for mu in (.36, 1., 2.25):
            step = 1e-5
            arguments = {"alpha_b": .4, "alpha_ni": -.7}
            fd_intensity = (self.model.intensity(self.x, mu + step, **arguments) - self.model.intensity(self.x, mu - step, **arguments)) / (2 * step)
            np.testing.assert_allclose(self.model.mu_derivative(self.x, mu, **arguments), fd_intensity, rtol=2e-8, atol=2e-8)
            fd_rate = (self.model.total_yield(mu + step, **arguments) - self.model.total_yield(mu - step, **arguments)) / (2 * step)
            self.assertAlmostEqual(self.model.yield_mu_derivative(mu, **arguments), fd_rate, delta=2e-7)
            fd_score = (np.log(self.model.pdf(self.x, mu + step, **arguments)) - np.log(self.model.pdf(self.x, mu - step, **arguments))) / (2 * step)
            np.testing.assert_allclose(self.model.score(self.x, mu, **arguments), fd_score, rtol=2e-7, atol=2e-9)

    def test_selected_normalization_and_extreme_tail_score(self):
        selected = {"S": 42., "SBI": 285., "B": 330., "NI": 700.}
        mu, step = 1.3, 1e-5
        log_plus = np.log(self.model.intensity(self.x, mu + step)) - np.log(self.model.total_yield(mu + step, selected_yields=selected))
        log_minus = np.log(self.model.intensity(self.x, mu - step)) - np.log(self.model.total_yield(mu - step, selected_yields=selected))
        np.testing.assert_allclose(self.model.score(self.x, mu, selected_yields=selected), (log_plus - log_minus) / (2 * step), rtol=2e-7, atol=2e-9)
        self.assertTrue(np.all(np.isfinite(self.model.score([[100., -100., 100.]], 1.))))
        with self.assertRaises(ValueError):
            self.model.score(self.x, 0.)

    def test_sbi_pdf_normalization_by_independent_importance_integration(self):
        rng = np.random.default_rng(111)
        n = 80_000
        choose_s = rng.random(n) < 1 / 11
        points = np.empty((n, 3))
        points[choose_s] = self.model.sample_component("S", int(choose_s.sum()), rng, alpha_b=1.)
        points[~choose_s] = self.model.sample_component("B", int((~choose_s).sum()), rng, alpha_b=1.)
        proposal = (self.model.component_density(points, "S") + self.model.component_density(points, "B", alpha_b=1.)) / 1100
        weights = self.model.component_pdf(points, "SBI", alpha_b=1.) / proposal
        self.assertLess(abs(weights.mean() - 1.), 5 * weights.std(ddof=1) / np.sqrt(n))

    def test_exact_sbi_sampler_first_moment(self):
        alpha = -.6
        sample = self.model.sample_component("SBI", 40_000, 829, alpha_b=alpha)
        ps = np.linalg.inv(np.asarray(self.model.cov_s))
        pb = np.linalg.inv(np.asarray(self.model.cov_b))
        mean_s, mean_b = self.model.mean("S"), self.model.mean("B", alpha_b=alpha)
        mean_overlap = np.linalg.solve(ps + pb, ps @ mean_s + pb @ mean_b)
        lam_i = self.model.component_yield("SBI", alpha_b=alpha) - 1100.
        expected_mean = (100 * mean_s + 1000 * mean_b + lam_i * mean_overlap) / self.model.component_yield("SBI", alpha_b=alpha)
        standard_error = sample.std(axis=0, ddof=1) / np.sqrt(len(sample))
        self.assertTrue(np.all(np.abs(sample.mean(axis=0) - expected_mean) < 5 * standard_error))

    def test_defaults_allow_useful_signal_preselection(self):
        # Oracle S/NI ordering: retain 70% of S and improve S/NI beyond 0.1.
        rng = np.random.default_rng(304)
        s = self.model.sample_component("S", 30_000, rng)
        ni = self.model.sample_component("NI", 50_000, rng)
        def discriminant(x):
            return np.log(self.model.component_pdf(x, "S")) - np.log(self.model.component_pdf(x, "NI"))
        threshold = np.quantile(discriminant(s), .3)
        ni_efficiency = np.mean(discriminant(ni) >= threshold)
        self.assertGreater(.01 * .7 / ni_efficiency, .1)

    def test_configuration_roundtrip_shapes_and_exposure(self):
        restored = PhysicsModel.from_dict(json.loads(json.dumps(self.model.to_dict())))
        np.testing.assert_allclose(restored.component_densities(self.x, alpha_b=.5), self.model.component_densities(self.x, alpha_b=.5), rtol=2e-15)
        twice = PhysicsModel(exposure=2.)
        np.testing.assert_allclose(twice.intensity(self.x, 1.2), 2 * self.model.intensity(self.x, 1.2), rtol=2e-15)
        np.testing.assert_allclose(twice.score(self.x, 1.2), self.model.score(self.x, 1.2), atol=3e-16)
        self.assertEqual(self.model.component_densities(self.x[0]).shape, (len(COMPONENTS),))
        self.assertEqual(self.model.sample_component("SBI", 0, 1).shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
