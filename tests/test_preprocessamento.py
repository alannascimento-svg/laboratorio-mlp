"""PCA por variância: transformação, validação e identidade do caminho legado."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np
from sklearn.preprocessing import StandardScaler

from preprocessamento import fit_preprocessor, transform_inputs, validate_pca, source_features, source_descriptions


class TestPreprocessamento(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(519)
        latent = rng.normal(size=(40, 2))
        self.X = latent @ np.array([[1., 2., -3., .4], [.2, -.8, .6, 2.]]) + [12., -6., 22., 7.]

    def test_disabled_is_bitwise_standard_scaler_including_float32(self):
        for dtype in (np.float64, np.float32):
            X = self.X.astype(dtype)
            expected = StandardScaler().fit(X)
            actual = fit_preprocessor(X)
            self.assertIsNone(actual.pca)
            np.testing.assert_array_equal(actual.mean_, expected.mean_)
            np.testing.assert_array_equal(actual.scale_, expected.scale_)
            self.assertEqual(actual.transform(X).tobytes(), expected.transform(X).tobytes())

    def test_projection_matches_explicit_formula_and_covariance(self):
        fitted = fit_preprocessor(self.X, True, .95)
        pca = fitted.pca
        C = np.asarray(pca['components'])
        Z = (self.X-fitted.mean_)/fitted.scale_
        centered = Z-np.asarray(pca['center'])
        np.testing.assert_allclose(fitted.transform(self.X), centered @ C.T, atol=1e-14)
        covariance = centered.T @ centered/(len(Z)-1)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        np.testing.assert_allclose(pca['explained_variance_ratio'], (eigenvalues/eigenvalues.sum())[:len(C)], atol=1e-14)
        self.assertEqual(pca['n_components'], 2)
        self.assertEqual(pca['fit_samples'], len(self.X))
        self.assertGreaterEqual(pca['retained_variance'], .95)
        self.assertLess(sum(pca['explained_variance_ratio'][:-1]), .95)
        np.testing.assert_allclose(C @ C.T, np.eye(len(C)), atol=1e-14)
        self.assertTrue(np.all(C[np.arange(len(C)), np.argmax(abs(C), axis=1)] >= 0))
        self.assertEqual(fit_preprocessor(self.X, True, .95).pca, pca)

    def test_exact_cut_is_inclusive_and_one_hundred_percent_supported(self):
        X = np.array([[1.,0.], [-1.,0.], [0.,1.], [0.,-1.]])
        self.assertEqual(fit_preprocessor(X, True, .5).pca['n_components'], 1)
        self.assertEqual(fit_preprocessor(X, True, 1.).pca['n_components'], 2)

    def test_transforming_other_rows_never_refits(self):
        fitted = fit_preprocessor(self.X[:25], True, .9)
        before = copy.deepcopy(fitted.pca)
        mean, scale = fitted.mean_.copy(), fitted.scale_.copy()
        fitted.transform(self.X[25:]*1000)
        self.assertEqual(before, fitted.pca)
        np.testing.assert_array_equal(fitted.mean_, mean)
        np.testing.assert_array_equal(fitted.scale_, scale)

    def test_source_names_and_shared_transform(self):
        fitted = fit_preprocessor(self.X, True, .95)
        pca = copy.deepcopy(fitted.pca)
        pca.update(source_features=['a','b','c','d'], source_feature_descriptions=['A','B','C','D'])
        net = SimpleNamespace(pca=pca, n_inputs=pca['n_components'], features=['PC1','PC2'])
        self.assertEqual(source_features(net), ['a','b','c','d'])
        self.assertEqual(source_descriptions(net), ['A','B','C','D'])
        np.testing.assert_allclose(transform_inputs(net, self.X, fitted.mean_, fitted.scale_), fitted.transform(self.X), atol=1e-14)
        legacy = SimpleNamespace(n_inputs=4, features=['a','b','c','d'])
        np.testing.assert_array_equal(transform_inputs(legacy, self.X, fitted.mean_, fitted.scale_),
                                      (self.X-fitted.mean_)/fitted.scale_)

    def test_zero_variance_bad_targets_and_malformed_data_rejected(self):
        with self.assertRaisesRegex(ValueError, 'variância total zero'):
            fit_preprocessor(np.ones((8,3)), True)
        with self.assertRaisesRegex(ValueError, 'duas amostras'):
            fit_preprocessor([[1., 2.]], True)
        for target in (0, -1, 1.01, float('nan'), float('inf'), True, '0.95'):
            with self.subTest(target=target), self.assertRaises(ValueError):
                fit_preprocessor(self.X, True, target)
        for X in ([], [1,2], [[float('nan')]], [[float('inf')]], [['não é número']]):
            with self.subTest(X=X), self.assertRaises(ValueError):
                fit_preprocessor(X, True)
        with self.assertRaises(ValueError):
            fit_preprocessor(self.X, enabled=1)

    def test_invalid_saved_transform_rejected(self):
        base = fit_preprocessor(self.X, True).pca
        for key, value in [('kind','other'), ('center',[0.]), ('components',[[1.,0.,0.,0.]]),
                           ('fit_samples',True), ('retained_variance',float('nan')),
                           ('n_components',3), ('source_features',['a']*4),
                           ('explained_variance_ratio',[.1,.1])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_pca({**base,key:value}, 4, 2)
        broken = copy.deepcopy(base)
        broken['components'][0][0] += 2
        with self.assertRaisesRegex(ValueError, 'ortonormais'):
            validate_pca(broken, 4, 2)
        with self.assertRaises(ValueError):
            validate_pca(None, 4, 2)


if __name__ == '__main__':
    unittest.main()
