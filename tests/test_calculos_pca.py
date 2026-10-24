"""Conferência didática do PCA e dos gradientes reais depois da projeção."""
import copy
import unittest

import numpy as np

from calculos import snapshot_report, update_report
from preprocessamento import fit_preprocessor
from rede import MLP, one_hot


class TestCalculosPCA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(41)
        latent = rng.normal(size=(16, 2))
        self.raw = latent @ np.array([[1.,2.,-1.,.5], [.2,-.8,.6,2.]]) + [7.,4.,-3.,9.]
        self.y = np.arange(len(self.raw)) % 2
        self.pre = fit_preprocessor(self.raw, True, .95)
        self.X = self.pre.transform(self.raw)
        k = self.X.shape[1]
        self.net = MLP(71, k, [f'PC{i+1}' for i in range(k)])
        self.net.pca = {**self.pre.pca, 'source_features':['a','b','c','d'],
                        'source_feature_descriptions':['A','B','C','D']}
        self.net.params['b1'] += 3.

    def payload(self, optimizer='sgd_momentum'):
        params = self.net.params
        _, gradients = self.net.loss_gradients(self.X, one_hot(self.y), .001)
        zero = {key:np.zeros_like(value) for key,value in params.items()}
        payload = {'X':self.X.tolist(), 'raw_X':self.raw.tolist(), 'y':self.y.tolist(),
                   'mean':self.pre.mean_.tolist(), 'scale':self.pre.scale_.tolist(),
                   'features':self.net.features, 'feature_descriptions':self.net.feature_descriptions,
                   'classes':self.net.classes, 'pca':copy.deepcopy(self.net.pca),
                   'l2':.001, 'learning_rate':.01, 'optimizer':optimizer,
                   'step':1, 'epoch':1, 'batch':1,
                   'before':{key:value.tolist() for key,value in params.items()},
                   'gradients':{key:value.tolist() for key,value in gradients.items()}}
        if optimizer == 'sgd_momentum':
            velocity = {key:-.01*value for key,value in gradients.items()}
            after = {key:params[key]+velocity[key] for key in params}
            payload.update(momentum=.3, velocity_before={key:value.tolist() for key,value in zero.items()},
                           velocity_after={key:value.tolist() for key,value in velocity.items()},
                           momentum_origin={'kind':'initial','momentum_initial':.3,'seed':71})
        else:
            m = {key:.1*value for key,value in gradients.items()}
            v = {key:.001*value**2 for key,value in gradients.items()}
            after = {key:params[key]-.01*(m[key]/(1-.9))/(np.sqrt(v[key]/(1-.999))+1e-8) for key in params}
            payload.update(adam_m_before={key:value.tolist() for key,value in zero.items()},
                           adam_v_before={key:value.tolist() for key,value in zero.items()},
                           adam_m_after={key:value.tolist() for key,value in m.items()},
                           adam_v_after={key:value.tolist() for key,value in v.items()})
        payload['after'] = {key:value.tolist() for key,value in after.items()}
        return payload

    def test_snapshot_has_true_raw_values_all_pca_terms_and_forward(self):
        original = {key:value.copy() for key,value in self.net.params.items()}
        report = snapshot_report(self.net, self.pre.mean_, self.pre.scale_, self.raw, self.y, 0, 3, .001)
        self.assertTrue(report['verification']['ok'])
        self.assertEqual(report['architecture'][0], self.X.shape[1])
        np.testing.assert_allclose(report['forward']['network_input'], self.X[3], atol=1e-14)
        np.testing.assert_allclose(report['forward']['scores'], self.net.predict_proba(self.X[3:4])[0], atol=1e-14)
        raw_rows = report['steps'][0]['tables'][0]['rows']
        np.testing.assert_array_equal([row['raw'] for row in raw_rows], self.raw[3])
        terms = report['steps'][1]['tables'][1]['rows']
        self.assertEqual(len(terms), self.raw.shape[1]*self.X.shape[1])
        for q in range(self.X.shape[1]):
            total = sum(row['product'] for row in terms if row['component'] == f'PC{q+1}')
            self.assertAlmostEqual(total, self.X[3,q], places=12)
        self.assertTrue(any('∂PC' in formula for formula in report['steps'][1]['formulas']))
        epsilon = 1e-6
        for i in range(self.raw.shape[1]):
            plus, minus = self.raw[3:4].copy(), self.raw[3:4].copy()
            plus[0,i] += epsilon
            minus[0,i] -= epsilon
            derivative = (self.pre.transform(plus)-self.pre.transform(minus))[0]/(2*epsilon)
            expected = [row['derivative_raw'] for row in terms if row['feature'] == self.net.pca['source_features'][i]]
            np.testing.assert_allclose(derivative, expected, rtol=1e-7, atol=1e-8)
        for key in original:
            np.testing.assert_array_equal(self.net.params[key], original[key])

    def test_snapshot_weight_derivatives_match_finite_differences(self):
        report = snapshot_report(self.net, self.pre.mean_, self.pre.scale_, self.raw, self.y, 0, 0, .001)
        target = one_hot(self.y[:1])
        epsilon = 1e-6
        for i in range(self.net.n_inputs):
            saved = self.net.params['W1'][i,0]
            self.net.params['W1'][i,0] = saved+epsilon
            plus = self.net.loss_gradients(self.X[:1], target, .001)[0]
            self.net.params['W1'][i,0] = saved-epsilon
            minus = self.net.loss_gradients(self.X[:1], target, .001)[0]
            self.net.params['W1'][i,0] = saved
            self.assertAlmostEqual(report['gradients']['W1'][i], (plus-minus)/(2*epsilon), places=7)

    def test_actual_updates_match_both_optimizers_with_real_raw_rows(self):
        for optimizer in ('sgd_momentum','adam'):
            with self.subTest(optimizer=optimizer):
                payload = self.payload(optimizer)
                unchanged = copy.deepcopy(payload)
                report = update_report(payload, 0, 1)
                self.assertTrue(report['verification']['ok'])
                self.assertLess(report['verification']['projection_max_abs_error'], 1e-12)
                self.assertEqual(report['batch']['size'], len(self.raw))
                raw_rows = report['steps'][0]['tables'][0]['rows']
                np.testing.assert_array_equal([row['raw'] for row in raw_rows], self.raw[1])
                self.assertIn('raw_X real gravado', report['steps'][0]['note'])
                self.assertEqual(payload, unchanged)
                self.assertIn('PCₙᵢ', ' '.join(report['steps'][10]['formulas']))

    def test_missing_originals_and_projection_mismatch_are_rejected(self):
        payload = self.payload()
        del payload['raw_X']
        with self.assertRaisesRegex(ValueError, 'raw_X'):
            update_report(payload, 0)
        payload = self.payload()
        payload['raw_X'][7][0] += 20.
        with self.assertRaisesRegex(ValueError, 'não correspondem'):
            update_report(payload, 0, 0)
        payload = self.payload()
        payload['raw_X'] = [[1.]]
        with self.assertRaises(ValueError):
            update_report(payload, 0)
        with self.assertRaises(ValueError):
            snapshot_report(self.net, self.pre.mean_[:-1], self.pre.scale_, self.raw, self.y, 0)


if __name__ == '__main__':
    unittest.main()
