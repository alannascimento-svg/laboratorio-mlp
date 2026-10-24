from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rede import (CLASSES, CODES, Config, FEATURES, MLP, fit_network, load_dataset,
                  load_model, make_folds, metrics, one_hot, predict_dataframe,
                  run_experiment, save_model, variable_sizes, wta)

ROOT = Path(__file__).resolve().parents[1]

class TestRede(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset(ROOT / 'dados' / 'cancear.txt')

    def test_01_dataset(self):
        d = self.data
        self.assertEqual(d.X.shape, (569, 30))
        np.testing.assert_array_equal(np.bincount(d.y), [212, 357])
        self.assertEqual(len(set(d.ids)), 569)
        self.assertTrue(np.isfinite(d.X).all())

    def test_02_encoding(self):
        np.testing.assert_array_equal(one_hot(np.array([1, 0])), [[0, 1], [1, 0]])
        p = wta(np.array([[.1, .9], [.9, .1], [.5, .5]]))
        self.assertEqual(CLASSES[p].tolist(), ['B', 'M', 'M'])
        self.assertEqual(CODES[p].tolist(), ['01', '10', '10'])

    def test_03_architecture_and_softmax(self):
        n = MLP(1)
        self.assertEqual(n.n_parameters, 530)
        self.assertEqual(n.params['W1'].shape, (30, 16))
        self.assertEqual(n.params['W2'].shape, (16, 2))
        X = np.random.default_rng(1).normal(size=(7, 30))
        scores = n.predict_proba(X)
        np.testing.assert_allclose(scores.sum(axis=1), 1)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

    def test_04_all_530_gradients(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(5, 30))
        y = one_hot(np.array([0, 1, 0, 1, 0]))
        net = MLP(87)
        _, grads = net.loss_gradients(X, y, l2=.0001)
        eps = 1e-6
        errors = []
        for k, arr in net.params.items():
            for index in np.ndindex(arr.shape):
                original = arr[index]
                arr[index] = original + eps
                plus, _ = net.loss_gradients(X, y, .0001)
                arr[index] = original - eps
                minus, _ = net.loss_gradients(X, y, .0001)
                arr[index] = original
                errors.append(abs((plus - minus) / (2 * eps) - grads[k][index]))
        self.assertLess(max(errors), 1e-6)

    def test_05_variable_partition_many_seeds(self):
        for seed in range(50):
            sizes = variable_sizes(569, seed)
            self.assertEqual(sizes[0], 57)
            self.assertEqual(sum(sizes), 569)
            self.assertTrue(all(57 <= s <= 113 for s in sizes))
            folds = make_folds(self.data.y, Config(seed=seed))
            all_tests = np.concatenate([test for _, test in folds])
            np.testing.assert_array_equal(np.sort(all_tests), np.arange(569))
            for train, test in folds:
                self.assertEqual(len(np.intersect1d(train, test)), 0)
                self.assertEqual(len(np.unique(self.data.y[test])), 2)
                self.assertLess(abs(np.mean(self.data.y[test] == 0) - 212/569), .03)

    def test_06_kfold(self):
        for mode, k in [('kfold5', 5), ('kfold10', 10)]:
            f = make_folds(self.data.y, Config(mode=mode))
            self.assertEqual(len(f), k)
            self.assertLessEqual(max(len(t) for _, t in f) - min(len(t) for _, t in f), 1)

    def test_07_scaling_train_only_and_constant(self):
        train = np.array([[1., 8.], [3., 8.]])
        scaler = StandardScaler().fit(train)
        np.testing.assert_allclose(scaler.mean_, [2, 8])
        np.testing.assert_allclose(scaler.scale_, [1, 1])
        np.testing.assert_allclose(scaler.transform([[101., 8.]]), [[99., 0.]])
        np.testing.assert_allclose(scaler.mean_, [2, 8])

    def test_08_metric_orientation(self):
        y = np.array([1, 1, 0, 0, 0])
        scores = np.array([[.1, .9], [.8, .2], [.9, .1], [.6, .4], [.3, .7]])
        m = metrics(y, scores)
        self.assertEqual(m['confusion_matrix'], [[1, 1], [1, 2]])
        self.assertAlmostEqual(m['sensitivity_M'], 2/3)
        self.assertAlmostEqual(m['specificity_B'], .5)
        self.assertEqual(m['false_negative_M_as_B'], 1)

    def test_09_serialization_and_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            net, scaler = MLP(123), StandardScaler().fit(self.data.X[:100])
            path = Path(tmp) / 'model.npz'
            save_model(path, net, scaler, {'test': True})
            net2, mean, scale, meta = load_model(path)
            np.testing.assert_array_equal(net.predict_proba(scaler.transform(self.data.X[:5])),
                                          net2.predict_proba((self.data.X[:5] - mean) / scale))
            frame = pd.DataFrame(self.data.X[:5], columns=FEATURES)
            out = predict_dataframe(path, frame)
            out_reverse = predict_dataframe(path, frame[FEATURES[::-1]])
            np.testing.assert_array_equal(out.score_M, out_reverse.score_M)
            with self.assertRaises(ValueError):
                predict_dataframe(path, frame.assign(diagnosis='B'))
            with self.assertRaises(ValueError):
                predict_dataframe(path, frame.drop(columns=[FEATURES[0]]))

    def test_10_malformed_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.txt'
            path.write_text('Documentação\n123,M,' + ','.join(['1'] * 29), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '32 colunas'):
                load_dataset(path)

    def test_11_learning_and_reproducibility(self):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(80, 30))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        cfg = Config(max_epochs=35)
        initial = MLP(6).cross_entropy(X, y)
        a, hist, _ = fit_network(X, y, cfg, 6, 35)
        b, _, _ = fit_network(X, y, cfg, 6, 35)
        self.assertLess(hist[-1]['train_cross_entropy'], initial)
        np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))

    def test_12_full_workflow_and_scaler_audit(self):
        # Smoke test funcional, apenas 2 épocas. Não é seleção de hiperparâmetros.
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment(ROOT / 'dados' / 'cancear.txt', tmp, Config(max_epochs=2))
            path = Path(tmp)
            oof = pd.read_csv(path / 'previsoes_oof.csv')
            self.assertEqual(len(oof), 569)
            self.assertTrue((oof.vezes_em_teste == 1).all())
            self.assertTrue((path / 'modelo_final.npz').exists())
            for folder in sorted((path / 'folds').iterdir()):
                import json
                ix = json.loads((folder / 'indices.json').read_text())
                norms = pd.read_csv(folder / 'normalizacao.csv')
                np.testing.assert_allclose(norms.mean_treino_externo, self.data.X[ix['train_0based']].mean(axis=0))
                np.testing.assert_allclose(norms.mean_fit_interno, self.data.X[ix['inner_fit_0based']].mean(axis=0))
                self.assertFalse(set(ix['test_0based']) & set(ix['inner_validation_0based']))
                self.assertFalse(set(ix['test_0based']) & set(ix['inner_fit_0based']))
            self.assertEqual(result['metrics']['n'], 569)
            with self.assertRaises(FileExistsError):
                run_experiment(ROOT / 'dados' / 'cancear.txt', tmp, Config(max_epochs=2))

if __name__ == '__main__':
    unittest.main(verbosity=2)
