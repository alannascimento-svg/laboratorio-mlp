"""PCA integrado: fases independentes, persistência e ausência de vazamento."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from calculos import update_report
from memoria import read_update, update_index
from preprocessamento import transform_inputs
from rede import (Config, experiment_signature, fit_network, hidden_size, load_model,
                  make_folds, model_details, predict_dataframe, run_experiment,
                  save_model, select_epochs_and_refit)


class TestPCATraining(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        latent = np.random.default_rng(876).normal(size=(100, 2))
        self.X = np.column_stack([latent[:, 0], latent[:, 1], latent[:, 0]*2,
                                  latent[:, 1]*3, latent[:, 0]+latent[:, 1], latent[:, 0]-latent[:, 1]])
        self.y = np.array([0, 1] * 50)
        self.features = [f'medida_{i}' for i in range(6)]
        self.cfg = Config(mode='kfold5', max_epochs=2, batch_size=23, pca_enabled=True)

    def dataset(self, X, name):
        frame = pd.DataFrame(X, columns=self.features)
        frame.insert(0, 'diagnosis', ['M', 'B']*50)
        frame.insert(0, 'id', [f'sintetico_{i}' for i in range(100)])
        source = self.root/name
        frame.to_csv(source, index=False)
        return source

    def test_disabled_pipeline_matches_original_scaler_and_network(self):
        cfg = replace(self.cfg, pca_enabled=False)
        net, scaler, chosen, ih, fh, fit, val, inner = select_epochs_and_refit(
            self.X, self.y, cfg, 51, self.features)
        expected_inner = StandardScaler().fit(self.X[fit])
        _, ih_old, chosen_old = fit_network(expected_inner.transform(self.X[fit]), self.y[fit],
            cfg, 52, cfg.max_epochs, (expected_inner.transform(self.X[val]), self.y[val]), self.features)
        expected_outer = StandardScaler().fit(self.X)
        expected_net, fh_old, _ = fit_network(expected_outer.transform(self.X), self.y,
            cfg, 53, chosen_old, features=self.features)
        self.assertEqual((chosen, ih, fh), (chosen_old, ih_old, fh_old))
        self.assertIsNone(net.pca)
        np.testing.assert_array_equal(scaler.mean_, expected_outer.mean_)
        for key in net.params:
            np.testing.assert_array_equal(net.params[key], expected_net.params[key])

    def test_validation_values_do_not_change_inner_preprocessing(self):
        first = select_epochs_and_refit(self.X, self.y, self.cfg, 61, self.features)
        fit, val, inner = first[5:]
        changed = self.X.copy()
        changed[val] = changed[val]*100 + 10000
        second = select_epochs_and_refit(changed, self.y, self.cfg, 61, self.features)
        np.testing.assert_array_equal(inner.mean_, self.X[fit].mean(axis=0))
        np.testing.assert_array_equal(inner.mean_, second[-1].mean_)
        self.assertEqual(inner.pca, second[-1].pca)
        self.assertEqual(inner.pca['fit_samples'], len(fit))
        self.assertEqual(first[1].pca['fit_samples'], len(self.X))
        self.assertFalse(np.array_equal(first[1].mean_, second[1].mean_))

    def test_model_reload_prediction_original_columns_and_real_update_memory(self):
        net, scaler, chosen, _, _, _, _, _ = select_epochs_and_refit(
            self.X, self.y, self.cfg, 71, self.features, self.root/'checkpoints')
        self.assertEqual(net.n_inputs, 2)
        self.assertEqual(net.n_hidden, hidden_size(2, 2))
        model = self.root/'modelo.npz'
        save_model(model, net, scaler, {'epochs':chosen})
        restored, mean, scale, metadata = load_model(model)
        self.assertEqual(restored.pca, net.pca)
        np.testing.assert_array_equal(scaler.transform(self.X), transform_inputs(restored, self.X, mean, scale))
        frame = pd.DataFrame(self.X, columns=self.features)
        prediction = predict_dataframe(model, frame[self.features[::-1]])
        np.testing.assert_array_equal(prediction[['score_M','score_B']], net.predict_proba(scaler.transform(self.X)))
        details = model_details(model)
        self.assertEqual(details['source_features'], self.features)
        self.assertEqual(details['feature_keys'], ['PC1','PC2'])
        self.assertEqual(mean.shape, (6,))
        path = self.root/'checkpoints/retreino/memoria_calculo.sqlite3'
        trace = read_update(path, update_index(path)['last_step'])
        np.testing.assert_array_equal(trace['raw_X'], self.X[trace['batch_indices_in_phase']])
        self.assertEqual(trace['pca'], net.pca)
        report = update_report(trace, 0, 0)
        self.assertTrue(report['verification']['ok'])
        self.assertIsNotNone(report['preprocessing'])
        with np.load(model, allow_pickle=False) as data:
            arrays = {key:data[key] for key in data.files}
        metadata['pca']['components'][0].pop()
        arrays['metadata'] = np.array(json.dumps(metadata))
        broken = self.root/'broken.npz'
        np.savez_compressed(broken, **arrays)
        with self.assertRaises(ValueError):
            load_model(broken)

    def test_external_test_cannot_change_its_fold_weights_scaler_or_pca(self):
        cfg = replace(self.cfg, max_epochs=1, record_calculations=False)
        _, test = make_folds(self.y, cfg)[0]
        changed = self.X.copy()
        changed[test] = changed[test]*500 + 1000
        for name, X in [('original', self.X), ('changed_test', changed)]:
            run_experiment(self.dataset(X, name+'.csv'), self.root/name, cfg)
        net1, mean1, scale1, _ = load_model(self.root/'original/folds/fold_01/modelo.npz')
        net2, mean2, scale2, _ = load_model(self.root/'changed_test/folds/fold_01/modelo.npz')
        self.assertEqual(net1.pca, net2.pca)
        np.testing.assert_array_equal(mean1, mean2)
        np.testing.assert_array_equal(scale1, scale2)
        for key in net1.params:
            np.testing.assert_array_equal(net1.params[key], net2.params[key])
        for name in ('original', 'changed_test'):
            folder = self.root/name
            config = json.loads((folder/'configuracao.json').read_text())
            metrics = json.loads((folder/'metricas_oof.json').read_text())
            model = load_model(folder/'modelo_final.npz')[0]
            self.assertEqual(config['architecture'], model.architecture)
            self.assertEqual(metrics['architecture'], model.architecture)
            self.assertEqual(len(metrics['fold_architectures']), 5)
            timing = json.loads((folder/'timing.json').read_text())
            self.assertGreater(timing['total_seconds'], 0)
            self.assertEqual(len(timing['folds']), 5)
            for phase in timing['folds'] + [timing['final']]:
                self.assertGreaterEqual(phase['total_seconds'], phase['selection_seconds'] + phase['refit_seconds'])
            signature = experiment_signature(folder)
            config['pca_variance'] = .8
            (folder/'configuracao.json').write_text(json.dumps(config))
            self.assertNotEqual(signature, experiment_signature(folder))
            config['pca_enabled'] = False
            (folder/'configuracao.json').write_text(json.dumps(config))
            without = experiment_signature(folder)
            config['pca_variance'] = .6
            (folder/'configuracao.json').write_text(json.dumps(config))
            self.assertEqual(without, experiment_signature(folder))

    def test_invalid_pca_configuration_is_rejected(self):
        for value in (True, 0, -1, 1.1, float('nan'), float('inf'), '95'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.cfg, pca_variance=value).validate()
        with self.assertRaises(ValueError):
            replace(self.cfg, pca_enabled='yes').validate()


if __name__ == '__main__':
    unittest.main()
