"""Demonstração incluída: consulta e exportação funcionam sem novo treino."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import app
from memoria import read_update
from rede import load_dataset, load_model
from public_fixtures import digest_files
from test_app_calculos import CalculationHTTPMixin


ROOT = Path(__file__).resolve().parents[1]


class TestPublicDemonstrationHTTP(CalculationHTTPMixin, unittest.TestCase):
    def setUp(self):
        self.folder = ROOT / 'resultados_exemplo'
        self.start_server()

    def get_json(self, path):
        status, headers, body = self.get(path)
        self.assertEqual(status, 200, body[:1000])
        self.assertIn('application/json', headers.get('Content-Type', ''))
        return json.loads(body)

    def test_included_metrics_model_checkpoints_calculations_and_exports_are_readonly(self):
        before = digest_files(self.folder)
        source = ROOT / 'dados' / 'cancear.txt'
        source_before = source.read_bytes()
        with patch.object(app, 'run_experiment', side_effect=AssertionError('Consulta iniciou treino')), \
                patch.object(app, 'save_prediction', side_effect=AssertionError('Consulta gravou previsão')):
            info = self.get_json('/api/info')
            runs = [run for run in info['runs'] if run['id'] == 'exemplo']
            self.assertEqual(len(runs), 1)
            self.assertIn('Demonstração pública', runs[0]['label'])
            self.assertEqual(info['n'], 569)
            self.assertFalse(self.get_json('/api/status')['busy'])

            result = self.get_json('/api/result?run=exemplo')
            saved_metrics = json.loads((self.folder / 'metricas_oof.json').read_text(encoding='utf8'))
            self.assertEqual(result['metrics'], saved_metrics)
            self.assertEqual(result['config']['optimizer'], 'adam')
            self.assertEqual(result['config']['max_epochs'], 3)
            self.assertTrue(result['config']['record_calculations'])
            self.assertEqual(result['metrics']['n'], 569)

            net, mean, scale, metadata = load_model(self.folder / 'modelo_final.npz')
            network = self.get_json('/api/network?run=exemplo&model=final')
            self.assertEqual(network['architecture'], [30, 16, 2])
            for key, values in net.params.items():
                np.testing.assert_array_equal(network['params'][key], values)
            checkpoints = self.get_json('/api/checkpoints?run=exemplo&model=final')
            self.assertIn('trained', [item['id'] for item in checkpoints])
            initial = 'retreino/epoca_00000.npz'
            self.assertIn(initial, [item['id'] for item in checkpoints])
            initial_network = self.get_json('/api/network?run=exemplo&model=final&checkpoint=' + initial)
            initial_net, _, _, _ = load_model(self.folder / 'checkpoints_final' / initial)
            for key, values in initial_net.params.items():
                np.testing.assert_array_equal(initial_network['params'][key], values)

            status, snapshot = self.calculation(run='exemplo', mode='snapshot', neuron=15, sample=568)
            self.assertEqual(status, 200, snapshot)
            self.assertTrue(snapshot['report']['verification']['ok'])
            self.assertEqual(snapshot['report']['mode'], 'snapshot')
            self.assertGreater(snapshot['context']['trace_count'], 0)
            X = load_dataset(source).X
            np.testing.assert_array_equal(snapshot['report']['forward']['scores'],
                                          net.predict_proba((X[568:569] - mean) / scale)[0])

            trace = read_update(self.folder / 'checkpoints_final' / 'retreino' / 'memoria_calculo.sqlite3', 1)
            status, recorded = self.calculation(run='exemplo', mode='update', step=1, neuron=15, sample=0)
            self.assertEqual(status, 200, recorded)
            self.assertEqual(recorded['report']['mode'], 'update')
            self.assertTrue(recorded['report']['verification']['ok'])
            self.assertEqual(recorded['context']['step'], 1)
            np.testing.assert_array_equal(recorded['report']['forward']['standardized'], trace['X'][0])
            for row in recorded['report']['update']['rows']:
                key, index = row['key'], tuple(row['index'])
                self.assertEqual(row['before'], np.asarray(trace['before'][key])[index])
                self.assertEqual(row['after'], np.asarray(trace['after'][key])[index])
                self.assertEqual(row['gradient'], np.asarray(trace['gradients'][key])[index])
            # O botão da página exporta este mesmo payload, sem perder valores.
            self.assertEqual(json.loads(json.dumps(recorded, allow_nan=False)), recorded)
            status, _, calculation_page = self.get('/calculos?run=exemplo')
            self.assertEqual(status, 200)
            self.assertIn(b'JSON.stringify(lastPayload,null,2)', calculation_page)

            downloaded = self.get_json('/download?run=exemplo&file=resumo_rodadas.json')
            self.assertEqual(downloaded['fold_summary'], result['fold_summary'])
            status, headers, bundle = self.get('/api/bundle?run=exemplo')
            self.assertEqual(status, 200)
            self.assertIn('application/zip', headers['Content-Type'])
            self.assertIn('attachment;', headers['Content-Disposition'])
            with zipfile.ZipFile(io.BytesIO(bundle)) as exported:
                exported_hashes = {name.removeprefix('exemplo/'): hashlib.sha256(exported.read(name)).hexdigest()
                                   for name in exported.namelist()}
            self.assertEqual(exported_hashes, before)
            self.assertFalse(self.get_json('/api/status')['busy'])
        self.assertEqual(source.read_bytes(), source_before)
        self.assertEqual(digest_files(self.folder), before)


if __name__ == '__main__':
    unittest.main()
