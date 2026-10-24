"""Tempos e entradas originais de PCA: consultas somente leitura, sem treinamento."""
import hashlib
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import app


class QuietHandler(app.Handler):
    def log_message(self, *args):
        pass


class TestResultTiming(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_explicit_timing_includes_each_phase_without_touching_files(self):
        timing = {'total_seconds': 101.5, 'folds': [
            {'fold': 1, 'selection_seconds': 20., 'refit_seconds': 10., 'total_seconds': 31.}],
            'final': {'selection_seconds': 30., 'refit_seconds': 25., 'total_seconds': 60.}}
        path = self.folder/'timing.json'
        path.write_text(json.dumps(timing))
        before = path.read_bytes()
        result = app.result_timing(self.folder)
        self.assertEqual(result, {**timing, 'source': 'timing.json'})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.folder.iterdir()), [path])

    def test_legacy_duration_uses_only_recorded_timestamps(self):
        path = self.folder/'status.json'
        path.write_text(json.dumps({'created_utc': '2026-09-19T17:26:38.000000+00:00',
                                    'completed_utc': '2026-09-19T18:11:23.500000+00:00'}))
        before = path.read_bytes()
        result = app.result_timing(self.folder)
        self.assertEqual(result['total_seconds'], 2685.5)
        self.assertEqual(result['source'], 'status.json')
        self.assertEqual(result['folds'], [])
        self.assertIsNone(result['final'])
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.folder/'timing.json').exists())

    def test_missing_invalid_or_reversed_history_is_unavailable_not_zero(self):
        self.assertIsNone(app.result_timing(self.folder)['total_seconds'])
        for status in ({}, {'created_utc': '2026-09-19T18:00:00+00:00'},
                       {'created_utc': 'invalid', 'completed_utc': '2026-09-19T18:00:00+00:00'},
                       {'created_utc': '2026-09-19T18:00:00+00:00', 'completed_utc': '2026-09-19T17:00:00+00:00'}):
            (self.folder/'status.json').write_text(json.dumps(status))
            self.assertIsNone(app.result_timing(self.folder)['total_seconds'])

    def test_explicit_negative_or_nonfinite_seconds_are_rejected(self):
        for value in (-1, float('inf'), float('nan'), True):
            (self.folder/'timing.json').write_text(json.dumps({'total_seconds': value}))
            with self.assertRaisesRegex(ValueError, 'Duração inválida'):
                app.result_timing(self.folder)


class TestPCAOriginalFeaturesHTTP(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.pca = {'kind': 'pca_v1', 'components': [[1., 0., 0.], [0., 1., 0.]],
                    'center': [0., 0., 0.], 'explained_variance_ratio': [.6, .38],
                    'source_features': ['medida_a', 'medida_b', 'medida_c'],
                    'n_original_features': 3, 'n_components': 2,
                    'variance_target': .95, 'retained_variance': .98, 'fit_samples': 50}
        self.net = SimpleNamespace(features=['PC1', 'PC2'], n_inputs=2, n_hidden=2,
                                   architecture=[2, 2, 2], classes=['M', 'B'], pca=self.pca)

    def test_csv_template_requests_original_measurements_not_components(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        try:
            with patch.object(app, 'run_folder', return_value=self.folder), \
                    patch.object(app, 'load_model', return_value=(self.net, None, None, {})):
                connection = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
                try:
                    connection.request('GET', '/api/template?run=exemplo')
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.read().decode().strip(), 'id,medida_a,medida_b,medida_c')
                finally:
                    connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_result_summary_uses_real_saved_pca_and_legacy_fallback(self):
        with patch.object(app, 'load_model', return_value=(self.net, None, None, {})):
            self.assertEqual(app.result_pca(self.folder), {
                'enabled': True, 'n_original_features': 3, 'n_components': 2,
                'variance_target': .95, 'retained_variance': .98, 'fit_samples': 50})
        legacy = SimpleNamespace(features=['medida_a', 'medida_b'], n_inputs=2)
        with patch.object(app, 'load_model', return_value=(legacy, None, None, {})):
            self.assertEqual(app.result_pca(self.folder), {
                'enabled': False, 'n_original_features': 2, 'n_components': 2})

    def test_calculation_matches_original_features_and_passes_raw_samples_to_report(self):
        raw = b'synthetic source for mocked calculation\n'
        (self.folder/'dados_utilizados.txt').write_bytes(raw)
        (self.folder/'configuracao.json').write_text(json.dumps({'data_sha256': hashlib.sha256(raw).hexdigest()}))
        data = SimpleNamespace(features=self.pca['source_features'], classes=['M', 'B'],
            X=np.array([[1., 2., 3.], [4., 5., 6.]]), y=np.array([0, 1]),
            ids=['a', 'b'], source_lines=[1, 2])
        with patch.object(app, 'run_folder', return_value=self.folder), \
                patch.object(app, 'resolve_model', return_value=self.folder/'modelo_final.npz'), \
                patch.object(app, 'load_model', return_value=(self.net, np.zeros(3), np.ones(3), {})), \
                patch.object(app, 'load_dataset', return_value=data), \
                patch.object(app, 'update_index', return_value={'count': 0, 'last_step': 0}), \
                patch('calculos.snapshot_report', return_value={'mode': 'snapshot'}) as report:
            result = app.calculation_result({'run': ['exemplo'], 'mode': ['snapshot']})
        self.assertEqual(result['context']['feature_count'], 2)
        self.assertEqual(result['context']['sample_count'], 2)
        self.assertIs(report.call_args.args[3], data.X)
        self.assertEqual(len(list(self.folder.iterdir())), 2)


if __name__ == '__main__':
    unittest.main()
