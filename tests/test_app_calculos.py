"""HTTP dos cálculos: legado somente leitura e uma execução sintética temporária."""
from __future__ import annotations

import csv
import hashlib
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

import numpy as np
from sklearn.preprocessing import StandardScaler

import app
from entrada import inspect_source
from memoria import read_update
from rede import (Config, MLP, fit_network, load_dataset, load_model, recorder_for,
                  run_experiment, save_model)
from public_fixtures import legacy_example


ROOT = Path(__file__).resolve().parents[1]


class QuietCalculationHandler(app.Handler):
    def log_message(self, format, *args):
        pass


class CalculationHTTPMixin:
    def start_server(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietCalculationHandler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive(), 'Servidor HTTP de teste não encerrou.')

    def get(self, path, host=None):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        try:
            connection.request('GET', path, headers={'Host': host} if host else {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def calculation(self, **query):
        status, headers, body = self.get('/api/calculation?' + urlencode(query))
        self.assertIn('application/json', headers.get('Content-Type', ''))
        return status, json.loads(body)


class TestLegacyCalculationHTTP(CalculationHTTPMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.folder = legacy_example(cls.root)

    def setUp(self):
        patcher = patch.object(app, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.start_server()

    def test_calculation_page_is_served_as_local_html(self):
        status, headers, body = self.get('/calculos?run=exemplo&model=final&neuron=0')
        self.assertEqual(status, 200)
        self.assertIn('text/html', headers['Content-Type'])
        page = body.decode('utf8')
        self.assertIn('<html lang="pt-BR">', page)
        self.assertIn('Memória de cálculo', page)
        self.assertIn('/api/calculation', page)
        self.assertIn('Voltar ao dashboard', page)

    def test_legacy_snapshot_matches_saved_parameters_without_inventing_updates(self):
        source = self.root / 'dados' / 'cancear.txt'
        tracked = [self.folder / 'modelo_final.npz', self.folder / 'configuracao.json', source]
        before = {path: path.read_bytes() for path in tracked}
        sqlite_before = sorted(str(path) for path in self.folder.rglob('*.sqlite*'))
        status, result = self.calculation(run='exemplo', model='final', neuron=15, sample=568)
        self.assertEqual(status, 200, result)
        self.assertEqual(set(result), {'context', 'samples', 'report', 'updates'})
        context, report = result['context'], result['report']
        self.assertEqual(context['architecture'], [30, 16, 2])
        self.assertEqual(context['neuron'], 15)
        self.assertEqual(context['sample_count'], 569)
        self.assertEqual(context['trace_count'], 0)
        self.assertEqual(context['last_step'], 0)
        self.assertEqual(result['updates'], [])
        self.assertEqual(report['mode'], 'snapshot')
        self.assertTrue(report['verification']['ok'])
        self.assertNotIn('update', report)
        self.assertIn('não reconstrução', report['notice'])
        data = load_dataset(source)
        net, mean, scale, _ = load_model(self.folder / 'modelo_final.npz')
        standardized = (data.X[568:569] - mean) / scale
        np.testing.assert_array_equal(report['forward']['standardized'], standardized[0])
        np.testing.assert_array_equal(report['forward']['scores'], net.predict_proba(standardized)[0])
        self.assertEqual(result['samples'][568]['id'], str(data.ids[568]))
        self.assertEqual(report['forward']['target'], [1.0, 0.0] if data.y[568] == 0 else [0.0, 1.0])
        self.assertEqual({path: path.read_bytes() for path in tracked}, before)
        self.assertEqual(sorted(str(path) for path in self.folder.rglob('*.sqlite*')), sqlite_before)

    def test_invalid_indices_mode_and_paths_are_rejected(self):
        cases = [
            {'neuron': -1}, {'neuron': 16}, {'neuron': 'abc'},
            {'sample': -1}, {'sample': 569}, {'sample': '0.5'},
            {'mode': 'retrain'}, {'model': '../modelo_final'}, {'model': 'fold_99'},
            {'checkpoint': '../modelo_final.npz'},
            {'checkpoint': 'retreino/../../modelo_final.npz'},
            {'checkpoint': 'retreino/epoca_99999.npz'},
            {'run': '../resultados_exemplo'},
        ]
        for case in cases:
            with self.subTest(query=case):
                status, result = self.calculation(**{'run': 'exemplo', **case})
                self.assertEqual(status, 400, result)
                self.assertTrue(result['error'])

    def test_legacy_update_request_reports_absence_of_saved_memory(self):
        for query in ({'mode': 'update', 'step': 1}, {'step': 1}, {'mode': 'update'}):
            with self.subTest(query=query):
                status, result = self.calculation(run='exemplo', **query)
                self.assertEqual(status, 400)
                self.assertIn('error', result)
                self.assertNotIn('report', result)

    def test_calculation_routes_reject_unapproved_host(self):
        for path in ['/calculos', '/api/calculation?run=exemplo']:
            with self.subTest(path=path):
                status, _, body = self.get(path, host='example.invalid')
                self.assertEqual(status, 403)
                self.assertIn('Host não autorizado', json.loads(body)['error'])


class TestTemporaryCalculationHTTP(CalculationHTTPMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.run_id = 'r20000101_000000_000001'
        cls.folder = cls.root / 'resultados' / cls.run_id
        generator = np.random.default_rng(123)
        measurements = generator.normal(size=(60, 3)) * [3.0, 10.0, 0.2] + [20.0, 100.0, 1.0]
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['id', 'classe', 'a', 'nota_ignorada', 'b', 'c'])
        for index, row in enumerate(measurements):
            writer.writerow([f'{index:04d}', 'M' if index % 2 == 0 else 'B',
                             row[0], '<anotação ignorada>', row[1], row[2]])
        text = stream.getvalue()
        table = inspect_source(text)
        cls.schema = {
            'delimiter': table['delimiter'], 'header': 'yes',
            'columns': [{key: column[key] for key in ('index', 'description', 'role')}
                        for column in table['columns']],
        }
        cls.descriptions = ['Altura anotada', 'Largura anotada', 'Densidade anotada']
        for index, description in zip([2, 4, 5], cls.descriptions):
            cls.schema['columns'][index]['description'] = description
        cls.source = cls.root / 'sintetica.csv'
        cls.source.write_text(text, encoding='utf8')
        cls.cfg = Config(seed=7, mode='kfold5', max_epochs=1, patience=1,
                         batch_size=8, snapshot_interval=1, record_calculations=True)
        run_experiment(cls.source, cls.folder, cls.cfg, input_schema=cls.schema)
        cls.dataset = load_dataset(cls.source, cls.schema)

    def setUp(self):
        patcher = patch.object(app, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.start_server()

    def digest_files(self):
        return {str(path.relative_to(self.folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.folder.rglob('*') if path.is_file()}

    def test_snapshot_uses_selected_columns_descriptions_and_original_ids(self):
        before = self.digest_files()
        status, result = self.calculation(run=self.run_id, neuron=2, sample=0, mode='snapshot')
        self.assertEqual(status, 200, result)
        self.assertEqual(result['context']['architecture'], [3, 3, 2])
        self.assertEqual(result['context']['epoch'], 1)
        self.assertEqual(result['samples'][0]['id'], '0000')
        self.assertEqual(result['samples'][0]['label'], 'ID 0000 · M · linha 2')
        self.assertEqual(result['report']['sample_count'], 60)
        rows = result['report']['steps'][0]['tables'][0]['rows']
        self.assertEqual([row['feature'] for row in rows], ['a', 'b', 'c'])
        self.assertEqual([row['description'] for row in rows], self.descriptions)
        np.testing.assert_array_equal([row['raw'] for row in rows], self.dataset.X[0])
        status, _, body = self.get('/api/network?' + urlencode({'run': self.run_id, 'model': 'final'}))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['feature_keys'], ['x3', 'x5', 'x6'])
        self.assertEqual(self.digest_files(), before)

    def test_recorded_updates_map_phase_samples_to_original_ids_and_exact_values(self):
        before = self.digest_files()
        for model in ['final', 'fold_01']:
            for phase in ['retreino', 'selecao_interna']:
                with self.subTest(model=model, phase=phase):
                    checkpoint = 'trained' if phase == 'retreino' else 'selecao_interna/epoca_00001.npz'
                    base = self.folder / 'checkpoints_final' if model == 'final' else self.folder / 'folds' / model / 'checkpoints'
                    trace = read_update(base / phase / 'memoria_calculo.sqlite3', 2)
                    if model == 'final':
                        audit = json.loads((self.folder / 'modelo_final_indices.json').read_text(encoding='utf8'))
                        phase_indices = list(range(60)) if phase == 'retreino' else audit['inner_fit_0based']
                    else:
                        audit = json.loads((self.folder / 'folds' / model / 'indices.json').read_text(encoding='utf8'))
                        phase_indices = audit['train_0based' if phase == 'retreino' else 'inner_fit_0based']
                    global_indices = [phase_indices[index] for index in trace['batch_indices_in_phase']]
                    status, result = self.calculation(run=self.run_id, model=model, checkpoint=checkpoint,
                                                      neuron=2, sample=1, mode='update', step=2)
                    self.assertEqual(status, 200, result)
                    context, report = result['context'], result['report']
                    self.assertEqual(context['phase'], phase)
                    self.assertEqual(context['step'], 2)
                    self.assertEqual(context['update_epoch'], 1)
                    self.assertEqual(context['trace_count'], math.ceil(len(phase_indices) / self.cfg.batch_size))
                    self.assertEqual(context['sample_count'], len(trace['y']))
                    self.assertEqual([sample['index'] for sample in result['samples']], list(range(len(global_indices))))
                    self.assertEqual([sample['id'] for sample in result['samples']],
                                     [str(self.dataset.ids[index]) for index in global_indices])
                    self.assertEqual([sample['label'] for sample in result['samples']],
                                     [f"ID {self.dataset.ids[index]} · {'M' if self.dataset.y[index] == 0 else 'B'} · linha {self.dataset.source_lines[index]}"
                                      for index in global_indices])
                    self.assertEqual(report['mode'], 'update')
                    self.assertEqual(report['update']['step'], 2)
                    self.assertEqual(report['batch']['size'], len(trace['y']))
                    self.assertTrue(report['verification']['ok'], report['verification'])
                    np.testing.assert_array_equal(report['forward']['standardized'], trace['X'][1])
                    before_net = MLP(0, 3, trace['features'], trace['feature_descriptions'])
                    before_net.params = {key: np.asarray(value) for key, value in trace['before'].items()}
                    np.testing.assert_array_equal(report['forward']['scores'],
                                                  before_net.predict_proba(np.asarray(trace['X'][1:2]))[0])
                    for row in report['update']['rows']:
                        key, index = row['key'], tuple(row['index'])
                        self.assertEqual(row['before'], np.asarray(trace['before'][key])[index])
                        self.assertEqual(row['after'], np.asarray(trace['after'][key])[index])
                        self.assertEqual(row['gradient'], np.asarray(trace['gradients'][key])[index])
                    self.assertEqual(report['forward']['target'],
                                     [1.0, 0.0] if self.dataset.y[global_indices[1]] == 0 else [0.0, 1.0])
        self.assertEqual(self.digest_files(), before)

    def test_initial_checkpoint_exposes_zero_updates_and_rejects_future_step(self):
        for model in ['final', 'fold_01']:
            for phase in ['retreino', 'selecao_interna']:
                with self.subTest(model=model, phase=phase):
                    query = {'run': self.run_id, 'model': model,
                             'checkpoint': phase + '/epoca_00000.npz'}
                    status, result = self.calculation(**query, mode='snapshot')
                    self.assertEqual(status, 200, result)
                    self.assertEqual(result['context']['epoch'], 0)
                    self.assertEqual(result['context']['trace_count'], 0)
                    self.assertEqual(result['context']['last_step'], 0)
                    status, result = self.calculation(**query, mode='update', step=1)
                    self.assertEqual(status, 400)
                    self.assertIn('intervalo deste checkpoint', result['error'])

    def test_default_update_is_latest_step_and_invalid_batch_sample_is_rejected(self):
        status, result = self.calculation(run=self.run_id, mode='update')
        self.assertEqual(status, 200, result)
        last_step = math.ceil(60 / self.cfg.batch_size)
        self.assertEqual(result['context']['step'], last_step)
        self.assertEqual(result['context']['last_step'], last_step)
        self.assertEqual(result['context']['sample_count'], 60 % self.cfg.batch_size)
        for query in [{'sample': -1}, {'sample': 4}, {'step': 0}, {'step': -1},
                      {'step': last_step + 1}, {'step': 'invalid'}]:
            with self.subTest(query=query):
                status, result = self.calculation(run=self.run_id, mode='update', **query)
                self.assertEqual(status, 400, result)
                self.assertTrue(result['error'])

    def test_changed_source_is_rejected_by_sha256(self):
        source = self.folder / 'dados_utilizados.txt'
        original = source.read_bytes()
        try:
            source.write_bytes(original + b'\n')
            status, result = self.calculation(run=self.run_id)
            self.assertEqual(status, 400)
            self.assertIn('mesmo SHA-256', result['error'])
        finally:
            source.write_bytes(original)


class TestMulticlassCalculationHTTP(CalculationHTTPMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.run_id = 'r20000101_000000_000010'
        cls.folder = cls.root/'resultados'/cls.run_id
        cls.folder.mkdir(parents=True)
        classes = ['MIT','CYT','NUC','ME3','ME2','ME1','EXC','VAC','POX','ERL']
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['id', *[f'medida_{i+1}' for i in range(8)], 'classe'])
        for index, values in enumerate(np.random.default_rng(321).normal(size=(60,8))):
            # ID duplicado: o rótulo visível também precisa da linha física.
            writer.writerow(['P000' if index in (0,10) else f'P{index:03d}', *values, classes[index % 10]])
        cls.source = cls.folder/'dados_utilizados.txt'
        cls.source.write_text(stream.getvalue(), encoding='utf8')
        table = inspect_source(stream.getvalue())
        cls.schema = {'delimiter':table['delimiter'], 'header':'yes',
                      'columns':[{key:column[key] for key in ('index','description','role')}
                                 for column in table['columns']]}
        cls.dataset = load_dataset(cls.source, cls.schema)
        cls.classes = cls.dataset.classes
        cls.scaler = StandardScaler().fit(cls.dataset.X)
        cfg = Config(max_epochs=1, batch_size=13, snapshot_interval=1, record_calculations=True)
        recorder = recorder_for(cls.folder/'checkpoints_final', 'retreino', cls.scaler, cfg, {'test_only':True})
        net, _, _ = fit_network(cls.scaler.transform(cls.dataset.X), cls.dataset.y, cfg, 13, 1,
                                 features=cls.dataset.features, feature_descriptions=cls.dataset.feature_descriptions,
                                 input_schema=cls.schema, classes=cls.classes, recorder=recorder)
        save_model(cls.folder/'modelo_final.npz', net, cls.scaler, {'epoch':1, 'test_only':True})
        (cls.folder/'configuracao.json').write_text(json.dumps({'data_sha256':cls.dataset.sha256,
            'input_schema':cls.schema,'classes':cls.classes,'l2':cfg.l2}), encoding='utf8')
        (cls.folder/'CONCLUIDO.txt').write_text('Fixture temporária de teste HTTP: uma época sintética.', encoding='utf8')

    def setUp(self):
        patcher = patch.object(app, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.start_server()

    def test_snapshot_outputs_and_duplicate_ids_keep_saved_class_order_and_lines(self):
        hashes = {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in self.folder.rglob('*') if path.is_file()}
        status, payload = self.calculation(run=self.run_id, neuron=8, sample=9)
        self.assertEqual(status,200,payload)
        self.assertEqual(payload['context']['classes'], self.classes)
        report = payload['report']
        self.assertEqual(report['classes'], self.classes)
        self.assertEqual(report['architecture'], [8,9,10])
        self.assertEqual(np.argmax(report['forward']['target']), self.classes.index('ERL'))
        self.assertTrue(report['verification']['ok'])
        self.assertEqual(payload['samples'][0]['id'], payload['samples'][10]['id'])
        self.assertEqual(payload['samples'][0]['label'], 'ID P000 · MIT · linha 2')
        self.assertEqual(payload['samples'][10]['label'], 'ID P000 · MIT · linha 12')
        for path, digest in hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_recorded_step_uses_all_outputs_and_real_batch_class_labels(self):
        trace = read_update(self.folder/'checkpoints_final'/'retreino'/'memoria_calculo.sqlite3', 2)
        status, payload = self.calculation(run=self.run_id, neuron=8, mode='update', step=2, sample=3)
        self.assertEqual(status,200,payload)
        report = payload['report']
        self.assertEqual(report['classes'], self.classes)
        self.assertEqual(report['architecture'], [8,9,10])
        self.assertTrue(report['verification']['ok'])
        np.testing.assert_array_equal(report['forward']['standardized'], trace['X'][3])
        for sample, index in zip(payload['samples'], trace['batch_indices_in_phase']):
            self.assertEqual(sample['label'], f'ID {self.dataset.ids[index]} · {self.classes[self.dataset.y[index]]} · linha {self.dataset.source_lines[index]}')
        self.assertEqual(len(report['update']['rows']),29)


if __name__ == '__main__':
    unittest.main(verbosity=2)
