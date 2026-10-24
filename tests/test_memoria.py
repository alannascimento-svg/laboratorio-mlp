"""Audita passos reais em bases sintéticas; todos os artefatos são temporários."""
from __future__ import annotations

from contextlib import closing
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

import numpy as np
from sklearn.preprocessing import StandardScaler

from memoria import append_update, read_update, update_index
from rede import (Config, MLP, fit_network, flat_gradient, load_model,
                  momentum_next, one_hot, recorder_for)


class TestMemoriaStorage(unittest.TestCase):
    def test_compression_and_duplicate_step_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'memoria.sqlite3'
            payload = {'step': 1, 'epoch': 1, 'batch': 1,
                       'description': 'Atualização real · x1',
                       'before': {'W1': [[1.25, -0.75]]},
                       'after': {'W1': [[1.3, -0.8]]}}
            append_update(path, payload)
            with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
                blob = connection.execute('SELECT payload FROM updates WHERE step = 1').fetchone()[0]
            self.assertIsInstance(blob, bytes)
            self.assertEqual(json.loads(zlib.decompress(blob)), payload)
            original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(sqlite3.IntegrityError):
                append_update(path, {**payload, 'after': {'W1': [[999.0, 999.0]]}})
            self.assertEqual(read_update(path, 1), payload)
            self.assertEqual(update_index(path), {'count': 1, 'last_step': 1})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_digest)

    def test_checkpoint_limits_and_read_only_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder / 'memoria.sqlite3'
            for step, epoch, batch in [(1, 1, 1), (2, 1, 2), (3, 2, 1)]:
                append_update(path, {'step': step, 'epoch': epoch, 'batch': batch})
            original_bytes = path.read_bytes()
            original_files = sorted(file.name for file in folder.iterdir())
            self.assertEqual(update_index(path), {'count': 3, 'last_step': 3})
            self.assertEqual(update_index(path, max_epoch=1), {'count': 2, 'last_step': 2})
            self.assertEqual(update_index(path, max_epoch=0), {'count': 0, 'last_step': 0})
            self.assertEqual(read_update(path, 2, max_epoch=1)['epoch'], 1)
            self.assertEqual(read_update(path, 3, max_epoch=2)['epoch'], 2)
            for step, limit in [(3, 1), (1, 0), (4, None)]:
                with self.subTest(step=step, max_epoch=limit):
                    with self.assertRaisesRegex(ValueError, 'intervalo deste checkpoint'):
                        read_update(path, step, max_epoch=limit)
            for step in [True, False, 0, -1, 1.5, '1', None]:
                with self.subTest(invalid_step=step):
                    with self.assertRaisesRegex(ValueError, 'passo de atualização válido'):
                        read_update(path, step)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(sorted(file.name for file in folder.iterdir()), original_files)

    def test_absent_memory_is_not_created_by_readers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'nao_existe' / 'memoria.sqlite3'
            self.assertEqual(update_index(path), {'count': 0, 'last_step': 0})
            self.assertEqual(update_index(path, max_epoch=2), {'count': 0, 'last_step': 0})
            with self.assertRaisesRegex(ValueError, 'não possui memória'):
                read_update(path, 1)
            self.assertFalse(path.exists())
            self.assertFalse(path.parent.exists())

    def test_nonfinite_payload_does_not_corrupt_existing_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'memoria.sqlite3'
            append_update(path, {'step': 1, 'epoch': 1, 'batch': 1})
            original_bytes = path.read_bytes()
            for value in [float('nan'), float('inf'), -float('inf')]:
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        append_update(path, {'step': 2, 'epoch': 1, 'batch': 2, 'gradient': value})
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(update_index(path), {'count': 1, 'last_step': 1})


class TestMemoriaTraining(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        # Escalas distintas permitem conferir que o registro contém X padronizado.
        self.raw_X = np.array([
            [8.0, 100.0, -4.0], [11.0, 120.0, -1.0],
            [13.0, 105.0, 3.0], [17.0, 150.0, 7.0],
            [19.0, 175.0, -2.0], [23.0, 160.0, 5.0],
            [29.0, 190.0, 2.0], [31.0, 220.0, 9.0],
        ])
        self.y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        self.scaler = StandardScaler().fit(self.raw_X)
        self.X = self.scaler.transform(self.raw_X)
        self.features = ['medida_a', 'medida_b', 'medida_c']
        self.descriptions = ['Altura', 'Largura', 'Densidade']
        self.schema = {
            'delimiter': ',', 'header': 'yes',
            'columns': [
                {'index': 0, 'description': 'ID', 'role': 'id'},
                {'index': 1, 'description': 'Diagnóstico', 'role': 'target'},
                *[{'index': index + 2, 'description': description, 'role': 'feature'}
                  for index, description in enumerate(self.descriptions)],
            ],
        }
        self.seed, self.epochs = 19, 2

    def train(self, name, optimizer='sgd_momentum', record=True, adaptive=False,
              learning_rate=0.02):
        cfg = Config(max_epochs=self.epochs, batch_size=3, learning_rate=learning_rate,
                     l2=0.001, optimizer=optimizer, adaptive_momentum=adaptive,
                     momentum_initial=0.35, record_calculations=record,
                     snapshot_interval=1)
        phase = 'treino_sintetico'
        recorder = recorder_for(self.folder / name, phase, self.scaler, cfg,
                                {'seed': self.seed, 'test_only': True})
        net, history, selected_epoch = fit_network(
            self.X, self.y, cfg, self.seed, self.epochs,
            features=self.features, recorder=recorder,
            feature_descriptions=self.descriptions, input_schema=self.schema)
        path = self.folder / name / phase / 'memoria_calculo.sqlite3'
        return cfg, net, history, selected_epoch, path

    def audited_traces(self, cfg, final_net, path):
        batches_per_epoch = math.ceil(len(self.X) / cfg.batch_size)
        count = batches_per_epoch * self.epochs
        self.assertEqual(update_index(path), {'count': count, 'last_step': count})
        self.assertEqual(update_index(path, max_epoch=1),
                         {'count': batches_per_epoch, 'last_step': batches_per_epoch})
        initial_net = MLP(self.seed, self.X.shape[1], self.features, self.descriptions)
        traces = [read_update(path, step) for step in range(1, count + 1)]
        for offset, trace in enumerate(traces):
            with self.subTest(step=offset + 1, optimizer=cfg.optimizer):
                self.assertEqual(trace['step'], offset + 1)
                self.assertEqual(trace['epoch'], offset // batches_per_epoch + 1)
                self.assertEqual(trace['batch'], offset % batches_per_epoch + 1)
                self.assertEqual(trace['optimizer'], cfg.optimizer)
                self.assertEqual(trace['phase'], 'treino_sintetico')
                self.assertEqual(trace['features'], self.features)
                self.assertEqual(trace['feature_descriptions'], self.descriptions)
                self.assertEqual(trace['input_schema'], self.schema)
                self.assertEqual(trace['seed'], self.seed)
                self.assertTrue(trace['test_only'])
                np.testing.assert_array_equal(trace['mean'], self.scaler.mean_)
                np.testing.assert_array_equal(trace['scale'], self.scaler.scale_)
                indices = np.asarray(trace['batch_indices_in_phase'])
                np.testing.assert_array_equal(trace['X'], self.X[indices])
                np.testing.assert_array_equal(trace['y'], self.y[indices])
                np.testing.assert_allclose(
                    np.asarray(trace['X']) * trace['scale'] + trace['mean'],
                    self.raw_X[indices], rtol=1e-14, atol=1e-14)
                before_net = MLP(self.seed, self.X.shape[1], self.features, self.descriptions)
                before_net.params = {key: np.asarray(value) for key, value in trace['before'].items()}
                _, recalculated = before_net.loss_gradients(
                    np.asarray(trace['X']), one_hot(np.asarray(trace['y'])), cfg.l2)
                for key in initial_net.params:
                    np.testing.assert_array_equal(trace['gradients'][key], recalculated[key])
                    expected_before = traces[offset - 1]['after'][key] if offset else initial_net.params[key]
                    np.testing.assert_array_equal(trace['before'][key], expected_before)
        for epoch in range(self.epochs):
            indices = [index for trace in traces[epoch * batches_per_epoch:(epoch + 1) * batches_per_epoch]
                       for index in trace['batch_indices_in_phase']]
            self.assertEqual(sorted(indices), list(range(len(self.X))))
        self.assertEqual([len(trace['y']) for trace in traces], [3, 3, 2, 3, 3, 2])
        for key, value in final_net.params.items():
            np.testing.assert_array_equal(traces[-1]['after'][key], value)
        return traces

    def test_sgd_records_each_real_update_and_signed_gradient(self):
        cfg, net, history, selected, path = self.train('sgd')
        self.assertEqual(selected, self.epochs)
        self.assertEqual(len(history), self.epochs)
        traces = self.audited_traces(cfg, net, path)
        negative_first_step = 0
        for offset, trace in enumerate(traces):
            self.assertNotIn('adam_m_before', trace)
            for key in net.params:
                before = np.asarray(trace['before'][key])
                after = np.asarray(trace['after'][key])
                gradient = np.asarray(trace['gradients'][key])
                velocity_before = np.asarray(trace['velocity_before'][key])
                velocity_after = np.asarray(trace['velocity_after'][key])
                expected_previous = traces[offset - 1]['velocity_after'][key] if offset else np.zeros_like(before)
                np.testing.assert_array_equal(velocity_before, expected_previous)
                expected_velocity = trace['momentum'] * velocity_before - trace['learning_rate'] * gradient
                np.testing.assert_allclose(velocity_after, expected_velocity, rtol=1e-14, atol=1e-16)
                np.testing.assert_allclose(after, before + expected_velocity, rtol=1e-14, atol=1e-16)
                if offset == 0:
                    negative = gradient < 0
                    negative_first_step += int(negative.sum())
                    self.assertTrue(np.all(after[negative] > before[negative]))
        self.assertGreater(negative_first_step, 0, 'O teste precisa exercitar gradientes negativos reais.')

    def audit_epoch_momentum_origin(self, cfg, net, history, path):
        traces = self.audited_traces(cfg, net, path)
        batches_per_epoch = math.ceil(len(self.X) / cfg.batch_size)
        first_epoch, second_epoch = traces[:batches_per_epoch], traces[batches_per_epoch:]
        initial_origin = {'kind': 'initial', 'momentum_initial': cfg.momentum_initial,
                          'seed': self.seed}
        for trace in first_epoch:
            self.assertEqual(trace['momentum_origin'], initial_origin)
            self.assertEqual(trace['momentum'], cfg.momentum_initial)

        # A origem da época 2 deve vir de TODO o treino nos dois estados salvos,
        # e não apenas dos gradientes do último lote da época 1.
        initial, initial_mean, initial_scale, _ = load_model(path.parent / 'epoca_00000.npz')
        epoch_one, epoch_mean, epoch_scale, _ = load_model(path.parent / 'epoca_00001.npz')
        for saved_mean, saved_scale in [(initial_mean, initial_scale), (epoch_mean, epoch_scale)]:
            np.testing.assert_array_equal(saved_mean, self.scaler.mean_)
            np.testing.assert_array_equal(saved_scale, self.scaler.scale_)
        initial_loss, initial_gradients = initial.loss_gradients(self.X, one_hot(self.y), cfg.l2)
        current_loss, current_gradients = epoch_one.loss_gradients(self.X, one_hot(self.y), cfg.l2)
        previous_vector, current_vector = flat_gradient(initial_gradients), flat_gradient(current_gradients)
        origin = second_epoch[0]['momentum_origin']
        self.assertEqual(origin['kind'], 'epoch_decision')
        self.assertEqual(origin['epoch'], 1)
        self.assertEqual(origin['seed'], self.seed)
        self.assertEqual(len(origin['previous_gradient']), net.n_parameters)
        self.assertEqual(len(origin['current_gradient']), net.n_parameters)
        np.testing.assert_array_equal(origin['previous_gradient'], previous_vector)
        np.testing.assert_array_equal(origin['current_gradient'], current_vector)
        self.assertEqual(origin['previous_loss'], initial_loss)
        self.assertEqual(origin['objective'], current_loss)
        denominator = np.linalg.norm(current_vector) * np.linalg.norm(previous_vector)
        alignment = float(np.clip(np.dot(current_vector, previous_vector) / denominator, -1, 1)) if denominator else 0.0
        self.assertAlmostEqual(origin['alignment'], alignment, places=14)
        self.assertEqual(origin['momentum_used'], cfg.momentum_initial)
        for name in ['adaptive_momentum', 'momentum_max', 'momentum_increase',
                     'momentum_decrease', 'adaptation_tolerance']:
            self.assertEqual(origin[name], getattr(cfg, name))
        expected_mu, expected_action, expected_reset = momentum_next(
            cfg.momentum_initial, initial_loss, current_loss, alignment, cfg)
        self.assertEqual(origin['momentum_next'], expected_mu)
        self.assertEqual(origin['action'], expected_action)
        self.assertIs(origin['reset'], expected_reset)
        self.assertEqual(history[0]['train_objective_L2'], current_loss)
        self.assertEqual(history[0]['momentum_action'], expected_action)
        self.assertEqual(history[0]['momentum_next'], expected_mu)
        self.assertEqual(history[1]['momentum_used'], expected_mu)

        if expected_reset:
            self.assertTrue(any(np.any(np.asarray(value) != 0)
                                for value in first_epoch[-1]['velocity_after'].values()),
                            'O reset deve exercitar velocidades anteriormente não nulas.')
        for offset, trace in enumerate(second_epoch):
            self.assertEqual(trace['momentum_origin'], origin)
            self.assertEqual(trace['momentum'], expected_mu)
            for key in net.params:
                before = np.asarray(trace['before'][key])
                if offset:
                    expected_before_velocity = second_epoch[offset - 1]['velocity_after'][key]
                else:
                    np.testing.assert_array_equal(before, epoch_one.params[key])
                    expected_before_velocity = np.zeros_like(before) if expected_reset else first_epoch[-1]['velocity_after'][key]
                np.testing.assert_array_equal(trace['velocity_before'][key], expected_before_velocity)
                expected_velocity = expected_mu * np.asarray(expected_before_velocity) - cfg.learning_rate * np.asarray(trace['gradients'][key])
                np.testing.assert_allclose(trace['velocity_after'][key], expected_velocity, rtol=1e-14, atol=1e-16)
                np.testing.assert_allclose(trace['after'][key], before + expected_velocity, rtol=1e-14, atol=1e-16)
        return origin

    def test_adaptive_momentum_origin_uses_complete_epoch_gradients_and_real_initialization(self):
        cfg, net, history, _, path = self.train('momentum_increase', adaptive=True)
        origin = self.audit_epoch_momentum_origin(cfg, net, history, path)
        self.assertEqual(origin['action'], 'aumentar_ate_limite')
        self.assertFalse(origin['reset'])
        self.assertLess(origin['objective'], origin['previous_loss'])
        self.assertGreater(origin['alignment'], 0)
        self.assertEqual(origin['momentum_next'], min(cfg.momentum_max,
                                                    cfg.momentum_initial + cfg.momentum_increase))

    def test_adaptive_momentum_origin_records_loss_increase_and_velocity_reset(self):
        # Taxa deliberadamente alta apenas para provocar piora neste caso de 8 linhas.
        cfg, net, history, _, path = self.train('momentum_reset', adaptive=True, learning_rate=5.0)
        origin = self.audit_epoch_momentum_origin(cfg, net, history, path)
        self.assertEqual(origin['action'], 'reduzir_e_zerar_velocidade')
        self.assertTrue(origin['reset'])
        self.assertGreater(origin['objective'], origin['previous_loss'] + cfg.adaptation_tolerance)
        self.assertEqual(origin['momentum_next'], cfg.momentum_initial * cfg.momentum_decrease)

    def test_adam_records_moments_and_bias_corrected_updates(self):
        cfg, net, _, _, path = self.train('adam', optimizer='adam')
        traces = self.audited_traces(cfg, net, path)
        for offset, trace in enumerate(traces):
            self.assertNotIn('velocity_before', trace)
            for key in net.params:
                before = np.asarray(trace['before'][key])
                gradient = np.asarray(trace['gradients'][key])
                previous_m = np.asarray(trace['adam_m_before'][key])
                previous_v = np.asarray(trace['adam_v_before'][key])
                expected_previous_m = traces[offset - 1]['adam_m_after'][key] if offset else np.zeros_like(before)
                expected_previous_v = traces[offset - 1]['adam_v_after'][key] if offset else np.zeros_like(before)
                np.testing.assert_array_equal(previous_m, expected_previous_m)
                np.testing.assert_array_equal(previous_v, expected_previous_v)
                expected_m = 0.9 * previous_m + 0.1 * gradient
                expected_v = 0.999 * previous_v + 0.001 * gradient ** 2
                np.testing.assert_allclose(trace['adam_m_after'][key], expected_m, rtol=1e-14, atol=1e-16)
                np.testing.assert_allclose(trace['adam_v_after'][key], expected_v, rtol=1e-14, atol=1e-16)
                m_corrected = expected_m / (1 - 0.9 ** trace['step'])
                v_corrected = expected_v / (1 - 0.999 ** trace['step'])
                expected_after = before - trace['learning_rate'] * m_corrected / (np.sqrt(v_corrected) + 1e-8)
                np.testing.assert_allclose(trace['after'][key], expected_after, rtol=1e-14, atol=1e-16)

    def test_disabling_memory_keeps_models_and_history_bitwise_identical(self):
        self.assertIs(Config().record_calculations, True)
        for optimizer in ['sgd_momentum', 'adam']:
            with self.subTest(optimizer=optimizer):
                _, recorded, recorded_history, recorded_epoch, recorded_path = self.train(
                    optimizer + '_recorded', optimizer, record=True, adaptive=True)
                _, disabled, disabled_history, disabled_epoch, disabled_path = self.train(
                    optimizer + '_disabled', optimizer, record=False, adaptive=True)
                self.assertTrue(recorded_path.is_file())
                self.assertFalse(disabled_path.exists())
                self.assertFalse(list(disabled_path.parent.glob('*.sqlite*')))
                self.assertEqual(update_index(disabled_path), {'count': 0, 'last_step': 0})
                self.assertEqual(recorded_epoch, disabled_epoch)
                self.assertEqual(recorded_history, disabled_history)
                for key in recorded.params:
                    self.assertEqual(recorded.params[key].dtype, disabled.params[key].dtype)
                    self.assertEqual(recorded.params[key].shape, disabled.params[key].shape)
                    self.assertEqual(recorded.params[key].tobytes(), disabled.params[key].tobytes())
                self.assertEqual(recorded.predict_proba(self.X).tobytes(), disabled.predict_proba(self.X).tobytes())
        with self.assertRaisesRegex(ValueError, 'record_calculations deve ser booleano'):
            replace(Config(), record_calculations='false').validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
