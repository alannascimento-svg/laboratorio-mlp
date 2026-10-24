"""Progresso HTTP em servidor efêmero; o motor é simulado, sem treinar uma base."""
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import app


class QuietHandler(app.Handler):
    def log_message(self, format, *args):
        pass


class TestTrainingProgressHTTP(unittest.TestCase):
    def setUp(self):
        self.saved_state = app.STATE.copy()
        self.addCleanup(self.restore_state)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def restore_state(self):
        with app.STATE_LOCK:
            app.STATE.clear()
            app.STATE.update(self.saved_state)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, payload=None):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            headers = {'Content-Type': 'application/json', 'X-Lab-Token': app.TOKEN}
            connection.request('GET' if payload is None else 'POST', path,
                               body=None if payload is None else json.dumps(payload), headers=headers)
            response = connection.getresponse()
            data = response.read()
            return response.status, json.loads(data) if 'application/json' in response.getheader('Content-Type', '') else data
        finally:
            connection.close()

    def test_active_reloaded_status_rejects_second_training_and_confirms_real_completion(self):
        entered, finish = threading.Event(), threading.Event()
        detail = {'stage': 'fold', 'fold': 2, 'folds_total': 5,
                  'phase': 'selecao_interna', 'epoch': 120, 'epochs_total': 800,
                  'phase_complete': False, 'phase_elapsed_seconds': 30.,
                  'fold_elapsed_seconds': 30., 'elapsed_seconds': 60.,
                  'updated_utc': '2026-09-19T20:00:00+00:00', 'event': 'epoch'}

        def fake_run(path, output, cfg, progress, input_schema=None, training_progress=None):
            progress(1, 6, 'Rodada 2/5')
            training_progress(detail)
            detail['epoch'] = 121  # A cópia no STATE não pode mudar junto com o emissor.
            entered.set()
            if not finish.wait(3):
                raise RuntimeError('Teste não liberou o motor simulado.')
            progress(6, 6, 'Concluído e salvo')
            return {'metrics': {'accuracy': .5}}

        with patch.object(app, 'ROOT', self.root), patch.object(app, 'imported_dataset', return_value=self.root/'synthetic.csv'), \
                patch.object(app, 'load_dataset'), patch.object(app, 'run_experiment', side_effect=fake_run):
            with ThreadPoolExecutor(max_workers=1) as pool:
                request = pool.submit(self.request, '/api/train', {'config': {'max_epochs': 1}})
                try:
                    self.assertTrue(entered.wait(3))
                    status, state = self.request('/api/status')
                    self.assertEqual(status, 200)
                    self.assertTrue(state['busy'])
                    self.assertEqual(state['status'], 'running')
                    self.assertEqual(state['detail']['epoch'], 120)
                    self.assertEqual((state['done'], state['total']), (1, 6))
                    self.assertRegex(state['run_id'], r'^r\d{8}_\d{6}_\d{6}$')
                    second_status, second = self.request('/api/train', {})
                    self.assertEqual(second_status, 409)
                    self.assertIn('Já existe', second['error'])
                    self.assertEqual(self.request('/api/status')[1], state)
                finally:
                    finish.set()
                status, response = request.result(timeout=3)
                self.assertEqual(status, 200)
                _, state = self.request('/api/status')
                self.assertFalse(state['busy'])
                self.assertEqual(state['status'], 'completed')
                self.assertEqual(state['done'], state['total'])
                self.assertEqual(state['run_id'], response['run_id'])
                self.assertIn('Todos os resultados foram salvos', state['message'])
        self.assertFalse(any(self.root.iterdir()), 'Teste não pode gravar resultados ou treinar a base real.')

    def test_failure_clears_old_phase_and_next_start_clears_error_before_validation(self):
        with app.STATE_LOCK:
            app.STATE.update(status='completed', detail={'epoch': 700}, run_id='old', done=6, total=6)
        observed = []

        def fail_load(*args):
            observed.append(self.request('/api/status')[1])
            raise ValueError('Base sintética inválida.')

        with patch.object(app, 'imported_dataset', return_value=self.root/'invalid.csv'), \
                patch.object(app, 'load_dataset', side_effect=fail_load):
            for _ in range(2):
                status, response = self.request('/api/train', {})
                self.assertEqual(status, 400)
                self.assertIn('Base sintética inválida', response['error'])
                _, state = self.request('/api/status')
                self.assertFalse(state['busy'])
                self.assertEqual(state['status'], 'error')
                self.assertIsNone(state['detail'])
                self.assertIsNone(state['run_id'])
                self.assertIn('Base sintética inválida', state['message'])
        for start in observed:
            self.assertTrue(start['busy'])
            self.assertEqual(start['status'], 'running')
            self.assertIsNone(start['detail'])
            self.assertIsNone(start['run_id'])
            self.assertEqual((start['done'], start['total']), (0, 1))
            self.assertNotIn('inválida', start['message'])
        self.assertFalse(app.TRAIN_LOCK.locked())

    def test_failure_during_training_hides_phase_and_preserves_run_identifier(self):
        def fake_failure(*args, training_progress=None, **kwargs):
            training_progress({'epoch': 2, 'phase': 'retreino'})
            raise OSError('Disco de teste indisponível.')

        with patch.object(app, 'ROOT', self.root), patch.object(app, 'imported_dataset'), \
                patch.object(app, 'load_dataset'), patch.object(app, 'run_experiment', side_effect=fake_failure):
            status, _ = self.request('/api/train', {})
        self.assertEqual(status, 400)
        _, state = self.request('/api/status')
        self.assertEqual(state['status'], 'error')
        self.assertIsNone(state['detail'])
        self.assertIsNotNone(state['run_id'])
        self.assertIn('Disco de teste', state['message'])
        self.assertFalse(state['busy'])

    def test_progress_script_is_served_locally(self):
        status, body = self.request('/training-progress.js')
        self.assertEqual(status, 200)
        self.assertIn(b'createPoller', body)

    def test_closed_client_does_not_change_completed_training_to_error(self):
        handler = object.__new__(QuietHandler)
        with patch.object(app, 'ROOT', self.root), patch.object(app, 'imported_dataset'), \
                patch.object(app, 'load_dataset'), patch.object(app, 'run_experiment', return_value={'metrics': {}}), \
                patch.object(handler, 'send_json', side_effect=BrokenPipeError('Página fechada.')):
            with self.assertRaises(BrokenPipeError):
                handler.train({})
        _, state = self.request('/api/status')
        self.assertEqual(state['status'], 'completed')
        self.assertFalse(state['busy'])
        self.assertIn('Todos os resultados foram salvos', state['message'])
        self.assertFalse(app.TRAIN_LOCK.locked())


if __name__ == '__main__':
    unittest.main()
