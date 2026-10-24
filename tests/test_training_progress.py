"""Progresso informa épocas concluídas sem participar da aprendizagem."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rede import Config, fit_network, run_experiment


class TestTrainingProgress(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(91).normal(size=(40, 3))
        self.y = np.array([0, 1] * 20)

    def test_progress_is_numerically_inert_for_both_optimizers(self):
        for optimizer in ('sgd_momentum', 'adam'):
            with self.subTest(optimizer=optimizer):
                cfg = Config(optimizer=optimizer, batch_size=7, max_epochs=3)
                plain, history, chosen = fit_network(self.X, self.y, cfg, 17, 3)
                events = []
                observed, other_history, other_chosen = fit_network(
                    self.X, self.y, cfg, 17, 3, progress=events.append)
                self.assertEqual(history, other_history)
                self.assertEqual(chosen, other_chosen)
                for key in plain.params:
                    np.testing.assert_array_equal(plain.params[key], observed.params[key])
                self.assertEqual([e['epoch'] for e in events], [0, 1, 2, 3, 3])
                self.assertEqual([e['event'] for e in events], ['start', 'epoch', 'epoch', 'epoch', 'complete'])
                self.assertEqual([e['phase_complete'] for e in events], [False] * 4 + [True])
                self.assertTrue(all(e['epochs_total'] == 3 for e in events))
                times = [e['phase_elapsed_seconds'] for e in events]
                self.assertEqual(times, sorted(times))

    def test_early_stop_reports_executed_epochs_and_selected_epoch_separately(self):
        events = []
        cfg = Config(max_epochs=10, patience=1, min_delta=100)
        _, history, chosen = fit_network(self.X, self.y, cfg, 17, 10,
            validation=(self.X, self.y), progress=events.append)
        self.assertEqual(chosen, 1)
        self.assertEqual(len(history), 2)
        self.assertEqual(events[-1]['epoch'], 2)
        self.assertEqual(events[-1]['epochs_total'], 10)
        self.assertEqual(events[-1]['chosen_epochs'], 1)
        self.assertTrue(events[-1]['phase_complete'])

    def test_epoch_event_follows_recording_and_recorder_closes(self):
        actions = []
        def record(net, entry, event):
            actions.append(('saved', entry['epoch'], event))
        record.close = lambda: actions.append(('closed',))
        def progress(detail):
            actions.append(('reported', detail['epoch'], detail['event']))
        fit_network(self.X, self.y, Config(), 17, 1, recorder=record, progress=progress)
        self.assertEqual(actions, [('saved', 0, 'inicial'), ('reported', 0, 'start'),
            ('saved', 1, 'epoca'), ('reported', 1, 'epoch'),
            ('saved', 1, 'selecionado'), ('reported', 1, 'complete'), ('closed',)])

    def test_recorder_closes_on_recording_or_progress_failure(self):
        for failure in ('initial', 'epoch', 'progress'):
            with self.subTest(failure=failure):
                closed, events = [], []
                def record(net, entry, event):
                    if ((failure == 'initial' and event == 'inicial') or
                            (failure == 'epoch' and entry['epoch'] == 1)):
                        raise OSError('falha de gravação simulada')
                record.close = lambda: closed.append(True)
                def progress(detail):
                    events.append(detail)
                    if failure == 'progress':
                        raise OSError('falha de callback simulada')
                with self.assertRaises(OSError):
                    fit_network(self.X, self.y, Config(), 17, 2, recorder=record, progress=progress)
                self.assertEqual(closed, [True])
                self.assertFalse(any(e['phase_complete'] for e in events))

    def test_experiment_reports_all_phases_and_legacy_completion_after_saving(self):
        events, legacy = [], []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            X = np.random.default_rng(21).normal(size=(100, 3))
            frame = pd.DataFrame(X, columns=['a', 'b', 'c'])
            frame.insert(0, 'diagnosis', ['M', 'B'] * 50)
            frame.insert(0, 'id', [f'synthetic_{i}' for i in range(100)])
            source = root / 'source.csv'
            frame.to_csv(source, index=False)
            output = root / 'run'
            def legacy_progress(done, total, message):
                legacy.append((done, total, message))
                if done == total:
                    self.assertTrue((output / 'CONCLUIDO.txt').is_file())
                    self.assertTrue((output / 'modelo_final.npz').is_file())
            run_experiment(source, output, Config(mode='kfold5', max_epochs=1, record_calculations=False),
                           legacy_progress, training_progress=events.append)
            starts = [e for e in events if e['event'] == 'start']
            self.assertEqual(len(starts), 12)
            self.assertEqual([(e['stage'], e['fold'], e['phase']) for e in starts],
                [('fold', fold, phase) for fold in range(1, 6) for phase in ('selecao_interna', 'retreino')]
                + [('final', None, phase) for phase in ('selecao_interna', 'retreino')])
            self.assertTrue(all(e['folds_total'] == 5 and e['updated_utc'].endswith('+00:00') for e in events))
            self.assertEqual([e['elapsed_seconds'] for e in events], sorted(e['elapsed_seconds'] for e in events))
            self.assertEqual(legacy[-1], (6, 6, 'Concluído e salvo'))
            self.assertTrue(events[-1]['phase_complete'])


if __name__ == '__main__':
    unittest.main()
