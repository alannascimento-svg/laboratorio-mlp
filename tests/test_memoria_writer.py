"""Persistência otimizada: todos os passos são duráveis e recuperados completos."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zlib

from memoria import FORMAT_VERSION, UpdateWriter, append_update, read_update, update_index


class TestUpdateWriter(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'memoria.sqlite3'

    def payload(self, step=1, epoch=1):
        return {'step': step, 'epoch': epoch, 'batch': step,
                'classes': ['CYT', 'ERL', 'MIT'], 'seed': 42,
                'optimizer': 'sgd_momentum', 'learning_rate': 0.001, 'l2': 0.0001,
                'phase': 'selecao_interna', 'features': ['x1', 'x2'],
                'feature_descriptions': ['Altura · á', 'Largura'],
                'mean': [3.2, -1.0], 'scale': [0.45, 2.0],
                'input_schema': {'columns': [{'index': 0, 'role': 'feature'}]},
                'X': [[0.25, -0.75]], 'y': [2], 'batch_indices_in_phase': [7],
                'before': {'W1': [[0.5, -0.75]], 'b1': [0.0, -0.0]},
                'after': {'W1': [[0.4, -0.70]], 'b1': [0.1, 0.0]},
                'gradients': {'W1': [[0.1, -0.05]], 'b1': [-0.1, 0.0]},
                'velocity_before': {'W1': [[0.0, 0.0]], 'b1': [0.0, 0.0]},
                'velocity_after': {'W1': [[-0.1, 0.05]], 'b1': [0.1, 0.0]},
                'momentum': 0.1,
                'momentum_origin': {'kind': 'epoch_decision', 'epoch': epoch - 1,
                                    'previous_gradient': [0.2, -0.3], 'current_gradient': [0.1, -0.2],
                                    'action': 'manter', 'reset': False},
                'future_field': {'preservado': [None, True, 'ç']}}

    def db_counts(self):
        with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as connection:
            return tuple(connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                         for table in ['updates', 'contexts', 'momentum_origins'])

    def test_complete_round_trip_reuses_context_and_epoch_origin(self):
        expected = [self.payload(1), self.payload(2), self.payload(3, epoch=2)]
        with UpdateWriter(self.path) as writer:
            for payload in expected:
                writer.append(payload)
                # Outro leitor já vê a atualização antes do close.
                self.assertEqual(read_update(self.path, payload['step']), payload)
            self.assertEqual(self.db_counts(), (3, 1, 2))
            with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as connection:
                self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0], FORMAT_VERSION)
                self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
                self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(), [])
                batch = json.loads(zlib.decompress(connection.execute('SELECT payload FROM updates WHERE step=1').fetchone()[0]))
                self.assertNotIn('input_schema', batch)
                self.assertNotIn('momentum_origin', batch)
        self.assertEqual(update_index(self.path), {'count': 3, 'last_step': 3})
        self.assertEqual(update_index(self.path, max_epoch=1), {'count': 2, 'last_step': 2})

    def test_legacy_file_read_and_append_without_migration(self):
        first, second = self.payload(), self.payload(2)
        append_update(self.path, first)
        with UpdateWriter(self.path) as writer:
            self.assertEqual(writer.version, 0)
            writer.append(second)
        with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0], 0)
            self.assertEqual(len(connection.execute('PRAGMA table_info(updates)').fetchall()), 4)
            blob = connection.execute('SELECT payload FROM updates WHERE step=1').fetchone()[0]
            self.assertEqual(json.loads(zlib.decompress(blob)), first)
        self.assertEqual(read_update(self.path, 1), first)
        self.assertEqual(read_update(self.path, 2), second)

    def test_new_file_reopen_and_single_append_api(self):
        with UpdateWriter(self.path) as writer:
            writer.append(self.payload())
        with UpdateWriter(self.path) as writer:
            writer.append(self.payload(2))
        append_update(self.path, self.payload(3))
        self.assertEqual(self.db_counts(), (3, 1, 1))
        self.assertEqual(read_update(self.path, 3), self.payload(3))

    def test_duplicate_rolls_back_new_shared_records_and_keeps_cache_valid(self):
        first = self.payload()
        with UpdateWriter(self.path) as writer:
            writer.append(first)
            before = self.path.read_bytes()
            duplicate = self.payload(epoch=2)
            duplicate['classes'] = ['A', 'B']
            with self.assertRaises(sqlite3.IntegrityError):
                writer.append(duplicate)
            self.assertEqual(self.path.read_bytes(), before)
            self.assertEqual(self.db_counts(), (1, 1, 1))
            self.assertEqual(read_update(self.path, 1), first)
            writer.append(self.payload(2))
            changed = {**duplicate, 'step': 3}
            writer.append(changed)
            self.assertEqual(read_update(self.path, 3), changed)
            self.assertEqual(self.db_counts(), (3, 2, 2))

    def test_failed_insert_rolls_back_context_and_origin(self):
        with UpdateWriter(self.path) as writer:
            with closing(sqlite3.connect(self.path)) as connection:
                connection.execute("CREATE TRIGGER fail_update BEFORE INSERT ON updates BEGIN SELECT RAISE(ABORT, 'simulated failure'); END")
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'simulated failure'):
                writer.append(self.payload())
            self.assertEqual(self.db_counts(), (0, 0, 0))
            with closing(sqlite3.connect(self.path)) as connection:
                connection.execute('DROP TRIGGER fail_update')
            writer.append(self.payload())
            self.assertEqual(read_update(self.path, 1), self.payload())

    def test_mutated_metadata_and_origin_are_preserved_not_cached_as_old_values(self):
        payload = self.payload()
        first = deepcopy(payload)
        with UpdateWriter(self.path) as writer:
            writer.append(payload)
            payload['step'] = 2
            payload['mean'][0] = 999
            payload['momentum_origin']['previous_gradient'][0] = -123
            writer.append(payload)
        self.assertEqual(read_update(self.path, 1), first)
        self.assertEqual(read_update(self.path, 2), payload)
        self.assertEqual(self.db_counts(), (2, 2, 2))

    def test_adam_and_optional_origin_are_not_invented(self):
        with UpdateWriter(self.path) as writer:
            adam = self.payload()
            del adam['momentum_origin']
            del adam['velocity_before']
            del adam['velocity_after']
            adam.update(optimizer='adam', adam_m_before={'W1': [[0.0]]}, adam_m_after={'W1': [[0.1]]},
                        adam_v_before={'W1': [[0.0]]}, adam_v_after={'W1': [[0.01]]})
            writer.append(adam)
            writer.append({'step': 2, 'epoch': 1, 'batch': 2, 'momentum_origin': None})
            writer.append({'step': 3, 'epoch': 1, 'batch': 3})
        self.assertEqual(read_update(self.path, 1), adam)
        self.assertIsNone(read_update(self.path, 2)['momentum_origin'])
        self.assertNotIn('momentum_origin', read_update(self.path, 3))

    def test_nan_and_infinite_rejected_without_any_update(self):
        with UpdateWriter(self.path) as writer:
            writer.append(self.payload())
            before = self.path.read_bytes()
            for field in ['before', 'mean', 'momentum_origin']:
                for invalid in [float('nan'), float('inf'), -float('inf')]:
                    with self.subTest(field=field, value=invalid):
                        payload = self.payload(2)
                        payload[field] = [invalid]
                        with self.assertRaises(ValueError):
                            writer.append(payload)
            self.assertEqual(self.path.read_bytes(), before)
            self.assertEqual(self.db_counts(), (1, 1, 1))
            writer.append(self.payload(2))

    def test_readers_are_read_only_and_checkpoint_limit_is_preserved(self):
        with UpdateWriter(self.path) as writer:
            writer.append(self.payload())
            writer.append(self.payload(2, epoch=2))
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        files = sorted(path.name for path in self.path.parent.iterdir())
        self.assertEqual(read_update(self.path, 1, max_epoch=1), self.payload())
        with self.assertRaisesRegex(ValueError, 'intervalo deste checkpoint'):
            read_update(self.path, 2, max_epoch=1)
        self.assertEqual(update_index(self.path, max_epoch=1), {'count': 1, 'last_step': 1})
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(files, sorted(path.name for path in self.path.parent.iterdir()))

    def test_committed_updates_survive_process_exit_without_close(self):
        script = """import os, sys
from memoria import UpdateWriter
writer = UpdateWriter(sys.argv[1])
writer.append({'step': 1, 'epoch': 1, 'batch': 1, 'mean': [1.0], 'momentum_origin': {'kind': 'initial'}})
writer.append({'step': 2, 'epoch': 1, 'batch': 2, 'mean': [1.0], 'momentum_origin': {'kind': 'initial'}})
os._exit(0)
"""
        subprocess.run([sys.executable, '-c', script, str(self.path)],
                       cwd=Path(__file__).resolve().parents[1], check=True, timeout=15)
        self.assertEqual(update_index(self.path), {'count': 2, 'last_step': 2})
        self.assertEqual(read_update(self.path, 2)['mean'], [1.0])
        self.assertEqual(read_update(self.path, 2)['momentum_origin'], {'kind': 'initial'})
        with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as connection:
            self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')

    def test_default_sqlite_durability_is_not_reduced(self):
        with UpdateWriter(self.path) as writer:
            self.assertEqual(writer._connection.execute('PRAGMA synchronous').fetchone()[0], 2)
            self.assertEqual(writer._connection.execute('PRAGMA journal_mode').fetchone()[0], 'delete')

    def test_close_idempotent_and_context_exception_releases_connection(self):
        writer = UpdateWriter(self.path)
        with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
            with writer:
                writer.append(self.payload())
                raise RuntimeError('simulated failure')
        writer.close()
        self.assertIsNone(writer._connection)
        with self.assertRaisesRegex(ValueError, 'já foi fechado'):
            writer.append(self.payload(2))
        self.assertEqual(read_update(self.path, 1), self.payload())


if __name__ == '__main__':
    unittest.main(verbosity=2)
