"""Memória local de atualizações reais, comprimida e consultada por passo."""
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import zlib


FORMAT_VERSION = 2
# Estes campos não mudam normalmente durante uma fase. Se mudarem, um novo
# contexto será gravado; nunca reutilizamos valores diferentes silenciosamente.
_CONTEXT_KEYS = frozenset({'seed', 'classes', 'optimizer', 'learning_rate', 'l2',
                         'phase', 'features', 'feature_descriptions', 'mean',
                         'scale', 'input_schema', 'pca'})


def _compress(payload: dict) -> bytes:
    # Nível 1 reduz CPU sem perda: o conteúdo JSON recuperado continua exato.
    return zlib.compress(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                   separators=(',', ':')).encode('utf8'), level=1)


class UpdateWriter:
    """Uma conexão por fase, com um COMMIT durável por atualização real.

    Novos arquivos usam user_version=2: contexto e origem do momentum são
    compartilhados, enquanto cada lote mantém seus próprios pesos/gradientes.
    Abrir um arquivo legado não o migra. Não há fila de passos em memória nem
    mudança em synchronous/journal_mode; close não é necessário para confirmar
    um append que já retornou. Use close/context manager ao terminar a fase.
    """

    def __init__(self, path: Path, *, _legacy_new: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._context_cache = None
        self._origin_cache = None
        try:
            connection = self._connection
            connection.execute('PRAGMA foreign_keys = ON')
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, FORMAT_VERSION):
                raise ValueError('Formato de memória de cálculo não suportado.')
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='updates'").fetchone()
            if not exists:
                if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchone():
                    raise ValueError('Arquivo não é uma memória de cálculo reconhecida.')
                self.version = 0 if _legacy_new else FORMAT_VERSION
                with connection:
                    # sqlite3 não abre transação implicitamente para DDL.
                    # Esquema e marcador de versão devem nascer juntos.
                    connection.execute('BEGIN')
                    if self.version == FORMAT_VERSION:
                        connection.execute('CREATE TABLE contexts (id INTEGER PRIMARY KEY, payload BLOB NOT NULL UNIQUE)')
                        connection.execute('CREATE TABLE momentum_origins (id INTEGER PRIMARY KEY, payload BLOB NOT NULL UNIQUE)')
                        connection.execute('CREATE TABLE updates (step INTEGER PRIMARY KEY, epoch INTEGER NOT NULL, '
                                           'batch INTEGER NOT NULL, payload BLOB NOT NULL, '
                                           'context_id INTEGER NOT NULL REFERENCES contexts(id), '
                                           'origin_id INTEGER REFERENCES momentum_origins(id))')
                        connection.execute('CREATE INDEX updates_epoch ON updates(epoch)')
                        connection.execute(f'PRAGMA user_version = {FORMAT_VERSION}')
                    else:
                        connection.execute('CREATE TABLE updates (step INTEGER PRIMARY KEY, epoch INTEGER NOT NULL, '
                                           'batch INTEGER NOT NULL, payload BLOB NOT NULL)')
            else:
                self.version = version
        except BaseException:
            self.close()
            raise

    def append(self, payload: dict) -> None:
        connection = self._connection
        if connection is None:
            raise ValueError('Gravador de memória já foi fechado.')
        if self.version == 0:
            blob = _compress(payload)
            with connection:
                connection.execute('INSERT INTO updates VALUES (?, ?, ?, ?)',
                                   (payload['step'], payload['epoch'], payload['batch'], blob))
            return

        context = {key: value for key, value in payload.items() if key in _CONTEXT_KEYS}
        origin = {'momentum_origin': payload['momentum_origin']} if 'momentum_origin' in payload else None
        batch = {key: value for key, value in payload.items()
                 if key not in _CONTEXT_KEYS and key != 'momentum_origin'}
        # Serializar antes da transação evita qualquer escrita para NaN/Inf.
        blob = _compress(batch)
        context_changed = self._context_cache is None or context != self._context_cache[0]
        origin_changed = origin is not None and (self._origin_cache is None or origin != self._origin_cache[0])
        context_blob = _compress(context) if context_changed else None
        origin_blob = _compress(origin) if origin_changed else None
        with connection:
            if context_changed:
                connection.execute('INSERT OR IGNORE INTO contexts(payload) VALUES (?)', (context_blob,))
                context_id = connection.execute('SELECT id FROM contexts WHERE payload = ?', (context_blob,)).fetchone()[0]
            else:
                context_id = self._context_cache[1]
            if origin_changed:
                connection.execute('INSERT OR IGNORE INTO momentum_origins(payload) VALUES (?)', (origin_blob,))
                origin_id = connection.execute('SELECT id FROM momentum_origins WHERE payload = ?', (origin_blob,)).fetchone()[0]
            else:
                origin_id = self._origin_cache[1] if origin is not None else None
            # INSERT recusa duplicatas; contexto/origem novos também voltam
            # atrás se esta atualização falhar. Só depois do COMMIT há cache.
            connection.execute('INSERT INTO updates VALUES (?, ?, ?, ?, ?, ?)',
                               (payload['step'], payload['epoch'], payload['batch'], blob, context_id, origin_id))
        if context_changed:
            self._context_cache = (deepcopy(context), context_id)
        if origin_changed:
            self._origin_cache = (deepcopy(origin), origin_id)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._context_cache = self._origin_cache = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def append_update(path: Path, payload: dict) -> None:
    """API avulsa preservada; para uma fase inteira, reutilize UpdateWriter."""
    # Mantém o blob completo para consumidores legados deste ponto de entrada.
    with UpdateWriter(path, _legacy_new=True) as writer:
        writer.append(payload)


def update_index(path: Path, max_epoch: int | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        return {'count': 0, 'last_step': 0}
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        query = 'SELECT COUNT(*), COALESCE(MAX(step),0) FROM updates'
        if max_epoch is None:
            row = connection.execute(query).fetchone()
        else:
            row = connection.execute(query + ' WHERE epoch <= ?', (max_epoch,)).fetchone()
    return {'count': row[0], 'last_step': row[1]}


def read_update(path: Path, step: int, max_epoch: int | None = None) -> dict:
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError('Escolha um passo de atualização válido.')
    if not Path(path).is_file():
        raise ValueError('Esta execução não possui memória de atualizações.')
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        if version == FORMAT_VERSION:
            row = connection.execute('SELECT u.epoch, u.payload, c.payload, o.payload FROM updates u '
                                     'JOIN contexts c ON c.id = u.context_id '
                                     'LEFT JOIN momentum_origins o ON o.id = u.origin_id WHERE u.step = ?',
                                     (step,)).fetchone()
        elif version == 0:
            row = connection.execute('SELECT epoch, payload FROM updates WHERE step = ?', (step,)).fetchone()
        else:
            raise ValueError('Formato de memória de cálculo não suportado.')
    if row is None or (max_epoch is not None and row[0] > max_epoch):
        raise ValueError('Atualização não pertence ao intervalo deste checkpoint.')
    payload = json.loads(zlib.decompress(row[1]))
    if version == FORMAT_VERSION:
        payload.update(json.loads(zlib.decompress(row[2])))
        if row[3] is not None:
            payload.update(json.loads(zlib.decompress(row[3])))
    return payload
