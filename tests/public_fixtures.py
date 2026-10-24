"""Fixtures descartáveis derivadas apenas da demonstração pública distribuída."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def legacy_example(root: Path) -> Path:
    """Simula artefatos anteriores ao esquema de entradas, sem mudar o exemplo.

    Os parâmetros e o normalizador continuam sendo os da demonstração pública.
    Somente a cópia temporária perde os campos recentes e os registros de lotes.
    """
    folder = root / 'resultados_exemplo'
    shutil.copytree(ROOT / 'resultados_exemplo', folder,
                    ignore=shutil.ignore_patterns('*.sqlite3', '*.sqlite3-*', '*.sqlite'))
    for filename in ('calculos.html',):
        shutil.copy2(ROOT / filename, root / filename)
    (root / 'dados').mkdir()
    shutil.copy2(ROOT / 'dados' / 'cancear.txt', root / 'dados' / 'cancear.txt')
    for path in folder.rglob('*.npz'):
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files
                      if key not in {'classes', 'codes'}}
        metadata = json.loads(str(arrays['metadata'].item()))
        for key in ('input_schema', 'feature_descriptions', 'classes', 'output_order', 'codes'):
            metadata.pop(key, None)
        arrays['metadata'] = np.array(json.dumps(metadata, ensure_ascii=False))
        np.savez_compressed(path, **arrays)
    config_path = folder / 'configuracao.json'
    config = json.loads(config_path.read_text(encoding='utf8'))
    config.pop('input_schema', None)
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf8')
    return folder


def digest_files(folder: Path) -> dict[str, str]:
    import hashlib
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob('*') if path.is_file()}
