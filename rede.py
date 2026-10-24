"""MLP: D -> ceil((D+S)/2) -> S. Backpropagation explícita.
Somente educacional. Teste externo nunca determina pesos, escala ou parada.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from resumos import fold_summary
from entrada import inspect_source, selected_data, TABULAR_PROFILE_TITLES
from preprocessamento import fit_preprocessor, source_features, transform_inputs, validate_pca
from memoria import UpdateWriter

VERSION = "2.4.0"
BASE_FEATURES = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
                 "concavity", "concave_points", "symmetry", "fractal_dimension"]
FEATURES = [f"{f}_{group}" for group in ["mean", "se", "worst"] for f in BASE_FEATURES]
CLASSES = np.array(["M", "B"])
CODES = np.array(["10", "01"])
WARNING = "Experimento educacional. Não usar para diagnóstico, triagem ou decisões clínicas."


def hidden_size(n_inputs: int, n_outputs: int = 2) -> int:
    """Regra do projeto, não teorema de arquitetura ótima. Ímpar: teto."""
    for n in (n_inputs, n_outputs):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("Dimensões devem ser inteiros positivos.")
    return (int(n_inputs) + int(n_outputs) + 1) // 2


@dataclass(frozen=True)
class Config:
    seed: int = 42
    mode: str = "variavel"
    max_epochs: int = 800
    batch_size: int = 32
    learning_rate: float = 0.001
    l2: float = 0.0001
    patience: int = 60
    min_delta: float = 0.00001
    validation_fraction: float = 0.15
    optimizer: str = "sgd_momentum"
    adaptive_momentum: bool = True
    momentum_initial: float = 0.10
    momentum_max: float = 0.90
    momentum_increase: float = 0.01
    momentum_decrease: float = 0.50
    adaptation_tolerance: float = 1e-8
    snapshot_interval: int = 25
    record_calculations: bool = True
    pca_enabled: bool = False
    pca_variance: float = 0.95

    def validate(self) -> None:
        for name in ["seed", "max_epochs", "batch_size", "patience", "snapshot_interval"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} deve ser um inteiro.")
        if not isinstance(self.adaptive_momentum, bool):
            raise ValueError("adaptive_momentum deve ser booleano.")
        if not isinstance(self.pca_enabled, bool):
            raise ValueError('pca_enabled deve ser booleano.')
        if (isinstance(self.pca_variance, bool) or not isinstance(self.pca_variance, (int, float, np.number))
                or not np.isfinite(self.pca_variance) or not 0 < self.pca_variance <= 1):
            raise ValueError('Variância preservada pelo PCA deve estar em (0, 1], por exemplo 0.95.')
        if not isinstance(self.record_calculations, bool):
            raise ValueError('record_calculations deve ser booleano.')
        if self.mode not in {"variavel", "kfold5", "kfold10"}:
            raise ValueError("Modo deve ser variavel, kfold5 ou kfold10.")
        if self.optimizer not in {"sgd_momentum", "adam"}:
            raise ValueError("Otimizador deve ser sgd_momentum ou adam.")
        if not 0 <= self.seed <= 1_000_000_000:
            raise ValueError("Semente deve estar entre 0 e 1.000.000.000.")
        if min(self.max_epochs, self.batch_size, self.patience, self.snapshot_interval) < 1:
            raise ValueError("Épocas, lote, paciência e intervalo devem ser positivos.")
        numeric = ["learning_rate", "l2", "min_delta", "validation_fraction", "momentum_initial",
                   "momentum_max", "momentum_increase", "momentum_decrease", "adaptation_tolerance"]
        if any(isinstance(getattr(self, k), bool) or not isinstance(getattr(self, k), (int,float,np.number))
               or not np.isfinite(getattr(self, k)) for k in numeric):
            raise ValueError("Hiperparâmetros devem ser números finitos.")
        if self.learning_rate <= 0 or min(self.l2, self.min_delta, self.adaptation_tolerance) < 0:
            raise ValueError("Taxa deve ser positiva; L2 e tolerâncias não negativos.")
        if not 0.05 <= self.validation_fraction <= 0.30:
            raise ValueError("Validação interna deve estar entre 5% e 30% do treino externo.")
        if not 0 <= self.momentum_initial <= self.momentum_max < 1:
            raise ValueError("Exigido 0 <= momentum inicial <= máximo < 1.")
        if not 0 < self.momentum_increase <= 1 or not 0 < self.momentum_decrease < 1:
            raise ValueError("Incremento deve estar em (0,1]; fator de redução em (0,1).")


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    source_lines: np.ndarray
    sha256: str
    features: list[str]
    feature_descriptions: list[str] | None = None
    input_schema: dict | None = None
    classes: list[str] = field(default_factory=lambda: ["M", "B"])
    groups: np.ndarray | None = None


def class_codes(classes):
    return ["".join("1" if i == j else "0" for j in range(len(classes))) for i in range(len(classes))]


def connected_groups(ids, X):
    """Componentes conexas de mesmo ID OU vetor selecionado idêntico, sem remover linhas."""
    parent = np.arange(len(X))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    seen_ids, seen_vectors = {}, {}
    for i, (identifier, row) in enumerate(zip(ids, X)):
        vector = tuple(float(value) for value in row)
        if identifier in seen_ids: union(i, seen_ids[identifier])
        if vector in seen_vectors: union(i, seen_vectors[vector])
        seen_ids[identifier], seen_vectors[vector] = i, i
    roots = [find(i) for i in range(len(X))]
    mapping = {root: i for i, root in enumerate(sorted(set(roots)))}
    return np.asarray([mapping[root] for root in roots], dtype=int)


def dataset_protocol(data: Dataset) -> dict:
    counts = {label: int(np.sum(data.y == i)) for i, label in enumerate(data.classes)}
    grouped = data.classes != ["M", "B"]
    group_counts = ({label: len(np.unique(data.groups[data.y == i]))
                     for i, label in enumerate(data.classes)} if grouped else counts.copy())
    modes = (["kfold5"] + (["kfold10"] if min(group_counts.values()) >= 10 else [])) if grouped else ["variavel", "kfold5", "kfold10"]
    return {"classes": data.classes.copy(), "class_counts": counts,
            "class_group_counts": group_counts, "supported_modes": modes,
            "grouped": grouped, "n_groups": len(np.unique(data.groups)) if grouped else len(data.y),
            "group_policy": "connected_id_or_selected_X_v1" if grouped else "reject_duplicates_legacy"}


def decode_source(raw: bytes) -> str:
    try:
        return raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        return raw.decode('latin-1')


def load_selected_dataset(raw: bytes, input_schema: dict) -> Dataset:
    """Valida a seleção de colunas mantendo o arquivo original e sua auditoria."""
    if not isinstance(input_schema, dict):
        raise ValueError('A configuração de entradas deve ser um objeto.')
    table = inspect_source(decode_source(raw), delimiter=input_schema.get('delimiter', 'auto'),
                           header=input_schema.get('header', 'auto'))
    selected = selected_data(table, input_schema)
    X = np.asarray(selected['X'], dtype=np.float64)
    y = np.asarray(selected['y'], dtype=np.int64)
    classes = selected['classes']
    grouped = classes != ['M', 'B']
    if not grouped and len(np.unique(X, axis=0)) != len(X):
        raise ValueError('Vetores de medidas repetidos após a seleção: revisar antes da validação cruzada.')
    if not grouped and (len(X) < 50 or min(np.bincount(y, minlength=2)) < 20):
        raise ValueError('Este protocolo requer >=50 amostras e >=20 exemplos de cada classe.')
    groups = connected_groups(selected['ids'], X) if grouped else None
    if grouped and (len(X) < 50 or min(len(np.unique(groups[y == i])) for i in range(len(classes))) < 5):
        raise ValueError('O protocolo multiclasse requer >=50 amostras e >=5 grupos de cada classe; IDs ou medidas iguais pertencem ao mesmo grupo.')
    return Dataset(X, y, np.asarray(selected['ids']), np.asarray(selected['source_lines']),
                   hashlib.sha256(raw).hexdigest(), selected['features'],
                   selected['descriptions'], selected['schema'], classes, groups)


def load_dataset(path: str | Path, input_schema: dict | None = None) -> Dataset:
    """TXT WDBC original ou CSV com cabeçalho id,diagnosis,<d medidas>.
    CSV nomeado permite d variável; TXT sem cabeçalho continua exigindo 30.
    """
    raw = Path(path).read_bytes()
    if input_schema is not None:
        return load_selected_dataset(raw, input_schema)
    # Perfis documentados também funcionam no CLI sem esquema manual.
    title = next((line.strip() for line in decode_source(raw).splitlines() if line.strip()), '')
    if title in TABULAR_PROFILE_TITLES or title.startswith('1. Title: Protein Localization Sites'):
        table = inspect_source(decode_source(raw))
        return load_selected_dataset(raw, {'delimiter': table['delimiter'], 'header': 'no',
            'columns': [{key: column[key] for key in ('index', 'role', 'description')}
                        for column in table['columns']]})
    try: text = raw.decode("utf-8-sig")
    except UnicodeDecodeError: text = raw.decode("latin-1")
    rows, line_numbers, started = [], [], False
    features, named = FEATURES.copy(), False
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        row = [v.strip() for v in next(csv.reader([line]))]
        if not started and len(row) >= 3 and [v.lower() for v in row[:2]] == ["id", "diagnosis"]:
            features = row[2:]
            if not 1 <= len(features) <= 256 or len(set(features)) != len(features):
                raise ValueError("Use de 1 a 256 características com nomes únicos.")
            if any(not f or len(f) > 100 or f.lower() in {"id", "diagnosis", "diagnostico", "diagnóstico"}
                   for f in features):
                raise ValueError("Nome de característica vazio, muito longo ou reservado.")
            started, named = True, True
            continue
        candidate = bool(re.match(r"^\s*\d+\s*,", line))
        if not candidate and not started: continue
        started = True
        if len(row) != len(features) + 2 or not row[0] or (not named and not row[0].isdigit()) or row[1] not in {"B", "M"}:
            raise ValueError(f"Linha {line_no}: esperados ID, B/M e {len(features)} medidas ({len(features)+2} colunas).")
        try: values = [float(v) for v in row[2:]]
        except ValueError as exc: raise ValueError(f"Linha {line_no}: medida vazia ou não numérica.") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"Linha {line_no}: NaN ou infinito não permitido.")
        rows.append((row[0], 0 if row[1] == "M" else 1, values)); line_numbers.append(line_no)
    if not rows: raise ValueError("Nenhum registro encontrado. CSV genérico: id,diagnosis,medida1,...")
    ids = np.array([r[0] for r in rows])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("IDs repetidos: revisar duplicações/grupos antes de dividir os dados.")
    X = np.array([r[2] for r in rows], dtype=np.float64)
    y = np.array([r[1] for r in rows], dtype=np.int64)
    if len(np.unique(X, axis=0)) != len(X):
        raise ValueError("Vetores de medidas repetidos: revisar antes da validação cruzada.")
    if len(X) < 50 or min(np.bincount(y, minlength=2)) < 20:
        raise ValueError("Este protocolo requer >=50 amostras e >=20 exemplos de cada classe.")
    return Dataset(X, y, ids, np.array(line_numbers), hashlib.sha256(raw).hexdigest(), features)


def one_hot(y: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Classe interna 0=M -> [1,0]; classe interna 1=B -> [0,1]."""
    y = np.asarray(y)
    if n_classes < 2 or y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError('Índices de classe incompatíveis com as saídas.')
    return np.eye(n_classes, dtype=np.float64)[y]


def wta(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError("WTA requer uma matriz finita com pelo menos duas colunas.")
    return np.argmax(scores, axis=1)  # Empate exato: M por convenção, registrado.


class MLP:
    def __init__(self, seed: int = 42, n_inputs: int = 30, features: list[str] | None = None,
                 feature_descriptions: list[str] | None = None, classes: list[str] | None = None):
        self.classes = list(classes) if classes is not None else ['M', 'B']
        if (not 2 <= len(self.classes) <= 256
                or any(not isinstance(label, str) or not label.strip() or len(label) > 100 for label in self.classes)
                or len(set(self.classes)) != len(self.classes)):
            raise ValueError('Classes da rede devem ser de 2 a 256 rótulos únicos e não vazios.')
        if set(self.classes) == {'M', 'B'} and self.classes != ['M', 'B']:
            raise ValueError('A ordem de saídas M/B deve permanecer [M,B].')
        if set(self.classes) != {'M', 'B'} and self.classes != sorted(self.classes):
            raise ValueError('Outras classes devem seguir ordem lexical determinística.')
        hidden = hidden_size(n_inputs, len(self.classes))
        self.n_inputs, self.n_hidden, self.n_outputs = n_inputs, hidden, len(self.classes)
        self.features = list(features) if features is not None else (FEATURES.copy() if n_inputs == 30
                          else [f"feature_{i+1:02d}" for i in range(n_inputs)])
        if len(self.features) != n_inputs or len(set(self.features)) != n_inputs:
            raise ValueError("Nomes de características incompatíveis com a arquitetura.")
        self.feature_descriptions = list(feature_descriptions) if feature_descriptions is not None else self.features.copy()
        if len(self.feature_descriptions) != n_inputs or any(not isinstance(v, str) or len(v) > 1000 for v in self.feature_descriptions):
            raise ValueError('Descrições de entradas incompatíveis com a arquitetura.')
        self.input_schema = None
        self.pca = None
        rng = np.random.default_rng(seed)
        self.params = {"W1": rng.normal(0, math.sqrt(2 / n_inputs), (n_inputs, hidden)),
                       "b1": np.zeros(hidden),
                       "W2": rng.normal(0, math.sqrt(1 / hidden), (hidden, self.n_outputs)), "b2": np.zeros(self.n_outputs)}

    @property
    def n_parameters(self): return sum(v.size for v in self.params.values())
    @property
    def architecture(self): return [self.n_inputs, self.n_hidden, self.n_outputs]

    def forward(self, X: np.ndarray):
        if X.ndim != 2 or X.shape[1] != self.n_inputs:
            raise ValueError(f"A rede recebe {self.n_inputs} características.")
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        h = np.maximum(z1, 0.0)
        logits = h @ p["W2"] + p["b2"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return z1, h, log_probs, np.exp(log_probs)

    def predict_proba(self, X): return self.forward(X)[3]

    def loss_gradients(self, X: np.ndarray, targets: np.ndarray, l2: float = 0.0):
        """Forward + backpropagation por regra da cadeia, sem biblioteca de autograd."""
        z1, h, log_probs, scores = self.forward(X)
        n = len(X)
        data_loss = -np.sum(targets * log_probs) / n
        penalty = 0.5 * l2 * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))
        delta2 = (scores - targets) / n
        delta1 = (delta2 @ self.params["W2"].T) * (z1 > 0)
        gradients = {"W1": X.T @ delta1 + l2 * self.params["W1"], "b1": delta1.sum(axis=0),
                     "W2": h.T @ delta2 + l2 * self.params["W2"], "b2": delta2.sum(axis=0)}
        return float(data_loss + penalty), gradients

    def cross_entropy(self, X, y):
        return float(-np.mean(self.forward(X)[2][np.arange(len(y)), y]))


class SGDMomentum:
    """v(t)=mu*v(t-1)-eta*g(t); theta(t)=theta(t-1)+v(t)."""
    def __init__(self, params: dict):
        self.velocity = {k: np.zeros_like(v) for k,v in params.items()}
    def reset(self):
        for value in self.velocity.values(): value.fill(0)
    def step(self, params: dict, gradients: dict, learning_rate: float, momentum: float):
        for key, grad in gradients.items():
            self.velocity[key] = momentum * self.velocity[key] - learning_rate * grad
            params[key] += self.velocity[key]


def momentum_next(mu: float, previous_loss: float, current_loss: float,
                  alignment: float, cfg: Config) -> tuple[float, str, bool]:
    """Heurística educacional explícita. Dados de TREINO apenas. Não é RPROP.
    Alinhamento é o cosseno entre gradientes completos consecutivos.
    Sinal de uma componente isolada nunca decide aumentar momentum.
    """
    if not cfg.adaptive_momentum: return mu, "fixo", False
    change = current_loss - previous_loss
    tol = cfg.adaptation_tolerance
    if change > tol or alignment < 0:
        return max(0.0, mu * cfg.momentum_decrease), "reduzir_e_zerar_velocidade", True
    if change < -tol and alignment > 0:
        return min(cfg.momentum_max, mu + cfg.momentum_increase), "aumentar_ate_limite", False
    return mu, "manter", False


def flat_gradient(gradients): return np.concatenate([gradients[k].ravel() for k in ["W1","b1","W2","b2"]])


def fit_network(X, y, cfg: Config, seed: int, epochs: int, validation=None,
                features: list[str] | None = None, recorder: Callable | None = None,
                feature_descriptions=None, input_schema=None, classes=None, progress=None,
                pca=None, raw_X=None):
    """Executa uma fase e libera seu gravador mesmo em caso de falha."""
    try:
        return _fit_network(X, y, cfg, seed, epochs, validation, features, recorder,
                            feature_descriptions, input_schema, classes, progress, pca, raw_X)
    finally:
        close = getattr(recorder, 'close', None)
        if close:
            close()


def _fit_network(X, y, cfg: Config, seed: int, epochs: int, validation=None,
                 features=None, recorder=None, feature_descriptions=None,
                 input_schema=None, classes=None, progress=None, pca=None, raw_X=None):
    """Reinicia pesos, momentum e Adam; grava histórico sem consultar teste."""
    phase_started = monotonic()
    def report(epoch, event, chosen_epochs=None):
        if progress:
            detail = {'event':event, 'epoch':epoch, 'epochs_total':epochs,
                      'phase_complete':event == 'complete',
                      'phase_elapsed_seconds':monotonic() - phase_started}
            if chosen_epochs is not None:
                detail['chosen_epochs'] = chosen_epochs
            progress(detail)
    cfg.validate()
    net, rng = MLP(seed, X.shape[1], features, feature_descriptions, classes), np.random.default_rng(seed + 1)
    net.input_schema = input_schema
    net.pca = deepcopy(pca)
    if pca is not None:
        raw_X = np.asarray(raw_X, dtype=float)
        if raw_X.ndim != 2 or len(raw_X) != len(X) or not np.isfinite(raw_X).all():
            raise ValueError('PCA requer as amostras originais correspondentes à fase.')
        validate_pca(pca, raw_X.shape[1], net.n_inputs)
    targets = one_hot(y, net.n_outputs)
    opt = SGDMomentum(net.params)
    m = {k: np.zeros_like(v) for k,v in net.params.items()}
    v = {k: np.zeros_like(a) for k,a in net.params.items()}
    history, step, mu = [], 0, cfg.momentum_initial
    best_loss, best_epoch, waited, best_params = float("inf"), 0, 0, None
    previous_loss, g = net.loss_gradients(X, targets, cfg.l2)
    previous_g = flat_gradient(g)
    if recorder: recorder(net, {"epoch": 0}, "inicial")
    report(0, 'start')
    update_recorder = getattr(recorder, 'record_update', None) if cfg.record_calculations else None
    momentum_origin = {'kind':'initial', 'momentum_initial':mu, 'seed':seed}
    for epoch in range(1, epochs + 1):
        mu_used = mu
        order = rng.permutation(len(X))
        for start in range(0, len(X), cfg.batch_size):
            idx = order[start:start+cfg.batch_size]
            loss, gradients = net.loss_gradients(X[idx], targets[idx], cfg.l2)
            if not np.isfinite(loss) or any(not np.isfinite(a).all() for a in gradients.values()):
                raise FloatingPointError("Treinamento divergiu; revise taxa e dados.")
            step += 1
            if update_recorder:
                trace = {'step':step, 'epoch':epoch, 'batch':start // cfg.batch_size + 1,
                    'classes':net.classes,
                    'optimizer':cfg.optimizer, 'learning_rate':cfg.learning_rate,
                    'momentum':mu_used, 'l2':cfg.l2, 'X':X[idx].tolist(), 'y':y[idx].tolist(),
                    'batch_indices_in_phase':idx.tolist(),
                    'before':{k:v.tolist() for k,v in net.params.items()},
                    'gradients':{k:v.tolist() for k,v in gradients.items()}}
                if net.pca is not None:
                    trace['raw_X'] = raw_X[idx].tolist()
                if cfg.optimizer == 'sgd_momentum':
                    trace['momentum_origin'] = momentum_origin
                    trace['velocity_before'] = {k:v.tolist() for k,v in opt.velocity.items()}
                else:
                    trace['adam_m_before'] = {k:v.tolist() for k,v in m.items()}
                    trace['adam_v_before'] = {k:v.tolist() for k,v in v.items()}
            if cfg.optimizer == "sgd_momentum":
                opt.step(net.params, gradients, cfg.learning_rate, mu_used)
            else:
                for key, grad in gradients.items():
                    m[key] = .9*m[key]+.1*grad
                    v[key] = .999*v[key]+.001*grad**2
                    net.params[key] -= cfg.learning_rate*(m[key]/(1-.9**step))/(np.sqrt(v[key]/(1-.999**step))+1e-8)
            if update_recorder:
                trace['after'] = {k:v.tolist() for k,v in net.params.items()}
                if cfg.optimizer == 'sgd_momentum':
                    trace['velocity_after'] = {k:v.tolist() for k,v in opt.velocity.items()}
                else:
                    trace['adam_m_after'] = {k:v.tolist() for k,v in m.items()}
                    trace['adam_v_after'] = {k:v.tolist() for k,v in v.items()}
                update_recorder(net, trace)
        objective, full_g = net.loss_gradients(X, targets, cfg.l2)
        full_g = flat_gradient(full_g)
        norm = float(np.linalg.norm(full_g))
        if not np.isfinite(objective) or not np.isfinite(norm):
            raise FloatingPointError("Perda ou gradiente não finito ao concluir a época.")
        denom = norm * float(np.linalg.norm(previous_g))
        alignment = float(np.clip(np.dot(full_g, previous_g)/denom, -1, 1)) if denom > 0 else 0.0
        if cfg.optimizer == "sgd_momentum":
            mu, action, reset = momentum_next(mu_used, previous_loss, objective, alignment, cfg)
            if update_recorder:
                # Esta decisão só será usada nos lotes da próxima época.
                momentum_origin = {'kind':'epoch_decision', 'epoch':epoch, 'seed':seed,
                    'previous_loss':previous_loss, 'objective':objective,
                    'previous_gradient':previous_g.tolist(), 'current_gradient':full_g.tolist(),
                    'alignment':alignment, 'momentum_used':mu_used, 'momentum_next':mu,
                    'action':action, 'reset':reset, 'adaptive_momentum':cfg.adaptive_momentum,
                    'momentum_max':cfg.momentum_max, 'momentum_increase':cfg.momentum_increase,
                    'momentum_decrease':cfg.momentum_decrease,
                    'adaptation_tolerance':cfg.adaptation_tolerance}
            if reset: opt.reset()
        else: action = "adam_beta1_0.9_beta2_0.999"
        entry = {"epoch": epoch, "train_cross_entropy": net.cross_entropy(X,y),
                 "train_objective_L2": objective, "objective_change": objective-previous_loss,
                 "gradient_norm": norm, "gradient_alignment": alignment,
                 "gradient_negative_fraction": float(np.mean(full_g < 0)),
                 "momentum_used": mu_used if cfg.optimizer == "sgd_momentum" else None,
                 "momentum_next": mu if cfg.optimizer == "sgd_momentum" else None,
                 "learning_rate": cfg.learning_rate, "momentum_action": action}
        previous_loss, previous_g = objective, full_g
        improved = False
        if validation is not None:
            val_loss = net.cross_entropy(*validation)
            if not np.isfinite(val_loss): raise FloatingPointError("Perda de validação não finita.")
            entry["validation_cross_entropy"] = val_loss
            if val_loss < best_loss - cfg.min_delta:
                best_loss, best_epoch, waited, improved = val_loss, epoch, 0, True
                best_params = {k:a.copy() for k,a in net.params.items()}
            else: waited += 1
        entry["best_internal_epoch_so_far"] = best_epoch if validation is not None else None
        history.append(entry)
        if recorder: recorder(net, entry, "melhor_interno" if improved else "epoca")
        report(epoch, 'epoch')
        if validation is not None and waited >= cfg.patience: break
    if validation is not None:
        if best_params is None: raise RuntimeError("Nenhum checkpoint interno válido.")
        net.params = best_params
    else: best_epoch = epochs
    if recorder: recorder(net, {"epoch": best_epoch}, "selecionado")
    report(len(history), 'complete', best_epoch)
    return net, history, best_epoch


def variable_sizes(n: int, seed: int) -> list[int]:
    """Primeiro teste ~10%; demais 10–20%; cobertura exata, sem resto minúsculo.

    Sorteio condicionado: só aceita tamanho que deixa uma partição viável do resto.
    Percentuais sempre relativos a n ORIGINAL, não ao conjunto ainda não testado.
    """
    low, high = math.ceil(n * 0.10), math.floor(n * 0.20)
    if n < 50 or low > high:
        raise ValueError("Poucas amostras para partições de 10–20%.")
    first = low
    rng = np.random.default_rng(seed)

    def feasible(remainder: int) -> bool:
        return remainder == 0 or (remainder >= low and math.ceil(remainder / high) <= remainder // low)

    sizes, remaining = [first], n - first
    while remaining:
        candidates = [s for s in range(low, min(high, remaining) + 1) if feasible(remaining - s)]
        if not candidates:
            raise ValueError("Não foi possível particionar com os limites estabelecidos.")
        size = int(rng.choice(candidates))
        sizes.append(size)
        remaining -= size
    return sizes


def _all_classes(indices, y):
    return np.array_equal(np.unique(y[indices]), np.unique(y))


def grouped_holdout(y, groups, fraction, seed):
    """Escolhe somente por rótulos, grupos e tamanho; nunca consulta pesos ou teste."""
    if min(len(np.unique(groups[y == label])) for label in np.unique(y)) < 2:
        raise ValueError('Validação interna requer ao menos dois grupos de cada classe no treino externo.')
    candidates = GroupShuffleSplit(n_splits=512, test_size=fraction, random_state=seed)
    best, best_score = None, None
    overall = np.bincount(y, minlength=int(y.max()) + 1) / len(y)
    for fit, val in candidates.split(np.arange(len(y)), y, groups):
        if len(val) / len(y) > .30 or not _all_classes(fit, y) or not _all_classes(val, y):
            continue
        score = (abs(len(val) / len(y) - fraction),
                 float(np.abs(np.bincount(y[val], minlength=len(overall)) / len(val) - overall).sum()))
        if best_score is None or score < best_score:
            best, best_score = (np.sort(fit), np.sort(val)), score
    if best is None:
        # Em bases pequenas, a fração pedida pode ter menos vagas do que
        # classes. Cobrir todas exige uma fração real maior, sempre auditada.
        unique_groups = np.unique(groups)
        members = {int(g): np.flatnonzero(groups == g) for g in unique_groups}
        labels_in = {g: set(y[ix].tolist()) for g, ix in members.items()}
        groups_of = {int(label): {g for g, labels in labels_in.items() if label in labels}
                     for label in np.unique(y)}
        rng = np.random.default_rng(seed)
        for _ in range(128):
            chosen, covered = set(), set()
            remaining_counts = {label: len(values) for label, values in groups_of.items()}
            while len(covered) < len(groups_of):
                missing = [label for label in groups_of if label not in covered]
                rng.shuffle(missing)
                label = min(missing, key=lambda value: len(groups_of[value] - chosen))
                possible = [g for g in sorted(groups_of[label] - chosen)
                            if all(remaining_counts[other] > 1 for other in labels_in[g])]
                if not possible:
                    break
                g = int(rng.choice(possible))
                chosen.add(g); covered.update(labels_in[g])
                for other in labels_in[g]: remaining_counts[other] -= 1
            if len(covered) != len(groups_of):
                continue
            count = sum(len(members[g]) for g in chosen)
            for g in rng.permutation(unique_groups):
                if (g not in chosen and all(remaining_counts[other] > 1 for other in labels_in[g])
                        and abs((count + len(members[g])) / len(y) - fraction) < abs(count / len(y) - fraction)):
                    chosen.add(int(g)); count += len(members[g])
                    for other in labels_in[g]: remaining_counts[other] -= 1
            val = np.flatnonzero(np.isin(groups, list(chosen)))
            fit = np.flatnonzero(~np.isin(groups, list(chosen)))
            if len(val) / len(y) > .30:
                continue
            score = (abs(len(val) / len(y) - fraction),
                     float(np.abs(np.bincount(y[val], minlength=len(overall)) / len(val) - overall).sum()))
            if best_score is None or score < best_score:
                best, best_score = (fit, val), score
    if best is None:
        raise ValueError('Não foi possível formar validação interna por grupos com todas as classes e no máximo 30% das amostras; ajuste a fração ou revise os grupos.')
    if set(groups[best[0]]) & set(groups[best[1]]):
        raise AssertionError('Grupo compartilhado entre treino e validação interna.')
    return best


def make_folds(y: np.ndarray, cfg: Config, groups=None) -> list[tuple[np.ndarray, np.ndarray]]:
    cfg.validate()
    indices = np.arange(len(y))
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape:
            raise ValueError('Grupos incompatíveis com as amostras.')
        if cfg.mode == 'variavel':
            raise ValueError('Bases multiclasse usam divisão por grupos: selecione K-fold 5 (ou K-fold 10 se houver grupos suficientes).')
        k = 5 if cfg.mode == 'kfold5' else 10
        group_counts = [len(np.unique(groups[y == label])) for label in np.unique(y)]
        if min(group_counts) < k:
            raise ValueError(f'K-fold {k} requer ao menos {k} grupos de cada classe; menor contagem: {min(group_counts)}. Selecione K-fold 5 quando disponível.')
        splits = None
        for attempt in range(256):
            candidate = list(StratifiedGroupKFold(k, shuffle=True,
                random_state=cfg.seed + attempt * 97).split(indices, y, groups))
            if all(_all_classes(train, y) and _all_classes(test, y) for train, test in candidate):
                splits = candidate
                break
        if splits is None:
            raise ValueError('Não foi possível formar folds por grupos com todas as classes; revise a base ou reduza o número de folds.')
        validate_folds(splits, len(y))
        if any(set(groups[train]) & set(groups[test]) for train, test in splits):
            raise AssertionError('Grupo compartilhado entre treino e teste.')
        return splits
    if cfg.mode != "variavel":
        k = 5 if cfg.mode == "kfold5" else 10
        splits = list(StratifiedKFold(k, shuffle=True, random_state=cfg.seed).split(indices, y))
    else:
        sizes = variable_sizes(len(y), cfg.seed)
        remaining = indices.copy()
        splits = []
        for i, size in enumerate(sizes):
            if size == len(remaining):
                test = remaining.copy()
                remaining = np.array([], dtype=int)
            else:
                remaining, test = train_test_split(
                    remaining, test_size=size, stratify=y[remaining], random_state=cfg.seed + 100 + i)
            # Reutilizar dados em TREINOS diferentes é permitido. Não em TESTES.
            train = np.setdiff1d(indices, test)
            splits.append((train, np.sort(test)))
    validate_folds(splits, len(y))
    return splits


def validate_folds(splits, n: int) -> None:
    count = np.zeros(n, dtype=int)
    for train, test in splits:
        if len(np.intersect1d(train, test)) or len(np.union1d(train, test)) != n:
            raise AssertionError("Treino/teste não formam partição disjunta e completa.")
        if len(np.unique(test)) != len(test) or len(np.unique(train)) != len(train):
            raise AssertionError("Índices repetidos dentro de uma partição.")
        count[test] += 1
    if not np.all(count == 1):
        raise AssertionError("Cada registro precisa aparecer EXATAMENTE uma vez em teste.")


def metrics(y: np.ndarray, scores: np.ndarray, classes=None) -> dict:
    predictions = wta(scores)
    classes = list(classes) if classes is not None else ['M', 'B']
    if len(classes) != scores.shape[1]:
        raise ValueError('Classes incompatíveis com os escores.')
    if classes != ['M', 'B']:
        indices = np.arange(len(classes))
        cm = confusion_matrix(y, predictions, labels=indices)
        per_class = {}
        for index, label in enumerate(classes):
            support = int(cm[index].sum())
            if not support:
                raise ValueError(f'Classe {label} ausente na avaliação; não calcular macro ocultando classes.')
            precision = float(precision_score(y == index, predictions == index, zero_division=0))
            recall = float(recall_score(y == index, predictions == index, zero_division=0))
            per_class[label] = {'precision': precision, 'recall': recall,
                'f1': float(f1_score(y == index, predictions == index, zero_division=0)), 'support': support}
        return {'n':len(y), 'correct':int(np.sum(predictions == y)), 'classes':classes,
                'accuracy':float(accuracy_score(y, predictions)),
                'precision_macro':float(np.mean([v['precision'] for v in per_class.values()])),
                'recall_macro':float(np.mean([v['recall'] for v in per_class.values()])),
                'f1_macro':float(np.mean([v['f1'] for v in per_class.values()])),
                'balanced_accuracy':float(balanced_accuracy_score(y, predictions)),
                'confusion_matrix_order':classes, 'confusion_matrix':cm.tolist(), 'per_class':per_class,
                'exact_ties':int(np.sum(np.sum(scores == scores.max(axis=1, keepdims=True), axis=1) > 1))}
    true_m, pred_m = (y == 0), (predictions == 0)
    # Matriz para apresentação: linhas/colunas B,M. M é classe positiva.
    cm = confusion_matrix(true_m, pred_m, labels=[False, True])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    return {
        "n": len(y), "correct": int(np.sum(predictions == y)), 'classes':classes,
        "accuracy": float(accuracy_score(y, predictions)),
        "sensitivity_M": float(recall_score(true_m, pred_m, zero_division=0)),
        "specificity_B": tn / (tn + fp) if tn + fp else None,
        "precision_M": float(precision_score(true_m, pred_m, zero_division=0)),
        "f1_M": float(f1_score(true_m, pred_m, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "roc_auc_M": float(roc_auc_score(true_m, scores[:, 0])) if len(np.unique(y)) == 2 else None,
        "true_negative_B": tn, "false_positive_B_as_M": fp,
        "false_negative_M_as_B": fn, "true_positive_M": tp,
        "confusion_matrix_order": ["B", "M"], "confusion_matrix": cm.tolist(),
        "exact_ties": int(np.sum(scores[:, 0] == scores[:, 1])),
    }


def write_json(path: Path, value) -> None:
    """Substituição atômica para não deixar JSON cortado em leitura concorrente."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_model(path: str | Path, net: MLP, scaler: StandardScaler, metadata: dict) -> None:
    """NPZ sem pickle, com nomes dinâmicos e normalizador do respectivo treino."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**metadata, 'feature_descriptions': net.feature_descriptions,
                'input_schema': net.input_schema, 'classes':net.classes,
                'output_order':net.classes, 'codes':dict(zip(net.classes, class_codes(net.classes)))}
    if net.pca is not None:
        metadata['pca'] = net.pca
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('wb') as f:
            np.savez_compressed(f, **net.params, mean=scaler.mean_, scale=scaler.scale_,
                features=np.array(net.features), classes=np.array(net.classes), codes=np.array(class_codes(net.classes)),
                metadata=np.array(json.dumps(metadata, ensure_ascii=False)))
        os.replace(temp, path)
    finally: temp.unlink(missing_ok=True)


def load_model(path: str | Path):
    with np.load(path, allow_pickle=False) as data:
        required = {'W1','b1','W2','b2','mean','scale','features','metadata'}
        if not required <= set(data.files) or set(data.files) - required - {'classes','codes'}:
            raise ValueError('Arquivo de modelo incompatível.')
        classes = data['classes'].tolist() if 'classes' in data.files else ['M','B']
        if 'codes' in data.files and data['codes'].tolist() != class_codes(classes):
            raise ValueError('Ordem de saídas incompatível.')
        features = data['features'].tolist()
        if not isinstance(features, list) or not 1 <= len(features) <= 256 or any(not isinstance(f,str) for f in features):
            raise ValueError('Nomes de entradas inválidos.')
        meta = json.loads(str(data['metadata'].item()))
        net = MLP(0, len(features), features, meta.get('feature_descriptions'), classes)
        if 'classes' in meta and meta['classes'] != net.classes:
            raise ValueError('Classes dos metadados diferem das saídas do modelo.')
        net.input_schema = meta.get('input_schema')
        net.pca = meta.get('pca')
        for key, arr in net.params.items():
            if data[key].shape != arr.shape or not np.isfinite(data[key]).all():
                raise ValueError(f'Pesos inválidos: {key}.')
            net.params[key] = data[key].copy()
        mean, scale = data['mean'].copy(), data['scale'].copy()
        if net.pca is not None:
            validate_pca(net.pca, len(mean), net.n_inputs)
        original_n = len(source_features(net))
        if mean.shape != (original_n,) or scale.shape != (original_n,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.all(scale > 0):
            raise ValueError('Normalizador inválido.')
        return net, mean, scale, meta


def predict_dataframe(model_path: str | Path, frame: pd.DataFrame) -> pd.DataFrame:
    net, mean, scale, _ = load_model(model_path)
    if frame.columns.duplicated().any(): raise ValueError('Colunas repetidas não permitidas.')
    original_features = source_features(net)
    missing = [f for f in original_features if f not in frame.columns]
    extra = [c for c in frame.columns if c not in original_features and c != 'id']
    if missing or extra:
        raise ValueError(f'CSV requer {len(original_features)} medidas nomeadas, ID opcional, SEM diagnóstico. Faltam: {missing}. Extras: {extra}.')
    if frame.empty: raise ValueError('CSV de previsão vazio.')
    X = frame[original_features].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all(): raise ValueError('Medidas devem ser numéricas e finitas.')
    standardized = (X - mean) / scale
    if not np.isfinite(standardized).all(): raise ValueError('Valores fora da faixa numérica.')
    scores = net.predict_proba(transform_inputs(net, X, mean, scale)); predicted = wta(scores)
    result = pd.DataFrame({'linha_csv': np.arange(1,len(X)+1), 'classe_prevista': np.asarray(net.classes)[predicted],
        'codigo_WTA': np.asarray(class_codes(net.classes))[predicted],
        **{f'score_{label}':scores[:,index] for index,label in enumerate(net.classes)},
        'empate_exato': np.sum(scores == scores.max(axis=1,keepdims=True),axis=1)>1,
        'medidas_com_abs_z_maior_5': np.sum(np.abs(standardized)>5,axis=1), 'aviso': WARNING})
    if 'id' in frame: result.insert(1,'id_auditoria',frame['id'].astype(str).to_numpy())
    return result


def weights_frame(net: MLP) -> pd.DataFrame:
    rows = []
    for key, arr in net.params.items():
        for index in np.ndindex(arr.shape):
            if key == 'W1': source, target = net.features[index[0]], f'H{index[1]+1:02d}'
            elif key == 'W2': source, target = f'H{index[0]+1:02d}', net.classes[index[1]]
            elif key == 'b1': source, target = 'bias', f'H{index[0]+1:02d}'
            else: source, target = 'bias', net.classes[index[0]]
            rows.append({'matriz': key, 'indice_0based': ','.join(map(str,index)),
                         'origem': source, 'destino': target, 'valor': float(arr[index])})
    return pd.DataFrame(rows)


def model_details(model_path):
    net, mean, scale, meta = load_model(model_path)
    return {'architecture': net.architecture, 'n_parameters': net.n_parameters,
            'classes':net.classes, 'codes':class_codes(net.classes),
            'features': net.features, 'feature_descriptions': net.feature_descriptions,
            'pca': net.pca, 'source_features':source_features(net),
            'feature_keys': net.features if net.pca is not None else [f"x{c['index']+1}" for c in net.input_schema['columns'] if c['role']=='feature'] if net.input_schema else None,
            'params': {k:v.tolist() for k,v in net.params.items()},
            'mean': mean.tolist(), 'scale': scale.tolist(), 'metadata': meta,
            'weights': weights_frame(net).to_dict(orient='records')}


def recorder_for(folder: Path, phase: str, scaler, cfg: Config, metadata: dict):
    """Escala+pesos inspecionáveis; não é retomada do estado do otimizador.
    Log escalar por época e último checkpoint persistidos imediatamente.
    Checkpoints numerados: época 0, a cada intervalo e escolhido final.
    """
    folder = folder / phase; folder.mkdir(parents=True, exist_ok=True)
    def record(net, entry, event):
        epoch = entry['epoch']
        info = {**metadata, 'epoch': epoch, 'phase': phase, 'event': event,
                'architecture': net.architecture, 'optimizer': cfg.optimizer,
                'resume_optimizer_supported': False, 'warning': WARNING}
        if event not in {'inicial', 'selecionado'}:
            with (folder / 'historico.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
            save_model(folder / 'ultimo.npz', net, scaler, info)
        if epoch == 0 or epoch % cfg.snapshot_interval == 0 or event == 'selecionado':
            save_model(folder / f'epoca_{epoch:05d}.npz', net, scaler, info)
        if event in {'melhor_interno', 'selecionado'}:
            save_model(folder / 'selecionado.npz', net, scaler, info)
    writer = None
    def record_update(net, payload):
        nonlocal writer
        if writer is None:
            writer = UpdateWriter(folder / 'memoria_calculo.sqlite3')
        writer.append({**metadata, **payload,
            'classes':net.classes,
            'phase':phase, 'features':net.features, 'feature_descriptions':net.feature_descriptions,
            'mean':scaler.mean_.tolist(), 'scale':scaler.scale_.tolist(),
            'input_schema':net.input_schema, **({'pca':net.pca} if net.pca is not None else {})})
    def close():
        if writer is not None:
            writer.close()
    record.record_update = record_update
    record.close = close
    return record


def select_epochs_and_refit(X, y, cfg: Config, seed: int, features=None, artifact_dir=None,
                           feature_descriptions=None, input_schema=None, classes=None, groups=None,
                           training_progress=None):
    """Ajusta normalização/PCA só no treino pertinente; teste não é recebido."""
    if groups is None:
        idx_fit, idx_val = train_test_split(np.arange(len(y)), test_size=cfg.validation_fraction,
                                          stratify=y, random_state=seed)
    else:
        idx_fit, idx_val = grouped_holdout(y, np.asarray(groups), cfg.validation_fraction, seed)
    original_features = list(features) if features is not None else MLP(0, X.shape[1], classes=classes).features
    original_descriptions = list(feature_descriptions) if feature_descriptions is not None else original_features.copy()
    def prepare(raw, phase):
        started = monotonic()
        scaler = fit_preprocessor(raw, enabled=cfg.pca_enabled, variance_target=cfg.pca_variance)
        pca = scaler.pca
        if pca is not None:
            pca.update(source_features=original_features.copy(),
                       source_feature_descriptions=original_descriptions.copy())
            names = [f'PC{i+1}' for i in range(pca['n_components'])]
            descriptions = [f'Componente principal {i+1} · {100*v:.2f}% da variância'
                            for i,v in enumerate(pca['explained_variance_ratio'])]
        else:
            names, descriptions = original_features, original_descriptions
        def report(detail):
            if training_progress:
                training_progress({**detail, 'phase':phase,
                    'phase_elapsed_seconds':monotonic() - started})
        return scaler, names, descriptions, report
    inner_scaler, inner_features, inner_descriptions, inner_progress = prepare(X[idx_fit], 'selecao_interna')
    rec = recorder_for(artifact_dir, 'selecao_interna', inner_scaler, cfg,
                      {'seed':seed+1}) if artifact_dir else None
    _, ih, chosen = fit_network(inner_scaler.transform(X[idx_fit]), y[idx_fit], cfg, seed+1,
        cfg.max_epochs, (inner_scaler.transform(X[idx_val]), y[idx_val]), inner_features, rec,
        inner_descriptions, input_schema, classes, progress=inner_progress,
        pca=inner_scaler.pca, raw_X=X[idx_fit] if inner_scaler.pca is not None else None)
    outer_scaler, outer_features, outer_descriptions, outer_progress = prepare(X, 'retreino')
    rec = recorder_for(artifact_dir, 'retreino', outer_scaler, cfg,
                      {'seed':seed+2}) if artifact_dir else None
    net, fh, _ = fit_network(outer_scaler.transform(X), y, cfg, seed+2, chosen,
                            features=outer_features, recorder=rec,
                            feature_descriptions=outer_descriptions, input_schema=input_schema, classes=classes,
                            progress=outer_progress, pca=outer_scaler.pca,
                            raw_X=X if outer_scaler.pca is not None else None)
    return net, outer_scaler, chosen, ih, fh, idx_fit, idx_val, inner_scaler


def metric_highlights(rows: list[dict], id_key: str = 'fold') -> dict:
    """Mantém todos os empates. Destaques descritivos, nunca critério de refit."""
    result = {}
    for metric, lower in [('accuracy',False),('sensitivity_M',False),('specificity_B',False),
                          ('f1_M',False),('false_negative_M_as_B',True),
                          ('precision_macro',False),('recall_macro',False),('f1_macro',False)]:
        valid = [r for r in rows if r.get(metric) is not None]
        if not valid: continue
        value = (min if lower else max)(float(r[metric]) for r in valid)
        result[metric] = {'value':value, 'ids':[r[id_key] for r in valid
                         if math.isclose(float(r[metric]),value,rel_tol=1e-12,abs_tol=1e-12)]}
    return result


def experiment_signature(folder: Path) -> str:
    """Comparação somente mesma base (bytes), mesmo protocolo e mesmas partições.
    Também funciona nos arquivos v1, sem alterar o experimento anterior.
    """
    conf = json.loads((folder/'configuracao.json').read_text(encoding='utf8'))
    splits = [json.loads((p/'indices.json').read_text()) for p in sorted((folder/'folds').iterdir())
              if p.is_dir() and (p/'indices.json').exists()]
    signature = {'data_sha256':conf['data_sha256'], 'mode':conf['mode'],
                 'validation_fraction':conf['validation_fraction'], 'splits':splits}
    if conf.get('input_schema') is not None:
        schema = conf['input_schema']
        signature['input_selection'] = {'protocol':'column_selection_v1',
            'delimiter':schema['delimiter'], 'header':schema['header'],
            'columns':[{k:c[k] for k in ('index','role')} for c in schema['columns']]}
    if conf.get('pca_enabled'):
        signature['preprocessing'] = {'policy':'standard_scaler_pca_train_only_full_svd_v1',
                                      'variance_target':conf['pca_variance']}
    if conf.get('classification_protocol'):
        signature['classification_protocol'] = conf['classification_protocol']
        signature['classes'] = conf['classes']
        signature['group_policy'] = conf['group_policy']
    return hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()


def save_prediction(model_path: Path, text: str, output_root: Path):
    frame = pd.read_csv(__import__('io').StringIO(text),dtype={'id':str})
    result = predict_dataframe(model_path,frame)
    name = 'p'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:6]
    target = Path(output_root)/name; target.mkdir(parents=True,exist_ok=False)
    (target/'entrada.csv').write_text(text,encoding='utf8')
    result.to_csv(target/'previsoes.csv',index=False)
    write_json(target/'metadados.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'model_sha256':hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        'model_path':str(model_path), 'n':len(result), 'warning':WARNING})
    return result, target


def run_experiment(dataset_path: str | Path, output_dir: str | Path, cfg: Config = Config(),
                   progress: Callable[[int,int,str],None] | None = None,
                   input_schema: dict | None = None, training_progress: Callable[[dict], None] | None = None) -> dict:
    cfg.validate(); data = load_dataset(dataset_path, input_schema)
    folds = make_folds(data.y, cfg, data.groups)
    # Falhar antes de criar artefatos se alguma validação agrupada não for viável.
    if data.groups is not None:
        for fold_no, (train, _) in enumerate(folds, 1):
            grouped_holdout(data.y[train], data.groups[train], cfg.validation_fraction, cfg.seed + 1000 * fold_no)
        grouped_holdout(data.y, data.groups, cfg.validation_fraction, cfg.seed + 100_000)
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('A pasta de saída deve estar vazia; escolha outra para preservar o histórico.')
    start = datetime.now(timezone.utc).isoformat()
    write_json(output/'status.json',{'state':'em_execucao','created_utc':start,'version':VERSION})
    try:
        return _run(data, Path(dataset_path), output, cfg, progress, start, folds, training_progress)
    except BaseException as exc:
        write_json(output/'status.json',{'state':'interrompido_ou_falhou','created_utc':start,
            'error':f'{type(exc).__name__}: {exc}', 'version':VERSION,
            'note':'Artefatos já gravados preservados. Encerramento forçado pode não atualizar este status.'})
        raise


def _run(data, dataset_path, output, cfg, progress, start, folds=None, training_progress=None):
    run_started = monotonic()
    timing = {'folds':[], 'final':{}, 'clock':'monotonic', 'includes_persistence':True}
    def stage_progress(stage, stage_timing, fold=None):
        stage_started = monotonic()
        def report(detail):
            if detail['event'] == 'complete':
                key = 'selection_seconds' if detail['phase'] == 'selecao_interna' else 'refit_seconds'
                stage_timing[key] = detail['phase_elapsed_seconds']
            if training_progress:
                training_progress({**detail, 'stage':stage, 'fold':fold, 'folds_total':len(folds),
                    'fold_elapsed_seconds':monotonic() - stage_started,
                    'elapsed_seconds':monotonic() - run_started,
                    'updated_utc':datetime.now(timezone.utc).isoformat()})
        return report
    X,y = data.X,data.y; features=data.features
    classes, codes = data.classes, class_codes(data.classes)
    protocol = dataset_protocol(data)
    arch=None if cfg.pca_enabled else MLP(0,X.shape[1],features,classes=classes).architecture
    folds=make_folds(y,cfg,data.groups) if folds is None else folds
    scores_oof=np.full((len(y),len(classes)),np.nan); fold_of=np.zeros(len(y),dtype=int); coverage=fold_of.copy()
    fold_rows, history_rows, membership = [], [], []
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn']}
    write_json(output/'configuracao.json',{**asdict(cfg),'version':VERSION,'architecture':arch,
        'hidden_formula':'ceil((n_inputs + n_outputs) / 2)','features':features,
        'architecture_scope':'modelo_final; cada fase ajusta suas próprias entradas efetivas',
        'preprocessing_policy':'standard_scaler_pca_train_only_full_svd_v1' if cfg.pca_enabled else 'standard_scaler_train_only',
        'feature_descriptions':data.feature_descriptions or features, 'input_schema':data.input_schema,
        'classes':classes,'output_order':classes,'codes':dict(zip(classes,codes)), 'python':platform.python_version(),
        'classification_protocol':'explicit_classes_groups_v1','group_policy':protocol['group_policy'],
        'internal_validation_policy':('grouped_all_classes_requested_fraction_max_30pct_v1' if data.groups is not None else 'legacy_stratified_holdout'),
        'class_counts':protocol['class_counts'],'class_group_counts':protocol['class_group_counts'],
        'packages':versions,'data_sha256':data.sha256, 'created_utc':start,
        'backpropagation':True, 'controller':'loss_and_full_gradient_alignment_v1',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    # Reprodutibilidade local: preservar os bytes efetivamente recebidos.
    shutil.copyfile(dataset_path,output/'dados_utilizados.txt')
    if data.input_schema is not None:
        write_json(output/'esquema_entradas.json',data.input_schema)
    if data.groups is not None:
        pd.DataFrame({'sample':np.arange(1,len(y)+1), 'linha_arquivo':data.source_lines,
            'id_auditoria':data.ids, 'grupo':data.groups, 'classe':np.asarray(classes)[y]}).to_csv(
                output/'grupos_auditoria.csv',index=False)
    for fold_no,(train,test) in enumerate(folds,1):
        fold_started = monotonic()
        fold_timing = {'fold':fold_no}
        seed=cfg.seed+1000*fold_no
        if progress: progress(fold_no-1,len(folds)+1,f'Rodada {fold_no}/{len(folds)}: {len(train)} treino, {len(test)} teste')
        fold_dir=output/'folds'/f'fold_{fold_no:02d}';fold_dir.mkdir(parents=True)
        net,scaler,chosen,ih,fh,inner_fit,inner_val,inner_scaler=select_epochs_and_refit(
            X[train],y[train],cfg,seed,features,fold_dir/'checkpoints',
            data.feature_descriptions,data.input_schema,classes,
            data.groups[train] if data.groups is not None else None,
            training_progress=stage_progress('fold', fold_timing, fold_no))
        scores=net.predict_proba(scaler.transform(X[test]));scores_oof[test]=scores;fold_of[test]=fold_no;coverage[test]+=1
        values=metrics(y[test],scores,classes)
        fold_rows.append({'fold':fold_no,'n_train':len(train),'n_test':len(test),
            'n_inputs':net.n_inputs, 'n_hidden':net.n_hidden,
            'selection_n_inputs':inner_scaler.pca['n_components'] if inner_scaler.pca is not None else len(features),
            'pca_retained_variance':net.pca['retained_variance'] if net.pca is not None else None,
            'train_percent':100*len(train)/len(y),'test_percent':100*len(test)/len(y),
            **({f'test_{label}':int(np.sum(y[test]==i)) for i,label in enumerate(classes)}),
            'inner_fit_n':len(inner_fit),'inner_validation_n':len(inner_val),
            'inner_validation_fraction_requested':cfg.validation_fraction,
            'inner_validation_fraction_effective':len(inner_val)/len(train),
            'chosen_epochs':chosen,'epochs_run_selection':len(ih),
            **{k:v for k,v in values.items() if not isinstance(v,(list,dict))}})
        for phase,hist in [('selecao_interna',ih),('retreino_externo',fh)]:
            history_rows += [{'fold':fold_no,'phase':phase,**h} for h in hist]
        fit_global,val_global=train[inner_fit],train[inner_val]
        for role,indices in [('treino_interno',fit_global),('validacao_interna',val_global),('teste_externo',test)]:
            membership += [{'fold':fold_no,'sample':int(i+1),'id_auditoria':data.ids[i],'role':role} for i in indices]
        save_model(fold_dir/'modelo.npz',net,scaler,{'fold':fold_no,'epochs':chosen,'seed':seed+2,
                   'data_sha256':data.sha256,'optimizer':cfg.optimizer,'warning':WARNING,
                   'group_policy':protocol['group_policy']})
        write_json(fold_dir/'metricas.json',values)
        weights_frame(net).to_csv(fold_dir/'pesos.csv',index=False)
        pd.DataFrame({'feature':features,'mean_fit_interno':inner_scaler.mean_,
                      'scale_fit_interno':inner_scaler.scale_,'mean_treino_externo':scaler.mean_,
                      'scale_treino_externo':scaler.scale_}).to_csv(fold_dir/'normalizacao.csv',index=False)
        write_json(fold_dir/'indices.json',{'train_0based':train.tolist(),'test_0based':test.tolist(),
            'inner_fit_0based':fit_global.tolist(),'inner_validation_0based':val_global.tolist(),
            **({'group_policy':protocol['group_policy'],
                'train_groups':np.unique(data.groups[train]).tolist(),'test_groups':np.unique(data.groups[test]).tolist(),
                'inner_fit_groups':np.unique(data.groups[fit_global]).tolist(),
                'inner_validation_groups':np.unique(data.groups[val_global]).tolist(),
                'validation_fraction_requested':cfg.validation_fraction,
                'validation_fraction_effective':len(inner_val)/len(train)} if data.groups is not None else {})})
        pd.DataFrame(fold_rows).to_csv(output/'rodadas.csv',index=False)
        pd.DataFrame(history_rows).to_csv(output/'historico_treinamento.csv',index=False)
        pd.DataFrame(membership).to_csv(output/'particoes_auditoria.csv',index=False)
        write_json(output/'status.json',{'state':'em_execucao','created_utc':start,'folds_completed':fold_no})
        fold_timing['total_seconds'] = monotonic() - fold_started
        timing['folds'].append(fold_timing)
        write_json(output/'timing.json', {**timing,'total_seconds':monotonic() - run_started})
    if not np.all(coverage==1) or not np.isfinite(scores_oof).all(): raise AssertionError('Cobertura de teste incorreta.')
    predicted=wta(scores_oof)
    oof=pd.DataFrame({'amostra':np.arange(1,len(y)+1),'linha_arquivo':data.source_lines,'id_auditoria':data.ids,
        'fold_teste':fold_of,'vezes_em_teste':coverage,'classe_real':np.asarray(classes)[y],'codigo_real':np.asarray(codes)[y],
        'classe_prevista':np.asarray(classes)[predicted],'codigo_WTA':np.asarray(codes)[predicted],
        **{f'score_{label}':scores_oof[:,i] for i,label in enumerate(classes)},'acertou':predicted==y})
    oof.to_csv(output/'previsoes_oof.csv',index=False);oof[~oof.acertou].to_csv(output/'erros_oof.csv',index=False)
    total=metrics(y,scores_oof,classes)
    total.update({'test_coverage':'Cada registro testado exatamente uma vez','n_folds':len(folds),
        'architecture':arch,'trainable_parameters':net.n_parameters,'class_counts':protocol['class_counts'],
        **({'B':int(sum(y==1)),'M':int(sum(y==0))} if classes == ['M','B'] else {}),
        'aggregation':'Métricas das N previsões OOF, não média simples de grupos desiguais.',
        'external_clinical_validation':False,'warning':WARNING})
    write_json(output/'metricas_oof.json',total)
    write_json(output/'resumo_rodadas.json',fold_summary(fold_rows, classes=classes))
    write_json(output/'destaques_rodadas.json',{'highlights':metric_highlights(fold_rows),
        'warning':'Destaques descritivos. Amostras distintas por rodada. Não selecionar modelo final pelo teste.'})
    if progress:progress(len(folds),len(folds)+1,'Modelo final: seleção interna e retreino completo, sem novo teste')
    final_started = monotonic()
    final_seed=cfg.seed+100_000
    net,scaler,chosen,ih,fh,inner_fit,inner_val,inner_scaler=select_epochs_and_refit(X,y,cfg,final_seed,
            features,output/'checkpoints_final',data.feature_descriptions,data.input_schema,classes,data.groups,
            training_progress=stage_progress('final', timing['final']))
    arch = net.architecture
    config_path = output/'configuracao.json'
    config_saved = json.loads(config_path.read_text(encoding='utf8'))
    config_saved['architecture'] = arch
    write_json(config_path, config_saved)
    total.update(architecture=arch, trainable_parameters=net.n_parameters,
                 pca_enabled=cfg.pca_enabled, architecture_scope='modelo_final',
                 fold_architectures=[[row['n_inputs'], row['n_hidden'], len(classes)] for row in fold_rows])
    write_json(output/'metricas_oof.json', total)
    meta={'architecture':arch,'trainable_parameters':net.n_parameters,'final_training_n':len(y),
        'pca':net.pca, 'source_features':features,
        'feature_descriptions':net.feature_descriptions,'input_schema':data.input_schema,
        'final_epochs':chosen,'seed':final_seed+2,'classes':classes,'output_order':classes,'codes':dict(zip(classes,codes)),
        'group_policy':protocol['group_policy'],
        'config':asdict(cfg),'data_sha256':data.sha256,'new_independent_test':False,'version':VERSION,'warning':WARNING}
    save_model(output/'modelo_final.npz',net,scaler,meta)
    weights_frame(net).to_csv(output/'pesos_final.csv',index=False)
    write_json(output/'rede_final.json',model_details(output/'modelo_final.npz'))
    write_json(output/'modelo_final_metadados.json',meta)
    write_json(output/'modelo_final_indices.json',{'inner_fit_0based':inner_fit.tolist(),'inner_validation_0based':inner_val.tolist(),
        **({'validation_fraction_requested':cfg.validation_fraction,
            'validation_fraction_effective':len(inner_val)/len(y),
            'inner_fit_groups':np.unique(data.groups[inner_fit]).tolist(),
            'inner_validation_groups':np.unique(data.groups[inner_val]).tolist()} if data.groups is not None else {})})
    pd.DataFrame(ih).to_csv(output/'modelo_final_selecao.csv',index=False)
    pd.DataFrame(fh).to_csv(output/'modelo_final_treinamento.csv',index=False)
    pd.DataFrame({'feature':features,'mean_fit_interno':inner_scaler.mean_,'scale_fit_interno':inner_scaler.scale_,
        'mean_final':scaler.mean_,'scale_final':scaler.scale_}).to_csv(output/'normalizacao_final.csv',index=False)
    pd.DataFrame(columns=['id']+features).to_csv(output/'modelo_previsao.csv',index=False)
    write_json(output/'comparacao.json',{'signature':experiment_signature(output),
        'warning':'Ranking OOF exploratório; ajustes repetidos por esses resultados exigem nova avaliação independente.'})
    timing['final']['total_seconds'] = monotonic() - final_started
    timing['total_seconds'] = monotonic() - run_started
    write_json(output/'timing.json', timing)
    write_json(output/'status.json',{'state':'concluido','created_utc':start,
        'completed_utc':datetime.now(timezone.utc).isoformat(),'version':VERSION})
    (output/'CONCLUIDO.txt').write_text(WARNING+'\nExperimento finalizado.\n',encoding='utf8')
    if progress:progress(len(folds)+1,len(folds)+1,'Concluído e salvo')
    return {'output_dir':str(output),'metrics':total,'folds':fold_rows,
            'fold_summary':fold_summary(fold_rows, classes=classes),'final_model':meta,'timing':timing}
