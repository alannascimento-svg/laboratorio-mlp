"""Resumo descritivo de métricas entre folds, separado das métricas OOF."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from statistics import stdev


FOLD_METRICS = ("accuracy", "precision_M", "sensitivity_M", "f1_M")
MACRO_METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


def fold_summary(rows: Sequence[Mapping[str, object]], classes: Sequence[str] | None = None) -> dict:
    """Calcula média simples e STD amostral dos valores disponíveis de cada fold.

    Cada fold tem o mesmo peso, independentemente de seu tamanho de teste.
    Campos ausentes ou None não contam em ``n``; valores inválidos são erros.
    Uma única observação permite média, mas não desvio-padrão amostral.
    """
    macro = (list(classes) != ['M', 'B']) if classes is not None else any(
        any(key in row for key in MACRO_METRICS[1:]) for row in rows)
    metric_keys = MACRO_METRICS if macro else FOLD_METRICS
    summary = {
        "n_folds": len(rows),
        "ddof": 1,
        "aggregation": (
            "Média simples entre folds, sem ponderação pelo tamanho do teste; "
            "STD amostral (ddof=1). Cada n informa os folds com valor disponível "
            "para a métrica. Métricas OOF globais são calculadas separadamente."
        ),
        "positive_class": None if macro else "M",
        "average": "macro" if macro else "positive_class",
        "metric_keys": list(metric_keys),
        "classes": list(classes) if classes is not None else (None if macro else ['M', 'B']),
        "metrics": {},
    }
    if macro:
        summary['aggregation'] += ' Precisão, recall e F1 usam média macro: cada classe tem o mesmo peso dentro de cada fold.'
    for metric in metric_keys:
        values = []
        for index, row in enumerate(rows, start=1):
            value = row.get(metric)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"Linha de fold {index}, {metric}: esperado número finito entre 0 e 1."
                )
            values.append(float(value))
        n = len(values)
        summary["metrics"][metric] = {
            "mean": math.fsum(values) / n if n else None,
            "std": stdev(values) if n > 1 else None,
            "n": n,
        }
    return summary
