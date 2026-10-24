"""Z-score e PCA opcional, ajustados somente nas amostras de treino recebidas."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def _matrix(values, name='Dados'):
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} devem ser uma matriz numérica.') from exc
    if values.ndim != 2 or not len(values) or not 1 <= values.shape[1] <= 256 or not np.isfinite(values).all():
        raise ValueError(f'{name} devem ser uma matriz não vazia, finita, com 1 a 256 colunas.')
    return values


def _target(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError('A variância preservada pelo PCA deve estar acima de 0% e até 100%.')
    return float(value)


def validate_pca(pca, source_dim, network_dim):
    """Valida a transformação salva sem reajustá-la e sem modificar seu conteúdo."""
    if pca is None:
        if source_dim != network_dim:
            raise ValueError('Dimensões originais e da rede incompatíveis sem PCA.')
        return None
    if not isinstance(pca, dict) or pca.get('kind') != 'pca_v1':
        raise ValueError('Formato PCA não reconhecido.')
    try:
        for key in ('n_components', 'n_original_features', 'fit_samples'):
            if isinstance(pca[key], (bool, np.bool_)) or not isinstance(pca[key], (int, np.integer)):
                raise ValueError('Dimensões PCA devem ser inteiras.')
        k, d, n = pca['n_components'], pca['n_original_features'], pca['fit_samples']
        if d != source_dim or k != network_dim or not 1 <= k <= d <= 256 or n < 2 or k > n:
            raise ValueError('Dimensões PCA incompatíveis com o modelo.')
        components, center, ratios = (np.asarray(pca[key], dtype=float) for key in
                                      ('components', 'center', 'explained_variance_ratio'))
        if components.shape != (k, d) or center.shape != (d,) or ratios.shape != (k,):
            raise ValueError('Matrizes PCA têm formas incompatíveis.')
        if any(not np.isfinite(value).all() for value in (components, center, ratios)):
            raise ValueError('Parâmetros PCA devem ser finitos.')
        if not np.allclose(components @ components.T, np.eye(k), rtol=1e-8, atol=1e-8):
            raise ValueError('Componentes PCA devem ser ortonormais.')
        target = _target(pca['variance_target'])
        retained = pca['retained_variance']
        if isinstance(retained, (bool, np.bool_)) or not isinstance(retained, (int, float, np.number)) or not np.isfinite(retained):
            raise ValueError('Variância PCA preservada inválida.')
        if (np.any(ratios < 0) or np.any(ratios > 1) or np.any(np.diff(ratios) > 1e-12)
                or not 0 < retained <= 1 + 1e-12 or abs(float(ratios.sum()) - retained) > 1e-10
                or retained < target - 1e-12 or (k > 1 and float(ratios[:-1].sum()) >= target)):
            raise ValueError('Variâncias PCA incompatíveis com o corte informado.')
        for key, descriptions in (('source_features', False), ('source_feature_descriptions', True)):
            if key in pca:
                values = pca[key]
                if (not isinstance(values, list) or len(values) != d
                        or any(not isinstance(value, str) or (not descriptions and not value) for value in values)
                        or (not descriptions and len(set(values)) != d)):
                    raise ValueError('Nomes ou descrições originais incompatíveis com o PCA.')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Parâmetros PCA incompletos ou inválidos.') from exc
    return pca


class FittedPreprocessor:
    def __init__(self, scaler, pca=None):
        self.scaler, self.pca = scaler, pca

    @property
    def mean_(self):
        return self.scaler.mean_

    @property
    def scale_(self):
        return self.scaler.scale_

    def transform(self, X):
        # Mantém exatamente StandardScaler.transform quando o PCA está desligado.
        _matrix(X, 'Entradas originais')
        Z = self.scaler.transform(X)
        if self.pca is None:
            return Z
        result = (Z - np.asarray(self.pca['center'])) @ np.asarray(self.pca['components']).T
        if not np.isfinite(result).all():
            raise ValueError('A projeção PCA produziu valores não finitos.')
        return result


def fit_preprocessor(X, enabled=False, variance_target=.95):
    if not isinstance(enabled, bool):
        raise ValueError('A opção PCA deve ser booleana.')
    _matrix(X)
    target = _target(variance_target)
    scaler = StandardScaler().fit(X)
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError('Não foi possível obter um normalizador finito.')
    if not enabled:
        return FittedPreprocessor(scaler)
    Z = np.asarray(scaler.transform(X), dtype=float)
    if len(Z) < 2:
        raise ValueError('PCA requer pelo menos duas amostras no treino da fase.')
    center = Z.mean(axis=0)
    centered = Z - center
    if not np.isfinite(centered).all():
        raise ValueError('PCA recebeu valores padronizados não finitos.')
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    if not len(singular) or singular[0] <= 0:
        raise ValueError('PCA não pode preservar variância: o treino da fase tem variância total zero.')
    # A divisão antes do quadrado evita overflow sem mudar a proporção.
    energy = (singular / singular[0]) ** 2
    ratios = energy / energy.sum()
    cumulative = np.cumsum(ratios)
    cumulative[-1] = 1.0  # Fecha a soma normalizada também para corte de 100%.
    k = min(len(ratios), int(np.searchsorted(cumulative, target, side='left')) + 1)
    components = components[:k].copy()
    # O sinal dos vetores singulares é arbitrário: fixa a maior carga positiva.
    anchors = np.argmax(np.abs(components), axis=1)
    signs = np.where(components[np.arange(k), anchors] < 0, -1.0, 1.0)
    components *= signs[:, None]
    pca = {'kind':'pca_v1', 'components':components.tolist(), 'center':center.tolist(),
           'explained_variance_ratio':ratios[:k].tolist(),
           'retained_variance':float(min(1.0, ratios[:k].sum())), 'variance_target':target,
           'n_components':k, 'n_original_features':Z.shape[1], 'fit_samples':len(Z)}
    validate_pca(pca, Z.shape[1], k)
    return FittedPreprocessor(scaler, pca)


def source_features(net):
    pca = getattr(net, 'pca', None)
    if pca is None:
        return list(net.features)
    if not isinstance(pca, dict) or 'source_features' not in pca:
        raise ValueError('Nomes das medidas originais ausentes no PCA.')
    validate_pca(pca, pca.get('n_original_features'), net.n_inputs)
    return list(pca['source_features'])


def source_descriptions(net):
    pca = getattr(net, 'pca', None)
    if pca is None:
        return list(getattr(net, 'feature_descriptions', net.features))
    return list(pca.get('source_feature_descriptions', source_features(net)))


def transform_inputs(net, raw_X, mean, scale):
    X = _matrix(raw_X, 'Entradas originais')
    mean, scale = np.asarray(mean, dtype=float), np.asarray(scale, dtype=float)
    if mean.shape != (X.shape[1],) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Normalizador original incompatível com as entradas.')
    pca = getattr(net, 'pca', None)
    validate_pca(pca, X.shape[1], net.n_inputs)
    result = (X - mean) / scale
    if pca is not None:
        result = (result - np.asarray(pca['center'])) @ np.asarray(pca['components']).T
    if not np.isfinite(result).all():
        raise ValueError('Pré-processamento produziu valores não finitos.')
    return result
