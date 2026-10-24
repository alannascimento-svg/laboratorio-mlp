"""Memória didática conferida contra a MLP, sem treinar ou alterar parâmetros.

Um snapshot permite refazer o cálculo neste estado. Apenas um registro completo
de lote e otimizador permite conferir a atualização que realmente foi gravada.
"""
from __future__ import annotations

import math
import numpy as np

from rede import MLP, one_hot
from preprocessamento import source_features, source_descriptions, transform_inputs, validate_pca


LEGEND = [
    ['D ou E', 'Quantidade de características usadas como entrada; ID e alvo ficam fora.'],
    ['S; H', 'S é a quantidade de classes de saída; H = teto((D + S)/2) neurônios ocultos.'],
    ['i; j; k; n', 'Índices de entrada, neurônio oculto, saída e amostra, respectivamente. A tela conta a partir de 1.'],
    ['xᵢ; mᵢ; sᵢ', 'Valor original, média de treino e escala de treino da característica i. Escala de uma característica constante é 1.'],
    ['Zᵢ', 'Entrada padronizada: (xᵢ − mᵢ)/sᵢ. Não recalculamos m ou s na amostra inspecionada.'],
    ['W1ᵢⱼ; b1ⱼ', 'Peso da entrada i para o oculto j e viés (bias) desse oculto.'],
    ['z1ⱼ; aⱼ', 'Soma antes da ativação e valor depois da ReLU do oculto j.'],
    ['W2ⱼₖ; b2ₖ', 'Peso do oculto j para a saída k e viés dessa saída.'],
    ['uₖ; pₖ', 'Logit (soma antes da Softmax) e escore Softmax da saída k.'],
    ['Yₖ; y', 'Alvo one-hot e índice interno da classe: Yₖ vale 1 na classe correta e 0 nas demais.'],
    ['Σ; exp; ln', 'Σ soma as parcelas; exp(r)=eʳ é a exponencial; ln é o logaritmo natural, inverso de exp.'],
    ['I[k=r]', 'Indicador que vale 1 quando os índices k e r são iguais e 0 quando são diferentes.'],
    ['ℓ; J; λ', 'ℓ é a entropia cruzada de um exemplo; J é a média da perda do lote + penalidade L2; λ é o coeficiente L2.'],
    ['∂; d; δ; g', '∂ é derivada parcial, variando um parâmetro e mantendo os demais fixos; d é derivada de uma variável; δ é a derivada em uma soma; g é o gradiente do parâmetro.'],
    ['N; t', 'N é a quantidade de exemplos do lote; t é o passo global do otimizador na fase, não a época. No cálculo isolado N=1.'],
    ['θ; η; μ; v', 'θ representa peso ou viés; η é taxa de aprendizado; μ é momentum do SGD; v é sua velocidade acumulada.'],
    ['g atual; g anterior; c; τ', 'Gradientes completos de treino em épocas consecutivas; c é o cosseno do ângulo entre eles; τ é a tolerância de mudança da perda do controlador de momentum. Vetores são achatados na ordem W1 (linha por linha), b1, W2 (linha por linha), b2.'],
    ['Normal(0, σ); √; semente', 'Distribuição normal com média zero e desvio-padrão σ; √ é raiz quadrada; a semente inicializa o gerador pseudoaleatório. Não é resultado do cálculo da perda.'],
    ['m Adam; v Adam; β₁; β₂; ε', 'Momentos de primeira e segunda ordem do Adam, seus coeficientes 0,9 e 0,999 e estabilizador ε=10⁻⁸. v Adam não é a velocidade do SGD.'],
    ['Tangente; integral', 'Uma derivada é a inclinação da reta tangente local. ReLU tem uma quina em zero: a derivada clássica não existe ali, e o código escolhe 0. Este algoritmo não usa integrais.'],
]


def _number(value):
    return format(float(value), '.12g')


def _classes(net):
    classes = list(getattr(net, 'classes', ['M', 'B']))
    if len(classes) != net.n_outputs or len(set(classes)) != len(classes):
        raise ValueError('Classes incompatíveis com as saídas do modelo.')
    if any(not isinstance(label, str) or not label for label in classes):
        raise ValueError('Rótulos de saída inválidos no modelo.')
    return classes


def _codes(classes):
    return [''.join('1' if k == r else '0' for k in range(len(classes)))
            for r in range(len(classes))]


def _legend(classes):
    legend = [row.copy() for row in LEGEND]
    legend[1][1] = f'S = {len(classes)} saídas, na ordem [{", ".join(classes)}]; H = teto((D + S)/2) neurônios ocultos.'
    legend[9][1] += ' Ordem das classes: [' + ', '.join(classes) + '].'
    if classes == ['M', 'B']:
        legend[9][1] += ' y=0 → M=[1,0] (10); y=1 → B=[0,1] (01). M é positiva nas métricas.'
    return legend


def _table(title, columns, rows):
    return {'title': title, 'columns': [{'key': k, 'label': label} for k, label in columns], 'rows': rows}


def _step(title, explanation, formulas, substitutions=None, tables=None, note=None):
    return {'title': title, 'explanation': explanation, 'formulas': formulas,
            'substitutions': substitutions or [], 'tables': tables or [], 'note': note}


def _safe(value):
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Cálculo não finito; revise os dados e os parâmetros.')
    return value


def _inputs(net, mean, scale, X, y, neuron, sample, l2):
    X, mean, scale = (np.asarray(v, dtype=float) for v in (X, mean, scale))
    labels = np.asarray(y)
    original_count = len(source_features(net))
    if X.ndim != 2 or not len(X) or X.shape[1] != original_count:
        raise ValueError('Amostras incompatíveis com as entradas do modelo.')
    validate_pca(getattr(net, 'pca', None), original_count, net.n_inputs)
    if mean.shape != (original_count,) or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError('Normalização incompatível: a escala deve ser positiva.')
    classes = _classes(net)
    if labels.shape != (len(X),) or not np.all(np.isin(labels, np.arange(net.n_outputs))):
        raise ValueError(f'Classes devem ser índices de 0 a {net.n_outputs - 1}, uma por amostra, na ordem {classes}.')
    if any(not np.isfinite(v).all() for v in (X, mean, scale)):
        raise ValueError('Entradas e normalizador precisam ser finitos.')
    for index, size, name in ((neuron, net.n_hidden, 'Neurônio'), (sample, len(X), 'Amostra')):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < size:
            raise ValueError(f'{name} fora do intervalo disponível.')
    if isinstance(l2, bool) or not isinstance(l2, (int, float, np.number)) or not np.isfinite(l2) or l2 < 0:
        raise ValueError('L2 deve ser finito e não negativo.')
    for key, shape in [('W1', (net.n_inputs, net.n_hidden)), ('b1', (net.n_hidden,)),
                       ('W2', (net.n_hidden, net.n_outputs)), ('b2', (net.n_outputs,))]:
        if np.shape(net.params[key]) != shape or not np.isfinite(net.params[key]).all():
            raise ValueError(f'Parâmetro inválido: {key}.')
    return X, mean, scale, labels.astype(int)


def _selected(gradients, j):
    return {'W1': gradients['W1'][:, j], 'b1': gradients['b1'][j],
            'W2': gradients['W2'][j, :], 'b2': gradients['b2']}


def _network_notation(value):
    """Nos passos da MLP, a entrada passa a ser PC; Z continua a medida padronizada."""
    if isinstance(value, str):
        return value.replace('Zₙᵢ', 'PCₙᵢ').replace('Zᵢ', 'PCᵢ').replace('Z × W', 'PC × W').replace('é Z e', 'é PC e').replace('₁ᴰ', '₁ᴷ').replace('√(2/D)', '√(2/K)')
    if isinstance(value, list):
        return [_network_notation(item) for item in value]
    if isinstance(value, dict):
        return {key:_network_notation(item) for key,item in value.items()}
    return value


def _pca_steps(steps, legend, net, raw, mean, scale, projected):
    pca = net.pca
    names, descriptions = source_features(net), source_descriptions(net)
    Z = (raw - mean) / scale
    center, components = np.asarray(pca['center']), np.asarray(pca['components'])
    ratios = np.asarray(pca['explained_variance_ratio'])
    d, k = len(raw), net.n_inputs
    raw_rows = [{'input':f'x{i+1}', 'feature':names[i], 'description':descriptions[i],
                 'raw':raw[i], 'mean':mean[i], 'scale':scale[i], 'standardized':Z[i]} for i in range(d)]
    steps[0]['formulas'] = ['D = medidas originais; K = componentes retidas; H = teto((K + S)/2)',
                            'Zᵢ = (xᵢ − mᵢ) / sᵢ']
    steps[0]['substitutions'] = [f'{d} medidas originais → {k} componentes; H = teto(({k} + {net.n_outputs})/2) = {net.n_hidden}.']
    steps[0]['tables'][0]['rows'] = raw_rows
    steps[1:] = _network_notation(steps[1:])
    steps[1]['explanation'] = 'Multiplicamos cada componente PCA pelo peso da ligação até o oculto e somamos o viés uma única vez.'
    rows = [{'component':f'PC{q+1}', 'feature':names[i], 'description':descriptions[i],
             'standardized':Z[i], 'center':center[i], 'centered':Z[i]-center[i],
             'loading':components[q,i], 'product':(Z[i]-center[i])*components[q,i],
             'derivative_raw':components[q,i]/scale[i]} for q in range(k) for i in range(d)]
    cumulative = np.cumsum(ratios)
    section = _step('1b. Aplicar o PCA ajustado no treino desta fase',
        'PCA combina as medidas padronizadas em componentes ordenadas por variância. Médias, escalas, centro e coeficientes foram ajustados somente no treino desta fase. A projeção desta amostra não reajusta nada. Os coeficientes PCA ficam fixos: são diferentes dos pesos W1/W2 aprendidos pela rede.',
        ['A = Z_treino − c; C = AᵀA / (N_treino − 1)',
         'A = U diag(σ) Qᵀ; variânciaⱼ = σⱼ² / (N_treino − 1)',
         'V = primeiras K linhas de Qᵀ, com convenção de sinal fixa',
         'rⱼ = variânciaⱼ / Σ variâncias; acumuladaₖ = Σⱼ₌₁ᵏ rⱼ',
         'K = menor k cuja acumuladaₖ ≥ corte; H = teto((K + S)/2)',
         'PCⱼ = Σᵢ₌₁ᴰ ((Zᵢ − cᵢ) × Vⱼᵢ)',
         '∂PCⱼ/∂Zᵢ = Vⱼᵢ; ∂PCⱼ/∂xᵢ = Vⱼᵢ / sᵢ'],
        [f'Ajuste salvo com {pca["fit_samples"]} amostras; corte = {_number(pca["variance_target"]*100)}%; variância preservada = {_number(pca["retained_variance"]*100)}%; K = {k}.']
        + [f'PC{q+1} = ' + ' + '.join(f'({_number((Z[i]-center[i])*components[q,i])})' for i in range(d))
           + f' = {_number(projected[q])}' for q in range(k)],
        [_table('Variância de cada componente retida e corte acumulado',
            [('component','Componente'),('ratio','Proporção explicada'),('cumulative','Proporção acumulada')],
            [{'component':f'PC{q+1}', 'ratio':ratios[q], 'cumulative':cumulative[q]} for q in range(k)]),
         _table('Todas as parcelas das componentes desta amostra',
            [('component','Componente'),('feature','Medida original'),('description','Descrição'),
             ('standardized','Zᵢ'),('center','cᵢ do PCA'),('centered','Zᵢ − cᵢ'),('loading','Vⱼᵢ fixo'),
             ('product','(Zᵢ − cᵢ) × Vⱼᵢ'),('derivative_raw','∂PCⱼ/∂xᵢ = Vⱼᵢ/sᵢ')],rows)],
        'SVD completa, sem whitening. O sinal de cada vetor é fixado tornando positiva sua maior carga em valor absoluto. As variâncias e cargas mostradas são as salvas no ajuste, não uma decomposição recalculada desta amostra. As derivadas da projeção são constantes durante a fase; o otimizador atualiza somente W1, b1, W2 e b2. Uma projeção reduzida não recupera exatamente todas as medidas originais.')
    steps.insert(1, section)
    legend[0] = ['D; K; E', 'D é a quantidade de medidas originais; K é a quantidade de componentes retidas e E=K é a entrada efetiva da MLP. ID e alvo não entram no PCA.']
    legend[1][1] = legend[1][1].replace('(D + S)', '(K + S)')
    legend.extend([
        ['PCⱼ; Vⱼᵢ; cᵢ', 'Componente principal j, coeficiente fixo da medida i nessa componente e centro do PCA. V não é um peso W treinado pela MLP.'],
        ['A; C; U; σ; Qᵀ; V', 'A contém os dados padronizados e centralizados de treino; C é sua covariância. U, valores singulares σ e Qᵀ vêm da SVD completa; ᵀ significa transposta. As linhas retidas de Qᵀ, com sinal fixado, formam os coeficientes V da projeção.'],
        ['rⱼ; corte; acumulada', 'rⱼ é a fração da variância explicada pela componente j. O menor conjunto inicial cuja soma atinge o corte é mantido.'],
    ])


def _example_report(net, mean, scale, raw_X, y, neuron, sample, l2, standardized=None):
    raw_X, mean, scale, y = _inputs(net, mean, scale, raw_X, y, neuron, sample, l2)
    j, p, d, h, s = int(neuron), net.params, net.n_inputs, net.n_hidden, net.n_outputs
    classes = _classes(net)
    codes = _codes(classes)
    pca = getattr(net, 'pca', None)
    transformed = transform_inputs(net, raw_X[sample:sample+1], mean, scale)
    if standardized is not None:
        standardized = np.asarray(standardized, dtype=float)
        if standardized.shape != (len(raw_X), d) or not np.isfinite(standardized).all():
            raise ValueError('Entradas efetivas do lote incompatíveis com a rede.')
    Z = transformed if standardized is None else standardized[sample:sample+1]
    if pca is not None and not np.allclose(transformed, Z, rtol=1e-10, atol=1e-12):
        raise ValueError('As componentes PCA gravadas não correspondem às entradas originais do lote.')
    targets = one_hot(y[sample:sample+1], s)
    z1, activation, log_probs, scores = net.forward(Z)
    logits = activation @ p['W2'] + p['b2']
    shifted = logits[0] - max(logits[0])
    exponentials = np.exp(shifted)
    residual = scores[0] - targets[0]
    chain_parts = residual * p['W2'][j]
    relu_derivative = float(z1[0, j] > 0)
    delta1 = float(chain_parts.sum() * relu_derivative)
    data_loss = float(-log_probs[0, y[sample]])
    squared_weights = float(np.sum(p['W1'] ** 2) + np.sum(p['W2'] ** 2))
    penalty = .5 * l2 * squared_weights
    objective, reference = net.loss_gradients(Z, targets, l2)
    selected_gradient = {'W1': Z[0] * delta1 + l2 * p['W1'][:, j], 'b1': delta1,
                         'W2': activation[0, j] * residual + l2 * p['W2'][j], 'b2': residual}
    ref_selected = _selected(reference, j)
    error = max(float(np.max(np.abs(np.asarray(selected_gradient[k]) - ref_selected[k]))) for k in selected_gradient)
    descriptions = getattr(net, 'feature_descriptions', net.features)
    in_rows = [{'input': f'PC{i+1}' if pca else f'Z{i+1}', 'feature': net.features[i], 'description': descriptions[i],
                'raw': None if pca else raw_X[sample, i], 'mean': None if pca else mean[i],
                'scale': None if pca else scale[i], 'standardized': Z[0, i],
                'weight': p['W1'][i, j], 'product': Z[0, i] * p['W1'][i, j]} for i in range(d)]
    products_sum = float(sum(row['product'] for row in in_rows))
    out_rows = [{'hidden': f'H{q+1}' + (' ← selecionado' if q == j else ''), 'preactivation': z1[0, q],
                 'activation': activation[0, q], 'class': classes[k], 'weight': p['W2'][q, k],
                 'product': activation[0, q] * p['W2'][q, k]} for k in range(s) for q in range(h)]
    steps = [
        _step('1. Identificar entradas e padronizar',
              'O normalizador é o que foi salvo com este estado. A característica mantém sua ordem no modelo; a descrição é apenas um rótulo.',
              ['S = quantidade de classes de saída; H = teto((D + S)/2)', 'Zᵢ = (xᵢ − mᵢ) / sᵢ'],
              [f'H = teto(({d} + {s})/2) = {h}; rede {d} → {h} → {s}'],
              [_table('Cada entrada antes e depois da padronização',
                      [('input', 'Entrada'), ('feature', 'Campo'), ('description', 'Descrição'), ('raw', 'x original'),
                       ('mean', 'm de treino'), ('scale', 's de treino'), ('standardized', '(x − m)/s = Z')], in_rows)],
              'Ordem das saídas: [' + ', '.join(classes) + ']. O alvo informado é ' + classes[y[sample]]
              + ' = [' + ', '.join(str(int(v)) for v in targets[0]) + '] = ' + codes[y[sample]] + '.'),
        _step(f'2. Somar as entradas de H{j+1}',
              'Multiplicamos cada entrada padronizada pelo peso da sua ligação e somamos o viés. O viés é somado uma única vez.',
              ['z1ⱼ = Σᵢ₌₁ᴰ (Zᵢ × W1ᵢⱼ) + b1ⱼ'],
              [f'z1 de H{j+1} = {_number(products_sum)} + ({_number(p["b1"][j])}) = {_number(z1[0,j])}'],
              [_table('Todas as parcelas da soma do neurônio selecionado',
                      [('input', 'Entrada'), ('description', 'Descrição'), ('standardized', 'Zᵢ'),
                       ('weight', 'W1ᵢⱼ'), ('product', 'Zᵢ × W1ᵢⱼ')], in_rows)]),
        _step('3. Aplicar ReLU',
              'ReLU deixa passar valores positivos e devolve zero para valores negativos ou zero. A sigmoide ainda não foi aplicada neste modelo.',
              ['aⱼ = ReLU(z1ⱼ) = máximo(0, z1ⱼ)'],
              [f'a de H{j+1} = máximo(0, {_number(z1[0,j])}) = {_number(activation[0,j])}']),
        _step('4. Somar todas as contribuições nas saídas',
              'O neurônio selecionado contribui para todas as classes, mas cada saída recebe a soma de TODOS os ocultos. Uma ligação isolada não determina a classe.',
              ['uₖ = Σⱼ₌₁ᴴ (aⱼ × W2ⱼₖ) + b2ₖ'],
              [f'u_{classes[k]} = {_number(np.sum(activation[0] * p["W2"][:,k]))} + ({_number(p["b2"][k])}) = {_number(logits[0,k])}' for k in range(s)],
              [_table('Ocultos → saídas, com o caminho selecionado marcado',
                      [('hidden', 'Oculto'), ('preactivation', 'z1'), ('activation', 'a = ReLU(z1)'),
                       ('class', 'Saída'), ('weight', 'W2 para esta saída'), ('product', 'a × W2')], out_rows)]),
        _step('5. Aplicar Softmax e decidir por WTA',
              'Subtraímos o maior logit antes da exponencial para evitar estouro numérico. Essa subtração não muda os escores. WTA escolhe o maior escore depois do cálculo; não entra nas derivadas.',
              ['c = máximo(u₁, …, uₛ)', 'pₖ = exp(uₖ − c) / Σᵣ₌₁ˢ exp(uᵣ − c)',
               'log(pₖ) = (uₖ − c) − ln[Σᵣ₌₁ˢ exp(uᵣ − c)]',
               'classe = argmax([p₁, …, pₛ]); empate exato escolhe a primeira classe empatada na ordem salva'],
              [f'c = {_number(max(logits[0]))}; soma das exponenciais = {_number(exponentials.sum())}',
               f'WTA = {classes[int(np.argmax(scores[0]))]} = {codes[int(np.argmax(scores[0]))]}'],
              [_table('Substituição para cada saída',
                      [('class', 'Saída'), ('logit', 'u'), ('shifted', 'u − c'), ('exp', 'exp(u − c)'), ('score', 'p'), ('log_score', 'log(p) estável')],
                      [{'class': classes[k], 'logit': logits[0,k], 'shifted': shifted[k], 'exp': exponentials[k],
                        'score': scores[0,k], 'log_score': log_probs[0,k]} for k in range(s)])]),
        _step('6. Medir o erro e a penalidade L2',
              'A entropia cruzada pune um escore baixo para a classe correta. L2 soma os quadrados de todos os pesos de ambas as camadas; os vieses não são penalizados.',
              ['ℓ = −Σₖ Yₖ ln(pₖ) = −ln(p da classe correta)',
               'J = (1/N) Σₙ ℓₙ + (λ/2) [Σᵢ,ⱼ W1ᵢⱼ² + Σⱼ,ₖ W2ⱼₖ²]'],
              [f'Neste exemplo isolado N = 1; ℓ = −({_number(log_probs[0,y[sample]])}) = {_number(data_loss)}',
               f'L2 = ({_number(l2)}/2) × {_number(squared_weights)} = {_number(penalty)}',
               f'J do exemplo = {_number(data_loss)} + {_number(penalty)} = {_number(objective)}']),
        _step('7. Entender derivada, Softmax e regra da cadeia',
              'A derivada mede como a perda muda se um valor variar muito pouco. Em um gráfico ela é a inclinação da tangente local. A regra da cadeia multiplica as derivadas ao longo de cada caminho e soma caminhos que se encontram.',
              ['f′(q) = limite, quando ε→0, de [f(q+ε) − f(q)]/ε',
               '∂pₖ/∂uᵣ = pₖ × (I[k=r] − pᵣ)',
               '∂ℓ/∂pₖ = −Yₖ/pₖ (expressão simbólica para pₖ > 0)',
               '∂ℓ/∂uᵣ = Σₖ (∂ℓ/∂pₖ)(∂pₖ/∂uᵣ) = pᵣ − Yᵣ',
               'δ2ₙₖ = ∂J/∂uₙₖ = (pₙₖ − Yₙₖ)/N'],
              ['Softmax e entropia cruzada se simplificam para p − Y; usamos a expressão estável mesmo quando um escore arredonda para zero.'],
              [_table(f'Derivadas da Softmax (Jacobiana {s} × {s})',
                      [('row', 'Derivada')] + [(f'output_{r}', 'Em relação a u_' + classes[r]) for r in range(s)],
                      [{'row': '∂p_'+classes[k], **{f'output_{r}': scores[0,k] * (int(k==r)-scores[0,r])
                                                 for r in range(s)}} for k in range(s)]),
               _table('Resíduo e derivada para N=1', [('class','Saída'), ('score','p'), ('target','Y'), ('delta','δ2 = p − Y')],
                      [{'class':classes[k], 'score':scores[0,k], 'target':targets[0,k], 'delta':residual[k]} for k in range(s)])],
              'Não há cálculo de integral neste algoritmo. A tangente explica a derivada; não calculamos um ângulo de tangente.'),
        _step(f'8. Levar o erro de volta a H{j+1}',
              'Primeiro somamos o erro que chega de cada uma das S saídas ponderado pelos pesos de saída. Depois multiplicamos pela derivada da ReLU. A ReLU bloqueada pode zerar a parcela do erro dos dados; L2 ainda pode alterar seus pesos.',
              ['ReLU′(z) = 1 se z > 0; 0 se z < 0; em z = 0 o código adota 0',
               '∂J/∂aⱼ = Σₖ δ2ₖ × W2ⱼₖ', 'δ1ⱼ = ∂J/∂z1ⱼ = (Σₖ δ2ₖ × W2ⱼₖ) × ReLU′(z1ⱼ)'],
              [f'ReLU′({_number(z1[0,j])}) = {_number(relu_derivative)}',
               f'δ1 de H{j+1} = (' + ' + '.join('(' + _number(term) + ')' for term in chain_parts)
               + f') × {_number(relu_derivative)} = {_number(delta1)}'],
              [_table(f'Os {s} caminhos da regra da cadeia', [('class','Saída'), ('delta','δ2'), ('weight','W2ⱼₖ'), ('product','δ2 × W2ⱼₖ')],
                      [{'class':classes[k], 'delta':residual[k], 'weight':p['W2'][j,k], 'product':chain_parts[k]} for k in range(s)])]),
        _step('9. Derivar pesos e vieses do caminho',
              'Para a soma Z × W + b, a derivada em W é Z e em b é 1. Para L2, a derivada de (λ/2)W² é λW. As tabelas mostram o gradiente de UMA amostra neste estado, incluindo a penalidade.',
              ['∂z1ⱼ/∂W1ᵢⱼ = Zᵢ; ∂z1ⱼ/∂b1ⱼ = 1',
               'g_W1ᵢⱼ = Σₙ Zₙᵢ δ1ₙⱼ + λW1ᵢⱼ; g_b1ⱼ = Σₙ δ1ₙⱼ',
               'g_W2ⱼₖ = Σₙ aₙⱼ δ2ₙₖ + λW2ⱼₖ; g_b2ₖ = Σₙ δ2ₙₖ'],
              [f'Neste cálculo N=1. g_b1 de H{j+1} = {_number(delta1)}; os vieses b2 pertencem às saídas compartilhadas.'],
              [_table('Pesos de entrada do neurônio selecionado', [('parameter','Parâmetro'), ('input','Zᵢ'), ('delta','δ1'), ('data','Zᵢ × δ1'), ('l2','λW1'), ('gradient','Gradiente total')],
                      [{'parameter':f'W1[{i+1},{j+1}]', 'input':Z[0,i], 'delta':delta1, 'data':Z[0,i]*delta1,
                        'l2':l2*p['W1'][i,j], 'gradient':selected_gradient['W1'][i]} for i in range(d)]),
               _table('Pesos de saída do neurônio selecionado', [('parameter','Parâmetro'), ('activation','aⱼ'), ('delta','δ2'), ('data','aⱼ × δ2'), ('l2','λW2'), ('gradient','Gradiente total')],
                      [{'parameter':f'W2[{j+1},{k+1}] → {classes[k]}', 'activation':activation[0,j], 'delta':residual[k],
                        'data':activation[0,j]*residual[k], 'l2':l2*p['W2'][j,k], 'gradient':selected_gradient['W2'][k]} for k in range(s)]),
               _table('Vieses: sem penalidade L2', [('parameter','Viés'), ('gradient','Gradiente')],
                      [{'parameter':f'b1[{j+1}]','gradient':delta1}]+[{'parameter':f'b2[{k+1}] → {classes[k]}','gradient':residual[k]} for k in range(s)])],
              'Gradiente negativo é válido. Não se troca seu sinal para torná-lo positivo.'),
    ]
    legend = _legend(classes)
    if pca is not None:
        _pca_steps(steps, legend, net, raw_X[sample], mean, scale, Z[0])
    report = {'mode':'snapshot', 'architecture':net.architecture, 'classes':classes, 'codes':codes,
            'neuron':j, 'sample':int(sample),
            'sample_count':len(raw_X), 'notice':'Cálculo conferido neste estado para esta amostra, não reconstrução do treinamento original.',
            'steps':steps, 'legend':legend, 'preprocessing':pca,
            'forward':{'standardized':(raw_X[sample]-mean)/scale if pca else Z[0],
                       'network_input':Z[0], 'z1':z1[0], 'activation':activation[0], 'logits':logits[0],
                       'scores':scores[0], 'target':targets[0], 'delta2':residual, 'delta1_selected':delta1,
                       'relu_derivative':relu_derivative, 'cross_entropy':data_loss, 'l2_penalty':penalty,
                       'objective':objective, 'predicted':classes[int(np.argmax(scores[0]))]},
            'gradients':selected_gradient,
            'verification':{'ok':bool(error < 1e-10 and abs(objective-data_loss-penalty) < 1e-10),
                            'gradient_max_abs_error':error, 'objective_abs_error':abs(objective-data_loss-penalty)},
            'history_note':'O snapshot não contém os lotes, gradientes e estado do otimizador que produziram estes pesos. Não é possível inventar esses passos. Na implementação atual, W1 começa com sorteios Normal(0, √(2/D)), W2 com Normal(0, √(1/H)) e os vieses começam em zero; o segundo argumento é o desvio-padrão. A semente determina o sorteio. Isso descreve a inicialização do código atual e não reconstrói o histórico de um modelo antigo. Para valores iniciais salvos, escolha o checkpoint inicial. Se houver atualizações gravadas, selecione essa fonte acima.'}
    if pca is not None:
        report['history_note'] = report['history_note'].replace('√(2/D)', '√(2/K)')
    return report


def snapshot_report(net, mean, scale, raw_X, y, neuron: int, sample: int = 0, l2: float = 0.0):
    """Forward e gradientes de um exemplo usando parâmetros e escala reais salvos."""
    return _safe(_example_report(net, mean, scale, raw_X, y, neuron, sample, l2))


def _arrays(payload, key, shapes):
    block = payload.get(key)
    if not isinstance(block, dict) or set(block) != set(shapes):
        raise ValueError(f'Registro incompleto: {key}.')
    arrays = {k:np.asarray(block[k], dtype=float) for k in shapes}
    if any(arrays[k].shape != shapes[k] or not np.isfinite(arrays[k]).all() for k in arrays):
        raise ValueError(f'Formas ou valores inválidos: {key}.')
    return arrays


def _momentum_origin_step(payload, net, mu):
    """Explica e confere a decisão gravada; não usa amostra/teste para mudá-la."""
    origin = payload.get('momentum_origin')
    initialization = f'Na inicialização atual: W1 ~ Normal(0, √(2/{net.n_inputs})); W2 ~ Normal(0, √(1/{net.n_hidden})); b1=b2=0. Os valores dentro de Normal são média e desvio-padrão. A semente organiza sorteios pseudoaleatórios; não há derivação do sorteio a partir da perda.'
    if origin is None:
        return _step('11. Identificar de onde veio o momentum',
            'Este registro preserva o μ e as velocidades aplicados, mas não a decisão anterior que escolheu μ. Sua origem está indisponível; não a reconstruímos por suposição.',
            ['vₜ = μ × vₜ₋₁ − ηgₜ'], [f'μ aplicado no lote = {_number(mu)}.'], note=initialization), None
    if not isinstance(origin, dict):
        raise ValueError('Origem do momentum inválida.')
    seed_text = f'Semente efetiva registrada: {origin["seed"]}. ' if 'seed' in origin else 'Semente não informada neste registro. '
    if origin.get('kind') == 'initial':
        initial = origin.get('momentum_initial')
        if isinstance(initial,bool) or not isinstance(initial,(int,float)) or not np.isfinite(initial) or not 0<=initial<1:
            raise ValueError('Momentum inicial inválido no registro.')
        section = _step('11. Identificar de onde veio o momentum',
            'Na primeira época, μ é o valor inicial configurado. A velocidade começou em zero antes do primeiro lote; em lotes seguintes da mesma época, ela acumula as atualizações anteriores.',
            ['μ da primeira época = μ inicial configurado', 'v antes do primeiro passo = 0',
             'W1 inicial ~ Normal(0, √(2/D)); W2 inicial ~ Normal(0, √(1/H)); b1=b2=0'],
            [f'μ inicial = {_number(initial)}; μ aplicado neste lote = {_number(mu)}.',seed_text+initialization])
        return section, {'ok':bool(abs(initial-mu)<1e-12 and payload.get('epoch')==1),
                         'momentum_abs_error':abs(initial-mu)}
    if origin.get('kind') != 'epoch_decision':
        raise ValueError('Tipo de origem do momentum desconhecido.')
    previous, current = (np.asarray(origin[k],dtype=float) for k in ('previous_gradient','current_gradient'))
    if previous.shape != (net.n_parameters,) or current.shape != previous.shape or not np.isfinite(previous).all() or not np.isfinite(current).all():
        raise ValueError('Gradientes completos da origem do momentum são inválidos.')
    fields = ['previous_loss','objective','alignment','momentum_used','momentum_next','momentum_max',
              'momentum_increase','momentum_decrease','adaptation_tolerance']
    if any(isinstance(origin.get(k),bool) or not isinstance(origin.get(k),(int,float)) or not np.isfinite(origin[k]) for k in fields):
        raise ValueError('Valores da decisão de momentum devem ser números finitos.')
    if not isinstance(origin.get('adaptive_momentum'),bool) or not isinstance(origin.get('reset'),bool):
        raise ValueError('Opções da decisão de momentum são inválidas.')
    if not (0 <= origin['momentum_used'] <= origin['momentum_max'] < 1
            and 0 < origin['momentum_increase'] <= 1 and 0 < origin['momentum_decrease'] < 1
            and origin['adaptation_tolerance'] >= 0):
        raise ValueError('Limites do controlador de momentum são inválidos.')
    dot = float(np.dot(current,previous))
    norm_previous, norm_current = float(np.linalg.norm(previous)),float(np.linalg.norm(current))
    denom = norm_current*norm_previous
    alignment = float(np.clip(dot/denom,-1,1)) if denom>0 else 0.
    change = origin['objective']-origin['previous_loss']
    tolerance, used = origin['adaptation_tolerance'],origin['momentum_used']
    if not origin['adaptive_momentum']:
        expected, action, reset = used,'fixo',False
    elif change>tolerance or alignment<0:
        expected, action, reset = max(0.,used*origin['momentum_decrease']),'reduzir_e_zerar_velocidade',True
    elif change < -tolerance and alignment>0:
        expected, action, reset = min(origin['momentum_max'],used+origin['momentum_increase']),'aumentar_ate_limite',False
    else:
        expected, action, reset = used,'manter',False
    alignment_error = abs(alignment-origin['alignment'])
    mu_error = max(abs(expected-origin['momentum_next']),abs(mu-origin['momentum_next']))
    epoch = origin.get('epoch')
    verified = bool(max(alignment_error,mu_error)<1e-10 and action==origin.get('action') and reset==origin['reset']
                    and isinstance(epoch,int) and payload.get('epoch')==epoch+1)
    rows = [
        {'quantity':'Gradiente completo anterior', 'terms':{'values':previous.tolist(),'label':'W1 (linhas), b1, W2 (linhas), b2'},'sum':None,'result':None},
        {'quantity':'Gradiente completo atual', 'terms':{'values':current.tolist(),'label':'W1 (linhas), b1, W2 (linhas), b2'},'sum':None,'result':None},
        {'quantity':'Produto escalar', 'terms':{'values':(current*previous).tolist(),'label':'g atual[q] × g anterior[q]'},'sum':dot,'result':dot},
        {'quantity':'Norma anterior', 'terms':{'values':(previous**2).tolist(),'label':'g anterior[q]²'},'sum':float(np.sum(previous**2)),'result':norm_previous},
        {'quantity':'Norma atual', 'terms':{'values':(current**2).tolist(),'label':'g atual[q]²'},'sum':float(np.sum(current**2)),'result':norm_current},
    ]
    section = _step('11. Conferir a decisão que escolheu o momentum',
        'A decisão foi tomada ao fim da época anterior usando a perda com L2 e os gradientes COMPLETOS de treino daquela fase, registrados antes e depois da época. Ela vale para todos os lotes da próxima época. Os vetores abaixo vêm desse registro; não são gradientes deste exemplo nem do teste externo.',
        ['g atual · g anterior = Σq (g atual[q] × g anterior[q])',
         '||g|| = √(Σq g[q]²)', 'c = (g atual · g anterior) / (||g atual|| × ||g anterior||); se alguma norma é zero, c = 0',
         'ΔJ = J atual − J anterior',
         'Se momentum fixo: μ próximo = μ usado',
         'Se ΔJ > τ OU c < 0: μ próximo = máximo(0, μ usado × fator de redução); zerar velocidade',
         'Senão, se ΔJ < −τ E c > 0: μ próximo = mínimo(μ máximo, μ usado + incremento)',
         'Nos outros casos: manter μ'],
        [f'Decisão registrada ao fim da época {origin.get("epoch", "não informada")}. '+seed_text,
         f'c = {_number(dot)} / ({_number(norm_current)} × {_number(norm_previous)}) = {_number(alignment)}'+(' (convenção de norma zero).' if denom==0 else '.'),
         f'ΔJ = {_number(origin["objective"])} − {_number(origin["previous_loss"])} = {_number(change)}; τ = {_number(tolerance)}.',
         f'Adaptativo: {"sim" if origin["adaptive_momentum"] else "não"}; μ usado = {_number(used)}; μ máximo = {_number(origin["momentum_max"])}; incremento = {_number(origin["momentum_increase"])}; fator de redução = {_number(origin["momentum_decrease"])}.',
         f'Ação calculada: {action}; ação gravada: {origin.get("action")}; zerar velocidade: {"sim" if origin["reset"] else "não"}.',
         f'μ próximo calculado = {_number(expected)}; gravado = {_number(origin["momentum_next"])}; aplicado neste lote = {_number(mu)}.'],
        [_table('Vetores completos e todas as parcelas (abrir para inspecionar)',
            [('quantity','Quantidade'),('terms','Valores / parcelas'),('sum','Soma das parcelas'),('result','Resultado após fórmula')],rows)],
        'c é o cosseno do ângulo entre vetores; não usamos a tangente trigonométrica. O sinal de um único gradiente não comanda o momentum. A regra é uma heurística limitada, sem garantia de convergência. '+initialization)
    return section, {'ok':verified,'alignment_abs_error':alignment_error,'momentum_abs_error':mu_error,
                     'expected_action':action,'expected_reset':reset,'expected_momentum':expected}


def update_report(payload, neuron: int, sample: int = 0):
    """Confere o lote/estado REAL registrado; não modifica ou reexecuta treino."""
    X = np.asarray(payload['X'], dtype=float)
    if X.ndim != 2 or not len(X) or not 1 <= X.shape[1] <= 256:
        raise ValueError('Lote registrado inválido.')
    net = MLP(0, X.shape[1], payload.get('features'), payload.get('feature_descriptions'),
              classes=payload.get('classes'))
    net.pca = payload.get('pca')
    classes = _classes(net)
    shapes = {k:v.shape for k,v in net.params.items()}
    before = _arrays(payload, 'before', shapes)
    after = _arrays(payload, 'after', shapes)
    saved_gradients = _arrays(payload, 'gradients', shapes)
    net.params = {k:v.copy() for k,v in before.items()}
    mean, scale = np.asarray(payload['mean'], dtype=float), np.asarray(payload['scale'], dtype=float)
    if net.pca is not None:
        if 'raw_X' not in payload:
            raise ValueError('Memória PCA precisa das entradas originais reais do lote (raw_X).')
        raw = np.asarray(payload['raw_X'], dtype=float)
    else:
        raw = X * scale + mean
    l2 = payload.get('l2', 0.0)
    _, mean, scale, y = _inputs(net, mean, scale, raw, payload['y'], neuron, sample, l2)
    if not np.isfinite(X).all():
        raise ValueError('Lote registrado deve ser finito.')
    if net.pca is not None:
        projected = transform_inputs(net, raw, mean, scale)
        if projected.shape != X.shape or not np.allclose(projected, X, rtol=1e-10, atol=1e-12):
            raise ValueError('As componentes PCA gravadas não correspondem às entradas originais do lote.')
    report = _example_report(net, mean, scale, raw, y, neuron, sample, l2, standardized=X)
    report['mode'] = 'update'
    report['notice'] = 'Atualização real registrada: cálculos da amostra no estado ANTES do passo e atualização pela média de TODO o lote.'
    report['history_note'] = 'Este registro permite conferir este passo do treinamento. As tabelas do exemplo não substituem o gradiente médio do lote usado pelo otimizador.'
    report['steps'][0]['note'] += (' Neste registro, x original vem de raw_X real gravado. A projeção é conferida, e o gradiente usa diretamente as componentes PC gravadas; não usamos uma reconstrução inversa do PCA.'
        if net.pca is not None else ' Neste registro, x original é reconstituído como Z × s + m; o cálculo usa diretamente o Z gravado, sem nova padronização.')
    targets = one_hot(y, net.n_outputs)
    objective, gradients = net.loss_gradients(X, targets, l2)
    z1, a, logs, scores = net.forward(X)
    n, j = len(X), int(neuron)
    delta2 = (scores-targets)/n
    delta1 = (delta2 @ before['W2'].T) * (z1>0)
    indices = [('W1',(i,j)) for i in range(net.n_inputs)] + [('b1',(j,))] + [('W2',(j,k)) for k in range(net.n_outputs)] + [('b2',(k,)) for k in range(net.n_outputs)]
    batch_rows, update_rows = [], []
    for key, ix in indices:
        if key == 'W1': terms = X[:,ix[0]] * delta1[:,j]
        elif key == 'b1': terms = delta1[:,j]
        elif key == 'W2': terms = a[:,j] * delta2[:,ix[1]]
        else: terms = delta2[:,ix[0]]
        label = key + '[' + ','.join(str(i+1) for i in ix) + ']'
        if key in ('W2', 'b2'):
            label += ' → ' + classes[ix[-1]]
        batch_rows.append({'parameter':label, 'terms':{'values':terms.tolist(),'label':'Contribuições das amostras (já divididas por N)'},
                           'data':float(terms.sum()), 'l2':l2*before[key][ix] if key.startswith('W') else 0.,
                           'gradient':gradients[key][ix], 'saved_gradient':saved_gradients[key][ix]})
        update_rows.append({'parameter':label, 'key':key, 'index':ix, 'before':before[key][ix],
                            'gradient':saved_gradients[key][ix], 'after':after[key][ix]})
    report['steps'].append(_step('10. Reunir o lote que realmente atualizou os pesos',
        'Cada exemplo do lote participou. O fator 1/N já está incluído em δ2 e δ1 abaixo; somar essas parcelas dá a média, sem dividir de novo. L2 entra uma única vez por atualização.',
        ['J_lote = (1/N) Σₙ ℓₙ + (λ/2)(Σ W1² + Σ W2²)',
         'δ2ₙₖ = (pₙₖ − Yₙₖ)/N', 'δ1ₙⱼ = (Σₖ δ2ₙₖ W2ⱼₖ) × ReLU′(z1ₙⱼ)',
         'g_W1ᵢⱼ = Σₙ (Zₙᵢ δ1ₙⱼ) + λW1ᵢⱼ', 'g_W2ⱼₖ = Σₙ (aₙⱼ δ2ₙₖ) + λW2ⱼₖ',
         'g_b1ⱼ = Σₙ δ1ₙⱼ; g_b2ₖ = Σₙ δ2ₙₖ'],
        [f'N = {n}; J_lote antes do passo = {_number(objective)}. Amostra visível no passo a passo: {sample+1}.'],
        [_table('Contribuição de cada exemplo no oculto selecionado',
            [('sample','Amostra no lote'), ('class','Alvo'), ('z1','z1ⱼ'), ('activation','aⱼ'), ('relu','ReLU′'),
             *[(f'delta_{k}', f'δ2_{classes[k]} (inclui 1/N)') for k in range(net.n_outputs)],
             ('delta1','δ1ⱼ (inclui 1/N)')],
            [{'sample':r+1,'class':classes[y[r]],'z1':z1[r,j],'activation':a[r,j],'relu':int(z1[r,j]>0),
              **{f'delta_{k}':delta2[r,k] for k in range(net.n_outputs)}, 'delta1':delta1[r,j]} for r in range(n)]),
         _table('Todas as parcelas para cada parâmetro do caminho',
            [('parameter','Parâmetro'), ('terms','Abrir parcelas por amostra'), ('data','Soma = gradiente dos dados'),
             ('l2','Parcela L2'), ('gradient','Total recalculado'), ('saved_gradient','Total gravado')], batch_rows)]))
    step = payload.get('step')
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError('Passo global inválido no registro.')
    eta = payload.get('learning_rate')
    if isinstance(eta, bool) or not isinstance(eta, (int, float)) or not np.isfinite(eta) or eta <= 0:
        raise ValueError('Taxa de aprendizado inválida no registro.')
    optimizer = payload.get('optimizer')
    state_errors = []
    origin_check = None
    if optimizer == 'sgd_momentum':
        mu = payload.get('momentum')
        if isinstance(mu, bool) or not isinstance(mu, (int,float)) or not np.isfinite(mu) or not 0 <= mu < 1:
            raise ValueError('Momentum inválido no registro.')
        origin_step, origin_check = _momentum_origin_step(payload,net,mu)
        report['steps'].append(_network_notation(origin_step) if net.pca is not None else origin_step)
        vb, va = _arrays(payload,'velocity_before',shapes), _arrays(payload,'velocity_after',shapes)
        origin = payload.get('momentum_origin') or {}
        if origin_check is not None and (payload['step']==1 or (origin.get('reset') and payload.get('batch')==1)):
            origin_check['ok'] = bool(origin_check['ok'] and all(np.count_nonzero(vb[k])==0 for k in shapes))
        expected_v = {k:mu*vb[k]-eta*saved_gradients[k] for k in shapes}
        expected_after = {k:before[k]+expected_v[k] for k in shapes}
        state_errors = [np.max(np.abs(expected_v[k]-va[k])) for k in shapes]
        for row in update_rows:
            k, ix = row['key'], row['index']
            row.update({'velocity_before':vb[k][ix], 'momentum_part':mu*vb[k][ix], 'gradient_part':-eta*saved_gradients[k][ix],
                        'velocity_after':va[k][ix], 'expected_after':expected_after[k][ix]})
        explanation = 'O momentum combina a velocidade anterior real com o novo gradiente do lote. A tabela usa o μ efetivamente aplicado neste lote. Um gradiente negativo contribui com −ηg positivo. Nenhuma velocidade foi presumida.'
        formulas = ['vₜ = μ × vₜ₋₁ − η × gₜ', 'θₜ = θₜ₋₁ + vₜ']
        substitutions = [f't = {step}; η = {_number(eta)}; μ = {_number(mu)}.']
        cols = [('parameter','Parâmetro'),('before','θ antes'),('velocity_before','v anterior'),('gradient','g gravado'),
                ('momentum_part','μ × v anterior'),('gradient_part','−η × g'),('velocity_after','v novo gravado'),
                ('expected_after','θ depois calculado'),('after','θ depois gravado')]
        note = 'O controlador de momentum atua ao fim de cada época com perda e gradientes completos de treino. Esta tela confere o μ e a velocidade efetivamente registrados para o lote; não tenta decidir outro μ usando esta amostra.'
    elif optimizer == 'adam':
        mb, vb = _arrays(payload,'adam_m_before',shapes), _arrays(payload,'adam_v_before',shapes)
        ma, va = _arrays(payload,'adam_m_after',shapes), _arrays(payload,'adam_v_after',shapes)
        if any((v < 0).any() for v in list(vb.values())+list(va.values())):
            raise ValueError('Segundo momento Adam não pode ser negativo.')
        em = {k:.9*mb[k]+.1*saved_gradients[k] for k in shapes}
        ev = {k:.999*vb[k]+.001*saved_gradients[k]**2 for k in shapes}
        mhat = {k:em[k]/(1-.9**step) for k in shapes}
        vhat = {k:ev[k]/(1-.999**step) for k in shapes}
        expected_after = {k:before[k]-eta*mhat[k]/(np.sqrt(vhat[k])+1e-8) for k in shapes}
        state_errors = [np.max(np.abs(em[k]-ma[k])) for k in shapes]+[np.max(np.abs(ev[k]-va[k])) for k in shapes]
        for row in update_rows:
            k, ix = row['key'], row['index']
            row.update({'m_before':mb[k][ix], 'v_before':vb[k][ix], 'm_after':ma[k][ix], 'v_after':va[k][ix],
                        'mhat':mhat[k][ix], 'vhat':vhat[k][ix], 'change':-eta*mhat[k][ix]/(np.sqrt(vhat[k][ix])+1e-8),
                        'expected_after':expected_after[k][ix]})
        explanation = 'Adam usa médias móveis do gradiente e do seu quadrado. A correção divide por 1−βᵗ porque os momentos começaram em zero. O passo t é global nesta fase, e não reinicia em cada época. μ do SGD não é usado.'
        formulas = ['mₜ = β₁mₜ₋₁ + (1−β₁)gₜ', 'vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²',
                    'm̂ₜ = mₜ / (1−β₁ᵗ); v̂ₜ = vₜ / (1−β₂ᵗ)', 'θₜ = θₜ₋₁ − η × m̂ₜ / (√v̂ₜ + ε)']
        substitutions = [f't = {step}; η = {_number(eta)}; β₁ = 0,9; β₂ = 0,999; ε = 10⁻⁸.',
                         f'1−β₁ᵗ = {_number(1-.9**step)}; 1−β₂ᵗ = {_number(1-.999**step)}.']
        cols = [('parameter','Parâmetro'),('before','θ antes'),('gradient','g gravado'),('m_before','m anterior'),
                ('v_before','v anterior'),('m_after','m novo gravado'),('v_after','v novo gravado'),('mhat','m̂ calculado'),
                ('vhat','v̂ calculado'),('change','Δθ calculado'),('expected_after','θ depois calculado'),('after','θ depois gravado')]
        note = 'v do Adam é segundo momento, não a velocidade do SGD. β₁ = 0,9 não é o μ adaptativo do outro otimizador.'
    else:
        raise ValueError('Otimizador desconhecido no registro.')
    gradient_error = max(float(np.max(np.abs(gradients[k]-saved_gradients[k]))) for k in shapes)
    update_error = max(float(np.max(np.abs(expected_after[k]-after[k]))) for k in shapes)
    state_error = float(max(state_errors))
    report['steps'].append(_step(('12.' if optimizer=='sgd_momentum' else '11.')+' Conferir a atualização real: '+('SGD com momentum' if optimizer=='sgd_momentum' else 'Adam'),
                               explanation, formulas, substitutions, [_table('Valores antes, cálculo e valores depois',cols,update_rows)],note))
    report['batch'] = {'size':n, 'objective':objective, 'gradients':_selected(gradients,j), 'gradient_rows':batch_rows,
                       'delta2':delta2, 'delta1_selected':delta1[:,j]}
    report['update'] = {'step':int(step),'epoch':payload.get('epoch'),'batch':payload.get('batch'),
                        'optimizer':optimizer,'learning_rate':eta,'rows':update_rows}
    report['verification'] = {'ok':bool(report['verification']['ok'] and max(gradient_error,update_error,state_error)<1e-10 and (origin_check is None or origin_check['ok'])),
                               'gradient_max_abs_error':gradient_error, 'update_max_abs_error':update_error,
                               'optimizer_state_max_abs_error':state_error,
                               'example_gradient_max_abs_error':report['verification']['gradient_max_abs_error']}
    report['momentum_origin_verification'] = origin_check
    if origin_check is not None:
        report['verification']['momentum_origin_abs_error'] = max(origin_check.get('alignment_abs_error',0.),origin_check['momentum_abs_error'])
    if net.pca is not None:
        # As seções adicionais de lote também recebem a notação PC.
        report['steps'][10:] = _network_notation(report['steps'][10:])
        report['verification']['projection_max_abs_error'] = float(np.max(np.abs(projected-X)))
    return _safe(report)
