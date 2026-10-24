"""Treinamento pela linha de comando, sem interface gráfica."""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from rede import Config, WARNING, run_experiment


def main() -> int:
    p = argparse.ArgumentParser(description=WARNING)
    p.add_argument('--dados', type=Path, default=Path(__file__).parent / 'dados' / 'cancear.txt')
    p.add_argument('--saida', type=Path, default=None)
    p.add_argument('--esquema-entradas', type=Path, help='JSON exportado com seleção e descrições das colunas.')
    p.add_argument('--pca', action='store_true', help='Reduzir entradas por PCA ajustado somente no treino de cada fase.')
    p.add_argument('--pca-variancia', type=float, default=.95, help='Fração de variância preservada (0,1], padrão 0.95; usada com --pca.')
    p.add_argument('--sem-memoria-calculo', action='store_true', help='Não registrar cada atualização; não altera o aprendizado.')
    p.add_argument('--modo', choices=['variavel', 'kfold5', 'kfold10'], default='variavel')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epocas', type=int, default=800)
    p.add_argument('--paciencia', type=int, default=60)
    p.add_argument('--lote', type=int, default=32)
    p.add_argument('--taxa', type=float, default=0.001)
    p.add_argument('--l2', type=float, default=0.0001)
    p.add_argument('--otimizador', choices=['sgd_momentum','adam'], default='sgd_momentum')
    p.add_argument('--momentum-inicial', type=float, default=.10)
    p.add_argument('--momentum-maximo', type=float, default=.90)
    p.add_argument('--incremento-momentum', type=float, default=.01)
    p.add_argument('--fator-reducao', type=float, default=.50)
    p.add_argument('--momentum-fixo', action='store_true')
    p.add_argument('--salvar-cada', type=int, default=25)
    p.add_argument('--validacao-interna', type=float, default=.15)
    p.add_argument('--melhora-minima', type=float, default=.00001)
    p.add_argument('--tolerancia-adaptacao', type=float, default=1e-8)
    args = p.parse_args()
    output = args.saida or Path(__file__).parent / 'resultados' / ('r'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    cfg = Config(seed=args.seed, mode=args.modo, max_epochs=args.epocas, patience=args.paciencia,
                 batch_size=args.lote, learning_rate=args.taxa, l2=args.l2,
                 optimizer=args.otimizador, momentum_initial=args.momentum_inicial,
                 momentum_max=args.momentum_maximo, momentum_increase=args.incremento_momentum,
                 momentum_decrease=args.fator_reducao, adaptive_momentum=not args.momentum_fixo,
                 snapshot_interval=args.salvar_cada, validation_fraction=args.validacao_interna,
                 min_delta=args.melhora_minima, adaptation_tolerance=args.tolerancia_adaptacao,
                 record_calculations=not args.sem_memoria_calculo,
                 pca_enabled=args.pca, pca_variance=args.pca_variancia)
    print(WARNING, flush=True)
    try:
        schema = json.loads(args.esquema_entradas.read_text(encoding='utf8')) if args.esquema_entradas else None
        result = run_experiment(args.dados, output, cfg,
                 progress=lambda done, total, msg: print(f'[{done}/{total}] {msg}', flush=True), input_schema=schema)
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        p.exit(1, f'ERRO: {exc}\n')
    m = result['metrics']
    print(f"\nAcurácia OOF: {m['accuracy']:.2%} ({m['correct']}/{m['n']})")
    if m.get('classes', ['M','B']) == ['M','B']:
        print(f"Sensibilidade M: {m['sensitivity_M']:.2%}; especificidade B: {m['specificity_B']:.2%}")
        print(f"Malignos previstos como B: {m['false_negative_M_as_B']}")
        print('\nMédia ± STD amostral entre folds (M é a classe positiva):')
        metric_labels = [('accuracy','Acurácia'),('precision_M','Precisão M'),
                         ('sensitivity_M','Recall M'),('f1_M','F1-score M')]
    else:
        print('Classes: ' + ', '.join(m['classes']))
        print('\nMédia ± STD amostral entre folds (macro: mesmo peso por classe):')
        metric_labels = [('accuracy','Acurácia'),('precision_macro','Precisão macro'),
                         ('recall_macro','Recall macro'),('f1_macro','F1-score macro')]
    for key, label in metric_labels:
        value = result['fold_summary']['metrics'][key]
        avg = f"{value['mean']:.2%}" if value['mean'] is not None else 'indisponível'
        std = f"{value['std']:.2%}" if value['std'] is not None else 'indisponível'
        print(f"{label}: {avg} ± {std} ({value['n']} folds)")
    print(f"Tempo total: {result['timing']['total_seconds']:.1f} s")
    if args.pca:
        pca=result['final_model']['pca']
        print(f"PCA final: {pca['n_components']} componentes; variância {pca['retained_variance']:.2%}; corte {args.pca_variancia:.2%}")
    print(f'Arquivos: {output.resolve()}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
