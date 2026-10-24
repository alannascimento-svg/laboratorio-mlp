"""Gera uma execução curta nova usando apenas o WDBC público incluído."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rede import Config, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saida', type=Path, required=True,
                        help='Pasta nova ou vazia; nunca sobrescreve uma execução.')
    args = parser.parse_args()
    cfg = Config(optimizer='adam', max_epochs=3, batch_size=256,
                 snapshot_interval=1, record_calculations=True)
    result = run_experiment(ROOT/'dados'/'cancear.txt', args.saida, cfg,
                            progress=lambda done,total,message: print(f'{done}/{total} · {message}', flush=True))
    manifest = {'description':'Demonstração pública curta, gerada separadamente dos históricos pessoais.',
                'dataset':'Breast Cancer Wisconsin (Diagnostic), UCI, DOI 10.24432/C5DW2B',
                'dataset_license':'CC BY 4.0', 'max_epochs':3, 'batch_size':256,
                'optimizer':'adam', 'purpose':'Explorar a interface e conferir equações; não otimizar desempenho.',
                'data_sha256':result['final_model']['data_sha256']}
    (args.saida/'DEMONSTRACAO.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    main()
