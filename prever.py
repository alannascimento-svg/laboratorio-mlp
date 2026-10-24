"""Classifica CSV usando o esquema salvo e arquiva todas as inferências."""
from __future__ import annotations
import argparse
from pathlib import Path
from rede import WARNING, save_prediction

def main():
    p=argparse.ArgumentParser(description=WARNING)
    p.add_argument('--modelo',type=Path,required=True)
    p.add_argument('--entrada',type=Path,required=True)
    p.add_argument('--saida',type=Path,default=None,help='Cópia CSV adicional; não sobrescreve')
    args=p.parse_args()
    try:
        if args.saida is not None and args.saida.exists():
            raise FileExistsError('A saída já existe; escolha outro nome.')
        result,folder=save_prediction(args.modelo,args.entrada.read_text(encoding='utf-8-sig'),
                                      args.modelo.parent/'inferencias')
        if args.saida is not None:
            args.saida.parent.mkdir(parents=True,exist_ok=True)
            # Exclusivo evita sobrescrita mesmo com processos simultâneos.
            with args.saida.open('x',encoding='utf8',newline='') as f:result.to_csv(f,index=False)
    except (ValueError,OSError) as exc:p.exit(1,f'ERRO: {exc}\n')
    print(WARNING)
    print(result[['classe_prevista','codigo_WTA','score_M','score_B']].to_string(index=False))
    print(f'Entrada, previsões e metadados: {folder.resolve()}')
    return 0
if __name__=='__main__':raise SystemExit(main())
