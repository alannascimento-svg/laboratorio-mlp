# Dados públicos incluídos

Fonte: [Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast-cancer).

Citação: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

Licença dos dados: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). A licença MIT do código não substitui a licença desta base.

Os arquivos foram obtidos diretamente do ZIP oficial da UCI em 24/10/2026. `wdbc.names` é a documentação original; `cancear.txt` concatena essa documentação (sem espaços finais, seguida de duas quebras de linha) e os 569 registros de `wdbc.data`, sem alterar as medidas ou os rótulos. O nome `cancear.txt` foi mantido para compatibilidade com o importador original. `exemplo_previsao.csv` contém as três primeiras amostras públicas, com cabeçalho e sem a coluna alvo.

SHA-256 de `wdbc.data` no ZIP oficial:
`d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a`

A demonstração em `resultados_exemplo/` foi gerada do zero sobre esta cópia pública: até três épocas, Adam, lote 256 e semente 42. Contém pesos, previsões, partições, checkpoints e atualizações reais para explicar os cálculos. Não contém históricos de pesquisa ou arquivos importados por usuários. Seu desempenho não é um modelo otimizado nem uma validação clínica.
