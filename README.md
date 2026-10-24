# Laboratório MLP

Laboratório educacional de redes neurais com **equações, explicações e valores reais em cada etapa**. Funciona localmente, em CPU, com painel no navegador. Não exige conta, chave de API, GPU ou Codex.

Inclui treinamento de dados tabulares, seleção das colunas, múltiplas classes, PCA opcional, resultados por rodada, visualização da rede e dos pesos, previsões e memória de cálculo. A MLP é implementada explicitamente com NumPy: ReLU, Softmax, backpropagation, SGD com momentum e Adam.

**Somente para estudo. Não usar para diagnóstico, triagem ou decisões clínicas.** Os escores Softmax não são probabilidades clínicas calibradas.

## Começar sem esperar um treinamento

1. Clique em **Code → Download ZIP** ou clone este repositório. Extraia a pasta inteira.
2. Instale **Python 3.12**; também são aceitos 3.11 e 3.13. No Windows, marque a opção de adicionar Python ao PATH.
3. Execute o instalador e, depois, o inicializador correspondente:

| Sistema | Instalar uma vez | Abrir o laboratório |
|---|---|---|
| Windows | `instalar_windows.bat` | `iniciar_windows.bat` |
| Linux | `bash instalar_linux.sh` | `bash iniciar_linux.sh` |
| macOS | `bash instalar_macos.sh` | `bash iniciar_macos.sh` |

O navegador abre em **http://127.0.0.1:8501/**. A instalação inicial requer internet; depois, a base e a demonstração incluídas funcionam offline. Não execute dentro do ZIP.

Selecione **Demonstração pública · adam · até 3 épocas**. Abra **Rede e pesos reais**, escolha um neurônio e clique em **Abrir memória de cálculo**. Não é necessário treinar para explorar essas telas.

Com o servidor aberto, estes atalhos dão acesso direto:

- [Painel](http://127.0.0.1:8501/)
- [Equações com valores reais](http://127.0.0.1:8501/calculos?run=exemplo&model=final&checkpoint=trained&neuron=0&sample=0)
- [Primeira atualização real dos pesos](http://127.0.0.1:8501/calculos?run=exemplo&model=final&checkpoint=trained&neuron=0&sample=0&step=1)
- [Guia completo](http://127.0.0.1:8501/guia)

Esses endereços acessam o computador de quem os abre. O GitHub distribui o programa; cada colega o executa no próprio computador.

## O que aparece nas equações

A página **Memória de cálculo** apresenta a fórmula, seu significado, a substituição numérica e tabelas com as parcelas. É possível trocar a amostra, o neurônio e o checkpoint, imprimir/salvar PDF e baixar os valores em JSON.

1. Padronização: `Zᵢ = (xᵢ − mᵢ) / sᵢ`, usando média e escala salvas do treino.
2. Soma das entradas: `zⱼ = Σᵢ Zᵢ W1ᵢⱼ + b1ⱼ`.
3. Ativação: `aⱼ = max(0, zⱼ)` (ReLU).
4. Soma de todos os ocultos nas saídas: `uₖ = Σⱼ aⱼ W2ⱼₖ + b2ₖ`.
5. Softmax estável: `pₖ = exp(uₖ − c) / Σᵣ exp(uᵣ − c)`, com `c = max(u)`; WTA escolhe a classe de maior escore.
6. Entropia cruzada e penalidade L2.
7. Derivadas parciais, Jacobiana da Softmax, regra da cadeia e backpropagation.
8. Gradientes de pesos e vieses, incluindo o sinal negativo quando correto.
9. Na fonte **Atualização real gravada**, gradiente médio do lote, estado do otimizador e pesos antes/depois do passo.

**Estado do modelo / checkpoint** calcula uma passagem e um gradiente neste estado. **Atualização real gravada** lê um lote que efetivamente ocorreu. A aplicação distingue essas duas fontes; não inventa trajetórias de modelos antigos. O motor não usa integrais. A derivada é explicada como inclinação local da tangente, com a convenção ReLU′(0) = 0.

## Arquitetura e treinamento

Para D entradas e S classes, a regra do projeto é **H = teto((D + S)/2)** e a rede é **D → H → S**. É uma convenção didática, não uma garantia de arquitetura ótima. O exemplo WDBC tem 30 → 16 → 2 e 530 parâmetros. ID e alvo não entram como medidas.

O editor reconhece CSV, TSV e TXT; confira separador, cabeçalho e funções das colunas antes de treinar. O exemplo binário usa saídas `[M, B]`, com M = `10` e B = `01`. Outras classes mantêm uma ordem determinística salva, com uma posição one-hot por classe. Há perfis de leitura para WDBC, Iris, Wine, Abalone e proteína; somente o WDBC está incluído nesta distribuição.

Cada fold começa com pesos e otimizador novos. A validação interna escolhe a duração; em seguida, normalização, PCA e pesos são reiniciados para o retreino externo. O teste externo não escolhe pesos, época, normalizador, PCA ou momentum. Cada amostra aparece uma vez em teste por execução. O modelo final é ajustado com toda a base e não recebe um novo teste independente.

PCA começa desligado; quando ativado, usa inicialmente 95% de variância preservada. É ajustado somente no treino de cada fase e salvo com o respectivo modelo. A memória mostra também a projeção Z → PC → MLP; previsões recebem as medidas originais.

Precisão, recall e F1 usam M como classe positiva no caso M/B e média macro nas demais classes. A média ± desvio-padrão amostral entre folds é separada das métricas OOF globais. Ranking e melhores rodadas são descritivos; escolher repetidamente configurações pelo teste exige avaliação adicional independente.

Para uma primeira experiência de treinamento, ajuste **Épocas máximas** para **3 ou 5** antes de iniciar. O padrão completo preservado do motor é 800, com parada antecipada e paciência 60; ele pode demorar, especialmente ao registrar cada lote. A memória detalhada continua ativada por padrão e aumenta o uso de disco. Consultar a demonstração não inicia treinamento.

## Dados e demonstração incluídos

`dados/cancear.txt` foi preparado de um download novo do **Breast Cancer Wisconsin (Diagnostic)** oficial da UCI: 569 amostras, 30 medidas e alvo M/B. A documentação e os registros públicos foram concatenados, sem alterar os valores. [Origem, licença CC BY 4.0, citação e hash](dados/ORIGEM.md).

`resultados_exemplo/` é uma demonstração **nova e curta**, gerada para esta distribuição: semente 42, Adam, lote 256, limite de três épocas, sete rodadas no protocolo variável e modelo final. Os checkpoints e as memórias correspondem aos cálculos reais desse exemplo. Ele serve para explorar o sistema, não representa o desempenho dos experimentos de pesquisa ou um modelo otimizado.

Para gerar outra demonstração, escolha uma pasta nova/vazia:

```bash
python scripts/gerar_demonstracao.py --saida resultados/r20261024_120000_000001
```

Execute com o Python do ambiente virtual. O programa recusa sobrescrever resultados existentes. A geração não acontece durante instalação ou abertura do painel.

## Arquivos locais e privacidade

As novas execuções ficam em `resultados/` e as bases importadas em `dados/importados/`. Previsões também guardam uma cópia da entrada. Essas pastas são ignoradas pelo Git, assim como ambientes virtuais, arquivos `.env`, chaves e logs. A demonstração incluída deriva somente da base pública identificada acima.

O projeto não contém credenciais, conversas, relatórios pessoais, arquivos importados ou históricos antigos. O processamento usa o computador local e não envia as bases a serviços externos. O servidor permanece em `127.0.0.1`; compartilhar o código no GitHub não publica seu servidor nem suas pastas pessoais.

Checkpoints são para inspeção; esta versão não restaura exatamente o otimizador de um treino interrompido. Para encerrar um treinamento em execução, use **Ctrl+C no terminal do servidor**; os artefatos já gravados são preservados, e a execução incompleta não vira resultado concluído. Fechar apenas o navegador não encerra o processo.

## Problemas comuns

- **Porta ocupada:** `bash iniciar_linux.sh --port 8502` (macOS: `iniciar_macos.sh`; Windows: `iniciar_windows.bat --port 8502`). Use então o endereço indicado no terminal.
- **Python 3.14 ou versão antiga:** instale 3.12. Para escolher no Linux/macOS, use `PYTHON=/caminho/python3.12 bash instalar_linux.sh` (ou `instalar_macos.sh`). No Windows, defina `PYTHON` com o caminho completo de `python.exe`, sem argumentos.
- **Ambiente incompatível:** o instalador preserva `.venv`; renomeie essa pasta e instale novamente. Não compartilhe ambientes virtuais entre computadores.
- **Falta do módulo venv no Linux:** instale o pacote `venv` correspondente ao seu Python pelo gerenciador do sistema.

## Desenvolvimento e verificação

Com o ambiente ativado:

```bash
python -m pip check
python -m unittest discover -s tests -v
node tests/dashboard_check.js
node tests/input_editor_check.js
node tests/training_progress_check.js
node tests/pca_dashboard_check.js
```

Os testes incluem diferenças finitas dos gradientes, separação treino/validação/teste, modelos salvos, memória por lote, PCA, importação, HTTP e a demonstração distribuída. O workflow do GitHub executa instalação e verificações em Linux, Windows e macOS com Python 3.12. Node é necessário somente para as verificações de desenvolvimento, não para usar o laboratório.

Código e documentação do projeto: [licença MIT](LICENSE). Dados WDBC: [CC BY 4.0](dados/ORIGEM.md).
