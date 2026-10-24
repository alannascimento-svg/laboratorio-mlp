# Regras de manutenção

- Leia README.md e rede.py antes de modificar o motor. Preserve a finalidade educacional e o funcionamento local.
- Arquitetura D → teto((D+S)/2) → S, ReLU oculta e Softmax estável. ID/alvo ficam fora de X. Preserve classes, nomes e ordem salva; legado M/B usa [M,B] e códigos10/01.
- Backpropagation explícita. Gradientes negativos são válidos. SGD: v=mu*v-eta*g; w+=v. Adaptação usa somente perda/gradientes completos de treino.
- Nunca ajustar normalização, PCA, pesos, parada ou hiperparâmetros pelo teste externo. Reinicie pesos, normalizador e otimizador em cada fase/fold. Preserve partições estratificadas e grupos disjuntos.
- Nunca sobrescreva ou apague resultados anteriores. Histórico ruim também é histórico. Consultas de modelos e SQLite são somente leitura, sem migrações automáticas.
- Snapshot explica um estado; memória de atualização representa um lote realmente gravado. Não invente trajetórias. Checkpoints não implementam retomada exata do otimizador.
- Preserve registro de cada lote quando habilitado, transações e fechamento do gravador em finally. Gravar a memória não pode alterar o resultado numérico.
- PCA é opcional, ajustado somente no treino de cada fase; preserve transformação, medidas originais e arquitetura de cada modelo.
- Não inclua bases próprias, resultados pessoais, credenciais, logs ou ambientes virtuais no Git. A demonstração pública tem origem documentada em dados/ORIGEM.md. Não a substitua por históricos locais.
- Após alterações, execute unittest completo, verificações JS pertinentes, HTTP e interface. Mudanças metodológicas exigem versão, documentação, assinatura de comparação e testes atualizados.
