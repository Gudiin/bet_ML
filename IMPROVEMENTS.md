# 🚀 Análise de Melhorias e Pontos Críticos

Este documento detalha os pontos de melhoria identificados no projeto após uma análise técnica profunda da arquitetura, código e metodologias utilizadas.

---

## 🚨 Pontos Críticos (Prioridade Alta)

Estes pontos podem afetar a confiabilidade das previsões ou a estabilidade do sistema.

### 1. Validação Temporal no Treinamento Padrão

- **Problema**: O método `train()` em `src/ml/model_improved.py` utiliza `train_test_split` com `random_state=42`. Embora as features usem janelas deslizantes (o que mitiga o vazamento de dados), misturar jogos de 2023 com 2024 no treino/teste pode criar um viés otimista. O futebol muda taticamente ao longo do tempo.
- **Solução**: Padronizar o uso de `TimeSeriesSplit` (já presente em `train_with_optimization`) ou fazer um split manual baseado em data (ex: Treino = Jan-Out, Teste = Nov-Dez).

### 2. Hardcoding no "Clamper" (Mecanismo de Segurança)

- **Problema**: Em `src/analysis/statistical.py`, o mecanismo que limita a previsão da IA (Clamper) tem um valor fixo de **30%** (`max_deviation = 0.30`).
- **Risco**: Em ligas muito voláteis ou jogos de copa, a IA pode estar correta ao prever algo fora da curva, mas será "censurada" por esse limite fixo.
- **Solução**: Tornar este parâmetro configurável ou dinâmico, baseado na variância histórica da liga específica.

### 3. Dependência de Bibliotecas Opcionais

- **Problema**: O código tenta importar `lightgbm` e `xgboost` e faz fallback para `RandomForest` se falhar.
- **Risco**: Se o ambiente de produção não tiver essas libs instaladas (o que pode acontecer silenciosamente), o modelo cairá para uma performance inferior sem um aviso muito explícito (apenas um print).
- **Solução**: Adicionar logs de alerta mais robustos ou falhar explicitamente se o modo "Ensemble" for solicitado mas as libs não estiverem presentes.

---

## ⚠️ Melhorias Técnicas (Prioridade Média)

Melhorias que visam a manutenibilidade e a qualidade do código.

### 1. Tratamento de "Cold Start" (Início de Temporada)

- **Problema**: O `feature_engineering.py` remove linhas com `NaN`. Isso significa que as primeiras 5 rodadas de cada time são ignoradas no treinamento.
- **Impacto**: Perdemos dados valiosos do início de campeonatos.
- **Sugestão**: Implementar uma janela dinâmica (ex: na rodada 2, usar média dos últimos 1 jogos) ou imputar dados com médias da temporada anterior.

### 2. Duplicação de Código de Modelos

- **Problema**: Existem arquivos `model.py`, `model_v2.py` e `model_improved.py`.
- **Impacto**: Confusão sobre qual é a "verdade" do projeto.
- **Sugestão**: Consolidar tudo em uma estrutura limpa, talvez movendo versões antigas para uma pasta `legacy/` ou refatorando para uma classe base única.

### 3. Logs e Observabilidade

- **Problema**: O sistema usa muitos `print()`.
- **Sugestão**: Implementar o módulo `logging` do Python. Isso permitiria salvar logs em arquivo para debug posterior ("Por que o sistema previu X naquele jogo de ontem?").

---

## 💡 Melhorias de Produto (Visão de Futuro)

Sugestões para evoluir o produto.

### 1. Análise de "Momentum" Intra-jogo

- **Ideia**: Se tivermos acesso a dados ao vivo, poderíamos ajustar a previsão do Poisson/Monte Carlo em tempo real (ex: saiu um gol aos 10min, a expectativa de escanteios muda).

### 2. Fator "Must Win"

- **Ideia**: Adicionar uma feature que indique a necessidade de vitória (ex: final de campeonato, luta contra rebaixamento). Times desesperados tendem a gerar mais escanteios no final do jogo.

### 3. Backtesting Automatizado

- **Ideia**: Criar um script que roda o modelo em todos os jogos de 2023 e calcula exatamente qual teria sido o ROI (Retorno sobre Investimento) se tivéssemos apostado R$ 10,00 em cada sugestão "Easy". Isso valida a estratégia financeiramente.
