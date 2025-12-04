# 🚀 Plano de Ação e Melhorias (Consolidado - 2 Relatórios)

Este documento unifica as análises de **dois especialistas** (Arquiteto Sênior & Data Scientist). Ambos concordam nos pontos críticos, e o segundo relatório forneceu soluções técnicas detalhadas.

---

## 🚨 Fase 1: Integridade e Correção (Prioridade Máxima)

**Objetivo:** Garantir que os números reportados sejam reais e que o modelo não esteja "trapaceando" (vazamento de dados).

### 1. Correção da Lógica Financeira (ROI Fictício)

- **Diagnóstico (Consenso)**: O código atual usa `avg_odd = 1.90` hardcoded. Isso gera resultados ilusórios.
- **Ação**:
  - Implementar cálculo de ROI baseado na **Odd Real** ou simulação dinâmica.
  - Adicionar métricas de negócio no log de treino: **Win Rate** e **ROI Estimado** (não apenas MAE).

### 2. Blindagem contra Data Leakage (Vazamento de Dados)

- **Diagnóstico (Consenso)**: O uso de `train_test_split` com `shuffle=True` mistura passado e futuro.
- **Ação**:
  - Padronizar o uso de `TimeSeriesSplit` ou corte manual por data (`train < data < test`).
  - Garantir que o dataset de treino contenha apenas jogos finalizados.

---

## ⚙️ Fase 2: Arquitetura e Performance (High Impact)

**Objetivo:** Otimizar o código para velocidade e robustez.

### 3. Feature Engineering Vetorizado (Novo!)

- **Diagnóstico (Relatório 2)**: O arquivo `feature_extraction.py` itera linha por linha (lento). O `features_v2.py` é melhor, mas pode ser aprimorado.
- **Ação**:
  - **Centralizar tudo em `features_v2.py`** usando abordagem 100% vetorizada (Pandas `groupby` + `shift`).
  - **Deletar `feature_extraction.py`** (código legado/lento).
  - Implementar a estratégia "Team-Centric" sugerida: transformar partidas em linhas de tempo por time para calcular médias móveis com precisão.

### 4. Monte Carlo "Clamper" (Novo!)

- **Diagnóstico (Relatório 2)**: Se o modelo de ML "alucinar" (ex: prever 20 escanteios), ele contamina a simulação de Monte Carlo.
- **Ação**:
  - Adicionar um **Limitador (Clamper)** na classe `StatisticalAnalyzer`.
  - Regra: A média ajustada não pode desviar mais de **30%** da média histórica, independente da previsão da IA.

---

## 🧠 Fase 3: Evolução do Modelo

### 5. Probabilidade Real (Poisson)

- **Diagnóstico (Consenso)**: O modelo deve prever probabilidade, não apenas média.
- **Ação**:
  - Confirmar uso de `objective='poisson'` no LightGBM.
  - Implementar `scipy.stats.poisson.sf` para decisão de aposta (+EV).

### 6. Correção do Viés de Liga

- **Ação**: Adicionar `tournament_id` como feature categórica e features relativas (`Média Time / Média Liga`).

---

## 📅 Roadmap de Implementação

1.  **Imediato (Correção)**:
    - Arrumar validação temporal (`TimeSeriesSplit`).
    - Implementar o "Clamper" no Monte Carlo (proteção rápida).
2.  **Curto Prazo (Refatoração)**:
    - Reescrever `features_v2.py` (Vetorizado) e apagar o antigo.
    - Corrigir cálculo de ROI nos logs.
3.  **Médio Prazo (Evolução)**:
    - Implementar lógica de Poisson (+EV) para apostas.
