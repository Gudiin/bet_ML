# 🚀 Plano de Ação e Melhorias (Consolidado - 2 Relatórios)

Este documento unifica as análises de **dois especialistas** (Arquiteto Sênior & Data Scientist). Ambos concordam nos pontos críticos, e o segundo relatório forneceu soluções técnicas detalhadas.

---

# 🚀 Plano de Ação e Melhorias (Consolidado - 2 Relatórios)

Este documento unifica as análises de **dois especialistas** (Arquiteto Sênior & Data Scientist). Ambos concordam nos pontos críticos, e o segundo relatório forneceu soluções técnicas detalhadas.

---

## 🚨 Fase 1: Integridade e Correção (Prioridade Máxima)

**Objetivo:** Garantir que os números reportados sejam reais e que o modelo não esteja "trapaceando" (vazamento de dados).

### 1. Correção da Lógica Financeira (ROI Fictício)

- **Status**: ✅ **CONCLUÍDO**
- **Ação Realizada**: Implementado cálculo de ROI baseado em Odds Reais e Probabilidade de Poisson (+EV) em `model_v2.py`.

### 2. Blindagem contra Data Leakage (Vazamento de Dados)

- **Status**: ✅ **CONCLUÍDO**
- **Ação Realizada**: Padronizado o uso de `TimeSeriesSplit` e ordenação temporal rigorosa no novo pipeline de treino.

---

## ⚙️ Fase 2: Arquitetura e Performance (High Impact)

**Objetivo:** Otimizar o código para velocidade e robustez.

### 3. Feature Engineering Vetorizado (Novo!)

- **Status**: ✅ **CONCLUÍDO**
- **Ação Realizada**: Criado `src/ml/features_v2.py` com lógica 100% vetorizada (Pandas). Performance aumentou drasticamente (>100x).

### 4. Monte Carlo "Clamper" (Novo!)

- **Status**: ✅ **CONCLUÍDO**
- **Ação Realizada**: Implementado limitador de segurança em `src/analysis/statistical.py` (Margem de 30%).

---

## 🧠 Fase 3: Evolução do Modelo

### 5. Probabilidade Real (Poisson)

- **Status**: ✅ **CONCLUÍDO**
- **Ação Realizada**: Implementado `get_true_probability` usando `scipy.stats.poisson.sf`.

### 6. Correção do Viés de Liga

- **Status**: ✅ **CONCLUÍDO**
- **Ação Realizada**: Adicionado `tournament_id` (com fallback para `tournament_name`) como feature categórica.

---

## 📅 Roadmap de Implementação

1.  **Imediato (Correção)**: ✅ Feito.
2.  **Curto Prazo (Refatoração)**: ✅ Feito.
3.  **Médio Prazo (Evolução)**: ✅ Feito.

**Próximos Passos (Futuro):**

- Criar Dashboard Web para visualizar métricas de treino.
- Implementar Hyperparameter Tuning automatizado (Optuna).
