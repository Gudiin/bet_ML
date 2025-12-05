# 🧠 Documentação Técnica: Machine Learning (V2)

Este documento detalha a engenharia e a matemática por trás do **Professional Predictor V2**, o cérebro do nosso sistema de previsões.

---

## 1. O Problema: Previsão de Escanteios

Escanteios são eventos de contagem (números inteiros não-negativos: 0, 1, 2...).

- **Erro Comum**: Tratar como regressão normal (Gaussiana), que assume distribuição simétrica e números contínuos.
- **Solução V2**: Usamos **Regressão de Poisson**, ideal para modelar a taxa de ocorrência de eventos raros em um intervalo de tempo.

---

## 2. Pipeline de Dados (Feature Engineering)

O arquivo `src/ml/features_v2.py` é responsável por transformar dados brutos de partidas em sinais matemáticos para o modelo.

### ⚡ A Revolução Vetorizada

Antigamente, iterávamos jogo a jogo (loop `for`), o que era lento. Agora, usamos **Vetorização do Pandas**:

1.  **Team-Centric View**: Duplicamos o dataset para ter uma linha por time, não por jogo.
2.  **GroupBy + Shift**: Agrupamos por time e deslocamos os dados 1 linha para baixo.
    - _Por que?_ Para garantir que a feature do jogo atual use apenas dados dos jogos **anteriores**. Isso elimina o **Data Leakage**.
3.  **Rolling Windows**: Calculamos médias móveis em janelas deslizantes.

### As Features (Variáveis)

O modelo aprende com:

- **Médias Móveis (3 e 5 jogos)**: Escanteios, Chutes, Gols.
- **Tendência (Trend)**: Diferença entre média curta (3j) e longa (5j). Indica se o time está melhorando ou piorando.
- **Força Relativa**: Diferença entre a média de escanteios do Mandante e do Visitante.
- **Contexto da Liga (`tournament_id`)**: O modelo aprende que a média de escanteios na Premier League é diferente do Brasileirão Série B.

---

## 3. O Modelo (LightGBM)

Usamos o **LightGBM**, um algoritmo de Gradient Boosting (árvores de decisão) extremamente rápido e eficiente.

- **Objective**: `poisson` (Otimiza a verossimilhança de Poisson).
- **Métrica**: `mae` (Erro Médio Absoluto) para monitoramento, mas o foco real é o ROI.

---

## 4. Validação Temporal (Time Series Split)

Em séries temporais (futebol), não podemos embaralhar os dados (`shuffle=True`). Se fizermos isso, o modelo aprenderá com jogos de 2025 para prever jogos de 2024 (trapaça!).

**Como fazemos na V2 (`model_v2.py`):**
Usamos `TimeSeriesSplit`. O treino cresce progressivamente:

- Split 1: Treina (Jan-Mar) -> Testa (Abr)
- Split 2: Treina (Jan-Abr) -> Testa (Mai)
- Split 3: Treina (Jan-Mai) -> Testa (Jun)

Isso simula o cenário real de produção.

---

## 5. Matemática Financeira (+EV)

Não basta acertar a média de escanteios. Precisamos saber se a aposta vale a pena.

### Probabilidade Real (Poisson)

O modelo prevê o **Lambda (λ)**, que é a média esperada de escanteios.
Para saber a probabilidade de sair **Mais de 9.5 escanteios** (Over 9.5), usamos a função de sobrevivência de Poisson:

$$ P(X > 9.5) = \text{poisson.sf}(9, \lambda) $$

### Valor Esperado (EV)

Calculamos o Valor Esperado de cada aposta:

$$ EV = (Probabilidade \times Odd) - 1 $$

Se $EV > 0.05$ (5%), o sistema sugere a aposta. Isso garante lucratividade a longo prazo, filtrando apostas onde o risco não compensa o retorno.

---

## 6. O "Clamper" (Segurança)

Para evitar que um erro do modelo (ex: prever 25 escanteios) quebre a banca, implementamos um **Limitador** na simulação de Monte Carlo.

- A previsão da IA nunca pode desviar mais de **30%** da média histórica dos times.
- Isso cria um sistema híbrido: **Inteligência da IA + Segurança da Estatística Clássica**.
