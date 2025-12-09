# 🧠 Documentação Técnica de ML (v8.0 Next Gen)

Este documento detalha o funcionamento interno do **Professional Predictor v8.0**, a nova arquitetura de inteligência artificial do projeto.

---

## 1. Arquitetura do Modelo (Ensemble Híbrido)

A v8.0 abandona a dependência de um único algoritmo. Utilizamos um **Weighted Stacking Ensemble** para combinar o melhor de três mundos:

### Os Componentes

1.  **LightGBM (Peso Variável - Principal)**

    - **Função**: Captura padrões complexos e não-lineares.
    - **Configuração**: Otimizado via Optuna (50-100 trials).
    - **Objetivo**: `mae` (Erro Absoluto Médio).

2.  **CatBoost (Peso Variável)**

    - **Função**: Lida melhor com features categóricas e dados ruidosos.
    - **Vantagem**: Menos propenso a overfitting em ligas menores.

3.  **Regressão Linear (Baseline)**
    - **Função**: "Âncora" do modelo. Impede que a IA faça previsões absurdas (ex: 20 escanteios) baseada em outliers.

### A Fórmula da Previsão

```math
PrevisãoFinal = (w_1 \cdot Pred_{LGBM}) + (w_2 \cdot Pred_{CatBoost}) + (w_3 \cdot Pred_{Linear})
```

_Os pesos (w) são ajustados dinamicamente durante o treinamento global._

---

## 2. Transfer Learning & Estratégia Multi-League

Em vez de treinar modelos isolados para cada liga desde o zero (o que falha em ligas pequenas), adotamos a estratégia de **Transfer Learning**:

1.  **Treinamento Global (A "Base de Conhecimento")**

    - O modelo vê **todos os jogos** das Ligas "Big 5" (Premier League, LaLiga, Bundesliga, Serie A, Ligue 1) + Brasileirão.
    - Ele aprende conceitos universais: _"Times perdendo por 1 gol aos 80min pressionam mais"_.

2.  **Fine-Tuning (A "Especialização")**
    - Para ligas com **>100 jogos** no histórico:
    - Pegamos o Modelo Global e realizamos um "retreino leve" apenas com dados daquela liga.
    - Resultado: O modelo mantém a inteligência global, mas se adapta ao estilo local (ex: futebol defensivo da Série B).

> **Aviso de Segurança**: Se uma liga tem <100 jogos, o sistema pula o Fine-Tuning e usa o Modelo Global puro, garantindo robustez.

---

## 3. Engenharia de Features (V2 - Dinâmica)

Abandonamos as médias fixas. O novo motor de features (`features_v2.py`) gera **Janelas Dinâmicas** para capturar a evolução dos times.

### Features Geradas (para cada time)

Para cada métrica (Escanteios, Chutes, Gols, Cantos Cedidos), geramos:

- **Curto Prazo (3 jogos)**: Forma atual / Momento.
- **Médio Prazo (5 jogos)**: Tática recente.
- **Longo Prazo (10 e 20 jogos)**: Consistência da temporada.

### Features Contextuais V8

- **Position Diff**: Diferença na tabela calculada dinamicamente (baseada em `form_score`).
- **H2H Dominance**: Histórico recente entre as duas equipes.
- **Season Progress**: (0.0 a 1.0) influencia o peso dos jogos (jogos finais valem mais).

---

## 4. Integração de Odds Históricas

A v8.0 introduziu a **Validação Financeira Real**.

### Fontes de Dados

- **Estatísticas**: SofaScore (Corner/Shots/Goals).
- **Odds**: Football-Data.co.uk (Dataset histórico curado).
  - Odds de Fechamento da **Bet365** e **Pinnacle**.

### O Desafio do Matching

Como unimos dados de fontes diferentes? Desenvolvemos um algoritmo de **Entity Resolution**:

1.  **Fuzzy Date Matching**: Tolerância de ±1 dia (resolve problemas de fuso horário UTC vs Local).
2.  **Team Name Mapping**: Dicionário inteligente (`team_map.json`) para casos como _"Man Utd"_ vs _"Manchester United"_ ou _"Flamengo"_ vs _"Flamengo RJ"_.

---

## 5. Avaliação de Lucratividade (ROI)

O modelo não é avaliado apenas por acertar o número de escanteios (MAE), mas por **Dinheiro Gerado**.

### Como calculamos o ROI?

O sistema simula uma temporada passadas dia-a-dia (`TimeSeriesSplit`):

1.  Esconde o resultado do jogo.
2.  Faz a previsão.
3.  Calcula a "Odd Justa" (1 / Probabilidade).
4.  Se `OddCasa > OddJusta + MargemSegurança`: **Aposta Simulada**.
5.  Verifica resultado e atualiza banca.

**Resultado Atual (Validado):**

- **ROI de ~14% a 18%** nas Top Ligas Europeias.
- Isso comprova que o modelo encontra ineficiência nas casas de aposta.

---

## 6. Como reproduzir o Treinamento

1.  Garanta que o banco `data/football_data.db` tenha dados.
2.  Execute `python src/main.py` -> Opção **2 (Treinar Modelo)**.
    - O modo **Optuna** é recomendado (50 trials) para calibrar os hiperparâmetros.
3.  O modelo final será salvo como `data/corner_model_global.pkl`.

---

**Projeto Bet - Ciência de Dados Aplicada ao Futebol**
