# 🤖 Sistema de Previsão de Escanteios (Professional v8.0 Next Gen)

> **Versão 8.0 - "Next Gen"**  
> _Transfer Learning + Ensemble + Odds Reais + Multi-League_

Sistema profissional de Machine Learning para previsão de escanteios (futebol), projetado para encontrar **Valor Esperado (+EV)** real usando IA avançada e odds históricas.

---

## 🎯 O Que Mudou na v8.0? (Next Gen)

A v8.0 representa um salto quântico na arquitetura do projeto. Saímos de análises estatísticas puras para um sistema de IA híbrida treinado com dados da elite europeia.

| Tecnologia      | Antes (v7.0)            | **Agora (v8.0 Next Gen)**                                      |
| :-------------- | :---------------------- | :------------------------------------------------------------- |
| **Arquitetura** | Modelo Único (LightGBM) | **Ensemble Híbrido** (LightGBM + CatBoost + Linear)            |
| **Aprendizado** | Treinamento Padrão      | **Transfer Learning** (Global Model ➔ Fine-Tuning por Liga)    |
| **Validação**   | Backtest Estatístico    | **ROI Real** usando Odds Históricas (Bet365/Pinnacle)          |
| **Features**    | Janelas Estáticas       | **Janelas Dinâmicas** (3, 5, 10, 20 jogos) + Posição Histórica |
| **Escopo**      | Foco Brasil             | **Multi-League** (Premier League, LaLiga, Serie A, etc.)       |

---

## 🏗️ Arquitetura do Sistema

```mermaid
graph TD
    A[SofaScore API] -->|Stats| C(Feature Engineering V2)
    B[Football-Data.co.uk] -->|Odds Históricas| C

    C --> D{Modelagem Híbrida}

    subgraph "Cérebro da IA (Ensemble)"
    D --> E[Global Model]
    E --> F[LightGBM (Velocidade)]
    E --> G[CatBoost (Precisão)]
    E --> H[Linear Regression (Tendência)]
    end

    D --> I{Transfer Learning}
    I -->|Ligas Grandes >100| J[Fine-Tuning Específico]
    I -->|Ligas Pequenas| K[Usa Global Model]

    J --> L[Previsão Final]
    K --> L

    L --> M[Scanner de Oportunidades]
    M --> N[Relatório +EV]
```

---

## 🧠 Inteligência Artificial

O sistema utiliza uma abordagem de **Stacking Ensemble** com calibração automática:

1.  **LightGBM (Tweedie)**: Captura a não-linearidade e picos de escanteios (ex: jogos com 15+ cantos).
2.  **CatBoost**: Excelente para lidar com features categóricas e evitar overfitting em datasets menores.
3.  **Regressão Linear**: Fornece uma base sólida e captura tendências de longo prazo.

### Transfer Learning

A IA aprende "futebol" observando 4.000+ jogos da Premier League, LaLiga e Serie A.

- **Fase 1 (Global):** Aprende padrões universais (ex: time perdendo ataca mais).
- **Fase 2 (Fine-Tuning):** Ajusta os detalhes para cada campeonato (ex: Brasileirão tem mais faltas, Premier League é mais rápida).

---

## 📊 Métricas de Performance (Validado em 4.000 Jogos)

Resultados baseados em **Validação Cruzada Temporal (Time Series Split)** usando odds reais de fechamento:

| Métrica              | Performance           | Significado                                   |
| :------------------- | :-------------------- | :-------------------------------------------- |
| **MAE** (Erro Médio) | **~2.6 - 2.8**        | A IA erra, em média, menos de 3 escanteios.   |
| **ROI** (Retorno)    | **+14% a +18%**       | Lucro consistente simulando apostas em valor. |
| **Cobertura**        | **Top 5 Europa + BR** | Testado nas ligas mais difíceis do mundo.     |

> **Nota:** O ROI é calculado apenas em situações onde a IA detecta uma discrepância significativa entre a probabilidade calculada e a Odd da casa (Value Bet).

---

## 📦 Instalação e Uso

### 1. Instalação

```bash
git clone https://github.com/seu-usuario/projeto-bet.git
cd projeto-bet
pip install -r requirements.txt
```

### 2. Executar o Sistema

```bash
python src/main.py
```

### 3. Menu Principal

1.  **Atualizar Base**: Baixa dados recentes (SofaScore).
2.  **Treinar Modelo (New)**: Executa o pipeline v8 (Optuna + Transfer Learning).
3.  **Scanner de Oportunidades**:
    - **Opção 7**: Varre jogos de Hoje, Amanhã ou Data Específica.
    - Analisa probabilidades vs Odds reais.
    - Indica **Verde** (Aposta Segura) ou **Vermelho** (Sem Valor).

---

## 📂 Estrutura de Pastas (Atualizada)

- `src/ml/model_v2.py`: O novo cérebro (Ensemble + Transfer Learning).
- `src/ml/features_v2.py`: Engenharia de features dinâmica.
- `src/data/external`: Gerenciadores de Odds externas (Football-Data).
- `src/scrapers`: Coleta de estatísticas (SofaScore).
- `data/football_data.db`: Banco SQLite unificado (Stats + Odds).

---

## ⚠️ Disclaimer

Apostas esportivas envolvem alto risco. Este software é uma ferramenta de **análise estatística** e não garante lucros futuros. O ROI passado não é garantia de ROI futuro. Use com responsabilidade.

---

**Desenvolvido com Python 3.12 + LightGBM + CatBoost**
