# 🤖 Sistema de Previsão de Escanteios (Professional V2)

> **Versão 2.0 - "The Vectorized Update"** > _Performance Extrema, Lógica Financeira Real e Anti-Leakage._

Este projeto é um sistema completo de Machine Learning para previsão de escanteios em futebol, focado em encontrar apostas de valor (+EV) usando dados históricos e estatísticas avançadas.

---

## 🚀 O Que Há de Novo? (V2)

Esta versão traz uma reformulação completa do núcleo de inteligência artificial:

- **⚡ Feature Engineering Vetorizado**: Processamento de dados >100x mais rápido usando operações vetoriais do Pandas.
- **💰 Lógica Financeira Real**: Cálculo de ROI baseado em Odds Reais e Probabilidade de Poisson (não mais odds fixas).
- **🛡️ Anti-Data Leakage**: Validação temporal rigorosa (`TimeSeriesSplit`) garante que o modelo nunca veja o futuro.
- **🧠 Inteligência de Liga**: O modelo agora entende o contexto do campeonato (`tournament_id`), diferenciando Premier League de Série B.
- **⚖️ Monte Carlo "Clamper"**: Proteção estatística que impede alucinações do modelo de contaminarem as simulações.

---

## 🛠️ Arquitetura

O sistema é dividido em três pilares principais:

1.  **Coleta de Dados (Scraper)**:

    - Automação via Selenium para extrair dados do SofaScore.
    - Armazenamento em SQLite (`football_data.db`).

2.  **Inteligência Artificial (Machine Learning)**:

    - **Modelo**: LightGBM Regressor (Objective: Poisson).
    - **Features**: Médias móveis (3/5 jogos), Tendências, Força Relativa, Confronto Direto.
    - **Validação**: Cross-Validation Temporal (respeita a ordem cronológica).

3.  **Análise Estatística (Monte Carlo)**:
    - Simula cada jogo 10.000 vezes.
    - Combina a previsão da IA com a variância histórica dos times.
    - Gera probabilidades para mercados de Over/Under.

---

## 📦 Instalação

1.  **Clone o repositório**:

    ```bash
    git clone https://github.com/seu-usuario/projeto-bet.git
    cd projeto-bet
    ```

2.  **Instale as dependências**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure o Driver**:
    - Certifique-se de ter o Google Chrome instalado.
    - O `webdriver-manager` gerencia o driver automaticamente.

---

## 🎮 Como Usar

Execute o menu principal:

```bash
python src/main.py
```

### Opções do Menu:

1.  **Atualizar Campeonato**: Baixa dados recentes do Brasileirão (ou outras ligas).
2.  **Treinar Modelo de IA**:
    - Escolha a **Opção 2 (Profissional V2)** para usar a nova arquitetura.
3.  **Analisar Jogo (URL)**: Cole o link de uma partida do SofaScore para receber previsões.
4.  **Consultar Análise (ID)**: Vê detalhes de uma análise já feita.
5.  **Atualizar Liga Específica**: Baixa histórico de 3 anos de ligas europeias.

---

## 📊 Métricas e Performance

O modelo é avaliado não apenas por erro estatístico (MAE), mas por **lucratividade**:

- **Win Rate**: Taxa de acerto das apostas sugeridas.
- **ROI (Return on Investment)**: Retorno financeiro sobre o capital investido.
- **EV (Expected Value)**: O modelo só sugere apostas onde a probabilidade calculada supera a probabilidade implícita na odd.

---

## 📝 Estrutura de Pastas

- `src/ml/`: Núcleo de Machine Learning (`features_v2.py`, `model_v2.py`).
- `src/analysis/`: Motor estatístico (`statistical.py`).
- `src/scrapers/`: Robôs de coleta de dados.
- `src/database/`: Gerenciamento do SQLite.
- `src/web/`: Interface Web (Flask).

---

> **Aviso**: Apostas esportivas envolvem risco financeiro. Este software é uma ferramenta de apoio à decisão e não garante lucros.
