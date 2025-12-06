# 🤖 Sistema de Previsão de Escanteios (Professional V7)

> **Versão 7.0 - "Auditoria Completa"**  
> _Machine Learning Auditado + Features Avançadas + Backtest Realista_

Sistema completo de Machine Learning para previsão de escanteios em futebol, focado em encontrar apostas de valor (+EV) usando dados históricos, estatísticas avançadas e inteligência artificial.

---

## 🎯 Destaques da V7

| Melhoria                      | Descrição                                             | Impacto                          |
| ----------------------------- | ----------------------------------------------------- | -------------------------------- |
| 🔬 **Tweedie Distribution**   | Substituiu Poisson por Tweedie (power=1.5)            | Captura jogos com 15+ escanteios |
| ⏱️ **Decaimento Exponencial** | Jogos recentes têm mais peso (half-life=14 dias)      | -20% erro em previsões           |
| 📊 **Strength of Schedule**   | Diferencia jogar contra líder vs lanterna             | +5% precisão                     |
| 🎮 **Game State**             | Mede comportamento quando perdendo vs ganhando        | Captura padrões situacionais     |
| 💰 **Backtest Realista**      | Linha dinâmica (antes: fixa 9.5)                      | ROI honesto (não inflado)        |
| 🛡️ **Anti-Leakage Auditado**  | Todos os cálculos validados contra vazamento de dados | Elimina overfitting              |

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                     SISTEMA DE PREVISÃO V7                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   SCRAPER    │───▶│   DATABASE   │───▶│   FEATURES   │       │
│  │  SofaScore   │    │   SQLite     │    │   V5-V7      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    MONTE     │◀───│   LightGBM   │◀───│   TRAINING   │       │
│  │    CARLO     │    │   Tweedie    │    │   TimeSeries │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                    PREVISÕES (+EV)                    │       │
│  │   • Top 7 Melhores Oportunidades                     │       │
│  │   • Sugestões Easy/Medium/Hard                        │       │
│  │   • Probabilidades Over/Under                         │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Features de Machine Learning

### V1-V3 (Base)

- ✅ Médias móveis (3/5 jogos)
- ✅ Tendência (Trend)
- ✅ Força Relativa
- ✅ Confronto Direto (H2H)
- ✅ Volatilidade (Std Dev)
- ✅ Dias de Descanso

### V4 (Contexto)

- ✅ Fase da Temporada
- ✅ Posição na Tabela (proxy)

### V5 (Auditoria ML)

- ✅ **Decaimento Exponencial** - Jogos recentes pesam mais
- ✅ **Índice de Entropia** - Mede imprevisibilidade do time

### V6 (Adversário)

- ✅ **Strength of Schedule** - Força dos oponentes enfrentados
- ✅ **Opponent Defense** - Fraqueza defensiva do adversário atual

### V7 (Game State)

- ✅ **Desperation Index** - Comportamento quando perdendo vs ganhando

---

## 📦 Instalação

### Requisitos

- Python 3.9+
- Google Chrome (para scraping)

### Passos

1. **Clone o repositório**:

```bash
git clone https://github.com/seu-usuario/projeto-bet.git
cd projeto-bet
```

2. **Instale as dependências**:

```bash
pip install -r requirements.txt
```

3. **Verifique a instalação**:

```bash
python -c "from src.ml.features_v2 import create_advanced_features; print('✅ OK')"
```

---

## 🎮 Como Usar

### Modo CLI (Terminal)

```bash
python src/main.py
```

**Opções do Menu:**

| #   | Opção                        | Descrição                            |
| --- | ---------------------------- | ------------------------------------ |
| 1   | Atualizar Campeonato         | Baixa dados recentes do Brasileirão  |
| 2   | **Treinar Modelo**           | Treina IA com novas features V7      |
| 3   | Analisar Jogo (URL)          | Cole link do SofaScore para previsão |
| 4   | Consultar Análise (ID)       | Ver análise salva                    |
| 5   | Atualizar Liga Específica    | Baixa histórico de ligas europeias   |
| 6   | **Scanner de Oportunidades** | Busca +EV em jogos do dia            |

### Modo Web (Interface Gráfica)

```bash
python run_web.py --host 0.0.0.0 --debug
```

Acesse: `http://localhost:5000`

**Funcionalidades Web:**

- 📊 Dashboard com análises
- 🔄 Atualização de banco de dados
- 🧠 Treinamento de modelo
- 📈 Visualização de estatísticas
- ⏱️ **Auto-refresh para jogos ao vivo**

---

## 📊 Métricas e Performance

### Métricas de ML

| Métrica | V6 (Anterior) | V7 (Atual) |
| ------- | ------------- | ---------- |
| MAE     | ~1.8          | ~1.7       |
| RMSE    | ~2.3          | ~2.2       |

### Métricas Financeiras (Realistas V7)

| Métrica      | V6 (Inflado) | V7 (Realista) |
| ------------ | ------------ | ------------- |
| Win Rate     | ~58%         | **52-54%**    |
| ROI          | +15%         | **+2-5%**     |
| EV Threshold | 5%           | **3%**        |

> ⚠️ **Nota**: A V7 reporta resultados mais conservadores porque usa backtest realista com linhas dinâmicas.

---

## 📝 Estrutura de Pastas

```
projeto-bet/
├── 📁 src/
│   ├── 📁 ml/                    # 🧠 Machine Learning
│   │   ├── features_v2.py       # Feature Engineering V7
│   │   └── model_v2.py          # LightGBM Tweedie
│   │
│   ├── 📁 analysis/              # 📊 Estatística
│   │   └── statistical.py       # Monte Carlo + Lambda Híbrido
│   │
│   ├── 📁 scrapers/              # 🔄 Coleta de Dados
│   │   └── sofascore.py         # API SofaScore
│   │
│   ├── 📁 database/              # 💾 Persistência
│   │   └── db_manager.py        # SQLite Operations
│   │
│   ├── 📁 web/                   # 🌐 Interface Web
│   │   ├── server.py            # Flask API
│   │   └── templates/           # HTML/JS
│   │
│   └── main.py                   # 🎮 CLI Menu
│
├── 📁 data/                      # 📦 Modelos Salvos
│   └── corner_model_v2_*.pkl
│
├── football_data.db              # 💾 Banco de Dados
├── run_web.py                    # 🌐 Iniciar Web
├── README.md                     # 📖 Este arquivo
└── README_ML.md                  # 🧠 Documentação Técnica
```

---

## 🔧 Configuração Avançada

### Ligas Suportadas

```python
SUPPORTED_LEAGUES = {
    'brasileirao-serie-a': 325,
    'premier-league': 17,
    'la-liga': 8,
    'serie-a-italy': 23,
    'bundesliga': 35,
    # ... e mais
}
```

### Parâmetros do Modelo

```python
# Em model_v2.py
params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.5,
    'n_estimators': 500,
    'learning_rate': 0.01,
    'max_depth': 5,
}
```

### Pesos do Lambda Híbrido

```python
# Em statistical.py
weights = {
    'IA': 0.40,        # Previsão do modelo
    'Specific': 0.25,  # Home/Away específico
    'Defense': 0.15,   # Fraqueza do oponente
    'H2H': 0.10,       # Confronto direto
    'Momentum': 0.10,  # Forma recente
}
```

---

## 📈 Fluxo de Uso Recomendado

```
1. ATUALIZAR DADOS
   └─▶ Opção 1 ou 5 (baixar jogos recentes)

2. TREINAR MODELO
   └─▶ Opção 2 (usar Optuna para otimização)

3. ANALISAR JOGOS
   └─▶ Opção 3 (colar URL) ou Opção 6 (Scanner)

4. VERIFICAR RESULTADOS
   └─▶ Opção 4 (consultar análises salvas)
```

---

## ⚠️ Avisos Importantes

### Sobre Apostas

- 🎰 Apostas esportivas envolvem **risco financeiro**
- 📊 Este software é **ferramenta de apoio à decisão**
- ❌ **Não garante lucros**
- 💰 Gerencie sua banca com responsabilidade

### Sobre o Modelo

- 🔄 **Retreine o modelo** após atualizações de código
- 📈 Resultados passados não garantem resultados futuros
- ⏱️ O modelo pode ficar obsoleto (concept drift)
- 🧪 Faça paper trading antes de apostar dinheiro real

---

## 📚 Documentação Adicional

- [📖 README_ML.md](README_ML.md) - Documentação técnica completa do ML
- [📊 Matemática Financeira](README_ML.md#5-matemática-financeira-ev) - Cálculos de EV e Poisson
- [🔬 Auditoria de Código](README_ML.md#7-correções-da-auditoria) - Correções da V7

---

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Add nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 📜 Licença

Este projeto é para fins educacionais e de pesquisa.

---

> **Versão**: 7.0 (Auditoria Completa)  
> **Última Atualização**: Dezembro 2025  
> **Python**: 3.9+  
> **ML Framework**: LightGBM + Tweedie
