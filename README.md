# ⚽ Sistema de Previsão de Escanteios com Inteligência Artificial

> **"Como se fosse a previsão do tempo, mas para o mercado de escanteios no futebol."**

Seja bem-vindo! Se você está começando agora no mundo da programação ou das apostas esportivas, este guia foi feito para você. Aqui explicamos **o que** este projeto faz, **por que** ele existe e **como** ele funciona, tudo de forma simples e direta.

---

## 🧐 O Que é Este Projeto?

Imagine que você quer saber se vai chover amanhã. Você pode:

1.  **Olhar para o céu** (Intuição).
2.  **Consultar a meteorologia**, que usa satélites e computadores para analisar nuvens, vento e temperatura (Ciência de Dados).

Este projeto é a **meteorologia do futebol**.

Em vez de prever chuva, ele prevê **escanteios**. Ele usa dados históricos, estatística e inteligência artificial para responder a uma pergunta simples:

> _"Neste jogo entre Time A e Time B, vão sair muitos ou poucos escanteios?"_

---

## 💰 Qual o Problema que Ele Resolve? (Regra de Negócio)

No mundo das apostas esportivas, as casas de apostas (como a Bet365) definem uma "linha" para cada jogo. Por exemplo: **10.5 escanteios**.

- Se você acha que vai ter **11 ou mais**, você aposta no **Over** (Mais de).
- Se você acha que vai ter **10 ou menos**, você aposta no **Under** (Menos de).

O problema é: **Como saber quem tem razão? Você ou a casa de apostas?**

As casas de apostas são muito boas em definir essas linhas. Para ganhar dinheiro a longo prazo, você precisa encontrar as **"Value Bets"** (Apostas de Valor). Uma Value Bet acontece quando a **sua** chance de ganhar é maior do que o preço (Odd) que a casa está pagando.

**Este sistema serve para encontrar essas agulhas no palheiro.** Ele analisa milhares de dados para calcular a **probabilidade real** de um evento acontecer. Se a nossa probabilidade for maior que a da casa, temos uma oportunidade!

---

## 🏗️ Como Funciona? (Visão Geral)

O sistema funciona como uma fábrica de decisões com 4 departamentos principais que conversam entre si:

### 1. O Olheiro (Coleta de Dados / Scraping) 🕵️‍♂️

- **Função**: Vai até o site do SofaScore e assiste aos replays dos jogos passados.
- **O que anota**: Chutes, ataques perigosos, posse de bola, e claro, escanteios.
- **Resultado**: Um banco de dados gigante com o histórico de cada time.

### 2. O Estudante (Machine Learning / IA) 🧠

- **Função**: Pega esse caderno de anotações e estuda os padrões.
- **O que aprende**: _"Quando o Time A joga em casa e chuta muito, costumam sair 12 escanteios"_.
- **Tecnologia**: Usa algoritmos avançados (LightGBM, XGBoost) para prever o número exato de escanteios do próximo jogo.

### 3. O Matemático (Estatística e Monte Carlo) 🎲

- **Função**: Testa a previsão do Estudante contra a sorte.
- **O que faz**: Simula a partida virtualmente **10.000 vezes** usando distribuições matemáticas (Poisson).
- **Resultado**: Uma probabilidade confiável. _"Em 85% das simulações, saíram mais de 9 escanteios"_.

### 4. O Juiz (Mecanismos de Segurança) ⚖️

- **Função**: Garante que ninguém está alucinando.
- **Regra do Clamper**: Se a IA prever algo muito absurdo (ex: 20 escanteios num jogo que a média é 10), o Juiz bloqueia e ajusta a previsão para um valor realista (máximo 30% de desvio da média).
- **Filtro Direcional**: Se a IA diz "Muitos Escanteios", o sistema proíbe apostar em "Poucos". Isso evita contradições.

### 5. O Consultor (Interface Web) 💻

- **Função**: Junta tudo isso e te mostra numa tela bonita.
- **Entrega**: _"Olha, a IA prevê um jogo movimentado, a estatística confirma com 85% de chance e o risco é baixo. É uma boa aposta!"_

---

## 🚀 Como Usar (Guia Rápido)

### 1. Instalação

Primeiro, precisamos preparar o terreno (instalar as ferramentas). No seu terminal:

```bash
# Instala as bibliotecas necessárias (os "ingredientes" do bolo)
pip install -r requirements.txt

# Instala o navegador que o robô vai usar
playwright install
```

### 2. Coletando Dados

O sistema precisa de dados para aprender. Vamos mandar o robô trabalhar:

```bash
python src/main.py
```

_Escolha a opção **1** para atualizar o banco de dados._

### 3. Treinando a IA

Agora que temos dados, vamos ensinar o cérebro do sistema:

```bash
python src/main.py
```

_Escolha a opção **2** para treinar o modelo._

### 4. Usando o Sistema (Interface Web)

A parte divertida! Vamos ver as previsões:

```bash
python run_web.py
```

_Abra o navegador no endereço que aparecer (geralmente `http://localhost:5000`)._

_Abra o navegador no endereço que aparecer (geralmente `http://localhost:5000`)._

### 5. Scanner de Oportunidades (Automático) 🆕

Quer analisar **todos** os jogos do dia de uma vez?

```bash
python src/main.py
```

_Escolha a opção **7**. O sistema vai buscar todos os jogos, analisar um por um e gerar um relatório com as melhores oportunidades (Confiança > 70%)._

---

## 📂 Onde Está Cada Coisa?

Para você não se perder nos arquivos:

- `src/scrapers/`: Onde mora o **Olheiro** (código que acessa a internet).
- `src/database/`: O **Caderno** (onde salvamos os dados).
- `src/ml/`: O **Estudante** (cérebro da Inteligência Artificial).
- `src/analysis/`: O **Matemático** (cálculos estatísticos e simulações).
- `src/web/`: O **Consultor** (site que você vê).

---

> **Quer saber os detalhes técnicos?**
> Leia o arquivo `README_ML.md` para uma explicação profunda sobre como a mágica acontece por baixo dos panos!
