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

O sistema funciona como uma fábrica com 4 departamentos principais:

1.  **O Olheiro (Coleta de Dados / Scraping)** 🕵️‍♂️

    - Vai até o site do SofaScore.
    - Anota tudo sobre os jogos passados: chutes, ataques perigosos, posse de bola, e claro, escanteios.
    - Guarda tudo num caderno organizado (Banco de Dados).

2.  **O Estudante (Machine Learning / IA)** 🧠

    - Pega esse caderno e estuda os padrões.
    - Aprende coisas como: _"Quando o time da casa chuta muito e o visitante defende mal, costumam sair 12 escanteios"_.
    - Faz uma previsão baseada no que aprendeu.

3.  **O Matemático (Simulação de Monte Carlo)** 🎲

    - Pega as estatísticas dos times e "joga" a partida virtualmente **10.000 vezes**.
    - Conta o que aconteceu nessas simulações.
    - _"Em 8.500 das 10.000 simulações, saíram mais de 9 escanteios"_. Logo, a probabilidade é de 85%.

4.  **O Consultor (Interface Web)** 💻
    - Junta tudo isso e te mostra numa tela bonita.
    - Te diz: _"Olha, a IA prevê um jogo movimentado e a estatística diz que tem 85% de chance de dar Over. É uma boa aposta!"_

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
