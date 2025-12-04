# 🧠 Manual Técnico: A Inteligência por Trás do Sistema

> **"O segredo não é a mágica, é saber como o truque é feito."**

Se no `README.md` explicamos _o que_ o sistema faz, aqui vamos explicar **como** ele faz. Vamos abrir o capô e entender as engrenagens de Dados, Estatística e Inteligência Artificial.

---

## 1. Coleta de Dados (Data Collection) 🕵️‍♂️

Tudo começa com dados. Sem dados, não há inteligência.

### Onde buscamos?

Usamos o **SofaScore**. É um dos maiores sites de estatísticas esportivas do mundo. Escolhemos ele porque fornece dados detalhados que outros sites não têm, como "Ataques Perigosos" e "Chutes Bloqueados".

### Como buscamos? (Web Scraping)

Não existe um botão "Baixar Dados" no site. Então, criamos um robô (script Python) que finge ser um usuário navegando.

- **Ferramenta**: Usamos o `Playwright`. Ele abre um navegador invisível (headless), clica nos jogos e copia os números.
- **Desafio**: O site tenta bloquear robôs. Para evitar isso, nosso robô "descansa" um pouco entre cada clique (rate limiting), agindo como um humano.

---

## 2. Pré-processamento (Data Cleaning) 🧹

Os dados brutos vêm "sujos". O computador não entende "10 escanteios". Ele precisa de números organizados.

### O que fazemos?

1.  **Limpeza**: Removemos jogos cancelados ou sem estatísticas.
2.  **Engenharia de Atributos (Feature Engineering)**: Criamos novas informações a partir das básicas.
    - _Exemplo_: O site diz que o Time A teve 5 escanteios e o Time B teve 3. Nós calculamos a **Média Móvel** dos últimos 5 jogos.
    - **Por que Média Móvel?** Porque o desempenho recente importa mais do que o desempenho de 3 meses atrás. Um time pode ter melhorado ou piorado.

---

## 3. Inteligência Artificial (Machine Learning) 🤖

Aqui é onde o computador "aprende". Utilizamos uma abordagem de **Ensemble** (união de forças).

### Os Modelos

Em vez de confiar em apenas um "especialista", usamos três:

1.  **LightGBM** (Principal): Extremamente rápido e preciso para dados tabulares.
2.  **XGBoost**: Robusto e excelente para capturar relações não-lineares.
3.  **Random Forest**: O clássico, bom para evitar overfitting.

O sistema faz uma "votação ponderada" entre eles para chegar ao número final.

### Validação Temporal (O Segredo do Sucesso) ⏳

Muitos iniciantes cometem o erro de misturar jogos de 2024 no treino e testar com jogos de 2023. Isso é **roubar**, pois você está usando o futuro para prever o passado.

Nós usamos **TimeSeriesSplit** (Cross-Validation Temporal):

- Treinamos com Jan-Fev -> Testamos em Março.
- Treinamos com Jan-Mar -> Testamos em Abril.
- Treinamos com Jan-Abr -> Testamos em Maio.

Isso simula o mundo real: a IA só sabe o que aconteceu _antes_ do jogo que ela está tentando prever. Além disso, nossas features usem janelas deslizantes (`shift(1)`) para garantir matematicamente que nenhum dado do jogo atual vaze para o treinamento.

O modelo final é treinado com **todos** os dados disponíveis, mas sua performance reportada é a média desses testes no tempo.

---

## 4. Análise Estatística (O Motor Matemático) 🎲

A IA nos dá um número (ex: "Vai ter 10.5 escanteios"). Mas futebol é caótico. Para modelar esse caos, usamos Distribuições de Probabilidade.

### Poisson vs. Binomial Negativa

O sistema é inteligente o suficiente para escolher qual matemática usar:

1.  **Poisson**: Usada quando o time é consistente (Média ≈ Variância). É o padrão para contagem de gols/escanteios.
2.  **Binomial Negativa**: Usada quando o time é "louco" (Variância > Média). Se um time faz 2 escanteios num jogo e 15 no outro, a Poisson falha. A Binomial Negativa captura essa **Overdispersion** (dispersão exagerada) e ajusta o risco.

### Simulação de Monte Carlo

Com a distribuição escolhida, ligamos a "máquina do tempo":

1.  Pegamos a média prevista (ajustada pela IA).
2.  Simulamos a partida virtualmente **10.000 vezes**.
3.  Contamos quantas vezes cada resultado aconteceu.

Isso cria uma **Curva de Probabilidade Real** que considera tanto a habilidade do time quanto a sorte.

---

## 5. O "Aperto de Mão" (Integração IA + Estatística) 🤝

Aqui está a mágica de como os cálculos "conversam entre si". Não usamos a IA sozinha, nem a Estatística sozinha.

### O Fluxo da Verdade:

1.  **IA Propõe**: "Acho que teremos 11.0 escanteios baseados na tática dos times."
2.  **Clamper (O Juiz) Verifica**:
    - O sistema olha a média histórica (ex: 9.0).
    - Calcula o limite aceitável (ex: ±30% = 6.3 a 11.7).
    - Se a IA dissesse 15.0, o Clamper reduziria para 11.7.
    - _Isso impede que um erro da IA quebre a banca._
3.  **Estatística Executa**:
    - O valor validado (11.0) vira o parâmetro `lambda` da distribuição de Poisson/Binomial.
    - As 10.000 simulações são rodadas usando esse novo centro de gravidade.

**Resultado**: Temos a precisão tática da IA, mas com a segurança matemática e as margens de erro da Estatística. Se a IA estiver otimista demais, o Clamper segura. Se a Estatística for conservadora demais, a IA puxa para cima. É o equilíbrio perfeito.

---

## 6. Geração de Saídas (Odds e Probabilidades) 📊

Finalmente, transformamos isso em dinheiro (ou potencial de).

### Probabilidade Real vs. Odd Justa

- **Probabilidade Real**: É a chance que calculamos (ex: 50% ou 0.50).
- **Odd Justa**: É o inverso da probabilidade.
  $$ Odd = \frac{1}{Probabilidade} $$
  - Se a chance é 50% (0.50), a Odd Justa é $1 / 0.50 = 2.00$.

### Value Bet (Aposta de Valor)

Comparamos a nossa **Odd Justa** com a **Odd da Casa de Apostas**.

- Nossa Odd Justa: **1.50** (Achamos que é muito provável).
- Odd da Bet365: **2.00** (Eles acham que é difícil).

Isso é uma **Value Bet**! Estamos comprando uma nota de 100 reais pagando 50. A longo prazo, a matemática garante o lucro.

Isso é uma **Value Bet**! Estamos comprando uma nota de 100 reais pagando 50. A longo prazo, a matemática garante o lucro.

---

## 7. Scanner de Oportunidades (Automação em Lote) 🚀

O **Scanner** é a evolução do sistema. Em vez de analisar um jogo por vez, ele analisa o dia inteiro.

### Como funciona?

1.  **Busca em Lote**: O Scraper vai ao calendário do SofaScore e baixa a lista de todos os jogos do dia (ex: 50 jogos).
2.  **Filtro de Ligas**: Ignoramos ligas obscuras (ex: 3ª divisão do Vietnã) para focar onde temos dados confiáveis.
3.  **Processamento Paralelo (Simulado)**: O sistema itera sobre cada jogo, aplica o modelo de IA e calcula a confiança.
4.  **Ranking de Oportunidades**:
    - Se a confiança da IA for **< 70%**, o jogo é descartado.
    - Se for **> 70%**, entra no relatório.
    - O relatório é ordenado: as melhores oportunidades aparecem no topo.

Isso transforma o sistema de uma ferramenta passiva ("O que você acha desse jogo?") em uma ferramenta ativa ("Quais são os melhores jogos de hoje?").

---

## Resumo da Ópera

1.  **Coletamos** o passado.
2.  **Limpamos** a sujeira.
3.  **A IA prevê** o futuro baseada em padrões.
4.  **Monte Carlo simula** os riscos.
5.  **Filtramos** as loucuras.
6.  **Calculamos** o preço justo.
7.  **Encontramos** o lucro.
