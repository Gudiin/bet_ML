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

Aqui é onde o computador "aprende".

### O Modelo: Random Forest (Floresta Aleatória)

Imagine que você quer saber se um filme é bom. Você pergunta para um amigo, e ele diz "Sim". Mas ele pode ter um gosto estranho.
Agora, imagine que você pergunta para **100 amigos diferentes**. Se 80 disserem "Sim", você tem muito mais certeza.

O **Random Forest** funciona assim. Ele cria 100 "Árvores de Decisão" (os amigos).

- Uma árvore olha só para "Chutes no Gol".
- Outra olha para "Posse de Bola".
- Outra olha para "Ataques Perigosos".

No final, o modelo faz uma votação. A média das opiniões dessas 100 árvores é a nossa previsão final.

### O que ele aprendeu?

O modelo analisou milhares de jogos e descobriu correlações matemáticas. Por exemplo:

- **Alta correlação**: Muitos chutes ao gol geralmente resultam em muitos escanteios (o goleiro espalma pra fora).
- **Baixa correlação**: Posse de bola no meio de campo não gera tantos escanteios.

---

## 4. Análise Estatística (Monte Carlo) 🎲

A IA nos dá um número (ex: "Vai ter 10.5 escanteios"). Mas futebol é caótico. E se der zebra?
Para lidar com a sorte (aleatoriedade), usamos o **Método de Monte Carlo**.

### Como funciona?

Imagine que temos uma máquina do tempo.

1.  Pegamos as estatísticas de ataque do Time A e defesa do Time B.
2.  Simulamos a partida virtualmente.
3.  Repetimos isso **10.000 vezes**.

### O Resultado

Desses 10.000 jogos simulados:

- Em 2.000 jogos, saíram 8 escanteios.
- Em 5.000 jogos, saíram 10 escanteios.
- Em 3.000 jogos, saíram 12 escanteios.

Isso cria uma **Curva de Probabilidade**. Podemos dizer: _"Existe 80% de chance de sair mais de 9 escanteios, porque isso aconteceu em 80% das nossas simulações"_.

---

## 5. O Filtro de Alinhamento (Directional Filter) ⚖️

Para garantir segurança, unimos o melhor dos dois mundos: a IA e a Estatística.

- A **IA** olha o cenário macro (O jogo vai ser movimentado?).
- A **Estatística** olha as linhas específicas (Over 9.5, Over 10.5).

**A Regra de Ouro:**

- Se a IA diz "Vai ser um jogo de MUITOS escanteios" (> 10.5), o sistema **proíbe** a gente de apostar em "Poucos escanteios" (Under).
- Se a IA diz "Vai ser um jogo PARADO" (< 9.5), o sistema **proíbe** apostar em "Muitos escanteios" (Over).

Isso evita que a gente vá contra a tendência óbvia do jogo.

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

---

## Resumo da Ópera

1.  **Coletamos** o passado.
2.  **Limpamos** a sujeira.
3.  **A IA prevê** o futuro baseada em padrões.
4.  **Monte Carlo simula** os riscos.
5.  **Filtramos** as loucuras.
6.  **Calculamos** o preço justo.
7.  **Encontramos** o lucro.
