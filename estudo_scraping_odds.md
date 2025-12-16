ESTUDO AVANÇADO: WEBSCRAPING DE ODDS (Foco Bet365 & Casas Asiáticas)
======================================================================
*Material compilado para fins de estudo e pesquisa acadêmica (TCC).*

1. INTRODUÇÃO: O CENÁRIO "GATO E RATO"
--------------------------------------
Casas de aposta como a Bet365 investem milhões em segurança. Elas usam serviços como **Akamai** ou **Cloudflare** para detectar robôs. Se você tentar um `requests.get(url)` simples, será bloqueado imediatamente (Erro 403).

Para ter sucesso, é preciso simular um comportamento 100% humano ou engenharia reversa profunda.

---

2. NÍVEL 1: AUTOMAÇÃO DE BROWSER (STEALTH)
------------------------------------------
O método mais acessível, mas lento e pesado. A ideia é controlar um Chrome real, mas escondendo que é um robô.

### As Ferramentas Certas
**NÃO USE:** Selenium padrão (`selenium webdriver`). Ele vaza variáveis como `navigator.webdriver = true` que gritam "SOU ROBÔ".

**USE:**
1.  **SeleniumBase (Recomendado):** Uma biblioteca Python moderna construída sobre o Selenium mas com modo "UC" (Undetected Chromedriver) nativo.
    *   *Comando:* `Driver(uc=True)`
    *   Ele baixa automaticamente drivers que passam nos testes da Cloudflare.
2.  **Playwright + playwright-stealth:** Mais rápido que Selenium. O plugin `stealth` altera o `navigator` e `user-agent` para parecer um usuário real.

### Técnicas de Evasão (Anti-Bot)
*   **Mouse Humano:** Robôs clicam instantaneamente nas coordenadas exatas. Humanos fazem curvas e aceleram/desaceleram. Use bibliotecas como `pyautogui` ou funções de "human mouse movement".
*   **TLS Fingerprinting:** A Bet365 olha como seu navegador faz o "aperto de mão" SSL (JA3 Fingerprint). O Python padrão tem uma digital diferente do Chrome. Ferramentas como `curl_cffi` podem simular a digital do Chrome.

---

3. NÍVEL 2: ENGENHARIA REVERSA DE WEBSOCKETS (O "GRAL")
-------------------------------------------------------
A Bet365 não carrega odds via HTML a cada segundo. Ela abre um túnel **WebSocket** (`wss://...`) e envia dados binários ou criptografados em tempo real. E interceptar isso é o método profissional.

### Como Funciona:
1.  **O Handshake:** O site envia um token de sessão (gerado por um Javascript ofuscado) para iniciar a conexão.
2.  **O Protocolo:** Os dados vêm compactados. A Bet365 usa um formato próprio (muitas vezes parecendo lixo visual como `F|Hg^...`).
3.  **O Desafio:** Você precisa descobrir como esse Javascript gera o token.

### Ferramentas para Estudo:
*   **MitMProxy / Burp Suite:** Permitem interceptar o tráfego do seu celular/PC e ver os dados brutos do WebSocket.
*   **DevTools:** Aba "Network" -> Filtro "WS" (WebSockets). Olhe as mensagens "Frames".

---

4. NÍVEL 3: ESTRATÉGIA VISUAL (OCR) - "OLHAR HUMANO"
---------------------------------------------------
Você sugeriu: *"Tirar print e converter para JSON"*. Sim, isso é totalmente possível e é a **tendência do futuro** para burlar anti-bots agressivos.

### Por que funciona?
Os sites conseguem embaralhar o código HTML (mudando nomes de classes `div class="x7z_a"` a cada segundo), mas eles **não podem embaralhar o visual**, senão o usuário humano não conseguiria ler. Se o humano vê, o robô vê.

### O Fluxo "Visual":
1.  **Print:** O Selenium/Playwright tira um screenshot apenas do elemento da tabela de odds (`element.screenshot()`).
2.  **OCR (Reconhecimento de Texto):** Uma IA lê a imagem.
    *   *Opção Grátis:* **Tesseract OCR** (Google). É bom para números, mas exige tratamento da imagem (preto e branco) antes.
    *   *Opção Moderna:* **EasyOCR** (Lib Python poderosa).
    *   *Opção "Nuclear" (A sua escolha):* **Google Gemini 1.5 Flash**.
    
### 💎 Usando seu Google Gemini Pago:
Você perguntou se o seu Gemini serve. **SIM, e é a melhor opção atual.**
*   **Por que?** O modelo `gemini-1.5-flash` é extremamente rápido, barato e tem visão nativa. Ele é mais barato que o GPT-4o e ideal para ler milhares de prints.
*   **Como conectar:**
    1.  Não use o chat do site (gemini.google.com) para automação.
    2.  Use o **Google AI Studio** para pegar sua API Key.
    3.  No Python, instale: `pip install google-generativeai`.
    
    ```python
    import google.generativeai as genai
    from PIL import Image

    # Sua chave API
    genai.configure(api_key="SUA_KEY_AQUI")

    # O "Flash" é mais rápido e barato para tarefas simples como ler tabelas
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Envia o print e pede o JSON
    sample_file = Image.open('print_odds.png')
    response = model.generate_content(["Extraia as odds desta tabela em JSON", sample_file])
    print(response.text) # Retorna o JSON prontinho
    ```

### Exemplo de Prompt para IA de Visão:
> "Esta imagem contém uma tabela de apostas. Identifique as colunas 'Over/Under' e 'Odds'. Retorne APENAS um JSON puro neste formato: `[{'market': 'Over 2.5', 'odd': 1.95}, ...]`."

---

5. INFRAESTRUTURA: COMO NÃO SER BANIDO
--------------------------------------
Se você fizer 1000 requisições do seu IP de casa, será banido.

### Proxies Residenciais (Essencial)
*   **Datacenter IPs (AWS, DigitalOcean):** Bloqueados automaticamente.
*   **Residential IPs:** São IPs de casas reais (lícitos). Serviços como BrightData ou Smartproxy vendem acesso. Para a Bet365, é obrigatório.

---

5. ARQUITETURA DE UM TCC (SUGESTÃO)
-----------------------------------
Se for transformar isso em um trabalho acadêmico (Engenharia de Software/Ciência da Computação):

**Tema:** "Arquitetura Distribuída para Coleta e Análise de Dados em Mercados de Alta Frequência"

**Capítulos Sugeridos:**
1.  **Revisão Bibliográfica:** Técnicas de Web Crawling, Ética de Scraping, Protocolos HTTP/WebSocket.
2.  **Engenharia Reversa:** Análise do tráfego de rede da Bet365 (sem expor segredos comerciais, focar na técnica).
3.  **Pipeline de Dados (ETL):**
    *   *Extraction:* SeleniumBase rotacionando Proxies.
    *   *Transformation:* Pandas para limpar nomes de times ("Man Utd" -> "Manchester United").
    *   *Loading:* Salvar em MongoDB (dados não estruturados) ou TimescaleDB (série temporal).
4.  **Estudo de Caso:** Comparação de latência entre coleta via HTML vs WebSocket.

---

6. LISTA DE RECURSOS (PARA PESQUISAR)
-------------------------------------
*   **Libs Python:** `SeleniumBase`, `playwright`, `Scrapy` (para sites simples), `websockets`.
*   **Ferramentas:** `Burp Suite Community`, `Postman`.
*   **Conceitos Chave:** JA3 Fingerprint, Canvas Fingerprinting, TCP/IP Headers.

7. ONDE PESQUISAR (FONTE DE CONHECIMENTO)
------------------------------------------

### Fóruns & Comunidades (Prática)
Para ver "como fazer no mundo real":
*   **BlackHatWorld (Seção Programming):** O melhor lugar para ver discussões sobre bypass de bot protection.
*   **Reddit (r/webscraping & r/algotrading):** Discussões técnicas de alto nível.
*   **StackOverlow:** Para dúvidas específicas de código (mas evite perguntar "como hackear bet365", pergunte "como lidar com websocket opaco").

### Fontes Acadêmicas (Teoria/TCC)
Para citar no seu trabalho:
*   **Google Scholar:** Busque por "Sports Betting Market Efficiency", "Arbitrage Betting Algorithms".
*   **arXiv.org:** Artigos de Ciência da Computação sobre "Web Scraping Anti-Bot Techniques".
*   **Repositórios de TCCs:** Busque nos repositórios da USP, Unicamp ou Federais termos como "Coleta de dados distribuída".

> **⚠️ Aviso Legal:** A raspagem de dados pode violar os Termos de Uso (ToS) das casas. Para fins acadêmicos/pessoais costuma ser "zona cinzenta", mas para fins comerciais é arriscado. O método mais seguro e ético é usar APIs pagas (ex: The Odds API) que já fazem esse trabalho sujo para você.
