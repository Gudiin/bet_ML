import google.generativeai as genai
from PIL import Image

# Sua chave API
genai.configure(api_key="AIzaSyC7x90aKAZFLX1iH0dlY7TOiknkBtMtOCA")

# O "Flash" é mais rápido e barato para tarefas simples como ler tabelas
model = genai.GenerativeModel('gemini-1.5-flash')

# Envia o print e pede o JSON
sample_file = Image.open('print_odds.png')
response = model.generate_content(["Extraia as odds desta tabela em JSON", sample_file])
print(response.text) # Retorna o JSON prontinho