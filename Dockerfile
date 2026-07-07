# Usa uma imagem base leve do Python
FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de dependência primeiro (melhora o cache)
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Expõe a porta que o Streamlit usa por padrão
EXPOSE 8501

# Comando para rodar a aplicação
CMD ["streamlit", "run", "dsa_app_com_rag.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
