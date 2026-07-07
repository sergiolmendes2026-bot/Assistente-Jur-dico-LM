# Usa uma versão oficial e estável do Python
FROM python:3.10-slim

# Define o diretório de trabalho
WORKDIR /app

# Instala dependências do sistema necessárias para compilar algumas bibliotecas de IA
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia o requirements e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

# Expõe a porta do Streamlit
EXPOSE 8501

# Comando de inicialização
CMD ["streamlit", "run", "dsa_app_com_rag.py", "--server.port=8501", "--server.address=0.0.0.0"]
