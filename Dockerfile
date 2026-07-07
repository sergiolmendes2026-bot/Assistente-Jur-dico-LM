FROM python:3.10-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copia e instala requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

# Expõe a porta do Streamlit
EXPOSE 8501

# Comando para rodar
CMD ["streamlit", "run", "dsa_app_com_rag.py", "--server.port=8501", "--server.address=0.0.0.0"]
