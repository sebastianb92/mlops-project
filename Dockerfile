FROM python:3.10-slim

# Dependencias para Pillow y ONNX Runtime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requisitos primero (optimiza el build cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la app
COPY app/ ./app/
COPY templates/ ./templates/

# Puerto de Flask
EXPOSE 8080

CMD ["python", "app/app.py"]
