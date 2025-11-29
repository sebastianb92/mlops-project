FROM python:3.10-slim
WORKDIR /app

# Copiar dependencias y scripts
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar app
COPY app/ app/
COPY templates/ templates/

# Variables de entorno (para usar dentro del contenedor)
ENV ENVIRONMENT=dev

EXPOSE 8080
CMD ["python", "app/app.py"]
