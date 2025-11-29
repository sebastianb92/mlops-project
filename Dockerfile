# 1. Imagen base ligera
FROM python:3.10-slim

# 2. Directorio de trabajo
WORKDIR /app

# 3. Copiar requirements
COPY requirements.txt .

# 4. Instalar dependencias + gunicorn
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn boto3 onnxruntime pillow flask

# 5. Copiar aplicación
COPY app/ app/
COPY templates/ templates/

# 6. Exponer puerto
EXPOSE 8080

# 7. Comando para producción usando gunicorn
ENV ENVIRONMENT=dev
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
