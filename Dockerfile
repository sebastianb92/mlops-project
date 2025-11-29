# 1. Usar una imagen base de Python oficial (ligera)
FROM python:3.10-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar solo los requisitos
COPY requirements.txt .

# 4. Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto de la aplicación y las plantillas
COPY app/ app/
COPY templates/ templates/
COPY imagenet_classes.txt .

# 6. Exponer el puerto de la aplicación
EXPOSE 8080

# 7. Comando para ejecutar la aplicación
ENV ENVIRONMENT=local
CMD ["python", "app/app.py"]
