FROM python:3.10-slim

# Install dependencies needed for Pillow + ONNX Runtime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application
COPY app/ /app/app/
COPY templates/ /app/templates/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "app/app.py"]
