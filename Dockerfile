FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy app
COPY app/ /app/app/
COPY templates/ /app/templates/
COPY requirements.txt /app/

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "app/app.py"]
