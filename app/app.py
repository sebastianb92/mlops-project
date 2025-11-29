import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import onnxruntime
import boto3

# -------------------------
# Configuración desde entorno
# -------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "mlops-project-bucket")
S3_MODEL_KEY = os.getenv("S3_MODEL_PATH", "models/mobilenetv2-7.onnx")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

LOCAL_MODEL_DIR = "app"
MODEL_FILENAME = os.path.basename(S3_MODEL_KEY)
MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, MODEL_FILENAME)
LABELS_FILENAME = "imagenet_classes.txt"
LABELS_PATH = os.path.join(LOCAL_MODEL_DIR, LABELS_FILENAME)

INPUT_SIZE = (224, 224)

# -------------------------
# Funciones de descarga desde S3
# -------------------------
def download_from_s3(bucket: str, key: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        s3.download_file(bucket, key, local_path)
        print(f"Archivo descargado: s3://{bucket}/{key}")
    except Exception as e:
        raise RuntimeError(f"Error descargando {key} desde S3: {e}")

# Descargar modelo si no existe
if not os.path.exists(MODEL_PATH):
    download_from_s3(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)

# Descargar etiquetas desde S3 si no existen
S3_LABELS_KEY = os.path.join(os.path.dirname(S3_MODEL_KEY), LABELS_FILENAME)
if not os.path.exists(LABELS_PATH):
    download_from_s3(S3_BUCKET, S3_LABELS_KEY, LABELS_PATH)

# -------------------------
# Cargar etiquetas
# -------------------------
try:
    with open(LABELS_PATH, 'r') as f:
        LABELS = [line.strip() for line in f]
except FileNotFoundError:
    LABELS = []
    print("Archivo de etiquetas no encontrado.")

# -------------------------
# Cargar modelo ONNX
# -------------------------
try:
    ort_session = onnxruntime.InferenceSession(MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo ONNX: {e}")

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB").resize(INPUT_SIZE)
    data = np.asarray(img, dtype=np.float32) / 255.0
    data = data.transpose([2, 0, 1])
    return np.expand_dims(data, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No se subió ningún archivo"}), 400

    try:
        tensor = preprocess(file.read())
        output = ort_session.run(None, {input_name: tensor})[0]
        idx = int(np.argmax(output))
        label = LABELS[idx] if LABELS and idx < len(LABELS) else "unknown"
        return jsonify({"predicted_label": label, "index": idx})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
