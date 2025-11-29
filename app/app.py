import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import onnxruntime
import boto3

# --- CONFIGURACIÓN DESDE VARIABLES DE ENTORNO ---
S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODEL_KEY = os.getenv("S3_MODEL_PATH")
S3_LABELS_KEY = os.getenv("S3_LABELS_PATH", "imagenet_classes.txt")  # opcional

MODEL_PATH = "app/mobilenetv2-7.onnx"
LABELS_PATH = "app/imagenet_classes.txt"

# --- FUNCIONES DE DESCARGA DESDE S3 ---
def download_file_if_missing(bucket: str, key: str, local_path: str):
    if os.path.exists(local_path):
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, key, local_path)
        print(f"Modelo descargado desde S3: s3://{bucket}/{key}")
    except Exception as e:
        print(f"Error descargando desde S3: {e}")
        raise

# Descargar modelo y clases si no existen
download_file_if_missing(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
download_file_if_missing(S3_BUCKET, S3_LABELS_KEY, LABELS_PATH)

# --- CARGAR ETIQUETAS ---
LABELS = []
try:
    with open(LABELS_PATH, "r") as f:
        LABELS = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"No se pudo cargar archivo de clases: {e}")

# --- CARGAR MODELO ONNX ---
ort_session = onnxruntime.InferenceSession(MODEL_PATH)
input_name = ort_session.get_inputs()[0].name

# --- FLASK ---
app = Flask(__name__)
INPUT_SIZE = (224, 224)

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
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    tensor = preprocess(file.read())
    output = ort_session.run(None, {input_name: tensor})[0]

    idx = int(np.argmax(output))
    label = LABELS[idx] if LABELS else "unknown"

    return jsonify({"predicted_label": label, "index": idx})

if __name__ == "__main__":
    # Solo para debug local
    app.run(host="0.0.0.0", port=8080)
