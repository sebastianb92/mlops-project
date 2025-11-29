import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import onnxruntime
import boto3

# Configuración S3 desde variables de entorno
S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODEL_PATH = os.getenv("S3_MODEL_PATH")
S3_LABELS_PATH = os.getenv("S3_LABELS_PATH")
S3_PREDICTIONS_DEV = os.getenv("S3_PREDICTIONS_DEV")

MODEL_LOCAL_PATH = "app/mobilenetv2-7.onnx"
LABELS_LOCAL_PATH = "app/imagenet_classes.txt"

def download_file_from_s3(s3_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3 = boto3.client("s3")
    try:
        s3.download_file(S3_BUCKET, s3_path, local_path)
    except Exception as e:
        raise RuntimeError(f"Error descargando {s3_path} desde S3: {e}")

# Descargar modelo y clases si no existen
if not os.path.exists(MODEL_LOCAL_PATH):
    download_file_from_s3(S3_MODEL_PATH, MODEL_LOCAL_PATH)

if not os.path.exists(LABELS_LOCAL_PATH):
    download_file_from_s3(S3_LABELS_PATH, LABELS_LOCAL_PATH)

# Cargar clases
with open(LABELS_LOCAL_PATH, "r") as f:
    LABELS = [line.strip() for line in f]

# Cargar modelo ONNX
ort_session = onnxruntime.InferenceSession(MODEL_LOCAL_PATH)
input_name = ort_session.get_inputs()[0].name

app = Flask(__name__)
INPUT_SIZE = (224, 224)

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(INPUT_SIZE)
    data = np.asarray(img, dtype=np.float32) / 255.0
    data = data.transpose([2, 0, 1])
    return np.expand_dims(data, axis=0)

def save_prediction_to_s3(prediction_str):
    s3 = boto3.client("s3")
    try:
        # Descargar archivo existente
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_PREDICTIONS_DEV)
        existing = obj["Body"].read().decode("utf-8")
    except s3.exceptions.NoSuchKey:
        existing = ""
    updated = existing + prediction_str + "\n"
    s3.put_object(Bucket=S3_BUCKET, Key=S3_PREDICTIONS_DEV, Body=updated.encode("utf-8"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    tensor = preprocess(file.read())
    output = ort_session.run(None, {input_name: tensor})[0]
    idx = int(np.argmax(output))
    label = LABELS[idx]
    # Guardar predicción en S3
    save_prediction_to_s3(f"{label},{idx}")
    return jsonify({"predicted_label": label, "index": idx})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
