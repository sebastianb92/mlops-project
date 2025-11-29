import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import onnxruntime
import boto3

# Config variables
S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODEL_PATH = os.getenv("S3_MODEL_PATH")
S3_LABELS_PATH = os.getenv("S3_LABELS_PATH")
S3_PREDICTIONS_PATH = os.getenv("S3_PREDICTIONS_PATH", "predictions/preds.txt")

MODEL_LOCAL_PATH = "app/mobilenetv2-7.onnx"
LABELS_LOCAL_PATH = "app/imagenet_classes.txt"


def safe_download(s3_path, local_path):
    """Download a file from S3 only if the path exists."""
    if not s3_path:
        print(f"[WARN] No S3 path provided for {local_path}, skipping download.")
        return

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3 = boto3.client("s3")
        print(f"[INFO] Downloading {s3_path} to {local_path}")
        s3.download_file(S3_BUCKET, s3_path, local_path)
    except Exception as e:
        print(f"[ERROR] Could not download {s3_path}: {e}")


# Ensure model + labels exist
if not os.path.exists(MODEL_LOCAL_PATH):
    safe_download(S3_MODEL_PATH, MODEL_LOCAL_PATH)

if not os.path.exists(LABELS_LOCAL_PATH):
    safe_download(S3_LABELS_PATH, LABELS_LOCAL_PATH)

# Load labels
LABELS = []
try:
    with open(LABELS_LOCAL_PATH, "r") as f:
        LABELS = [line.strip() for line in f]
except Exception as e:
    print(f"[ERROR] Could not load label file: {e}")
    LABELS = ["unknown"]

# Load ONNX model
ort_session = onnxruntime.InferenceSession(MODEL_LOCAL_PATH)
input_name = ort_session.get_inputs()[0].name

app = Flask(__name__)
INPUT_SIZE = (224, 224)


def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(INPUT_SIZE)
    data = np.asarray(img, dtype=np.float32) / 255.0
    data = data.transpose([2, 0, 1])
    return np.expand_dims(data, axis=0)


def save_prediction_to_s3(pred_str):
    """Append predictions to a file in S3."""
    try:
        s3 = boto3.client("s3")
        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key=S3_PREDICTIONS_PATH)
            existing = response["Body"].read().decode("utf-8")
        except s3.exceptions.NoSuchKey:
            existing = ""

        updated = existing + pred_str + "\n"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=S3_PREDICTIONS_PATH,
            Body=updated.encode("utf-8")
        )
    except Exception as e:
        print(f"[ERROR] Could not save prediction to S3: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    tensor = preprocess(file.read())

    output = ort_session.run(None, {input_name: tensor})[0]
    idx = int(np.argmax(output))
    label = LABELS[idx] if idx < len(LABELS) else "unknown"

    save_prediction_to_s3(f"{label},{idx}")

    return jsonify({"predicted_label": label, "index": idx})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
