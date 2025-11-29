import boto3
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import pytest

# Configuración de S3
BUCKET_NAME = "mlops-project-deploy-bucket"
TEST_FOLDER = "test_data/"  # Carpeta donde están las imágenes
MODEL_PATH = "models/mobilenetv2-7.onnx"

# Inicializar cliente S3
s3 = boto3.client("s3")

# Descargar modelo desde S3
s3.download_file(BUCKET_NAME, MODEL_PATH, "temp_model.onnx")

# Inicializar ONNX Runtime
session = ort.InferenceSession("temp_model.onnx")

# Función para preprocesar imagen
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224)).convert("RGB")
    np_image = np.array(image).astype("float32") / 255.0
    np_image = np_image.transpose(2, 0, 1)  # channels first
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

# Función para listar todas las imágenes en la carpeta de test
def get_test_images():
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=TEST_FOLDER)
    return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].lower().endswith(('.jpg', '.png', '.jpeg'))]

# Parametrizar test con pytest
@pytest.mark.parametrize("img_name", get_test_images())
def test_model_with_image(img_name):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=img_name)
    image_bytes = obj["Body"].read()
    input_tensor = preprocess_image(image_bytes)
    outputs = session.run(None, {"input": input_tensor})
    assert outputs is not None, f"Modelo no devolvió resultados para {img_name}"
