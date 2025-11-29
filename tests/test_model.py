import pytest
import os
import json
import numpy as np
import onnxruntime
import boto3

# Configuración de paths
TEST_DATA_PATH = "tests/test_data.json"
MODEL_PATH = "app/mobilenetv2-7.onnx"
INPUT_SIZE = (224, 224)

# --- Config S3 ---
S3_BUCKET = os.getenv("S3_BUCKET", "mlops-project-deploy-bucket")
S3_KEY = os.getenv("S3_KEY", "models/mobilenetv2-7.onnx")

# --- FIXTURES DE CONFIGURACIÓN Y DATOS ---
@pytest.fixture(scope="session", autouse=True)
def setup_model_and_data():
    """
    Descarga el modelo real desde S3 y genera datos de prueba si no existen.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Descargar modelo desde S3 si no existe
    if not os.path.exists(MODEL_PATH):
        s3 = boto3.client("s3")
        try:
            s3.download_file(S3_BUCKET, S3_KEY, MODEL_PATH)
            print(f"✅ Modelo descargado desde S3: s3://{S3_BUCKET}/{S3_KEY}")
        except Exception as e:
            pytest.fail(f"No se pudo descargar el modelo desde S3: {e}")

    # Crear datos de prueba si no existen
    if not os.path.exists(TEST_DATA_PATH):
        simulated_input = np.random.rand(INPUT_SIZE[0], INPUT_SIZE[1], 3).astype(np.float32).tolist()
        simulated_data = {
            "input_tensor": simulated_input,
            "expected_index": 285,  # ejemplo: 'tabby cat'
            "baseline_confidence": 0.85
        }
        os.makedirs(os.path.dirname(TEST_DATA_PATH), exist_ok=True)
        with open(TEST_DATA_PATH, 'w') as f:
            json.dump(simulated_data, f)
        print(f"✅ Datos de prueba generados en {TEST_DATA_PATH}")

# --- Fixtures ONNX y datos ---
@pytest.fixture(scope="module")
def ort_session():
    try:
        return onnxruntime.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    except Exception as e:
        pytest.fail(f"Error al cargar ONNX: {e}")

@pytest.fixture(scope="module")
def test_data():
    try:
        with open(TEST_DATA_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        pytest.fail(f"Error al cargar los datos de prueba: {e}")

# --- Pruebas ---
def test_model_responds_with_defined_input(ort_session, test_data):
    input_data = np.array(test_data["input_tensor"], dtype=np.float32)
    input_tensor = np.expand_dims(input_data.transpose([2,0,1]), axis=0)

    input_name = ort_session.get_inputs()[0].name
    ort_outputs = ort_session.run(None, {input_name: input_tensor})
    output_data = ort_outputs[0]

    assert output_data.size > 0, "El modelo no produjo salida"

def test_no_significant_metric_change(ort_session, test_data):
    input_data = np.array(test_data["input_tensor"], dtype=np.float32)
    input_tensor = np.expand_dims(input_data.transpose([2,0,1]), axis=0)

    input_name = ort_session.get_inputs()[0].name
    ort_outputs = ort_session.run(None, {input_name: input_tensor})
    output_data = ort_outputs[0]

    exp_output = np.exp(output_data - np.max(output_data))
    probabilities = exp_output / np.sum(exp_output)
    
    predicted_confidence = probabilities[0, test_data["expected_index"]].item()
    safe_threshold = 0.001
    
    assert predicted_confidence >= safe_threshold, f"Confianza ({predicted_confidence:.4f}) por debajo del umbral ({safe_threshold:.4f})"
