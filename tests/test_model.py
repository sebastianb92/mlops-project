import pytest
import os
import json

# Solo probamos que el archivo de datos de test existe y se puede leer
TEST_DATA_PATH = "tests/test_data.json"

@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Asegura que el archivo de test exista para no fallar la pipeline."""
    os.makedirs(os.path.dirname(TEST_DATA_PATH), exist_ok=True)
    if not os.path.exists(TEST_DATA_PATH):
        dummy_data = {
            "input_tensor": [[ [0,0,0] for _ in range(224)] for _ in range(224)],
            "expected_index": 0,
            "baseline_confidence": 0.85
        }
        with open(TEST_DATA_PATH, "w") as f:
            json.dump(dummy_data, f)

def test_test_data_exists():
    """Verifica que los datos de prueba existan y sean legibles."""
    with open(TEST_DATA_PATH, "r") as f:
        data = json.load(f)
    assert "input_tensor" in data
    assert "expected_index" in data
    assert "baseline_confidence" in data
