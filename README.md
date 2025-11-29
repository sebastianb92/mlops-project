# MLOps Project Deployment

## Descripción

Este proyecto implementa un modelo de clasificación de imágenes con **MobileNetV2** en formato ONNX. El modelo se despliega en AWS ECS con Docker y se gestiona con **GitHub Actions** para un pipeline **CI/CD** completo para entornos `dev` y `prod`.

Incluye pruebas unitarias automáticas con **pytest** y pruebas de inferencia con imágenes de prueba almacenadas en **S3**.

---

## Estructura del proyecto

```
mlops-project/
├── .github/
│ └── workflows/
│ ├── ci_cd_dev.yaml # Pipeline para el entorno de desarrollo
│ └── ci_cd_prod.yaml # Pipeline para el entorno de producción
├── app/
│ └── app.py # Script principal de la aplicación
├── templates/
│ └── index.html # Plantilla HTML usada por la aplicación
├── tests/
│ └── test_model.py # Pruebas unitarias y de integración
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## Requisitos

* Python 3.10
* Docker
* AWS CLI configurado
* GitHub con Secrets para `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY`
* Bucket S3 con:

  * Modelo: `models/mobilenetv2-7.onnx`
  * Imágenes de prueba: `test_data/`
* AWS ECS:

  * Cluster dev: `mlops-deploy-cluster`
  * Cluster prod: `mlops-prod-cluster` o mismo cluster dev
  * Servicios dev/prod: `mlops-deploy-dev` / `mlops-deploy-prod`

---

## Configuración de GitHub Actions

### Dev (`.github/workflows/ci_cd_dev.yaml`)

* Se ejecuta con push a la rama `dev`.
* Descarga imágenes de prueba de S3 y ejecuta tests.
* Construye y publica la imagen Docker en ECR con tag `dev-latest`.
* Actualiza o crea el servicio ECS correspondiente.
* Variables de entorno:

```yaml
AWS_REGION: us-east-1
ECR_REPOSITORY: mlops-deploy-repository
ECR_REGISTRY: 804923754854.dkr.ecr.us-east-1.amazonaws.com
IMAGE_TAG: dev-latest
CLUSTER_NAME: mlops-deploy-cluster
SERVICE_NAME: mlops-dev-service
TASK_DEFINITION: mlops-deploy-dev:4
SUBNETS: subnet-067b18d21d7a147a8
S3_BUCKET: mlops-project-deploy-bucket
S3_TEST_PATH: test_data/
```

### Prod (`.github/workflows/ci_cd_prod.yaml`)

* Se ejecuta con push a la rama `prod` o manual (`workflow_dispatch`).
* Igual que dev pero con tag `prod-latest` y servicio `mlops-prod-service`.

---

## Dockerfile

```dockerfile
FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY templates/ templates/

ENV ENVIRONMENT=dev
EXPOSE 8080
CMD ["python", "app/app.py"]
```

* `templates/` usada es la de la raíz del proyecto.
* `app/app.py` es el entrypoint principal.
* Cambiar `ENVIRONMENT` según dev o prod.

---

## Pruebas

* Se descargan imágenes desde S3 en `test_data/`.
* Se ejecuta `pytest tests/` para validar que el modelo produce inferencias correctamente.
* Test de ejemplo (`tests/test_model.py`):

```python
def test_model_with_image(img_name):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=img_name)
    image_bytes = obj["Body"].read()
    input_tensor = preprocess_image(image_bytes)
    outputs = session.run(None, {"data": input_tensor})  # data es el input del ONNX
    assert outputs is not None
```

---

## Despliegue

1. Push a `dev` para pruebas en ECS dev.
2. Push a `prod` para producción.
3. GitHub Actions:

   * Corre tests
   * Construye imagen Docker
   * Publica a ECR
   * Actualiza ECS Service (o crea si no existe)
4. Para asegurar que siempre se usen cambios recientes en el index, Docker build se hace sin cache:

```bash
docker build --no-cache -t $DOCKER_IMAGE_TAG .
```


## Endpoints

A continuación se listan los endpoints públicos para acceder a la aplicación desplegada en cada entorno:

### Desarrollo (DEV)
- **URL:** http://13.220.23.40:8080  

### Producción (PROD)
- **URL:** http://34.201.23.88:8080  

---

## Tips

* Para ver qué `index` está usando la app, revisar `CMD` en Dockerfile y las rutas de templates.
* El `task_definition` se puede fijar en el workflow o usar la última versión publicada.
* Si se borran imágenes antiguas, GitHub Actions recrea la imagen automáticamente al hacer push.

---

## Contacto

* Autor: Johan Sebastian Bonilla y Jousé Cobaleda