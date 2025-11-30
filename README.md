# MLOps Project Deployment

## Descripción

Este proyecto implementa un modelo de clasificación de imágenes con **MobileNetV2** en formato ONNX. El modelo se despliega en AWS ECS con Docker y se gestiona con **GitHub Actions** para un pipeline **CI/CD** completo para entornos `dev` y `prod`.

Incluye pruebas unitarias automáticas con **pytest** y pruebas de inferencia con imágenes de prueba almacenadas en **S3**.

---

## Estructura del proyecto

```
mlops-project/
├── .github/
│   └── workflows/
│       ├── ci_cd_dev.yaml
│       └── ci_cd_prod.yaml
│
├── app/                      # Carpeta principal del backend
│   ├── app.py                # Script principal Flask
│   ├── model/                # (Opcional) Carpeta para modelo descargado
│   └── templates/            # Plantillas renderizadas por Flask
│       └── index.html
│
├── tests/
│   └── test_model.py         # Pruebas unitarias
│
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

  * Cluster  `mlops-deploy-cluster`
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
TASK_DEFINITION: mlops-deploy-dev
SUBNETS: subnet-067b18d21d7a147a8
S3_BUCKET: mlops-project-deploy-bucket
S3_TEST_PATH: test_data/
```

### Prod (`.github/workflows/ci_cd_prod.yaml`)

* Se ejecuta con push a la rama `prod` o manual (`workflow_dispatch`).
* Igual que dev pero con tag `prod-latest` y servicio `mlops-prod-service`.



---

## Pruebas

* Se descargan imágenes desde S3 en `test_data/`.
* Se ejecuta `pytest tests/` para validar que el modelo produce inferencias correctamente.
* Test de ejemplo (`tests/test_model.py`):



---

## Despliegue

1. Push a `dev` para pruebas en ECS dev.
2. Push a `prod` para producción.
3. GitHub Actions:

   * Corre tests
   * Construye imagen Docker
   * Publica a ECR
   * Actualiza ECS Service (o crea si no existe)




## Endpoints

A continuación se listan los endpoints públicos para acceder a la aplicación desplegada en cada entorno:

### Desarrollo (DEV)
- **URL:** http://98.80.174.195:8080

### Producción (PROD)
- **URL:** http://54.91.223.74:8080

---



## Contacto

* Autor: Johan Sebastian Bonilla y Jousé Cobaleda