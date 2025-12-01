# Proyecto MLOps: Despliegue Automático de un Modelo ONNX con CI/CD

## Objetivo del Repositorio y Sistema Propuesto

Este repositorio demuestra un sistema de MLOps para el despliegue automático de modelos de Machine Learning en formato ONNX usando prácticas de CI/CD. El propósito central es implementar una pipeline continuo que tome un modelo entrenado (exportado a ONNX) y automatice su prueba e implementación en producción. De esta manera, cualquier cambio en el código o en el modelo desencadena procesos automáticos de integración continua (CI) – para asegurar calidad – y de despliegue continuo (CD) – para actualizar el servicio de inferencia sin intervención manual.

En resumen, el sistema propuesto permite que, tras cada actualización en la rama de desarrollo, se ejecuten pruebas automáticamente, y al promover esos cambios a la rama de producción, el modelo se containeriza y despliega en AWS. Esto logra un flujo confiable donde un modelo de ML (en formato ONNX) pasa de la fase de desarrollo a un servicio activo en producción de forma reproducible y controlada. La elección de ONNX (Open Neural Network Exchange) como formato del modelo garantiza portabilidad y rendimiento consistente en distintos entornos, facilitando su ejecución mediante runtimes optimizados en la infraestructura de despliegue.


---

## Estructura del proyecto

```
mlops-project/
├── .github/
│   └── workflows/
│       ├── ci_cd_dev.yaml
│       └── ci_cd_prod.yaml
│
├── app/                     
│   ├── app.py              
│   ├── model/                
│   └── templates/           
│       └── index.html
│
├── tests/
│   └── test_model.py         
│
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt

```

El proyecto está organizado de forma clara para separar el trabajo en desarrollo del código listo para producción. Las principales ramas en el control de versiones Git son:

*	`dev` – Rama de desarrollo: aquí se integran las nuevas funcionalidades, ajustes de modelo y cambios de código. Todas las contribuciones se prueban primero en dev.

*	`prod` – Rama de producción: contiene la versión estable del código y configuraciones que se despliegan al entorno productivo. Solo se actualiza mediante merge desde dev una vez que los cambios han sido validados.

Dentro del repositorio, la estructura de archivos y carpetas destaca los siguientes elementos clave:

*	**Dockerfile** – Archivo de configuración Docker que define la imagen del contenedor para la aplicación de predicción. Incluye las instrucciones para construir un entorno con las dependencias necesarias (por ejemplo, instalación de ONNX Runtime, FastAPI/Streamlit, etc.) y para incorporar el código de la aplicación en la imagen.

*	**main.py** – Script principal de la aplicación de inferencia. Contiene el código para cargar el modelo ONNX y servirlo a través de una interfaz (por ejemplo, inicializando una API con FastAPI, definiendo endpoints de predicción, etc.). Este es el punto de entrada de la aplicación dentro del contenedor.

*	**.github/workflows/** – Directorio con las definiciones de **GitHub Actions** usadas para CI/CD. Aquí se encuentran los flujos de trabajo (archivos YAML) que automatizan las etapas de pruebas y despliegue:

*	Un workflow de **test** (por ejemplo, ci-test.yml o similar) que se ejecuta en cada push a la rama dev. Este workflow contiene los pasos para preparar el entorno, obtener el modelo y datos de prueba, y ejecutar pruebas unitarias.

*	Un workflow de **deploy** (por ejemplo, cd-deploy.yml) que se desencadena cuando hay cambios en la rama prod. Este define los pasos de construcción de la imagen Docker, push al registro de contenedores, y actualización del servicio en AWS.

*	**Archivos de configuración** – Es posible que existan otros archivos de soporte, como un requirements.txt (con las dependencias Python requeridas, e.g. onnxruntime, fastapi, etc.), scripts utilitarios para descargar datos o manejar el modelo, o definiciones de infraestructura (p. ej., un JSON de definición de tarea ECS). Estos ayudan a reproducir el entorno localmente y en la pipeline.

*	**Directorio de pruebas (tests/)** – Si se incluyen tests unitarios o de integración, normalmente residirán aquí. Por ejemplo, podría haber un tests/test_prediction.py que verifica que el modelo ONNX cargue correctamente y que la función de predicción devuelva resultados esperados con datos conocidos.

Esta organización permite navegar fácilmente por el proyecto. Los desarrolladores realizan sus commits en dev junto a las pruebas correspondientes; los archivos de workflow en .github/workflows aseguran que las acciones adecuadas ocurran en cada rama; y los archivos como Dockerfile y main.py encapsulan la lógica para construir y ejecutar la aplicación de forma consistente.


---

## Pipeline CI/CD con GitHub Actions (Pruebas, Construcción y Despliegue)

El repositorio implementa un pipeline de CI/CD con GitHub Actions que consta de dos etapas principales: Pruebas (CI) y Construcción/Promoción a Producción (CD). A continuación, se detallan estas etapas y cómo están automatizadas:

1.	**Etapa de Pruebas (CI) – Continuous Integration:**
Esta fase se ejecuta automáticamente en cada push a la rama dev (es decir, cada vez que se introduce un cambio nuevo en desarrollo). El objetivo es verificar que tanto el código como el modelo funcionan correctamente antes de considerar desplegarlos. Los pasos incluidos son:

2.	**Descarga de Modelo y Datos:** El workflow utiliza credenciales configuradas (por ejemplo, variables de entorno o secretos de GitHub) para acceder al almacenamiento externo donde se encuentran el modelo ONNX y los datos de prueba. Por motivos de eficiencia y tamaño, estos archivos no están versionados en el repositorio, sino que residen en un bucket externo (ver sección siguiente). En esta etapa, se emplea la CLI de AWS (o la librería boto3 en un script Python) para descargar el modelo ONNX y, si aplica, un conjunto de datos de prueba desde el bucket al runner de GitHub Actions.
   
3.	**Instalación de Dependencias:** El runner de GitHub Actions (usualmente Ubuntu) instala las dependencias listadas (por ejemplo, via pip install -r requirements.txt), incluyendo frameworks como FastAPI, ONNX Runtime, etc., necesarios para ejecutar el código y las pruebas.
   
4.	**Ejecución de Pruebas Unitarias:** Se ejecutan los tests automatizados para validar la lógica del modelo y de la aplicación. Por ejemplo, se podría cargar el modelo ONNX y ejecutar una predicción con datos conocidos, verificando que el resultado tenga el formato esperado. Asimismo, se comprueba el correcto funcionamiento de funciones auxiliares y de los endpoints de la API (utilizando herramientas como pytest).
   
Si cualquiera de estos pasos falla (por ejemplo, el modelo no se descarga, o una prueba da error), GitHub Actions marcará el flujo como fallido y notificará en la sección de checks del pull request o commit. Solo una vez que esta etapa de CI pasa con éxito (indicando que los cambios en dev son estables) se procede a integrar esos cambios en la rama prod para desplegarlos.

1.	**Etapa de Construcción y Despliegue (CD) – Continuous Deployment:**
Esta fase se activa cuando los cambios han sido aprobados/probados en dev y son promovidos a la rama prod (por ejemplo, mediante un merge o pull request exitoso). El pipeline de despliegue realiza los siguientes pasos automáticamente:

2.	**Construcción de la Imagen Docker:** Usando el Dockerfile del repositorio, GitHub Actions crea la imagen de contenedor que encapsula la aplicación de predicción junto con el modelo ONNX. Durante este proceso, se asegura de incluir todo lo necesario para que el contenedor sea autónomo: el código de main.py, las dependencias instaladas y el modelo. En algunos casos, el modelo ONNX puede integrarse en la imagen en esta etapa (descargándolo dentro del Dockerfile) o alternativamente el contenedor obtendrá el modelo al iniciarse.
   
3.	**Push de la Imagen a AWS ECR:** Una vez construida la imagen, la pipeline inicia sesión en AWS (utilizando secrets de GitHub para las credenciales de AWS) y etiqueta la imagen apropiadamente (por ejemplo, mlops-project:prod con un tag único, como el commit SHA o un número de versión). Luego, sube esta imagen al Amazon Elastic Container Registry (ECR) del proyecto. ECR actúa como un registro centralizado donde se almacena la imagen del contenedor de nuestro modelo, asegurando que AWS ECS pueda obtenerla para desplegar.
   
4.	**Despliegue en AWS ECS:** Con la nueva imagen disponible, se actualiza el servicio en Amazon Elastic Container Service (ECS) para usar esta versión. En esta arquitectura, ECS maneja un cluster de contenedores; en nuestro caso, se ha configurado un cluster (y servicio) que ejecuta la aplicación de inferencia. El pipeline puede utilizar una acción oficial de AWS o comandos de la CLI (como aws ecs update-service) para indicar que el servicio ECS debe tirar la nueva imagen de ECR y reemplazar la versión anterior de la aplicación. El despliegue se realiza en un entorno de AWS EC2/ECS – es decir, el cluster ECS corre sobre instancias EC2 (modo de lanzamiento EC2) o Fargate según la configuración, pero en este proyecto se menciona EC2, implicando que existe una o más instancias EC2 registradas en el cluster ECS donde los contenedores se ejecutan.
   
5.	**Verificación Post-despliegue:** Aunque gran parte de esto es automatizado, es buena práctica comprobar que el nuevo contenedor esté corriendo correctamente en ECS. El pipeline podría incluir un paso de salud (health-check) llamando a algún endpoint de la API desplegada (por ejemplo, un /ping o similar) para confirmar que la instancia responde. Si este paso está implementado, se notificaría en la consola de Actions cualquier problema inmediatamente.
   
En conjunto, este pipeline CI/CD garantiza que cada cambio pase por pruebas rigurosas antes de llegar a producción, y luego realice el despliegue de forma consistente y repetible. Al utilizar GitHub Actions, todo el proceso ocurre directamente desde el repositorio con cada commit o merge, proporcionando trazabilidad. Además, el uso de AWS ECR/ECS integra el despliegue con infraestructura cloud escalable. Esto sigue las prácticas recomendadas: primero probar, luego construir y versionar el artefacto (imagen Docker) y finalmente desplegarlo en el servicio en la nube[1]. El resultado es un flujo continuo donde los desarrolladores pueden enfocarse en mejorar el modelo o el código, confiando en que el pipeline se encargará de llevar esos cambios a producción de manera segura.


---

## Obtención del Modelo y Datos desde Buckets Externos

Para mantener el repositorio ligero y cumplir buenas prácticas (no incluir archivos pesados o sensibles en Git), el modelo ONNX y los datos necesarios no están contenidos en este repositorio. En su lugar, dichos artefactos se almacenan en buckets externos (por ejemplo, un bucket S3 de AWS u otro almacenamiento compartido proporcionado para el proyecto).

En la configuración de este proyecto, tanto la etapa de pruebas como la propia aplicación en ejecución recurren a estos almacenamientos externos para obtener el modelo y, de ser necesario, datos de ejemplo. A continuación se detalla cómo se maneja esto:

*	**Descarga del Modelo ONNX:** El nombre del bucket y la ruta del archivo del modelo ONNX están definidos en la configuración (podrían estar codificados en el script o suministrados mediante variables de entorno/secretos). Cuando se corre la pipeline de pruebas (CI), un paso explícito se encarga de conectarse al bucket y descargar el archivo del modelo. Usualmente, se utiliza el CLI de AWS (aws s3 cp s3://<bucket>/<ruta_modelo> .) o un script Python con la biblioteca boto3 para bajar el modelo al sistema de archivos local (en el runner de GitHub Actions). De igual forma, al desplegar el contenedor, la aplicación sabe dónde obtener el modelo: si el modelo no se incluyó durante la construcción de la imagen, el main.py podría contener lógica para al inicio conectarse al bucket S3 y descargar el último modelo ONNX antes de levantar el servicio de predicción. Esto asegura que la aplicación siempre use el modelo más reciente aprobado.

*	**Descarga de Datos de Prueba:** Para validar el modelo, es común contar con algunos datos o casos de prueba. Estos también se almacenan externamente. El pipeline de pruebas en la rama dev incluye un paso para descargar un conjunto de datos de prueba (por ejemplo, un CSV con muestras de entrada y salidas esperadas) desde el bucket. Nuevamente, se usa AWS S3 u otro mecanismo similar. Una vez obtenidos, los datos se utilizan en las pruebas unitarias para verificar que el modelo ONNX produce la predicción correcta para entradas conocidas.

*	**Configuración de Credenciales:** Dado que el bucket no es público, el acceso se maneja con credenciales seguras. En GitHub Actions, se han configurado Secretos (como AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY y posiblemente tokens de sesión) que permiten autenticarse contra AWS para realizar las descargas. Estos secretos están encriptados en el repositorio y se utilizan en los workflows sin exponerse públicamente. Asimismo, en el entorno de producción (AWS ECS), la tarea que ejecuta el contenedor puede estar asociada a un IAM Role con permisos de lectura/escritura en el bucket S3 correspondiente. De este modo, la aplicación dentro del contenedor puede acceder al bucket directamente sin embeber credenciales estáticas.

**¿Cómo obtener el modelo y datos manualmente?** Si un desarrollador clona este repositorio y desea ejecutar la aplicación o las pruebas localmente, deberá obtener estos artefactos externamente. Por ejemplo, se tendría que descargar el archivo .onnx del modelo desde el bucket configurado (se puede proporcionar la URL o las instrucciones para solicitar acceso, según las políticas del curso). Igualmente, habría que descargar los archivos de datos de prueba necesarios. Una vez descargados, se ubican en los directorios o rutas donde la aplicación los espera (según la documentación interna o variables de entorno). En este proyecto, se asume que los paths o nombres necesarios estarán indicados ya sea en el código o en la documentación, por ejemplo: MODEL_PATH=/tmp/model.onnx o similar, apuntando al lugar donde colocar el modelo descargado.
Con esta estrategia de almacenaje externo, el proyecto mantiene el repositorio limpio, a la vez que posibilita la reproducibilidad: cualquier persona con los accesos adecuados puede obtener el mismo modelo y datos que el equipo de desarrollo está usando, asegurando coherencia entre ambientes. También evita sobrecargar GitHub con ficheros de gran tamaño o datos sensibles, delegando ese manejo a infraestructuras diseñadas para tal fin (como S3).



---

## Aplicación de Predicción (API de Inferencia en Dev y Prod)

El corazón del sistema es la aplicación de predicción que sirve el modelo entrenado para responder a solicitudes de inferencia. En este proyecto, la aplicación está implementada como un servicio web de API utilizando FastAPI (un framework web ligero y de alto rendimiento en Python) para exponer endpoints HTTP. (Alternativamente, se podría haber utilizado una interfaz visual con Streamlit, pero la solución con FastAPI permite una separación más clara entre los entornos de desarrollo y producción mediante endpoints).

Características de la aplicación:

*	**Carga del Modelo ONNX:** Al iniciarse, la aplicación (ver main.py) carga el modelo de inferencia desde el archivo ONNX. Usa para ello ONNX Runtime u otra librería compatible, garantizando que las predicciones sean rápidas y que el modelo permanezca en memoria para servir múltiples solicitudes. Si el modelo no fue incluido dentro de la imagen Docker, la aplicación primero lo descarga del bucket S3 (usando las credenciales/roles configurados) y luego lo carga en memoria. Este paso inicial ocurre tanto en el entorno de desarrollo como en producción, asegurando que ambos usen el mismo artefacto de modelo aprobado.
  
*	**Endpoints de la API:** La aplicación expone uno o varios endpoints HTTP. El principal es normalmente un endpoint de predicción (por ejemplo, POST /predict) al que un cliente puede enviarle datos de entrada (ej. un JSON con las características requeridas por el modelo) y obtener como respuesta la predicción generada. También suele haber endpoints auxiliares, como un endpoint de health check (GET /health o /) que simplemente devuelve un mensaje de estado ("OK") para indicar que el servicio está en pie. FastAPI automáticamente genera una documentación interactiva (en /docs) donde se pueden explorar y probar los endpoints, lo cual es útil durante desarrollo.
  
*	**Entornos Dev y Prod (Endpoints separados):** Gracias a la estrategia de ramas y despliegue, existen dos instancias de esta aplicación corriendo: una asociada a la rama dev (entorno de desarrollo) y otra asociada a la rama prod (entorno de producción). En la práctica, esto podría traducirse en dos URLs o endpoints desplegados, por ejemplo: http://98.80.174.195:8080/ para desarrollo y http://3.235.92.139:8080/ para producción. La versión de dev se actualiza con cada push a la rama de desarrollo (posiblemente desplegada en un ECS de staging o una instancia separada) y permite probar nuevas funcionalidades del modelo o código en un entorno controlado. La versión prod es la que consume el usuario final o la aplicación cliente real, y sólo se actualiza cuando las nuevas características han pasado las pruebas. Esta separación garantiza que lo que esté en producción esté siempre estable, mientras que el equipo puede iterar rápidamente en dev.
  
*	**Almacenamiento de Predicciones en S3:** Un requerimiento específico de este proyecto es que cada predicción realizada por la API se guarda como un archivo .txt en el bucket S3. Tanto la instancia dev como la prod realizan este guardado de resultados, aunque pueden organizarse en diferentes ubicaciones dentro del bucket para no interferir. Por ejemplo, el endpoint de dev podría escribir los archivos de resultado en una ruta s3://<bucket>/predicciones/dev/, mientras que el de prod lo hace en s3://<bucket>/predicciones/prod/. El contenido de cada archivo de predicción típicamente incluye alguna identificación de la solicitud (p.ej., un timestamp o ID único) y el resultado devuelto por el modelo para esa solicitud. De esta manera, si un usuario envía datos para predicción, el servicio devuelve la respuesta inmediatamente vía API y adicionalmente registra asíncronamente ese resultado en un archivo de texto en el bucket.
  
 	Este mecanismo de logging de predicciones en S3 sirve para varios propósitos: - Permite auditoría y seguimiento: se puede revisar posteriormente qué predicciones se hicieron (especialmente útil en producción para monitorear el comportamiento del modelo con datos reales). - Facilita la comparación entre entornos: dado que se conservan los resultados tanto de dev como de prod, el equipo puede comparar predicciones de ambas versiones para asegurarse de que los cambios introducidos no afectaron negativamente las respuestas. - Ofrece un rastro de datos de inferencia que podría usarse para re-entrenar o mejorar el modelo en el futuro (por ejemplo, almacenando también las entradas junto al resultado, se pueden construir nuevos conjuntos de entrenamiento con datos reales).
 	
*	**Uso y pruebas de la aplicación:** Para interactuar con la API localmente (por ejemplo, durante desarrollo), se puede ejecutar main.py con un servidor Uvicorn: uvicorn main:app --reload lanzará el servicio en http://localhost:8000. Desde allí, un desarrollador puede acceder a http://localhost:8000/docs para ver la documentación Swagger auto-generada por FastAPI y probar el endpoint de /predict. En ese entorno local, es necesario tener el modelo ONNX en la ruta esperada o accesible (siguiendo las instrucciones de descarga previas). En producción, los usuarios o sistemas clientes harán solicitudes HTTP al endpoint publicado de prod. Por diseño, los endpoints de dev no estarían abiertos al público general; su uso queda limitado a pruebas internas o del equipo hasta que los cambios se promocionen a prod.


---


## Endpoints

A continuación se listan los endpoints públicos para acceder a la aplicación desplegada en cada entorno:

### Desarrollo (DEV)
- **URL:** http://98.80.174.195:8080/

### Producción (PROD)
- **URL:** http://3.235.92.139:8080/



---



## Autores

* Johan Sebastian Bonilla
* Jousé Cobaleda
