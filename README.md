# Redes convolucionales y Transformers para clasificar la pudrición negra del cacao (*Phytophthora* spp.)

Código, particiones, predicciones y tablas del artículo:

> Valero Gómez, J. C.; Clares Perca, J. C.; Zúñiga Incalla, A. P. *Redes Convolucionales y Transformers para Clasificar la Pudrición Negra del Cacao (Phytophthora spp.): Auditoría de Identidad de Imágenes y Sensibilidad al Contexto de Captura*. Nativa (en evaluación).

Se comparan Swin-T, ConvNeXt-T, DeiT-Ti, EfficientNetV2-S y ResNet50 (timm, ImageNet-1k) en la clasificación de mazorcas sanas y con pudrición negra (BPR) con validación cruzada estratificada por grupos (3 repeticiones × 5 pliegues). Antes de formar las particiones se auditó la identidad de las imágenes (MD5, pHash, DINOv2, SIFT + RANSAC y revisión visual) para que las tomas de una misma mazorca o escena no se repartan entre entrenamiento y prueba.

*English summary.* Code, data splits, out-of-fold predictions and result tables for a comparison of five ImageNet-pretrained CNN and Transformer architectures for cacao black pod rot classification, with an image identity audit (capture-sequence grouping), repeated group cross-validation, DeLong/McNemar tests, calibration, Grad-CAM, computational cost and an evaluation on unused healthy images by capture scene.

## Contenido

| Ruta | Contenido |
|---|---|
| `01_auditoria_identidad_evidencia.ipynb` | MD5, pHash, descriptores DINOv2, pares candidatos y correspondencias SIFT + RANSAC |
| `02_auditoria_identidad_grupos.ipynb` | Regla de agrupación por secuencia de captura, herramienta de revisión visual y grupos finales |
| `03_entrenamiento_validacion_cruzada.ipynb` | Particiones, entrenamiento de los 5 modelos (75 ejecuciones), métricas, DeLong, McNemar, figuras, complejidad y Grad-CAM |
| `04_evaluacion_sanas_no_utilizadas.ipynb` | Evaluación de los 75 modelos en las sanas del repositorio que no se usaron, por escena de captura |
| `train_job.py`, `bench_job.py` | Script de entrenamiento (lo escribe y lanza el notebook 03) y medición de costo en procesos independientes |
| `scripts/` | Preparación de datos y generación de las figuras del artículo |
| `dataT/data.csv` | Lista de las 2211 imágenes analizadas (1268 sanas y 943 con BPR) |
| `grupos/grupos_finales.csv` | Grupo por secuencia de captura de cada imagen (664 grupos) |
| `auditoria_identidad/` | Evidencia de la auditoría: pares candidatos, descriptores, decisiones de la revisión visual, censo de escenas y vínculos de las sanas no usadas |
| `resultados/` | Configuración, particiones (`folds.csv`), predicciones fuera de pliegue (`preds/`), historiales de entrenamiento (`hist/`), tablas y figuras de trabajo |
| `resultados_sanas_no_usadas/` | Predicciones y tablas de la evaluación complementaria |
| `figuras_articulo/` | Figuras 1 a 7 del artículo (ver `figuras_articulo/LEEME.txt`) |
| `Dockerfile`, `compose.yml`, `requirements-docker.txt` | Entorno reproducible (PyTorch 2.14.0 + CUDA 12.6, timm 1.0.30) |

Los puntos de control de los modelos (`resultados/ckpt/`, unos 6 GB) no se incluyen por su tamaño; están disponibles a solicitud de los autores. Las tablas y predicciones incluidas permiten verificar todas las cifras del artículo sin reentrenar.

## Datos

Las imágenes provienen del repositorio público *Cacao Diseases* (Pagaduan, 2021, Kaggle: https://www.kaggle.com/datasets/zaldyjr/cacao-diseases) y no se redistribuyen aquí. Para reproducir:

1. Descargar el repositorio de Kaggle y descomprimirlo en `dataset_complete/` (debe quedar `dataset_complete/cacao_diseases/cacao_photos/{healthy,black_pod_rot,pod_borer}/`).
2. Ejecutar `python scripts/preparar_datos.py`, que copia en `dataT/healthy/` y `dataT/pod_rot_black/` exactamente las imágenes listadas en `dataT/data.csv` y verifica que estén todas.

## Entorno

Probado en Linux con 2 × NVIDIA A100 80 GB (controlador 560.35.03).

```bash
echo "JUPYTER_TOKEN=elija-un-token" > .env
docker compose up -d --build      # Jupyter Lab en http://localhost:8889/lab?token=elija-un-token
```

Las versiones exactas están en `requirements-docker.txt` y `resultados/requirements.txt`.

## Orden de ejecución

1. `01_auditoria_identidad_evidencia.ipynb` (10 a 20 min).
2. `02_auditoria_identidad_grupos.ipynb`. La revisión visual es opcional: con `auditoria_identidad/revision_decisiones.csv` (396 pares revisados) se reproducen los grupos del estudio. El archivo de referencia es `grupos/grupos_finales.csv`.
3. `03_entrenamiento_validacion_cruzada.ipynb`. Con `SMOKE_TEST = True` hace una prueba rápida en `prueba/`; con `False` entrena las 75 ejecuciones en `resultados/` (alrededor de 1 h con 2 A100 y 6 procesos por GPU). Si `resultados/` ya contiene predicciones, las secciones de métricas y figuras se pueden ejecutar directamente.
4. `04_evaluacion_sanas_no_utilizadas.ipynb` (requiere los puntos de control).
5. Figuras del artículo: `python scripts/figuras.py`, `python scripts/figura_flujo.py` y `python scripts/gradcam_figuras.py` (este último requiere los puntos de control y `pytorch-grad-cam`).

## Cómo citar

Ver `CITATION.cff`. Si usa este material, cite el artículo y el repositorio archivado en Zenodo (DOI en la página del repositorio).

## Licencia

Código bajo licencia MIT (`LICENSE`). Las imágenes pertenecen a su autor original y se rigen por la licencia del repositorio de Kaggle.
