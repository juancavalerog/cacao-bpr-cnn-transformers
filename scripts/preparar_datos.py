"""Copia en dataT/ las 2211 imágenes del estudio a partir del repositorio Cacao Diseases (Kaggle).

Uso (desde la raíz del repositorio):
    python scripts/preparar_datos.py
Requiere el repositorio descomprimido en dataset_complete/cacao_diseases/cacao_photos/.
"""
import csv, shutil, sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ORIGEN = RAIZ / 'dataset_complete' / 'cacao_diseases' / 'cacao_photos'
CARPETA_REPO = {'healthy': 'healthy', 'pod_rot_black': 'black_pod_rot'}   # carpeta en dataT -> carpeta en Kaggle

faltan, copiadas = [], 0
with open(RAIZ / 'dataT' / 'data.csv', newline='', encoding='utf-8') as fh:
    for fila in csv.DictReader(fh):
        destino = RAIZ / fila['image_path']
        carpeta, nombre = destino.parent.name, destino.name
        origen = ORIGEN / CARPETA_REPO[carpeta] / nombre
        if not origen.exists():
            faltan.append(str(origen))
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        if not destino.exists():
            shutil.copy2(origen, destino)
        copiadas += 1
print('Imágenes disponibles en dataT/: %d' % copiadas)
if faltan:
    print('No se encontraron %d archivos en el repositorio descargado, por ejemplo: %s' % (len(faltan), faltan[:3]))
    sys.exit(1)
