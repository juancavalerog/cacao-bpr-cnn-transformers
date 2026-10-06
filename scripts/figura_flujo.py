import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from pathlib import Path
P=str(Path(__file__).resolve().parents[1]); R=P+'/resultados'; O=P+'/figuras_articulo'; os.makedirs(O,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'savefig.dpi':400,'mathtext.fontset':'stix'})
CM=1/2.54
folds=pd.read_csv(R+'/folds.csv')
fig=plt.figure(figsize=(15.5*CM,20.1*CM)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(20,135); ax.axis('off')
def box(x,y,w,h,title,lines,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.5',fc=fc,ec='#333333',lw=0.8))
    ax.text(x+w/2,y+h-2.3,title,ha='center',va='top',fontsize=10,fontweight='bold')
    ax.text(x+w/2,y+h-6.6,'\n'.join(lines),ha='center',va='top',fontsize=8.2,linespacing=1.4)
def arrow(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=1.1,color='#333333'))
# fila de miniaturas
rng=np.random.default_rng(3)
ids=list(rng.choice(folds[(folds.label==0)&(folds.fuente=='A')].index,3,replace=False))+list(rng.choice(folds[folds.label==1].index,3,replace=False))
for i,k in enumerate(ids):
    im=Image.open(P+'/'+folds.loc[k,'image_path']).convert('RGB'); im.thumbnail((400,400))
    a=fig.add_axes([0.06+i*0.148,0.855,0.13,0.13*15.5/20.1]); a.imshow(im); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor('#4d9e4d' if folds.loc[k,'label']==0 else '#5b3a29'); s.set_linewidth(2)
ax.text(27,113.5,'Sana (n = 1268)',ha='center',fontsize=8.5,color='#2e6b2e',fontweight='bold')
ax.text(72,113.5,'BPR (n = 943)',ha='center',fontsize=8.5,color='#5b3a29',fontweight='bold')
ax.text(50,110.3,'Repositorio Cacao Diseases (Kaggle): 2211 imágenes',ha='center',fontsize=8)
arrow(50,108.8,50,105.3)
box(5,90,90,14.5,'1. Auditoría del conjunto de datos',
    ['MD5 · pHash (≤ 6) · similitud DINOv2 · correspondencias SIFT + RANSAC',
     'Grupos por secuencia de captura: 2211 imágenes → 664 grupos',
     'Control de atajos: regresión logística con tamaño, borde y color'],'#eef3fb')
arrow(50,89.4,50,86.3)
box(5,70.5,90,15,'2. Particiones',
    ['Validación cruzada estratificada por grupos: 3 repeticiones × 5 pliegues (semillas 42, 142, 242)',
     'Ningún grupo se reparte entre conjuntos; la prueba solo se usa en la evaluación final',
     'Resto: entrenamiento y validación interna (1/7) para detención temprana y umbral'],'#eef8ee')
arrow(50,69.9,50,66.8)
box(5,44,90,22,'3. Entrenamiento (idéntico para los cinco modelos)',
    ['Swin-T · ConvNeXt-T · DeiT-Ti · EfficientNetV2-S · ResNet50 (ImageNet-1k, timm)',
     'Cabecera: agrupación global + Dropout 0.5 + lineal (1 salida) + sigmoide',
     '',
     'Fase 1: extractor congelado · AdamW 1 × $10^{-3}$ · ≤ 20 épocas · paciencia 5',
     'Fase 2: red completa · AdamW 1 × $10^{-5}$ · ≤ 40 épocas · paciencia 8',
     'Lote 32 · ReduceLROnPlateau · BatchNorm en modo de inferencia · entropía cruzada binaria',
     'Aumento: recorte 80 a 100 %, volteo, rotación ±20°, color ±20 % · entrada 224 × 224 px'],'#fdf3e7')
arrow(50,43.4,50,40.3)
box(5,23,90,17,'4. Evaluación',
    ['Predicciones fuera de pliegue (2211 imágenes por repetición): AUC, sensibilidad,',
     'especificidad, F1, Brier, ECE · DeLong · McNemar · Holm · bootstrap por grupos',
     'Sanas no utilizadas del repositorio (1958): especificidad por escena de captura',
     'Grad-CAM (casos seleccionados) · costo: parámetros, FLOPs, latencia y memoria'],'#f5eef8')
fig.savefig(O+'/Figura_2_flujo_metodologico.png',bbox_inches='tight',pad_inches=0.05); plt.close(fig); print('ok')
