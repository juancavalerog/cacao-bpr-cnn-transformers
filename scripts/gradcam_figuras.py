"""Figuras 6 y 7: mapas Grad-CAM (repetición 1, pliegue 1). Requiere resultados/ckpt/*_r0_f0_best.pt y pytorch-grad-cam."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, torch, timm, torchvision.transforms as T
from timm.data import resolve_data_config
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman','DejaVu Serif'],'savefig.dpi':350})
RAIZ = Path(__file__).resolve().parents[1]
U = str(RAIZ)
df=pd.read_csv(U+'/resultados/folds.csv')
sel=json.load(open(RAIZ/'scripts'/'gradcam_seleccion.json'))   # índices de folds.csv (repetición 1, pliegue 1)
ARCH={'SwinT':'swin_tiny_patch4_window7_224.ms_in1k','ConvNeXtT':'convnext_tiny.fb_in1k','DeiT':'deit_tiny_patch16_224.fb_in1k',
      'EfficientNetV2S':'tf_efficientnetv2_s.in1k','ResNet50':'resnet50.tv_in1k'}
NM={'SwinT':'Swin-T','ConvNeXtT':'ConvNeXt-T','DeiT':'DeiT-Ti','EfficientNetV2S':'EfficientNetV2-S','ResNet50':'ResNet50'}
M=list(ARCH)
class Obj:
    def __init__(s,c): s.sg=1.0 if c==1 else -1.0
    def __call__(s,o): return s.sg*o.sum()
def rs(t):
    if t.ndim==4: return t.permute(0,3,1,2)
    b,l,c=t.shape; s=int(round(l**0.5)); return t.reshape(b,s,s,c).permute(0,3,1,2)
def capas(n,m):
    a=ARCH[n]
    if a.startswith('resnet'): return [m.layer4[-1]],None
    if a.startswith('convnext'): return [m.stages[-1]],None
    if 'efficientnet' in a: return [m.conv_head],None
    if a.startswith('swin'): return [m.layers[-1].blocks[-1].norm2],rs
    npt=m.num_prefix_tokens
    def rv(t):
        t=t[:,npt:,:]; b,l,c=t.shape; s=int(round(l**0.5)); return t.reshape(b,s,s,c).permute(0,3,1,2)
    return [m.blocks[-1].norm1],rv
def path(i): return U+'/'+df.loc[i,'cache_path'].split('/workspace/cacao/')[1]
def fig(idx,archivo):
    CM=1/2.54; n=len(idx)
    fig,axes=plt.subplots(n,6,figsize=(15.5*CM,15.5*CM*n/6*1.08))
    base=[np.asarray(Image.open(path(i)).convert('RGB').resize((224,224))).astype(np.float32)/255 for i in idx]
    for r,i in enumerate(idx):
        axes[r,0].imshow(base[r]); axes[r,0].set_ylabel('Real: '+('BPR' if df.loc[i,'label']==1 else 'Sana'),fontsize=8.5,fontweight='bold')
    for c,nm in enumerate(M,start=1):
        m=timm.create_model(ARCH[nm],pretrained=False,num_classes=1)
        m.load_state_dict(torch.load(U+'/resultados/ckpt/%s_r0_f0_best.pt'%nm,map_location='cpu')); m.eval()
        for p in m.parameters(): p.requires_grad_(True)
        dc=resolve_data_config({},model=m); tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize(dc['mean'],dc['std'])])
        L,R=capas(nm,m)
        with GradCAM(model=m,target_layers=L,reshape_transform=R) as cam:
            for r,i in enumerate(idx):
                x=tf(Image.open(path(i)).convert('RGB')).unsqueeze(0)
                with torch.no_grad(): p=torch.sigmoid(m(x)).item()
                mp=cam(input_tensor=x,targets=[Obj(int(p>=0.5))])[0]
                axes[r,c].imshow(show_cam_on_image(base[r],mp,use_rgb=True))
                ok=(p>=0.5)==(df.loc[i,'label']==1)
                axes[r,c].set_xlabel('%s (p = %.2f)'%('BPR' if p>=0.5 else 'Sana',p),fontsize=7.5,color='black' if ok else '#c00000',labelpad=1.5)
        print(nm,'ok',flush=True)
    axes[0,0].set_title('Imagen',fontsize=9,fontweight='bold')
    for c,nm in enumerate(M,start=1): axes[0,c].set_title(NM[nm],fontsize=9,fontweight='bold')
    for a in axes.flat: a.set_xticks([]); a.set_yticks([])
    plt.subplots_adjust(wspace=0.04,hspace=0.18,left=0.04,right=0.995,top=0.96,bottom=0.02)
    os.makedirs(U+'/figuras_articulo',exist_ok=True); fig.savefig(U+'/figuras_articulo/'+archivo,bbox_inches='tight'); plt.close(fig)
fig(sel['sel'],'Figura_6_gradcam_ejemplos.png'); fig(sel['err'],'Figura_7_gradcam_errores_swin.png')
