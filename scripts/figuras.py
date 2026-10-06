import os, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
P=str(Path(__file__).resolve().parents[1]); R=P+'/resultados'; O=P+'/figuras_articulo'; os.makedirs(O,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'font.size':10,
                     'axes.titlesize':10,'axes.labelsize':10,'legend.fontsize':8.5,'xtick.labelsize':9,'ytick.labelsize':9,
                     'savefig.dpi':400})
M=['SwinT','ConvNeXtT','DeiT','EfficientNetV2S','ResNet50']
NM={'SwinT':'Swin-T','ConvNeXtT':'ConvNeXt-T','DeiT':'DeiT-Ti','EfficientNetV2S':'EfficientNetV2-S','ResNet50':'ResNet50'}
COL={'SwinT':'#1f77b4','ConvNeXtT':'#ff7f0e','DeiT':'#2ca02c','EfficientNetV2S':'#d62728','ResNet50':'#9467bd'}
CM=1/2.54
folds=pd.read_csv(R+'/folds.csv')
pr=pd.concat([pd.read_csv(f) for f in glob.glob(R+'/preds/*_r0_f*.csv')])
te=pr[pr.split=='test']
oof=te.pivot_table(index=['image_path','label'],columns='model',values='prob').reset_index()
y=oof['label'].values.astype(int)
def roc(y,p):
    o=np.argsort(-p,kind='mergesort'); ys=y[o]; ps=p[o]
    tp=np.cumsum(ys); fp=np.cumsum(1-ys); last=np.r_[ps[1:]!=ps[:-1],True]
    return np.r_[0,fp[last]/(1-ys).sum()], np.r_[0,tp[last]/ys.sum()]
ci=pd.read_csv(R+'/tablas/tabla_auc_oof_ic95.csv'); ci=ci[ci['Repetición']==1].set_index('Modelo')

# Figura 1
rng=np.random.default_rng(7)
fig=plt.figure(figsize=(15.5*CM,15.5*CM))
gs=fig.add_gridspec(3,3,height_ratios=[1,1,1.15],hspace=0.35,wspace=0.06)
for r,(lab,nom) in enumerate([(0,'Sana'),(1,'BPR')]):
    sub=folds[(folds.label==lab)&(folds.fuente=='A')]
    for c,k in enumerate(rng.choice(sub.index,3,replace=False)):
        ax=fig.add_subplot(gs[r,c]); im=Image.open(P+'/'+folds.loc[k,'image_path']).convert('RGB'); im.thumbnail((700,700))
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        if c==0: ax.set_ylabel(nom,fontsize=10,fontweight='bold')
        if r==0 and c==0: ax.set_title('A',loc='left',fontweight='bold',fontsize=11)
ax=fig.add_subplot(gs[2,:])
tab=pd.crosstab(folds['fuente'],folds['label'])
x=np.arange(2); w=0.36
for i,(lab,nom,colr) in enumerate([(0,'Sana','#4d9e4d'),(1,'BPR','#5b3a29')]):
    v=[tab.loc['A',lab],tab.loc['B',lab]]
    ax.bar(x+(i-0.5)*w,v,w,label=nom,color=colr)
    for xx,vv in zip(x+(i-0.5)*w,v): ax.text(xx,vv+15,str(vv),ha='center',fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(['Estrato A (1080 × 1080 px)\nn = %d'%tab.loc['A'].sum(),'Estrato B (2160 × 2160 px)\nn = %d'%tab.loc['B'].sum()])
ax.set_ylabel('Número de imágenes'); ax.legend(frameon=False,title='Total: %d sanas y %d BPR'%(tab[0].sum(),tab[1].sum()),title_fontsize=8.5)
ax.set_ylim(0,1080); ax.spines[['top','right']].set_visible(False); ax.set_title('B',loc='left',fontweight='bold',fontsize=11)
fig.savefig(O+'/Figura_1_clases_y_estratos.png',bbox_inches='tight'); plt.close(fig)

# Figura 3
fig,axes=plt.subplots(5,2,figsize=(15.5*CM,21*CM),sharex=True)
for i,m in enumerate(M):
    for k in range(5):
        h=pd.read_csv(R+'/hist/%s_r0_f%d.csv'%(m,k)); c=plt.cm.tab10(k)
        axes[i,0].plot(h.epoch_global,h.val_loss,lw=1,color=c,label='Pliegue %d'%(k+1))
        axes[i,0].plot(h.epoch_global,h.train_loss,lw=0.8,ls='--',color=c,alpha=0.7)
        axes[i,1].plot(h.epoch_global,h.val_auc,lw=1,color=c)
        cut=(h.phase==1).sum()+0.5
        for a in axes[i]: a.axvline(cut,color='grey',ls=':',lw=0.7)
    axes[i,0].set_ylabel(NM[m]+'\nPérdida (BCE)'); axes[i,1].set_ylabel('AUC de validación')
    axes[i,0].set_ylim(0,0.7); axes[i,1].set_ylim(0.85,1.002)
    for a in axes[i]: a.grid(alpha=0.25); a.spines[['top','right']].set_visible(False)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,1.035))
axes[-1,0].set_xlabel('Época'); axes[-1,1].set_xlabel('Época')
axes[0,0].set_title('Pérdida (— validación, -- entrenamiento)'); axes[0,1].set_title('AUC de validación')
fig.tight_layout(); fig.savefig(O+'/Figura_3_curvas_entrenamiento.png',bbox_inches='tight'); plt.close(fig)

# Figura 4
fig,axes=plt.subplots(3,2,figsize=(13.5*CM,17.5*CM))
fn,fpv=[],[]
for ax,m in zip(axes.flat,M):
    p=oof[m].values; yp=(p>=0.5).astype(int)
    cm=np.array([[((y==0)&(yp==0)).sum(),((y==0)&(yp==1)).sum()],[((y==1)&(yp==0)).sum(),((y==1)&(yp==1)).sum()]])
    fn.append(int(cm[1,0])); fpv.append(int(cm[0,1]))
    ax.imshow(cm,cmap='Greys',vmin=0,vmax=cm.max()*1.15)
    for a in range(2):
        for b in range(2):
            ax.text(b,a,'%d\n(%.1f %%)'%(cm[a,b],100*cm[a,b]/cm[a].sum()),ha='center',va='center',fontsize=9.5,
                    color='white' if cm[a,b]>cm.max()*0.55 else 'black')
    ax.set_xticks([0,1]); ax.set_xticklabels(['Sana','BPR']); ax.set_yticks([0,1]); ax.set_yticklabels(['Sana','BPR'])
    ax.set_xlabel('Predicción'); ax.set_ylabel('Real'); ax.set_title('%s (n = %d)'%(NM[m],cm.sum()))
ax=axes.flat[5]; x=np.arange(5); w=0.38
ax.bar(x-w/2,fn,w,label='Falsos negativos',color='#5b3a29'); ax.bar(x+w/2,fpv,w,label='Falsos positivos',color='#9e9e9e')
for xx,v in zip(x-w/2,fn): ax.text(xx,v+0.8,str(v),ha='center',fontsize=8)
for xx,v in zip(x+w/2,fpv): ax.text(xx,v+0.8,str(v),ha='center',fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([NM[m] for m in M],rotation=35,ha='right',fontsize=8.5)
ax.set_ylabel('Imágenes'); ax.legend(frameon=False,fontsize=8); ax.spines[['top','right']].set_visible(False)
ax.set_title('Errores por modelo'); ax.set_ylim(0,max(fn+fpv)*1.25)
fig.tight_layout(); fig.savefig(O+'/Figura_4_matrices_confusion.png',bbox_inches='tight'); plt.close(fig)
print('FN',fn,'FP',fpv)

# Figura 5 (vertical: A arriba, B abajo)
fig=plt.figure(figsize=(13.5*CM,23*CM))
gsf=fig.add_gridspec(2,1,height_ratios=[1,1.18],hspace=0.62)
ax=fig.add_subplot(gsf[0]); ins=ax.inset_axes([0.40,0.08,0.56,0.52])
for m in M:
    f,t=roc(y,oof[m].values)
    ax.plot(f,t,lw=1.3,color=COL[m],label='%s: %.3f (%.3f a %.3f)'%(NM[m],ci.loc[m,'AUC_OOF'],ci.loc[m,'IC95_inf'],ci.loc[m,'IC95_sup']))
    ins.plot(f,t,lw=1.3,color=COL[m])
ax.plot([0,1],[0,1],'k--',lw=0.7); ins.set_xlim(0,0.1); ins.set_ylim(0.85,1.0); ins.tick_params(labelsize=8); ins.set_xticks([0,0.05,0.1]); ins.grid(alpha=0.3)
ax.indicate_inset_zoom(ins,edgecolor='grey')
ax.set_xlabel('Tasa de falsos positivos (1 − especificidad)'); ax.set_ylabel('Sensibilidad')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.17),fontsize=8.5,frameon=False,title='AUC fuera de pliegue (IC95 %)',title_fontsize=8.5,ncol=1)
ax.set_title('A',loc='left',fontweight='bold',fontsize=12)
gsb=gsf[1].subgridspec(2,1,height_ratios=[3.2,1],hspace=0.08)
ax=fig.add_subplot(gsb[0]); axh=fig.add_subplot(gsb[1],sharex=ax)
edges=np.linspace(0,1,11); cen=(edges[:-1]+edges[1:])/2
for m in M:
    p=oof[m].values; b=np.minimum((p*10).astype(int),9)
    ok=[k for k in range(10) if (b==k).any()]
    mp=[p[b==k].mean() for k in ok]; fo=[y[b==k].mean() for k in ok]
    ax.plot(mp,fo,marker='o',ms=3,lw=1.1,color=COL[m],label='%s (Brier %.3f)'%(NM[m],np.mean((p-y)**2)))
    axh.step(cen,[(b==k).sum() for k in range(10)],where='mid',color=COL[m],lw=1)
ax.plot([0,1],[0,1],'k--',lw=0.7); ax.set_ylabel('Frecuencia observada de BPR'); ax.grid(alpha=0.25)
plt.setp(ax.get_xticklabels(),visible=False)
axh.set_yscale('log'); axh.set_ylabel('Imágenes\npor intervalo',fontsize=8.5); axh.set_xlabel('Probabilidad predicha de BPR (10 intervalos de igual anchura)')
axh.set_xlim(0,1); axh.grid(alpha=0.25)
ax.legend(loc='upper left',fontsize=8,frameon=False)
ax.set_title('B',loc='left',fontweight='bold',fontsize=12)
fig.savefig(O+'/Figura_5_roc_calibracion.png',bbox_inches='tight'); plt.close(fig)
print('ok')
