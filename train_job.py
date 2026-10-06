import argparse, json, os, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from timm.data import resolve_data_config
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, precision_recall_curve

CFG = json.load(open(os.environ['CACAO_CFG']))
OUT = CFG['paths']['out']
DTYPE = torch.bfloat16

# Estado de avance que lee la celda de monitoreo del notebook (logs/progress_w<proceso>.json)
PROG = {}
PROG_FILE = None


def prog(**kw):
    PROG.update(kw)
    PROG['t'] = time.time()
    if PROG_FILE:
        tmp = PROG_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(PROG, f)
        os.replace(tmp, PROG_FILE)


def ahora():
    return time.strftime('%H:%M:%S')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init(wid):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# Imágenes en memoria (256 px, uint8), compartidas con los workers del DataLoader
_RAM = {}


def cargar(paths):
    for p in paths:
        if p not in _RAM:
            _RAM[p] = np.asarray(Image.open(p).convert('RGB'))
    return [_RAM[p] for p in paths]


class ImgDS(Dataset):
    def __init__(self, frame, tf):
        self.imgs = cargar(frame['cache_path'].tolist())
        self.y = frame['label'].astype('float32').tolist()
        self.tf = tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.imgs[i])), torch.tensor(self.y[i], dtype=torch.float32)


def build_transforms(model):
    dc = resolve_data_config({}, model=model)
    mean, std = dc['mean'], dc['std']
    s = CFG['img_size']
    a = CFG['aug']
    cj = a['color_jitter']
    train_tf = T.Compose([
        T.RandomResizedCrop(s, scale=(a['rrc_scale_min'], 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(a['rotation_deg']),
        T.ColorJitter(cj, cj, cj),
        T.ToTensor(),
        T.Normalize(mean, std)])
    eval_tf = T.Compose([T.Resize((s, s)), T.ToTensor(), T.Normalize(mean, std)])
    return train_tf, eval_tf


def set_backbone_trainable(model, trainable):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad = trainable or (id(p) in head_ids)


def bn_inferencia(model, phase):
    # BatchNorm en modo inferencia (estadísticas de ImageNet). Con bn_frozen también en la fase 2:
    # con lotes de 32 el recálculo desestabiliza la cabecera. gamma y beta se entrenan en la fase 2.
    if phase == 1 or CFG.get('bn_frozen', False):
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def run_epoch(model, loader, dev, opt=None, phase=1, etapa='train'):
    train = opt is not None
    model.train(train)
    if train:
        bn_inferencia(model, phase)
    crit = nn.BCEWithLogitsLoss()
    tot, n, ps, ys = 0.0, 0, [], []
    nb_total = len(loader)
    prog(etapa=etapa, lote=0, n_lotes=nb_total, loss_media=None)
    for b, (x, y) in enumerate(loader, start=1):
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(dev, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast('cuda', dtype=DTYPE):
            logit = model(x).squeeze(1)
            loss = crit(logit.float(), y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        tot += loss.item() * len(y)
        n += len(y)
        ps.append(torch.sigmoid(logit.float()).detach().cpu())
        ys.append(y.cpu())
        if b % 5 == 0 or b == nb_total:
            prog(lote=b, loss_media=tot / n)
    p = torch.cat(ps).numpy()
    yy = torch.cat(ys).numpy()
    acc = float(((p >= 0.5) == (yy == 1)).mean())
    auc = float(roc_auc_score(yy, p)) if len(np.unique(yy)) > 1 else float('nan')
    return tot / n, acc, auc


@torch.no_grad()
def predict(model, loader, dev):
    # Predicción final en float32, sin TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.eval()
    ps = []
    for x, _ in loader:
        x = x.to(dev).to(memory_format=torch.channels_last)
        ps.append(torch.sigmoid(model(x).squeeze(1).float()).cpu())
    return torch.cat(ps).numpy()


def umbral_estable(y, p):
    # Umbral: punto medio del intervalo más ancho entre los que dan F1 máximo en validación
    # (el primer máximo variaba mucho entre pliegues con clases casi separadas)
    prec, rec, thr = precision_recall_curve(y, p)
    if not len(thr):
        return 0.5, 0.5, 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    optimos = thr[f1 >= f1.max() - 1e-9]
    ps = np.sort(np.unique(p))
    mejor = None
    for t in optimos:
        abajo = ps[ps < t]
        lo = float(abajo[-1]) if len(abajo) else 0.0
        if mejor is None or t - lo > mejor[1] - mejor[0]:
            mejor = (lo, float(t))
    return (mejor[0] + mejor[1]) / 2, mejor[0], mejor[1]


def atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_one(job, dev):
    name, split, tag, seed = job['model'], job['split'], job['tag'], job['seed']
    arch = CFG['models'][name]
    done_flag = os.path.join(OUT, 'preds', tag + '.done')
    if os.path.exists(done_flag):
        return
    seed_all(seed)

    folds = pd.read_csv(os.path.join(OUT, 'folds.csv'))
    col = 'split_' + split
    tr, va, te = [folds[folds[col] == s].reset_index(drop=True) for s in ('train', 'val', 'test')]

    model = timm.create_model(arch, pretrained=True, num_classes=1, drop_rate=CFG['drop_rate'])
    # Inicialización común de la cabecera (timm usa U(-1, 1) en EfficientNet con una sola salida)
    head = model.get_classifier()
    nn.init.trunc_normal_(head.weight, std=0.02)
    nn.init.zeros_(head.bias)
    model = model.to(dev).to(memory_format=torch.channels_last)
    train_tf, eval_tf = build_transforms(model)
    g = torch.Generator()
    g.manual_seed(seed)
    nw = CFG['num_workers']
    kw = dict(batch_size=CFG['batch_size'], num_workers=nw, pin_memory=True,
              persistent_workers=nw > 0, worker_init_fn=worker_init)
    if nw > 0:
        kw['prefetch_factor'] = 4
    dl_tr = DataLoader(ImgDS(tr, train_tf), shuffle=True, drop_last=True, generator=g, **kw)
    dl_va = DataLoader(ImgDS(va, eval_tf), shuffle=False, **kw)
    dl_te = DataLoader(ImgDS(te, eval_tf), shuffle=False, **kw)

    ck_last = os.path.join(OUT, 'ckpt', tag + '_last.pt')
    ck_best = os.path.join(OUT, 'ckpt', tag + '_best.pt')
    st = {'phase': 1, 'epoch': 0, 'best_val': float('inf'), 'bad': 0, 'history': [],
          'opt': None, 'sched': None, 'train_seconds': 0.0, 'epochs_phase': {1: 0, 2: 0}}
    if os.path.exists(ck_last):
        st = torch.load(ck_last, map_location='cpu', weights_only=False)
        model.load_state_dict(st.pop('model'))
        print(ahora(), '[reanudando]', tag, 'fase', st['phase'], 'época', st['epoch'], flush=True)

    while st['phase'] <= 2:
        ph = st['phase']
        P = CFG['phase' + str(ph)]
        set_backbone_trainable(model, trainable=(ph == 2))
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=P['lr'], weight_decay=CFG['weight_decay'])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=CFG['plateau']['factor'],
            patience=CFG['plateau']['patience'], min_lr=CFG['plateau']['min_lr'])
        if st['opt'] is not None:
            opt.load_state_dict(st['opt'])
            sch.load_state_dict(st['sched'])
        while st['epoch'] < P['epochs'] and st['bad'] < P['patience']:
            t0 = time.time()
            prog(fase=ph, epoca=st['epoch'] + 1, max_epocas=P['epochs'], sin_mejora=st['bad'],
                 paciencia=P['patience'], mejor_val=None if st['best_val'] == float('inf') else st['best_val'])
            trl, tra, tru = run_epoch(model, dl_tr, dev, opt, ph, 'train')
            vl, vacc, vauc = run_epoch(model, dl_va, dev, None, ph, 'val')
            sch.step(vl)
            st['epoch'] += 1
            st['epochs_phase'][ph] += 1
            dt = time.time() - t0
            st['train_seconds'] += dt
            st['history'].append({'phase': ph, 'epoch_global': len(st['history']) + 1,
                                  'train_loss': trl, 'train_acc': tra, 'train_auc': tru,
                                  'val_loss': vl, 'val_acc': vacc, 'val_auc': vauc,
                                  'lr': opt.param_groups[0]['lr'], 'seconds': dt})
            if vl < st['best_val'] - 1e-4:
                st['best_val'] = vl
                st['bad'] = 0
                atomic_save(model.state_dict(), ck_best)
            else:
                st['bad'] += 1
            prog(ultima={'fase': ph, 'epoca': st['epoch'], 'train_loss': trl, 'train_acc': tra,
                         'val_loss': vl, 'val_acc': vacc, 'val_auc': vauc, 'seg': dt},
                 sin_mejora=st['bad'], mejor_val=st['best_val'])
            print('%s %s | fase %d | época %d/%d | train loss %.4f acc %.3f | val loss %.4f acc %.3f AUC %.4f'
                  ' | lr %.1e | sin mejora %d/%d | %.0f s' % (
                      ahora(), tag, ph, st['epoch'], P['epochs'], trl, tra, vl, vacc, vauc,
                      opt.param_groups[0]['lr'], st['bad'], P['patience'], dt), flush=True)
            st['opt'] = opt.state_dict()
            st['sched'] = sch.state_dict()
            atomic_save(dict(st, model=model.state_dict()), ck_last)
        # Fin de la fase: se recuperan los mejores pesos y se pasa a la siguiente
        model.load_state_dict(torch.load(ck_best, map_location=dev))
        st.update({'phase': ph + 1, 'epoch': 0, 'bad': 0, 'opt': None, 'sched': None})
        atomic_save(dict(st, model=model.state_dict()), ck_last)

    prog(etapa='predicción', lote=0, n_lotes=0)
    p_va = predict(model, dl_va, dev)
    p_te = predict(model, dl_te, dev)
    thr_val, thr_lo, thr_hi = umbral_estable(va['label'].values, p_va)
    partes = []
    for sp, frame, p in (('val', va, p_va), ('test', te, p_te)):
        o = frame[['image_path', 'label', 'group', 'fuente']].copy()
        o['prob'] = p
        o['split'] = sp
        o['particion'] = split
        o['model'] = name
        partes.append(o)
    pd.concat(partes).to_csv(os.path.join(OUT, 'preds', tag + '.csv'), index=False)
    pd.DataFrame(st['history']).to_csv(os.path.join(OUT, 'hist', tag + '.csv'), index=False)
    meta = {'model': name, 'arch': arch, 'particion': split, 'seed': seed,
            'thr_val': thr_val, 'thr_val_intervalo': [thr_lo, thr_hi],
            'best_val_loss': st['best_val'], 'epochs_phase1': st['epochs_phase'][1],
            'epochs_phase2': st['epochs_phase'][2], 'train_seconds': st['train_seconds'],
            'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
            'test_auc': float(roc_auc_score(te['label'].values, p_te))}
    with open(os.path.join(OUT, 'preds', tag + '.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    with open(done_flag, 'w') as f:
        f.write('ok')
    if os.path.exists(ck_last):
        os.remove(ck_last)
    print('%s [terminado] %s | AUC prueba %.4f | épocas %d+%d | %.1f min' % (
        ahora(), tag, meta['test_auc'], meta['epochs_phase1'], meta['epochs_phase2'],
        meta['train_seconds'] / 60), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(2)              # varios procesos comparten los hilos de CPU
    jobs = json.load(open(os.path.join(OUT, 'jobs.json')))
    PROG_FILE = os.path.join(OUT, 'logs', 'progress_w%d.json' % a.worker)
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
    prog(worker=a.worker, gpu=gpu, estado='corriendo', inicio=time.time(), trabajo=None, n_hechos=0)
    print(ahora(), 'Proceso', a.worker, 'en GPU', gpu, flush=True)
    # Cola de trabajos: cada proceso reserva el siguiente con preds/<trabajo>.claim (O_EXCL)
    k = 0
    for job in jobs:
        tag = job['tag']
        if os.path.exists(os.path.join(OUT, 'preds', tag + '.done')):
            continue
        try:
            fd = os.open(os.path.join(OUT, 'preds', tag + '.claim'), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(a.worker).encode())
            os.close(fd)
        except FileExistsError:
            continue
        k += 1
        prog(i_trabajo=k, trabajo=tag, inicio_trabajo=time.time(), fase=None, epoca=None,
             ultima=None, etapa='cargando modelo', lote=0, n_lotes=0)
        print(ahora(), 'Proceso', a.worker, 'toma', tag, flush=True)
        try:
            train_one(job, dev)
            prog(n_hechos=k)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print('[ERROR]', tag, repr(e), flush=True)
        torch.cuda.empty_cache()
    prog(estado='finalizado', etapa='-', trabajo=None)
    print(ahora(), 'Proceso', a.worker, 'finalizó.', flush=True)
