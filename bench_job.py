import json, sys, time
import torch, timm
from torch.utils.flop_counter import FlopCounterMode
arch, S = sys.argv[1], int(sys.argv[2])
dev = torch.device('cuda:0')

def latencia_gpu(model, x, n=200, warm=30, amp=False):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        for _ in range(warm):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
m = timm.create_model(arch, pretrained=False, num_classes=1).eval().to(dev)
x1 = torch.randn(1, 3, S, S, device=dev)
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    m(x1)
gflops = fc.get_total_flops() / 1e9
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev)
ms_fp32 = latencia_gpu(m, x1)
mem_mb = torch.cuda.max_memory_allocated(dev) / 1024**2
ms_bf16 = latencia_gpu(m, x1, amp=True)
xb = torch.randn(64, 3, S, S, device=dev)
img_s = 64 / (latencia_gpu(m, xb, n=30, warm=5, amp=True) / 1000)
del xb
m = m.cpu(); torch.set_num_threads(4)
xc = torch.randn(1, 3, S, S)
with torch.no_grad():
    for _ in range(5):
        m(xc)
    t0 = time.time()
    for _ in range(30):
        m(xc)
ms_cpu = (time.time() - t0) / 30 * 1000
npar = sum(p.numel() for p in m.parameters())
print(json.dumps({'npar': npar, 'gflops': gflops, 'ms_fp32': ms_fp32, 'ms_bf16': ms_bf16, 'img_s': img_s,
                  'mem_mb': mem_mb, 'ms_cpu': ms_cpu}))
