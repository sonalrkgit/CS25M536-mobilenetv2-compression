import argparse, copy
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from torch.utils.data import DataLoader
from data import test_tf
from model import get_model

# ---------------- bit packing: store k-bit integers densely in uint8 ----------------
def pack_bits(u, b):
    """u: 1-D int64 tensor, values in [0, 2^b - 1]  ->  uint8 tensor, b bits per value."""
    shifts = torch.arange(b - 1, -1, -1, device=u.device)
    bits = ((u.unsqueeze(1) >> shifts) & 1).flatten()
    pad = (-bits.numel()) % 8
    if pad:
        bits = torch.cat([bits, bits.new_zeros(pad)])
    w = 2 ** torch.arange(7, -1, -1, device=u.device)
    return (bits.view(-1, 8) * w).sum(1).to(torch.uint8)

def unpack_bits(packed, n, b):
    shifts = torch.arange(7, -1, -1, device=packed.device)
    bits = ((packed.unsqueeze(1) >> shifts) & 1).flatten()[: n * b].view(n, b)
    w = 2 ** torch.arange(b - 1, -1, -1, device=packed.device)
    return (bits * w).sum(1)

# ---------------- weight quantization (per output channel, symmetric) ----------------
def quantize_per_channel(w, bits, search=True):
    qmax = 2 ** (bits - 1) - 1
    shp = (-1,) + (1,) * (w.dim() - 1)
    maxv = w.reshape(w.shape[0], -1).abs().max(1)[0].clamp(min=1e-8)
    ratios = [float(r) for r in torch.linspace(0.5, 1.0, 11)] if search else [1.0]
    best_err = torch.full_like(maxv, float('inf'))
    best_scale = maxv / qmax
    for r in ratios:
        s = maxv * r / qmax
        q = torch.round(w / s.view(shp)).clamp(-qmax, qmax)
        err = ((q * s.view(shp) - w) ** 2).reshape(w.shape[0], -1).sum(1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, s, best_scale)
    q = torch.round(w / best_scale.view(shp)).clamp(-qmax, qmax)
    return q, best_scale

class PackedWeight(nn.Module):
    """Holds a layer's weights as packed integers + one FP32 scale per output channel."""
    def __init__(self, q, scale, bits):
        super().__init__()
        self.bits, self.qmax, self.shape = bits, 2 ** (bits - 1) - 1, tuple(q.shape)
        self.register_buffer('packed', pack_bits((q + self.qmax).long().flatten(), bits))
        self.register_buffer('scale', scale.float().clone())

    def dequant(self):
        n = 1
        for d in self.shape: n *= d
        q = unpack_bits(self.packed, n, self.bits).float() - self.qmax
        return q.view(self.shape) * self.scale.view((-1,) + (1,) * (len(self.shape) - 1))

class QConv2d(nn.Module):
    def __init__(self, conv, bits, search=True):
        super().__init__()
        q, s = quantize_per_channel(conv.weight.data, bits, search)
        self.pw = PackedWeight(q, s, bits)
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.data.clone(), requires_grad=False)
        self.stride, self.padding, self.dilation, self.groups = conv.stride, conv.padding, conv.dilation, conv.groups
    def forward(self, x):
        return F.conv2d(x, self.pw.dequant(), self.bias, self.stride, self.padding, self.dilation, self.groups)

class QLinear(nn.Module):
    def __init__(self, lin, bits, search=True):
        super().__init__()
        q, s = quantize_per_channel(lin.weight.data, bits, search)
        self.pw = PackedWeight(q, s, bits)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.clone(), requires_grad=False)
    def forward(self, x):
        return F.linear(x, self.pw.dequant(), self.bias)

# ---------------- activation quantization (after every ReLU6) ----------------
class ActQuant(nn.Module):
    def __init__(self, bits, pct=100.0):
        super().__init__()
        self.bits, self.pct, self.qmax = bits, pct, 2 ** bits - 1
        self.register_buffer('scale', torch.tensor(6.0 / self.qmax))
        self.calibrating, self.obs, self.numel = False, [], 0
    def forward(self, x):
        self.numel = x[0].numel()                       # activation size per image (for reporting)
        if self.calibrating:
            flat = x.detach().flatten()
            if flat.numel() > 1_000_000:
                flat = flat[torch.randint(0, flat.numel(), (1_000_000,), device=flat.device)]
            hi = flat.max() if self.pct >= 100 else torch.quantile(flat.float(), self.pct / 100)
            self.obs.append(hi.item())
            return x
        q = torch.clamp(torch.round(x / self.scale), 0, self.qmax)
        return q * self.scale

# ---------------- convert a model ----------------
def _convert(module, cfg, prefix=''):
    for name, child in module.named_children():
        full = f'{prefix}.{name}' if prefix else name
        if isinstance(child, nn.Conv2d):
            if full == 'features.0.0':
                bits = cfg['first_last_bits']                       # first conv
            elif child.groups == child.in_channels and child.groups > 1:
                bits = cfg['dw_bits'] or cfg['w_bits']              # depthwise conv
            else:
                bits = cfg['w_bits']                                # pointwise convs
            setattr(module, name, QConv2d(child, bits, cfg['clip_search']))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QLinear(child, cfg['first_last_bits'], cfg['clip_search']))
        elif isinstance(child, nn.ReLU6):
            setattr(module, name, nn.Sequential(child, ActQuant(cfg['a_bits'], cfg['act_pct'])))
        else:
            _convert(child, cfg, full)

def compress_model(model, cfg):
    qm = copy.deepcopy(model)
    _convert(qm, cfg)
    return qm.eval()

@torch.no_grad()
def calibrate(qmodel, loader, n_batches, device):
    aqs = [m for m in qmodel.modules() if isinstance(m, ActQuant)]
    for m in aqs: m.calibrating, m.obs = True, []
    qmodel.eval()
    for i, (x, _) in enumerate(loader):
        if i >= n_batches: break
        qmodel(x.to(device))
    for m in aqs:
        m.scale.fill_(max(min(max(m.obs), 6.0), 1e-4) / m.qmax)
        m.calibrating = False

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval(); correct = n = 0
    for x, y in loader:
        out = model(x.to(device))
        correct += (out.argmax(1).cpu() == y).sum().item(); n += len(y)
    return 100 * correct / n

# ---------------- storage accounting (real tensor sizes) ----------------
def tb(t): return t.numel() * t.element_size()

def memory_report(model):
    r = dict(packed_weights=0, weight_scales=0, fp32_weights=0, bias=0,
             batchnorm=0, act_scales=0, metadata=0)
    for m in model.modules():
        if isinstance(m, PackedWeight):
            r['packed_weights'] += tb(m.packed); r['weight_scales'] += tb(m.scale)
            r['metadata'] += 16            # estimate: bit-width + shape per layer
        elif isinstance(m, (QConv2d, QLinear)):
            if m.bias is not None: r['bias'] += tb(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            r['fp32_weights'] += tb(m.weight)
            if m.bias is not None: r['bias'] += tb(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            for t in list(m.parameters(recurse=False)) + list(m.buffers(recurse=False)):
                r['batchnorm'] += tb(t)
        elif isinstance(m, ActQuant):
            r['act_scales'] += tb(m.scale)
    r['total'] = sum(r.values())
    return r

def activation_report(qmodel, a_bits):
    n = [m.numel for m in qmodel.modules() if isinstance(m, ActQuant)]
    fp32 = sum(n) * 4
    q = sum(n) * a_bits / 8 + 4 * len(n)
    return dict(layers=len(n), fp32_bytes=fp32, quant_bytes=q, ratio=fp32 / q,
                peak_fp32=max(n) * 4, peak_quant=max(n) * a_bits / 8)

# ---------------- one full compression experiment ----------------
def run_compression(weights, w_bits, a_bits, first_last_bits=8, dw_bits=None,
                    act_pct=100.0, clip_search=True, calib_batches=10, device='cuda'):
    cfg = dict(w_bits=w_bits, a_bits=a_bits, first_last_bits=first_last_bits,
               dw_bits=dw_bits, act_pct=act_pct, clip_search=clip_search)
    tr = torchvision.datasets.CIFAR10('./data', True, download=True, transform=test_tf)
    te = torchvision.datasets.CIFAR10('./data', False, download=True, transform=test_tf)
    cal_loader = DataLoader(tr, 256, shuffle=False, num_workers=2)
    te_loader = DataLoader(te, 256, shuffle=False, num_workers=2)

    base = get_model().to(device)
    base.load_state_dict(torch.load(weights, map_location=device)); base.eval()
    base_acc, base_rep = accuracy(base, te_loader, device), memory_report(base)

    qm = compress_model(base, cfg)
    calibrate(qm, cal_loader, calib_batches, device)
    q_acc, q_rep = accuracy(qm, te_loader, device), memory_report(qm)
    act = activation_report(qm, a_bits)

    q_w = q_rep['packed_weights'] + q_rep['weight_scales'] + q_rep['fp32_weights']
    return dict(weight_quant_bits=w_bits, activation_quant_bits=a_bits,
                baseline_acc=base_acc, quantized_acc=q_acc,
                baseline_size_mb=base_rep['total'] / 1e6, model_size_mb=q_rep['total'] / 1e6,
                compression_ratio=base_rep['total'] / q_rep['total'],
                weight_ratio=base_rep['fp32_weights'] / q_w,
                act_ratio=act['ratio'], act_report=act, mem_report=q_rep, base_report=base_rep)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='mbv2_baseline.pt')
    p.add_argument('--w_bits', type=int, default=8)
    p.add_argument('--a_bits', type=int, default=8)
    p.add_argument('--first_last_bits', type=int, default=8)
    p.add_argument('--dw_bits', type=int, default=None)
    p.add_argument('--act_pct', type=float, default=100.0)
    p.add_argument('--no_clip_search', action='store_true')
    a = p.parse_args()
    res = run_compression(a.weights, a.w_bits, a.a_bits, a.first_last_bits, a.dw_bits,
                          a.act_pct, not a.no_clip_search)
    for k, v in res.items():
        print(k, ':', v)
