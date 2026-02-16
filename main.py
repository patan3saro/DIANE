"""
NP-TTA toy implementation (CIFAR-10) — compact refactored version.
Maintains all features: teacher distillation, BN banks, SPRT, proximal updates.
"""

from __future__ import annotations
import argparse, copy, csv, math, os, random, tarfile, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- PATCH: RoTTA/NP-TTA sanity defaults ---
DEBUG_MEM_EVERY = 50          # stampa memoria 1 volta ogni N batch (non per-sample)
ROTTTA_FREQ_FAST = 8          # update RoTTA più frequente per stream corto
LOSS_HARD_CAP = 50.0          # se loss explode, skip step

# ============ Utils ============
def ensure_cifar10c(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    labels = os.path.join(root, "labels.npy")
    if os.path.exists(labels): return
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    tar_path = os.path.join(root, "CIFAR-10-C.tar")
    print(f"[C10-C] downloading: {url}")
    urllib.request.urlretrieve(url, tar_path)
    print(f"[C10-C] extracting: {tar_path}")
    with tarfile.open(tar_path, "r") as tf: tf.extractall(path=os.path.dirname(root))
    if not os.path.exists(labels): raise FileNotFoundError(f"labels.npy not found in {root}")

def unsharp_mask(x: torch.Tensor, sigma: float = 1.0, amount: float = 0.7) -> torch.Tensor:
    blur = torchvision.transforms.functional.gaussian_blur(x, kernel_size=5, sigma=sigma)
    return (x + amount * (x - blur)).clamp(0.0, 1.0)

def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed); torch.manual_seed(worker_seed)

def make_stream_loader(test_set, batch_size: int, num_workers: int, seed: int, shuffle: bool = True) -> DataLoader:
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                     pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=g, persistent_workers=False)

def next_batch(it, loader: DataLoader):
    try: return next(it), it
    except StopIteration: it = iter(loader); return next(it), it

class CUDATimer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.start_evt = torch.cuda.Event(enable_timing=True) if self.enabled else None
        self.end_evt = torch.cuda.Event(enable_timing=True) if self.enabled else None
    def start(self) -> None:
        if self.enabled: self.start_evt.record()
    def stop_ms(self) -> float:
        if not self.enabled: return 0.0
        self.end_evt.record(); torch.cuda.synchronize()
        return float(self.start_evt.elapsed_time(self.end_evt))

def peak_mem_mb() -> float:
    return float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()

def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps); return -(p * p.log()).sum(dim=1)

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0

@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(momentum).add_(sp.data, alpha=(1.0 - momentum))

def freeze_bn_stats(m: nn.Module) -> None:
    if isinstance(m, nn.BatchNorm2d): m.eval(); m.track_running_stats = True

def bn_buffers_state(m: nn.Module) -> Dict[str, torch.Tensor]:
    sd = {}
    for name, mod in m.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            sd[f"{name}.running_mean"] = mod.running_mean.detach().clone()
            sd[f"{name}.running_var"] = mod.running_var.detach().clone()
            sd[f"{name}.num_batches_tracked"] = mod.num_batches_tracked.detach().clone()
    return sd

@torch.no_grad()
def load_bn_buffers(m: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    for name, mod in m.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.copy_(sd[f"{name}.running_mean"].to(mod.running_mean.device))
            mod.running_var.copy_(sd[f"{name}.running_var"].to(mod.running_var.device))
            mod.num_batches_tracked.copy_(sd[f"{name}.num_batches_tracked"].to(mod.num_batches_tracked.device))



# ============ RoTTA: RobustBN + Memory ============
class RobustBN2d(nn.BatchNorm2d):
    """
    BN che permette un update esplicito delle running stats usando le stats del batch corrente.
    Forward usa sempre running stats (stabile), mentre l'update lo fai con robustbn_update(...)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(
            x, self.running_mean, self.running_var,
            self.weight, self.bias,
            False,  # training=False: usa running stats
            0.0,
            self.eps
        )

@torch.no_grad()
def replace_bn_with_robustbn(module: nn.Module) -> None:
    """Sostituisce ricorsivamente nn.BatchNorm2d -> RobustBN2d copiando pesi e running stats."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, RobustBN2d):
            # crea su stesso device/dtype del vecchio BN
            dev = child.running_mean.device
            dtype = child.running_mean.dtype

            rb = RobustBN2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats
            ).to(device=dev, dtype=dtype)

            # copia parametri/BN buffers
            if child.affine:
                rb.weight.data.copy_(child.weight.data.to(dev))
                rb.bias.data.copy_(child.bias.data.to(dev))
            rb.running_mean.data.copy_(child.running_mean.data.to(dev))
            rb.running_var.data.copy_(child.running_var.data.to(dev))
            rb.num_batches_tracked.data.copy_(child.num_batches_tracked.data.to(dev))

            setattr(module, name, rb)

        else:
            replace_bn_with_robustbn(child)

@torch.no_grad()
def robustbn_update_from_input(model: nn.Module, x: torch.Tensor, mom: float) -> None:
    """
    Esegue un forward con hook per aggiornare running_mean/var di RobustBN2d con stats del batch.
    mom piccolo (es 0.01) = update lento e stabile.
    """
    handles = []

    def make_hook(m: RobustBN2d):
        def hook(mod, inp, out):
            z = inp[0]
            mean = z.mean(dim=(0, 2, 3))
            var = z.var(dim=(0, 2, 3), unbiased=False)
            mod.running_mean.mul_(1 - mom).add_(mom * mean)
            mod.running_var.mul_(1 - mom).add_(mom * var)
        return hook

    for m in model.modules():
        if isinstance(m, RobustBN2d):
            handles.append(m.register_forward_hook(make_hook(m)))

    # forward “dummy” solo per triggerare gli hook
    _ = model(x)

    for h in handles:
        h.remove()

from dataclasses import dataclass

@dataclass
class MemItem:
    x: torch.Tensor  # (C,H,W) su CPU
    y: int
    unc: float
    t: int

def unc_from_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (entropy_from_probs(p, eps=eps) / math.log(p.size(1))).clamp(0.0, 1.0)
def strong_aug(x: torch.Tensor) -> torch.Tensor:
    return make_teacher_views_general(x, 2)[1]  # 1 strong-ish view cheap (riusa la tua pipeline)


class CSTUMemory:
    """
    Memory semplice: capacity fissa, eviction bilanciata per classe "peggiore" (over-represented).
    Timeliness: peso exp(-alpha * age)
    Uncertainty: unc in [0,1], certainty = 1-unc
    """
    def __init__(self, capacity: int, num_classes: int):
        self.capacity = int(capacity)
        self.num_classes = int(num_classes)
        self.items: List[MemItem] = []
        self.time = 0

    def __len__(self) -> int:
        return len(self.items)

    def step_time(self) -> None:
        self.time += 1

    def _counts(self) -> List[int]:
        c = [0] * self.num_classes
        for it in self.items:
            c[it.y] += 1
        return c

    @staticmethod
    def w_timeliness(age: int, alpha: float) -> float:
        return math.exp(-alpha * float(age))

    def score_keep(self, it: MemItem, alpha: float, lambda_t: float, lambda_u: float) -> float:
        age = self.time - it.t
        wt = self.w_timeliness(age, alpha)
        certainty = 1.0 - float(it.unc)
        return lambda_t * wt + lambda_u * certainty

    # === PATCH 2/3: Memory time = per-batch (non per-sample) ===
    # (A) SOSTITUISCI tutto CSTUMemory.add(...) con questa versione:

    def add(self, x_cpu_chw: torch.Tensor, y: int, unc: float, alpha: float, lambda_t: float, lambda_u: float,
            step_time: bool = True) -> None:
        if step_time:
            self.step_time()
        item = MemItem(x=x_cpu_chw.detach().cpu(), y=int(y), unc=float(unc), t=self.time)

        if len(self.items) < self.capacity:
            self.items.append(item)
            return

        counts = self._counts()
        target_cls = int(np.argmax(counts))
        cand = [i for i, it in enumerate(self.items) if it.y == target_cls]
        if not cand:
            cand = list(range(len(self.items)))

        worst_i, worst_s = cand[0], float("inf")
        for i in cand:
            s = self.score_keep(self.items[i], alpha, lambda_t, lambda_u)
            if s < worst_s:
                worst_s = s
                worst_i = i

        self.items[worst_i] = item

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = min(int(batch_size), len(self.items))
        idxs = random.sample(range(len(self.items)), bs)
        xs = torch.stack([self.items[i].x for i in idxs], dim=0).to(device, non_blocking=True)
        ys = torch.tensor([self.items[i].y for i in idxs], device=device, dtype=torch.long)
        ages = torch.tensor([self.time - self.items[i].t for i in idxs], device=device, dtype=torch.float32)
        uncs = torch.tensor([self.items[i].unc for i in idxs], device=device, dtype=torch.float32)
        return xs, ys, ages, uncs


# ============ Corruptions ============
def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return (x + sigma * torch.randn_like(x)).clamp(0.0, 1.0)

def gaussian_blur_batch(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    return torchvision.transforms.functional.gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)

def jpeg_like(x: torch.Tensor, strength: float) -> torch.Tensor:
    levels = int(2 + (1.0 - strength) * 30)
    xq = torch.round(x * levels) / float(levels)
    return (xq + 0.02 * strength * torch.randn_like(x)).clamp(0.0, 1.0)

def apply_corruption(x: torch.Tensor, kind: str, severity: int) -> torch.Tensor:
    if kind == "clean": return x
    if kind == "gauss":
        sigma = {1: 0.04, 2: 0.08, 3: 0.12, 4: 0.18, 5: 0.25}[severity]
        return add_gaussian_noise(x, sigma)
    if kind == "blur":
        sigma = {1: 0.6, 2: 0.9, 3: 1.2, 4: 1.6, 5: 2.0}[severity]
        ks = 5 if severity <= 3 else 7
        return gaussian_blur_batch(x, kernel_size=ks, sigma=sigma)
    if kind == "jpeg":
        strength = {1: 0.15, 2: 0.25, 3: 0.35, 4: 0.5, 5: 0.7}[severity]
        return jpeg_like(x, strength=strength)
    raise ValueError(f"Unknown corruption kind: {kind}")

class CIFAR10C(torch.utils.data.Dataset):
    def __init__(self, root: str, corruption: str, severity: int, transform=None):
        assert 1 <= severity <= 5
        self.root, self.corruption, self.severity, self.transform = Path(root), corruption, severity, transform
        x_path, y_path = self.root / f"{corruption}.npy", self.root / "labels.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10-C files in {self.root}")
        x_all, y_all = np.load(x_path), np.load(y_path)
        start, end = (severity - 1) * 10000, severity * 10000
        self.x, self.y = x_all[start:end], y_all[start:end]
    def __len__(self): return int(self.y.shape[0])
    def __getitem__(self, idx: int):
        img, y = self.x[idx], int(self.y[idx])
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        if self.transform is not None: x = self.transform(x)
        return x, torch.tensor(y, dtype=torch.long)

def c10c_kind_tag(corruption: str) -> str:
    c = corruption.lower()
    if "blur" in c: return "blur"
    if c in ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"]: return "gauss"
    if "jpeg" in c: return "jpeg"
    return "other"

# ============ Model ============
class ResNet18WithAdapter(nn.Module):
    def __init__(self, num_classes: int = 10, adapter_rank: int = 128):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.in_adapter = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
        nn.init.zeros_(self.in_adapter.weight)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.adapter = nn.Sequential(nn.Linear(feat_dim, adapter_rank, bias=False), nn.ReLU(inplace=True),
                                     nn.Linear(adapter_rank, feat_dim, bias=False))
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.reset_adapter_identity()
    def reset_adapter_identity(self) -> None:
        nn.init.kaiming_normal_(self.adapter[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.in_adapter.weight)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + self.in_adapter(x)).clamp(0.0, 1.0)
        h = self.backbone(x)
        h = h + self.adapter(h)
        return self.classifier(h)

def freeze_all_except_adapter(model: ResNet18WithAdapter) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = ("adapter" in name) or ("in_adapter" in name)

@torch.no_grad()
def adapter_decay_step(model: ResNet18WithAdapter, decay: float) -> None:
    if decay <= 0.0: return
    for p in model.adapter.parameters(): p.mul_(1.0 - float(decay))

@torch.no_grad()
def adapter_soft_restore_to_state(model: ResNet18WithAdapter, ref_sd: Dict[str, torch.Tensor], rho: float) -> None:
    if rho <= 0.0: return
    cur = model.adapter.state_dict()
    for k in cur.keys(): cur[k].mul_(1.0 - rho).add_(ref_sd[k].to(cur[k].device), alpha=rho)
    model.adapter.load_state_dict(cur, strict=True)

@torch.no_grad()
def stochastic_restore_backbone(model: ResNet18WithAdapter, source: ResNet18WithAdapter, prob: float) -> None:
    if prob <= 0.0: return
    msd, ssd = model.state_dict(), source.state_dict()
    for k, w in msd.items():
        if "adapter" in k or k not in ssd: continue
        w0 = ssd[k].to(w.device)
        if (not torch.is_floating_point(w)) or w.ndim == 0:
            if random.random() < prob: w.copy_(w0)
            continue
        mask = torch.rand_like(w) < prob
        w[mask] = w0[mask]

# ============ Teacher views ============
def make_teacher_views_blur(x: torch.Tensor, n_views: int) -> torch.Tensor:
    if n_views <= 1: return x.unsqueeze(0)
    views, B, C, H, W = [], *x.shape
    for i in range(n_views):
        v = x
        if i > 0:
            if random.random() < 0.5: v = torch.flip(v, dims=[3])
            if random.random() < 0.3:
                s = 28
                v = F.interpolate(v, size=(s, s), mode="bilinear", align_corners=False)
                v = F.interpolate(v, size=(H, W), mode="bilinear", align_corners=False)
            if random.random() < 0.3: v = unsharp_mask(v, sigma=1.2, amount=0.5)
        views.append(v.clamp(0.0, 1.0))
    return torch.stack(views, dim=0)

def make_teacher_views_general(x: torch.Tensor, n_views: int) -> torch.Tensor:
    if n_views <= 1: return x.unsqueeze(0)
    if n_views <= 1: return x.unsqueeze(0)
    views, B, C, H, W = [], *x.shape
    # Aumenta diversità tra view per ridurre correlazione
    for _ in range(n_views):
        v = x
        if random.random() < 0.5: v = torch.flip(v, dims=[3])
        v = (v + 0.02 * torch.randn(B, 1, 1, 1, device=v.device)).clamp(0.0, 1.0)
        if random.random() < 0.35: v = (v + 0.03 * torch.randn_like(v)).clamp(0.0, 1.0)  # più noise
        if random.random() < 0.25: v = torchvision.transforms.functional.gaussian_blur(v, kernel_size=5, sigma=0.6)
        if random.random() < 0.25:
            s = random.choice([24, 28])
            v = F.interpolate(v, size=(s, s), mode="bilinear", align_corners=False)
            v = F.interpolate(v, size=(H, W), mode="bilinear", align_corners=False)
        if random.random() < 0.25:
            levels = 16
            v = (torch.round(v * levels) / float(levels)).clamp(0.0, 1.0)
        views.append(v)
    return torch.stack(views, dim=0)

@torch.no_grad()
def teacher_aggregate(teacher: nn.Module, x: torch.Tensor, n_views: int, amp: bool, view_mode: str, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xv = make_teacher_views_blur(x, n_views) if view_mode == "blur" else make_teacher_views_general(x, n_views)
    V, B, C, H, W = xv.shape
    probs_list = []
    with torch.inference_mode():
        for v in range(V):
            with torch.amp.autocast("cuda", enabled=amp):
                logits = teacher(xv[v])
                probs = F.softmax(logits, dim=1)
            probs_list.append(probs)
    probs_v = torch.stack(probs_list, dim=0)
    p_t = probs_v.mean(dim=0)
    conf, _ = p_t.max(dim=1)
    ent = entropy_from_probs(p_t, eps=eps)
    log_probs_v, log_pt = probs_v.clamp_min(eps).log(), p_t.clamp_min(eps).log()
    kl_vb = (probs_v * (log_probs_v - log_pt.unsqueeze(0))).sum(dim=2)
    disagree = kl_vb.mean(dim=0)
    return p_t, conf, ent, disagree

# ============ Config ============
@dataclass
class NPTTAConfig:
    teacher_momentum: float = 0.9995
    entropy_thresh: float = 1.2
    # --- RoTTA core ---
    rotta_enable: bool = False
    rotta_memory_size: int = 64
    rotta_update_frequency: int = 64
    rotta_batch_size: int = 64
    rotta_steps: int = 1
    rotta_alpha: float = 0.05
    rotta_lambda_t: float = 1.0
    rotta_lambda_u: float = 1.0
    rotta_robustbn_mom: float = 0.01
    rotta_train_bn_affine_only: bool = False  # False=resti adapter-only (consigliato col tuo setup)

    disagree_thresh: float = 0.03
    confidence_min: float = 0.6
    conf_frac_min: float = 0.2
    lr: float = 3e-3
    adapt_steps: int = 1
    aug_views: int = 8
    adapter_decay_when_clean: float = 0.0
    restore_prob_when_clean: float = 0.0
    restore_prob_when_shift: float = 0.0005
    blur_lr_scale: float = 0.2
    blur_loss: str = "distill"
    blur_im_lambda: float = 1.0
    blur_distill_temp: float = 0.7
    blur_distill_w: float = 1.0
    in_adapter_lr_scale: float = 1.0
    blur_update_teacher: bool = False
    blur_bn_bank: bool = True
    blur_phys_w: float = 1.0
    blur_disable_phys_when_distill: bool = True
    blur_entropy_scale: float = 0.7  # lower entropy threshold for blur gating (more adaptation)
    use_sprt_for_blur: bool = True
    sprt_alpha: float = 0.05
    sprt_beta: float = 0.10
    sprt_mu0: float = 0.03
    sprt_mu1: float = 0.08
    sprt_sigma: float = 0.03
    prox_w: float = 0.05
    prox_w_id: float = 0.10   # prox verso identità/source per proteggere clean


def build_cfg_from_args(args: argparse.Namespace, cfg_overrides: Optional[dict] = None) -> NPTTAConfig:
    ov = cfg_overrides or {}
    return NPTTAConfig(
        entropy_thresh=ov.get("entropy_thresh", args.entropy_thresh),
        # RoTTA core
        # (B) in build_cfg_from_args, SOSTITUISCI la riga rotta_enable=... con:
        rotta_enable=ov.get("rotta_enable", args.rotta_enable),

        rotta_memory_size=ov.get("rotta_memory_size", args.rotta_memory_size),
        rotta_update_frequency=ov.get("rotta_update_frequency", args.rotta_update_frequency),
        rotta_batch_size=ov.get("rotta_batch_size", args.rotta_batch_size),
        rotta_steps=ov.get("rotta_steps", args.rotta_steps),
        rotta_alpha=ov.get("rotta_alpha", args.rotta_alpha),
        rotta_lambda_t=ov.get("rotta_lambda_t", args.rotta_lambda_t),
        rotta_lambda_u=ov.get("rotta_lambda_u", args.rotta_lambda_u),
        rotta_robustbn_mom=ov.get("rotta_robustbn_mom", args.rotta_robustbn_mom),
        rotta_train_bn_affine_only=ov.get("rotta_train_bn_affine_only", args.rotta_train_bn_affine_only),

        disagree_thresh=ov.get("disagree_thresh", args.disagree_thresh),
        lr=ov.get("tta_lr", args.tta_lr),
        teacher_momentum=ov.get("teacher_momentum", args.teacher_momentum),
        confidence_min=ov.get("confidence_min", args.confidence_min),
        conf_frac_min=ov.get("conf_frac_min", args.conf_frac_min),
        adapt_steps=ov.get("adapt_steps", args.adapt_steps),
        aug_views=ov.get("aug_views", args.aug_views),
        adapter_decay_when_clean=ov.get("adapter_decay_when_clean", args.adapter_decay_when_clean),
        blur_loss=ov.get("blur_loss", args.blur_loss),
        blur_im_lambda=ov.get("blur_im_lambda", args.blur_im_lambda),
        blur_distill_temp=ov.get("blur_distill_temp", args.blur_distill_temp),
        blur_distill_w=ov.get("blur_distill_w", args.blur_distill_w),
        in_adapter_lr_scale=ov.get("in_adapter_lr_scale", args.in_adapter_lr_scale),
        blur_update_teacher=ov.get("blur_update_teacher", args.blur_update_teacher),
        blur_lr_scale=ov.get("blur_lr_scale", args.blur_lr_scale),
        blur_bn_bank=ov.get("blur_bn_bank", (not args.no_blur_bn_bank)),
        blur_phys_w=ov.get("blur_phys_w", args.blur_phys_w),
        blur_disable_phys_when_distill=ov.get("blur_disable_phys_when_distill", (not args.blur_keep_phys_with_distill)),
        blur_entropy_scale=ov.get("blur_entropy_scale", args.blur_entropy_scale),
        use_sprt_for_blur=args.use_sprt_for_blur,
        sprt_alpha=args.sprt_alpha,
        sprt_beta=args.sprt_beta,
        sprt_mu0=args.sprt_mu0,
        sprt_mu1=args.sprt_mu1,
        sprt_sigma=args.sprt_sigma,
        prox_w=args.prox_w,
        prox_w_id=args.prox_w_id,
    )

# ============ Training ============
def train_source(model: nn.Module, device: torch.device, epochs: int, batch_size: int, num_workers: int, lr: float, amp: bool, out_path: str) -> None:
    model.train()
    if hasattr(model, "in_adapter"):
        nn.init.zeros_(model.in_adapter.weight)
        for p in model.in_adapter.parameters(): p.requires_grad = False
    transform_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    transform_test = T.ToTensor()
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs * len(train_loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    @torch.no_grad()
    def eval_once() -> float:
        model.eval(); accs = []
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            accs.append(accuracy(model(x), y))
        model.train(); return float(sum(accs) / len(accs))
    for ep in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"train ep {ep}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            pbar.set_postfix(loss=float(loss.item()))
        acc = eval_once()
        print(f"[source] epoch {ep}: test_acc={acc:.4f}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict()}, out_path)
    print(f"Saved source checkpoint to: {out_path}")

# ============ Stream helpers ============
def build_schedule(stream_len_batches: int) -> Tuple[Tuple[str, int, int], ...]:
    seg = max(1, stream_len_batches // 6)
    return (("clean", 0, seg), ("gauss", 3, seg), ("clean", 0, seg), ("blur", 3, seg), ("clean", 0, seg), ("gauss", 3, stream_len_batches - 5 * seg))

def make_stream_len_and_schedule(args: argparse.Namespace) -> Tuple[int, Tuple[Tuple[str, int, int], ...]]:
    stream_len_batches = max(1, args.stream_len // args.batch_size)
    return stream_len_batches, build_schedule(stream_len_batches)

def avg_clean(seg_acc: Dict[str, float]) -> float:
    vals = [v for k, v in seg_acc.items() if k.split(":")[1].startswith("clean")]
    return mean(vals)

def last_seg_key(schedule: Tuple[Tuple[str, int, int], ...], kind: str, sev: int) -> str:
    idx = max(i for i, (k, s, _) in enumerate(schedule) if k == kind and s == sev)
    return f"{idx}:{kind}{sev}"

# ============ Baseline eval ============
@torch.no_grad()
def eval_c10c_baseline(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); accs, it = [], iter(loader)
    for _ in range(len(loader)):
        (x, y), it = next_batch(it, loader)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        accs.append(accuracy(model(x), y))
    return mean(accs)

def eval_c10c_nptta(model: ResNet18WithAdapter, source_model: ResNet18WithAdapter, loader: DataLoader, device: torch.device, cfg: NPTTAConfig, amp: bool, kind_tag: str, sev: int) -> float:
    schedule = ((kind_tag, sev, len(loader)),)
    seg = eval_stream_nptta(model, source_model, loader, device, cfg, schedule, amp, False, lambda x, k, s: x)
    return seg[f"0:{kind_tag}{sev}"]

@torch.no_grad()
def eval_stream_baseline(model: nn.Module, loader: DataLoader, device: torch.device, schedule: Tuple[Tuple[str, int, int], ...]) -> Dict[str, float]:
    model.eval(); seg_acc, it = {}, iter(loader)
    for si, (kind, sev, nb) in enumerate(schedule):
        accs = []
        for _ in range(nb):
            (x, y), it = next_batch(it, loader)
            x = apply_corruption(x.to(device, non_blocking=True), kind, sev)
            y = y.to(device, non_blocking=True)
            accs.append(accuracy(model(x), y))
        seg_acc[f"{si}:{kind}{sev}"] = mean(accs)
    return seg_acc

# ============ NP-TTA eval (condensed) ============
def eval_stream_nptta(model: ResNet18WithAdapter, source_model: ResNet18WithAdapter, loader: DataLoader, device: torch.device,
                      cfg: NPTTAConfig, schedule: Tuple[Tuple[str, int, int], ...], amp: bool, debug: bool = True, corrupt_fn=apply_corruption) -> Dict[str, float]:
    model.eval()
    # --- RoTTA init ---
    if cfg.rotta_enable:
        replace_bn_with_robustbn(model.backbone)
        model.backbone.to(device)  # safety: riallinea moduli nuovi
        if cfg.rotta_enable:
            nrb = 0
            devs = set()
            for m in model.backbone.modules():
                if isinstance(m, RobustBN2d):
                    nrb += 1
                    devs.add(str(m.running_mean.device))
            print(f"[SANITY][RobustBN] n={nrb} devices={sorted(list(devs))}")

        # memory
        rotta_mem = CSTUMemory(capacity=cfg.rotta_memory_size, num_classes=10)
    else:
        rotta_mem = None

    # Memoria running per teacher disagreement - aiuta a filtrare outlier
    running_disagree_mean = 0.03  # valore iniziale
    running_disagree_std = 0.02
    disagree_history_gen = []
    disagree_history_blur = []
    # --- CoTTA-style: keep EMA teacher(s) separate from student ---

    teacher_main = copy.deepcopy(model).eval()
    teacher_blur = copy.deepcopy(model).eval()
    for p in teacher_main.parameters(): p.requires_grad = False
    for p in teacher_blur.parameters(): p.requires_grad = False

    teacher = teacher_main
    teacher_blur_inited = False

    freeze_all_except_adapter(model)
    model.backbone.eval(); model.classifier.eval()

    opt = torch.optim.AdamW([{"params": list(model.adapter.parameters()), "lr": cfg.lr},
                             {"params": list(model.in_adapter.parameters()), "lr": 0.0}], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    adapter_identity = copy.deepcopy(model.adapter.state_dict())
    in_identity = copy.deepcopy(model.in_adapter.state_dict())
    adapter_main_state, in_main_state = copy.deepcopy(adapter_identity), copy.deepcopy(in_identity)
    adapter_blur_state, in_blur_state = copy.deepcopy(adapter_identity), copy.deepcopy(in_identity)
    bn_main_state = bn_buffers_state(model.backbone)
    bn_blur_state = copy.deepcopy(bn_main_state)

    seg_acc, it = {}, iter(loader)
    A = math.log((1.0 - cfg.sprt_beta) / max(1e-12, cfg.sprt_alpha)) if cfg.use_sprt_for_blur else 0.0
    B = math.log(cfg.sprt_beta / max(1e-12, (1.0 - cfg.sprt_alpha))) if cfg.use_sprt_for_blur else 0.0
    llr_blur, sprt_state, total_batches, adapted_batches = 0.0, "unknown", 0, 0
    all_ms, adapt_ms, prev_kind = [], [], None

    for si, (kind, sev, nb) in enumerate(schedule):
        timer = CUDATimer(enabled=True)
        gate_true, conf_true, updates = 0, 0, 0
        loss_phys_ms, mask_sz_ms, dx_ms, ent_ms, dis_ms, conf_ms = [], [], [], [], [], []
        conf_fs, loss_klds, ent_ss, loss_ts, grad_ns = [], [], [], [], []
        seg_batch_ms, seg_adapt_ms = [], []
        printed_blur_cfg, printed_blur_branch = False, False
        seg_adapted = 0

        if cfg.use_sprt_for_blur and prev_kind == "blur" and kind != "blur":
            llr_blur, sprt_state = 0.0, "unknown"
        prev_kind = kind

        # BN bank switch for blur
        if kind == "blur" and cfg.blur_bn_bank:
            adapter_main_state, in_main_state = copy.deepcopy(model.adapter.state_dict()), copy.deepcopy(model.in_adapter.state_dict())
            bn_main_state = bn_buffers_state(model.backbone)
            model.adapter.load_state_dict(adapter_blur_state, strict=True)
            model.in_adapter.load_state_dict(in_blur_state, strict=True)
            load_bn_buffers(model.backbone, bn_blur_state)

            # teacher for blur: init once; if blur teacher updates are disabled, just sync at segment entry
            if not teacher_blur_inited:
                teacher_blur.load_state_dict(model.state_dict(), strict=True)
                teacher_blur_inited = True

            teacher = teacher_blur

        if debug and kind == "blur" and not printed_blur_cfg:
            print(f"[DEBUG][enter blur] si={si} sev={sev} cfg.blur_loss={cfg.blur_loss} use_distill={(cfg.blur_loss=='distill')} "
                  f"teacher_x={'unsharp' if (cfg.blur_loss=='distill') else 'blur'} bn_bank={cfg.blur_bn_bank} "
                  f"phys_w={cfg.blur_phys_w} disable_phys_when_distill={cfg.blur_disable_phys_when_distill}")
            printed_blur_cfg = True

        accs = []
        adapter_anchor_sd = copy.deepcopy(model.adapter.state_dict())
        in_anchor_sd = copy.deepcopy(model.in_adapter.state_dict())

        for _ in range(nb):
            timer.start()
            (x, y), it = next_batch(it, loader)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = corrupt_fn(x, kind, sev)
            # --- RoTTA RobustBN update on current batch (pre-teacher) ---
            if cfg.rotta_enable and kind != "clean":
                robustbn_update_from_input(model.backbone, x, mom=cfg.rotta_robustbn_mom)

            with torch.no_grad():
                # Source confidence decides whether to pay the price of multi-view averaging
                with torch.amp.autocast("cuda", enabled=amp):
                    p0 = F.softmax(source_model(x), dim=1)
                src_conf = float(p0.max(dim=1).values.mean().item())

                view_mode = "blur" if kind == "blur" else "general"
                use_distill = (kind == "blur") and (cfg.blur_loss == "distill")
                x_t = unsharp_mask(x, sigma=1.0, amount=0.7) if use_distill else x
                views_max = min(cfg.aug_views * (2 if kind == "blur" else 1), 12)

                # 1) almeno 2 view per misurare disagree (gating)
                views_gate = 2 if src_conf >= cfg.confidence_min else max(2, views_max)

                # 2) puoi tenere 1 view per il target se vuoi velocità,
                #    oppure usare views_gate anche qui per stabilità
                views_target = 1 if src_conf >= cfg.confidence_min else views_max

                # gating statistics
                p_t_gate, conf, ent, disagree = teacher_aggregate(
                    teacher, x_t, views_gate, amp=amp, view_mode=view_mode
                )
                # --- RoTTA memory add (pseudo-label + uncertainty) ---
                if cfg.rotta_enable:
                    # pseudo-label da teacher e uncertainty semplice
                    # unc in [0,1] = 1 - confidence
                    y_hat = p_t_gate.argmax(dim=1)
                    unc = unc_from_entropy(p_t_gate)  # uncertainty = entropy normalizzata


                    # store sample-by-sample (CPU) per memory
                    # FIX: step_time once per batch here, then never inside add()
                    rotta_mem.step_time()
                    for bi in range(x.size(0)):
                        rotta_mem.add(
                            x_cpu_chw=x[bi].detach().cpu(),
                            y=int(y_hat[bi].item()),
                            unc=float(unc[bi].item()),
                            alpha=cfg.rotta_alpha,
                            lambda_t=cfg.rotta_lambda_t,
                            lambda_u=cfg.rotta_lambda_u,
                            step_time=False,  # FIX: already stepped above, don't double-count
                        )
                    # 3) PATCH: limita lo spam di log memoria (stesso blocco)
                    # sostituisci:
                    # if cfg.rotta_enable and (total_batches % 5 == 0):
                    # con:

                    if cfg.rotta_enable and (total_batches % DEBUG_MEM_EVERY == 0):
                        print(f"[SANITY][Mem] t={rotta_mem.time} len={len(rotta_mem)} cap={cfg.rotta_memory_size}")

                # target per distill/KLD (opzionale: usa p_t_gate oppure ricalcola)
                p_t, _, _, _ = teacher_aggregate(
                    teacher, x_t, views_target, amp=amp, view_mode=view_mode
                )

            dis_m, ent_m, conf_m, conf_f = float(torch.quantile(disagree, 0.5).item()), float(ent.mean().item()), float(conf.mean().item()), float((conf > cfg.confidence_min).float().mean().item())
            # Aggiorna statistiche running per adaptive threshold
            # Aggiorna statistiche running per adaptive threshold (history separata per blur vs resto)
            _hist = disagree_history_blur if kind == "blur" else disagree_history_gen
            _hist.append(dis_m)
            if len(_hist) > 100:
                _hist.pop(0)
            running_disagree_mean = float(np.mean(_hist)) if _hist else 0.0
            running_disagree_std = float(np.std(_hist)) if _hist else 0.0

            # Dynamic disagreement threshold (robust across scales)
            dis_thr_dyn = max(cfg.disagree_thresh, running_disagree_mean + 1.0 * running_disagree_std)

            ent_ms.append(ent_m);
            dis_ms.append(dis_m);
            conf_ms.append(conf_m);
            conf_fs.append(conf_f)

            # SPRT for blur - usa statistica più robusta
            if cfg.use_sprt_for_blur and kind == "blur":
                # Usa percentile invece di media per robustezza agli outlier
                # per blur, la "disagree" teacher non è un buon segnale: usa entropia teacher normalizzata
                s = float((ent.mean() / math.log(p_t_gate.size(1))).item())

                sig2, mu0, mu1 = max(1e-12, float(cfg.sprt_sigma) ** 2), float(cfg.sprt_mu0), float(cfg.sprt_mu1)
                # FIX: correct SPRT LLR for Gaussian: log(L1/L0) = [-(s-mu1)^2 + (s-mu0)^2] / (2*sig2)
                llr_inc = (-(s - mu1) ** 2 + (s - mu0) ** 2) / (2.0 * sig2)
                llr_blur += llr_inc
                sprt_state = "shift" if llr_blur >= A else ("no_shift" if llr_blur <= B else "unknown")
                if total_batches % DEBUG_MEM_EVERY == 0:
                    print(f"[SANITY][SPRT] llr={llr_blur:.4f} A={A:.4f} B={B:.4f} state={sprt_state} s={s:.4f}")

            # Decide adaptation
            if kind == "blur":
                blur_ent_thr = cfg.entropy_thresh * cfg.blur_entropy_scale
                do_adapt_heur = (ent_m > blur_ent_thr) or (dis_m > dis_thr_dyn * 0.5)
                # fallback: se entropia è davvero alta, adatta anche se SPRT dice no_shift
                if cfg.use_sprt_for_blur:
                    do_adapt = do_adapt_heur and ((sprt_state != "no_shift") or (ent_m > blur_ent_thr * 2.0))
                else:
                    do_adapt = do_adapt_heur
            else:
                do_adapt = (ent_m > cfg.entropy_thresh) or (dis_m > dis_thr_dyn)

            if kind == "clean":
                do_adapt = False

            confident = (conf_f >= cfg.conf_frac_min)

            gate_true += int(do_adapt);
            conf_true += int(confident)
            # --- RoTTA periodic update on memory (continual-stable) ---
            freq = int(cfg.rotta_update_frequency)
            if cfg.rotta_enable and (len(rotta_mem) >= cfg.rotta_batch_size) and (
                    total_batches % freq == 0) and total_batches > 0:

                # update usa loss KL(teacher || student) su batch campionato dal memory
                model.train()
                for _rs in range(cfg.rotta_steps):
                    xm, ym, ages, uncs = rotta_mem.sample(cfg.rotta_batch_size, device)

                    # timeliness + certainty weights
                    w = torch.exp(-cfg.rotta_alpha * ages).clamp(min=1e-6, max=1.0)
                    w = w / (w.mean() + 1e-6)

                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=amp):
                        # teacher target (single-view, veloce)
                        pt = F.softmax(teacher(xm), dim=1).detach()
                        ls = model(strong_aug(xm))  # strong aug SOLO sullo student (RoTTA core)

                        ps = F.softmax(ls, dim=1)

                        eps = 1e-6
                        log_pt = (pt.clamp_min(eps)).log()
                        log_ps = (ps.clamp_min(eps)).log()
                        kld = (pt * (log_pt - log_ps)).sum(dim=1)   # (B,)
                        loss_rotta = (w * kld).mean()
                        print(f"[SANITY][RoTTA loss] {float(loss_rotta.detach().item()):.4f}")

                    scaler.scale(loss_rotta).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()

                    # aggiorna teacher EMA (stessa regola tua)
                    ema_update(teacher, model, momentum=cfg.teacher_momentum)

                model.eval()


            # Learning rate adattivo basato su confidence e disagreement
            lr_scale_dyn = 1.0
            if do_adapt:
                # Riduci LR se alta incertezza (alto disagree o bassa conf)
                if dis_m > cfg.disagree_thresh * 2.0:
                    lr_scale_dyn = 0.5
                elif conf_m < 0.5:
                    lr_scale_dyn = 0.7

            lr_adapt = cfg.lr * (cfg.blur_lr_scale if kind == "blur" else 1.0) * lr_scale_dyn
            lr_in = (lr_adapt * cfg.in_adapter_lr_scale) if kind == "blur" else 0.0
            opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = lr_adapt, lr_in
            total_batches += 1

            # --- (A) BN stats-only update (opzionale) ---
            # --- (A) BN stats-only update (opzionale) ---
            # Non-blur shifts (gauss/jpeg/other): aggiorna solo running stats BN senza fare step
            # Non-blur shifts (gauss/jpeg/other): DISABILITATO per non contaminare BN e far crollare clean
            # Se vuoi riattivarlo, fallo con una BN-bank separata per "noise".
            if False and kind != "blur" and kind != "clean":
                model.backbone.train()
                model.adapter.eval()
                model.in_adapter.eval()
                model.classifier.eval()
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
                    _ = model(x)
                model.backbone.eval()
                model.adapter.train()


            # Blur: se usi blur_bn_bank, aggiorna stats BN del bank blur (stats-only)
            if kind == "blur" and cfg.blur_bn_bank:
                model.backbone.train()
                model.adapter.eval()
                model.in_adapter.eval()
                model.classifier.eval()
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
                    _ = model(x)
                model.backbone.eval()
                model.adapter.train()
                model.in_adapter.train()
                model.classifier.eval()


            # --- (B) Adattamento vero (step di ottimizzazione) ---
            # --- (B) Adattamento vero (step di ottimizzazione) ---
            if do_adapt and confident:
                if kind == "blur":
                    confident = True
                adapted_batches += 1  # conta davvero i batch adattati
                seg_adapted += 1

                steps_now = cfg.adapt_steps * (2 if kind == "blur" else 1)
                prev_loss = float("inf")

                for _k in range(steps_now):
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=amp):
                        logits = model(x)
                        p_s = F.softmax(logits, dim=1)
                        eps = 1e-12

                        # ===== LOSS BLOCK (drop-in) =====
                        # Uses teacher target p_t (already computed above).
                        # Requires: p_t (B,K), p_s (B,K), logits (B,K), x, kind, cfg, use_distill, amp
                        # Produces: mask, loss_kld, ent_s, loss_phys, loss_cons, loss_im, loss_prox, loss

                        # student log-probs / entropy
                        log_p_s = (p_s.clamp_min(eps)).log()
                        ent_s = -(p_s * log_p_s).sum(dim=1)  # (B,)

                        # teacher mask (confidence-based)
                        with torch.no_grad():
                            conf_t = p_t.max(dim=1).values  # (B,)
                            thr = cfg.confidence_min if kind != "blur" else (cfg.confidence_min - 0.15)
                            mask = (conf_t >= thr).float()

                            # if everything is masked out, keep a tiny mass to avoid NaNs
                            mask_mean = mask.mean().clamp_min(1e-6)

                        # --- (1) Distillation / KLD objective ---
                        # KLD(p_t || p_s) = sum p_t (log p_t - log p_s)
                        # Use temperature only for distill (blur branch)
                        if kind == "blur" and use_distill:
                            Tt = float(cfg.blur_distill_temp)
                            # re-compute softened distributions (teacher already prob-space -> approximate via log/renorm)
                            # safer: soften student from logits, teacher from log(p_t)
                            p_s_T = F.softmax(logits / max(1e-6, Tt), dim=1)
                            log_p_s_T = (p_s_T.clamp_min(eps)).log()

                            log_p_t = (p_t.clamp_min(eps)).log()
                            p_t_T = F.softmax(log_p_t / max(1e-6, Tt), dim=1)
                            log_p_t_T = (p_t_T.clamp_min(eps)).log()

                            kld_per = (p_t_T * (log_p_t_T - log_p_s_T)).sum(dim=1)  # (B,)
                            loss_kld = (kld_per * mask).sum() / mask_mean
                            loss_kld = loss_kld * (Tt * Tt)  # standard distill scaling
                            loss_kld = loss_kld * float(cfg.blur_distill_w)
                        else:
                            log_p_t = (p_t.clamp_min(eps)).log()
                            kld_per = (p_t * (log_p_t - log_p_s)).sum(dim=1)  # (B,)
                            loss_kld = (kld_per * mask).sum() / mask_mean

                        # --- (2) IM loss option (only if selected) ---
                        # Encourage low entropy on student predictions (masked)
                        if kind == "blur" and (cfg.blur_loss == "im"):
                            loss_im = (ent_s * mask).sum() / mask_mean
                            loss_im = float(cfg.blur_im_lambda) * loss_im
                        else:
                            loss_im = torch.zeros((), device=logits.device)

                        # --- (3) Consistency loss (multi-view student) ---
                        # Cheap 2-view consistency: student(x) vs student(aug(x)).
                        # Keep small weight; helps stabilità.
                        if kind != "clean":
                            with torch.no_grad():
                                x2 = x
                                if random.random() < 0.5:
                                    x2 = torch.flip(x2, dims=[3])
                                if random.random() < 0.25:
                                    x2 = (x2 + 0.02 * torch.randn_like(x2)).clamp(0.0, 1.0)
                            logits2 = model(x2)
                            p2 = F.softmax(logits2, dim=1)
                            # symmetric KL (masked)
                            log_p2 = (p2.clamp_min(eps)).log()
                            kl12 = (p_s * (log_p_s - log_p2)).sum(dim=1)
                            kl21 = (p2 * (log_p2 - log_p_s)).sum(dim=1)
                            loss_cons = (((kl12 + kl21) * 0.5) * mask).sum() / mask_mean
                            loss_cons = 0.1 * loss_cons
                        else:
                            loss_cons = torch.zeros((), device=logits.device)

                        # --- (4) Physics loss (blur only, optional) ---
                        # Encourage invariance between blur and unsharp (or original) student predictions.
                        # Disable if distill to avoid redundancy/conflict (your cfg flag).
                        if kind == "blur" and (float(cfg.blur_phys_w) > 0.0) and not (
                                use_distill and cfg.blur_disable_phys_when_distill):
                            x_sh = unsharp_mask(x, sigma=1.0, amount=0.7)
                            logits_sh = model(x_sh)
                            p_sh = F.softmax(logits_sh, dim=1)
                            log_p_sh = (p_sh.clamp_min(eps)).log()
                            # KL(p_s || p_sh) masked
                            phys_per = (p_s * (log_p_s - log_p_sh)).sum(dim=1)
                            loss_phys = (phys_per * mask).sum() / mask_mean
                            loss_phys = float(cfg.blur_phys_w) * loss_phys
                        else:
                            loss_phys = torch.zeros((), device=logits.device)

                        # --- (5) Proximal loss to anchor adapter near segment entry (prevents drift) ---
                        # L2 distance on adapter weights between current and anchor_sd.
                        if float(cfg.prox_w) > 0.0:
                            prox = torch.zeros((), device=logits.device)
                            cur_sd = model.adapter.state_dict()
                            for kk in cur_sd.keys():
                                prox = prox + (cur_sd[kk] - adapter_anchor_sd[kk].to(cur_sd[kk].device)).pow(2).mean()
                            loss_prox = float(cfg.prox_w) * prox
                        else:
                            loss_prox = torch.zeros((), device=logits.device)

                        # --- (6) Prox verso identità/source (protezione clean) ---
                        if float(getattr(cfg, "prox_w_id", 0.0)) > 0.0:
                            prox_id = torch.zeros((), device=logits.device)
                            cur_sd = model.adapter.state_dict()
                            for kk in cur_sd.keys():
                                prox_id = prox_id + (cur_sd[kk] - adapter_identity[kk].to(cur_sd[kk].device)).pow(2).mean()
                            loss_prox_id = float(cfg.prox_w_id) * prox_id
                        else:
                            loss_prox_id = torch.zeros((), device=logits.device)


                        # --- Final loss selection/mix ---
                        if kind == "blur":
                            if cfg.blur_loss == "kld":
                                loss = loss_kld + loss_cons + loss_phys + loss_prox + loss_prox_id
                            elif cfg.blur_loss == "im":
                                # IM + (small) distill stabilizer
                                loss = loss_im + 0.25 * loss_kld + loss_cons + loss_phys + loss_prox + loss_prox_id
                            else:  # "distill"
                                loss = loss_kld + loss_cons + loss_phys + loss_prox + loss_prox_id

                        else:
                            # non-blur: plain kld + consistency + prox
                            loss = loss_kld + loss_cons + loss_phys + loss_prox + loss_prox_id

                        # ===== END LOSS BLOCK =====

                        mask_sz_ms.append(float(mask.float().mean().item()))
                        loss_klds.append(float(loss_kld.detach().item()))
                        ent_ss.append(float(((ent_s.detach() * mask).sum() / mask_mean).item()))
                        loss_ts.append(float(loss.detach().item()))
                        # opzionale: se calcoli phys
                        # loss_phys_ms.append(float(loss_phys.detach().item()))
                        # dx_ms.append(float((x - x_sh).abs().mean().item()))

                        # ===== END LOSS BLOCK =====
                    if loss.detach().item() > LOSS_HARD_CAP:
                        opt.zero_grad(set_to_none=True)
                        break
                    # [A] SAFETY: stop/rollback if loss is NaN/Inf
                    if not torch.isfinite(loss):
                        model.adapter.load_state_dict(adapter_anchor_sd, strict=True)
                        model.in_adapter.load_state_dict(in_anchor_sd, strict=True)
                        opt.zero_grad(set_to_none=True)
                        break

                    scaler.scale(loss).backward()

                    # unscale SOLO UNA VOLTA per step
                    scaler.unscale_(opt)

                    # [B] SAFETY: skip if any grad is NaN/Inf
                    bad = False
                    for p in list(model.adapter.parameters()) + (
                    list(model.in_adapter.parameters()) if kind == "blur" else []):
                        if p.grad is not None and (not torch.isfinite(p.grad).all()):
                            bad = True
                            break
                    if bad:
                        opt.zero_grad(set_to_none=True)
                        scaler.update()  # IMPORTANT: reset stato scaler per evitare "unscale already called"
                        break

                    clip_val = 1.0 if dis_m < cfg.disagree_thresh * 1.5 else 0.5
                    if kind == "blur":
                        torch.nn.utils.clip_grad_norm_(
                            list(model.adapter.parameters()) + list(model.in_adapter.parameters()),
                            clip_val
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), clip_val)

                    gn = sum((p.grad.detach() ** 2).sum().item()
                             for p in model.adapter.parameters() if p.grad is not None)
                    grad_ns.append(math.sqrt(gn))

                    scaler.step(opt)
                    scaler.update()

                    updates += 1

                    if _k > 0 and abs(loss.item() - prev_loss) < 1e-5:
                        break
                    prev_loss = loss.item()

                model.eval()


            else:
                # niente update, applica decay/restore “safe”
                decay = cfg.adapter_decay_when_clean * (10.0 if kind == "clean" else 1.0)
                adapter_decay_step(model, decay)

                if kind == "clean":
                    adapter_soft_restore_to_state(model, adapter_identity, rho=0.02)

            # Reset periodico per prevenire drift eccessivo
            if total_batches % 500 == 0 and total_batches > 0:
                adapter_soft_restore_to_state(model, adapter_identity, rho=0.15)
                print(f"[periodic reset] batch {total_batches}: soft restore adapter")
            restore_prob = cfg.restore_prob_when_shift if do_adapt else cfg.restore_prob_when_clean
            stochastic_restore_backbone(model, source_model, prob=restore_prob)

            # Update teacher solo se sample affidabile per evitare error propagation
            if (kind != "blur") or cfg.blur_update_teacher:
                # Usa momentum più alto (slower update) quando incertezza è alta
                mom = cfg.teacher_momentum
                if dis_m > cfg.disagree_thresh * 1.5:
                    mom = 0.9998  # quasi nessun update
                ema_update(teacher, model, momentum=mom)

            # CoTTA-style: predict with EMA teacher (more stable than hot-updated student)
            # usa teacher per eval (stabile); student serve per update
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
                logits_eval = teacher(x)

            accs.append(accuracy(logits_eval, y))

            ms = timer.stop_ms()
            all_ms.append(ms); seg_batch_ms.append(ms)
            if do_adapt and confident:
                adapt_ms.append(ms); seg_adapt_ms.append(ms)

        # Restore BN bank after blur
        if kind == "blur" and cfg.blur_bn_bank:
            adapter_blur_state, in_blur_state = copy.deepcopy(model.adapter.state_dict()), copy.deepcopy(model.in_adapter.state_dict())
            bn_blur_state = bn_buffers_state(model.backbone)
            model.adapter.load_state_dict(adapter_main_state, strict=True)
            model.in_adapter.load_state_dict(in_main_state, strict=True)
            load_bn_buffers(model.backbone, bn_main_state)
            teacher = teacher_main

        seg_acc[f"{si}:{kind}{sev}"] = mean(accs)

        if debug:
            print(
                f"[seg {si}:{kind}{sev}] gate={gate_true}/{nb} conf_ok={conf_true}/{nb} "
                f"seg_adapted={seg_adapted}/{nb} updates={updates} | "
                f"ent={mean(ent_ms):.3f} dis={mean(dis_ms):.3f} conf_m={mean(conf_ms):.3f} conf_f={mean(conf_fs):.3f} | "
                f"loss_kld={mean(loss_klds):.4f} ent_s={mean(ent_ss):.4f} loss={mean(loss_ts):.4f} grad={mean(grad_ns):.3f} | "
                f"acc={seg_acc[f'{si}:{kind}{sev}']:.4f} | ms/batch={mean(seg_batch_ms):.2f} ms/adapt={mean(seg_adapt_ms):.2f} "
                f"peakVRAM={peak_mem_mb():.0f}MB phys={mean(loss_phys_ms):.4f} dx={mean(dx_ms):.4f} mask={mean(mask_sz_ms):.3f}"
            )


    print(f"[NP-TTA] adapted {adapted_batches}/{total_batches} batches = {adapted_batches / max(1,total_batches):.3f}")
    print(f"[perf] avg_ms/batch={mean(all_ms):.2f} avg_ms/adapt_batch={mean(adapt_ms):.2f} peakVRAM={peak_mem_mb():.0f}MB")
    return seg_acc

# ============ Sweep ============
def run_one_tta_eval(args, cfg_overrides: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=T.ToTensor())
    _, schedule = make_stream_len_and_schedule(args)

    base_model = ResNet18WithAdapter(num_classes=10, adapter_rank=args.adapter_rank).to(device)
    base_model.load_state_dict(ckpt["model"], strict=False)
    base_model.reset_adapter_identity(); base_model.eval()

    source_model = ResNet18WithAdapter(num_classes=10, adapter_rank=args.adapter_rank).to(device)
    source_model.load_state_dict(ckpt["model"], strict=False); source_model.eval()

    stream_loader_base = make_stream_loader(test_set, args.batch_size, args.num_workers, seed=args.seed)
    base = eval_stream_baseline(base_model, stream_loader_base, device, schedule)

    tta_model = ResNet18WithAdapter(num_classes=10, adapter_rank=args.adapter_rank).to(device)
    tta_model.load_state_dict(ckpt["model"], strict=False)
    tta_model.reset_adapter_identity(); tta_model.eval()

    cfg = build_cfg_from_args(args, cfg_overrides=cfg_overrides)
    set_seed(cfg_overrides.get("run_seed", args.seed))

    stream_loader_tta = make_stream_loader(test_set, args.batch_size, args.num_workers, seed=args.seed)
    tta = eval_stream_nptta(tta_model, source_model, stream_loader_tta, device, cfg, schedule, amp=args.amp, debug=False)

    k_gauss3_revisit, k_blur3 = last_seg_key(schedule, "gauss", 3), last_seg_key(schedule, "blur", 3)
    out = {}
    out.update(cfg_overrides)
    out["base_clean"], out["tta_clean"] = avg_clean(base), avg_clean(tta)
    out["d_clean"] = out["tta_clean"] - out["base_clean"]
    out["base_gauss3_revisit"], out["tta_gauss3_revisit"] = base[k_gauss3_revisit], tta[k_gauss3_revisit]
    out["d_gauss3_revisit"] = out["tta_gauss3_revisit"] - out["base_gauss3_revisit"]
    out["base_blur3"], out["tta_blur3"] = base[k_blur3], tta[k_blur3]
    out["d_blur3"] = out["tta_blur3"] - out["base_blur3"]
    clean_penalty = max(0.0, -out["d_clean"]) * 5.0
    out["score"] = out["d_gauss3_revisit"] + out["d_blur3"] - clean_penalty
    return out

def sweep(args) -> None:
    space = {
        "tta_lr": [1e-4, 3e-4, 5e-4, 1e-3, 2e-3],
        "entropy_thresh": [0.15, 0.25, 0.35, 0.5, 0.8],
        "disagree_thresh": [0.01, 0.02, 0.03, 0.05, 0.08],
        "teacher_momentum": [0.99, 0.995, 0.999, 0.9995],
        "confidence_min": [0.3, 0.4, 0.5, 0.6],
        "conf_frac_min": [0.05, 0.1, 0.2],
        "adapt_steps": [1, 2, 4],
        "aug_views": [1, 2, 4, 8],
        "adapter_decay_when_clean": [0.0, 0.01, 0.03, 0.05],
        "blur_lr_scale": [0.1, 0.2, 0.3, 0.5, 1.0],
        "blur_loss": ["distill", "im", "kld"],
        "blur_distill_temp": [0.5, 0.7, 0.9],
        "blur_distill_w": [0.5, 1.0, 2.0],
        "blur_bn_bank": [True, False],
        "blur_phys_w": [0.0, 0.25, 0.5, 1.0],
        "blur_disable_phys_when_distill": [True, False],
        "blur_entropy_scale": [0.5, 0.7, 0.85, 1.0],
    }
    rng, rows = random.Random(args.sweep_seed), []
    for i in range(args.sweep_trials):
        cfg = {k: rng.choice(v) for k, v in space.items()}
        cfg["run_seed"] = args.seed + i
        print(f"[sweep {i+1}/{args.sweep_trials}] cfg={cfg}")

        r = run_one_tta_eval(args, cfg)
        print(
            f"[sweep {i+1}/{args.sweep_trials}] "
            f"score={r['score']:+.4f} d_blur3={r['d_blur3']:+.4f} "
            f"d_gauss={r['d_gauss3_revisit']:+.4f} d_clean={r['d_clean']:+.4f}"
        )
        rows.append(r)

    rows.sort(key=lambda r: r["score"], reverse=True)
    os.makedirs(os.path.dirname(args.sweep_out) or ".", exist_ok=True)
    with open(args.sweep_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nTOP-10 by score:")
    for r in rows[:10]:
        print(
            f"score={r['score']:+.4f} d_gauss={r['d_gauss3_revisit']:+.4f} "
            f"d_blur={r['d_blur3']:+.4f} d_clean={r['d_clean']:+.4f} | "
            f"lr={r['tta_lr']:.0e} ent={r['entropy_thresh']} dis={r['disagree_thresh']} mom={r['teacher_momentum']} "
            f"steps={r['adapt_steps']} views={r['aug_views']} cmin={r['confidence_min']} cfrac={r['conf_frac_min']} "
            f"decay={r['adapter_decay_when_clean']} blur_lr={r['blur_lr_scale']} blur_loss={r.get('blur_loss','?')} "
            f"T={r.get('blur_distill_temp','?')} w={r.get('blur_distill_w','?')} bn_bank={r.get('blur_bn_bank','?')} "
            f"phys_w={r.get('blur_phys_w','?')} disable_phys_when_distill={r.get('blur_disable_phys_when_distill','?')}"
        )

    print(f"\nSaved sweep results to: {args.sweep_out}")


# ============ CLI ============
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--eval_c10c", action="store_true")

    # C10-C
    ap.add_argument("--c10c_root", type=str, default="./data/CIFAR-10-C")
    ap.add_argument("--c10c_corruptions", type=str,
                    default="gaussian_noise,gaussian_blur,motion_blur,jpeg_compression")
    ap.add_argument("--c10c_severities", type=str, default="1,2,3,4,5")
    ap.add_argument("--c10c_shuffle", action="store_true")

    # train / data
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.1)  # source training LR (SGD)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", type=str, default="./checkpoints/cifar10_resnet18_adapter.pt")

    # stream / model
    ap.add_argument("--stream_len", type=int, default=20000)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--adapter_rank", type=int, default=128)

    # sweep
    ap.add_argument("--sweep_trials", type=int, default=30)
    ap.add_argument("--sweep_seed", type=int, default=123)
    ap.add_argument("--sweep_out", type=str, default="./checkpoints/sweep_results.csv")

    # NP-TTA core (defaults: “sane” per far partire adattamento senza esplodere)
    ap.add_argument("--entropy_thresh", type=float, default=0.35)
    ap.add_argument("--tta_lr", type=float, default=3e-4)              # AdamW adapter LR
    ap.add_argument("--disagree_thresh", type=float, default=0.02)
    ap.add_argument("--teacher_momentum", type=float, default=0.9995)
    ap.add_argument("--confidence_min", type=float, default=0.6)
    ap.add_argument("--conf_frac_min", type=float, default=0.1)
    ap.add_argument("--adapt_steps", type=int, default=1)
    ap.add_argument("--aug_views", type=int, default=8)
    ap.add_argument("--adapter_decay_when_clean", type=float, default=0.0)

    # blur branch
    ap.add_argument("--blur_loss", type=str, default="distill", choices=["distill", "im", "kld"])
    ap.add_argument("--blur_im_lambda", type=float, default=1.0)
    ap.add_argument("--blur_distill_temp", type=float, default=0.7)
    ap.add_argument("--blur_distill_w", type=float, default=1.0)
    ap.add_argument("--in_adapter_lr_scale", type=float, default=1.0)
    ap.add_argument("--blur_update_teacher", action="store_true")
    ap.add_argument("--blur_lr_scale", type=float, default=0.2)
    ap.add_argument("--no_blur_bn_bank", action="store_true")
    ap.add_argument("--blur_phys_w", type=float, default=0.25)
    ap.add_argument("--blur_keep_phys_with_distill", action="store_true")
    ap.add_argument("--blur_entropy_scale", type=float, default=0.7,
                    help="Scale entropy_thresh for blur gating (lower=adapt more, default 0.7)")

    # SPRT
    ap.add_argument("--use_sprt_for_blur", action="store_true")
    ap.add_argument("--sprt_alpha", type=float, default=0.05)
    ap.add_argument("--sprt_beta", type=float, default=0.10)
    ap.add_argument("--sprt_mu0", type=float, default=0.03)
    ap.add_argument("--sprt_mu1", type=float, default=0.08)
    ap.add_argument("--sprt_sigma", type=float, default=0.03)

    # regularization
    ap.add_argument("--prox_w", type=float, default=0.05)
    ap.add_argument("--prox_w_id", type=float, default=0.10)
    # RoTTA core
    ap.add_argument("--rotta_enable", action="store_true")
    ap.add_argument("--rotta_memory_size", type=int, default=64)
    ap.add_argument("--rotta_update_frequency", type=int, default=64)
    ap.add_argument("--rotta_batch_size", type=int, default=64)
    ap.add_argument("--rotta_steps", type=int, default=1)
    ap.add_argument("--rotta_alpha", type=float, default=0.05)
    ap.add_argument("--rotta_lambda_t", type=float, default=1.0)
    ap.add_argument("--rotta_lambda_u", type=float, default=1.0)
    ap.add_argument("--rotta_robustbn_mom", type=float, default=0.01)
    ap.add_argument("--rotta_train_bn_affine_only", action="store_true")


    return ap.parse_args()



def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    model = ResNet18WithAdapter(num_classes=10, adapter_rank=args.adapter_rank).to(device)
    print(f"Total params: {count_params(model)/1e6:.2f}M")

    if args.sweep:
        if not os.path.exists(args.ckpt): raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}. Run with --train first.")
        sweep(args); return

    if args.train:
        train_source(model, device, args.epochs, args.batch_size, args.num_workers, args.lr, args.amp, args.ckpt)

    if args.eval_c10c:
        ensure_cifar10c(args.c10c_root)
        if not os.path.exists(args.ckpt): raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}. Run with --train first.")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        base_model = ResNet18WithAdapter(num_classes=10, adapter_rank=args.adapter_rank).to(device)
        base_model.load_state_dict(ckpt["model"], strict=False)
        base_model.reset_adapter_identity(); base_model.eval()
        source_model = ResNet18WithAdapter(num_classes=10, adapter_rank=args.adapter_rank).to(device)
        source_model.load_state_dict(ckpt["model"], strict=False); source_model.eval()
        cfg = build_cfg_from_args(args, cfg_overrides=None)
        corrs = [c.strip() for c in args.c10c_corruptions.split(",") if c.strip()]
        sevs = [int(s.strip()) for s in args.c10c_severities.split(",") if s.strip()]
        print("[C10-C] running:", {"corruptions": corrs, "severities": sevs})
        rows = []
        for corr in corrs:
            kind_tag = c10c_kind_tag(corr)
            for sev in sevs:
                ds = CIFAR10C(args.c10c_root, corr, sev, transform=None)
                loader = make_stream_loader(ds, args.batch_size, args.num_workers, seed=args.seed, shuffle=args.c10c_shuffle)
                b = eval_c10c_baseline(base_model, loader, device)
                tta_model = ResNet18WithAdapter(num_classes=10, adapter_rank=args.adapter_rank).to(device)
                tta_model.load_state_dict(ckpt["model"], strict=False)
                tta_model.reset_adapter_identity(); tta_model.eval()
                set_seed(args.seed)
                t = eval_c10c_nptta(tta_model, source_model, loader, device, cfg, amp=args.amp, kind_tag=kind_tag, sev=sev)
                rows.append((corr, sev, b, t, t - b))
                print(f"{corr:>18s} sev={sev} | base={b:.4f} tta={t:.4f} delta={t - b:+.4f}")
        if rows:
            dmean = sum(r[4] for r in rows) / len(rows)
            print(f"\n[C10-C] mean delta over {len(rows)} points: {dmean:+.4f}")
        return

    if args.eval:
        if not os.path.exists(args.ckpt): raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}. Run with --train first.")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        model.reset_adapter_identity(); model.eval()
        source_model = ResNet18WithAdapter(num_classes=10, adapter_rank=args.adapter_rank).to(device)
        source_model.load_state_dict(ckpt["model"], strict=False); source_model.eval()
        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=T.ToTensor())
        _, schedule = make_stream_len_and_schedule(args)
        print("Schedule (kind, severity, num_batches):", schedule)
        stream_loader_base = make_stream_loader(test_set, args.batch_size, args.num_workers, seed=args.seed)
        base = eval_stream_baseline(model, stream_loader_base, device, schedule)
        print("\n[Baseline] segment accuracies:")
        for k, v in base.items(): print(f"  {k:>10s}: {v:.4f}")
        cfg = build_cfg_from_args(args, cfg_overrides=None)
        print(f"[CFG] blur_loss(args={args.blur_loss} cfg={cfg.blur_loss}) bn_bank={cfg.blur_bn_bank} "
              f"phys_w={cfg.blur_phys_w} disable_phys_when_distill={cfg.blur_disable_phys_when_distill}")
        tta_model = ResNet18WithAdapter(num_classes=10, adapter_rank=args.adapter_rank).to(device)
        tta_model.load_state_dict(ckpt["model"], strict=False)
        tta_model.reset_adapter_identity(); tta_model.eval()
        set_seed(args.seed)
        stream_loader_tta = make_stream_loader(test_set, args.batch_size, args.num_workers, seed=args.seed)
        tta = eval_stream_nptta(tta_model, source_model, stream_loader_tta, device, cfg, schedule, amp=args.amp)
        print("\n[NP-TTA] segment accuracies:")
        for k, v in tta.items(): print(f"  {k:>10s}: {v:.4f}")
        k_gauss3_revisit, k_blur3 = last_seg_key(schedule, "gauss", 3), last_seg_key(schedule, "blur", 3)
        print("\nSummary:")
        print(f"  baseline avg clean        : {avg_clean(base):.4f}")
        print(f"  NP-TTA   avg clean        : {avg_clean(tta):.4f}")
        print(f"  baseline gauss3 revisit   : {base[k_gauss3_revisit]:.4f}")
        print(f"  NP-TTA   gauss3 revisit   : {tta[k_gauss3_revisit]:.4f}")
        print(f"  baseline blur3            : {base[k_blur3]:.4f}")
        print(f"  NP-TTA   blur3            : {tta[k_blur3]:.4f}")

if __name__ == "__main__":
    main()
