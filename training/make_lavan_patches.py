"""Generate EXTERNAL attack patches (LaVAN-style): optimize small targeted
patches against a pretrained ImageNet classifier with location jitter, then
composite them into COCO scenes. Neither the optimizer, the classifier, nor
these textures appear anywhere in the SDNL training pool.

Outputs (kept OUTSIDE the training pool; external eval only):
  data/ext_patches/lavan_<target>.png      optimized patch patterns
  data/preprocessed/ext_attack/*.npy       4-channel tensors of patched scenes

Requires: torch, torchvision (pretrained resnet18 weights download once).
CPU-friendly (~1 min/patch). Deterministic seeds.
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from sdnl import build_4ch_tensor

TARGETS = {  # ImageNet class ids: thematically varied targets
    "toaster": 859,
    "goose": 99,
    "cassette": 482,
}
PATCH_PX = 60
STEPS = 150
BATCH = 8
N_SCENES_PER_PATCH = 40


def load_coco_bases():
    bases = []
    for p in sorted(glob.glob(os.path.join("data", "coco_clean", "images", "*.jpg")))[:40]:
        img = cv2.imread(p)
        if img is not None:
            bases.append(img)
    for p in sorted(glob.glob(os.path.join("data", "crops", "clean", "coco_*.jpg")))[:40]:
        img = cv2.imread(p)
        if img is not None:
            bases.append(cv2.resize(img, (256, 256)))
    assert bases, "no COCO bases found"
    return bases


def optimize_patch(clf, mean, std, bases224, target_id, seed):
    import torch
    import torch.nn as nn
    g = torch.Generator().manual_seed(seed)
    patch = torch.rand(3, PATCH_PX, PATCH_PX, generator=g)  # [0,1] RGB
    patch.requires_grad_(True)
    opt = torch.optim.Adam([patch], lr=0.1)
    ce = nn.CrossEntropyLoss()
    idx = torch.randperm(len(bases224), generator=g)[:BATCH]
    batch = torch.stack([bases224[i] for i in idx])  # normalized NCHW
    tgt = torch.full((BATCH,), target_id, dtype=torch.long)
    for _ in range(STEPS):
        opt.zero_grad()
        adv = batch.clone()
        for b in range(BATCH):
            top = int(torch.randint(0, 224 - PATCH_PX + 1, (1,), generator=g))
            left = int(torch.randint(0, 224 - PATCH_PX + 1, (1,), generator=g))
            adv[b, :, top:top + PATCH_PX, left:left + PATCH_PX] = patch
        loss = ce(clf((adv - mean) / std), tgt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            patch.clamp_(0, 1)
    return (patch.detach() * 255).byte().permute(1, 2, 0).numpy()[..., ::-1]  # BGR


def main():
    import torch
    import torchvision.models as models
    torch.manual_seed(0)
    device = torch.device("cpu")

    patch_dir = os.path.join("data", "ext_patches")
    prep_dir = os.path.join("data", "preprocessed", "ext_attack")
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)

    weights = models.ResNet18_Weights.DEFAULT
    clf = models.resnet18(weights=weights).to(device).eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
    std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)

    # COCO bases normalized for the classifier
    import torchvision.transforms as T
    tf = T.Compose([T.ToPILImage(), T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    bases = load_coco_bases()
    bases224 = [tf(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)) for b in bases]

    rng = np.random.default_rng(0)
    made = 0
    for name, tid in TARGETS.items():
        patch = optimize_patch(clf, mean, std, bases224, tid, seed=100 + tid)
        cv2.imwrite(os.path.join(patch_dir, f"lavan_{name}.png"), patch)
        for i in range(N_SCENES_PER_PATCH):
            scene = bases[int(rng.integers(0, len(bases)))].copy()
            h, w = scene.shape[:2]
            side = int(min(h, w) * rng.uniform(0.15, 0.40))
            pr = cv2.resize(patch, (side, side), interpolation=cv2.INTER_LINEAR)
            y0, x0 = int(rng.integers(0, h - side)), int(rng.integers(0, w - side))
            comp = scene.copy()
            comp[y0:y0 + side, x0:x0 + side] = pr
            comp128 = cv2.resize(comp, (128, 128))
            tag = f"ext_lavan_{name}_{i:03d}"
            np.save(os.path.join(prep_dir, tag + ".npy"), build_4ch_tensor(comp128))
            made += 1
        print(f"[lavan] patch {name} (ImageNet {tid}) + {N_SCENES_PER_PATCH} scenes")
    print(f"[lavan] wrote {made} external attack tensors -> {prep_dir}")


if __name__ == "__main__":
    main()
