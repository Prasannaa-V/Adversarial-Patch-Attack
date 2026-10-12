"""Synthesize hard CLEAN negatives: the texture-rich everyday patterns that
false-positive the DCT channel (phone screens, text pages, icon grids,
QR-like codes, barcodes, halftone print, checkered fabric).

Writes 128x128 JPGs to data/crops/synth_clean/ and 4-channel .npy tensors
(via inference/sdnl.py:build_4ch_tensor) to data/preprocessed/clean/synth_*.npy
so the next training run learns them as CLEAN. Deterministic (seed=7).
No new dependencies beyond numpy/opencv (already required).
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from sdnl import build_4ch_tensor

SEED = 7
SIZE = 128


def _save(name, img, crops_dir, prep_dir, npy=True):
    cv2.imwrite(os.path.join(crops_dir, name + ".jpg"), img)
    if npy:
        np.save(os.path.join(prep_dir, name + ".npy"), build_4ch_tensor(img))


def text_pages(rng, n, crops_dir, prep_dir):
    for i in range(n):
        bg = int(rng.integers(225, 255))
        img = np.full((SIZE, SIZE, 3), bg, np.uint8)
        y = 4
        while y < SIZE - 6:
            x = 4
            h = int(rng.integers(3, 6))
            while x < SIZE - 8:
                w = int(rng.integers(6, 26))
                col = (30, 30, 30) if rng.random() < 0.85 else (180, 40, 20)
                img[y:y + h, x:x + w] = col
                x += w + int(rng.integers(2, 6))
            y += h + int(rng.integers(2, 5))
        if i % 3 == 0:  # headline bar
            img[4:12, 4:90] = (25, 25, 25)
        _save(f"synth_text_{i:03d}", img, crops_dir, prep_dir)


def icon_grids(rng, n, crops_dir, prep_dir):
    for i in range(n):
        bg = (15, 15, 22) if i % 2 == 0 else (235, 235, 240)
        img = np.full((SIZE, SIZE, 3), bg, np.uint8)
        cell = SIZE // 4
        for gy in range(4):
            for gx in range(4):
                col = tuple(int(v) for v in rng.integers(50, 255, 3))
                x0, y0 = gx * cell + 5, gy * cell + 5
                cv2.rectangle(img, (x0, y0), (x0 + cell - 10, y0 + cell - 10), col, -1)
                cv2.circle(img, (x0 + (cell - 10) // 2, y0 + (cell - 10) // 2), 5,
                           (255, 255, 255), -1)
        _save(f"synth_icons_{i:03d}", img, crops_dir, prep_dir)


def _finder(img, ox, oy, s, rng):
    cv2.rectangle(img, (ox, oy), (ox + s, oy + s), (0, 0, 0), -1)
    cv2.rectangle(img, (ox + 2, oy + 2), (ox + s - 2, oy + s - 2), (255, 255, 255), -1)
    cv2.rectangle(img, (ox + 4, oy + 4), (ox + s - 4, oy + s - 4), (0, 0, 0), -1)


def qr_like(rng, n, crops_dir, prep_dir):
    for i in range(n):
        img = np.full((SIZE, SIZE, 3), 255, np.uint8)
        m = 21  # modules per side
        cell = SIZE // m
        _finder(img, 0, 0, cell * 7, rng)
        _finder(img, SIZE - cell * 7, 0, cell * 7, rng)
        _finder(img, 0, SIZE - cell * 7, cell * 7, rng)
        for gy in range(m):
            for gx in range(m):
                in_f = ((gx < 8 and gy < 8) or (gx >= m - 8 and gy < 8)
                        or (gx < 8 and gy >= m - 8))
                if in_f or rng.random() < 0.52:
                    continue
                img[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell] = (0, 0, 0)
        if i % 2 == 0:  # slight rotation like a photographed code
            ctr, ang = (SIZE / 2, SIZE / 2), float(rng.uniform(-8, 8))
            img = cv2.warpAffine(img, cv2.getRotationMatrix2D(ctr, ang, 1.0),
                                 (SIZE, SIZE), borderValue=(255, 255, 255))
        _save(f"synth_qr_{i:03d}", img, crops_dir, prep_dir)


def barcodes(rng, n, crops_dir, prep_dir):
    for i in range(n):
        img = np.full((SIZE, SIZE, 3), 255, np.uint8)
        x = 4
        while x < SIZE - 4:
            w = int(rng.integers(1, 5))
            if rng.random() < 0.55:
                img[10:100, x:x + w] = (0, 0, 0)
            x += w + int(rng.integers(1, 4))
        img[104:120, 10:118] = (20, 20, 20)  # digits strip
        _save(f"synth_barcode_{i:03d}", img, crops_dir, prep_dir)


def moire_screens(rng, n, crops_dir, prep_dir):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    for i in range(n):
        base = np.full((SIZE, SIZE, 3), int(rng.integers(15, 60)), np.uint8).astype(np.float32)
        # app-ish blobs + text bars
        for _ in range(6):
            cx, cy = int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE))
            cv2.circle(base, (cx, cy), int(rng.integers(6, 18)),
                       tuple(float(v) for v in rng.integers(80, 230, 3)), -1)
        for yy0 in range(8, SIZE, 16):
            base[yy0:yy0 + 4, 8:120] = 210.0
        f1, f2 = float(rng.uniform(3.5, 6.5)), float(rng.uniform(4.5, 8.0))
        moire = (10 * np.sin(2 * np.pi * xx / f1)
                 + 8 * np.sin(2 * np.pi * (xx + yy) / f2))[..., None]
        img = np.clip(base + moire + rng.normal(0, 3, (SIZE, SIZE, 1)), 0, 255).astype(np.uint8)
        _save(f"synth_moire_{i:03d}", img, crops_dir, prep_dir)


def halftone(rng, n, crops_dir, prep_dir):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    for i in range(n):
        shade = 100 + 60 * np.sin(xx / 18.0) * np.cos(yy / 22.0)
        dots = ((np.sin(xx / 2.2) + np.sin(yy / 2.2)) > 0.4)
        img = np.where(dots[..., None], 25, shade[..., None])
        img = np.repeat(img, 3, axis=2).astype(np.uint8)
        img[4:12, 4:100] = (20, 20, 20)  # headline
        _save(f"synth_halftone_{i:03d}", img, crops_dir, prep_dir)


def checks_stripes(rng, n, crops_dir, prep_dir):
    for i in range(n):
        kind = i % 3
        if kind == 0:  # checkerboard
            s = int(rng.choice([6, 8, 12]))
            pat = (np.indices((SIZE, SIZE)).sum(0) // s) % 2
            img = np.where(pat[..., None] == 0, 20, 235).astype(np.uint8)
        elif kind == 1:  # stripes
            s = int(rng.choice([3, 5, 9]))
            pat = (np.arange(SIZE)[None, :].repeat(SIZE, 0) // s) % 2
            img = np.where(pat[..., None] == 0, 30, 225).astype(np.uint8)
        else:  # noisy fabric weave
            img = rng.integers(90, 170, (SIZE, SIZE, 1)).repeat(3, axis=2).astype(np.uint8)
            img[::3] = np.clip(img[::3].astype(int) - 40, 0, 255)
        img = np.repeat(img[:, :, :1], 3, axis=2).astype(np.uint8) if img.shape[2] == 1 else img
        _save(f"synth_fabric_{i:03d}", img, crops_dir, prep_dir)


def main():
    rng = np.random.default_rng(SEED)
    crops_dir = os.path.join("data", "crops", "synth_clean")
    prep_dir = os.path.join("data", "preprocessed", "clean")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)
    # Remove stale synth tensors so counts are exact
    for f in os.listdir(prep_dir):
        if f.startswith("synth_"):
            os.remove(os.path.join(prep_dir, f))
    text_pages(rng, 60, crops_dir, prep_dir)
    icon_grids(rng, 60, crops_dir, prep_dir)
    qr_like(rng, 60, crops_dir, prep_dir)
    barcodes(rng, 40, crops_dir, prep_dir)
    moire_screens(rng, 40, crops_dir, prep_dir)
    halftone(rng, 40, crops_dir, prep_dir)
    checks_stripes(rng, 60, crops_dir, prep_dir)
    total = len([f for f in os.listdir(prep_dir) if f.startswith("synth_")])
    print(f"[synth] wrote {total} hard-clean tensors -> {prep_dir}")


if __name__ == "__main__":
    main()
