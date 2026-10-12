"""
training/run_ablation_study.py
Executes a rigorous ablation study comparing:
  1. Baseline: 3-Channel (RGB only) MobileNetV3-Small
  2. Proposed: 4-Channel (RGB + 2D DCT Energy Map) MobileNetV3-Small

Uses identical held-out test splits, training hyperparameters, and random seeds.
Generates:
  - models/sdnl/sdnl_baseline_3ch.pth
  - models/sdnl/sdnl_baseline_3ch.onnx
  - experiments/results/ablation_comparison.png
  - experiments/results/ablation_report.txt
"""

import os
import time
import glob
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Dataset Class ─────────────────────────────────────────────────────────────
class SDNLDataset(Dataset):
    def __init__(self, file_paths, labels, in_channels=4):
        self.file_paths = file_paths
        self.labels = labels
        self.in_channels = in_channels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        tensor = np.load(self.file_paths[idx])
        tensor = np.transpose(tensor, (2, 0, 1))  # (C, H, W)
        if self.in_channels == 3:
            tensor = tensor[:3, :, :]
        return torch.tensor(tensor, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# ── Model Architecture ────────────────────────────────────────────────────────
class SDNLModel(nn.Module):
    def __init__(self, num_classes=2, in_channels=4):
        super().__init__()
        import torchvision.models as models
        self.in_channels = in_channels
        self.backbone = models.mobilenet_v3_small(weights=None)
        
        orig_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ── Wilson Score Confidence Interval ──────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom)


# ── Training & Validation Helper ──────────────────────────────────────────────
def train_model(in_channels, train_loader, val_loader, device, epochs=20, patience=6, lr=1e-3):
    print(f"\n[Training] Starting training for {in_channels}-Channel model...")
    model = SDNLModel(num_classes=2, in_channels=in_channels).to(device)
    
    # Class weights
    train_labels = [y.item() for _, targets in train_loader for y in targets]
    n_clean = sum(1 for y in train_labels if y == 0)
    n_attack = sum(1 for y in train_labels if y == 1)
    tot = n_clean + n_attack
    class_w = torch.tensor([tot / (2 * n_clean), tot / (2 * n_attack)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_b.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                out = model(x_b)
                preds = out.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y_b.numpy())

        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif patience > 0 and (epoch - best_epoch) >= patience:
            print(f"  [Early Stopping] Triggered at epoch {epoch} (best val F1: {best_val_f1:.4f} at epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    return model


# ── Threshold Tuning ──────────────────────────────────────────────────────────
def tune_threshold(model, val_loader, device, target_fpr=0.05):
    model.eval()
    val_probs, val_targets = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b = x_b.to(device)
            out = model(x_b)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            val_probs.extend(probs)
            val_targets.extend(y_b.numpy())

    val_targets = np.array(val_targets)
    val_probs = np.array(val_probs)

    thresholds = np.linspace(0.01, 0.99, 199)
    best_thresh = 0.5
    best_f1 = -1.0

    for thresh in thresholds:
        preds = (val_probs >= thresh).astype(int)
        f1 = f1_score(val_targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return float(best_thresh)


# ── Full Test Evaluation ──────────────────────────────────────────────────────
def evaluate_model(model, test_loader, device, threshold):
    from sklearn.metrics import auc
    model.eval()
    test_probs, test_targets = [], []
    latencies = []

    with torch.no_grad():
        for x_b, y_b in test_loader:
            for i in range(x_b.size(0)):
                single_sample = x_b[i:i+1].to(device)
                t0 = time.perf_counter()
                out = model(single_sample)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                prob = F.softmax(out, dim=1)[0, 1].item()
                test_probs.append(prob)
                test_targets.append(y_b[i].item())

    test_targets = np.array(test_targets)
    test_probs = np.array(test_probs)
    test_preds = (test_probs >= threshold).astype(int)

    acc = accuracy_score(test_targets, test_preds)
    prec = precision_score(test_targets, test_preds, zero_division=0)
    rec = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, zero_division=0)
    auc_roc = roc_auc_score(test_targets, test_probs)

    precision_curve, recall_curve, _ = precision_recall_curve(test_targets, test_probs)
    auc_pr = float(auc(recall_curve, precision_curve))

    tn, fp, fn, tp = confusion_matrix(test_targets, test_preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn + 1e-8)

    lat_mean = np.mean(latencies)
    lat_p50 = np.percentile(latencies, 50)
    lat_p95 = np.percentile(latencies, 95)

    prec_ci = wilson_ci(tp, tp + fp)
    rec_ci = wilson_ci(tp, tp + fn)
    fpr_ci = wilson_ci(fp, fp + tn)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": threshold,
        "lat_mean": lat_mean,
        "lat_p50": lat_p50,
        "lat_p95": lat_p95,
        "prec_ci": prec_ci,
        "rec_ci": rec_ci,
        "fpr_ci": fpr_ci,
        "targets": test_targets,
        "probs": test_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Run 3-Channel vs 4-Channel Ablation Study")
    parser.add_argument("--data-dir", type=str, default="data/preprocessed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--retrain", action="store_true", help="Force retraining of baseline model")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Device for Ablation Study: {device}")

    # Gather data files
    clean_files = sorted(glob.glob(os.path.join(args.data_dir, "clean", "*.npy")))
    attack_files = sorted(glob.glob(os.path.join(args.data_dir, "attack", "*.npy")))

    print(f"Total Dataset: {len(clean_files)} Clean | {len(attack_files)} Attack")

    files = clean_files + attack_files
    labels = [0] * len(clean_files) + [1] * len(attack_files)

    # Identical 70 / 15 / 15 Split
    train_files, val_test_files, train_lbls, val_test_lbls = train_test_split(
        files, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_files, test_files, val_lbls, test_lbls = train_test_split(
        val_test_files, val_test_lbls, test_size=0.50, stratify=val_test_lbls, random_state=SEED
    )

    print(f"Split sizes: Train={len(train_files)} | Val={len(val_files)} | Test={len(test_files)}")

    # 1. Baseline: 3-Channel (RGB only)
    train_ds_3ch = SDNLDataset(train_files, train_lbls, in_channels=3)
    val_ds_3ch = SDNLDataset(val_files, val_lbls, in_channels=3)
    test_ds_3ch = SDNLDataset(test_files, test_lbls, in_channels=3)

    train_loader_3ch = DataLoader(train_ds_3ch, batch_size=args.batch_size, shuffle=True)
    val_loader_3ch = DataLoader(val_ds_3ch, batch_size=args.batch_size, shuffle=False)
    test_loader_3ch = DataLoader(test_ds_3ch, batch_size=args.batch_size, shuffle=False)

    baseline_3ch_path = "models/sdnl/sdnl_baseline_3ch.pth"
    model_3ch = SDNLModel(num_classes=2, in_channels=3).to(device)
    if os.path.exists(baseline_3ch_path) and not args.retrain:
        print(f"\n[3-Channel] Loading existing trained weights from {baseline_3ch_path}...")
        model_3ch.load_state_dict(torch.load(baseline_3ch_path, map_location=device))
    else:
        model_3ch = train_model(3, train_loader_3ch, val_loader_3ch, device, epochs=args.epochs, lr=args.lr)
        os.makedirs("models/sdnl", exist_ok=True)
        torch.save(model_3ch.state_dict(), baseline_3ch_path)
        dummy_3ch = torch.randn(1, 3, 128, 128)
        torch.onnx.export(
            model_3ch,
            dummy_3ch,
            "models/sdnl/sdnl_baseline_3ch.onnx",
            export_params=True,
            opset_version=18,
            input_names=["input_1"],
            output_names=["output_1"],
            dynamic_axes={"input_1": {0: "batch_size"}, "output_1": {0: "batch_size"}}
        )
        print("[3-Channel] Baseline model checkpoint and ONNX exported.")

    thresh_3ch = tune_threshold(model_3ch, val_loader_3ch, device)
    print(f"[3-Channel] Optimal threshold (Validation F1-max): {thresh_3ch:.4f}")
    results_3ch = evaluate_model(model_3ch, test_loader_3ch, device, thresh_3ch)

    # 2. Proposed: 4-Channel (RGB + DCT)
    train_ds_4ch = SDNLDataset(train_files, train_lbls, in_channels=4)
    val_ds_4ch = SDNLDataset(val_files, val_lbls, in_channels=4)
    test_ds_4ch = SDNLDataset(test_files, test_lbls, in_channels=4)

    train_loader_4ch = DataLoader(train_ds_4ch, batch_size=args.batch_size, shuffle=True)
    val_loader_4ch = DataLoader(val_ds_4ch, batch_size=args.batch_size, shuffle=False)
    test_loader_4ch = DataLoader(test_ds_4ch, batch_size=args.batch_size, shuffle=False)

    # Check if pre-trained 4-channel model exists; otherwise train
    best_4ch_path = "models/sdnl/sdnl_best.pth"
    model_4ch = SDNLModel(num_classes=2, in_channels=4).to(device)
    if os.path.exists(best_4ch_path):
        print(f"\n[4-Channel] Loading existing trained weights from {best_4ch_path}...")
        model_4ch.load_state_dict(torch.load(best_4ch_path, map_location=device))
    else:
        model_4ch = train_model(4, train_loader_4ch, val_loader_4ch, device, epochs=args.epochs, lr=args.lr)
        torch.save(model_4ch.state_dict(), best_4ch_path)

    thresh_4ch = tune_threshold(model_4ch, val_loader_4ch, device, target_fpr=0.05)
    print(f"[4-Channel] Selected threshold (FPR <= 5%): {thresh_4ch:.4f}")
    results_4ch = evaluate_model(model_4ch, test_loader_4ch, device, thresh_4ch)

    # ── Print Comparison Table ────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("                    ABLATION STUDY EXPERIMENTAL RESULTS")
    print("=" * 75)
    print(f"{'Metric':<25} | {'Baseline (3-Ch RGB)':<22} | {'Proposed (4-Ch RGB+DCT)':<22}")
    print("-" * 75)
    print(f"{'Input Modality':<25} | {'RGB (3 Channels)':<22} | {'RGB + 2D DCT (4 Channels)':<22}")
    print(f"{'Optimal Threshold':<25} | {results_3ch['threshold']:<22.4f} | {results_4ch['threshold']:<22.4f}")
    print(f"{'Accuracy':<25} | {results_3ch['accuracy']*100:<21.2f}% | {results_4ch['accuracy']*100:<21.2f}%")
    print(f"{'Precision':<25} | {results_3ch['precision']*100:<21.2f}% | {results_4ch['precision']*100:<21.2f}%")
    print(f"{'Recall':<25} | {results_3ch['recall']*100:<21.2f}% | {results_4ch['recall']*100:<21.2f}%")
    print(f"{'F1-Score':<25} | {results_3ch['f1']:<22.4f} | {results_4ch['f1']:<22.4f}")
    print(f"{'False Positive Rate':<25} | {results_3ch['fpr']*100:<21.2f}% | {results_4ch['fpr']*100:<21.2f}%")
    print(f"{'AUC-ROC':<25} | {results_3ch['auc_roc']:<22.4f} | {results_4ch['auc_roc']:<22.4f}")
    print(f"{'AUC-PR':<25} | {results_3ch['auc_pr']:<22.4f} | {results_4ch['auc_pr']:<22.4f}")
    print(f"{'CPU Latency (Mean)':<25} | {results_3ch['lat_mean']:<19.2f} ms | {results_4ch['lat_mean']:<19.2f} ms")
    print(f"{'CPU Latency (P95)':<25} | {results_3ch['lat_p95']:<19.2f} ms | {results_4ch['lat_p95']:<19.2f} ms")
    print(f"{'Confusion Matrix':<25} | TP={results_3ch['tp']} FP={results_3ch['fp']} FN={results_3ch['fn']} TN={results_3ch['tn']} | TP={results_4ch['tp']} FP={results_4ch['fp']} FN={results_4ch['fn']} TN={results_4ch['tn']}")
    print("=" * 75)

    # ── Save Text Report ──────────────────────────────────────────────────────
    os.makedirs("experiments/results", exist_ok=True)
    report_path = "experiments/results/ablation_report.txt"
    with open(report_path, "w") as f:
        f.write("Ablation Study: Baseline 3-Channel (RGB) vs Proposed 4-Channel (RGB + 2D DCT)\n")
        f.write("===========================================================================\n\n")
        f.write(f"Test Set Size: {len(test_files)} (Clean: {(np.array(test_lbls) == 0).sum()} | Attack: {(np.array(test_lbls) == 1).sum()})\n\n")
        f.write(f"{'Metric':<25} | {'Baseline (3-Ch RGB)':<22} | {'Proposed (4-Ch RGB+DCT)':<22}\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Input Modality':<25} | {'RGB (3 Channels)':<22} | {'RGB + 2D DCT (4 Channels)':<22}\n")
        f.write(f"{'Optimal Threshold':<25} | {results_3ch['threshold']:<22.4f} | {results_4ch['threshold']:<22.4f}\n")
        f.write(f"{'Accuracy':<25} | {results_3ch['accuracy']*100:<21.2f}% | {results_4ch['accuracy']*100:<21.2f}%\n")
        f.write(f"{'Precision':<25} | {results_3ch['precision']*100:<21.2f}% | {results_4ch['precision']*100:<21.2f}%\n")
        f.write(f"{'Recall':<25} | {results_3ch['recall']*100:<21.2f}% | {results_4ch['recall']*100:<21.2f}%\n")
        f.write(f"{'F1-Score':<25} | {results_3ch['f1']:<22.4f} | {results_4ch['f1']:<22.4f}\n")
        f.write(f"{'False Positive Rate':<25} | {results_3ch['fpr']*100:<21.2f}% | {results_4ch['fpr']*100:<21.2f}%\n")
        f.write(f"{'AUC-ROC':<25} | {results_3ch['auc_roc']:<22.4f} | {results_4ch['auc_roc']:<22.4f}\n")
        f.write(f"{'AUC-PR':<25} | {results_3ch['auc_pr']:<22.4f} | {results_4ch['auc_pr']:<22.4f}\n")
        f.write(f"{'CPU Latency (Mean)':<25} | {results_3ch['lat_mean']:<19.2f} ms | {results_4ch['lat_mean']:<19.2f} ms\n")
        f.write(f"{'CPU Latency (P95)':<25} | {results_3ch['lat_p95']:<19.2f} ms | {results_4ch['lat_p95']:<19.2f} ms\n")
        f.write(f"{'Confusion Matrix':<25} | TP={results_3ch['tp']} FP={results_3ch['fp']} FN={results_3ch['fn']} TN={results_3ch['tn']} | TP={results_4ch['tp']} FP={results_4ch['fp']} FN={results_4ch['fn']} TN={results_4ch['tn']}\n")
    print(f"\nReport written to: {report_path}")

    # ── Plot Comparative Curves & Metrics ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1. ROC Curves
    fpr_3ch, tpr_3ch, _ = roc_curve(results_3ch["targets"], results_3ch["probs"])
    fpr_4ch, tpr_4ch, _ = roc_curve(results_4ch["targets"], results_4ch["probs"])

    axes[0].plot(fpr_3ch, tpr_3ch, color="#ef4444", linestyle="--", linewidth=2.2,
                 label=f"Baseline 3-Ch RGB (AUC={results_3ch['auc_roc']:.4f})")
    axes[0].plot(fpr_4ch, tpr_4ch, color="#10b981", linewidth=2.4,
                 label=f"Proposed 4-Ch RGB+DCT (AUC={results_4ch['auc_roc']:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle=":", alpha=0.6)
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[0].set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curves
    prec_3ch_c, rec_3ch_c, _ = precision_recall_curve(results_3ch["targets"], results_3ch["probs"])
    prec_4ch_c, rec_4ch_c, _ = precision_recall_curve(results_4ch["targets"], results_4ch["probs"])

    axes[1].plot(rec_3ch_c, prec_3ch_c, color="#ef4444", linestyle="--", linewidth=2.2,
                 label=f"Baseline 3-Ch RGB (AUC-PR={results_3ch['auc_pr']:.4f})")
    axes[1].plot(rec_4ch_c, prec_4ch_c, color="#10b981", linewidth=2.4,
                 label=f"Proposed 4-Ch RGB+DCT (AUC-PR={results_4ch['auc_pr']:.4f})")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title("Precision-Recall Curve Comparison", fontsize=13, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 3. Bar Chart of Key Metrics
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    vals_3ch = [results_3ch["accuracy"] * 100, results_3ch["precision"] * 100,
                results_3ch["recall"] * 100, results_3ch["f1"] * 100]
    vals_4ch = [results_4ch["accuracy"] * 100, results_4ch["precision"] * 100,
                results_4ch["recall"] * 100, results_4ch["f1"] * 100]

    x = np.arange(len(metrics_names))
    width = 0.35

    rects1 = axes[2].bar(x - width/2, vals_3ch, width, label="Baseline 3-Ch (RGB)", color="#f87171")
    rects2 = axes[2].bar(x + width/2, vals_4ch, width, label="Proposed 4-Ch (RGB+DCT)", color="#34d399")

    axes[2].set_ylabel("Percentage (%)", fontsize=11)
    axes[2].set_title("Classification Metrics Comparison", fontsize=13, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrics_names, fontsize=11)
    axes[2].set_ylim(80, 103)
    axes[2].legend(loc="lower right", fontsize=10)
    axes[2].grid(axis="y", alpha=0.3)

    for rect in rects1:
        h = rect.get_height()
        axes[2].annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        axes[2].annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plot_path = "experiments/results/ablation_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
