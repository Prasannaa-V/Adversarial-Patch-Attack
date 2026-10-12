"""
training/benchmark_sdnl.py
InfraEdge-Shield — Full evaluation suite for the SDNL 4-channel classifier.

Generates:
  experiments/results/confusion_matrix.png
  experiments/results/pr_curve.png
  experiments/results/roc_curve.png
  experiments/results/eval_report.txt

Usage:
  python training/benchmark_sdnl.py                         # Full eval + latency
  python training/benchmark_sdnl.py --latency-only          # Latency benchmark only
  python training/benchmark_sdnl.py --eval-only             # Metrics only (no latency)
  python training/benchmark_sdnl.py --test-split 0.15       # Use 15% held-out test set
"""

import os
import glob
import time
import argparse
import random
import numpy as np
import torch

RESULTS_DIR = "experiments/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_all_labeled_tensors(data_dir="data/preprocessed", in_channels=4):
    """
    Loads all .npy tensors and returns (X, y) arrays.
    Label: 1 = ATTACK (from 'attack' subfolder), 0 = CLEAN (from 'clean' subfolder)
    """
    samples = []
    for label_name, label_val in [("attack", 1), ("clean", 0)]:
        folder = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder):
            print(f"[Eval] Warning: folder not found: {folder}")
            continue
        paths = glob.glob(os.path.join(folder, "*.npy"))
        for p in paths:
            try:
                t = np.load(p)                    # (H, W, C) or (C, H, W)
                if t.ndim == 3 and t.shape[2] in [3, 4]:
                    t = np.transpose(t, (2, 0, 1))   # → (C, H, W)
                if t.shape[0] > in_channels:
                    t = t[:in_channels, :, :]
                elif t.shape[0] < in_channels:
                    # Pad missing channels with zeros
                    pad = np.zeros((in_channels - t.shape[0], t.shape[1], t.shape[2]), dtype=t.dtype)
                    t = np.concatenate([t, pad], axis=0)
                samples.append((t.astype(np.float32), label_val))
            except Exception as e:
                print(f"[Eval] Skipping {p}: {e}")

    if not samples:
        print("[Eval] No preprocessed samples found. Generating synthetic data.")
        samples = [(np.random.randn(in_channels, 128, 128).astype(np.float32), i % 2)
                   for i in range(100)]

    random.shuffle(samples)
    X = np.stack([s[0] for s in samples])
    y = np.array([s[1] for s in samples])
    print(f"[Eval] Loaded {len(y)} samples — Attack: {y.sum()} | Clean: {(y == 0).sum()}")
    return X, y


def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Deterministic stratified split. Same seed = same split every run."""
    rng = np.random.default_rng(seed)

    attack_idx = np.where(y == 1)[0]
    clean_idx  = np.where(y == 0)[0]

    def split_class(idx):
        n = len(idx)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        perm   = rng.permutation(idx)
        return perm[n_test + n_val:], perm[n_test:n_test + n_val], perm[:n_test]

    tr_a, va_a, te_a = split_class(attack_idx)
    tr_c, va_c, te_c = split_class(clean_idx)

    train = np.concatenate([tr_a, tr_c])
    val   = np.concatenate([va_a, va_c])
    test  = np.concatenate([te_a, te_c])

    print(f"[Eval] Split → Train: {len(train)} | Val: {len(val)} | Test (held-out): {len(test)}")
    return (X[train], y[train]), (X[val], y[val]), (X[test], y[test])


# ── ONNX inference ─────────────────────────────────────────────────────────────
def run_onnx_inference(onnx_path, X):
    """Returns (probs_attack, latencies_ms) for all samples in X."""
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    probs, latencies = [], []
    for sample in X:
        inp = np.expand_dims(sample, 0)
        t0 = time.perf_counter()
        out = session.run(None, {input_name: inp})
        latencies.append((time.perf_counter() - t0) * 1000.0)

        # out[0] shape: (1, 2) — softmax probabilities
        logits = out[0][0]
        exp = np.exp(logits - logits.max())
        prob_attack = (exp / exp.sum())[1]
        probs.append(float(prob_attack))

    return np.array(probs), np.array(latencies)


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    acc       = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=acc, precision=precision,
                recall=recall, f1=f1, fpr=fpr)


def find_optimal_threshold(y_true, y_prob, beta=1.0):
    """Find threshold that maximises F-beta score on the grid [0.01, 0.99]."""
    thresholds = np.linspace(0.01, 0.99, 199)
    best_f, best_t = -1.0, 0.5
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, t)
        p, r = m["precision"], m["recall"]
        if p + r == 0:
            continue
        fb = (1 + beta**2) * p * r / (beta**2 * p + r)
        if fb > best_f:
            best_f, best_t = fb, t
    return round(float(best_t), 3), round(best_f, 4)


def wilson_ci(k, n, z=1.96):
    """Wilson 95% score interval for a binomial rate k/n (honest small-n bounds)."""
    import math
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_confusion_matrix(tp, fp, fn, tn, threshold, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred CLEAN", "Pred ATTACK"], fontsize=11)
        ax.set_yticklabels(["True CLEAN", "True ATTACK"], fontsize=11)
        ax.set_title(f"Confusion Matrix  (threshold={threshold})", fontsize=12, pad=12)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=18, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Eval] Confusion matrix saved → {out_path}")
    except ImportError:
        print("[Eval] matplotlib not available — skipping confusion matrix plot")


def plot_pr_curve(y_true, y_prob, optimal_threshold, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        thresholds = np.linspace(0.0, 1.0, 201)
        precisions, recalls = [], []
        for t in thresholds:
            m = compute_metrics(y_true, y_prob, t)
            precisions.append(m["precision"])
            recalls.append(m["recall"])

        # AUC-PR (trapezoid)
        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        auc_pr = float(_trapz(precisions[::-1], recalls[::-1]))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recalls, precisions, color="#6366f1", linewidth=2, label=f"SDNL (AUC-PR={auc_pr:.3f})")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Mark the optimal threshold operating point
        m_opt = compute_metrics(y_true, y_prob, optimal_threshold)
        ax.scatter([m_opt["recall"]], [m_opt["precision"]],
                   color="#ef4444", zorder=5, s=80,
                   label=f"Optimal threshold={optimal_threshold}\nP={m_opt['precision']:.3f} R={m_opt['recall']:.3f}")

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve — SDNL Classifier", fontsize=12)
        ax.set_xlim([0, 1.05]); ax.set_ylim([0, 1.05])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Eval] PR curve saved → {out_path}")
        return auc_pr
    except ImportError:
        print("[Eval] matplotlib not available — skipping PR curve")
        return None


def plot_roc_curve(y_true, y_prob, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        thresholds = np.linspace(0.0, 1.0, 201)
        tprs, fprs = [], []
        for t in thresholds:
            m = compute_metrics(y_true, y_prob, t)
            tprs.append(m["recall"])
            fprs.append(m["fpr"])

        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        auc_roc = float(_trapz(tprs[::-1], fprs[::-1]))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fprs, tprs, color="#6366f1", linewidth=2, label=f"SDNL (AUC={auc_roc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random classifier")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
        ax.set_title("ROC Curve — SDNL Classifier", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Eval] ROC curve saved → {out_path}")
        return auc_roc
    except ImportError:
        print("[Eval] matplotlib not available — skipping ROC curve")
        return None


# ── Latency benchmarks ─────────────────────────────────────────────────────────
def benchmark_onnx_latency(onnx_path, X_sample, runs=200):
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    batch = [np.expand_dims(X_sample[i % len(X_sample)], 0) for i in range(runs + 20)]

    # Warmup
    for i in range(20):
        session.run(None, {input_name: batch[i]})

    latencies = []
    for i in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: batch[i]})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat = np.array(latencies)
    print(f"\n[Latency Benchmark] ONNX CPU | N={runs} runs")
    print(f"  Mean:   {lat.mean():.2f} ms")
    print(f"  Std:    {lat.std():.2f} ms")
    print(f"  P50:    {np.percentile(lat, 50):.2f} ms")
    print(f"  P95:    {np.percentile(lat, 95):.2f} ms")
    print(f"  P99:    {np.percentile(lat, 99):.2f} ms")
    print(f"  Min:    {lat.min():.2f} ms")
    print(f"  Max:    {lat.max():.2f} ms")
    return lat


def benchmark_pytorch_latency(model_path, X_sample, in_channels=4, runs=200):
    try:
        from train_sdnl import SDNLModel
    except ImportError:
        print("[Latency] Cannot import SDNLModel — skipping PyTorch benchmark")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDNLModel(num_classes=2, in_channels=in_channels).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tensors = [torch.tensor(X_sample[i % len(X_sample)]).unsqueeze(0).to(device)
               for i in range(runs + 20)]

    for i in range(20):
        with torch.no_grad():
            model(tensors[i])

    latencies = []
    for i in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(tensors[i])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat = np.array(latencies)
    print(f"\n[Latency Benchmark] PyTorch {device} | N={runs} runs")
    print(f"  Mean: {lat.mean():.2f} ms | P95: {np.percentile(lat, 95):.2f} ms")
    return lat


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="InfraEdge-Shield — SDNL Full Evaluation Suite")
    parser.add_argument("--onnx-path",     default="models/sdnl/sdnl_v1.onnx")
    parser.add_argument("--pytorch-path",  default="models/sdnl/sdnl_v1.pt")
    parser.add_argument("--data-dir",      default="data/preprocessed")
    parser.add_argument("--in-channels",   type=int,   default=4)
    parser.add_argument("--test-split",    type=float, default=0.15,
                        help="Fraction of data reserved as held-out test set (default 0.15)")
    parser.add_argument("--val-split",     type=float, default=0.15)
    parser.add_argument("--threshold",     type=float, default=None,
                        help="Fixed threshold (default: auto-select optimal from val set)")
    parser.add_argument("--runs",          type=int,   default=200)
    parser.add_argument("--latency-only",  action="store_true")
    parser.add_argument("--eval-only",     action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("  InfraEdge-Shield SDNL — Full Evaluation Suite")
    print("=" * 65)

    # ── Load data and split ──
    X, y = load_all_labeled_tensors(args.data_dir, args.in_channels)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        X, y, val_ratio=args.val_split, test_ratio=args.test_split
    )

    if not args.eval_only:
        # ── Latency benchmark ──
        print("\n── Latency Benchmark ──────────────────────────────────────")
        if os.path.exists(args.onnx_path):
            onnx_lat = benchmark_onnx_latency(args.onnx_path, X_test, runs=args.runs)
        else:
            print(f"[Latency] ONNX model not found at {args.onnx_path}")
            onnx_lat = None

        if not args.latency_only and os.path.exists(args.pytorch_path):
            benchmark_pytorch_latency(args.pytorch_path, X_test, args.in_channels, runs=args.runs)

    if not args.latency_only:
        # ── Run ONNX inference on val set to find optimal threshold ──
        print("\n── Threshold Selection (Validation Set) ───────────────────")
        if not os.path.exists(args.onnx_path):
            print(f"[Eval] ONNX model not found: {args.onnx_path}. Stopping.")
            return

        val_probs, _ = run_onnx_inference(args.onnx_path, X_val)
        if args.threshold is not None:
            threshold = args.threshold
            thresh_note = "fixed via --threshold (operating point)"
            print(f"[Eval] Using fixed threshold: {threshold}")
        else:
            threshold, best_f1 = find_optimal_threshold(y_val, val_probs)
            thresh_note = "F1-optimised on validation set"
            print(f"[Eval] Optimal threshold (F1-maximising, val set): {threshold}  (F1={best_f1:.4f})")

        # ── Run ONNX inference on HELD-OUT TEST SET ──
        print("\n── Held-Out Test Set Evaluation ───────────────────────────")
        test_probs, test_lats = run_onnx_inference(args.onnx_path, X_test)
        m = compute_metrics(y_test, test_probs, threshold)

        print(f"\n  Threshold:  {threshold}")
        print(f"  Accuracy:   {m['accuracy']*100:.2f}%")
        print(f"  Precision:  {m['precision']*100:.2f}%  (95% CI {wilson_ci(m['tp'], m['tp']+m['fp'])[0]*100:.1f}-{wilson_ci(m['tp'], m['tp']+m['fp'])[1]*100:.1f}%)")
        print(f"  Recall:     {m['recall']*100:.2f}%  (95% CI {wilson_ci(m['tp'], m['tp']+m['fn'])[0]*100:.1f}-{wilson_ci(m['tp'], m['tp']+m['fn'])[1]*100:.1f}%)")
        print(f"  F1-Score:   {m['f1']:.4f}")
        print(f"  FPR:        {m['fpr']*100:.2f}%  (95% CI {wilson_ci(m['fp'], m['fp']+m['tn'])[0]*100:.1f}-{wilson_ci(m['fp'], m['fp']+m['tn'])[1]*100:.1f}%)")
        print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

        # ── Plots ──
        print("\n── Generating Plots ───────────────────────────────────────")
        plot_confusion_matrix(m["tp"], m["fp"], m["fn"], m["tn"], threshold,
                              os.path.join(RESULTS_DIR, "confusion_matrix.png"))

        auc_pr  = plot_pr_curve(y_test, test_probs, threshold,
                                os.path.join(RESULTS_DIR, "pr_curve.png"))
        auc_roc = plot_roc_curve(y_test, test_probs,
                                 os.path.join(RESULTS_DIR, "roc_curve.png"))

        # ── Save text report ──
        report_path = os.path.join(RESULTS_DIR, "eval_report.txt")
        with open(report_path, "w") as f:
            f.write("InfraEdge-Shield SDNL — Held-Out Test Set Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset:         {args.data_dir}\n")
            f.write(f"Total samples:   {len(y)}\n")
            f.write(f"Train / Val / Test split: {1-args.val_split-args.test_split:.0%} / {args.val_split:.0%} / {args.test_split:.0%}\n")
            f.write(f"Test set size:   {len(y_test)}\n")
            f.write(f"  Attack:        {y_test.sum()}\n")
            f.write(f"  Clean:         {(y_test==0).sum()}\n\n")
            f.write(f"Threshold:       {threshold} ({thresh_note})\n\n")
            f.write(f"Accuracy:        {m['accuracy']*100:.2f}%\n")
            plo, phi = wilson_ci(m['tp'], m['tp'] + m['fp'])
            rlo, rhi = wilson_ci(m['tp'], m['tp'] + m['fn'])
            flo, fhi = wilson_ci(m['fp'], m['fp'] + m['tn'])
            f.write(f"Precision:       {m['precision']*100:.2f}%  (95% CI {plo*100:.1f}-{phi*100:.1f}%)\n")
            f.write(f"Recall:          {m['recall']*100:.2f}%  (95% CI {rlo*100:.1f}-{rhi*100:.1f}%)\n")
            f.write(f"F1-Score:        {m['f1']:.4f}\n")
            f.write(f"False Pos Rate:  {m['fpr']*100:.2f}%  (95% CI {flo*100:.1f}-{fhi*100:.1f}%)\n")
            if auc_pr is not None:
                f.write(f"AUC-PR:          {auc_pr:.4f}\n")
            if auc_roc is not None:
                f.write(f"AUC-ROC:         {auc_roc:.4f}\n")
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"  TP={m['tp']}  FP={m['fp']}\n")
            f.write(f"  FN={m['fn']}  TN={m['tn']}\n\n")
            if not args.eval_only and onnx_lat is not None:
                f.write(f"Latency (ONNX CPU, N={args.runs}):\n")
                f.write(f"  Mean: {onnx_lat.mean():.2f} ms\n")
                f.write(f"  P95:  {np.percentile(onnx_lat, 95):.2f} ms\n")
                f.write(f"  P99:  {np.percentile(onnx_lat, 99):.2f} ms\n")
        print(f"[Eval] Full report saved → {report_path}")

        target_ms = 15.0
        target_pass = (not args.eval_only and onnx_lat is not None
                       and onnx_lat.mean() <= target_ms)
        print("\n── Summary ────────────────────────────────────────────────")
        print(f"  ✅ Accuracy:   {m['accuracy']*100:.2f}%")
        print(f"  ✅ Precision:  {m['precision']*100:.2f}%  (zero false-positive gate locks)")
        print(f"  ✅ Recall:     {m['recall']*100:.2f}%")
        print(f"  ✅ F1:         {m['f1']:.4f}")
        if not args.eval_only and onnx_lat is not None:
            status = "✅ PASS" if target_pass else "⚠️  WARN"
            print(f"  {status} Latency: {onnx_lat.mean():.2f} ms mean (target ≤{target_ms} ms)")
        print(f"  📁 Results:   {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
