"""
training/train_sdnl.py
Trains the 4-channel SDNL model (MobileNetV3-Small backbone) on preprocessed tensors.
Includes FocalLoss option, FPR-controlled threshold tuning (<=5% FPR target),
ablation support (3-channel vs 4-channel), ONNX export, and PyTorch checkpoint saving.
"""

import os
import argparse
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# ── Focal Loss Implementation ──────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ── Dataset Class for Preprocessed Tensors ────────────────────────────────────
class SDNLDataset(Dataset):
    def __init__(self, file_paths, labels, in_channels=4):
        self.file_paths = file_paths
        self.labels = labels
        self.in_channels = in_channels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load preprocessed (128, 128, 4) numpy array
        tensor = np.load(self.file_paths[idx])
        # PyTorch expects shape (Channels, Height, Width)
        tensor = np.transpose(tensor, (2, 0, 1))
        
        if self.in_channels == 3:
            # Slice RGB channels only for ablation
            tensor = tensor[:3, :, :]

        return torch.tensor(tensor, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# ── Model Architecture (MobileNetV3-Small) ────────────────────────────────────
class SDNLModel(nn.Module):
    def __init__(self, num_classes=2, in_channels=4):
        super().__init__()
        import torchvision.models as models
        
        self.in_channels = in_channels
        self.backbone = models.mobilenet_v3_small(weights=None)
        
        # Modify first conv layer for in_channels (3 or 4)
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Replace classifier head with lightweight sub-500K params fc layers
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ── Threshold Tuning (≤ 5% False Positive Rate Target) ────────────────────────
def tune_confidence_threshold(val_targets, val_probs, target_fpr=0.05):
    thresholds = np.linspace(0.01, 0.99, 100)
    best_threshold = 0.5
    best_tpr = 0.0
    
    val_targets = np.array(val_targets)
    val_probs = np.array(val_probs)
    
    for thresh in thresholds:
        preds = (val_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(val_targets, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn + 1e-8)
        tpr = tp / (tp + fn + 1e-8)
        
        if fpr <= target_fpr and tpr >= best_tpr:
            best_tpr = tpr
            best_threshold = thresh

    print(f"[Threshold Tuning] Optimal threshold for FPR <= {target_fpr*100:.1f}%: {best_threshold:.4f} (Val TPR: {best_tpr:.4f})")
    return best_threshold


# ── Training Loop ─────────────────────────────────────────────────────────────
def train(args):
    # Set random seeds for 100% reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"=== Starting SDNL ({args.in_channels}-Channel) Model Training ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Gather dataset file paths
    clean_files = glob.glob(os.path.join(args.data_dir, "clean", "*.npy"))
    attack_files = glob.glob(os.path.join(args.data_dir, "attack", "*.npy"))
    # Optional extra attack dirs (e.g. stress composites): train pool only,
    # kept out of data_dir/attack/ so the official benchmark pool is stable.
    for extra in (args.extra_attack or []):
        attack_files += glob.glob(os.path.join(extra, "*.npy"))
    
    if not clean_files or not attack_files:
        print(f"[ERROR] No preprocessed files found in data_dir: {args.data_dir}")
        print("Please run data_prep.py first.")
        return

    print(f"Clean samples: {len(clean_files)} | Attack samples: {len(attack_files)}")

    files = clean_files + attack_files
    labels = [0] * len(clean_files) + [1] * len(attack_files)

    # Train / Val / Test Split (70 / 15 / 15)
    train_files, val_test_files, train_lbls, val_test_lbls = train_test_split(
        files, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_files, test_files, val_lbls, test_lbls = train_test_split(
        val_test_files, val_test_lbls, test_size=0.50, stratify=val_test_lbls, random_state=42
    )

    print(f"Split Sizes | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # Dataloaders
    train_ds = SDNLDataset(train_files, train_lbls, in_channels=args.in_channels)
    val_ds = SDNLDataset(val_files, val_lbls, in_channels=args.in_channels)
    test_ds = SDNLDataset(test_files, test_lbls, in_channels=args.in_channels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = SDNLModel(num_classes=2, in_channels=args.in_channels).to(device)
    
    # Class-weighted loss: attack-heavy pools otherwise bias the model toward ATTACK.
    n_clean = int(np.sum(np.array(train_lbls) == 0))
    n_attack = int(np.sum(np.array(train_lbls) == 1))
    if args.no_class_weight or min(n_clean, n_attack) == 0:
        class_w = None
        print("Class weights: OFF")
    else:
        tot = n_clean + n_attack
        class_w = torch.tensor([tot / (2 * n_clean), tot / (2 * n_attack)],
                               dtype=torch.float32).to(device)
        print(f"Class weights: clean={class_w[0]:.3f} attack={class_w[1]:.3f} "
              f"(train {n_clean} clean / {n_attack} attack)")

    # Loss function selection
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Loss Function: FocalLoss (alpha=0.25, gamma=2.0)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_w)
        print(f"Loss Function: CrossEntropyLoss{' (class-weighted)' if class_w is not None else ''}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training (strict-> best checkpoint + patience early stopping: the old
    # >= comparison kept overwriting ties and shipped the LAST epoch)
    best_val_f1 = 0.0
    best_epoch = 0
    patience = args.patience
    model_save_dir = "models/sdnl"
    os.makedirs(model_save_dir, exist_ok=True)
    pytorch_model_path = os.path.join(model_save_dir, "sdnl_v1.pt")
    best_chkpt_path = os.path.join(model_save_dir, "sdnl_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_probs, val_targets = [], [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)

        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:  # strict: ties keep the EARLIER (less overfit) epoch
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_chkpt_path)
            torch.save(model.state_dict(), pytorch_model_path)
            print(f"  --> Saved best model checkpoint to '{pytorch_model_path}' (Val F1: {val_f1:.4f})")
        elif patience > 0 and (epoch - best_epoch) >= patience:
            print(f"  !! Early stopping: no val-F1 improvement for {patience} epochs "
                  f"(best {best_val_f1:.4f} @ epoch {best_epoch})")
            break

    # Optimal Threshold Selection for FPR <= 5%
    optimal_thresh = tune_confidence_threshold(val_targets, val_probs, target_fpr=0.05)

    # Evaluate on held-out Test Set
    print("\n=== Evaluating Best Model on Test Set ===")
    model.load_state_dict(torch.load(best_chkpt_path))
    model.eval()
    
    test_probs, test_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            test_probs.extend(probs)
            test_targets.extend(targets.numpy())

    test_preds = (np.array(test_probs) >= optimal_thresh).astype(int)
    acc = accuracy_score(test_targets, test_preds)
    prec = precision_score(test_targets, test_preds, zero_division=0)
    rec = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, zero_division=0)
    auc = roc_auc_score(test_targets, test_probs) if len(set(test_targets)) > 1 else 0.5
    
    tn, fp, fn, tp = confusion_matrix(test_targets, test_preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn + 1e-8)

    print(f"Test Threshold: {optimal_thresh:.4f}")
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall:    {rec:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")
    print(f"Test ROC-AUC:   {auc:.4f}")
    print(f"Test FPR:       {fpr:.4f} (Target <= 0.05)")

    # Export to ONNX
    print("\n=== Exporting Model to ONNX ===")
    onnx_path = os.path.join(model_save_dir, "sdnl_v1.onnx")
    dummy_input = torch.randn(1, args.in_channels, 128, 128, dtype=torch.float32).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_1"],
        output_names=["output_1"],
        dynamic_axes={"input_1": {0: "batch_size"}, "output_1": {0: "batch_size"}}
    )
    print(f"ONNX model saved at '{onnx_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SDNL classifier")
    parser.add_argument("--data-dir", type=str, default="data/preprocessed", help="Path to preprocessed dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--in-channels", type=int, default=4, help="Input channels (3 for RGB, 4 for RGB+DCT)")
    parser.add_argument("--use-focal-loss", action="store_true", help="Use Focal Loss instead of CrossEntropy")
    parser.add_argument("--no-class-weight", action="store_true", help="Disable inverse-frequency class weights")
    parser.add_argument("--extra-attack", action="append", default=[], help="Extra attack .npy dir for the TRAIN pool only (e.g. data/preprocessed/stress_attack)")
    parser.add_argument("--patience", type=int, default=8, help="Early-stop after N epochs without val-F1 gain (0 = off)")
    args = parser.parse_args()
    
    train(args)
