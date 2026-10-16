import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_tiny_imagenet import get_tiny_imagenet_loaders
from multi_exit_model import MultiExitConvNeXt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./checkpoints/multi_exit_tiny_imagenet_best.pt"

# Profiled FLOPs for Tiny-ImageNet (64x64 resolution, 200 classes)
F1 = 0.1254 * 1e9  # Exit 1 (Stage 1 & 2)
F2 = 0.3031 * 1e9  # Exit 2 (Stage 3)
F3 = 0.3650 * 1e9  # Exit 3 (Stage 4)

def main():
    print("=======================================================")
    print("Ablation 1: Tiny-ImageNet-200 Pareto Frontier")
    print("=======================================================")
    os.makedirs("./results", exist_ok=True)

    _, val_loader = get_tiny_imagenet_loaders(batch_size=64)

    model = MultiExitConvNeXt(num_classes=200).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True), strict=False)
    model.eval()

    all_l1, all_l2, all_l3, all_targets = [], [], [], []
    print("\n--- 1. Gathering Validation Logits ---")
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Forward Pass (Tiny-ImageNet)"):
            images = images.to(DEVICE)
            l1, l2, l3 = model(images)
            all_l1.append(l1.cpu())
            all_l2.append(l2.cpu())
            all_l3.append(l3.cpu())
            all_targets.append(targets)

    logits1 = torch.cat(all_l1)
    logits2 = torch.cat(all_l2)
    logits3 = torch.cat(all_l3)
    targets = torch.cat(all_targets)
    total_samples = len(targets)

    conf1, preds1 = torch.max(F.softmax(logits1, dim=1), dim=1)
    conf2, preds2 = torch.max(F.softmax(logits2, dim=1), dim=1)
    preds3 = logits3.argmax(dim=1)

    thresholds = [0.0, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98, 1.00]
    flops_list, acc_list = [], []

    print("\n--- 2. Sweeping Threshold Grid ---")
    print(f"{'Tau':<6} | {'Top-1 Acc':<10} | {'Avg GFLOPs':<11} | {'Exit 1 %':<10} | {'Exit 2 %':<10} | {'Exit 3 %':<10}")
    print("-" * 72)

    for tau in thresholds:
        if tau >= 1.00:
            e1 = torch.zeros(total_samples, dtype=torch.bool)
            e2 = torch.zeros(total_samples, dtype=torch.bool)
            e3 = torch.ones(total_samples, dtype=torch.bool)
        elif tau <= 0.0:
            e1 = torch.ones(total_samples, dtype=torch.bool)
            e2 = torch.zeros(total_samples, dtype=torch.bool)
            e3 = torch.zeros(total_samples, dtype=torch.bool)
        else:
            e1 = conf1 >= tau
            e2 = (~e1) & (conf2 >= tau)
            e3 = ~(e1 | e2)

        correct = (preds1[e1] == targets[e1]).sum().item() + \
                  (preds2[e2] == targets[e2]).sum().item() + \
                  (preds3[e3] == targets[e3]).sum().item()

        acc = (correct / total_samples) * 100.0
        p1 = e1.sum().item() / total_samples
        p2 = e2.sum().item() / total_samples
        p3 = e3.sum().item() / total_samples

        avg_flops = (p1 * F1) + (p2 * F2) + (p3 * F3)
        flops_list.append(avg_flops / 1e9)
        acc_list.append(acc)

        print(f"{tau:<6.2f} | {acc:<9.2f}% | {avg_flops/1e9:<11.4f} | {p1*100:<9.1f}% | {p2*100:<9.1f}% | {p3*100:<9.1f}%")

    # Static Baselines
    acc_static_e1 = (preds1 == targets).sum().item() / total_samples * 100.0
    acc_static_e2 = (preds2 == targets).sum().item() / total_samples * 100.0
    acc_static_e3 = (preds3 == targets).sum().item() / total_samples * 100.0

    # Plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(flops_list, acc_list, marker="o", linewidth=2.4, color="#1f77b4", label="Adaptive Multi-Exit Frontier")

    plt.scatter([F1/1e9], [acc_static_e1], color="orange", marker="^", s=140, zorder=5, label=f"Static Exit 1 ({acc_static_e1:.1f}%)")
    plt.scatter([F2/1e9], [acc_static_e2], color="green", marker="D", s=120, zorder=5, label=f"Static Exit 2 ({acc_static_e2:.1f}%)")
    plt.scatter([F3/1e9], [acc_static_e3], color="red", marker="*", s=220, zorder=5, label=f"Static Exit 3 Full ({acc_static_e3:.1f}%)")

    plt.title("Pareto Frontier: Accuracy vs. Compute (Tiny-ImageNet Ablation)", fontsize=13, fontweight="bold")
    plt.xlabel("Average Computation per Image (GFLOPs)", fontsize=11)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="lower right", fontsize=10)
    plt.tight_layout()

    out_fig = "./results/pareto_tiny_imagenet.png"
    plt.savefig(out_fig)
    print(f"\nPareto plot saved to {out_fig}")

if __name__ == "__main__":
    main()