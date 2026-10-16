import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from tqdm import tqdm

from dataset import get_pcam_loaders
from multi_exit_model import MultiExitConvNeXt

CHECKPOINT_FP32 = "./checkpoints/multi_exit_pcam_best.pt"
CHECKPOINT_INT8 = "./checkpoints/multi_exit_int8.pt"

# Profiled FLOPs for PCam (96x96, 2 classes)
F1 = 0.2821 * 1e9  # Stage 1 & 2
F2 = 0.6818 * 1e9  # Stage 3
F3 = 0.8210 * 1e9  # Stage 4

def load_int8_model(sample_batch, device):
    torch.backends.quantized.engine = "fbgemm"
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    shell = MultiExitConvNeXt().to(device)
    shell.eval()
    prepared = prepare_fx(shell, qconfig_mapping, (sample_batch,))
    int8_model = convert_fx(prepared)
    int8_model.load_state_dict(torch.load(CHECKPOINT_INT8, map_location=device, weights_only=True))
    int8_model.eval()
    return int8_model

def extract_logits(model, test_loader, device, desc):
    model.eval()
    l1_list, l2_list, l3_list, target_list = [], [], [], []
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc=desc):
            images = images.to(device)
            l1, l2, l3 = model(images)
            l1_list.append(l1.cpu())
            l2_list.append(l2.cpu())
            l3_list.append(l3.cpu())
            target_list.append(targets)
    return torch.cat(l1_list), torch.cat(l2_list), torch.cat(l3_list), torch.cat(target_list)

def compute_metrics(preds, targets):
    # Class 1 is Tumor, Class 0 is Normal
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()

    acc = (tp + tn) / len(targets) * 100.0
    recall = (tp / (tp + fn + 1e-8)) * 100.0
    precision = (tp / (tp + fp + 1e-8)) * 100.0
    f1 = (2 * precision * recall / (precision + recall + 1e-8))
    return acc, recall, f1

def sweep_clinical_thresholds(logits1, logits2, logits3, targets, thresholds):
    total = len(targets)
    conf1, preds1 = torch.max(F.softmax(logits1, dim=1), dim=1)
    conf2, preds2 = torch.max(F.softmax(logits2, dim=1), dim=1)
    preds3 = logits3.argmax(dim=1)

    flops_list, recalls, f1s, accs = [], [], [], []

    for tau in thresholds:
        if tau >= 1.00:
            e1 = torch.zeros(total, dtype=torch.bool)
            e2 = torch.zeros(total, dtype=torch.bool)
            e3 = torch.ones(total, dtype=torch.bool)
        elif tau <= 0.50:
            e1 = torch.ones(total, dtype=torch.bool)
            e2 = torch.zeros(total, dtype=torch.bool)
            e3 = torch.zeros(total, dtype=torch.bool)
        else:
            e1 = conf1 >= tau
            e2 = (~e1) & (conf2 >= tau)
            e3 = ~(e1 | e2)

        preds_routed = torch.empty_like(targets)
        preds_routed[e1] = preds1[e1]
        preds_routed[e2] = preds2[e2]
        preds_routed[e3] = preds3[e3]

        acc, rec, f1 = compute_metrics(preds_routed, targets)

        p1 = e1.sum().item() / total
        p2 = e2.sum().item() / total
        p3 = e3.sum().item() / total
        avg_flops = (p1 * F1) + (p2 * F2) + (p3 * F3)

        flops_list.append(avg_flops / 1e9)
        recalls.append(rec)
        f1s.append(f1)
        accs.append(acc)

    return flops_list, recalls, f1s, accs

def main():
    print("===============================================================")
    print("PCam Clinical Frontier: Tumor Recall & F1-Score vs. FLOPs")
    print("===============================================================")
    os.makedirs("./results", exist_ok=True)

    _, _, test_loader = get_pcam_loaders(pipeline_type="augmented", batch_size=64, num_workers=2)
    sample_batch = next(iter(test_loader))[0]

    thresholds = np.concatenate((np.arange(0.50, 0.95, 0.05), np.arange(0.95, 1.01, 0.01)))

    # 1. Forward Pass: FP32 (CUDA)
    gpu_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/2] Evaluating FP32 on {gpu_dev}...")
    fp32_model = MultiExitConvNeXt().to(gpu_dev)
    fp32_model.load_state_dict(torch.load(CHECKPOINT_FP32, map_location=gpu_dev, weights_only=True), strict=False)
    l1_f, l2_f, l3_f, targets = extract_logits(fp32_model, test_loader, gpu_dev, "FP32 Test")
    flops_f, rec_f, f1_f, acc_f = sweep_clinical_thresholds(l1_f, l2_f, l3_f, targets, thresholds)

    # 2. Forward Pass: INT8 (CPU)
    cpu_dev = torch.device("cpu")
    print("\n[2/2] Evaluating INT8 on CPU...")
    int8_model = load_int8_model(sample_batch.to(cpu_dev), cpu_dev)
    l1_i, l2_i, l3_i, _ = extract_logits(int8_model, test_loader, cpu_dev, "INT8 Test")
    flops_i, rec_i, f1_i, acc_i = sweep_clinical_thresholds(l1_i, l2_i, l3_i, targets, thresholds)

    # Static Baselines
    _, rec_e1, f1_e1 = compute_metrics(l1_f.argmax(dim=1), targets)
    _, rec_e2, f1_e2 = compute_metrics(l2_f.argmax(dim=1), targets)
    _, rec_e3, f1_e3 = compute_metrics(l3_f.argmax(dim=1), targets)
    _, rec_int8_static, f1_int8_static = compute_metrics(l3_i.argmax(dim=1), targets)

    # Print Summary Table
    print("\n" + "=" * 80)
    print(f"{'Tau':<5} | {'INT8 GFLOPs':<11} | {'Accuracy':<10} | {'Tumor Recall':<13} | {'F1-Score':<10}")
    print("-" * 80)
    for tau, fl, ac, rc, f1 in zip(thresholds, flops_i, acc_i, rec_i, f1_i):
        print(f"{tau:<5.2f} | {fl:<11.4f} | {ac:<9.2f}% | {rc:<12.2f}% | {f1:<9.2f}%")
    print("=" * 80)

    # -------------------------------------------------------------
    # Dual-Panel Figure: Recall & F1 vs. FLOPs
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Panel 1: Tumor Recall (Clinical Sensitivity) vs. FLOPs
    ax1.plot(flops_f, rec_f, 'b-', marker='o', linewidth=2.0, label='Adaptive FP32')
    ax1.plot(flops_i, rec_i, 'm--', marker='s', linewidth=2.0, label='Adaptive INT8')
    ax1.scatter([F1/1e9], [rec_e1], color='orange', marker='^', s=130, zorder=5, label=f'Static Exit 1 ({rec_e1:.1f}%)')
    ax1.scatter([F2/1e9], [rec_e2], color='green', marker='D', s=110, zorder=5, label=f'Static Exit 2 ({rec_e2:.1f}%)')
    ax1.scatter([F3/1e9], [rec_e3], color='red', marker='*', s=200, zorder=5, label=f'Static Exit 3 FP32 ({rec_e3:.1f}%)')
    ax1.scatter([F3/1e9], [rec_int8_static], color='black', marker='X', s=150, zorder=5, label=f'Static Exit 3 INT8 ({rec_int8_static:.1f}%)')
    ax1.set_title('Clinical Safety: Tumor Recall vs. Compute', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Average Computation per Image (GFLOPs)', fontsize=11)
    ax1.set_ylabel('Tumor Recall / Sensitivity (%)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=9.5)

    # Panel 2: F1-Score vs. FLOPs
    ax2.plot(flops_f, f1_f, 'b-', marker='o', linewidth=2.0, label='Adaptive FP32')
    ax2.plot(flops_i, f1_i, 'm--', marker='s', linewidth=2.0, label='Adaptive INT8')
    ax2.scatter([F1/1e9], [f1_e1], color='orange', marker='^', s=130, zorder=5, label=f'Static Exit 1 ({f1_e1:.1f}%)')
    ax2.scatter([F2/1e9], [f1_e2], color='green', marker='D', s=110, zorder=5, label=f'Static Exit 2 ({f1_e2:.1f}%)')
    ax2.scatter([F3/1e9], [f1_e3], color='red', marker='*', s=200, zorder=5, label=f'Static Exit 3 FP32 ({f1_e3:.1f}%)')
    ax2.scatter([F3/1e9], [f1_int8_static], color='black', marker='X', s=150, zorder=5, label=f'Static Exit 3 INT8 ({f1_int8_static:.1f}%)')
    ax2.set_title('Harmonic Balance: F1-Score vs. Compute', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Average Computation per Image (GFLOPs)', fontsize=11)
    ax2.set_ylabel('F1-Score (%)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=9.5)

    plt.tight_layout()
    out_fig = "./results/pcam_clinical_pareto.png"
    plt.savefig(out_fig)
    print(f"\nSaved clinical Pareto plots to {out_fig}")

if __name__ == "__main__":
    main()