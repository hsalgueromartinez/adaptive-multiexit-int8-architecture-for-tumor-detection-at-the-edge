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


F1 = 0.2821 * 1e9  # Exit 1
F2 = 0.6818 * 1e9  # Exit 2 
F3 = 0.8210 * 1e9  # Exit 3

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

def evaluate_threshold_sweep(logits1, logits2, logits3, targets, thresholds):
    total = len(targets)
    conf1, preds1 = torch.max(F.softmax(logits1, dim=1), dim=1)
    conf2, preds2 = torch.max(F.softmax(logits2, dim=1), dim=1)
    preds3 = logits3.argmax(dim=1)

    flops_list, acc_list = [], []

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

        correct = (preds1[e1] == targets[e1]).sum().item() + \
                  (preds2[e2] == targets[e2]).sum().item() + \
                  (preds3[e3] == targets[e3]).sum().item()

        acc = (correct / total) * 100.0
        p1 = e1.sum().item() / total
        p2 = e2.sum().item() / total
        p3 = e3.sum().item() / total

        avg_flops = (p1 * F1) + (p2 * F2) + (p3 * F3)
        flops_list.append(avg_flops / 1e9)
        acc_list.append(acc)

    return flops_list, acc_list

def main():
    print("=======================================================")
    print("Block 7: Benchmarking, Pareto Analysis & Reflection")
    print("=======================================================")

    _, _, test_loader = get_pcam_loaders(pipeline_type="augmented", batch_size=64, num_workers=2)
    sample_batch = next(iter(test_loader))[0]

    thresholds = np.concatenate((np.arange(0.50, 0.95, 0.05), np.arange(0.95, 1.01, 0.01)))

    # Forward Pass: FP32 Model (CUDA if available)
    gpu_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 1. Evaluating FP32 Model on Test Set ({gpu_dev}) ---")
    fp32_model = MultiExitConvNeXt().to(gpu_dev)
    fp32_model.load_state_dict(torch.load(CHECKPOINT_FP32, map_location=gpu_dev, weights_only=True), strict=False)
    l1_f, l2_f, l3_f, targets = extract_logits(fp32_model, test_loader, gpu_dev, "FP32 Test")
    fp32_flops, fp32_accs = evaluate_threshold_sweep(l1_f, l2_f, l3_f, targets, thresholds)

    # Forward Pass: INT8 Model (CPU)
    cpu_dev = torch.device("cpu")
    print("\n--- 2. Evaluating INT8 Model on Test Set (CPU FBGEMM) ---")
    int8_model = load_int8_model(sample_batch.to(cpu_dev), cpu_dev)
    l1_i, l2_i, l3_i, _ = extract_logits(int8_model, test_loader, cpu_dev, "INT8 Test")
    int8_flops, int8_accs = evaluate_threshold_sweep(l1_i, l2_i, l3_i, targets, thresholds)

    # Static Baselines
    acc_static_e1 = (l1_f.argmax(dim=1) == targets).sum().item() / len(targets) * 100.0
    acc_static_e2 = (l2_f.argmax(dim=1) == targets).sum().item() / len(targets) * 100.0
    acc_static_e3 = (l3_f.argmax(dim=1) == targets).sum().item() / len(targets) * 100.0
    acc_int8_static = (l3_i.argmax(dim=1) == targets).sum().item() / len(targets) * 100.0


    with open("pareto_summary.txt", "w") as f:
        f.write("--- Block 7: Final Test Set Evaluation ---\n")
        f.write(f"Static Exit 1: {F1/1e9:.4f} GFLOPs | Acc: {acc_static_e1:.2f}%\n")
        f.write(f"Static Exit 2: {F2/1e9:.4f} GFLOPs | Acc: {acc_static_e2:.2f}%\n")
        f.write(f"Static Exit 3: {F3/1e9:.4f} GFLOPs | Acc: {acc_static_e3:.2f}%\n")
        f.write(f"Static INT8:   {F3/1e9:.4f} GFLOPs | Acc: {acc_int8_static:.2f}%\n\n")
        f.write(f"{'Tau':<5} | {'FP32 Acc':<9} | {'INT8 Acc':<9} | {'FP32 GFLOPs':<11} | {'INT8 GFLOPs':<11}\n")
        f.write("-" * 55 + "\n")
        for tau, fa, ia, ff, ifl in zip(thresholds, fp32_accs, int8_accs, fp32_flops, int8_flops):
            f.write(f"{tau:<5.2f} | {fa:<8.2f}% | {ia:<8.2f}% | {ff:<11.4f} | {ifl:<11.4f}\n")


    plt.figure(figsize=(11, 6.5), dpi=300)


    plt.plot(fp32_flops, fp32_accs, 'b-', marker='o', linewidth=2.2, label='Adaptive Multi-Exit (FP32)')
    plt.plot(int8_flops, int8_accs, 'm--', marker='s', linewidth=2.2, label='Adaptive Multi-Exit (INT8 Quantized)')


    plt.scatter([F3/1e9], [acc_static_e3], color='red', marker='*', s=220, zorder=5, label='Static Full-Depth FP32 Baseline')
    plt.scatter([F3/1e9], [acc_int8_static], color='black', marker='X', s=160, zorder=5, label='Static Full-Depth INT8 Baseline')


    plt.scatter([F1/1e9], [acc_static_e1], color='orange', marker='^', s=130, zorder=5, label='Static Exit 1 Only (Stage 2)')
    plt.scatter([F2/1e9], [acc_static_e2], color='green', marker='D', s=110, zorder=5, label='Static Exit 2 Only (Stage 3)')

    plt.annotate('Emergency Edge Mode\n(High E1 Exit, ~0.29 GFLOPs)', 
                 xy=(int8_flops[2], int8_accs[2]), 
                 xytext=(int8_flops[2] + 0.04, int8_accs[2] - 0.35), 
                 arrowprops=dict(arrowstyle="->", color="#4b0082", lw=1.5),
                 fontsize=9.5, color="#4b0082")

    plt.annotate('Peak Balanced Operating Point\n(τ=0.80, 60.4% Compute Saved)', 
                 xy=(int8_flops[6], int8_accs[6]), 
                 xytext=(int8_flops[6] + 0.05, int8_accs[6] - 0.25), 
                 arrowprops=dict(arrowstyle="->", color="#800080", lw=1.8),
                 fontweight='bold', fontsize=10, color="#800080")

    plt.ylim(87.2, 88.4)
    plt.xlim(0.26, 0.86)

    plt.title('Pareto Frontier: Accuracy vs. Computation Cost (PatchCamelyon Test Set)', fontsize=13, fontweight='bold')
    plt.xlabel('Average Computation per Image (GFLOPs)', fontsize=11)
    plt.ylabel('Test Accuracy (%)', fontsize=11)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    out_plot = "./pareto_frontier.png"
    plt.savefig(out_plot)
    print(f"\nSUCCESS! Pareto Frontier saved to {out_plot} and summary to pareto_summary.txt")

if __name__ == "__main__":
    main()