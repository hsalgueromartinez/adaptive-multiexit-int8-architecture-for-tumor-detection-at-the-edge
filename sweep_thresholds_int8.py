import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from tqdm import tqdm

from dataset import get_pcam_loaders
from multi_exit_model import MultiExitConvNeXt

DEVICE = torch.device("cpu")
INT8_CHECKPOINT = "./checkpoints/multi_exit_int8.pt"

FLOPS_E1 = 0.2821
FLOPS_E2 = 0.6818
FLOPS_E3 = 0.8210

def load_quantized_model_from_statedict(sample_batch):
    print("\n--- 1. Reconstructing FX Graph Shell & Loading INT8 State Dict ---")
    torch.backends.quantized.engine = "fbgemm"
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")

    # Instantiate unquantized architecture shell on CPU
    shell_model = MultiExitConvNeXt().to(DEVICE)
    shell_model.eval()

    prepared = prepare_fx(shell_model, qconfig_mapping, (sample_batch,))
    int8_model = convert_fx(prepared)

    state_dict = torch.load(INT8_CHECKPOINT, map_location=DEVICE, weights_only=True)
    int8_model.load_state_dict(state_dict)
    int8_model.eval()
    print("--> INT8 Checkpoint loaded successfully!")
    return int8_model

def sweep_int8():
    print("=======================================================")
    print("Block 6: INT8 Quantized Threshold Sweep & Calibration")
    print("=======================================================")

    # Use strictly the validation split
    _, val_loader, _ = get_pcam_loaders(pipeline_type="augmented", batch_size=64, num_workers=2)


    sample_batch = next(iter(val_loader))[0].to(DEVICE)
    int8_model = load_quantized_model_from_statedict(sample_batch)

   
    all_l1, all_l2, all_l3, all_targets = [], [], [], []
    print("\n--- 2. Gathering INT8 Validation Logits (CPU FBGEMM) ---")
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="INT8 Forward Pass"):
            images = images.to(DEVICE)
            l1, l2, l3 = int8_model(images)
            all_l1.append(l1)
            all_l2.append(l2)
            all_l3.append(l3)
            all_targets.append(targets)

    logits1 = torch.cat(all_l1)
    logits2 = torch.cat(all_l2)
    logits3 = torch.cat(all_l3)
    targets = torch.cat(all_targets)
    total_samples = len(targets)

    conf1, preds1 = torch.max(F.softmax(logits1, dim=1), dim=1)
    conf2, preds2 = torch.max(F.softmax(logits2, dim=1), dim=1)
    preds3 = logits3.argmax(dim=1)

    # Sweep Threshold Grid
    print("\n--- 3. Sweeping INT8 Thresholds ---")
    print(f"{'Tau':<6} | {'Accuracy':<10} | {'Tumor Recall':<13} | {'Avg GFLOPs':<11} | {'Exit 1 %':<10} | {'Exit 2 %':<10} | {'Exit 3 %':<10}")
    print("-" * 84)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]

    for tau in thresholds:
        if tau >= 1.00:
            exit1_mask = torch.zeros(total_samples, dtype=torch.bool)
            exit2_mask = torch.zeros(total_samples, dtype=torch.bool)
            exit3_mask = torch.ones(total_samples, dtype=torch.bool)
        elif tau <= 0.50:
            exit1_mask = torch.ones(total_samples, dtype=torch.bool)
            exit2_mask = torch.zeros(total_samples, dtype=torch.bool)
            exit3_mask = torch.zeros(total_samples, dtype=torch.bool)
        else:
            exit1_mask = conf1 >= tau
            exit2_mask = (~exit1_mask) & (conf2 >= tau)
            exit3_mask = ~(exit1_mask | exit2_mask)

        # Accuracy
        correct = 0
        correct += (preds1[exit1_mask] == targets[exit1_mask]).sum().item()
        correct += (preds2[exit2_mask] == targets[exit2_mask]).sum().item()
        correct += (preds3[exit3_mask] == targets[exit3_mask]).sum().item()
        acc = (correct / total_samples) * 100.0

        # Tumor Recall (Sensitivity)
        preds_all = torch.empty_like(targets)
        preds_all[exit1_mask] = preds1[exit1_mask]
        preds_all[exit2_mask] = preds2[exit2_mask]
        preds_all[exit3_mask] = preds3[exit3_mask]
        tumor_mask = (targets == 1)
        recall = ((preds_all[tumor_mask] == 1).sum().item() / tumor_mask.sum().item()) * 100.0

        # Average FLOPs
        pct1 = exit1_mask.sum().item() / total_samples
        pct2 = exit2_mask.sum().item() / total_samples
        pct3 = exit3_mask.sum().item() / total_samples
        avg_flops = (pct1 * FLOPS_E1) + (pct2 * FLOPS_E2) + (pct3 * FLOPS_E3)

        print(f"{tau:<6.2f} | {acc:<9.2f}% | {recall:<12.2f}% | {avg_flops:<11.4f} | {pct1*100:<9.1f}% | {pct2*100:<9.1f}% | {pct3*100:<9.1f}%")

if __name__ == "__main__":
    sweep_int8()