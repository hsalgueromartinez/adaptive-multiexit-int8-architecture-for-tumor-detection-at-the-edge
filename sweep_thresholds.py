import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from dataset import get_pcam_loaders
from multi_exit_model import MultiExitConvNeXt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./checkpoints/multi_exit_pcam_best.pt"

def sweep_thresholds():
    print("=======================================================")
    print("Block 6: Threshold Sweeping & Operating Points")
    print("=======================================================")

    # Use strictly the validation split for tuning
    _, val_loader, _ = get_pcam_loaders(pipeline_type="augmented", batch_size=64, num_workers=2)

    model = MultiExitConvNeXt().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True), strict=False)
    model.eval()

    all_logits1, all_logits2, all_logits3, all_targets = [], [], [], []

    print("\n--- 1. Gathering Validation Logits ---")
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Forward Pass"):
            images = images.to(DEVICE)
            # Use Operational Mode 1 to get all exits simultaneously
            l1, l2, l3 = model(images)
            all_logits1.append(l1.cpu())
            all_logits2.append(l2.cpu())
            all_logits3.append(l3.cpu())
            all_targets.append(targets)

    logits1 = torch.cat(all_logits1)
    logits2 = torch.cat(all_logits2)
    logits3 = torch.cat(all_logits3)
    targets = torch.cat(all_targets)
    total_samples = len(targets)

    # Step 6.1: Confidence Policy Definition (Maximum Softmax Probability)
    conf1, preds1 = torch.max(F.softmax(logits1, dim=1), dim=1)
    conf2, preds2 = torch.max(F.softmax(logits2, dim=1), dim=1)
    preds3 = logits3.argmax(dim=1)

    print("\n--- 2. Sweeping Thresholds [0.50 to 0.99] ---")
    print(f"{'Tau':<6} | {'Accuracy':<10} | {'Exit 1 %':<10} | {'Exit 2 %':<10} | {'Exit 3 %':<10}")
    print("-" * 60)

    # Sweep grid: 0.50 to 0.94 in 0.05 steps, then 0.95 to 0.99 in 0.01 steps
    thresholds = np.concatenate((np.arange(0.50, 0.95, 0.05), np.arange(0.95, 1.00, 0.01)))
    
    for tau in thresholds:
        # Simulate dynamic routing
        exit1_mask = conf1 >= tau
        exit2_mask = (~exit1_mask) & (conf2 >= tau)
        exit3_mask = ~(exit1_mask | exit2_mask)
        
        # Calculate accuracy for this routing combination
        correct = 0
        correct += (preds1[exit1_mask] == targets[exit1_mask]).sum().item()
        correct += (preds2[exit2_mask] == targets[exit2_mask]).sum().item()
        correct += (preds3[exit3_mask] == targets[exit3_mask]).sum().item()
        
        acc = correct / total_samples
        pct1 = (exit1_mask.sum().item() / total_samples) * 100
        pct2 = (exit2_mask.sum().item() / total_samples) * 100
        pct3 = (exit3_mask.sum().item() / total_samples) * 100
        
        print(f"{tau:<6.2f} | {acc*100:<9.2f}% | {pct1:<9.1f}% | {pct2:<9.1f}% | {pct3:<9.1f}%")

if __name__ == "__main__":
    sweep_thresholds()