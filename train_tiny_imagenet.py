import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

from dataset_tiny_imagenet import get_tiny_imagenet_loaders
from multi_exit_model import MultiExitConvNeXt
from train_multi_exit import MultiExitDistillationLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def profile_tiny_imagenet_flops():
    print("\n--- Profiling FLOPs for Tiny-ImageNet (64x64, 200 classes) ---")
    model = MultiExitConvNeXt(num_classes=200).to(DEVICE)
    model.eval()
    dummy = torch.randn(1, 3, 64, 64, device=DEVICE)

    class SubExit(nn.Module):
        def __init__(self, feat, head):
            super().__init__()
            self.f = feat
            self.h = head
        def forward(self, x): return self.h(self.f(x))

    flops1 = FlopCountAnalysis(SubExit(model.features[0:4], model.exit1), dummy).total()
    flops2 = FlopCountAnalysis(SubExit(model.features[0:6], model.exit2), dummy).total()
    flops3 = FlopCountAnalysis(SubExit(model.features[0:8], model.exit3), dummy).total()

    print(f"Exit 1 (Stage 2) : {flops1:,} FLOPs ({flops1 / 1e9:.3f} GFLOPs)")
    print(f"Exit 2 (Stage 3) : {flops2:,} FLOPs ({flops2 / 1e9:.3f} GFLOPs)")
    print(f"Exit 3 (Stage 4) : {flops3:,} FLOPs ({flops3 / 1e9:.3f} GFLOPs)")
    return flops1, flops2, flops3

def train_tiny_imagenet(epochs=5, batch_size=64, lr=1e-4):
    os.makedirs("./checkpoints", exist_ok=True)
    train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=batch_size)

    # Instantiate model with 200 class heads
    model = MultiExitConvNeXt(num_classes=200).to(DEVICE)
    criterion = MultiExitDistillationLoss(alphas=(1.0, 1.0, 1.0), betas=(0.5, 0.5), temperature=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Tiny-ImageNet]"):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            logits1, logits2, logits3 = model(images)
            loss = criterion(logits1, logits2, logits3, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()

        # Validation
        model.eval()
        corr1, corr2, corr3, total = 0, 0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                l1, l2, l3 = model(images)
                
                corr1 += (l1.argmax(1) == targets).sum().item()
                corr2 += (l2.argmax(1) == targets).sum().item()
                corr3 += (l3.argmax(1) == targets).sum().item()
                total += targets.size(0)

        acc1, acc2, acc3 = corr1 / total, corr2 / total, corr3 / total
        print(f"Epoch {epoch+1} Val Accuracies -> Exit 1: {acc1*100:.2f}% | Exit 2: {acc2*100:.2f}% | Exit 3: {acc3*100:.2f}%")

        if acc3 > best_val_acc:
            best_val_acc = acc3
            torch.save(model.state_dict(), "./checkpoints/multi_exit_tiny_imagenet_best.pt")
            print("--> Saved best Tiny-ImageNet checkpoint!")

if __name__ == "__main__":
    profile_tiny_imagenet_flops()
    train_tiny_imagenet(epochs=10)