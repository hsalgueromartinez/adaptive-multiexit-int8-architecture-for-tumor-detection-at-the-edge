import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

from dataset import get_pcam_loaders
from multi_exit_model import MultiExitConvNeXt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Step 4.1: Joint Distillation Loss ---
class MultiExitDistillationLoss(nn.Module):
    def __init__(self, alphas=(1.0, 1.0, 1.0), betas=(0.5, 0.5), temperature=3.0):
        super().__init__()
        self.alphas = alphas
        self.betas = betas
        self.T = temperature
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits1, logits2, logits3, targets):
        # 1. Supervised Cross-Entropy on all heads
        l_ce1 = self.ce(logits1, targets)
        l_ce2 = self.ce(logits2, targets)
        l_ce3 = self.ce(logits3, targets)
        supervised_loss = self.alphas[0] * l_ce1 + self.alphas[1] * l_ce2 + self.alphas[2] * l_ce3

        # 2. Knowledge Distillation: Stage 4 teacher supervises Exits 1 & 2
        teacher_soft = F.softmax(logits3.detach() / self.T, dim=1)
        
        kd1 = self.kl(F.log_softmax(logits1 / self.T, dim=1), teacher_soft) * (self.T ** 2)
        kd2 = self.kl(F.log_softmax(logits2 / self.T, dim=1), teacher_soft) * (self.T ** 2)
        distillation_loss = self.betas[0] * kd1 + self.betas[1] * kd2

        return supervised_loss + distillation_loss

# --- Step 4.3: Standalone Exit Profiling ---
def profile_individual_exits():
    print("\n--- Step 4.3: Profiling Individual Standalone Exits (FLOPs) ---")
    model = MultiExitConvNeXt().to(DEVICE)
    model.eval()
    dummy = torch.randn(1, 3, 96, 96, device=DEVICE)

    # Standalone Exit 1
    class Exit1Model(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.f = m.features[0:4]
            self.h = m.exit1
        def forward(self, x): return self.h(self.f(x))

    # Standalone Exit 2
    class Exit2Model(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.f = m.features[0:6]
            self.h = m.exit2
        def forward(self, x): return self.h(self.f(x))

    # Standalone Exit 3
    class Exit3Model(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.f = m.features[0:8]
            self.h = m.exit3
        def forward(self, x): return self.h(self.f(x))

    flops1 = FlopCountAnalysis(Exit1Model(model), dummy).total()
    flops2 = FlopCountAnalysis(Exit2Model(model), dummy).total()
    flops3 = FlopCountAnalysis(Exit3Model(model), dummy).total()

    print(f"Exit 1 Alone (Stage 2) : {flops1:,} FLOPs ({flops1 / 1e9:.3f} GFLOPs)")
    print(f"Exit 2 Alone (Stage 3) : {flops2:,} FLOPs ({flops2 / 1e9:.3f} GFLOPs)")
    print(f"Exit 3 Alone (Stage 4) : {flops3:,} FLOPs ({flops3 / 1e9:.3f} GFLOPs)")
    return flops1, flops2, flops3

# --- Step 4.2: Multi-Exit Fine-Tuning Execution ---
def train_multi_exit(epochs=10, batch_size=64, lr=1e-4):
    os.makedirs("./checkpoints", exist_ok=True)
    train_loader, val_loader, test_loader = get_pcam_loaders(pipeline_type="augmented", batch_size=batch_size)

    model = MultiExitConvNeXt().to(DEVICE)
    criterion = MultiExitDistillationLoss(alphas=(1.0, 1.0, 1.0), betas=(0.5, 0.5), temperature=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            logits1, logits2, logits3 = model(images)
            loss = criterion(logits1, logits2, logits3, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()

        # Evaluate each head independently on the validation split
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
            torch.save(model.state_dict(), "./checkpoints/multi_exit_convnext_best.pt")
            print("--> Saved best checkpoint to ./checkpoints/multi_exit_convnext_best.pt")

if __name__ == "__main__":
    profile_individual_exits()
    # Execute fine-tuning
    train_multi_exit(epochs=10)