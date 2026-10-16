import os
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from tqdm import tqdm
from dataset import get_pcam_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)
    # Replace final linear layer for binary classification (PCam: tumor vs non-tumor)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    return model.to(DEVICE)

def train_static_baseline(epochs=10, target_acc=0.90):
    print(f"\n=======================================================")
    print(f"Starting Block 2: Static Baseline Fine-Tuning")
    print(f"Pipeline: AUGMENTED | Data subset: 100% | Device: {DEVICE}")
    print(f"=======================================================")
    
    # Load 100% of the data using the chosen augmented pipeline
    train_loader, val_loader, _ = get_pcam_loaders(
        data_root="./data",
        pipeline_type="augmented",
        batch_size=64,
        num_workers=2,
        subset_fraction=1.0  # Full dataset
    )
    
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Cosine Annealing scheduler to help stabilize late-stage convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "static_fp32_baseline.pt")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training Loop
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        train_acc = correct / total
        train_loss = running_loss / total
        scheduler.step()

        # Validation Loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                preds = model(images).argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Train Acc = {train_acc*100:.2f}% | Val Acc = {val_acc*100:.2f}%")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"*** New best validation accuracy! Saving checkpoint to {checkpoint_path} ***")
            torch.save(model.state_dict(), checkpoint_path)
            
            # Optional early stopping if target is comfortably reached
            if best_val_acc >= target_acc:
                print(f"\nTarget accuracy of {target_acc*100}% reached. Training complete.")
                break

    print(f"\nBaseline fine-tuning finished. Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"Static FP32 checkpoint permanently saved at: {checkpoint_path}")

if __name__ == "__main__":
    # Start training for up to 10 epochs, stopping early if it hits 90% validation accuracy
    train_static_baseline(epochs=10, target_acc=0.90)