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

def train_eval_ablation(pipeline_type, epochs=3, subset_fraction=0.1):
    print(f"\n=======================================================")
    print(f"Starting Ablation: Pipeline [{pipeline_type.upper()}] (Data subset: {subset_fraction*100}%)")
    print(f"=======================================================")
    
    train_loader, val_loader, _ = get_pcam_loaders(
        data_root="./data",
        pipeline_type=pipeline_type,
        batch_size=64,
        num_workers=2,
        subset_fraction=subset_fraction
    )
    
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{pipeline_type}]"):
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

        # Evaluate on validation split
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                preds = model(images).argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Train Acc = {train_acc*100:.2f}% | Val Acc = {val_acc*100:.2f}%")

    return val_acc

if __name__ == "__main__":
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    print(f"Running on device: {DEVICE}")
    
    # 1. Run Raw Baseline
    #raw_val_acc = train_eval_ablation("raw", epochs=3, subset_fraction=0.1)
    
    # 2. Run Histology-Informed Pipeline
    aug_val_acc = train_eval_ablation("augmented", epochs=3, subset_fraction=0.1)
    
    print("\n---------------- FINAL DECISION ----------------")
    #print(f"Pipeline A (Raw) Validation Accuracy        : {raw_val_acc * 100:.2f}%")
    print(f"Pipeline B (Augmented) Validation Accuracy  : {aug_val_acc * 100:.2f}%")
    # if aug_val_acc >= raw_val_acc:
    #     print("Decision: ADOPT Pipeline B (Augmented) for Block 2 fine-tuning.")
    # else:
    #     print("Decision: ADOPT Pipeline A (Raw) or adjust jitter strengths.")