import torch
import torchvision.models as models
from fvcore.nn import FlopCountAnalysis

def verify_convnext_flops():
    model = models.convnext_tiny(weights="DEFAULT")
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
    model.eval()

    # PCam input dimensions: 1 image, 3 channels, 96x96 resolution
    dummy_input = torch.randn(1, 3, 96, 96)
    
    flop_counter = FlopCountAnalysis(model, dummy_input)
    total_flops = flop_counter.total()
    
    print(f"--- ConvNeXt-Tiny Static Baseline Profile ---")
    print(f"Input Shape : {tuple(dummy_input.shape)}")
    print(f"Total MACs / FLOPs : {total_flops:,} ({total_flops / 1e9:.3f} GFLOPs)")
    print(f"Methodology validated. Keep fvcore constant for all future exits.")

if __name__ == "__main__":
    verify_convnext_flops()