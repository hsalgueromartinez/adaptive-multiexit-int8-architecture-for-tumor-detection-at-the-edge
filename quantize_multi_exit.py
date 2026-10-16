import os
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from tqdm import tqdm

from dataset import get_pcam_loaders
from multi_exit_model import MultiExitConvNeXt

# int8 quantized kernels in PyTorch are CPU-only, so we force CPU here.
DEVICE = torch.device("cpu")
CHECKPOINT_PATH = "./checkpoints/multi_exit_convnext_best.pt"

def evaluate_model(model, data_loader, num_batches=None, desc="Evaluating"):
    model.eval()
    corr1, corr2, corr3, total = 0, 0, 0, 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(data_loader, desc=desc)):
            if num_batches and i >= num_batches:
                break
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            # Operational Mode 1: Returns all 3 logits
            l1, l2, l3 = model(images)
            
            corr1 += (l1.argmax(1) == targets).sum().item()
            corr2 += (l2.argmax(1) == targets).sum().item()
            corr3 += (l3.argmax(1) == targets).sum().item()
            total += targets.size(0)
            
    return corr1/total, corr2/total, corr3/total

def main():
    print("=======================================================")
    print("Block 5: Post-Training Static Quantization (FX Mode)")
    print("=======================================================")
    
    # Load loaders (Val for calibration, Test for parity check)
    _, val_loader, test_loader = get_pcam_loaders(pipeline_type="augmented", batch_size=64, num_workers=2)

    # 1. Load FP32 Multi-Exit Model
    print("\n--- Loading FP32 Baseline ---")
    fp32_model = MultiExitConvNeXt().to(DEVICE)
    fp32_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True), strict=False)
    fp32_model.eval()

    # Get Ground-Truth FP32 Accuracy
    fp32_acc1, fp32_acc2, fp32_acc3 = evaluate_model(fp32_model, test_loader, desc="FP32 Test Eval")
    print(f"FP32 Accuracies -> Exit 1: {fp32_acc1*100:.2f}% | Exit 2: {fp32_acc2*100:.2f}% | Exit 3: {fp32_acc3*100:.2f}%")

    # 2. Step 5.1: Observer Calibration
    print("\n--- Step 5.1: Observer Calibration ---")
    # Set the backend engine to fbgemm[cite: 18].
    torch.backends.quantized.engine = "fbgemm" 
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    
    # FX mode requires an example input to trace the model graph[cite: 18].
    sample_batch = next(iter(val_loader))[0].to(DEVICE)
    example_inputs = (sample_batch, )
    
    # prepare_fx inserts observers and auto-fuses layers like Conv+BN+ReLU[cite: 18].
    prepared_model = prepare_fx(fp32_model, qconfig_mapping, example_inputs)
    
    # Calibrate by running validation batches through in eval mode[cite: 18].
    evaluate_model(prepared_model, val_loader, num_batches=50, desc="Calibrating (Val Subset)")

    # 3. Step 5.2: INT8 Conversion
    print("\n--- Step 5.2: INT8 Conversion ---")
    # Convert to a fully quantized INT8 model[cite: 18].
    static_quantized = convert_fx(prepared_model)
    
    int8_path = "./checkpoints/multi_exit_int8.pt"
    torch.save(static_quantized.state_dict(), int8_path)
    print(f"INT8 weights saved to {int8_path}")

    # 4. Step 5.3: Quantization Parity Check
    print("\n--- Step 5.3: Quantization Parity Check ---")
    int8_acc1, int8_acc2, int8_acc3 = evaluate_model(static_quantized, test_loader, desc="INT8 Test Eval")
    print(f"INT8 Accuracies -> Exit 1: {int8_acc1*100:.2f}% | Exit 2: {int8_acc2*100:.2f}% | Exit 3: {int8_acc3*100:.2f}%")
    
    print("\n--- Final Degradation Analysis (< 1% Expected) ---")
    print(f"Exit 1 Degradation: {fp32_acc1*100 - int8_acc1*100:.2f}%")
    print(f"Exit 2 Degradation: {fp32_acc2*100 - int8_acc2*100:.2f}%")
    print(f"Exit 3 Degradation: {fp32_acc3*100 - int8_acc3*100:.2f}%")

if __name__ == "__main__":
    main()