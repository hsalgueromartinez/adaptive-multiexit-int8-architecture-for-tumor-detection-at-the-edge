import os
import time
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from multi_exit_model import MultiExitConvNeXt

CHECKPOINT_FP32 = "./checkpoints/multi_exit_pcam_best.pt"
CHECKPOINT_INT8 = "./checkpoints/multi_exit_int8.pt"


FLOPS_E1 = 0.2821
FLOPS_E2 = 0.6818
FLOPS_E3 = 0.8210

def load_int8_model(sample_input):
    torch.backends.quantized.engine = "fbgemm"
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    shell = MultiExitConvNeXt().to(torch.device("cpu"))
    shell.eval()
    prepared = prepare_fx(shell, qconfig_mapping, (sample_input,))
    int8_model = convert_fx(prepared)
    int8_model.load_state_dict(torch.load(CHECKPOINT_INT8, map_location="cpu", weights_only=True))
    int8_model.eval()
    return int8_model

def measure_latency_ms(forward_fn, sample_input, warmup=20, runs=100):
    with torch.no_grad():
        for _ in range(warmup):
            _ = forward_fn(sample_input)
        
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = forward_fn(sample_input)
        t1 = time.perf_counter()

    return ((t1 - t0) / runs) * 1000.0

def main():
    print("===============================================================")
    print("Edge Benchmarking: Single-Image Latency & Memory Footprint")
    print("===============================================================")

    # Memory on Disk
    fp32_mb = os.path.getsize(CHECKPOINT_FP32) / (1024 * 1024)
    int8_mb = os.path.getsize(CHECKPOINT_INT8) / (1024 * 1024)
    print(f"\n1. Storage / Checkpoint Footprint on Disk:")
    print(f"   • Multi-Exit FP32 Model : {fp32_mb:.2f} MB")
    print(f"   • Multi-Exit INT8 Model : {int8_mb:.2f} MB  ({fp32_mb / int8_mb:.1f}x reduction)")

    dummy_patch = torch.randn(1, 3, 96, 96)

    # Stage-by-Stage Latency Benchmarking (FP32 on CPU)
    print(f"\n2. Stage-by-Stage Latency on CPU (FP32 PyTorch Runtime)...")
    fp32_model = MultiExitConvNeXt().to(torch.device("cpu"))
    fp32_model.load_state_dict(torch.load(CHECKPOINT_FP32, map_location="cpu", weights_only=True), strict=False)
    fp32_model.eval()

    lat_fp32_e1 = measure_latency_ms(lambda x: fp32_model.exit1(fp32_model.stage1_2(x)), dummy_patch)
    lat_fp32_e2 = measure_latency_ms(lambda x: fp32_model.exit2(fp32_model.stage3(fp32_model.stage1_2(x))), dummy_patch)
    lat_fp32_full = measure_latency_ms(lambda x: fp32_model(x), dummy_patch)

    print(f"   • Exit 1 (Stage 2)   : {lat_fp32_e1:.2f} ms")
    print(f"   • Exit 2 (Stage 3)   : {lat_fp32_e2:.2f} ms")
    print(f"   • Exit 3 (Full Stage): {lat_fp32_full:.2f} ms")

    # Full-Model Quantized Latency (INT8 fbgemm on CPU)
    print(f"\n3. Quantized INT8 Latency on CPU (FBGEMM Engine)...")
    int8_model = load_int8_model(dummy_patch)
    lat_int8_full = measure_latency_ms(lambda x: int8_model(x), dummy_patch)
    print(f"   • INT8 Full Forward Pass: {lat_int8_full:.2f} ms  ({lat_fp32_full / lat_int8_full:.1f}x speedup vs FP32)")

    scale = lat_int8_full / lat_fp32_full
    lat_int8_e1 = lat_fp32_e1 * scale
    lat_int8_e2 = lat_fp32_e2 * scale

    # Dynamic Average Latency per Operating Point
    print("\n4. Dynamic Latencies across Evaluated Operating Points (INT8 Edge):")
    print(f"{'Operating Point':<24} | {'Threshold':<9} | {'Avg Latency (ms)':<16} | {'Speedup vs Baseline':<20}")
    print("-" * 78)

    operating_points = [
        ("Emergency Edge Mode",   "tau = 0.60", 0.978, 0.020, 0.002),
        ("Peak Balanced (Knee)", "tau = 0.80", 0.924, 0.050, 0.027),
        ("Conservative Mode",    "tau = 0.98", 0.682, 0.151, 0.167),
        ("Static Full Baseline", "tau = 1.00", 0.000, 0.000, 1.000),
    ]

    for name, tau_str, r1, r2, r3 in operating_points:
        avg_lat = (r1 * lat_int8_e1) + (r2 * lat_int8_e2) + (r3 * lat_int8_full)
        speedup = lat_int8_full / avg_lat
        print(f"{name:<24} | {tau_str:<9} | {avg_lat:<16.2f} | {speedup:<.2f}x faster")
    print("=" * 78)

if __name__ == "__main__":
    main()