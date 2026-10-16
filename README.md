# Adaptive Multi-Exit Deep Neural Networks for Edge Inference

Edge deployment of adaptive deep learning models for medical pathology screening and computer vision under strict battery, compute, and latency budgets. This project implements dynamic early-exit branching combined with post-training static INT8 quantization across **PatchCamelyon (PCam)**, **PathMNIST**, and **Tiny-ImageNet-200**.

---

## Pre-trained Checkpoints & Quickstart

To evaluate the pre-trained baselines, multi-exit networks, and INT8 quantized models without retraining from scratch, download the checkpoint archive from Google Drive:

```bash
# 1. Install gdown
pip install gdown

# 2. Download pre-trained weights archive from Google Drive
gdown 1KdYwuz4KSEdVkrPgY-zybhE7W89mjH6I -O checkpoints.zip

# 3. Extract checkpoints into project root
unzip checkpoints.zip
rm checkpoints.zip
```

---

## Repository Structure

```text
.
├── checkpoints/                    # Saved weights (.pth) for baselines, multi-exit, and INT8 models
├── data/                           # Local storage for PatchCamelyon and PathMNIST datasets
├── data_tiny_imagenet/             # Tiny-ImageNet-200 dataset storage
├── FLOP/                           # FLOP calculation logs and profiling outputs
├── results/                        # Generated figures, Pareto frontiers, and benchmark summaries
├── statement/                      # Project documentation and specifications
│
├── dataset.py                      # Data loaders and transforms for PatchCamelyon & PathMNIST
├── dataset_tiny_imagenet.py        # Automated download and data loading for Tiny-ImageNet-200
├── multi_exit_model.py             # ResNet-18 architecture with early exits at intermediate stages
├── train_static_baseline.py        # Trains standard single-exit baseline models
├── train_multi_exit.py             # Trains multi-exit networks with weighted joint cross-entropy loss
├── train_tiny_imagenet.py          # Multi-exit training on Tiny-ImageNet-200
├── quantize_multi_exit.py          # Post-training static INT8 quantization with calibration
├── sweep_thresholds.py             # Grid search over confidence thresholds (tau) for FP32 models
├── sweep_thresholds_int8.py        # Threshold sweep for quantized INT8 models
├── test_flops.py                   # Computes theoretical FLOPs per stage using torchprofile
├── measure_latency_memory_pcam.py  # Measures real edge latency (CPU) and checkpoint memory footprint
├── pareto_pcam.py                  # Evaluates and plots Accuracy vs. FLOPs Pareto frontier on PCam
├── pareto_recall_f1_pcam.py        # Evaluates clinical safety (Recall, F1-Score) vs. FLOPs on PCam
├── pareto_tiny_imagenet.py         # Evaluates Accuracy vs. FLOPs Pareto frontier on Tiny-ImageNet
└── run_ablation.py                 # Full ablation pipeline running baselines across all datasets
```

---

## Script Descriptions

| Script | Purpose |
| :--- | :--- |
| `dataset.py` | Handles dataset downloads, train/val/test splits, and augmentations for PatchCamelyon and MedMNIST benchmarks. |
| `dataset_tiny_imagenet.py` | Downloads, unpacks, and preprocesses the Tiny-ImageNet-200 dataset ($64\times 64$, 200 classes). |
| `multi_exit_model.py` | Implements the shared-backbone multi-exit model with adaptive exits attached after intermediate residual stages. |
| `train_static_baseline.py` | Trains standard, unbranched backbones to serve as latency, FLOP, and accuracy baselines. |
| `train_multi_exit.py` | Jointly trains all exit heads using a multi-task loss objective ($\mathcal{L}_{\text{total}} = \sum \alpha_i \mathcal{L}_i$). |
| `train_tiny_imagenet.py` | Training script for multi-exit models on the Tiny-ImageNet-200 benchmark. |
| `quantize_multi_exit.py` | Performs post-training static INT8 quantization (fusing Conv+BN+ReLU, calibrating activations, and converting weights). |
| `sweep_thresholds.py` | Sweeps decision thresholds ($\tau \in [0.5, 0.99]$) on FP32 multi-exit models to compute dynamic exit rates. |
| `sweep_thresholds_int8.py` | Sweeps decision thresholds on quantized INT8 models to assess Pareto optimality at low bit-widths. |
| `test_flops.py` | Uses `torchprofile` to benchmark theoretical FLOPs stage-by-stage. |
| `measure_latency_memory_pcam.py` | Measures single-sample edge CPU latency, throughput (FPS), and checkpoint storage footprints. |
| `pareto_pcam.py` | Plots the multi-exit trade-off curves (Accuracy vs. FLOPs) on PatchCamelyon. |
| `pareto_recall_f1_pcam.py` | Computes sector-aware clinical safety metrics (Tumor Recall and F1-Score vs. FLOPs). |
| `pareto_tiny_imagenet.py` | Evaluates representational scaling and trade-off frontiers on Tiny-ImageNet-200. |
| `run_ablation.py` | Runs the full ablation suite across all benchmarks and architectures. |

---

## Step-by-Step Reproduction Guide

### 1. Environment Setup

```bash
git clone [https://github.com/](https://github.com/)<username>/<repo-name>.git
cd <repo-name>

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Pre-trained Weights

```bash
gdown 1KdYwuz4KSEdVkrPgY-zybhE7W89mjH6I -O checkpoints.zip
unzip checkpoints.zip
rm checkpoints.zip
```

### 3. Threshold Sweeps & Profiling

```bash
# Analyze FP32 dynamic early exits
python sweep_thresholds.py

# Analyze INT8 dynamic early exits
python sweep_thresholds_int8.py

# Benchmark stage-by-stage FLOPs
python test_flops.py
```

### 4. Edge Latency & Memory Benchmark

```bash
python measure_latency_memory_pcam.py
```

### 5. Generate Pareto Frontiers & Clinical Recall Figures

```bash
# Accuracy vs. FLOPs frontier (PatchCamelyon)
python pareto_pcam.py

# Clinical safety frontier (Recall and F1 vs. FLOPs)
python pareto_recall_f1_pcam.py

# Natural vision ablation frontier (Tiny-ImageNet)
python pareto_tiny_imagenet.py
```

---

## Training from Scratch (Optional)

```bash
# Train standard single-exit baseline on PCam
python train_static_baseline.py

# Jointly train multi-exit model on PCam
python train_multi_exit.py

# Apply post-training static INT8 quantization
python quantize_multi_exit.py
```