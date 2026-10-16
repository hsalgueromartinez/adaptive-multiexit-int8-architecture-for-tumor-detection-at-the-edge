# Adaptive Multi-Exit Architecture with Knowledge Distillation and Static INT8 Quantization for Clinical Tumor Detection at the Edge

Edge deployment of adaptive deep learning models for medical pathology screening and computer vision under strict battery, compute, and latency budgets. This project implements dynamic early-exit branching combined with post-training static INT8 quantization across **PatchCamelyon (PCam)** (primary clinical histology benchmark) and **Tiny-ImageNet-200** (representational scaling ablation).

---

## Pre-trained Checkpoints & Quickstart

To evaluate the pre-trained baselines, multi-exit networks, and INT8 quantized models without retraining from scratch, download our model weights from Google Drive:

```bash
# 1. Install gdown
pip install gdown

# 2. Download the pre-trained weights archive from Google Drive
gdown 1KdYwuz4KSEdVkrPgY-zybhE7W89mjH6I -O checkpoints.zip

# 3. Extract checkpoints into project root
unzip checkpoints.zip
rm checkpoints.zip
```

---

## Dataset Acquisition & Setup

Because raw medical slides and image archives are large, the dataset directories (`data/` and `data_tiny_imagenet/`) are excluded from Git version control. Follow these instructions to set them up:

### 1. PatchCamelyon (PCam) — Primary Benchmark
The `dataset.py` module uses `torchvision.datasets.PCAM(..., download=True)`. When executing any training or evaluation script on PCam for the first time, PyTorch automatically downloads and verifies the official `.h5` splits (`camelyonpatch_level_2_split_*.h5`) into the `data/pcam/` folder.

If downloading manually or transferring from another cluster:
1. Create directory: `mkdir -p data/pcam`
2. Place the following files inside `data/pcam/`:
   - `camelyonpatch_level_2_split_train_x.h5`, `camelyonpatch_level_2_split_train_y.h5`
   - `camelyonpatch_level_2_split_valid_x.h5`, `camelyonpatch_level_2_split_valid_y.h5`
   - `camelyonpatch_level_2_split_test_x.h5`, `camelyonpatch_level_2_split_test_y.h5`

### 2. Tiny-ImageNet-200 — Visual Ablation Benchmark
To download and extract the Tiny-ImageNet-200 benchmark:

```bash
# 1. Create dataset directory
mkdir -p data_tiny_imagenet
cd data_tiny_imagenet

# 2. Download official Stanford CS231n archive
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

# 3. Extract and clean up
unzip -q tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
cd ..
```
*(Ensure the directory structure matches: `data_tiny_imagenet/tiny-imagenet-200/{train,val,test}`)*

---

## Repository Structure

```text
.
├── checkpoints/                    # Saved weights (.pt) for baselines, multi-exit, and INT8 models
├── data/                           # Local storage for PatchCamelyon dataset (downloaded on first run)
├── data_tiny_imagenet/             # Tiny-ImageNet-200 dataset storage
├── FLOP/                           # FLOP calculation logs and profiling outputs
├── results/                        # Generated figures, Pareto frontiers, and benchmark summaries
│
├── dataset.py                      # Data loaders, geometric invariances & stain augmentations for PCam
├── dataset_tiny_imagenet.py        # Dataset loader and transforms for Tiny-ImageNet-200
├── multi_exit_model.py             # Shared-backbone ResNet with dynamic early exits
├── train_static_baseline.py        # Trains standard single-exit baseline models
├── train_multi_exit.py             # Jointly trains all exit heads with weighted multi-task loss
├── train_tiny_imagenet.py          # Multi-exit training on Tiny-ImageNet-200
├── quantize_multi_exit.py          # Post-training static INT8 quantization and activation calibration
├── sweep_thresholds.py             # Grid search over confidence thresholds (tau) for FP32 models
├── sweep_thresholds_int8.py        # Threshold sweep for quantized INT8 models
├── test_flops.py                   # Benchmarks theoretical FLOPs stage-by-stage using torchprofile
├── measure_latency_memory_pcam.py  # Measures real edge latency (CPU FPS) and checkpoint memory footprint
├── pareto_pcam.py                  # Evaluates and plots Accuracy vs. FLOPs Pareto frontier on PCam
├── pareto_recall_f1_pcam.py        # Evaluates clinical safety (Tumor Recall and F1-Score vs. FLOPs)
├── pareto_tiny_imagenet.py         # Evaluates Accuracy vs. FLOPs Pareto frontier on Tiny-ImageNet
└── run_ablation.py                 # Ablation on preprocessing the baseline model
```

---

## Script Descriptions

| Script | Purpose |
| :--- | :--- |
| `dataset.py` | Handles PatchCamelyon downloads, train/val/test data loaders, histology-specific discrete 90-degree rotations, and stain simulation. |
| `dataset_tiny_imagenet.py` | Loads and transforms the Tiny-ImageNet-200 dataset ($64\times 64$, 200 classes). |
| `multi_exit_model.py` | Shared-backbone ResNet with early-exit heads attached after intermediate stages for runtime dynamic execution. |
| `train_static_baseline.py` | Trains standard unbranched backbones to serve as latency, FLOP, and accuracy baselines. |
| `train_multi_exit.py` | Jointly trains all exit heads using a multi-task loss objective ($\mathcal{L}_{\text{total}} = \sum \alpha_i \mathcal{L}_i$). |
| `train_tiny_imagenet.py` | Multi-exit training on Tiny-ImageNet-200 to assess representational capacity scaling. |
| `quantize_multi_exit.py` | Performs post-training static INT8 quantization (Conv+BN+ReLU fusion, calibration, weight quantization). |
| `sweep_thresholds.py` | Sweeps decision thresholds ($\tau \in [0.5, 0.99]$) on FP32 multi-exit models to compute dynamic exit rates. |
| `sweep_thresholds_int8.py` | Sweeps decision thresholds on quantized INT8 models to assess Pareto optimality at low bit-widths. |
| `test_flops.py` | Computes theoretical FLOPs per stage using `torchprofile`. |
| `measure_latency_memory_pcam.py` | Measures single-sample edge CPU latency, throughput (FPS), and checkpoint storage footprints. |
| `pareto_pcam.py` | Generates Accuracy vs. FLOPs Pareto frontier curve on PatchCamelyon. |
| `pareto_recall_f1_pcam.py` | Computes sector-aware clinical safety metrics (Tumor Recall and F1-Score vs. FLOPs). |
| `pareto_tiny_imagenet.py` | Generates the representational scaling trade-off frontier on Tiny-ImageNet-200. |
| `run_ablation.py` | Runs the full ablation suite across benchmarks and architectures. |

---

## Step-by-Step Reproduction Guide

### 1. Environment Setup

```bash
git clone https://github.com/hsalgueromartinez/adaptive-multiexit-int8-architecture-for-tumor-detection-at-the-edge.git
cd adaptive-multiexit-int8-architecture-for-tumor-detection-at-the-edge

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

# Clinical safety frontier (Tumor Recall and F1 vs. FLOPs)
python pareto_recall_f1_pcam.py

# Visual ablation frontier (Tiny-ImageNet)
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

---

## Authors

*   **David Piera** - *Developer & Researcher* — [@hsalgueromartinez](https://github.com/hsalgueromartinez)
*   **Hector Salguero** - *Developer & Researcher* — [@hsalgueromartinez](https://github.com/hsalgueromartinez)
---
