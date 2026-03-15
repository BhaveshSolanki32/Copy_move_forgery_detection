# SciForge: Copy-Move Forgery Detection in Scientific Images

> **Instance-level copy-move forgery segmentation in biomedical research images using contrastive pixel embeddings, dual-encoder feature fusion, and SAM3-powered synthetic data generation.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [The Problem: Why Copy-Move Forgery is Hard](#the-problem-why-copy-move-forgery-is-hard)
- [Dataset & EDA](#dataset--eda)
- [Synthetic Data Generation Pipeline](#synthetic-data-generation-pipeline)
- [Model Architectures](#model-architectures)
  - [Model 1 — Hybrid Stem Swin+UNet with Contrastive Learning](#model-1--hybrid-stem-swinunet-with-contrastive-learning)
  - [Model 2 — Dual-Encoder Swin + ConvNeXt Fusion](#model-2--dual-encoder-swin--convnext-fusion)
  - [Model 3 — DINOv2 + MLP + HDBSCAN Clustering](#model-3--dinov2--mlp--hdbscan-clustering)
- [Loss Functions](#loss-functions)
- [Training Infrastructure](#training-infrastructure)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [Key Takeaways](#key-takeaways)

---

## Project Overview

**Team:** 3 members &nbsp;|&nbsp; **Duration:** ~1 month &nbsp;|&nbsp; **Hardware:** 2× NVIDIA T4 GPUs

This project tackles **instance-level copy-move image forgery detection** in biomedical research images — a task where a region of a scientific image (e.g., a microscopy slide or gel blot) is copied, possibly transformed, and pasted back onto the same image to misrepresent experimental results. The goal is to predict binary pixel-level segmentation masks identifying every forged region, evaluated using the **Object F1 (OF1)** metric — which uses the Hungarian algorithm to optimally match predicted masks to ground truth instances.

The project involved three major components built over the course of a month:

1. **EDA** of the original dataset (image size distributions, DPI, forgery area coverage, instance counts)
2. **Synthetic data generation** at scale using SAM3 to combat data scarcity
3. **Three successive model architectures**, each addressing limitations discovered in the previous iteration

---

## The Problem: Why Copy-Move Forgery is Hard

Most image forgery detection literature focuses on splicing (pasting from a different image). Copy-move forgery within scientific images is fundamentally harder for two reasons:

**1. The dual-region problem:** The model must not only find *where* the forged region is (the destination), but also implicitly reason about *where it came from* (the source region in the same image). These two regions look identical — because one is literally a copy of the other. A model that learns only "what looks weird" will miss source regions entirely.

**2. Background homogeneity:** Scientific images (microscopy, gel electrophoresis, western blots) contain large areas of repetitive textures. A patch from the background looks nearly identical to a neighboring background patch. This causes naive patch-matching or texture-anomaly approaches to produce massive numbers of false positives. The true signal — *this exact patch appears in two distinct spatial locations* — is subtle and requires learning dense, discriminative patch embeddings rather than pixel-level texture features.

This insight drove all three model architectures: the core idea was to make the embedding space such that **pixels within a forged instance cluster together** and are **pushed away from the background cluster**, enabling the model to find duplicate regions by searching for anomalously similar embeddings in distant spatial locations.

---

## Dataset & EDA

The original dataset contains **5,128 RGB biomedical images** from multiple scientific imaging modalities:

| Split | Authentic | Forged |
|---|---|---|
| Train | ~1,500 | ~3,600 |
| Supplemental HQ | 48 | 48 |

**EDA findings (performed by teammates):**

- Image resolutions varied widely (not square, not fixed DPI) — required letterbox resizing to preserve aspect ratio
- Forgery area coverage: median ~8% of total image area, highly skewed — most forgeries are small
- Instance count distribution: 1 instance dominates (~55%), with a long tail up to 6
- DPI distributions differed across imaging modalities (blot vs. microscopy vs. fluorescence)

<!-- INSERT: EDA graphs — image size distribution, forgery area histogram, instance count bar chart -->

These findings informed several design choices: letterbox padding instead of stretch resizing, focal loss to handle extreme foreground-background imbalance, and capping maximum instances at 6.

---

## Synthetic Data Generation Pipeline

The original dataset's ~3,600 forged training images were insufficient for training deep segmentation models. We built a custom synthetic data generator using **Meta's SAM3 (Segment Anything Model 3)** that programmatically creates copy-move forgeries with ground-truth masks.

### Pipeline Architecture

```
Input Image (COCO / Scientific)
        │
        ▼
  SAM3 Segmentation
  (Text-prompted: "object", "animal", "human", "thing")
        │
        ▼
  Candidate Patch Selection
  (CONFIDENCE > 0.4, 1–8 candidates)
        │
        ▼
  Copy-Move Augmentation
  ├── Random rotation: U(-180°, +180°)
  ├── Random scale: U(0.8, 1.2)
  └── Overlap rejection: IoU < 0.05
        │
        ▼
  RLE Mask Encoding & Save
  (Fortran-order, JSON-serialized, semicolon-separated per instance)
```

### Two Generated Datasets

| Dataset | Source | Forged Images | Notes |
|---|---|---|---|
| `synthetic_forgery_dataset` | Scientific domain images | ~11,000 | Same image modality as test set |
| `coco_forgery_dataset` | COCO 2017 (80k natural images) | ~84,700 | Scale diversity, generalizable features |

**Instance distribution (per image):**

```
1 instance: 30%   2 instances: 25%   3 instances: 20%
4 instances: 12%  5 instances: 8%    6 instances: 5%
```

**Copy count distribution per object:**
```
1 copy: 65%   2 copies: 25%   3 copies: 8%   4 copies: 2%
```

DDP was used across both GPUs during generation to parallelize SAM3 inference, enabling generation of the full 84.7k COCO dataset in a single session.

<!-- INSERT: Sample synthetic images — original / forged / mask overlay -->

### Mask Format

Masks are stored as RLE-encoded `.npy` files. Each file contains a semicolon-separated string of JSON-encoded `[start, length, start, length, ...]` run-length pairs (1-based indexing, Fortran column-major order — matching the competition ground truth format). A `rle_convert.py` utility converts these to raw `(N, H, W)` NumPy arrays for model consumption.

---

## Model Architectures

All three models share the same high-level philosophy: **learn discriminative dense embeddings that cluster by semantic identity**, then decode those embeddings into instance masks. The key insight is that copy-move forgeries produce regions with matching embeddings separated in space — a signature that pixel-level or purely local features cannot capture.

---

### Model 1 — Hybrid Stem Swin+UNet with Contrastive Learning

**File:** `sci-forge-b.ipynb`

#### Architecture

```
Input (3, 896, 896)
     │
     ▼
ConvolutionalStem
 Block1: Conv(3→64, s=2) + Conv(64→64) → (64, 448, 448)
 Block2: Conv(64→128, s=2) + Conv(128→128) → (128, 224, 224)
     │
     ▼
Swin-Small Encoder (swin_small_patch4_window7_224)
  └── 4 stage outputs: (96,56), (192,28), (384,14), (768,7)
     │
     ▼
UNet Decoder (ConvTranspose2d + skip connections)
  Channels: 512 → 256 → 128 → 64 → 32
     │
     ├──▶ Segmentation Head → (NUM_INSTANCES, 224, 224) mask logits
     ├──▶ Active Channel Head → (NUM_INSTANCES,) — is instance active?
     └──▶ ContrastiveEmbeddingHead → (128, 224, 224) L2-normalized embeddings
```

**Key design decisions:**

- **ConvolutionalStem:** Swin's patch embedding discards fine-grained texture. The CNN stem first processes the 896px input at full resolution through two strided convolution blocks before handing 224px feature maps to Swin, preserving edge and boundary information that matters for tight mask boundaries.
- **Contrastive embedding head:** Alongside the segmentation head, a separate branch produces 128-dim L2-normalized embeddings per spatial location. These are trained with a margin-based contrastive loss so that forged-instance pixels embed closer to each other and farther from the background.
- **Active channel prediction:** Since instances can vary from 0–6, a binary classification head predicts which of the 6 output channels actually contains a forgery, reducing false positive masks.

#### Training Config

| Parameter | Value |
|---|---|
| Input size | 896 × 896 |
| Mask output size | 224 × 224 |
| Max instances | 6 |
| Epochs | 45 |
| Batch size | 26 (w/ grad accumulation ×10) |
| Optimizer | AdamW (lr=4e-4, wd=1e-4) |
| Scheduler | CosineAnnealing (η_min=1e-6) |
| Mixed precision | ✅ AMP |
| Encoder freeze | First 1 epoch |

<!-- INSERT: Model 1 training curves — loss components, OF1 vs epoch, LR schedule -->

---

### Model 2 — Dual-Encoder Swin + ConvNeXt Fusion

**File:** `sci-forge-b-tri.ipynb`

#### Motivation

Model 1's single Swin encoder had to simultaneously handle global context (where are the duplicate regions?) and local texture (what are the exact boundaries?). Disentangling these two tasks into dedicated encoders — one for similarity, one for localization — was the core hypothesis here.

#### Architecture

```
Input (3, 224, 224)  Input (3, 448, 448)
        │                     │
        ▼                     ▼
  Swin-Small            ConvNeXt-Small
  Encoder               Encoder
  (global context,      (local texture,
   similarity)          boundary detail)
        │                     │
        └──────┬───────────────┘
               ▼
         FusionBlock
         (BN → Concat → Conv 1×1 → 2/3 channel reduction)
               │
        ┌──────┴──────────────┐
        ▼                     ▼
   Auxiliary Heads        UNet Decoder
   ├── Swin: Contrastive  (512→256→128→64→32)
   │         embedding         │
   └── ConvNeXt: Coarse        ▼
       segmentation      Segmentation Head +
                         Active Channel Head
```

**Key design decisions:**

- **CoordConv:** Coordinate channels (normalized x, y) are concatenated to the input before the ConvNeXt branch. This gives the model explicit spatial awareness — critical for detecting that *two identical patches appear at different spatial coordinates*.
- **FusionBlock:** Rather than naively concatenating encoder features, the fusion block first normalizes each independently (removing scale differences between architectures), concatenates, then reduces to 2/3 of the combined channels via a learned 1×1 convolution.
- **Auxiliary heads:** The Swin branch drives contrastive learning (find similar patches); the ConvNeXt branch drives coarse segmentation (find any forged region). This division of labor allows loss gradients to specifically train each encoder for its intended function.

#### Training Config

| Parameter | Value |
|---|---|
| Swin input | 224 × 224 |
| ConvNeXt input | 448 × 448 |
| Epochs | 50 (with early stopping, patience=5) |
| Batch size | 17 |
| Optimizer | AdamW (lr=5e-4) |
| Loss weights | Swin contrastive: 2.0, ConvNeXt coarse: 0.8, Dice: 2.0 |

<!-- INSERT: Model 2 training curves — per-head loss breakdown, validation OF1 -->

---

### Model 3 — DINOv2 + MLP + HDBSCAN Clustering

**File:** `sci-forge-b-fins.ipynb`

#### Motivation

Models 1 and 2 treated mask prediction as a *supervised segmentation* task. But the fundamental nature of copy-move forgery is *self-similarity* — the forged region matches a source region. This calls for an *unsupervised clustering* approach: produce high-quality patch embeddings, then cluster them, and identify which clusters are duplicates.

DINOv2 was chosen because its self-supervised pre-training already produces semantically rich, spatially coherent patch-level features — ideal for a clustering-based approach.

#### Architecture

```
Input (3, 462, 462)
      │
      ▼
DINOv2 ViT-B/14
(462/14 = 33 patches per side → 33×33 patch grid)
      │
      ▼
 Patch tokens: (B, 1089, 768)
      │
      ▼
MLP Projection Head
  fc1: 768 → 1024 (LayerNorm + ReLU)
  fc2: 1024 → 768 (LayerNorm + ReLU)
  fc3: 768 → 512
  L2-normalize → (B, 1089, 512)
      │
      ▼
Reshape → (B, 33, 33, 512) embeddings
      │
   ┌──┴──── TRAINING ─────┐         ┌── INFERENCE ──────────────┐
   │                      │         │                           │
   ▼                      │         ▼                           │
Uniformity Loss            │     HDBSCAN Clustering              │
(low variance per instance)│     on (33×33, 512) embeddings      │
Separation Loss            │         │                           │
(high cosine distance      │     Cluster Validation              │
 between instances)        │     (cosine dist from bg centroid)  │
Triplet Loss               │         │                           │
(hard mining)              │     Mask Upsampling (33×33 → 462)   │
   └──────────────────────┘     └──────────────────────────────┘
```

**Key design decisions:**

- **Input size 462px:** Chosen so that `462 / 14 = 33` — a whole number of patches. This avoids partial patches at image boundaries that confuse the transformer attention pattern.
- **HDBSCAN inference:** Rather than a fixed-threshold decoder, HDBSCAN discovers the natural cluster structure of the patch embeddings without requiring the number of clusters in advance. This is important because the number of forged instances varies per image.
- **Cluster validation:** Not every HDBSCAN cluster corresponds to a forgery. A validation step computes the cosine distance between each candidate cluster's centroid and the background cluster's centroid. Only sufficiently distant clusters are promoted to mask predictions, filtering out clusters driven by genuine texture variation rather than duplication.
- **Fine-tuning strategy:** After initial training on the synthetic dataset, the model is fine-tuned on a mixture of 100% competition test images + 6× augmented HQ images + 30% synthetic data. DINOv2 and MLP are both unfrozen at extremely low learning rate (1e-6) to prevent forgetting the pre-trained representations.

#### Encoder Freeze Schedule

| Phase | DINOv2 | MLP |
|---|---|---|
| Epochs 1–10 | ❄️ Frozen | 🔥 Training (lr=1e-3) |
| Epochs 11–40 | 🔥 Fine-tuning (lr=1e-5) | 🔥 Training (lr=1e-3) |
| Fine-tune phase | 🔥 (lr=1e-6) | 🔥 (lr=1e-6) |

<!-- INSERT: Model 3 training curves — uniformity loss, separation loss, HDBSCAN parameter search -->

---

## Loss Functions

### Segmentation Losses (Models 1 & 2)

**Dice Loss:** Directly optimizes region overlap. Critical for small forgery regions that BCE would otherwise ignore.

$$\mathcal{L}_{Dice} = 1 - \frac{2 \cdot |P \cap G| + \varepsilon}{|P| + |G| + \varepsilon}$$

**Segmentation Focal Loss:** Addresses foreground-background imbalance (typical forgery covers ~8% of image pixels).

$$\mathcal{L}_{Focal} = -\alpha (1-p_t)^\gamma \log(p_t)$$

with α=0.9, γ=2.0

**Active Channel Loss:** Binary focal loss on the instance-presence prediction head, preventing empty channels from generating false positive masks.

### Contrastive Loss (Models 1 & 2)

For each image in a batch, pixels are sampled from:
- Background: up to 256 samples
- Each active forgery instance: up to 64 samples per instance

A margin-based contrastive loss is applied:
- **Background–Foreground margin:** 1.5 (push forgery pixels away from background)
- **Foreground–Foreground margin:** 1.0 (push pixels of *different* instances away from each other)

```
Loss_contrastive = w_bg_fg * max(0, margin_bg_fg - dist(bg, fg))
                 + w_fg_fg * max(0, margin_fg_fg - dist(fg_i, fg_j))
```

### Embedding Losses (Model 3)

**Uniformity Loss:** Penalizes high intra-instance variance. Only activated when variance exceeds threshold (0.1), allowing the model to focus on separation rather than collapse.

**Separation Loss:** Maximizes cosine distance between centroids of different instances (background + forgeries). Weight 2.5× — prioritized over uniformity.

**Triplet Loss (hard mining):** Anchors from one instance, positives from the same instance, negatives from another instance — selected as the *hardest* (closest) negative to maximize margin efficiency.

---

## Training Infrastructure

All models were trained with **PyTorch DistributedDataParallel (DDP)** across 2 T4 GPUs.

```python
# DDP launch pattern used across all notebooks
mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

**Shared infrastructure across all models:**

| Component | Implementation |
|---|---|
| Multi-GPU training | PyTorch DDP (NCCL backend) |
| Mixed precision | `torch.cuda.amp` (GradScaler + autocast) |
| Gradient accumulation | 2–10 steps depending on model |
| Letterbox resizing | Aspect-ratio-preserving padding |
| Data augmentation | Albumentations (flip, rotate, color jitter, Gaussian blur, JPEG compression) |
| Checkpointing | Best + current checkpoints, saved per epoch |
| Threshold tuning | Grid search on active-channel threshold × mask threshold, evaluated on 15% of training data |
| Validation | Sampled validation (20–50% of val set) with 30/70 authentic/forged ratio |

**Evaluation metric — OF1:**

The Object F1 score uses the Hungarian algorithm (linear sum assignment) to optimally match predicted masks to ground truth masks, then computes F1 based on the best-matched pairs. Authentic images contribute score=1.0 if no masks are predicted, 0.0 otherwise. The final score is the average across all images.

```python
# Simplified OF1 logic
from scipy.optimize import linear_sum_assignment

cost_matrix = compute_iou_matrix(pred_masks, gt_masks)
row_ind, col_ind = linear_sum_assignment(-cost_matrix)
matched_ious = cost_matrix[row_ind, col_ind]
```

---

## Results

| Model | Architecture | Best Val OF1 |
|---|---|---|
| Model 1 | Hybrid CNN Stem + Swin-Small + UNet + Contrastive | — |
| Model 2 | Dual Encoder (Swin + ConvNeXt) + Fusion + Auxiliary Heads | — |
| Model 3 | DINOv2 ViT-B/14 + MLP + HDBSCAN Clustering | **0.3727** |

> **Note on scores:** An OF1 of ~0.37 is substantially harder to achieve than it looks. The metric requires the model to precisely identify the *forged region* (not just "something is wrong here") at the instance level, with matching penalized by the Hungarian algorithm. Perfect localization of the background — the dominant class — contributes nothing; only correct, tightly-bounded forgery masks score points. Additionally, the source region (where the copy came from) must also be found and masked — the model effectively needs to identify two visually identical regions and flag both.

<!-- INSERT: Final per-model OF1 comparison bar chart -->

<!-- INSERT: Qualitative predictions — best / average / worst cases (image | ground truth | prediction overlay) -->

<!-- INSERT: Training curves for best model (Model 3) — loss, separation, uniformity, LR -->

---

## Repository Structure

```
sci-forge/
├── sci-forge-b.ipynb               # Model 1: Hybrid Swin+UNet
│   ├── config.py                   # Hyperparameters & paths
│   ├── model.py                    # ConvStem + Swin + UNet + Contrastive head
│   ├── dataset.py                  # ForgeryDataset (letterbox, multi-instance masks)
│   ├── losses.py                   # Dice + SegFocal + Contrastive + Active focal
│   ├── train.py                    # DDP training loop
│   ├── threshold.py                # Grid search over active/mask thresholds
│   ├── test.py                     # Evaluation + visualization
│   └── utils.py                    # RLE encode/decode, OF1 metric
│
├── sci-forge-b-tri.ipynb           # Model 2: Dual-Encoder Fusion
│   ├── config.py
│   ├── model.py                    # CoordConv + Swin + ConvNeXt + FusionBlock + UNet
│   ├── dataset.py                  # Dual-resolution loader (224 + 448)
│   ├── losses.py                   # Dice + Focal + Contrastive + Coarse seg
│   ├── train.py                    # DDP training + early stopping
│   ├── threshold.py
│   └── test.py
│
├── sci-forge-b-fins.ipynb          # Model 3: DINOv2 + HDBSCAN
│   ├── config.py
│   ├── model.py                    # DINOv2 ViT-B/14 + 3-layer MLP projection
│   ├── dataset.py                  # 462px letterbox loader, 33×33 mask downsampling
│   ├── losses.py                   # Uniformity + Separation + Triplet
│   ├── clustering.py               # HDBSCAN + candidate validation
│   ├── train.py                    # DDP training loop
│   ├── tuning.py                   # HDBSCAN hyperparameter search (parallel CPU)
│   ├── finetune_config.py          # Fine-tuning config (mixed dataset)
│   ├── finetune_dataset.py         # RepeatedDataset + SubsampledDataset
│   ├── finetune_train.py           # Fine-tuning script
│   ├── extract_embeddings.py       # GPU embedding extraction → CPU HDBSCAN tuning
│   └── test.py
│
├── sci-forge-dataset-generation.ipynb  # SAM3 synthetic data generation
│   └── generate_final.py           # Multi-GPU generation pipeline
│
├── rle_convert.py                  # RLE string → raw (N, H, W) mask converter
├── test.ipynb                      # Dataset visualization & mask verification
└── test2.ipynb                     # Grid visualization, checkpoint inspection
```

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision timm albumentations \
            hdbscan numba scipy scikit-learn \
            opencv-python Pillow tqdm
```

For synthetic data generation:

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .
pip install --force-reinstall numpy==1.26.4 numba
```

### Training (Model 3 — DINOv2)

Update paths in `sci-forge-b-fins/config.py`, then:

```bash
# Initial training
python train.py

# HDBSCAN parameter search
python tuning.py

# Fine-tuning on mixed dataset
python finetune_train.py

# Evaluation
python test.py
```

### Converting Masks

```bash
python rle_convert.py
# Reads from: synthetic_forgery_dataset/masks/
# Writes to:  synthetic_forgery_dataset/masks_converted/
```

### Synthetic Data Generation

Edit `START_INDEX` and `END_INDEX` in `generate_final.py` for batch processing, then:

```bash
python generate_final.py
```

---

## Key Takeaways

**What worked:**

- Contrastive / embedding-based learning consistently outperformed pure segmentation approaches for this task. The self-similarity structure of copy-move forgery is naturally suited to metric learning.
- DINOv2's self-supervised pre-training provided extremely strong patch representations out of the box, requiring only a lightweight MLP projection to be useful for this domain.
- HDBSCAN's density-based clustering handled variable instance counts without needing a fixed "number of objects" — a significant practical advantage over K-means-style approaches.
- Synthetic data from SAM3 at scale (~85k COCO-derived images) gave the model exposure to diverse patch-copy scenarios beyond the domain-specific training set.

**What was challenging:**

- Scientific images' homogeneous backgrounds caused both high false positive rates (background patches incorrectly clustered as forgeries) and low recall (forged regions with subtle boundaries getting absorbed into the background cluster).
- The OF1 metric penalizes imprecise mask boundaries heavily — models that correctly *detected* forgeries but produced oversized or shifted masks still scored poorly.
- The source region (the patch that was *copied from*) is as important to detect as the destination region, but has no visual anomaly on its own — requiring the model to compare across the entire image rather than making local predictions.

**If given more time:**

- Correlation-map based approaches (comparing patch embeddings pairwise across the full image to find duplicate pairs) would be the natural next step
- Ensemble of Models 1 and 3 (segmentation backbone + embedding backbone) for better boundary precision

---

*Built with PyTorch, timm, DINOv2, SAM3, HDBSCAN, and Albumentations.*
