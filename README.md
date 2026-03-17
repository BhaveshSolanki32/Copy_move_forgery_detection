# SciForge: Copy-Move Forgery Detection in Scientific Images

> **Instance-level copy-move forgery segmentation in scientific images using contrastive pixel embeddings, dual-encoder feature fusion, and SAM3-powered synthetic data generation.**

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

This project tackles **instance-level copy-move image forgery detection** in scientific research images — a task where a region is copied, optionally transformed, and pasted back onto the same image to misrepresent experimental results. The goal is to predict pixel-level segmentation masks for every forged region, evaluated using the **Object F1 (OF1)** metric, which uses the Hungarian algorithm to optimally match predicted masks to ground truth instances before computing F1.

The project involved three major components built across roughly one month:

1. **EDA** of the original dataset — image counts, size distributions, forgery area coverage, instance counts per image, and mask format analysis
2. **Synthetic data generation** at scale using SAM3 to address the limited training set size
3. **Three successive model architectures**, each motivated by the limitations discovered in the previous one

---

## The Problem: Why Copy-Move Forgery is Hard

Most forgery detection literature focuses on splicing (pasting from an *external* image). Copy-move forgery within scientific images is fundamentally harder for two reasons:

**1. The dual-region problem.** The model must find not just *where* the forged region is (the destination), but also *where it came from* (the source region in the same image). These two regions look identical — because one is literally a copy of the other. A model that learns only "what looks anomalous" will miss source regions entirely, since source regions have no visual artifact of their own.

**2. Background homogeneity.** Scientific images contain large areas of repetitive texture. A patch from the background looks nearly identical to a neighbouring background patch. This makes naive patch-matching approaches produce massive false positives. The true signal — *this exact patch appears in two distinct spatial locations* — requires learning dense, discriminative embeddings rather than pixel-level texture anomaly detectors.

This drove all three model architectures: the core idea is to make the embedding space such that **pixels within a forged instance cluster together** and are **pushed apart from the background**, enabling the model to find duplicate regions by detecting anomalously similar embeddings at distant spatial coordinates.

---

## Dataset & EDA

The original dataset contains **5,128 RGB scientific images**:

| Split | Count |
|---|---|
| Authentic images | 2,377 |
| Forged images | 2,751 |
| Mask files (`.npy`, one per forged image) | 2,751 |
| Images with both authentic and forged versions | 2,377 |

A key structural observation: 2,377 filenames appear in *both* the authentic and forged folders — meaning these images have a clean original and a manipulated version with the same name, forming a paired structure.

**Mask format:** All masks are stored as `(N, H, W)` NumPy arrays where N is the number of forged instances (confirmed across all 2,751 masks). Instance labels range 1–5 (max 5 instances per image in the ground truth).

**Instance counts per image** (range: 1–5):

Total instances across all 2,751 masks: **3,823** (average ≈ 1.39 instances per forged image).

<!-- INSERT: Bar chart — instance count distribution (x: 1 to 5, y: number of images) -->

**Forgery area coverage** (per instance, as fraction of total image area):

| Statistic | Value |
|---|---|
| Mean | 3.97% |
| Median | 1.17% |
| Std | 5.98% |
| Min | ~0% |
| Max | 52.27% |

The heavily skewed distribution (median 1.17% vs mean 3.97%) confirms most forged regions are small — directly motivating focal loss to counter severe foreground-background imbalance.

**Image dimension distribution** across all 5,128 images:

| Statistic | Width | Height |
|---|---|---|
| Mean | 931 px | 693 px |
| Median | 696 px | 512 px |
| Min | 74 px | 64 px |
| Max | 3,888 px | 3,888 px |
| Unique (W, H) combinations | 763 | — |

Top 5 most common sizes: 1000×666 (686 images), 256×256 (605), 1600×1200 (530), 696×520 (256), 320×256 (229).

763 unique dimension combinations made fixed-resolution training impossible — letterbox resizing was required throughout. EDA also explored mask-to-YOLO polygon format conversion and visualised authentic/forged pairs side-by-side with mask contour overlays.

<!-- INSERT: EDA visualisations — instance count bar chart, area ratio histogram with KDE, 2D image dimension heatmap, sample authentic/forged/mask overlay grid -->

---

## Synthetic Data Generation Pipeline

The 2,751 forged training images were insufficient for training deep segmentation models. We built a custom synthetic data generator using **Meta's SAM3 (Segment Anything Model 3)** that programmatically creates copy-move forgeries with pixel-accurate ground-truth masks.

### How It Works

```
Input Image (COCO 2017 / Scientific domain images)
        │
        ▼
  SAM3 Segmentation
  (text-prompted: "object", "animal", "human", "thing")
  Confidence threshold: 0.4
  Candidates per image: 1–8
        │
        ▼
  Copy-Move Augmentation
  ├── Objects to forge per image: 1–6
  ├── Copies per object: 1–4
  ├── Random rotation: U(−180°, +180°)
  ├── Random scale: U(0.8, 1.2)
  └── Overlap rejection: IoU < 0.05
        │
        ▼
  RLE Mask Encoding → .npy save
  (Fortran-order, JSON-serialised, semicolon-separated per instance)
```

Copy count distribution per object: 1 copy (65%), 2 copies (25%), 3 copies (8%), 4 copies (2%).

Multi-GPU DDP was used across both T4s during generation to parallelise SAM3 inference. Generation ran in configurable index chunks (e.g. 0–28,000 images per run) to fit within session time limits.

### Two Generated Datasets

| Dataset | Source | Forged Images | Authentic Images |
|---|---|---|---|
| Scientific domain | Scientific paper images | ~11,000 | ~13,000 |
| COCO-derived | COCO 2017 | ~84,700 | — |

<!-- INSERT: Sample generated images — original / forged image / mask overlay -->

### Mask Conversion

Generated masks are stored as RLE-encoded `.npy` files. `rle_convert.py` converts them to raw `(N, H, W)` NumPy arrays for model consumption:

```bash
python rle_convert.py
# Edit WRONG_MASK_DIR, IMG_DIR, NEW_MASK_DIR paths inside the script
```

---

## Model Architectures

All three models share the same core hypothesis: **learn discriminative dense embeddings that cluster by semantic/spatial identity**, then decode those embeddings into instance masks. Copy-move forgeries create regions with matching embeddings separated in space — a signature that purely local or texture-based features cannot detect.

---

### Model 1 — Hybrid Stem Swin+UNet with Contrastive Learning

**File:** `sci-forge-b.ipynb`

#### Architecture

```
Input (3, 896, 896)
     │
     ▼
ConvolutionalStem
  Block1: Conv(3→64, stride=2) + Conv(64→64)    → (64, 448, 448)
  Block2: Conv(64→128, stride=2) + Conv(128→128) → (128, 224, 224)
     │
     ▼
Swin-Small Encoder  (swin_small_patch4_window7_224)
  Stage outputs: (96, 56×56), (192, 28×28), (384, 14×14), (768, 7×7)
     │
     ▼
UNet Decoder  (ConvTranspose2d + skip connections)
  Channels: 512 → 256 → 128 → 64 → 32
     │
     ├──▶ Segmentation Head        → (6, 224, 224) mask logits
     ├──▶ Active Channel Head      → (6,) — is this channel a real instance?
     └──▶ ContrastiveEmbeddingHead → (128, 224, 224) L2-normalised embeddings
```

**Key design decisions:**

- **ConvolutionalStem:** Swin's standard patch embedding discards fine-grained texture. The CNN stem processes the 896px input through two strided conv blocks before handing 224px feature maps to Swin — preserving boundary detail for tight mask edges.
- **Contrastive embedding head:** Alongside the segmentation decoder, a separate branch produces 128-dim L2-normalised per-pixel embeddings, trained to push forged-instance pixels together and away from background.
- **Active channel prediction:** A binary classification head predicts which of the 6 output channels contains a real forgery instance, suppressing false-positive masks from empty channels. Weight 4.0 in the combined loss.

#### Training Config

| Parameter | Value |
|---|---|
| Input size | 896 × 896 |
| Mask output size | 224 × 224 |
| Max output instances | 6 |
| Epochs | 45 |
| Batch size | 26 (gradient accumulation ×10) |
| Optimizer | AdamW (lr=4e-4, wd=1e-4) |
| Scheduler | CosineAnnealing (η_min=1e-6) |
| Mixed precision | AMP |
| Encoder freeze | First 1 epoch |
| Contrastive temperature | 0.07 |
| BG samples / instance samples | 256 / 64 |

<!-- INSERT: Model 1 training curves — component losses, validation OF1 vs epoch, LR schedule -->

---

### Model 2 — Dual-Encoder Swin + ConvNeXt Fusion

**File:** `sci-forge-b-tri.ipynb`

#### Motivation

Model 1's single encoder had to simultaneously handle global context (where are the duplicate regions?) and local texture (what are the exact boundaries?). Model 2 **disentangles** these two responsibilities into dedicated encoders.

#### Architecture

```
Input (3, 224, 224)          Input (3, 448, 448)
        │                             │
        ▼                             ▼
  Swin-Small                    ConvNeXt-Small
  (global context,              (local texture,
   similarity learning)          boundary detail)
        │                             │
        └──────────┬──────────────────┘
                   ▼
             FusionBlock
             BN(Swin) + BN(ConvNeXt) → Concat → Conv1×1 → 2/3 channels
                   │
          ┌────────┴──────────────────┐
          ▼                           ▼
   Auxiliary Heads               UNet Decoder
   ├── Swin: Contrastive loss     (512→256→128→64→32)
   └── ConvNeXt: Coarse seg            │
                                  Segmentation Head +
                                  Active Channel Head
```

**Key design decisions:**

- **CoordConv:** Normalised x, y coordinate channels are concatenated to the input before the ConvNeXt branch, giving the model explicit spatial awareness — useful for detecting that two identical patches exist at different spatial coordinates.
- **FusionBlock:** Independently batch-normalises each encoder's features before concatenation, then reduces to 2/3 of combined channels via a learned 1×1 conv to avoid one encoder dominating.
- **Separate loss weights per head:** Swin branch driven by contrastive loss (weight 2.0); ConvNeXt branch driven by coarse segmentation (weight 0.8), so gradients specifically train each encoder for its intended function.

#### Training Config

| Parameter | Value |
|---|---|
| Swin input | 224 × 224 |
| ConvNeXt input | 448 × 448 |
| Epochs | 50 (early stopping, patience=5) |
| Batch size | 17 |
| Optimizer | AdamW (lr=5e-4, wd=1e-4) |
| Loss weights | Dice: 2.0, Swin contrastive: 2.0, ConvNeXt coarse: 0.8, Active focal: 4.0 |
| Encoder freeze | First 10 epochs |

<!-- INSERT: Model 2 training curves — per-head loss breakdown, validation OF1 -->

---

### Model 3 — DINOv2 + MLP + HDBSCAN Clustering

**File:** `sci-forge-b-fins.ipynb`

#### Motivation

Models 1 and 2 treated mask prediction as supervised segmentation. But copy-move forgery is fundamentally a *self-similarity* problem. An **unsupervised clustering** approach better fits: produce high-quality patch embeddings, cluster them, and identify clusters that represent spatial duplicates.

DINOv2 was chosen because its self-supervised pre-training already produces spatially coherent, semantically rich patch-level features — well suited for clustering without a dense prediction head.

#### Architecture

```
Input (3, 462, 462)
       │
       ▼
DINOv2 ViT-B/14
462 / 14 = 33 patches per side → 33×33 patch grid
Patch token dim: 768
       │
       ▼
MLP Projection Head
  fc1: 768 → 1024  (LayerNorm + ReLU)
  fc2: 1024 → 768  (LayerNorm + ReLU)
  fc3: 768 → 512
  L2-normalise → (B, 1089, 512)
       │
       ▼
Reshape → (B, 33, 33, 512) patch embeddings
       │
   TRAINING                          INFERENCE
   Uniformity Loss                   HDBSCAN Clustering
   Separation Loss                   on (33×33, 512) embeddings
   Triplet Loss (hard mining)              │
                                     Cluster Validation
                                     (cosine dist from bg centroid)
                                           │
                                     Mask upsampling 33×33 → 462×462
```

**Key design decisions:**

- **Input size 462px:** `462 / 14 = 33` exactly — whole-number patches, no boundary artefacts from partial patches at image edges.
- **HDBSCAN inference:** Discovers natural cluster structure without requiring the number of instances in advance — important since instance count varies per image.
- **Cluster validation:** Each candidate cluster's centroid cosine distance from the background centroid is computed; only sufficiently distant clusters become mask predictions, filtering spurious clusters from genuine texture variation.
- **Fine-tuning strategy:** After initial training on synthetic data, fine-tuned on 100% competition test images + 6× augmented HQ images (48 × 6 = 288 samples/epoch) + 30% synthetic data (~12,488 total samples/epoch). Both DINOv2 and MLP unfrozen at lr=1e-6.

#### Encoder Freeze Schedule

| Phase | DINOv2 | MLP lr |
|---|---|---|
| Epochs 1–10 | Frozen | 1e-3 |
| Epochs 11–40 | Fine-tuning (lr=1e-5) | 1e-3 |
| Fine-tune phase (30 epochs) | Fine-tuning (lr=1e-6) | 1e-6 |

<!-- INSERT: Model 3 training curves — uniformity loss, separation loss, LR schedule, HDBSCAN parameter search heatmap -->

---

## Loss Functions

### Segmentation Losses (Models 1 & 2)

**Dice Loss** — directly optimises region overlap; critical for small forgery regions (median area 1.17%) that BCE alone would mostly ignore.

$$\mathcal{L}_{Dice} = 1 - \frac{2 \cdot |P \cap G| + \varepsilon}{|P| + |G| + \varepsilon}$$

**Segmentation Focal Loss** — addresses severe foreground-background imbalance.
$$\mathcal{L}_{Focal} = -\alpha (1-p_t)^\gamma \log(p_t)$$
Model 1: α=0.9, γ=2.0 &nbsp;|&nbsp; Model 2: α=1.0, γ=2.0

**Active Channel Focal Loss** — binary focal loss on the instance-presence head. Weight 4.0 in both models.

### Contrastive Loss (Models 1 & 2)

Per image: up to 256 background samples and up to 64 samples per active forgery instance.

| Margin | Value | Weight |
|---|---|---|
| Background–Foreground | 1.5 | 2.0 |
| Foreground–Foreground (different instances) | 1.0 | 1.2 |

### Embedding Losses (Model 3)

**Uniformity Loss** (weight 0.5) — penalises high intra-instance embedding variance only when variance exceeds 0.1 threshold, preventing loss collapse.

**Separation Loss** (weight 2.5) — maximises cosine distance between instance centroids. Highest weight — the primary detection signal.

**Triplet Loss** (weight 0.4) — hard-mined triplets with the closest negative, adding fine-grained separation pressure beyond centroid-level losses.

---

## Training Infrastructure

All models used **PyTorch DistributedDataParallel (DDP)** across 2 T4 GPUs with NCCL backend.

| Component | Implementation |
|---|---|
| Multi-GPU | PyTorch DDP, NCCL backend |
| Mixed precision | `torch.cuda.amp` (GradScaler + autocast) |
| Gradient accumulation | ×2–×10 depending on model |
| Image resizing | Letterbox (aspect-ratio-preserving centre pad) |
| Augmentation | HFlip, VFlip, Rotate, ColorJitter, GaussianBlur, GaussNoise, JPEG compression (Albumentations) |
| Checkpointing | `best_checkpoint.pth` + `current_checkpoint.pth` saved per epoch |
| Threshold tuning | Grid search over active-channel threshold × mask threshold on stratified 15% of training data |
| Validation | Sampled validation (20–50% of val set), 30% authentic / 70% forged split |

**OF1 metric:**

```python
from scipy.optimize import linear_sum_assignment

cost_matrix = compute_iou_matrix(pred_masks, gt_masks)
row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # Hungarian matching
# Authentic images: score = 1.0 if no masks predicted, else 0.0
```

---

## Results

| Model | Architecture | Best Val OF1 |
|---|---|---|
| Model 1 | Hybrid CNN Stem + Swin-Small + UNet + Contrastive | — |
| Model 2 | Dual Encoder (Swin + ConvNeXt) + Fusion + Auxiliary Heads | — |
| Model 3 | DINOv2 ViT-B/14 + MLP + HDBSCAN | **0.3727** (epoch 39) |

> **Note on OF1 scores:** An OF1 of ~0.37 on this task is harder to achieve than the number suggests. The metric requires precise instance-level mask predictions matched via the Hungarian algorithm — detecting a forgery is not enough, the mask boundary must be accurate. More importantly, the *source region* (the patch copied from) carries no visual anomaly yet must also be detected, requiring the model to compare patches across the entire image rather than making local predictions.

<!-- INSERT: OF1 bar chart comparing all three models -->

<!-- INSERT: Qualitative predictions — best / median / worst cases (image | ground truth | prediction overlay) -->

---

## Repository Structure

```
sci-forge/
├── sci-forge-b.ipynb                  # Model 1: Hybrid Swin+UNet
│   ├── config.py
│   ├── model.py                       # ConvStem + Swin + UNet + ContrastiveHead
│   ├── dataset.py                     # ForgeryDataset with letterbox resizing
│   ├── losses.py                      # Dice + SegFocal + Contrastive + ActiveFocal
│   ├── train.py                       # DDP training loop
│   ├── threshold.py                   # Grid search over active/mask thresholds
│   ├── test.py                        # Evaluation + visualisation
│   └── utils.py                       # RLE encode/decode, OF1 metric
│
├── sci-forge-b-tri.ipynb              # Model 2: Dual-Encoder Fusion
│   ├── config.py
│   ├── model.py                       # CoordConv + Swin + ConvNeXt + FusionBlock + UNet
│   ├── dataset.py                     # Dual-resolution loader (224 + 448)
│   ├── losses.py                      # Dice + Focal + Contrastive + CoarseSeg
│   ├── train.py                       # DDP + early stopping (patience=5)
│   ├── threshold.py
│   └── test.py
│
├── sci-forge-b-fins.ipynb             # Model 3: DINOv2 + HDBSCAN
│   ├── config.py
│   ├── model.py                       # DINOv2 ViT-B/14 + 3-layer MLP projection
│   ├── dataset.py                     # 462px letterbox, 33×33 mask downsampling
│   ├── losses.py                      # Uniformity + Separation + Triplet
│   ├── clustering.py                  # HDBSCAN + cluster candidate validation
│   ├── train.py                       # DDP training loop
│   ├── tuning.py                      # Parallel CPU HDBSCAN hyperparameter search
│   ├── extract_embeddings.py          # GPU embedding extraction for CPU tuning
│   ├── finetune_config.py
│   ├── finetune_dataset.py            # RepeatedDataset + SubsampledDataset
│   ├── finetune_train.py
│   └── test.py
│
├── sci-forge-dataset-generation.ipynb # SAM3 synthetic data generation
│   └── generate_final.py              # Multi-GPU DDP generation pipeline
│
├── forgery-detection-eda-visual-annotations.ipynb  # EDA
├── rle_convert.py                     # RLE .npy → raw (N,H,W) mask converter
└── test.ipynb / test2.ipynb           # Dataset visualisation, checkpoint inspection
```

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision timm albumentations \
            hdbscan numba scipy scikit-learn \
            opencv-python Pillow tqdm seaborn pandas
```

For synthetic data generation:

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .
pip install --force-reinstall numpy==1.26.4 numba  # Fix numpy/numba incompatibility
```

### Training (Model 3 — DINOv2)

Update paths in `config.py`, then:

```bash
python train.py               # Initial training (40 epochs)
python extract_embeddings.py  # Extract embeddings for CPU-side HDBSCAN tuning
python tuning.py              # HDBSCAN hyperparameter search
python finetune_train.py      # Fine-tuning on mixed dataset (30 epochs)
python test.py                # Evaluation + visualisation
```

### Synthetic Data Generation

Edit `START_INDEX` / `END_INDEX` in `generate_final.py` for batched runs:

```bash
python generate_final.py
```

---

## Key Takeaways

**What worked:**

- Embedding-based learning consistently outperformed pure segmentation for this task. The self-similarity structure of copy-move forgery is naturally suited to metric learning approaches.
- DINOv2's self-supervised patch representations needed only a lightweight MLP projection — pre-trained features already captured enough structure to support HDBSCAN clustering.
- HDBSCAN handled variable instance counts (1–5 per image) without needing a fixed cluster count as input.
- Synthetic data at scale (SAM3-generated) was essential; 2,751 forged training images alone were insufficient for deep models.

**What was difficult:**

- Repetitive backgrounds in scientific images caused high false positive rates — background texture clusters were sometimes incorrectly promoted as forgeries.
- OF1 penalises imprecise mask boundaries heavily; correctly detecting a forgery but producing an oversized mask still scores poorly.
- Source regions carry no visual anomaly and must be detected purely through spatial duplication — requiring full-image patch comparison rather than local predictions.

**What would come next:**

- Explicit correlation-map approaches: compute pairwise similarity between all patch embedding pairs across an image to directly localise spatial duplicates
- Ensemble of segmentation (Model 1/2) and embedding (Model 3) approaches for complementary precision and recall

---

*Built with PyTorch, timm, DINOv2, SAM3, HDBSCAN, and Albumentations.*
