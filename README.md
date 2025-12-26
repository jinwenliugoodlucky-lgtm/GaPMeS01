# GapMeS Setup Guide

This guide provides step-by-step instructions for setting up and running GapMeS experiments.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Model Initialization](#model-initialization)
- [Quick Validation](#quick-validation)
- [Custom LLFF Dataset](#custom-llff-dataset)
- [Training](#training)
- [Troubleshooting](#troubleshooting)
- [Tips](#tips)

---

## Environment Setup

### Requirements
- Python: **3.10.16**
- PyTorch: **2.0.0+cu118** (or compatible CUDA version)
- CUDA: **11.8** (recommended)

### Installation

1. **Create conda environment**
```bash
conda create -n gapmes python=3.10.16
conda activate gapmes
```

2. **Install PyTorch**
```bash
# For CUDA 11.8
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu121
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Data Preparation

### DL3DV Dataset

#### Prerequisites
- HuggingFace account registration
- HuggingFace CLI login: `huggingface-cli login`

#### Step 1: Download Dataset

**Training Set:**
```bash
python download_dl3dv.py --output_dir ./datasets/dl3dv/train
```

**Test Set (Benchmark):**
```bash
python down_load.py --output_dir ./datasets/dl3dv/test
```

#### Step 2: Data Conversion

**Convert training data:**
```bash
python scripts/convert_dl3dv_train.py \
    --input_dir ./datasets/dl3dv/train \
    --output_dir ./datasets/dl3dv/train_processed
```

**Convert test data:**
```bash
python scripts/convert_dl3dv_test.py \
    --input_dir ./datasets/dl3dv/test \
    --output_dir ./datasets/dl3dv/test_processed
```

#### Step 3: Generate Index Files

```bash
python scripts/generate_dl3dv_index.py \
    --train_dir ./datasets/dl3dv/train_processed \
    --test_dir ./datasets/dl3dv/test_processed \
    --output_dir ./assets
```

This will generate:
- `dl3dv_train_index.json`
- `dl3dv_test_index.json`

#### Alternative: Use Pre-generated Data

If you prefer to skip the data preparation steps, download our pre-processed dataset:

**Baidu Netdisk (百度网盘):**
```
通过网盘分享的文件：datasets
链接: https://pan.baidu.com/s/12pirxibjWaC-6hZ90BADBA?pwd=1234 提取码: 1234 
--来自百度网盘超级会员v5的分享
```
---

## Model Initialization

### Step 1: Create Pretrained Directory

```bash
mkdir -p pretrained
```

### Step 2: Download Pretrained Models

Download the following models to `./pretrained/`:

1. **GapMeS Base Model**
   ```bash
   wget https://huggingface.co/models/gapmes-gs-base-re10k-256x448-view2-fea94f65.pth \
        -O pretrained/gapmes-gs-base-re10k-256x448-view2-fea94f65.pth
   ```

2. **Depth Anything V2**
   ```bash
   wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth \
        -O pretrained/depth_anything_v2_vitb.pth
   ```

3. **GMFlow (Optical Flow)**
   ```bash
   wget https://huggingface.co/models/gmflow-scale1-things-e9887eda.pth \
        -O pretrained/gmflow-scale1-things-e9887eda.pth
   ```

### Pretrained Models Overview

| Model | Size | Purpose | Path |
|-------|------|---------|------|
| GapMeS Base | ~200MB | Main model | `gapmes-gs-base-re10k-256x448-view2-fea94f65.pth` |
| GapMeS Small | ~100MB | Lightweight | `gapmes-gs-small-re10k-256x256-49b2d15c.pth` |
| Depth Anything V2 Base | ~100MB | Depth estimation | `depth_anything_v2_vitb.pth` |
| Depth Anything V2 Small | ~50MB | Fast depth | `depth_anything_v2_vits.pth` |
| GMFlow | ~50MB | Optical flow | `gmflow-scale1-things-e9887eda.pth` |

### Alternative: Baidu Netdisk Download

**Pretrained Models (百度网盘):**
```
通过网盘分享的文件：pretrained
链接: https://pan.baidu.com/s/1twEGoNDDPmJXAZ44S47EPQ?pwd=1234 提取码: 1234 
--来自百度网盘超级会员v5的分享
```

After downloading, extract to `./pretrained/` directory.

---

## Quick Validation

### Using GAPMES Encoder

If you have a working GapMeS installation, you can quickly validate using GAPMES encoder:

1. **Replace encoder**
```bash
cp src/temp_core/gapmes.py src/model/encoder/encoder_depth.py
```


---

## Custom LLFF Dataset

### Using utils_llff.py Tool

We provide a convenient tool to generate LLFF format data from your own images.

#### Step 1: Prepare Your Images

Place all images in a directory:
```
my_scene/
├── IMG_0001.jpg
├── IMG_0002.jpg
├── IMG_0003.jpg
└── ...
```

#### Step 2: Generate LLFF Format

```bash
python utils_llff.py \
    --input_dir ./my_scene \
    --output_dir ./datasets/custom/my_scene \
    --camera_model SIMPLE_RADIAL \
    --max_size 1024
```

#### Step 3: Verify Output Structure

```
datasets/custom/my_scene/
├── images/              # Processed images
├── sparse/              # COLMAP reconstruction
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── poses_bounds.npy     # (N, 17) camera poses and bounds
```

#### Step 4: Run GapMeS on Custom Data

```bash
python src/main.py \
    experiment=custom \
    dataset.roots=[datasets/custom/my_scene] \
    mode=test
```

### Advanced Options

**Camera models:**
- `SIMPLE_PINHOLE`: Single focal length
- `PINHOLE`: Separate fx, fy
- `SIMPLE_RADIAL`: Single focal + radial distortion (default)
- `RADIAL`: Full radial distortion
- `OPENCV`: OpenCV distortion model

**Matching strategies:**
- `exhaustive`: All image pairs (accurate, slow)
- `sequential`: Adjacent images only (fast, for videos)

**Example with custom parameters:**
```bash
python utils_llff.py \
    --input_dir ./my_video_frames \
    --output_dir ./datasets/custom/my_video \
    --camera_model OPENCV \
    --matcher sequential \
    --max_size 0 \
    --bd_factor 0.75
```

---

## Training

### DL3DV Training

```bash
bash scripts/dl3dv_train.sh
```

### RE10K Training

```bash
bash scripts/re10k_train.sh
```

### Monitor Training

```bash
# Weights & Biases
wandb login
# Training will automatically log to wandb
```
---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or image resolution:
```yaml
# config/experiment/dl3dv.yaml
dataset:
  image_shape: [256, 256]  # Reduce from [256, 448]
  
data_loader:
  train:
    batch_size: 2  # Reduce from 4
```

### COLMAP Reconstruction Failed

Check image quality and variety:
- Ensure sufficient overlap between images (>60%)
- Avoid motion blur or overexposure
- Provide at least 10 images for robust reconstruction

### Missing Dependencies

```bash
# Install missing packages
pip install pycolmap opencv-python tqdm einops

# For visualization
pip install matplotlib plotly
```

---

## Tips

⚠️ **Important Notes:**

1. **Work in Progress**: The overall code encapsulation is still incomplete. We are actively organizing and refining other parts of the codebase.

2. **Weights & Biases Setup Required**: Before running the code, you need to:
   - Install wandb: `pip install wandb`
   - Register at [wandb.ai](https://wandb.ai)
   - Configure your API key in `src/main.py`:
     ```python
     # Replace with your own API key
     wandb.init(
         project="gapmes",
         entity="YOUR_WANDB_USERNAME",
         api_key="YOUR_API_KEY"
     )
     ```
   - Or login via command line: `wandb login`

3. **Baidu Netdisk Downloads**: For easier access, we provide Baidu Netdisk (百度网盘) versions of datasets and pretrained models.

4. **Code Organization**: Some utility scripts and modules are still being refactored. If you encounter import errors, please check the file paths and ensure all dependencies are properly installed.


