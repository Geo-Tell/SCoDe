# SCoDe: Scale-aware Co-visible Region Detection for Image Matching

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://xupan.top/Projects/scode)
[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-green)](https://www.sciencedirect.com/science/article/abs/pii/S0924271625003260)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.isprsjprs.2025.08.015-orange)](https://doi.org/10.1016/j.isprsjprs.2025.08.015)
[![HuggingFace Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/SSSSphinx/SCoDe)
[![HuggingFace Dataset](https://img.shields.io/badge/Dataset-HuggingFace-lightgrey)](https://huggingface.co/SSSSphinx/SCoDe)
[![GitHub](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/SSSSphinx/SCoDe)

</div>

<div align="center">

<span class="author-block">
  <a href="https://xupan.top/" target="_blank">Xu Pan</a><sup>1</sup>,</span>
<span class="author-block">
  <a href="https://ziminxia.github.io/" target="_blank">Zimin Xia</a><sup>2</sup>,</span>
<span class="author-block">
  <a href="http://cvrs.whu.edu.cn/index.php?m=content&c=index&a=show&catid=173&id=6" target="_blank">Xianwei Zheng</a><sup>1,*</sup></span>

<p style="font-size: 0.7em; margin: 10px 0 0 0;">
<sup>1</sup>State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University<br/>
<sup>2</sup>The VITA Lab, École Polytechnique Fédérale de Lausanne (EPFL)<br/>
<sup>*</sup>Corresponding Author
</p>

</div>

This repository contains the official code for the paper:
**[Scale-aware Co-visible Region Detection for Image Matching](https://www.sciencedirect.com/science/article/abs/pii/S0924271625003260)**
Published in *ISPRS Journal of Photogrammetry and Remote Sensing*.

## Overview

SCoDe is a scale-aware co-visible region detection model designed for robust image matching. It detects overlapping regions between image pairs while being invariant to scale variations, making it particularly effective for structure-from-motion and 3D reconstruction tasks.

This model is built upon the CCOE (Co-visible region detection with Overlap Estimation) architecture and has been trained on the MegaDepth dataset.

### Key Features

- **Scale-aware overlap region detection** - Robust to scale variations
- **Rotation-invariant matching** - Handles image rotations up to 360°
- **Multi-scale attention** - Transformer with multi-scale attention mechanisms
- **End-to-end trainable** - Fully differentiable pipeline
- **Feature extractor compatible** - Works with SIFT, SuperPoint, D2-Net, R2D2, DISK

---

## Installation

### Prerequisites

```bash
git clone https://github.com/Geo-Tell/SCoDe.git
cd SCoDe
pip install -r requirements.txt
```

### Initialize from OETR

For model initialization, we follow the approach used in [OETR](https://github.com/TencentYoutuResearch/ImageMatching-OETR). Please refer to their repository for detailed initialization procedures:

```bash
# Clone OETR for reference
git clone https://github.com/TencentYoutuResearch/ImageMatching-OETR.git
```

---

## Model Details

- **Architecture**: CCOE-based transformer with multi-scale attention
- **Backbone**: ResNet-50
- **Input Size**: 1024×1024 (configurable)
- **Training Dataset**: MegaDepth
- **Framework**: PyTorch

---

## Quick Start

### Basic Inference

```python
import torch
from src.config.default import get_cfg_defaults
from src.model import CCOE

# Load configuration
cfg = get_cfg_defaults()
cfg.merge_from_file('configs/scode_config.py')

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CCOE(cfg.CCOE).eval().to(device)

# Load pre-trained weights
model.load_state_dict(torch.load('weights/scode.pth', map_location=device))

# Model is ready for inference
with torch.no_grad():
    # Process image pair (example)
    image1 = torch.randn(1, 3, 1024, 1024).to(device)
    image2 = torch.randn(1, 3, 1024, 1024).to(device)
    output = model({'image1': image1, 'image2': image2})
    
    # Output contains:
    # - overlap_map: Co-visible region probability map
    # - scale_estimation: Scale ratio between images
```

---

## Dataset Preparation

### MegaDepth Dataset

SCoDe leverages the [MegaDepth](https://github.com/zhengqili/MegaDepth) dataset for training and evaluation. To prepare the dataset:

1. Download MegaDepth from the official repository:
   ```bash
   git clone https://github.com/zhengqili/MegaDepth.git
   ```

2. Follow the MegaDepth instructions to download and process the depth maps and image pairs.

3. Prepare dataset with:
   ```bash
   python dataset_preparation.py \
       --base_path dataset/megadepth/MegaDepth \
       --num_per_scene 5000
   ```

### Pre-trained Models and Datasets

We provide pre-trained models and datasets on Hugging Face:
- **Models**: [SSSSphinx/SCoDe](https://huggingface.co/SSSSphinx/SCoDe)
- **Dataset**: [SSSSphinx/SCoDe](https://huggingface.co/SSSSphinx/SCoDe)

Please refer to the scripts and tools in the repository for training and evaluation.

---

## Training

### Single GPU Training

```bash
python train_scode.py --num_workers 4 --epoch 15 --batch_size 4 --validation --learning_rate 1e-5
```

### Multi-GPU Distributed Training

```bash
# Training with 4 GPUs
python -m torch.distributed.launch --nproc_per_node 4 --master_port=29501 train_scode.py \
    --num_workers 4 --epoch 15 --batch_size 4 --validation --learning_rate 1e-5
```

### Configuration

Main configuration files:
- [`configs/scode_config.py`](configs/scode_config.py) - SCoDe model configuration
- [`src/config/default.py`](src/config/default.py) - Default configuration template

#### Key Parameters

```python
# Training
cfg.DATASET.TRAIN.IMAGE_SIZE = [1024, 1024]
cfg.DATASET.TRAIN.BATCH_SIZE = 4
cfg.DATASET.TRAIN.PAIRS_LENGTH = 128000
cfg.TRAINING.LEARNING_RATE = 1e-5
cfg.TRAINING.EPOCHS = 15

# Validation
cfg.DATASET.VAL.IMAGE_SIZE = [1024, 1024]

# Model
cfg.CCOE.BACKBONE.NUM_LAYERS = 50
cfg.CCOE.BACKBONE.STRIDE = 32
cfg.CCOE.CCA.DEPTH = [2, 2, 2, 2]
cfg.CCOE.CCA.NUM_HEADS = [8, 8, 8, 8]
```

---

## Evaluation

### Rotation Invariance Evaluation

Test the model's robustness to image rotations:

```bash
python rot_inv_eval.py \
    --extractors superpoint d2net r2d2 disk \
    --image_pairs path/to/image/pairs \
    --output_dir outputs/scode_rot_eval
```

### Pose Estimation Evaluation

Evaluate camera pose estimation on MegaDepth benchmark:

```bash
python eval_pose_estimation.py \
    --results_dir outputs/megadepth_results \
    --dataset megadepth
```

### Radar Evaluation

```bash
python eval_radar.py \
    --results_dir outputs/radar_results
```

---

## Model Performance

SCoDe demonstrates strong performance on:

- **Rotation Invariance**: Robust to image rotations up to 360°
- **Scale Invariance**: Effective across multiple image scales
- **Pose Estimation**: Improved camera pose estimation on MegaDepth benchmark
- **Feature Matching**: Enhanced matching accuracy with various feature extractors

### Supported Feature Extractors

The model works seamlessly with:
- SIFT (with brute-force matcher)
- SuperPoint (with NN matcher)
- D2-Net
- R2D2
- DISK

---

## Citation

If you find this project useful in your research, please consider citing our paper:

```bibtex
@article{pan2025scale,
  title={Scale-aware co-visible region detection for image matching},
  author={Pan, Xu and Xia, Zimin and Zheng, Xianwei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={229},
  pages={122--137},
  year={2025},
  publisher={Elsevier}
}
```

---

## License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

---

## Acknowledgments

We acknowledge the following projects that were instrumental to our research:
- [MegaDepth](https://github.com/zhengqili/MegaDepth) for providing comprehensive depth data and image matching benchmarks.
- [OETR](https://github.com/TencentYoutuResearch/ImageMatching-OETR) for model initialization strategies.
- PyTorch team for the excellent framework.

---

## Contact

For questions or issues, please visit the [GitHub repository](https://github.com/SSSSphinx/SCoDe) or contact the authors.
