# Multimodal Fusion for Cancer Diagnosis and Prognosis (TGCA)

This repository contains an advanced framework for integrating histology images, graph-based cellular structures, and genomic features for cancer diagnosis and prognosis. This project builds upon the **Pathomic Fusion** framework and introduces several modern enhancements tailored for robust multimodal analysis.

## Key Features & Modernizations

- **Vision Foundation Models**: Integration of the **UNI** model (from Mahmood Lab) as a powerful histology encoder for extracting high-dimensional features from Whole Slide Images (WSIs).
- **Genos Transformer**: A transformer-based architecture for encoding tabular genomic data, allowing for complex relationship modeling within -omic features.
- **Advanced Fusion Strategies**:
    - **Gated Tensor Fusion**: Inspired by the original Pathomic Fusion for efficient modality gating.
    - **Cross-Attention Fusion**: Multi-head attention mechanism to dynamically model interactions between pathology and genomic embeddings.
- **Disentangled Representation Learning**: Support for learning shared latent features (common across modalities) and specific features (unique to each modality) to improve interpretability and performance.
- **Modernized GNNs**: Robust Graph Neural Networks for cellular topology analysis, featuring **SAGPooling** and support for **SAGE**, **GAT**, and **GraphConv** layers.
- **Feature Extraction Support**: Ability to use pre-extracted features (VGG, UNI, etc.) to significantly speed up training and reduce GPU memory overhead.

## Setup

### Prerequisites
- Python 3.8+
- PyTorch >= 1.10
- PyTorch Geometric
- `timm` (for foundation models)
- `huggingface_hub`
- NVIDIA GPU with CUDA support

### Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install timm huggingface_hub
pip install -r requirements.txt
```

## Code Base Structure

- **train_cv.py**: Cross-validation script for training.
- **networks.py**: Contains PyTorch model definitions (including UNI, GenosTransformer, and GraphNet).
- **fusion.py**: Implementation of various fusion strategies (Bilinear, Trilinear, Cross-Attention, Concat).
- **options.py**: Configuration and command-line arguments.
- **data_loaders.py**: Multi-modal dataset loaders.
- **utils.py**: Utility functions for survival loss, evaluation, and visualization.

## Training Examples

### Using UNI Foundation Model and Cross-Attention
To train a survival model using the UNI histology encoder and Cross-Attention fusion:
```bash
python train_cv.py --mode AC --use_uni 1 --fusion_type crossattn --task surv --exp_name uni_crossattn_exp
```

### Using Genos Transformer for Genomic Data
To train using a transformer-based genomic encoder:
```bash
python train_cv.py --mode C --use_transformer 1 --task surv --exp_name transformer_omic_exp
```

### Disentangled Representation Learning
To enable explicit latent disentanglement:
```bash
python train_cv.py --mode AC --use_disentanglement 1 --task surv --exp_name disentangle_exp
```

## Acknowledgements
This project is an extension of the work by [Chen et al. (Pathomic Fusion)](https://github.com/mahmoodlab/PathomicFusion). Special thanks to the Mahmood Lab for providing the UNI foundation model.

---
© **Muhammad Nouman Saleem** - MSc Thesis Project
