# 🐟 Vision Transformer for Out-of-Water Fish Classification

Fine-tuned a pre-trained Vision Transformer (ViT) model to classify fish species from out-of-water images, achieving **82.5% accuracy** — outperforming a ResNet50 baseline (75%) on the same dataset.

---

## Overview

Most existing fish classification models are trained on underwater images, which struggle with the different lighting and environmental conditions present when fish are caught out of water. This project addresses that gap by building a deep learning model specifically for out-of-water fish classification.

---

## Results

| Model               | Accuracy  | Precision | Recall    | F1-Score  |
| ------------------- | --------- | --------- | --------- | --------- |
| **ViT (ours)**      | **82.5%** | **73.8%** | **70.5%** | **71.5%** |
| ResNet50 (baseline) | 75.0%     | 68.5%     | 65.2%     | 66.7%     |

- Training and validation loss decreased consistently over 5 epochs, indicating effective optimization with no overfitting
- Ablation studies showed that removing data augmentation dropped accuracy by ~5%
- Fine-tuning the ViT backbone improved accuracy by ~2% over frozen weights

---

## Approach

### Dataset

- **Source:** [Affine dataset on Kaggle](https://www.kaggle.com/datasets/jorritvenema/affine) — out-of-water fish images across 30 species
- Train / Validation / Test split: 80 / 10 / 10

### Preprocessing & Augmentation

- Random rotation (±30°)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Random resized crop to 224×224
- Normalization with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Model

- **Base model:** `google/vit-base-patch16-224-in21k` from Hugging Face Model Hub
- Replaced classification head for 30-class output
- Fine-tuned all layers

### Training

- **Optimizer:** AdamW (lr=5e-5)
- **Loss:** Label smoothing cross-entropy
- **Batch size:** 8 (hardware constrained)
- **Epochs:** 5

---

## Model Selection

Initially experimented with YOLOv7 for object detection, but found it struggled with the fine-grained classification needed to distinguish between fish species. Switched to ViT for its superior ability to capture spatial relationships and fine-grained patterns across image patches.

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers & Datasets
- scikit-learn
- Matplotlib
- KaggleHub

---

## How to Run

1. Clone this repo
2. Install dependencies:

```bash
pip install torch torchvision transformers datasets kagglehub scikit-learn matplotlib pillow
```

3. Open `Fish_Classification_Project.ipynb` in Jupyter or Google Colab
4. Run all cells — the dataset will be automatically downloaded via KaggleHub

---

## Limitations & Future Work

- Dataset is limited to bright outdoor conditions — low-light generalization is weak
- Some fish species are underrepresented, leading to higher misclassification for rare species
- Real-time deployment (e.g., mobile/wearable) would require model quantization
- Future work: custom ViT architecture, Bayesian hyperparameter optimization, multi-GPU training

---

## Authors

- **Aarshi Gala** — Dataset preprocessing, ViT model architecture, fine-tuning, hyperparameter optimization, ablation studies, results analysis
- **Swadhi Sathish Kumar** — Dataset preprocessing, YOLOv7 experimentation, presentation

---

## References

- Dosovitskiy et al. ["An Image Is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) (2020)
- Hugging Face [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k)
- Venema, Jorrit. [Affine Dataset](https://www.kaggle.com/datasets/jorritvenema/affine), Kaggle (2022)
