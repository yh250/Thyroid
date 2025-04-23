# GANs: Generative Adversarial Networks for Thyroid Image Augmentation

This folder contains two GAN architectures used for data augmentation on a thyroid SPECT scan dataset: **DCGAN** and **ACGAN**. These models help generate synthetic images to combat data imbalance.

## 📁 Files

| File | Description |
|------|-------------|
| [`DCGAN.ipynb`][1] | Deep Convolutional GAN to generate synthetic thyroid SPECT scan images without label conditioning. |
| [`ACGAN.ipynb`][2] | Auxiliary Classifier GAN that generates class-conditioned synthetic images, enabling targeted augmentation for underrepresented classes. |

---

##  Objective

This module focuses on synthetically increasing the size of the dataset using GANs to aid in balanced classification. Both models are used to generate thyroid SPECT scans, helping improve generalization and reduce overfitting in the downstream classifier.

---

##  Architecture Overview

###  DCGAN

- Input: Random noise vector (latent space).
- Generator: Deep convolutional layers with batch normalization and ReLU activations.
- Discriminator: Convolutional network with LeakyReLU, predicts real vs fake.
- Output: Generated grayscale 64x64 images.

### ACGAN

- Input: Random noise + Class label embedding.
- Generator: Takes both latent noise and class embeddings to generate labeled images.
- Discriminator: Outputs both real/fake score and predicted class label.
- Dual loss:
  - Real/fake discrimination.
  - Auxiliary classification loss for correct label generation.

---
---

## 📦Requirements

Ensure the following libraries are installed before running the notebooks:
- ```numpy```
- ```matplotlib```
- ```pandas```
- ```scikit-learn```
- ```tensorflow keras```

```bash
pip install numpy matplotlib pandas scikit-learn tensorflow keras
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yh250/Thyroid
   cd Thyroid/Minor\ Project\ \(Vinay\ \&\ Harsh,\ Msc\ 2024\)/Minor_codebase/Augmentation/GANs
   ```
2. Open either of the notebooks in Jupyter:

```bash

jupyter notebook DCGAN.ipynb
# or
jupyter notebook ACGAN.ipynb
```
3. Run all cells sequentially after ensuring dataset paths are correct.

---
[1]: https://github.com/yh250/Thyroid/blob/c55f8f8a3be1734681f94657dd0fa3e4159f02e0/Minor%20Project%20(%20Vinay%20%26%20Harsh%2C%20Msc%202024)/Minor_codebase/Augmentation/GANs/DCGAN.ipynb
[2]: https://github.com/yh250/Thyroid/blob/c55f8f8a3be1734681f94657dd0fa3e4159f02e0/Minor%20Project%20(%20Vinay%20%26%20Harsh%2C%20Msc%202024)/Minor_codebase/Augmentation/GANs/ACGAN.ipynb
