# Data Augmentation and Balancing of a Highly Imbalanced SPECT Scan Dataset

*Minor Project Submission by Harsh Yadav, Vinay Dagar (M.Sc. 2025)*  
*Department of Computer Science, University of Delhi*

---

## Project Summary

**Aim:**  
To address class imbalance in small-scale medical imaging datasets and develop a more generalizable diagnostic model for thyroid disorders.

**Problem Statement:**  
Medical imaging datasets are often limited and imbalanced due to the high cost and effort required for annotation.  
This class imbalance negatively affects the performance of deep learning models, leading to biased predictions and reduced diagnostic reliability.

**Overview:**  
This project focuses on the classification of hyperthyroidism using SPECT scan images.  
The dataset inherently suffers from severe class imbalance, a common trait in medical datasets.  
Our objective was to explore and compare various strategies for balancing and augmenting this data to improve model generalization and robustness.

---

##  Our Approach

### Data Augmentation Techniques:
- Traditional methods (rotation, scaling, flipping)  
- Advanced hybrid method: **STEM** (SMOTE + ENN + MixUp)  
- GAN-based synthetic image generation  
- Diffusion-based augmentation techniques  

### Model Evaluation:
- Baseline: Custom 3-layer CNN  
- Pretrained architectures: **AlexNet**, **MobileNetV2**  
- Transfer learning using **ResNet50V2**

---

A comprehensive comparative analysis was conducted to evaluate the effectiveness of each augmentation method and model architecture in addressing class imbalance and improving diagnostic accuracy.
