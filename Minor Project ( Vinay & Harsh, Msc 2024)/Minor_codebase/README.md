# Minor Project Codebase - MSc 2024
- This directory contains all essential code components for our MSc Minor Project on thyroid disorder classification using SPECT scan images. The project implements multiple deep learning models and advanced data augmentation techniques to address class imbalance and improve classification performance.
---
## 📁 Directory Structure

```bash
Minor_codebase/
│
├── notebooks/           # Jupyter Notebooks for model development and evaluation
├── scripts/             # Python script versions of the models for modular usage
├── augmentation/        # Data augmentation techniques including SMOTE, GANs, etc.
├── requirements.txt     # Required dependencies for running the project
```
---

### 📘 `notebooks/`
This folder contains interactive Jupyter notebooks used during the experimentation and development phase. Each notebook corresponds to different model architectures or augmentation strategies, and can be run step-by-step for reproducibility and analysis.

### ⚙️ `scripts/`
This folder holds Python scripts (.py files) that modularize the models and their training logic. These scripts are meant for direct execution and integration into larger pipelines.

###  `augmentation/`
All the augmentation techniques explored in this project are present here. It includes:
- Basic image augmentation
- Synthetic data generation using STEM
- GAN-based data augmentation
- Diffusion based data augmentation
  
NOTE: Basic augmentation is present in ```Data_loader.py```, located at ```/scripts/utils```

### 📦 `requirements.txt`
Contains a list of dependencies and libraries required to run the notebooks and scripts. Install them using:

```bash
pip install -r requirements.txt
