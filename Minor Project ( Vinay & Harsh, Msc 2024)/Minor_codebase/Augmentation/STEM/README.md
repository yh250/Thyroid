# STEMBalancer: A Hybrid Image Dataset Balancing Framework

STEMBalancer is a Python-based image dataset balancing tool that integrates SMOTE-ENN and Mixup augmentation strategies to reduce class imbalance in image classification tasks. This tool is particularly useful for medical imaging and other domains where minority class samples are limited and difficult to collect.

##  Key Features

- **Folder-based image ingestion** for each class label.
- **SMOTE-ENN algorithm** for synthetic oversampling followed by noise reduction.
- **Mixup augmentation** for further sample diversification.
- **Configurable parameters** for controlling augmentation behavior.
- **Intermediate and final results saving** to visualize and verify the augmentation impact.

---
## Requirements
- Python 3.x
- NumPy
- OpenCV
- Pandas
- scikit-learn

---

##  How It Works

The STEM pipeline is composed of the following steps:

1. **Data Loading**  
   Loads grayscale images from class-labeled folders, resizes them to a fixed size, and flattens for processing.

2. **Imbalance Ratio Calculation**  
   Computes the ratio between minority and majority class counts.

3. **SMOTE-ENN**  
   - **SMOTE**: Generates synthetic samples using k-nearest neighbors.
   - **ENN**: Removes noisy or borderline samples using neighborhood voting.

4. **Mixup Augmentation**  
   Randomly blends pairs of samples within the same class to create new training examples.

5. **Data Saving**  
   Writes the augmented images back to disk for inspection and future model training.

---

## 🛠️ Configuration

You can customize the balancing behavior via the `STEMBalancer` class initializer:

```python
STEMBalancer(
    target_ratio=0.7,         # Desired imbalance ratio
    k_neighbors=4,            # k-NN used in SMOTE
    max_iterations=30,        # Max SMOTE-ENN cycles
    min_improvement=5,        # Minimum sample improvement to continue
    image_size=(256, 256)     # Resize all images to this size
)
```
---
## Folder structure 
Each input folder should contain images for a specific class. The script expects all classes to be located in subfolders under the specified base_path.
```
project_root/
│
├── class_0/
│   ├── img001.png
│   ├── ...
│
├── class_1/
│   ├── img001.png
│   ├── ...
│
└── balanced_output/
    ├── after_smote_enn/
    ├── final_balanced/
    └── balance_summary.txt
```
---
## Output 
- Augmented image files in ```balanced_output/```

- Summary statistics in balance_summary.txt

- Print logs for:
   - Class distribution before/after each step
   - Iteration progress
   - Improvements in balancing ratio
---


