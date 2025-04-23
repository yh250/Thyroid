# STABLE DIFFUSION based image augmentation 

This Python-based image generation pipeline leverages **Stable Diffusion** to generate realistic grayscale **SPECT thyroid scans**. It is designed to augment small or imbalanced medical image datasets using class-specific prompts and conditioning images.

Useful for:
- Synthetic data generation for medical imaging tasks
- Balancing datasets for training classification models
- Exploring text-to-image synthesis with domain-specific conditioning

---

## Key Features

- **Text-to-Image Generation** using Stable Diffusion
-  **Conditioning Support** via input thyroid scan images
-  **Balanced Dataset Generation** from underrepresented classes
-  **Grayscale and Medical-Format Optimization** for clarity and detail
-  Supports original + SMOTE-generated images for prompt conditioning

---

##  How It Works

1. **Conditioning Image Loader**  
   Loads grayscale images from the original and SMOTE folders to use as diffusion conditions.

2. **Stable Diffusion Inference**  
   For each class label, generates synthetic scans based on class-specific prompts and optional conditioning images.

3. **Balanced Dataset Builder**  
   Ensures each class meets a target sample count by generating missing images in controlled batches.

---

##  Example Usage

```python
if __name__ == "__main__":
    generator = ThyroidImageGenerator()

    class_prompts = {
        "AFTN": "thyroid SPECT scan showing normal uptake pattern, medical imaging, grayscale, high detail",
        "graves": "thyroid SPECT scan showing focal adenoma, increased uptake, medical imaging, detailed grayscale",
        "MHG": "thyroid SPECT scan showing toxic multinodular goiter pattern, medical imaging, clear contrast",
        "thyroiditis": "thyroid SPECT scan showing reduced uptake pattern of thyroiditis, medical imaging, precise"
    }

    generator.generate_balanced_dataset(
        original_path="path/to/original",
        smote_path="path/to/smote",
        class_prompts=class_prompts,
        target_samples=150,
        output_format='.png'
    )
```
---
## Requirements
Install dependencies with:

```Bash 
pip install torch torchvision diffusers transformers opencv-python numpy pillow
```
Make sure to have a CUDA-capable GPU for performance. The script runs significantly slower on CPU.

Required Libraries
- torch, diffusers, transformers – for running Stable Diffusion

- Pillow, numpy, opencv-python – for image processing and I/O

- torchvision – for grayscale and resizing transformations


---
## Folder Structure
The directory layout should be:

```markdown
project_root/
│
├── original/
│   ├── graves/
│   ├── thyroiditis/
│   ├── MHG/
│   └── AFTN/
│
├── smote/
│   ├── graves/
│   ├── thyroiditis/
│   ├── MHG/
│   └── AFTN/
│
└── enhanced_dataset/
    ├── graves/
    ├── thyroiditis/
    ├── MHG/
    └── AFTN/
```
Each class folder should contain .png, .jpg, or other supported images for conditioning.


---
### ⚠️ Notes
- Images are converted to grayscale and resized to 256x256 by default.

- This tool disables safety checks in Stable Diffusion to enable medical image use.

- Make sure you have enough VRAM (≥6GB) to run Stable Diffusion with conditioning.
---
