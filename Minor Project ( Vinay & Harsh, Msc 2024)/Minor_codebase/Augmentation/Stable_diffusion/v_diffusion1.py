
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Dict, List
import random
import transformers

class ThyroidImageGenerator:
    def __init__(self, model_id="CompVis/stable-diffusion-v1-4"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Define supported image formats
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    def load_conditioning_images(self, paths):
        """Load and prepare images from multiple source directories."""
        images = []
        for path in paths:
            if path.exists():
                # Look for all supported image formats
                for format in self.supported_formats:
                    for img_path in path.glob(f"*{format}"):
                        try:
                            images.append(self.condition_image(img_path))
                        except Exception as e:
                            print(f"Failed to load {img_path}: {e}")
        return images

    def condition_image(self, image_path):
        """Prepare an input image for conditioning."""
        image = Image.open(image_path)
        # Convert to RGB if image is in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = self.transform(image)
        return image

    def generate_images(self, prompt, num_images=5, guidance_scale=7.5,
                       num_inference_steps=50, conditioning_image=None):
        images = []
        for i in range(num_images):
            if conditioning_image is not None:
                image = self.pipe(
                    prompt=prompt,
                    image=conditioning_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            else:
                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]

            image = image.convert('L')
            images.append(image)
        return images

    def generate_balanced_dataset(self, original_path, smote_path, class_prompts,
                                target_samples=150, output_dir="enhanced_dataset",
                                output_format='.png'):
        """
        Generate a balanced dataset using both original and SMOTE-generated images.

        Args:
            output_format (str): The format to save generated images ('.png', '.jpg', etc.)
        """
        # Ensure output_format starts with a dot
        if not output_format.startswith('.'):
            output_format = '.' + output_format

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        for class_name, prompt in class_prompts.items():
            print(f"\nProcessing class: {class_name}")

            class_dir = output_path / class_name
            class_dir.mkdir(exist_ok=True)

            original_class_dir = Path(original_path) / class_name
            smote_class_dir = Path(smote_path) / class_name

            cond_images = self.load_conditioning_images([original_class_dir, smote_class_dir])

            if not cond_images:
                print(f"Warning: No conditioning images found for class {class_name}")
                continue

            # Count existing images across all formats
            existing_count = 0
            for format in self.supported_formats:
                existing_count += len(list(original_class_dir.glob(f"*{format}")))
            print(f"Found {existing_count} original images")

            to_generate = max(0, target_samples - existing_count)
            print(f"Will generate {to_generate} new images")

            batch_size = 10
            for batch_start in range(0, to_generate, batch_size):
                batch_count = min(batch_size, to_generate - batch_start)
                cond_img = random.choice(cond_images)

                images = self.generate_images(
                    prompt=prompt,
                    num_images=batch_count,
                    conditioning_image=cond_img,
                    guidance_scale=7.0,
                    num_inference_steps=40
                )

                for idx, img in enumerate(images):
                    image_number = batch_start + idx
                    output_file = class_dir / f"{class_name}_gen_{image_number}{output_format}"
                    img.save(output_file)

                print(f"Generated batch {batch_start//batch_size + 1}")

# Example usage
if __name__ == "__main__":
    generator = ThyroidImageGenerator()

    class_prompts = {
        "AFTN": "thyroid SPECT scan showing normal uptake pattern, medical imaging, grayscale, high detail",
        "graves": "thyroid SPECT scan showing focal adenoma, increased uptake, medical imaging, detailed grayscale",
        "MHG": "thyroid SPECT scan showing toxic multinodular goiter pattern, medical imaging, clear contrast",
        "thyroiditis": "thyroid SPECT scan showing reduced uptake pattern of thyroiditis, medical imaging, precise"
    }

    # Generate dataset
    generator.generate_balanced_dataset(
        original_path=r"C:\Users\PC\Downloads\og",
        smote_path=r"C:\Users\PC\Downloads\sm",
        class_prompts=class_prompts,
        target_samples=150,
        output_format='.png'  # You can change this to '.jpg', '.tiff', etc.
    )