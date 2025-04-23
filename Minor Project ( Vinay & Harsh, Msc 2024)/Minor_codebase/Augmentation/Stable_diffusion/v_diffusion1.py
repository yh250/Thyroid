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
    """
    A class for generating synthetic thyroid images using the Stable Diffusion model.
    Provides functionality to preprocess conditioning images, generate synthetic images,
    and create balanced datasets for machine learning applications.
    """

    def __init__(self, model_id="CompVis/stable-diffusion-v1-4"):
        """
        Initialize the ThyroidImageGenerator class with the specified Stable Diffusion model.

        Args:
            model_id (str): The model ID for the Stable Diffusion pipeline.
        """
        # Load the Stable Diffusion pipeline and scheduler
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None  # Disable safety checks for medical image generation
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        # Move the pipeline to GPU if available
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")

        # Define preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # Convert images to grayscale
            transforms.Resize((256, 256)),  # Resize images to a fixed size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])

        # Define supported image formats for input and output
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    def load_conditioning_images(self, paths: List[Path]) -> List[torch.Tensor]:
        """
        Load and preprocess images from multiple directories.

        Args:
            paths (List[Path]): List of directories containing conditioning images.

        Returns:
            List[torch.Tensor]: List of processed conditioning images.
        """
        images = []
        for path in paths:
            if path.exists():
                # Search for images in all supported formats
                for format in self.supported_formats:
                    for img_path in path.glob(f"*{format}"):
                        try:
                            images.append(self.condition_image(img_path))
                        except Exception as e:
                            print(f"Failed to load {img_path}: {e}")
        return images

    def condition_image(self, image_path: Path) -> torch.Tensor:
        """
        Preprocess an input image for conditioning.

        Args:
            image_path (Path): Path to the input image.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        image = Image.open(image_path)
        # Convert RGBA images to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        # Apply transformations
        image = self.transform(image)
        return image

    def generate_images(self, prompt: str, num_images: int = 5, guidance_scale: float = 7.5,
                        num_inference_steps: int = 50, conditioning_image: torch.Tensor = None) -> List[Image.Image]:
        """
        Generate synthetic images using Stable Diffusion.

        Args:
            prompt (str): Text prompt for image generation.
            num_images (int): Number of images to generate.
            guidance_scale (float): Scale for classifier-free guidance.
            num_inference_steps (int): Number of inference steps for the generation process.
            conditioning_image (torch.Tensor): Optional image for conditioning.

        Returns:
            List[Image.Image]: List of generated images.
        """
        images = []
        for i in range(num_images):
            if conditioning_image is not None:
                # Generate an image based on a conditioning image
                image = self.pipe(
                    prompt=prompt,
                    image=conditioning_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            else:
                # Generate an image without conditioning
                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]

            # Convert the generated image to grayscale
            image = image.convert('L')
            images.append(image)
        return images

    def generate_balanced_dataset(self, original_path: str, smote_path: str, class_prompts: Dict[str, str],
                                  target_samples: int = 150, output_dir: str = "enhanced_dataset",
                                  output_format: str = '.png'):
        """
        Generate a balanced dataset using original and SMOTE-generated images.

        Args:
            original_path (str): Path to the directory containing original images.
            smote_path (str): Path to the directory containing SMOTE-generated images.
            class_prompts (Dict[str, str]): Mapping of class names to text prompts for generation.
            target_samples (int): Target number of samples per class.
            output_dir (str): Directory to save the generated dataset.
            output_format (str): Format for saving the generated images (e.g., '.png', '.jpg').
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

            # Define paths for original and SMOTE-generated images
            original_class_dir = Path(original_path) / class_name
            smote_class_dir = Path(smote_path) / class_name

            # Load conditioning images from both directories
            cond_images = self.load_conditioning_images([original_class_dir, smote_class_dir])

            if not cond_images:
                print(f"Warning: No conditioning images found for class {class_name}")
                continue

            # Count existing images across all supported formats
            existing_count = 0
            for format in self.supported_formats:
                existing_count += len(list(original_class_dir.glob(f"*{format}")))
            print(f"Found {existing_count} original images")

            # Calculate the number of images to generate
            to_generate = max(0, target_samples - existing_count)
            print(f"Will generate {to_generate} new images")

            # Generate images in batches
            batch_size = 10
            for batch_start in range(0, to_generate, batch_size):
                batch_count = min(batch_size, to_generate - batch_start)
                # Randomly select a conditioning image
                cond_img = random.choice(cond_images)

                # Generate a batch of images
                images = self.generate_images(
                    prompt=prompt,
                    num_images=batch_count,
                    conditioning_image=cond_img,
                    guidance_scale=7.0,
                    num_inference_steps=40
                )

                # Save generated images
                for idx, img in enumerate(images):
                    image_number = batch_start + idx
                    output_file = class_dir / f"{class_name}_gen_{image_number}{output_format}"
                    img.save(output_file)

                print(f"Generated batch {batch_start//batch_size + 1}")

# Example usage
if __name__ == "__main__":
    generator = ThyroidImageGenerator()

    # Define class-specific prompts for image generation
    class_prompts = {
        "AFTN": "thyroid SPECT scan showing normal uptake pattern, medical imaging, grayscale, high detail",
        "graves": "thyroid SPECT scan showing focal adenoma, increased uptake, medical imaging, detailed grayscale",
        "MHG": "thyroid SPECT scan showing toxic multinodular goiter pattern, medical imaging, clear contrast",
        "thyroiditis": "thyroid SPECT scan showing reduced uptake pattern of thyroiditis, medical imaging, precise"
    }

    # Generate a balanced dataset
    generator.generate_balanced_dataset(
        original_path=r"C:\Users\PC\Downloads\og",
        smote_path=r"C:\Users\PC\Downloads\sm",
        class_prompts=class_prompts,
        target_samples=150,
        output_format='.png'  # You can change this to '.jpg', '.tiff', etc.
    )
