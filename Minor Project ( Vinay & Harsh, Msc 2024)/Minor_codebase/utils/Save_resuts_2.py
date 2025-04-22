"""
Create the Result folder only. NO LOGING  
"""
import os
from datetime import datetime


def create_results_folder(base_dir="results", model_name="model", augmentation_type="none"):
    """
    Creates a results folder with a name based on the current date, time, model name, and augmentation type.

    Parameters:
    - base_dir: str, base directory to store results.
    - model_name: str, name of the model being used.
    - augmentation_type: str, type of augmentation applied.

    Returns:
    - folder_path: str, the path to the created folder.
    """
    # Get the current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Sanitize model and augmentation names for file system compatibility
    model_name = model_name.replace(" ", "_").lower()
    augmentation_type = augmentation_type.replace(" ", "_").lower()

    # Create the folder name
    folder_name = f"{timestamp}_{model_name}_{augmentation_type}"
    folder_path = os.path.join(base_dir, folder_name)

    # Create the directory
    os.makedirs(folder_path, exist_ok=True)

    return folder_path


# Example usage
if __name__ == "__main__":
    results_folder = create_results_folder(model_name="ResNet50", augmentation_type="SMOTE")
    print(f"Results folder created: {results_folder}")
