import os
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf


def create_results_folder(base_dir="/mnt/f/Reena/Temp_Minor_project/Temp_Minor_project/Results", model_name="model", augmentation_type="none"):
    """
    Creates a results folder with a name based on the current date and time.

    Parameters:
    - base_dir: str, base directory to store results.

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

    # Create the folder path
    folder_path = os.path.join(base_dir, folder_name)

    # Create the directory
    os.makedirs(folder_path, exist_ok=True)

    return folder_path


def save_training_logs(folder_path, logs):
    """
    Saves training logs to a text file.

    Parameters:
    - folder_path: str, path to the results folder.
    - logs: dict, training logs to save.
    """
    log_path = os.path.join(folder_path, "training_logs.txt")
    with open(log_path, "w") as log_file:
        for key, value in logs.items():
            log_file.write(f"{key}: {value}\n")
    print(f"Logs saved at {log_path}")


def save_loss_curve(folder_path, history):
    """
    Saves a plot of the training and validation loss.

    Parameters:
    - folder_path: str, path to the results folder.
    - history: keras.callbacks.History, history object from model training.
    """
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plot_path = os.path.join(folder_path, "loss_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curve saved at {plot_path}")


def save_model_checkpoint(folder_path, model):
    """
    Saves the trained model to the results folder.

    Parameters:
    - folder_path: str, path to the results folder.
    - model: keras.Model, the trained model to save.
    """
    checkpoint_path = os.path.join(folder_path, "model_checkpoint.h5")
    model.save(checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")
if __name__ == "__main__":
    results_folder = create_results_folder()
    print(f"Results folder created: {results_folder}")

