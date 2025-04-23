import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import pandas as pd
from pathlib import Path
import cv2

class STEMBalancer:
    """
    A class that implements the STEM balancing algorithm which combines 
    SMOTE-ENN and Mixup techniques to address class imbalance in image datasets.
    """

    def __init__(self, target_ratio=0.7, k_neighbors=5, max_iterations=30, min_improvement=5, image_size=(256, 256)):
        """
        Initialize the STEMBalancer with configuration parameters.

        Parameters:
        - target_ratio (float): Desired ratio of minority to majority class.
        - k_neighbors (int): Number of neighbors used in SMOTE.
        - max_iterations (int): Maximum iterations for SMOTE-ENN.
        - min_improvement (int): Minimum number of samples added per iteration to continue.
        - image_size (tuple): Image dimensions for resizing.
        """
        self.target_ratio = target_ratio
        self.k_neighbors = k_neighbors
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.image_size = image_size

    def load_data_from_folders(self, base_path):
        """
        Load and preprocess grayscale images from class-wise directories.

        Parameters:
        - base_path (str): Path to the root directory containing class folders.

        Returns:
        - X (np.ndarray): Flattened and normalized image data.
        - y (np.ndarray): Corresponding class labels.
        """
        X = []
        y = []
        print("Loading data from folders...")

        for class_idx, class_folder in enumerate(os.listdir(base_path)):
            folder_path = os.path.join(base_path, class_folder)
            if os.path.isdir(folder_path):
                print(f"Processing class {class_folder} (index: {class_idx})")
                folder_count = 0

                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        file_path = os.path.join(folder_path, file_name)
                        try:
                            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img_resized = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
                                img_flat = img_resized.flatten().astype(np.float32) / 255.0
                                X.append(img_flat)
                                y.append(class_idx)
                                folder_count += 1
                            else:
                                print(f"Failed to load image: {file_path}")
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")

                print(f"Loaded {folder_count} images from class {class_folder}")

        if not X:
            raise ValueError("No valid images found in the specified directories")

        X = np.array(X)
        y = np.array(y)

        print(f"\nTotal dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")

        return X, y

    def calculate_imbalance_ratio(self, y):
        """
        Calculate the class imbalance ratio (minority/majority).

        Parameters:
        - y (np.ndarray): Class labels.

        Returns:
        - ratio (float): Calculated imbalance ratio.
        """
        class_counts = Counter(y)
        min_class = min(class_counts.values())
        max_class = max(class_counts.values())
        return min_class / max_class if max_class > 0 else 0

    def apply_smote_enn(self, X, y):
        """
        Perform SMOTE followed by ENN to balance the dataset iteratively.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Class labels.

        Returns:
        - X_current (np.ndarray): Balanced feature matrix.
        - y_current (np.ndarray): Balanced labels.
        """
        class_counts = Counter(y)
        max_class_count = max(class_counts.values())
        X_current, y_current = X.copy(), y.copy()

        iteration = 0
        prev_total_samples = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}/{self.max_iterations}")
            print(f"Current class distribution: {Counter(y_current)}")
            print(f"Current imbalance ratio: {self.calculate_imbalance_ratio(y_current):.3f}")

            X_new, y_new = [], []
            samples_added = 0

            for class_label in class_counts:
                if class_counts[class_label] < max_class_count:
                    class_samples = X_current[y_current == class_label]
                    if len(class_samples) < 2:
                        continue

                    n_samples_needed = max_class_count - len(class_samples)

                    nn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(class_samples)))
                    nn.fit(class_samples)

                    for _ in range(n_samples_needed):
                        idx = np.random.randint(0, len(class_samples))
                        sample = class_samples[idx]

                        distances, indices = nn.kneighbors([sample])
                        neighbor_idx = np.random.choice(indices[0][1:])
                        neighbor = class_samples[neighbor_idx]

                        delta = np.random.random()
                        synthetic_sample = sample + delta * (neighbor - sample)

                        X_new.append(synthetic_sample)
                        y_new.append(class_label)
                        samples_added += 1

            if samples_added == 0:
                print("No new samples generated. Breaking SMOTE loop.")
                break

            X_current = np.vstack([X_current] + X_new)
            y_current = np.append(y_current, y_new)

            # ENN: Remove ambiguous samples
            nn = NearestNeighbors(n_neighbors=4)
            nn.fit(X_current)
            distances, indices = nn.kneighbors(X_current)

            keep_indices = []
            for idx in range(len(X_current)):
                neighbor_labels = y_current[indices[idx][1:]]
                if sum(neighbor_labels == y_current[idx]) >= 2:
                    keep_indices.append(idx)

            X_current = X_current[keep_indices]
            y_current = y_current[keep_indices]

            current_total = len(X_current)
            samples_improvement = current_total - prev_total_samples

            print(f"Samples added in this iteration: {samples_added}")
            print(f"Samples after ENN: {current_total}")
            print(f"Net improvement: {samples_improvement}")

            if samples_improvement < self.min_improvement:
                print(f"Insufficient improvement ({samples_improvement} < {self.min_improvement}). Breaking loop.")
                break

            prev_total_samples = current_total

            current_ratio = self.calculate_imbalance_ratio(y_current)
            if current_ratio >= self.target_ratio:
                print(f"Target ratio {self.target_ratio} achieved. Current ratio: {current_ratio:.3f}")
                break

        return X_current, y_current

    def apply_mixup(self, X, y):
        """
        Apply Mixup augmentation to expand the dataset.

        Parameters:
        - X (np.ndarray): Input features.
        - y (np.ndarray): Class labels.

        Returns:
        - X_mixed (np.ndarray): Augmented feature matrix.
        - y_mixed (np.ndarray): Augmented labels.
        """
        print("\nApplying Mixup augmentation...")
        X_mixed, y_mixed = X.copy(), y.copy()

        for class_label in np.unique(y):
            class_indices = np.where(y == class_label)[0]
            n_samples = len(class_indices)

            for _ in range(n_samples // 2):
                idx1, idx2 = np.random.choice(class_indices, 2, replace=False)
                lambda_mix = np.random.beta(0.4, 0.4)
                mixed_sample = lambda_mix * X[idx1] + (1 - lambda_mix) * X[idx2]

                X_mixed = np.vstack([X_mixed, mixed_sample])
                y_mixed = np.append(y_mixed, class_label)

        print(f"Final class distribution after Mixup: {Counter(y_mixed)}")
        print(f"Final imbalance ratio: {self.calculate_imbalance_ratio(y_mixed):.3f}")
        return X_mixed, y_mixed

    def save_balanced_data(self, X, y, output_path, stage="final"):
        """
        Save balanced images to class-wise folders.

        Parameters:
        - X (np.ndarray): Balanced image data.
        - y (np.ndarray): Corresponding labels.
        - output_path (str): Directory to save the balanced dataset.
        - stage (str): Label for the augmentation stage (e.g., "smote_enn", "final").
        """
        os.makedirs(output_path, exist_ok=True)

        unique_classes = np.unique(y)
        for class_label in unique_classes:
            class_folder = os.path.join(output_path, f"class_{class_label}")
            os.makedirs(class_folder, exist_ok=True)

            class_indices = np.where(y == class_label)[0]

            for idx, sample_idx in enumerate(class_indices):
                img = X[sample_idx].reshape(self.image_size)
                img = (img * 255).astype(np.uint8)
                img_path = os.path.join(class_folder, f"{stage}_sample_{idx}.png")
                cv2.imwrite(img_path, img)

        print(f"\nSaved {stage} balanced dataset to: {output_path}")
        print(f"Class distribution in saved data: {Counter(y)}")

    def fit_resample(self, base_path, output_base_path="balanced_output"):
        """
        Apply the complete STEM process: load, balance (SMOTE-ENN + Mixup), and save.

        Parameters:
        - base_path (str): Input directory containing class-wise folders.
        - output_base_path (str): Directory name for saving outputs.

        Returns:
        - X_final (np.ndarray): Final balanced features.
        - y_final (np.ndarray): Final balanced labels.
        """
        X, y = self.load_data_from_folders(base_path)
        print("\nInitial class distribution:", Counter(y))
        print(f"Initial imbalance ratio: {self.calculate_imbalance_ratio(y):.3f}")
        print("Initial dataset size:", len(X))

        output_base_path = os.path.join(base_path, output_base_path)
        smote_enn_path = os.path.join(output_base_path, "after_smote_enn")
        final_path = os.path.join(output_base_path, "final_balanced")

        print("\nApplying SMOTE-ENN...")
        X_balanced, y_balanced = self.apply_smote_enn(X, y)
        self.save_balanced_data(X_balanced, y_balanced, smote_enn_path, "smote_enn")

        print("\nApplying Mixup...")
        X_final, y_final = self.apply_mixup(X_balanced, y_balanced)
        self.save_balanced_data(X_final, y_final, final_path, "final")

        with open(os.path.join(output_base_path, "balance_summary.txt"), "w") as f:
            f.write("STEM Balancing Summary\n")
            f.write("=====================\n\n")
            f.write(f"Initial distribution: {Counter(y)}\n")
            f.write(f"After SMOTE-ENN: {Counter(y_balanced)}\n")
            f.write(f"Final distribution: {Counter(y_final)}\n\n")
            f.write(f"Initial imbalance ratio: {self.calculate_imbalance_ratio(y):.3f}\n")
            f.write(f"Final imbalance ratio: {self.calculate_imbalance_ratio(y_final):.3f}\n")

        return X_final, y_final


if __name__ == "__main__":
    # Initialize the balancer with desired parameters
    balancer = STEMBalancer(
        target_ratio=0.7,
        k_neighbors=4,
        max_iterations=30,
        min_improvement=5,
        image_size=(256,256)
    )

    # Path to dataset
    data_folder = r"/content/drive/MyDrive/Thyroid_Spect_minor /DU_original"

    try:
        # Perform dataset balancing using STEM
        X_balanced, y_balanced = balancer.fit_resample(data_folder)
        print("Successfully completed STEM balancing")
    except Exception as e:
        print(f"An error occurred: {e}")
