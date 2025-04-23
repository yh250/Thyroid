"""
Trainer function for the alternate PYTORCH framework 
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class PyTorchModelTrainer:
    def __init__(self, model, device, criterion, optimizer):
        """
        Initializes the trainer with model, device, loss function, and optimizer.
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def fit(self, train_loader, val_loader, epochs):
        """
        Trains the model with the provided data loaders.

        Parameters:
        - train_loader: DataLoader for training data.
        - val_loader: DataLoader for validation data.
        - epochs: Number of training epochs.

        Returns:
        - history: Dictionary containing training and validation loss and accuracy.
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Training phase
            self.model.train()
            train_loss, correct, total = 0.0, 0, 0
            for images, labels in tqdm(train_loader, desc="Training"):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total
            self.history["train_loss"].append(train_loss / len(train_loader))
            self.history["train_acc"].append(train_acc)

            # Validation phase
            self.model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation"):
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    # Metrics
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = 100 * correct / total
            self.history["val_loss"].append(val_loss / len(val_loader))
            self.history["val_acc"].append(val_acc)

            print(f"Train Loss: {self.history['train_loss'][-1]:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {self.history['val_loss'][-1]:.4f}, Val Acc: {val_acc:.2f}%")

        return self.history
