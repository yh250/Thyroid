import torch
import torch.nn as nn
import torchvision.models as models

def build_resnet18(num_classes, pretrained=False):
    """
    Builds a ResNet-18 model using torchvision, adapted for the given number of classes.

    Parameters:
    - num_classes: int, the number of output classes.
    - pretrained: bool, whether to use a pretrained ResNet-18 model.

    Returns:
    - model: torch.nn.Module, the customized ResNet-18 model.
    """
    # Load the ResNet-18 model
    model = models.resnet18(pretrained=pretrained)

    # Adjust the input layer if needed (e.g., grayscale images)
    # Replace the first convolutional layer for single-channel (grayscale) input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Adjust the fully connected layer for the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model
