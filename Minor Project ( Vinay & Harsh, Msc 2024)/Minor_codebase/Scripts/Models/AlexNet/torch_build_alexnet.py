import torch
import torch.nn as nn
import torchvision.models as models

def build_alexnet(num_classes, pretrained=True):
    """
    Builds an AlexNet model using torchvision, adapted for the given number of classes.

    Parameters:
    - num_classes: int, the number of output classes.
    - pretrained: bool, whether to use a pretrained AlexNet model.

    Returns:
    - model: torch.nn.Module, the customized AlexNet model.
    """
    # Load the AlexNet model
    model = models.alexnet(pretrained=pretrained)

    # Adjust the first convolutional layer for grayscale images
    model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

    # Adjust the final classifier for the number of classes
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)

    return model
