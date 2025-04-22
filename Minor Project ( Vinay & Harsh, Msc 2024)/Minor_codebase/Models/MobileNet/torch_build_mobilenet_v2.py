import torch
import torch.nn as nn
import torchvision.models as models

def build_mobilenetv2(num_classes, pretrained=True):
    """
    Builds a MobileNetV2 model using torchvision, adapted for the given number of classes.

    Parameters:
    - num_classes: int, the number of output classes.
    - pretrained: bool, whether to use a pretrained MobileNetV2 model.

    Returns:
    - model: torch.nn.Module, the customized MobileNetV2 model.
    """
    # Load the MobileNetV2 model
    model = models.mobilenet_v2(pretrained=pretrained)

    # Adjust the first convolutional layer for grayscale images
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    # Adjust the final classifier for the number of classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    return model