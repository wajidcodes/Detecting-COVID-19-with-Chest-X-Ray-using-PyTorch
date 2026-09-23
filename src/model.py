import torch
import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=4, pretrained=True):
    """
    Creates a standard PyTorch ResNet-18 model for image classification.
    
    Args:
        num_classes (int): Number of classes to predict (default: 4)
        pretrained (bool): Whether to use ImageNet pre-trained weights
                           (highly recommended for faster training and better accuracy)
                           
    Returns:
        PyTorch ResNet-18 model ready for training or inference.
    """
    
    # 1. Load the ResNet-18 model
    if pretrained:
        # Load weights pre-trained on the ImageNet dataset
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        # Start from scratch (random weights)
        model = models.resnet18(weights=None)
        
    # 2. Replace the final classification layer (fully connected layer - 'fc')
    # ResNet-18 was originally designed to classify 1000 classes (ImageNet).
    # We need to change the final layer to output only 4 classes for our COVID-19 dataset.
    
    # Get the number of input features to the final layer (which is 512 for ResNet-18)
    in_features = model.fc.in_features
    
    # Replace it with a new Linear layer that outputs our desired number of classes
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    # Test the model creation
    print("Testing model creation...")
    model = get_resnet18()
    print("Model created successfully!")
    print(f"Final layer structure: {model.fc}")
