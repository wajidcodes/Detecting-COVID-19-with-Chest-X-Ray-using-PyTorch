import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Define our 4 target classes
CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

def get_transforms():
    """
    Returns data transformations for training and validation/testing.
    We resize images to 224x224 (standard for ResNet) and convert them to PyTorch tensors.
    """
    # Training transforms include data augmentation to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),         # Resize image for ResNet-18
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images left/right
        transforms.RandomRotation(degrees=10),   # Randomly rotate slightly
        transforms.ToTensor(),                   # Convert PIL image to PyTorch tensor
        transforms.Normalize(                    # Standard ImageNet normalization
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Validation transforms only resize and normalize (no random flipping/rotation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return train_transform, val_transform

class CovidXRayDataset(Dataset):
    """
    PyTorch Dataset class for loading COVID-19 Chest X-Ray images.
    """
    def __init__(self, data_dir, csv_file=None, split='train', transform=None):
        """
        Args:
            data_dir (str): Path to the data folder containing class subfolders.
            csv_file (str, optional): Path to splits.csv. If None, it reads directly from folders.
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Load from splits.csv if provided
        if csv_file and os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            # Filter for the requested split (train, val, test)
            df_split = df[df['split'] == split]
            
            for _, row in df_split.iterrows():
                # Get the relative path and full path
                rel_path = row['image_path']
                # Sometimes image_path in csv might contain 'data/' prefix, handle it
                if rel_path.startswith('data/'):
                    rel_path = rel_path[5:]
                    
                full_path = os.path.join(data_dir, rel_path)
                class_id = int(row['class_id'])
                
                if os.path.exists(full_path):
                    self.samples.append((full_path, class_id))
                    
        else:
            # Fallback: Just read directly from folders (good for Kaggle/Colab simplicity)
            print(f"Warning: No valid CSV found. Reading all images from folders directly. Treating as '{split}' data.")
            for class_name in CLASSES:
                class_id = CLASS_TO_IDX[class_name]
                # Try standard folder, or with single quotes like 'Viral Pneumonia'
                class_dir1 = os.path.join(data_dir, class_name, 'images')
                class_dir2 = os.path.join(data_dir, f"'{class_name}'", 'images')
                
                class_dir = class_dir1 if os.path.exists(class_dir1) else class_dir2
                
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_dir, img_name)
                            self.samples.append((img_path, class_id))

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Loads and returns a single sample (image and label) from the dataset."""
        img_path, label = self.samples[idx]
        
        # Open image and convert to RGB (since ResNet expects 3 channels)
        # Even though X-rays are grayscale, we duplicate the channel to make it RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
