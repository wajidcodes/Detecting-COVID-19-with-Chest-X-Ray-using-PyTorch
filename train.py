import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import CovidXRayDataset, get_transforms, CLASSES
from src.model import get_resnet18

def train():
    print("=== COVID-19 X-Ray Classification Training ===")
    
    # 1. Setup Data Paths & Device
    data_dir = "data"
    csv_file = "data/splits.csv"
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Setup Hyperparameters (Scientifically Rigorous Setup)
    batch_size = 32
    num_epochs = 25     # 25 epochs is excellent for ResNet-18 to fully converge
    learning_rate = 0.001
    
    # 3. Load Data
    train_transform, val_transform = get_transforms()
    
    print("\nLoading datasets...")
    train_dataset = CovidXRayDataset(data_dir, csv_file, split='train', transform=train_transform)
    val_dataset = CovidXRayDataset(data_dir, csv_file, split='val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 4. Initialize Model, Loss Function, and Optimizer
    print("\nInitializing ResNet-18 model...")
    model = get_resnet18(num_classes=len(CLASSES), pretrained=True)
    model = model.to(device)
    
    # --- SCIENTIFIC RIGOR: CLASS WEIGHTS ---
    # The dataset is highly imbalanced (Normal: 10k, Viral Pneumonia: 1.3k).
    # To prevent the model from just guessing "Normal", we mathematically weight the loss.
    # Formula: Total_Images / (Number_of_Classes * Class_Count)
    # Order: ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    weights = torch.tensor([1.463, 0.880, 0.519, 3.934], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning Rate Scheduler: Reduces learning rate if validation accuracy stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # 5. Training Loop
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # --- TRAINING PHASE ---
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if (i+1) % 100 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
        train_accuracy = 100 * correct_train / total_train
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_accuracy:.2f}%")
        
        # --- VALIDATION PHASE ---
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): # Don't track gradients during validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_accuracy = 100 * correct_val / total_val
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_accuracy:.2f}%")
        
        # Update Learning Rate Scheduler
        scheduler.step(val_accuracy)
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_path = os.path.join(save_dir, "resnet18_covid.pth")
            torch.save(model.state_dict(), save_path)
            print(f"*** Model improved! Saved to {save_path} ***")
            
    print(f"\nTraining complete. Best Validation Accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    train()
