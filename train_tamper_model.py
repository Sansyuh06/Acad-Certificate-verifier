#!/usr/bin/env python3
"""
Train Tampering Detection Model for Certificate Verification

This script trains a binary classifier to detect tampered/forged certificates.
Uses ResNet18 architecture matching the inference code.

Dataset Structure Expected:
    dataset/
        authentic/
            cert1.jpg
            cert2.jpg
            ...
        tampered/
            fake1.jpg
            fake2.jpg
            ...

Usage:
    python train_tamper_model.py --data_dir ./dataset --epochs 30
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import json
from datetime import datetime

class CertificateDataset(Dataset):
    """Dataset for authentic vs tampered certificates"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Load authentic certificates (label 0)
        authentic_dir = self.data_dir / "authentic"
        if authentic_dir.exists():
            for img_path in authentic_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 0))
        
        # Load tampered certificates (label 1)
        tampered_dir = self.data_dir / "tampered"
        if tampered_dir.exists():
            for img_path in tampered_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 1))
        
        print(f"Found {len(self.samples)} total samples")
        authentic_count = sum(1 for _, label in self.samples if label == 0)
        tampered_count = len(self.samples) - authentic_count
        print(f"  - Authentic: {authentic_count}")
        print(f"  - Tampered: {tampered_count}")
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def prepare_datasets_from_kaggle(base_dir):
    """
    Organize your Kaggle datasets into authentic/tampered structure
    This is a helper function - adjust paths based on your actual dataset structure
    """
    base_dir = Path(base_dir)
    output_dir = base_dir / "organized_dataset"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "authentic").mkdir(exist_ok=True)
    (output_dir / "tampered").mkdir(exist_ok=True)
    
    print("Organizing datasets...")
    
    # Example: Professional Certification Programs (likely authentic)
    cert_dir = base_dir / "Professional Certification Programs"
    if cert_dir.exists():
        print(f"Processing {cert_dir}")
        for img in cert_dir.rglob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    # Copy to authentic folder
                    dest = output_dir / "authentic" / f"cert_{img.stem}{img.suffix}"
                    if not dest.exists():
                        import shutil
                        shutil.copy2(img, dest)
                except Exception as e:
                    print(f"Error copying {img}: {e}")
    
    # Original and Tampered Image Dataset
    tampered_dataset = base_dir / "Original and Tampered Image Dataset"
    if tampered_dataset.exists():
        print(f"Processing {tampered_dataset}")
        
        # Look for 'original' or 'authentic' subfolder
        for subfolder in ['original', 'Original', 'authentic', 'Authentic']:
            orig_path = tampered_dataset / subfolder
            if orig_path.exists():
                for img in orig_path.rglob("*"):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        try:
                            dest = output_dir / "authentic" / f"orig_{img.stem}{img.suffix}"
                            if not dest.exists():
                                import shutil
                                shutil.copy2(img, dest)
                        except Exception as e:
                            print(f"Error: {e}")
        
        # Look for 'tampered' or 'forged' subfolder
        for subfolder in ['tampered', 'Tampered', 'forged', 'Forged', 'fake', 'Fake']:
            tamp_path = tampered_dataset / subfolder
            if tamp_path.exists():
                for img in tamp_path.rglob("*"):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        try:
                            dest = output_dir / "tampered" / f"tamp_{img.stem}{img.suffix}"
                            if not dest.exists():
                                import shutil
                                shutil.copy2(img, dest)
                        except Exception as e:
                            print(f"Error: {e}")
    
    # defacto-copymove (tampered images with copy-move forgery)
    copymove_dir = base_dir / "defacto-copymove"
    if copymove_dir.exists():
        print(f"Processing {copymove_dir}")
        for img in copymove_dir.rglob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # These are tampered images
                try:
                    dest = output_dir / "tampered" / f"copymove_{img.stem}{img.suffix}"
                    if not dest.exists():
                        import shutil
                        shutil.copy2(img, dest)
                except Exception as e:
                    print(f"Error: {e}")
    
    print(f"\nDataset organized in: {output_dir}")
    print(f"Authentic: {len(list((output_dir / 'authentic').glob('*')))}")
    print(f"Tampered: {len(list((output_dir / 'tampered').glob('*')))}")
    
    return str(output_dir)

def train_model(data_dir, epochs=30, batch_size=32, learning_rate=0.001, 
                output_dir="./models"):
    """Train the tampering detection model"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("\nLoading dataset...")
    full_dataset = CertificateDataset(data_dir, transform=train_transform)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model (ResNet18)...")
    model = timm.create_model('resnet18', pretrained=True, num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f"{train_loss/len(train_loader):.4f}",
                'acc': f"{100.*train_correct/train_total:.2f}%"
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f"{val_loss/len(val_loader):.4f}",
                    'acc': f"{100.*val_correct/val_total:.2f}%"
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            model_path = output_path / "tamper_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Best model saved: {model_path} (Val Acc: {val_acc:.2f}%)")
    
    # Save training history
    history_path = Path(output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_dir}/tamper_model.pth")
    print(f"History saved to: {history_path}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Train tampering detection model")
    parser.add_argument('--data_dir', type=str, default='./dataset',
                       help='Directory with authentic/tampered subdirectories')
    parser.add_argument('--organize_kaggle', action='store_true',
                       help='Organize Kaggle datasets first')
    parser.add_argument('--kaggle_base', type=str, default='D:/fyeshi/project/certificate/models',
                       help='Base directory containing Kaggle datasets')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Directory to save trained model')
    
    args = parser.parse_args()
    
    # Organize datasets if requested
    if args.organize_kaggle:
        print("Organizing Kaggle datasets...")
        data_dir = prepare_datasets_from_kaggle(args.kaggle_base)
    else:
        data_dir = args.data_dir
    
    # Train model
    train_model(
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()