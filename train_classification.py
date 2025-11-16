"""
Training script for classification-based face recognition model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, CLASSIFICATION_CONFIG, TRAINING_CONFIG, PATHS
from models.classification_model import create_model, save_model
from utils.data_loader import get_classification_loaders
from utils.metrics import evaluate_classification, evaluate_verification


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = correct / total
    
    return val_loss, val_acc


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()  # Close instead of show


def train():
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory with embedding size suffix
    embed_size = CLASSIFICATION_CONFIG.get('embedding_size', 512)
    results_dir = os.path.join(PATHS['results'], f'classification_embed{embed_size}')
    checkpoints_dir = os.path.join(PATHS['checkpoints'], f'classification_embed{embed_size}')
    logs_dir = os.path.join(PATHS['logs'], f'classification_embed{embed_size}')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"\nResults will be saved to: {results_dir}")
    print(f"Checkpoints will be saved to: {checkpoints_dir}")
    print(f"Embedding size: {embed_size}-dim")
    
    # Get data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, num_classes = get_classification_loaders()
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(num_classes)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {MODEL_CONFIG['epochs']} epochs...")
    print("=" * 70)
    
    for epoch in range(1, MODEL_CONFIG['epochs'] + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{MODEL_CONFIG['epochs']}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model only
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint_path = os.path.join(checkpoints_dir, 'best_classification_model.pth')
            save_model(model, optimizer, epoch, val_acc, checkpoint_path)
            print(f"  New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        print("-" * 70)
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training history
    history_plot_path = os.path.join(results_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    test_acc = evaluate_classification(model, test_loader, device)
    
    print("\n" + "=" * 70)
    print("Final Results:")
    print(f"  Embedding Size: {embed_size}-dim")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print("=" * 70)
    
    # Save final results
    results_file = os.path.join(results_dir, 'classification_results.txt')
    with open(results_file, 'w') as f:
        f.write("CLASSIFICATION MODEL RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Embedding Size: {embed_size}-dim\n")
        f.write(f"Backbone: {CLASSIFICATION_CONFIG['backbone']}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nResults saved to: {results_file}")
    
    return model, history


if __name__ == '__main__':
    model, history = train()
    
    print("\n" + "=" * 70)
    print("Training script completed!")
    print("Next steps:")
    print("1. Run evaluate_verification.py to test on verification pairs")
    print("2. Check results/ folder for training plots")
    print("3. Check checkpoints/ folder for saved models")
    print("=" * 70)
