"""
Training script for metric learning face recognition model using triplet loss
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import METRIC_LEARNING_CONFIG, TRAINING_CONFIG, PATHS, VERIFICATION_CONFIG
from models.metric_learning_model import create_metric_model, save_metric_model
from utils.data_loader import get_triplet_loader, get_verification_loader
from utils.triplet_mining import triplet_loss
from utils.metrics import evaluate_verification


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch using triplet loss"""
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_images, batch_labels in pbar:
        # Squeeze batch dimension since TripletDataset returns [1, P*K, C, H, W]
        batch_images = batch_images.squeeze(0)  # [P*K, C, H, W]
        batch_labels = batch_labels.squeeze(0)  # [P*K]
        
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass - get embeddings
        embeddings = model(batch_images)
        
        # Compute triplet loss
        loss = triplet_loss(
            embeddings=embeddings,
            labels=batch_labels,
            margin=METRIC_LEARNING_CONFIG['margin'],
            strategy=METRIC_LEARNING_CONFIG['mining_strategy'],
            squared=False
        )
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / num_batches
        })
    
    epoch_loss = running_loss / num_batches
    
    return epoch_loss


def validate_one_epoch(model, val_loader, device):
    """Validate for one epoch using triplet loss"""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            # Squeeze batch dimension
            batch_images = batch_images.squeeze(0)
            batch_labels = batch_labels.squeeze(0)
            
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            embeddings = model(batch_images)
            
            # Compute triplet loss
            loss = triplet_loss(
                embeddings=embeddings,
                labels=batch_labels,
                margin=METRIC_LEARNING_CONFIG['margin'],
                strategy=METRIC_LEARNING_CONFIG['mining_strategy'],
                squared=False
            )
            
            running_loss += loss.item()
            num_batches += 1
    
    epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0
    
    return epoch_loss


def plot_training_history(history, save_path):
    """Plot training history with validation loss"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Triplet Loss', fontsize=12)
    plt.title('Metric Learning - Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def train():
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories with embedding size suffix
    embed_size = METRIC_LEARNING_CONFIG['embedding_size']
    results_dir = os.path.join(PATHS['results'], f'metric_learning_embed{embed_size}')
    checkpoints_dir = os.path.join(PATHS['checkpoints'], f'metric_learning_embed{embed_size}')
    logs_dir = os.path.join(PATHS['logs'], f'metric_learning_embed{embed_size}')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"\nResults will be saved to: {results_dir}")
    print(f"Checkpoints will be saved to: {checkpoints_dir}")
    print(f"Embedding size: {embed_size}-dim")
    
    # Get data loaders
    print("\nLoading datasets...")
    train_loader, num_classes = get_triplet_loader(batch_size=1)
    val_loader, _ = get_triplet_loader(batch_size=1)  # Separate validation loader
    print(f"Number of classes: {num_classes}")
    
    # Get verification loader for evaluation
    verification_loader = get_verification_loader()
    
    # Create model
    print("\nCreating model...")
    model = create_metric_model()
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=METRIC_LEARNING_CONFIG['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_auc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {METRIC_LEARNING_CONFIG['epochs']} epochs...")
    print(f"Triplet loss margin: {METRIC_LEARNING_CONFIG['margin']}")
    print(f"Mining strategy: {METRIC_LEARNING_CONFIG['mining_strategy']}")
    print("=" * 70)
    
    for epoch in range(1, METRIC_LEARNING_CONFIG['epochs'] + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate_one_epoch(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{METRIC_LEARNING_CONFIG['epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        print("-" * 70)
    
    print("\nTraining completed!")
    
    # Evaluate verification performance at the end
    print("\n" + "=" * 70)
    print("Final Verification Evaluation:")
    final_results = evaluate_verification(
        model=model,
        verification_loader=verification_loader,
        device=device,
        metric=VERIFICATION_CONFIG['distance_metric'],
        save_prefix=f'metric_learning_embed{embed_size}_final'
    )
    
    best_auc = final_results['auc']
    
    # Save best model only
    checkpoint_path = os.path.join(checkpoints_dir, 'best_metric_learning_model.pth')
    save_metric_model(model, optimizer, METRIC_LEARNING_CONFIG['epochs'], best_auc, checkpoint_path)
    
    print(f"Best verification AUC: {best_auc:.4f}")
    
    # Plot training history
    history_plot_path = os.path.join(results_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)
    
    # Save final results
    results_file = os.path.join(results_dir, 'metric_learning_results.txt')
    with open(results_file, 'w') as f:
        f.write("METRIC LEARNING RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Embedding Size: {embed_size}-dim\n")
        f.write(f"Model: {METRIC_LEARNING_CONFIG['backbone']}\n")
        f.write(f"Embedding Size: {METRIC_LEARNING_CONFIG['embedding_size']}\n")
        f.write(f"Margin: {METRIC_LEARNING_CONFIG['margin']}\n")
        f.write(f"Mining Strategy: {METRIC_LEARNING_CONFIG['mining_strategy']}\n")
        f.write(f"Distance Metric: {VERIFICATION_CONFIG['distance_metric']}\n")
        f.write(f"\nFinal AUC: {best_auc:.4f}\n")
        f.write(f"Optimal Threshold: {final_results['optimal_threshold']:.4f}\n")
        f.write(f"Accuracy at Optimal Threshold: {final_results['accuracy']:.4f}\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nResults saved to {results_file}")
    
    print("\n" + "=" * 70)
    print("Final Results:")
    print(f"  Final Verification AUC: {best_auc:.4f}")
    print("=" * 70)
    
    return model, history


if __name__ == '__main__':
    model, history = train()
    
    print("\n" + "=" * 70)
    print("Metric Learning Training Completed!")
    print("Next steps:")
    print("1. Compare with classification model results")
    print("2. Check results/ folder for training plots and ROC curves")
    print("3. Check checkpoints/ folder for saved models")
    print("=" * 70)
