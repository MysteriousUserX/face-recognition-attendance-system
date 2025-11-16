"""
Evaluation metrics for face verification
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS


def compute_similarity(embedding1, embedding2, metric='cosine'):
    """
    Compute similarity between two embeddings
    
    Args:
        embedding1: First embedding vector (numpy or torch)
        embedding2: Second embedding vector (numpy or torch)
        metric: 'cosine' or 'euclidean'
        
    Returns:
        similarity_score: Similarity score (higher means more similar for cosine)
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(embedding1):
        embedding1 = embedding1.cpu().numpy()
    if torch.is_tensor(embedding2):
        embedding2 = embedding2.cpu().numpy()
    
    if metric == 'cosine':
        # Cosine similarity (1 - cosine distance)
        # Higher value means more similar
        return 1 - cosine(embedding1, embedding2)
    elif metric == 'euclidean':
        # Negative euclidean distance (higher is more similar)
        return -euclidean(embedding1, embedding2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def calculate_roc_auc(y_true, y_scores, plot=True, save_path=None):
    """
    Calculate ROC curve and AUC score
    
    Args:
        y_true: True labels (1 for same person, 0 for different)
        y_scores: Similarity scores
        plot: Whether to plot ROC curve
        save_path: Path to save the plot
        
    Returns:
        dict with fpr, tpr, thresholds, auc_score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    
    if plot:
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        plt.close()  # Close instead of show to avoid popup
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': auc_score
    }


def find_optimal_threshold(fpr, tpr, thresholds):
    """
    Find optimal threshold that maximizes (TPR - FPR)
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Threshold values
        
    Returns:
        dict with optimal_threshold, optimal_tpr, optimal_fpr
    """
    # Find threshold that maximizes TPR - FPR (Youden's index)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"True Positive Rate: {optimal_tpr:.4f}")
    print(f"False Positive Rate: {optimal_fpr:.4f}")
    
    return {
        'threshold': optimal_threshold,
        'tpr': optimal_tpr,
        'fpr': optimal_fpr
    }


def evaluate_verification(model, verification_loader, device, metric='cosine', save_prefix=''):
    """
    Evaluate model on verification task
    
    Args:
        model: Trained model
        verification_loader: DataLoader with verification pairs
        device: Device to run on
        metric: Distance metric ('cosine' or 'euclidean')
        save_prefix: Prefix for saved files (e.g., 'classification' or 'metric_learning')
        
    Returns:
        dict with scores, labels, and metrics
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    
    print("Evaluating verification pairs...")
    with torch.no_grad():
        for img1, img2, labels in verification_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Get embeddings
            emb1 = model.get_embedding(img1)
            emb2 = model.get_embedding(img2)
            
            # Compute similarity for each pair
            for i in range(len(labels)):
                score = compute_similarity(emb1[i], emb2[i], metric=metric)
                all_scores.append(score)
                all_labels.append(labels[i].item())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Calculate ROC and AUC
    if save_prefix:
        save_path = os.path.join(PATHS['results'], f'{save_prefix}_roc_curve.png')
    else:
        save_path = os.path.join(PATHS['results'], 'roc_curve.png')
    
    roc_results = calculate_roc_auc(all_labels, all_scores, plot=True, save_path=save_path)
    
    # Find optimal threshold
    optimal = find_optimal_threshold(roc_results['fpr'], roc_results['tpr'], roc_results['thresholds'])
    
    # Calculate accuracy at optimal threshold
    predictions = (all_scores >= optimal['threshold']).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    
    print(f"\nVerification Results:")
    print(f"AUC Score: {roc_results['auc']:.4f}")
    print(f"Accuracy at optimal threshold: {accuracy:.4f}")
    
    return {
        'scores': all_scores,
        'labels': all_labels,
        'auc': roc_results['auc'],
        'optimal_threshold': optimal['threshold'],
        'accuracy': accuracy,
        'roc_results': roc_results
    }


def evaluate_classification(model, test_loader, device):
    """
    Evaluate model on classification task
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data
        device: Device to run on
        
    Returns:
        accuracy
    """
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Classification Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy
