"""
Evaluate face verification performance using trained models
Supports both classification and metric learning models
"""

import torch
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PATHS, VERIFICATION_CONFIG
from models.classification_model import load_model as load_classification_model
from models.metric_learning_model import load_metric_model
from utils.data_loader import get_verification_loader, get_classification_loaders
from utils.metrics import evaluate_verification


def main(model_type='classification', embed_size=128):
    """
    Main evaluation function
    
    Args:
        model_type: 'classification' or 'metric_learning'
        embed_size: embedding dimension to evaluate (default: 128)
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    print(f"Embedding size: {embed_size}-dim")
    
    # Set checkpoint directory based on embedding size
    checkpoint_dir = os.path.join(PATHS['checkpoints'], f'{model_type}_embed{embed_size}')
    results_dir = os.path.join(PATHS['results'], f'{model_type}_embed{embed_size}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model based on type
    if model_type == 'classification':
        # Get number of classes from train data
        print("\nLoading dataset info...")
        _, _, _, num_classes = get_classification_loaders()
        print(f"Number of classes: {num_classes}")
        
        # Load trained model
        checkpoint_path = os.path.join(checkpoint_dir, 'best_classification_model.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"\nError: Model checkpoint not found at {checkpoint_path}")
            print("Please train the model first using train_classification.py")
            return None
        
        print(f"\nLoading classification model from {checkpoint_path}...")
        model = load_classification_model(checkpoint_path, num_classes, device)
        
    elif model_type == 'metric_learning':
        # Load metric learning model
        checkpoint_path = os.path.join(checkpoint_dir, 'best_metric_learning_model.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"\nError: Model checkpoint not found at {checkpoint_path}")
            print("Please train the model first using train_metric_learning.py")
            return None
        
        print(f"\nLoading metric learning model from {checkpoint_path}...")
        model = load_metric_model(checkpoint_path, device)
        
    else:
        print(f"Error: Unknown model type '{model_type}'")
        print("Use 'classification' or 'metric_learning'")
        return None
    
    model = model.to(device)
    
    # Get verification data loader
    print("\nLoading verification pairs...")
    verification_loader = get_verification_loader(
        pairs_file=PATHS['verification_pairs_val']
    )
    
    # Evaluate verification
    print("\n" + "=" * 70)
    print(f"Evaluating Face Verification - {model_type.upper()} Model ({embed_size}-dim)")
    print("=" * 70)
    
    results = evaluate_verification(
        model=model,
        verification_loader=verification_loader,
        device=device,
        metric=VERIFICATION_CONFIG['distance_metric'],
        save_prefix=f'{model_type}_embed{embed_size}'
    )
    
    # Print results summary
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model Type: {model_type}")
    print(f"Embedding Size: {embed_size}-dim")
    print(f"Distance Metric: {VERIFICATION_CONFIG['distance_metric']}")
    print(f"AUC Score: {results['auc']:.4f}")
    print(f"Optimal Threshold: {results['optimal_threshold']:.4f}")
    print(f"Accuracy at Optimal Threshold: {results['accuracy']:.4f}")
    print("=" * 70)
    
    # Save results to file in the embed-specific directory
    results_file = os.path.join(results_dir, f'{model_type}_verification_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"{model_type.upper()} MODEL - FACE VERIFICATION RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Embedding Size: {embed_size}-dim\n")
        f.write(f"Distance Metric: {VERIFICATION_CONFIG['distance_metric']}\n")
        f.write(f"AUC Score: {results['auc']:.4f}\n")
        f.write(f"Optimal Threshold: {results['optimal_threshold']:.4f}\n")
        f.write(f"Accuracy at Optimal Threshold: {results['accuracy']:.4f}\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Save ROC curve with model type and embedding size in filename
    roc_path = os.path.join(results_dir, f'{model_type}_roc_curve.png')
    print(f"ROC curve saved to {roc_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate face verification models')
    parser.add_argument('--model', type=str, default=None,
                       choices=['classification', 'metric_learning', 'both'],
                       help='Model type to evaluate')
    
    args = parser.parse_args()
    
    # If no argument provided, show interactive menu
    if args.model is None:
        print("\n" + "=" * 70)
        print("FACE VERIFICATION EVALUATION")
        print("=" * 70)
        print("\nSelect which model(s) to evaluate:")
        print("  1. Classification Model only")
        print("  2. Metric Learning Model only")
        print("  3. Both Models (with comparison)")
        print("  4. Exit")
        print("=" * 70)
        
        while True:
            try:
                choice = input("\nEnter your choice (1-4): ").strip()
                
                if choice == '1':
                    args.model = 'classification'
                    break
                elif choice == '2':
                    args.model = 'metric_learning'
                    break
                elif choice == '3':
                    args.model = 'both'
                    break
                elif choice == '4':
                    print("\nExiting...")
                    sys.exit(0)
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                sys.exit(0)
    
    # Execute evaluation based on choice
    if args.model == 'both':
        # Evaluate both models
        print("\n" + "=" * 70)
        print("EVALUATING BOTH MODELS FOR COMPARISON")
        print("=" * 70)
        
        # Get embedding size from config
        from config import MODEL_CONFIG
        embed_size = MODEL_CONFIG.get('embedding_size', 128)
        
        print(f"\nEvaluating both models with {embed_size}-dim embeddings...\n")
        
        results_classification = main(model_type='classification', embed_size=embed_size)
        print("\n")
        results_metric = main(model_type='metric_learning', embed_size=embed_size)
        
        # Compare results
        if results_classification and results_metric:
            print("\n" + "=" * 70)
            print("DETAILED COMPARISON - CLASSIFICATION vs METRIC LEARNING")
            print("=" * 70)
            print(f"\nEmbedding Size: {embed_size}-dim")
            print(f"Distance Metric: {VERIFICATION_CONFIG['distance_metric']}")
            print("\n" + "-" * 70)
            print(f"{'Metric':<30} {'Classification':<20} {'Metric Learning':<20}")
            print("-" * 70)
            print(f"{'AUC Score':<30} {results_classification['auc']:<20.4f} {results_metric['auc']:<20.4f}")
            print(f"{'Optimal Threshold':<30} {results_classification['optimal_threshold']:<20.4f} {results_metric['optimal_threshold']:<20.4f}")
            print(f"{'Accuracy at Threshold':<30} {results_classification['accuracy']:<20.4f} {results_metric['accuracy']:<20.4f}")
            print("-" * 70)
            
            # Determine winner
            if results_metric['auc'] > results_classification['auc']:
                diff = results_metric['auc'] - results_classification['auc']
                improvement_pct = (diff / results_classification['auc']) * 100
                print(f"\n🏆 WINNER: Metric Learning Model")
                print(f"   AUC Improvement: +{diff:.4f} ({improvement_pct:.2f}%)")
            elif results_classification['auc'] > results_metric['auc']:
                diff = results_classification['auc'] - results_metric['auc']
                improvement_pct = (diff / results_metric['auc']) * 100
                print(f"\n🏆 WINNER: Classification Model")
                print(f"   AUC Improvement: +{diff:.4f} ({improvement_pct:.2f}%)")
            else:
                print(f"\n🤝 TIE: Both models have equal AUC scores")
            
            print("=" * 70)
            
            # Save comparison results
            comparison_file = os.path.join(PATHS['results'], f'model_comparison_embed{embed_size}.txt')
            with open(comparison_file, 'w') as f:
                f.write("MODEL COMPARISON - FACE VERIFICATION\n")
                f.write("=" * 70 + "\n")
                f.write(f"Embedding Size: {embed_size}-dim\n")
                f.write(f"Distance Metric: {VERIFICATION_CONFIG['distance_metric']}\n")
                f.write("\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'Metric':<30} {'Classification':<20} {'Metric Learning':<20}\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'AUC Score':<30} {results_classification['auc']:<20.4f} {results_metric['auc']:<20.4f}\n")
                f.write(f"{'Optimal Threshold':<30} {results_classification['optimal_threshold']:<20.4f} {results_metric['optimal_threshold']:<20.4f}\n")
                f.write(f"{'Accuracy at Threshold':<30} {results_classification['accuracy']:<20.4f} {results_metric['accuracy']:<20.4f}\n")
                f.write("-" * 70 + "\n")
                
                if results_metric['auc'] > results_classification['auc']:
                    diff = results_metric['auc'] - results_classification['auc']
                    improvement_pct = (diff / results_classification['auc']) * 100
                    f.write(f"\nWINNER: Metric Learning Model\n")
                    f.write(f"AUC Improvement: +{diff:.4f} ({improvement_pct:.2f}%)\n")
                elif results_classification['auc'] > results_metric['auc']:
                    diff = results_classification['auc'] - results_metric['auc']
                    improvement_pct = (diff / results_metric['auc']) * 100
                    f.write(f"\nWINNER: Classification Model\n")
                    f.write(f"AUC Improvement: +{diff:.4f} ({improvement_pct:.2f}%)\n")
                else:
                    f.write(f"\nRESULT: Both models have equal AUC scores\n")
                
                f.write("=" * 70 + "\n")
            
            print(f"\nComparison results saved to {comparison_file}")
    else:
        # Get embedding size from config
        from config import MODEL_CONFIG
        embed_size = MODEL_CONFIG.get('embedding_size', 128)
        
        results = main(model_type=args.model, embed_size=embed_size)
