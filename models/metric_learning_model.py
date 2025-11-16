"""
Metric Learning Model using Triplet Loss
"""

import torch
import torch.nn as nn
import torchvision.models as models

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import METRIC_LEARNING_CONFIG


class MetricLearningModel(nn.Module):
    """Face recognition model using metric learning (triplet loss)"""
    
    def __init__(self, embedding_size=128, backbone='resnet50', pretrained=True):
        """
        Args:
            embedding_size: Size of embedding vector
            backbone: Backbone architecture ('resnet18', 'resnet50', etc.)
            pretrained: Use ImageNet pretrained weights
        """
        super(MetricLearningModel, self).__init__()
        
        self.embedding_size = embedding_size
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Embedding layer with BatchNorm and optional ReLU
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            # No ReLU - allows negative values in embeddings
        )
        
        print(f"Created {backbone} metric learning model")
        print(f"Embedding size: {embedding_size}")
    
    def forward(self, x):
        """
        Forward pass - returns L2 normalized embeddings
        
        Args:
            x: Input images [batch_size, 3, H, W]
            
        Returns:
            embeddings: L2-normalized embeddings [batch_size, embedding_size]
        """
        # Extract features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Get embeddings
        embeddings = self.embedding(features)
        
        # L2 normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding(self, x):
        """Get normalized embeddings (same as forward for metric learning)"""
        return self.forward(x)


def create_metric_model():
    """Create metric learning model based on config"""
    model = MetricLearningModel(
        embedding_size=METRIC_LEARNING_CONFIG['embedding_size'],
        backbone=METRIC_LEARNING_CONFIG['backbone'],
        pretrained=METRIC_LEARNING_CONFIG['pretrained']
    )
    
    return model


def load_metric_model(checkpoint_path, device='cuda'):
    """Load a trained metric learning model from checkpoint"""
    model = create_metric_model()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'auc' in checkpoint:
        print(f"AUC: {checkpoint['auc']:.4f}")
    
    return model


def save_metric_model(model, optimizer, epoch, auc, save_path):
    """Save metric learning model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'auc': auc,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved model to {save_path}")
