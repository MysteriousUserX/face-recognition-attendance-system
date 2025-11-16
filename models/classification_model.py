"""
Classification-based Face Recognition Model using ResNet
"""

import torch
import torch.nn as nn
import torchvision.models as models

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, CLASSIFICATION_CONFIG


class ClassificationModel(nn.Module):
    """Face recognition model based on classification"""
    
    def __init__(self, num_classes, embedding_size=512, backbone='resnet50', pretrained=True):
        """
        Args:
            num_classes: Number of identities to classify
            embedding_size: Size of embedding vector
            backbone: Backbone architecture ('resnet18', 'resnet50', etc.)
            pretrained: Use ImageNet pretrained weights
        """
        super(ClassificationModel, self).__init__()
        
        self.num_classes = num_classes
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
        
        # Add embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        
        # Add classification layer (for training)
        self.classifier = nn.Linear(embedding_size, num_classes)
        
        print(f"Created {backbone} model with {num_classes} classes")
        print(f"Embedding size: {embedding_size}")
    
    def forward(self, x, return_embedding=False):
        """
        Forward pass
        
        Args:
            x: Input images [batch_size, 3, H, W]
            return_embedding: If True, return embeddings instead of logits
            
        Returns:
            logits or embeddings depending on return_embedding flag
        """
        # Extract features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Get embeddings
        embeddings = self.embedding(features)
        
        if return_embedding:
            # L2 normalize embeddings for verification
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings
        
        # Get classification logits
        logits = self.classifier(embeddings)
        return logits
    
    def get_embedding(self, x):
        """Get normalized embeddings for face verification"""
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(x, return_embedding=True)
        return embeddings
    
    def freeze_backbone(self):
        """Freeze backbone parameters (useful for fine-tuning)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")


def create_model(num_classes):
    """Create classification model based on config"""
    model = ClassificationModel(
        num_classes=num_classes,
        embedding_size=MODEL_CONFIG['embedding_size'],
        backbone=CLASSIFICATION_CONFIG['backbone'],
        pretrained=CLASSIFICATION_CONFIG['pretrained']
    )
    
    # Freeze backbone if specified
    if CLASSIFICATION_CONFIG['freeze_backbone']:
        model.freeze_backbone()
    
    return model


def load_model(checkpoint_path, num_classes, device='cuda'):
    """Load a trained model from checkpoint"""
    model = create_model(num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'accuracy' in checkpoint:
        print(f"Accuracy: {checkpoint['accuracy']:.4f}")
    
    return model


def save_model(model, optimizer, epoch, accuracy, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved model to {save_path}")
