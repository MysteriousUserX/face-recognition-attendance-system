"""
Data loaders for face classification and verification
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, CLASSIFICATION_CONFIG, AUGMENTATION_CONFIG, PATHS


class FaceDataset(Dataset):
    """Dataset for face classification"""
    
    def __init__(self, root_dir, transform=None, max_images_per_class=None):
        """
        Args:
            root_dir: Directory with subdirectories for each person
            transform: Optional transform to be applied on images
            max_images_per_class: Maximum number of images per person (None for all)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.max_images_per_class = max_images_per_class
        
        # Get all person IDs (subdirectories)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = []
        self.targets = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all images in this class
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit number of images if specified
            if max_images_per_class and len(images) > max_images_per_class:
                images = random.sample(images, max_images_per_class)
            
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append(img_path)
                self.targets.append(class_idx)
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        if max_images_per_class:
            print(f"Limited to {max_images_per_class} images per class")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.targets[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', MODEL_CONFIG['image_size'], color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx):
        """Get class name from index"""
        return self.classes[idx]


class VerificationDataset(Dataset):
    """Dataset for face verification (pairs of images)"""
    
    def __init__(self, pairs_file, verification_data_dir=None, transform=None):
        """
        Args:
            pairs_file: Path to file containing verification pairs
            verification_data_dir: Root directory for verification images (optional, paths may already be in file)
            transform: Optional transform to be applied on images
        """
        self.verification_data_dir = verification_data_dir
        self.transform = transform
        
        # Load pairs from file
        self.pairs = []
        self.labels = []
        
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # Check if paths already contain directory or are relative
                    img1_rel = parts[0]
                    img2_rel = parts[1]
                    
                    # If paths don't start with verification_data/, prepend it
                    if not img1_rel.startswith('verification_data'):
                        img1_path = os.path.join(verification_data_dir, img1_rel) if verification_data_dir else img1_rel
                    else:
                        img1_path = img1_rel
                    
                    if not img2_rel.startswith('verification_data'):
                        img2_path = os.path.join(verification_data_dir, img2_rel) if verification_data_dir else img2_rel
                    else:
                        img2_path = img2_rel
                    
                    label = int(parts[2])
                    
                    self.pairs.append((img1_path, img2_path))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.pairs)} verification pairs")
        print(f"Positive pairs: {sum(self.labels)}, Negative pairs: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        # Load images
        try:
            img1 = Image.open(img1_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img1_path}: {e}")
            img1 = Image.new('RGB', MODEL_CONFIG['image_size'], color='black')
        
        try:
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img2_path}: {e}")
            img2 = Image.new('RGB', MODEL_CONFIG['image_size'], color='black')
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label


def get_transforms(train=True):
    """Get image transforms for training or validation"""
    
    if train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize(MODEL_CONFIG['image_size']),
            transforms.RandomHorizontalFlip(p=0.5) if AUGMENTATION_CONFIG['horizontal_flip'] else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(AUGMENTATION_CONFIG['rotation_range']),
            transforms.ColorJitter(
                brightness=AUGMENTATION_CONFIG['brightness'],
                contrast=AUGMENTATION_CONFIG['contrast']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize(MODEL_CONFIG['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def get_classification_loaders(batch_size=None):
    """Get train, validation, and test data loaders for classification"""
    
    if batch_size is None:
        batch_size = MODEL_CONFIG['batch_size']
    
    # Get transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    # Create datasets
    train_dataset = FaceDataset(
        root_dir=PATHS['train_data'],
        transform=train_transform,
        max_images_per_class=CLASSIFICATION_CONFIG['max_images_per_class']
    )
    
    val_dataset = FaceDataset(
        root_dir=PATHS['val_data'],
        transform=val_transform,
        max_images_per_class=None  # Use all validation images
    )
    
    test_dataset = FaceDataset(
        root_dir=PATHS['test_data'],
        transform=val_transform,
        max_images_per_class=None  # Use all test images
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=MODEL_CONFIG['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=MODEL_CONFIG['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=MODEL_CONFIG['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, len(train_dataset.classes)


def get_verification_loader(pairs_file=None, batch_size=None):
    """Get data loader for verification pairs"""
    
    if batch_size is None:
        batch_size = MODEL_CONFIG['batch_size']
    
    if pairs_file is None:
        pairs_file = PATHS['verification_pairs_val']
    
    # Get transform (no augmentation for verification)
    transform = get_transforms(train=False)
    
    # Create dataset (paths in file already include verification_data/)
    verification_dataset = VerificationDataset(
        pairs_file=pairs_file,
        verification_data_dir=None,  # Paths already complete in file
        transform=transform
    )
    
    # Create data loader
    verification_loader = DataLoader(
        verification_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=MODEL_CONFIG['num_workers'],
        pin_memory=True
    )
    
    return verification_loader


def get_triplet_loader(batch_size=1):
    """
    Get data loader for triplet learning
    Note: batch_size should be 1 as TripletDataset returns P*K samples
    """
    from config import METRIC_LEARNING_CONFIG
    
    # Get transform with augmentation
    transform = get_transforms(train=True)
    
    # Create dataset
    triplet_dataset = TripletDataset(
        root_dir=PATHS['train_data'],
        transform=transform,
        P=METRIC_LEARNING_CONFIG['P'],
        K=METRIC_LEARNING_CONFIG['K'],
        max_images_per_class=METRIC_LEARNING_CONFIG['max_images_per_class']
    )
    
    # Create data loader with batch_size=1
    # Each "batch" is actually P*K samples returned by dataset
    triplet_loader = DataLoader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for triplet dataset to avoid multiprocessing issues
        pin_memory=True
    )
    
    return triplet_loader, len(triplet_dataset.classes)


# Triplet Dataset for metric learning
class TripletDataset(Dataset):
    """Dataset for triplet learning using PK sampling"""
    
    def __init__(self, root_dir, transform=None, P=32, K=4, max_images_per_class=None):
        """
        Args:
            root_dir: Directory with subdirectories for each person
            transform: Optional transform to be applied on images
            P: Number of identities per batch
            K: Number of images per identity
            max_images_per_class: Maximum images per person
        """
        self.root_dir = root_dir
        self.transform = transform
        self.P = P
        self.K = K
        
        # Get all person IDs
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        
        # Create a dictionary mapping class to list of image paths
        self.class_to_images = {}
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit images per class if specified
            if max_images_per_class and len(images) > max_images_per_class:
                images = random.sample(images, max_images_per_class)
            
            if len(images) >= K:  # Need at least K images for sampling
                self.class_to_images[class_name] = images
        
        # Filter classes and create class to index mapping
        self.classes = list(self.class_to_images.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Loaded triplet dataset with {len(self.classes)} classes")
        if max_images_per_class:
            print(f"Limited to {max_images_per_class} images per class")
        print(f"Batch composition: P={P} identities × K={K} images = {P*K} samples per batch")
    
    def __len__(self):
        # Number of batches we can create
        return len(self.classes) // self.P
    
    def __getitem__(self, idx):
        """
        Return a batch of P*K images with P identities and K images each
        This is designed to work with batch_size=1 in DataLoader
        """
        # Select P random classes
        selected_classes = random.sample(self.classes, self.P)
        
        images = []
        labels = []
        
        for cls in selected_classes:
            # Sample K images from this class
            class_images = random.sample(self.class_to_images[cls], 
                                        min(self.K, len(self.class_to_images[cls])))
            
            # If less than K images, repeat some
            while len(class_images) < self.K:
                class_images.append(random.choice(self.class_to_images[cls]))
            
            # Load and transform images
            for img_path in class_images:
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    img = Image.new('RGB', MODEL_CONFIG['image_size'], color='black')
                
                if self.transform:
                    img = self.transform(img)
                
                images.append(img)
                labels.append(self.class_to_idx[cls])
        
        # Stack images and convert labels to tensor
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return images, labels
