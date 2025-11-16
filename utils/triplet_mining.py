"""
Triplet mining strategies for metric learning
"""

import torch
import numpy as np


def pairwise_distances(embeddings, squared=False):
    """
    Compute pairwise distances between embeddings
    
    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        squared: If True, return squared Euclidean distances
        
    Returns:
        pairwise_dist: Tensor of shape [batch_size, batch_size]
    """
    # Compute dot product
    dot_product = torch.matmul(embeddings, embeddings.t())
    
    # Get squared L2 norm for each embedding
    square_norm = torch.diag(dot_product)
    
    # Compute squared distances: ||a - b||^2 = ||a||^2 - 2 <a, b> + ||b||^2
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    
    # Ensure distances are non-negative (numerical stability)
    distances = torch.clamp(distances, min=0.0)
    
    if not squared:
        # Add small epsilon for numerical stability when taking sqrt
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        # Undo epsilon addition
        distances = distances * (1.0 - mask)
    
    return distances


def get_triplet_mask(labels):
    """
    Return a 3D mask for valid triplets
    
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    
    Args:
        labels: Tensor of shape [batch_size]
        
    Returns:
        mask: Boolean tensor of shape [batch_size, batch_size, batch_size]
    """
    # Check that i, j, k are distinct
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    
    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k
    
    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)
    
    valid_labels = i_equal_j & (~i_equal_k)
    
    return distinct_indices & valid_labels


def batch_all_triplet_loss(embeddings, labels, margin, squared=False):
    """
    Batch-all triplet loss
    Uses all valid triplets in the batch
    
    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        margin: Margin for triplet loss
        squared: If True, use squared distances
        
    Returns:
        loss: Scalar tensor
        fraction_positive_triplets: Fraction of triplets with positive loss
    """
    # Get pairwise distances
    pairwise_dist = pairwise_distances(embeddings, squared=squared)
    
    # Get anchor-positive distances
    anchor_positive_dist = pairwise_dist.unsqueeze(2)  # [batch, batch, 1]
    
    # Get anchor-negative distances
    anchor_negative_dist = pairwise_dist.unsqueeze(1)  # [batch, 1, batch]
    
    # Compute triplet loss: d(a,p) - d(a,n) + margin
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    
    # Get valid triplets mask
    mask = get_triplet_mask(labels)
    triplet_loss = triplet_loss * mask.float()
    
    # Remove negative losses (easy triplets)
    triplet_loss = torch.clamp(triplet_loss, min=0.0)
    
    # Count positive triplets
    positive_triplets = (triplet_loss > 1e-16).float()
    num_positive_triplets = torch.sum(positive_triplets)
    num_valid_triplets = torch.sum(mask.float())
    
    fraction_positive = num_positive_triplets / (num_valid_triplets + 1e-16)
    
    # Average over positive triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    
    return triplet_loss, fraction_positive


def batch_hard_triplet_loss(embeddings, labels, margin, squared=False):
    """
    Batch-hard triplet loss
    For each anchor, select the hardest positive and hardest negative
    
    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        margin: Margin for triplet loss
        squared: If True, use squared distances
        
    Returns:
        loss: Scalar tensor
    """
    # Get pairwise distances
    pairwise_dist = pairwise_distances(embeddings, squared=squared)
    
    # For each anchor, get the hardest positive
    # (farthest positive sample)
    mask_anchor_positive = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask_anchor_positive = mask_anchor_positive - torch.eye(labels.size(0), device=labels.device)
    
    # Add large value to distances of negatives and anchor itself
    anchor_positive_dist = pairwise_dist + (1.0 - mask_anchor_positive) * 1e9
    hardest_positive_dist = torch.min(anchor_positive_dist, dim=1)[0]
    
    # For each anchor, get the hardest negative
    # (closest negative sample)
    mask_anchor_negative = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
    
    # Subtract large value from distances of positives
    anchor_negative_dist = pairwise_dist + (1.0 - mask_anchor_negative) * 1e9
    hardest_negative_dist = torch.min(anchor_negative_dist, dim=1)[0]
    
    # Combine to get triplet loss
    triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)
    
    # Average over batch
    triplet_loss = torch.mean(triplet_loss)
    
    return triplet_loss


def batch_semi_hard_triplet_loss(embeddings, labels, margin, squared=False):
    """
    Batch semi-hard triplet loss
    For each anchor-positive pair, select negatives that are:
    - Farther than positive (d(a,n) > d(a,p))
    - But still within margin (d(a,p) < d(a,n) < d(a,p) + margin)
    
    Falls back to hard negatives if no semi-hard negatives exist
    
    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        margin: Margin for triplet loss
        squared: If True, use squared distances
        
    Returns:
        loss: Scalar tensor
    """
    # Get pairwise distances
    pairwise_dist = pairwise_distances(embeddings, squared=squared)
    
    # Get anchor-positive mask
    mask_anchor_positive = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask_anchor_positive = mask_anchor_positive - torch.eye(labels.size(0), device=labels.device)
    
    # Get anchor-negative mask
    mask_anchor_negative = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
    
    # For each anchor-positive pair, find semi-hard negatives
    anchor_positive_dist = pairwise_dist.unsqueeze(2)  # [batch, batch, 1]
    anchor_negative_dist = pairwise_dist.unsqueeze(1)  # [batch, 1, batch]
    
    # Semi-hard condition: d(a,p) < d(a,n) < d(a,p) + margin
    semi_hard_negatives = (anchor_negative_dist > anchor_positive_dist) & \
                         (anchor_negative_dist < anchor_positive_dist + margin)
    
    # Combine with negative mask
    mask_anchor_negative_3d = mask_anchor_negative.unsqueeze(1)
    semi_hard_mask = semi_hard_negatives.float() * mask_anchor_negative_3d
    
    # If no semi-hard negatives, use hardest negative
    num_semi_hard = torch.sum(semi_hard_mask, dim=2, keepdim=True)
    use_hard = (num_semi_hard == 0).float()
    
    # Get hardest negative for each anchor
    hardest_negative_dist = pairwise_dist + (1.0 - mask_anchor_negative) * 1e9
    hardest_negative_dist = torch.min(hardest_negative_dist, dim=1, keepdim=True)[0]
    
    # Select semi-hard or hard negatives
    masked_negative_dist = anchor_negative_dist * semi_hard_mask + \
                          hardest_negative_dist.unsqueeze(1) * use_hard * mask_anchor_negative_3d
    
    # Get the selected negative
    masked_negative_dist = masked_negative_dist + (1.0 - semi_hard_mask - use_hard * mask_anchor_negative_3d) * 1e9
    final_negative_dist = torch.min(masked_negative_dist, dim=2)[0]
    
    # Calculate triplet loss for each anchor-positive pair
    triplet_loss = anchor_positive_dist.squeeze(2) - final_negative_dist + margin
    triplet_loss = torch.clamp(triplet_loss, min=0.0)
    
    # Mask valid anchor-positive pairs
    triplet_loss = triplet_loss * mask_anchor_positive
    
    # Average over valid pairs
    num_positive_pairs = torch.sum(mask_anchor_positive)
    triplet_loss = torch.sum(triplet_loss) / (num_positive_pairs + 1e-16)
    
    return triplet_loss


def triplet_loss(embeddings, labels, margin=0.5, strategy='semi-hard', squared=False):
    """
    Compute triplet loss with specified mining strategy
    
    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        margin: Margin for triplet loss
        strategy: 'batch-all', 'hard', or 'semi-hard'
        squared: If True, use squared distances
        
    Returns:
        loss: Scalar tensor
    """
    if strategy == 'batch-all':
        loss, fraction = batch_all_triplet_loss(embeddings, labels, margin, squared)
        return loss
    elif strategy == 'hard':
        return batch_hard_triplet_loss(embeddings, labels, margin, squared)
    elif strategy == 'semi-hard':
        return batch_semi_hard_triplet_loss(embeddings, labels, margin, squared)
    else:
        raise ValueError(f"Unknown mining strategy: {strategy}")
