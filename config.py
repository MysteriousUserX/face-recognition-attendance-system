"""
Configuration file for the Face Recognition Attendance System
"""

# Model Configuration (shared settings)
MODEL_CONFIG = {
    'image_size': (160, 160),  # Input image size
    'batch_size': 32,
    'learning_rate': 0.0001,  # Same for both models
    'epochs': 30,  # Same for both models
    'num_workers': 4,
    'embedding_size': 128,  # Embedding dimension for both models
}

# Classification Model Config
CLASSIFICATION_CONFIG = {
    'num_classes': None,  # Will be set based on dataset
    'backbone': 'resnet50',  # Same backbone as metric learning
    'pretrained': True,
    'freeze_backbone': False,
    'embedding_size': 128,  # Same as metric learning for fair comparison
    'max_images_per_class': 20,  # Same data limit
}

# Metric Learning Config
METRIC_LEARNING_CONFIG = {
    'embedding_size': 128,  # Same as classification
    'margin': 0.5,  # Triplet loss margin
    'mining_strategy': 'semi-hard',  # 'hard', 'semi-hard', 'batch-all'
    'P': 32,  # Number of identities per batch
    'K': 4,   # Images per identity (32*4=128 batch size)
    'backbone': 'resnet50',  # Same backbone as classification
    'pretrained': True,
    'max_images_per_class': 20,  # Same data limit
    'learning_rate': 0.0001,  # Same as classification
    'epochs': 30,  # Same as classification for fair comparison
}

# Paths
PATHS = {
    'train_data': 'train_data',
    'val_data': 'val_data',
    'test_data': 'test_data',
    'verification_data': 'verification_data',
    'verification_pairs_val': 'verification_pairs_val.txt',
    'verification_pairs_test': 'verification_pairs_test.txt',
    'registered_faces': 'data/registered_faces',
    'checkpoints': 'checkpoints',
    'results': 'results',
    'logs': 'logs',
}

# Data Augmentation
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'rotation_range': 10,
    'brightness': 0.2,
    'contrast': 0.2,
}

# Face Detection
FACE_DETECTION_CONFIG = {
    'method': 'opencv',  # 'mtcnn', 'opencv'
    'min_face_size': 20,
}

# Verification Thresholds
VERIFICATION_CONFIG = {
    'distance_metric': 'cosine',  # 'cosine' or 'euclidean'
    'threshold': 0.6,  # Will be optimized based on ROC curve
}

# Anti-Spoofing Config
ANTI_SPOOFING_CONFIG = {
    'enabled': True,
    'model_type': 'deeppixbis',  # 'deeppixbis' (pre-trained DeepPixBiS ONNX model)
    'threshold': 0.5,  # Liveness threshold (0.0-1.0) - 0.5 for balanced, 0.03 for lenient
}

# Emotion Detection Config
EMOTION_CONFIG = {
    'enabled': True,
    'emotions': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
    'model_type': 'fer',  # Pre-trained model type
}

# Training
TRAINING_CONFIG = {
    'early_stopping_patience': 5,
    'save_best_only': True,
    'device': 'cuda',  # Will auto-detect
}
