"""
Liveness Detection using DeepPixBiS (Deep Pixel-wise Binary Supervision)
Based on: https://github.com/ffletcherr/face-recognition-liveness
"""

import urllib.request
from pathlib import Path
import cv2
import numpy as np
import onnxruntime
from PIL import Image
from torchvision import transforms as T
import os


class LivenessDetector:
    """
    Liveness detection using DeepPixBiS ONNX model
    Detects face presentation attacks (photos, videos, masks, etc.)
    """
    
    def __init__(self, checkpoint_path=None):
        """
        Initialize liveness detector
        
        Args:
            checkpoint_path: Path to ONNX model (will download if not exists)
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join('checkpoints', 'OULU_Protocol_2_model_0_0.onnx')
        
        # Download model if not exists
        if not Path(checkpoint_path).is_file():
            print("Downloading DeepPixBiS anti-spoofing model...")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            urllib.request.urlretrieve(
                "https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/OULU_Protocol_2_model_0_0.onnx",
                Path(checkpoint_path).absolute().as_posix()
            )
            print(f"✓ Model downloaded to {checkpoint_path}")
        
        # Load ONNX model
        self.model = onnxruntime.InferenceSession(
            checkpoint_path, providers=["CPUExecutionProvider"]
        )
        
        # Image preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"✓ Loaded DeepPixBiS liveness detection model from {checkpoint_path}")
    
    def predict(self, face_image):
        """
        Predict if face is real or spoofed
        
        Args:
            face_image: Face image (numpy array, BGR format)
            
        Returns:
            dict with:
                - is_real (bool): True if real, False if fake
                - confidence (float): Liveness score [0, 1]
                - label (str): "Real" or "Fake"
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # Preprocess
        face_tensor = self.transform(face_pil).unsqueeze(0).detach().cpu().numpy()
        
        # Predict
        output_pixel, output_binary = self.model.run(
            ["output_pixel", "output_binary"], 
            {"input": face_tensor.astype(np.float32)}
        )
        
        # Calculate liveness score (average of pixel-wise and binary outputs)
        liveness_score = (
            np.mean(output_pixel.flatten()) + np.mean(output_binary.flatten())
        ) / 2.0
        
        is_real = liveness_score > 0.3
        
        return {
            'is_real': is_real,
            'confidence': float(liveness_score),
            'label': "Real" if is_real else "Fake",
            'real_prob': float(liveness_score),
            'fake_prob': float(1.0 - liveness_score)
        }
    
    def __call__(self, face_image):
        """Allow calling detector directly"""
        return self.predict(face_image)


def create_liveness_detector(checkpoint_path=None):
    """
    Create liveness detector
    
    Args:
        checkpoint_path: Path to ONNX model (optional)
        
    Returns:
        LivenessDetector instance
    """
    return LivenessDetector(checkpoint_path=checkpoint_path)
