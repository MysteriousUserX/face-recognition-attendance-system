"""
Face detection utilities using OpenCV or MTCNN
"""

import cv2
import numpy as np
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import FACE_DETECTION_CONFIG
except:
    FACE_DETECTION_CONFIG = {'method': 'opencv', 'min_face_size': 20}


class FaceDetector:
    def __init__(self, method='opencv'):
        """
        Initialize face detector
        
        Args:
            method: 'mtcnn' or 'opencv'
        """
        self.method = method
        
        if method == 'mtcnn':
            if not MTCNN_AVAILABLE:
                print("MTCNN not available, falling back to OpenCV")
                self.method = 'opencv'
                self._init_opencv()
            else:
                try:
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.detector = MTCNN(
                        keep_all=False,
                        min_face_size=FACE_DETECTION_CONFIG.get('min_face_size', 20),
                        device=device
                    )
                except Exception as e:
                    print(f"Error initializing MTCNN: {e}")
                    print("Falling back to OpenCV")
                    self.method = 'opencv'
                    self._init_opencv()
        else:
            self._init_opencv()
    
    def _init_opencv(self):
        """Initialize OpenCV face detector"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise ValueError("Failed to load Haar Cascade classifier")
    
    def detect_face(self, image):
        """
        Detect face in image
        
        Args:
            image: numpy array (BGR format)
            
        Returns:
            face_bbox: Bounding box [x, y, w, h] or None
            face_img: Cropped face image or None
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if self.method == 'mtcnn':
            return self._detect_mtcnn(image)
        else:
            return self._detect_opencv(image)
    
    def _detect_mtcnn(self, image):
        """Detect face using MTCNN"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        try:
            boxes, _ = self.detector.detect(Image.fromarray(image_rgb))
            
            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                x, y, x2, y2 = box
                w, h = x2 - x, y2 - y
                
                x = max(0, x)
                y = max(0, y)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                face_img = image[y:y2, x:x2]
                return [x, y, w, h], face_img
        except Exception as e:
            print(f"MTCNN detection error: {e}")
        
        return None, None
    
    def _detect_opencv(self, image):
        """Detect face using OpenCV Haar Cascade"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Return the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            x = max(0, x)
            y = max(0, y)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)
            
            face_img = image[y:y2, x:x2]
            return [x, y, w, h], face_img
        
        return None, None
