"""
Emotion Detection Model
Uses DeepFace library with pre-trained models
Enhanced with temporal smoothing for consistent predictions
"""

import numpy as np
import cv2
from collections import deque

# Import DeepFace library
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    raise ImportError("DeepFace is required. Install with: pip install deepface tensorflow")


class EmotionDetector:
    """Emotion detection using DeepFace pre-trained model with temporal smoothing"""
    
    def __init__(self, model_type='deepface', device='cpu', smoothing_window=5):
        self.device = device
        self.model_type = 'deepface'
        self.smoothing_window = smoothing_window
        
        # Emotion labels
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Temporal smoothing: store recent predictions
        self.emotion_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)
        
        # Frame skip counter for performance
        self.frame_count = 0
        self.skip_frames = 2  # Process every 3rd frame
        self.last_result = None
        
        # Verify DeepFace is available
        if not DEEPFACE_AVAILABLE:
            raise ImportError("DeepFace is required but not installed. Run: pip install deepface tensorflow")
        
        print("Using DeepFace for emotion detection (with temporal smoothing)")
    
    def detect_emotion(self, face_img):
        """
        Detect emotion from face image using DeepFace with temporal smoothing
        
        Args:
            face_img: BGR image (numpy array)
            
        Returns:
            dict with emotion prediction
        """
        # Skip frames for performance
        self.frame_count += 1
        if self.frame_count % (self.skip_frames + 1) != 0:
            # Return last result if available
            if self.last_result is not None:
                return self.last_result
        
        # Check face quality
        if not self._is_face_quality_good(face_img):
            # If face quality is poor, return last result or neutral
            if self.last_result is not None:
                return self.last_result
            return {
                'emotion': 'Neutral',
                'confidence': 0.5,
                'probabilities': {e: 1.0/7 for e in self.emotions}
            }
        
        # Detect emotion with DeepFace
        result = self._detect_with_deepface(face_img)
        
        # Apply temporal smoothing
        smoothed_result = self._apply_temporal_smoothing(result)
        
        # Store as last result
        self.last_result = smoothed_result
        
        return smoothed_result
    
    def _is_face_quality_good(self, face_img):
        """
        Check if face image quality is good enough for reliable detection
        
        Args:
            face_img: Face image
            
        Returns:
            bool: True if quality is good
        """
        # Check minimum size
        h, w = face_img.shape[:2]
        if h < 48 or w < 48:
            return False
        
        # Check brightness
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 30 or brightness > 225:  # Too dark or too bright
            return False
        
        # Check blur (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:  # Too blurry
            return False
        
        return True
    
    def _apply_temporal_smoothing(self, current_result):
        """
        Apply temporal smoothing to reduce flickering between emotions
        
        Args:
            current_result: Current emotion detection result
            
        Returns:
            Smoothed result
        """
        # Add current prediction to history
        self.emotion_history.append(current_result['emotion'])
        self.confidence_history.append(current_result['confidence'])
        
        # If not enough history, return current result
        if len(self.emotion_history) < 3:
            return current_result
        
        # Count emotion occurrences in recent history
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Get most common emotion
        most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Average confidence for the most common emotion
        relevant_confidences = [
            conf for emotion, conf in zip(self.emotion_history, self.confidence_history)
            if emotion == most_common_emotion
        ]
        avg_confidence = np.mean(relevant_confidences) if relevant_confidences else current_result['confidence']
        
        # If current emotion matches most common, boost confidence
        if current_result['emotion'] == most_common_emotion:
            final_confidence = min(avg_confidence * 1.1, 1.0)
        else:
            final_confidence = avg_confidence
        
        return {
            'emotion': most_common_emotion,
            'confidence': final_confidence,
            'probabilities': current_result['probabilities']
        }
    
    def reset_smoothing(self):
        """Reset temporal smoothing history (call when changing person)"""
        self.emotion_history.clear()
        self.confidence_history.clear()
        self.last_result = None
    
    def _detect_with_deepface(self, face_img):
        """Detect emotion using DeepFace library"""
        try:
            # Ensure face_img is in the right format (numpy array, BGR or RGB)
            if not isinstance(face_img, np.ndarray):
                face_img = np.array(face_img)
            
            # DeepFace.analyze with default settings
            result = DeepFace.analyze(
                img_path=face_img, 
                actions=['emotion'], 
                enforce_detection=False,  # Don't fail if face not detected
                silent=True,  # Suppress verbose output
                detector_backend='opencv'  # Use default opencv backend
            )
            
            # Result is a list if multiple faces, or dict if single face
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            elif not isinstance(result, dict):
                raise ValueError("DeepFace returned unexpected format")
            
            # Get emotion predictions (DeepFace returns percentages)
            emotions_dict = result['emotion']
            dominant_emotion = result['dominant_emotion']
            
            # Map DeepFace emotion names to our format (capitalize first letter)
            emotion_map = {
                'angry': 'Angry',
                'disgust': 'Disgust',
                'fear': 'Fear',
                'happy': 'Happy',
                'sad': 'Sad',
                'surprise': 'Surprise',
                'neutral': 'Neutral'
            }
            
            # Get the properly formatted emotion name
            emotion = emotion_map.get(dominant_emotion, dominant_emotion.capitalize())
            
            # Get confidence (convert from percentage to 0-1 range)
            confidence = emotions_dict.get(dominant_emotion, 0) / 100.0
            
            # Convert all probabilities to our format
            probabilities = {}
            for key, value in emotions_dict.items():
                formatted_key = emotion_map.get(key, key.capitalize())
                probabilities[formatted_key] = value / 100.0
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probabilities
            }
                
        except Exception as e:
            # DeepFace failed
            print(f"DeepFace error: {e}")
            # Return neutral as fallback
            return {
                'emotion': 'Neutral',
                'confidence': 0.5,
                'probabilities': {e: 1.0/7 for e in self.emotions}
            }


def create_emotion_detector(model_type='deepface', device='cpu'):
    """
    Factory function to create emotion detector
    Note: Only DeepFace is supported. Requires TensorFlow.
    """
    return EmotionDetector(model_type='deepface', device='cpu')


if __name__ == '__main__':
    # Test emotion detection
    print("Testing Emotion Detection with DeepFace...")
    
    detector = create_emotion_detector(model_type='deepface')
    
    # Create test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = detector.detect_emotion(test_img)
    print(f"\nDetected Emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("\nProbabilities:")
    for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {prob:.3f}")
