# Face Recognition Attendance System

**COS30082 - Applied Machine Learning - Final Project**

A face recognition attendance system with anti-spoofing and emotion detection capabilities.

## Features

- **Face Recognition**: Metric learning (Triplet Loss) vs Classification approach
- **Anti-Spoofing**: DeepPixBiS liveness detection to prevent photo/video attacks
- **Emotion Detection**: Real-time emotion recognition using DeepFace
- **GUI Interface**: PyQt5-based application for easy deployment

## Requirements

- **Python 3.10 or lower** (Required for TensorFlow compatibility)
- PyTorch, OpenCV, PyQt5, DeepFace, TensorFlow, ONNX Runtime

```bash
pip install -r requirements.txt
```

## Project Structure

```
Final_project/
├── models/
│   ├── classification_model.py      # ResNet-50 classification
│   ├── metric_learning_model.py     # Triplet loss metric learning
│   ├── liveness_detection.py        # DeepPixBiS anti-spoofing
│   └── emotion_model.py             # DeepFace emotion recognition
├── utils/
│   ├── data_loader.py               # Dataset handling
│   ├── triplet_mining.py            # Triplet loss implementation
│   ├── metrics.py                   # ROC/AUC evaluation
│   └── face_detector.py             # Face detection
├── gui/
│   └── attendance_gui.py            # Main GUI application
├── train_classification.py          # Train classification model
├── train_metric_learning.py         # Train metric learning model
├── evaluate_verification.py         # Evaluate and compare models
├── test_anti_spoofing.py            # Test liveness detection
├── test_emotion.py                  # Test emotion detection
├── config.py                        # Configuration settings
└── checkpoints/                     # Trained models (auto-created)
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv310
.\venv310\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train metric learning model (recommended)
python train_metric_learning.py

# Train classification model (optional)
python train_classification.py
```

### 3. Evaluate Models

```bash
python evaluate_verification.py
```

Select option 3 to compare both models.

### 4. Run GUI

```bash
python gui/attendance_gui.py
```

Or use the startup script:

```bash
.\start_gui.ps1  # Windows PowerShell
```

## GUI Usage

1. **Start Camera**: Activate webcam
2. **Register Employee**: Enter name, capture 5 photos (anti-spoofing verified)
3. **Verify & Check In**: Verify identity with liveness detection and emotion tracking
4. **Manage Employees**: View/delete registered faces

## Results

| Model               | AUC Score | Embedding Size |
| ------------------- | --------- | -------------- |
| **Metric Learning** | 0.8898    | 128-dim        |
| Classification      | 0.8662    | 128-dim        |

**Winner**: Metric Learning (better performance)

## Models

- **Backbone**: ResNet-50 pretrained on ImageNet
- **Anti-Spoofing**: DeepPixBiS (ONNX, auto-downloads on first run)
- **Emotion**: DeepFace (7 emotions)

## Configuration

Edit `config.py` to customize:

- Embedding size, learning rate, batch size
- Anti-spoofing threshold (default: 0.3)
- Training epochs, early stopping patience

## Troubleshooting

**Python Version**: Must use Python ≤ 3.10 (TensorFlow requirement)

**CUDA Out of Memory**: Reduce batch size in `config.py`

**Camera Not Working**: Check permissions and camera index (default: 0)

---

**Author**: Duc Tam Nguyen, COS30082 - Applied Machine Learning, 2025
