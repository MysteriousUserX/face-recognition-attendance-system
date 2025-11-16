"""
GUI for Face Recognition Attendance System
Integrates: Face Verification + Anti-Spoofing + Emotion Detection
"""

import sys
import cv2
import torch
import numpy as np
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLabel, QComboBox,
                                 QTextEdit, QGroupBox, QFileDialog, QInputDialog,
                                 QMessageBox, QLineEdit)
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QImage, QPixmap
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("Warning: PyQt5 not installed. GUI will not work.")

from models.metric_learning_model import MetricLearningModel
from models.classification_model import ClassificationModel
from models.liveness_detection import create_liveness_detector
from models.emotion_model import create_emotion_detector
from utils.face_detector import FaceDetector
from config import MODEL_CONFIG, METRIC_LEARNING_CONFIG, CLASSIFICATION_CONFIG


class AttendanceGUI(QMainWindow):
    """Main GUI for attendance system"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Attendance System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.emotion_detector = create_emotion_detector(model_type='deepface')
        self.anti_spoofing_detector = create_liveness_detector()  # DeepPixBiS liveness detector
        
        # Load face recognition model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_type = None  # 'metric' or 'classification'
        self.registered_faces = {}  # {name: embedding}
        self.database_path = "data/registered_faces/database.pth"
        
        # Camera
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Registration mode
        self.registration_mode = False
        self.registration_name = None
        self.registration_embeddings = []
        self.registration_frame_skip = 0  # Skip frames between captures for diversity
        self.required_captures = 5
        
        # Verification mode
        self.verification_mode = False
        
        # Attendance log
        self.attendance_log = []
        self.attendance_today = set()  # Track who already checked in today
        
        # Setup UI
        self.init_ui()
        
        # Auto-load metric learning model
        self.auto_load_metric_model()
        
        # Load database after UI is initialized
        self.load_database()
        
    def init_ui(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Camera view
        left_panel = QVBoxLayout()
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid black;")
        left_panel.addWidget(self.camera_label)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn = QPushButton("Stop Camera")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)
        
        camera_controls.addWidget(self.start_camera_btn)
        camera_controls.addWidget(self.stop_camera_btn)
        left_panel.addLayout(camera_controls)
        
        # Registration and Verification buttons
        action_controls = QHBoxLayout()
        self.register_btn = QPushButton("Register New Employee")
        self.register_btn.clicked.connect(self.start_registration)
        self.register_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        
        self.verify_btn = QPushButton("Verify & Check In")
        self.verify_btn.clicked.connect(self.start_verification)
        self.verify_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.verify_btn.setEnabled(False)  # Enable after camera starts
        
        action_controls.addWidget(self.register_btn)
        action_controls.addWidget(self.verify_btn)
        left_panel.addLayout(action_controls)
        
        # Employee Management button
        manage_controls = QHBoxLayout()
        self.manage_btn = QPushButton("Manage Registered Employees")
        self.manage_btn.clicked.connect(self.manage_employees)
        self.manage_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        
        manage_controls.addWidget(self.manage_btn)
        left_panel.addLayout(manage_controls)
        
        # Right panel - Controls and info
        right_panel = QVBoxLayout()
        
        # Model status (removed selection - auto-loads metric model)
        model_group = QGroupBox("Model Status")
        model_layout = QVBoxLayout()
        
        self.model_status_label = QLabel("Model: Metric Learning (Auto-loaded)")
        self.model_status_label.setStyleSheet("font-weight: bold; color: green;")
        model_layout.addWidget(self.model_status_label)
        
        model_group.setLayout(model_layout)
        right_panel.addWidget(model_group)
        
        # Features group
        features_group = QGroupBox("Features")
        features_layout = QVBoxLayout()
        
        self.antispoofing_check = QPushButton("Enable Anti-Spoofing")
        self.antispoofing_check.setCheckable(True)
        self.antispoofing_check.setChecked(True)
        
        self.emotion_check = QPushButton("Enable Emotion Detection")
        self.emotion_check.setCheckable(True)
        self.emotion_check.setChecked(True)
        
        features_layout.addWidget(self.antispoofing_check)
        features_layout.addWidget(self.emotion_check)
        features_group.setLayout(features_layout)
        right_panel.addWidget(features_group)
        
        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        self.name_label = QLabel("Name: -")
        self.confidence_label = QLabel("Confidence: -")
        self.liveness_label = QLabel("Liveness: -")
        self.emotion_label = QLabel("Emotion: -")
        
        status_layout.addWidget(self.name_label)
        status_layout.addWidget(self.confidence_label)
        status_layout.addWidget(self.liveness_label)
        status_layout.addWidget(self.emotion_label)
        
        status_group.setLayout(status_layout)
        right_panel.addWidget(status_group)
        
        # Attendance log
        log_group = QGroupBox("Attendance Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_controls = QHBoxLayout()
        self.save_log_btn = QPushButton("Save Log")
        self.save_log_btn.clicked.connect(self.save_log)
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        
        log_controls.addWidget(self.save_log_btn)
        log_controls.addWidget(self.clear_log_btn)
        log_layout.addLayout(log_controls)
        
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
    def auto_load_metric_model(self):
        """Automatically load metric learning model on startup (128-dim embeddings)"""
        try:
            # Use the newly trained 128-dim metric learning model
            embed_size = METRIC_LEARNING_CONFIG.get('embedding_size', 128)
            checkpoint_path = f"checkpoints/metric_learning_embed{embed_size}/best_metric_learning_model.pth"
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model not found at {checkpoint_path}")
            
            # Load metric learning model
            self.model = MetricLearningModel(
                embedding_size=embed_size,
                backbone=METRIC_LEARNING_CONFIG['backbone'],
                pretrained=False
            ).to(self.device)
            self.model_type = "metric"
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.model_status_label.setText(f"Model: Metric Learning ({embed_size}-dim) ✓")
            self.model_status_label.setStyleSheet("font-weight: bold; color: green;")
            print(f"Metric Learning model loaded successfully ({embed_size}-dim embeddings)")
            print(f"Checkpoint: {checkpoint_path}")
            
        except Exception as e:
            self.model_status_label.setText(f"Model: Error - {str(e)[:30]}")
            self.model_status_label.setStyleSheet("font-weight: bold; color: red;")
            print(f"Error loading model: {str(e)}")

    
    def start_camera(self):
        """Start camera capture"""
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened():
            self.timer.start(30)  # 30ms = ~33 FPS
            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)
            self.verify_btn.setEnabled(True)
            self.status_label.setText("Camera Active")
        else:
            self.status_label.setText("Error: Cannot open camera")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.timer.stop()
        if self.camera:
            self.camera.release()
        self.camera_label.clear()
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.verify_btn.setEnabled(False)
        self.registration_mode = False
        self.status_label.setText("Camera Stopped")
    
    def load_database(self):
        """Load registered faces database"""
        if os.path.exists(self.database_path):
            try:
                self.registered_faces = torch.load(self.database_path)
                self.log_message(f"Loaded {len(self.registered_faces)} registered faces")
            except Exception as e:
                self.log_message(f"Error loading database: {str(e)}")
                self.registered_faces = {}
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            self.registered_faces = {}
    
    def save_database(self):
        """Save registered faces database"""
        try:
            torch.save(self.registered_faces, self.database_path)
            self.log_message(f"Database saved ({len(self.registered_faces)} faces)")
        except Exception as e:
            self.log_message(f"Error saving database: {str(e)}")
    
    def start_registration(self):
        """Start registration process"""
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Please load a model first!")
            return
        
        if not self.camera or not self.camera.isOpened():
            QMessageBox.warning(self, "No Camera", "Please start the camera first!")
            return
        
        # Get employee name
        name, ok = QInputDialog.getText(self, "Register New Employee", 
                                       "Enter employee name:")
        
        if ok and name:
            # Show instructions
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Registration Started")
            msg.setText(f"Registering: {name}")
            msg.setInformativeText(
                "INSTRUCTIONS:\n\n"
                f"1. System will capture {self.required_captures} photos\n"
                "2. Look straight at the camera\n"
                "3. Keep your face in the GREEN box\n"
                "4. Maintain good lighting\n\n"
                "IMPORTANT: Only REAL faces accepted!\n"
                "   Photos/screens will be rejected.\n\n"
                "Click OK to start registration."
            )
            msg.exec_()  # Wait for user to close dialog
            
            # Reset emotion smoothing for new person
            if self.emotion_detector:
                self.emotion_detector.reset_smoothing()
            
            # Now activate registration mode
            self.registration_mode = True
            self.registration_name = name.strip()
            self.registration_embeddings = []
            self.registration_frame_skip = 0
            self.status_label.setText(f"REGISTRATION MODE: {self.registration_name}")
            self.log_message(f"Started registration for: {self.registration_name}")
    
    def start_verification(self):
        """Start verification/check-in process"""
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Please load a model first!")
            return
        
        if not self.camera or not self.camera.isOpened():
            QMessageBox.warning(self, "No Camera", "Camera is not active!")
            return
        
        if len(self.registered_faces) == 0:
            QMessageBox.warning(self, "No Registered Faces", 
                              "No employees registered yet!\n"
                              "Please register employees first.")
            return
        
        # Toggle verification mode
        self.verification_mode = not self.verification_mode
        
        if self.verification_mode:
            # Reset emotion smoothing when starting verification
            if self.emotion_detector:
                self.emotion_detector.reset_smoothing()
            
            self.verify_btn.setStyleSheet("background-color: #FF5722; color: white; font-weight: bold; padding: 10px;")
            self.verify_btn.setText("Stop Verification")
            self.status_label.setText("VERIFICATION MODE: Present your face")
            self.log_message("Verification mode activated")
        else:
            self.verify_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
            self.verify_btn.setText("Verify & Check In")
            self.status_label.setText("Verification mode deactivated")
            self.log_message("Verification mode deactivated")
    
    def manage_employees(self):
        """Manage registered employees - view and delete"""
        if len(self.registered_faces) == 0:
            QMessageBox.information(self, "No Employees", 
                                   "No employees registered yet!\n\n"
                                   "Use 'Register New Employee' to add employees.")
            return
        
        # Create employee list dialog
        from PyQt5.QtWidgets import QDialog, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Registered Employees")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"Total Registered Employees: {len(self.registered_faces)}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Employee list
        list_widget = QListWidget()
        for name in sorted(self.registered_faces.keys()):
            item = QListWidgetItem(f"{name}")
            list_widget.addItem(item)
        layout.addWidget(list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        delete_btn = QPushButton("Delete Selected")
        delete_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        
        delete_all_btn = QPushButton("Delete All")
        delete_all_btn.setStyleSheet("background-color: #d32f2f; color: white; padding: 8px;")
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("background-color: #757575; color: white; padding: 8px;")
        
        button_layout.addWidget(delete_btn)
        button_layout.addWidget(delete_all_btn)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        # Button actions
        def delete_selected():
            current_item = list_widget.currentItem()
            if current_item:
                name = current_item.text()
                
                reply = QMessageBox.question(dialog, "Confirm Delete",
                                            f"Delete employee: {name}?",
                                            QMessageBox.Yes | QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    del self.registered_faces[name]
                    self.attendance_today.discard(name)
                    self.save_database()
                    list_widget.takeItem(list_widget.row(current_item))
                    title.setText(f"Total Registered Employees: {len(self.registered_faces)}")
                    self.log_message(f"Deleted employee: {name}")
                    
                    if len(self.registered_faces) == 0:
                        QMessageBox.information(dialog, "No Employees",
                                              "All employees deleted.\n"
                                              "Closing manager...")
                        dialog.close()
        
        def delete_all():
            reply = QMessageBox.warning(dialog, "Confirm Delete All",
                                       f"Delete ALL {len(self.registered_faces)} employees?\n\n"
                                       "This action cannot be undone!",
                                       QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                count = len(self.registered_faces)
                self.registered_faces.clear()
                self.attendance_today.clear()
                self.save_database()
                self.log_message(f"Deleted all {count} employees")
                QMessageBox.information(dialog, "Deleted",
                                       f"All {count} employees deleted.")
                dialog.close()
        
        delete_btn.clicked.connect(delete_selected)
        delete_all_btn.clicked.connect(delete_all)
        close_btn.clicked.connect(dialog.close)
        
        # Show dialog
        dialog.exec_()
    
    def update_frame(self):
        """Update camera frame and handle registration/verification"""
        ret, frame = self.camera.read()
        if not ret:
            return
        
        # Detect face
        bbox, face_img = self.face_detector.detect_face(frame)
        
        if bbox is not None and face_img is not None:
            x, y, w, h = bbox
            
            # ===== STEP 1: LIVENESS CHECK (Anti-Spoofing) =====
            is_real = True
            liveness_score = 1.0
            if self.antispoofing_check.isChecked():
                spoof_result = self.anti_spoofing_detector.predict(face_img)
                is_real = spoof_result['is_real']
                liveness_score = spoof_result['confidence']
                self.liveness_label.setText(f"Liveness: {liveness_score:.2f} ({'Real' if is_real else 'FAKE'})")
            
            # If face is FAKE, reject immediately
            if not is_real:
                color = (0, 0, 255)  # Red for fake
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, "FAKE FACE DETECTED!", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                self.status_label.setText("LIVENESS FAILED - SPOOFED FACE")
                self.name_label.setText("Name: REJECTED")
                self.confidence_label.setText("Confidence: -")
                self.emotion_label.setText("Emotion: -")
            
            # If face is REAL, proceed with registration or verification
            else:
                color = (0, 255, 0)  # Green for real
                
                # Detect emotion (always, if enabled)
                emotion = "N/A"
                emotion_conf = 0.0
                if self.emotion_check.isChecked():
                    emotion_result = self.emotion_detector.detect_emotion(face_img)
                    emotion = emotion_result['emotion']
                    emotion_conf = emotion_result['confidence']
                    self.emotion_label.setText(f"Emotion: {emotion} ({emotion_conf:.2f})")
                
                # ===== REGISTRATION MODE =====
                if self.registration_mode:
                    # Skip frames to get more diverse captures (capture every 15 frames = ~0.5 seconds)
                    self.registration_frame_skip += 1
                    
                    if self.registration_frame_skip >= 15:
                        self.registration_frame_skip = 0
                        
                        # Extract embedding
                        embedding = self.extract_embedding(face_img)
                        
                        if embedding is not None:
                            # Quality check: ensure face is large enough
                            face_area = w * h
                            frame_area = frame.shape[0] * frame.shape[1]
                            face_ratio = face_area / frame_area
                            
                            if face_ratio < 0.05:  # Face too small
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 3)  # Orange
                                cv2.putText(frame, "Move closer to camera!", (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                                self.status_label.setText(f"⚠️ Face too small - move closer!")
                            else:
                                # Good capture!
                                self.registration_embeddings.append(embedding)
                                remaining = self.required_captures - len(self.registration_embeddings)
                                
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                                cv2.putText(frame, f"Captured {len(self.registration_embeddings)}/{self.required_captures}", 
                                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                
                                if remaining > 0:
                                    instruction = self._get_registration_instruction(len(self.registration_embeddings))
                                    self.status_label.setText(f"REGISTRATION: {self.registration_name} - {instruction}")
                                    self.log_message(f"  Captured {len(self.registration_embeddings)}/{self.required_captures}")
                                
                                # If we have enough embeddings, complete registration
                                if len(self.registration_embeddings) >= self.required_captures:
                                    # Average the embeddings
                                    avg_embedding = torch.stack(self.registration_embeddings).mean(dim=0)
                                    self.registered_faces[self.registration_name] = avg_embedding
                                    self.save_database()
                                    
                                    self.registration_mode = False
                                    self.status_label.setText(f"Registration Complete: {self.registration_name}")
                                    self.log_message(f"Successfully registered: {self.registration_name}")
                                    
                                    QMessageBox.information(self, "Registration Complete",
                                                          f"Successfully registered: {self.registration_name}\n\n"
                                                          f"Captured {len(self.registration_embeddings)} high-quality images\n"
                                                          f"Total registered employees: {len(self.registered_faces)}\n\n"
                                                          f"You can now use 'Verify & Check In' to test recognition!")
                    else:
                        # Waiting for next capture
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        progress = int((self.registration_frame_skip / 15) * 100)
                        cv2.putText(frame, f"Ready in {15 - self.registration_frame_skip} frames...", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # ===== VERIFICATION MODE =====
                else:
                    # Only verify if verification mode is active
                    if not self.verification_mode:
                        # Just show the face with green box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, "Real Face", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        # Extract embedding and verify
                        embedding = self.extract_embedding(face_img)
                    
                        if embedding is not None and len(self.registered_faces) > 0:
                            # Find best match
                            best_match = None
                            best_similarity = -1
                            
                            for name, stored_embedding in self.registered_faces.items():
                                # Cosine similarity
                                similarity = torch.nn.functional.cosine_similarity(
                                    embedding.unsqueeze(0), 
                                    stored_embedding.unsqueeze(0)
                                ).item()
                                
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = name
                            
                            # Threshold for match
                            threshold = 0.6
                            
                            if best_similarity > threshold:
                                # ===== MATCH FOUND =====
                                # Update display
                                self.name_label.setText(f"Name: {best_match}")
                                self.confidence_label.setText(f"Similarity: {best_similarity:.4f}")
                                
                                # Check if already checked in today
                                if best_match not in self.attendance_today:
                                    self.attendance_today.add(best_match)
                                    self.log_message(f"CHECK IN: {best_match} | Emotion: {emotion} | Similarity: {best_similarity:.4f}")
                                    self.status_label.setText(f"Welcome, {best_match}!")
                                else:
                                    self.status_label.setText(f"Already checked in: {best_match}")
                                
                                # Draw on frame
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                                info_text = f"{best_match} | {emotion}"
                                cv2.putText(frame, info_text, (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
                            else:
                                self.name_label.setText("Name: UNKNOWN")
                                self.confidence_label.setText(f"Similarity: {best_similarity:.4f} (too low)")
                                self.emotion_label.setText("Emotion: -")
                                self.status_label.setText("User not found. Please register.")
                                
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 3)  # Orange
                                cv2.putText(frame, "UNKNOWN - Please Register", (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            # No face detected
            self.liveness_label.setText("Liveness: -")
            self.emotion_label.setText("Emotion: -")
            self.name_label.setText("Name: -")
            self.confidence_label.setText("Confidence: -")
        
        # Convert to QPixmap and display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio))
    
    def extract_embedding(self, face_img):
        """Extract embedding from face image"""
        if self.model is None:
            return None
        
        try:
            # Preprocess face
            face_resized = cv2.resize(face_img, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Extract embedding
            with torch.no_grad():
                if self.model_type == "classification":
                    # Classification model needs return_embedding=True
                    embedding = self.model(face_tensor.to(self.device), return_embedding=True)
                else:
                    # Metric learning model returns embedding directly
                    embedding = self.model(face_tensor.to(self.device))
            
            return embedding.cpu().squeeze()
        
        except Exception as e:
            self.log_message(f"Error extracting embedding: {str(e)}")
            return None
    
    def _get_registration_instruction(self, capture_num):
        """Get instruction for current registration capture"""
        instructions = [
            "Look straight at camera",
            "Turn head slightly LEFT",
            "Turn head slightly RIGHT", 
            "Tilt head slightly UP",
            "Tilt head slightly DOWN"
        ]
        if capture_num < len(instructions):
            return instructions[capture_num]
        return "Almost done!"
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        self.attendance_log.append(log_entry)
    
    def save_log(self):
        """Save attendance log to file"""
        if not self.attendance_log:
            self.status_label.setText("No log to save")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "attendance_log.txt", "Text Files (*.txt)"
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write('\n'.join(self.attendance_log))
            self.status_label.setText(f"Log saved to {filename}")
    
    def clear_log(self):
        """Clear attendance log"""
        self.log_text.clear()
        self.attendance_log.clear()
        self.status_label.setText("Log cleared")
    
    def closeEvent(self, event):
        """Clean up when closing"""
        if self.camera:
            self.camera.release()
        event.accept()


def main():
    """Main function"""
    if not PYQT_AVAILABLE:
        print("Error: PyQt5 is required for GUI")
        print("Install with: pip install PyQt5")
        return
    
    app = QApplication(sys.argv)
    window = AttendanceGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
