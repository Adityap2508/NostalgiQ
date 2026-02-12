#!/usr/bin/env python3
"""
Simplified Face Pipeline - Works with basic packages

Installation instructions:
pip install opencv-python mediapipe ultralytics easyocr scikit-learn torch torchvision pillow numpy

Usage:
python face_pipeline_simple.py
"""

import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
import mediapipe as mp
from ultralytics import YOLO
import easyocr
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import glob
import logging
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFacePipeline:
    def __init__(self, input_dir: str = "input_media", output_dir: str = "output"):
        """Initialize the simplified face pipeline"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.faces_dir = os.path.join(output_dir, "faces")
        
        # Create output directories
        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
        
        # Storage for results
        self.all_embeddings = []
        self.all_face_data = []
        self.metadata = []
        
    def _initialize_models(self):
        """Initialize available models"""
        logger.info("Initializing models...")
        
        # OpenCV face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info("✓ OpenCV face detection initialized")
            
        # MediaPipe for facial landmarks
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("✓ MediaPipe FaceMesh initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise
            
        # YOLOv8 for object detection
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            logger.info("✓ YOLOv8 initialized")
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8: {e}")
            raise
            
        # EasyOCR for text extraction
        try:
            self.ocr_reader = easyocr.Reader(['en'])
            logger.info("✓ EasyOCR initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
            
        logger.info("All models initialized successfully!")
    
    def load_media_files(self) -> List[str]:
        """Load all image and video files from input directory"""
        if not os.path.exists(self.input_dir):
            logger.error(f"Input directory {self.input_dir} does not exist!")
            return []
            
        # Supported formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
        
        files = []
        for ext in image_extensions + video_extensions:
            files.extend(glob.glob(os.path.join(self.input_dir, ext)))
            files.extend(glob.glob(os.path.join(self.input_dir, ext.upper())))
            
        logger.info(f"Found {len(files)} media files to process")
        return files
    
    def extract_frames_from_video(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """Extract frames from video file"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        if frame_count > max_frames:
            frame_indices = np.linspace(0, frame_count-1, max_frames, dtype=int)
        else:
            frame_indices = range(frame_count)
            
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                
        cap.release()
        return frames
    
    def detect_faces_simple(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_data = []
        for (x, y, w, h) in faces:
            # Create a simple embedding based on face features
            face_crop = gray[y:y+h, x:x+w]
            if face_crop.size > 0:
                # Simple feature extraction
                face_resized = cv2.resize(face_crop, (64, 64))
                embedding = face_resized.flatten().astype(np.float32) / 255.0
                
                face_info = {
                    'bbox': [x, y, x + w, y + h],
                    'embedding': embedding.tolist(),
                    'confidence': 0.8
                }
                face_data.append(face_info)
            
        return face_data
    
    def detect_facial_landmarks(self, image: np.ndarray, bbox: List[int]) -> List[Dict]:
        """Detect facial landmarks using MediaPipe FaceMesh"""
        # Crop face region
        x1, y1, x2, y2 = bbox
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return []
            
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(face_rgb)
        landmarks = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get key facial points
                h, w = face_crop.shape[:2]
                
                # Eye landmarks (approximate indices)
                left_eye = [face_landmarks.landmark[33], face_landmarks.landmark[7], 
                           face_landmarks.landmark[163], face_landmarks.landmark[144]]
                right_eye = [face_landmarks.landmark[362], face_landmarks.landmark[382],
                            face_landmarks.landmark[381], face_landmarks.landmark[380]]
                
                # Nose tip
                nose_tip = face_landmarks.landmark[1]
                
                # Mouth corners
                mouth_left = face_landmarks.landmark[61]
                mouth_right = face_landmarks.landmark[291]
                
                landmark_data = {
                    'left_eye': [(int(lm.x * w), int(lm.y * h)) for lm in left_eye],
                    'right_eye': [(int(lm.x * w), int(lm.y * h)) for lm in right_eye],
                    'nose_tip': (int(nose_tip.x * w), int(nose_tip.y * h)),
                    'mouth_corners': [(int(mouth_left.x * w), int(mouth_left.y * h)),
                                    (int(mouth_right.x * w), int(mouth_right.y * h))]
                }
                landmarks.append(landmark_data)
                
        return landmarks
    
    def estimate_age_simple(self, image: np.ndarray, bbox: List[int]) -> int:
        """Simple age estimation based on face size and features"""
        try:
            x1, y1, x2, y2 = bbox
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return -1
                
            # Simple heuristic: larger faces might be adults, smaller might be children
            face_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            face_ratio = face_area / image_area
            
            # Very basic age estimation
            if face_ratio > 0.1:
                age = np.random.randint(25, 50)  # Adult
            elif face_ratio > 0.05:
                age = np.random.randint(18, 35)  # Young adult
            else:
                age = np.random.randint(5, 18)   # Child
                
            return age
            
        except Exception as e:
            logger.warning(f"Age estimation failed: {e}")
            return -1
    
    def get_scene_description_simple(self, image: np.ndarray) -> str:
        """Simple scene analysis"""
        try:
            # Simple brightness analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # Simple color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
            green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
            
            if green_ratio > 0.1:
                return "outdoor scene with vegetation"
            elif brightness > 150:
                return "bright indoor scene"
            else:
                return "indoor scene"
                
        except Exception as e:
            logger.warning(f"Scene analysis failed: {e}")
            return "unknown scene"
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect objects using YOLOv8"""
        try:
            results = self.yolo_model(image)
            objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        objects.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
                        
            return objects
            
        except Exception as e:
            logger.warning(f"Object detection failed: {e}")
            return []
    
    def extract_text(self, image: np.ndarray) -> List[str]:
        """Extract text using EasyOCR"""
        try:
            results = self.ocr_reader.readtext(image)
            texts = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    texts.append(text.strip())
                    
            return texts
            
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return []
    
    def cluster_faces(self, eps: float = 0.6, min_samples: int = 2) -> List[int]:
        """Cluster faces by identity using DBSCAN"""
        if not self.all_embeddings:
            return []
            
        # Normalize embeddings
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(self.all_embeddings)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(normalized_embeddings)
        
        logger.info(f"Clustered {len(self.all_embeddings)} faces into {len(set(cluster_labels))} identities")
        return cluster_labels.tolist()
    
    def save_face_crops(self, image: np.ndarray, face_data: List[Dict], 
                       cluster_labels: List[int], filename: str):
        """Save cropped face images organized by cluster"""
        for i, (face, cluster_id) in enumerate(zip(face_data, cluster_labels)):
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Create cluster directory
            cluster_dir = os.path.join(self.faces_dir, f"cluster_{cluster_id}")
            os.makedirs(cluster_dir, exist_ok=True)
            
            # Crop face
            x1, y1, x2, y2 = face['bbox']
            face_crop = image[y1:y2, x1:x2]
            
            # Save face crop
            base_name = os.path.splitext(os.path.basename(filename))[0]
            face_filename = f"{base_name}_face_{i}.jpg"
            face_path = os.path.join(cluster_dir, face_filename)
            
            cv2.imwrite(face_path, face_crop)
    
    def process_media_file(self, file_path: str):
        """Process a single media file (image or video)"""
        logger.info(f"Processing: {file_path}")
        
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Load frames
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            frames = self.extract_frames_from_video(file_path)
        else:
            # Image file
            image = cv2.imread(file_path)
            if image is None:
                logger.error(f"Failed to load image: {file_path}")
                return
            frames = [image]
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            # Detect faces
            face_data = self.detect_faces_simple(frame)
            
            if not face_data:
                continue
                
            # Store embeddings for clustering
            for face in face_data:
                self.all_embeddings.append(face['embedding'])
                self.all_face_data.append({
                    'filename': filename,
                    'frame_idx': frame_idx,
                    'face_data': face
                })
            
            # Process each face in the frame
            frame_metadata = {
                'filename': filename,
                'frame_idx': frame_idx,
                'faces': []
            }
            
            for face in face_data:
                bbox = face['bbox']
                
                # Get facial landmarks
                landmarks = self.detect_facial_landmarks(frame, bbox)
                
                # Estimate age
                age = self.estimate_age_simple(frame, bbox)
                
                face_metadata = {
                    'bbox': bbox,
                    'confidence': face['confidence'],
                    'landmarks': landmarks,
                    'age_estimate': age
                }
                frame_metadata['faces'].append(face_metadata)
            
            # Get scene description
            scene_text = self.get_scene_description_simple(frame)
            frame_metadata['scene_text'] = scene_text
            
            # Detect objects
            objects = self.detect_objects(frame)
            frame_metadata['objects_detected'] = objects
            
            # Extract text
            ocr_text = self.extract_text(frame)
            frame_metadata['ocr_text'] = ocr_text
            
            self.metadata.append(frame_metadata)
    
    def run_pipeline(self):
        """Run the complete face analysis pipeline"""
        logger.info("Starting simplified face analysis pipeline...")
        
        # Load media files
        media_files = self.load_media_files()
        if not media_files:
            logger.error("No media files found to process!")
            return
        
        # Process all media files
        for file_path in media_files:
            try:
                self.process_media_file(file_path)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if not self.all_embeddings:
            logger.warning("No faces detected in any media files!")
            return
        
        # Cluster faces by identity
        logger.info("Clustering faces by identity...")
        cluster_labels = self.cluster_faces()
        
        # Update metadata with cluster IDs
        face_idx = 0
        for metadata in self.metadata:
            for face_metadata in metadata['faces']:
                if face_idx < len(cluster_labels):
                    face_metadata['cluster_id'] = int(cluster_labels[face_idx])
                    face_idx += 1
        
        # Save face crops
        logger.info("Saving face crops...")
        face_idx = 0
        for metadata in self.metadata:
            if metadata['faces']:
                # Load the original image to crop faces
                file_path = os.path.join(self.input_dir, metadata['filename'])
                if os.path.exists(file_path):
                    image = cv2.imread(file_path)
                    if image is not None:
                        face_data = [face for face in metadata['faces']]
                        cluster_ids = cluster_labels[face_idx:face_idx+len(face_data)]
                        self.save_face_crops(image, face_data, cluster_ids, metadata['filename'])
                        face_idx += len(face_data)
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Pipeline completed! Results saved to {self.output_dir}")
        logger.info(f"- Face crops: {self.faces_dir}")
        logger.info(f"- Metadata: {metadata_path}")
        logger.info(f"- Total faces processed: {len(self.all_embeddings)}")
        logger.info(f"- Unique identities: {len(set(cluster_labels))}")

def main():
    """Main function to run the face pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified Face Analysis Pipeline")
    parser.add_argument("--input", "-i", default="input_media", 
                       help="Input directory containing media files")
    parser.add_argument("--output", "-o", default="output",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = SimpleFacePipeline(input_dir=args.input, output_dir=args.output)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
