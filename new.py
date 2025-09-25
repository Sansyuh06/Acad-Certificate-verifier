#!/usr/bin/env python3
"""
Acad - Enhanced Certificate Authenticity Verification System
ML-powered verification with database integration and comprehensive forensic analysis

Requirements:
pip install flask pytesseract opencv-python pillow pyzbar qrcode[pil] numpy scikit-image pdf2image
pip install sqlalchemy tensorflow torch torchvision transformers sentence-transformers
pip install pandas matplotlib seaborn plotly dash psycopg2-binary sqlite3

System Requirements:
- Install Tesseract OCR on your system:
  Ubuntu: sudo apt install tesseract-ocr poppler-utils
  macOS: brew install tesseract poppler
  Windows: Download from https://github.com/tesseract-ocr/tesseract and install poppler

Usage:
python enhanced_app.py
Then open http://127.0.0.1:5000 in your browser
"""

from flask import Flask, render_template_string, request, jsonify, redirect, url_for
import os
import uuid
import base64
import io
import re
import time
import logging
import hashlib
from datetime import datetime, timedelta
from PIL import Image, ImageEnhance, ImageFilter, ExifTags, ImageDraw
import cv2
import numpy as np
import pytesseract
import qrcode
from pyzbar import pyzbar
import json
from werkzeug.utils import secure_filename
import tempfile
import subprocess
import sys
import sqlite3
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pickle
from pathlib import Path

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ML imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.utils import to_categorical
    import tensorflow.keras.backend as K
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from skimage import feature, measure, morphology
    from skimage.filters import threshold_otsu
    from skimage.feature import local_binary_pattern, hog
    from skimage.segmentation import clear_border
    ADVANCED_ANALYSIS = True
except ImportError:
    ADVANCED_ANALYSIS = False

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['DATABASE_URL'] = os.getenv('DATABASE_URL', 'sqlite:///certificates.db')
app.config['ML_MODELS_PATH'] = os.path.join(os.path.dirname(__file__), 'ml_models')

# Create models directory
os.makedirs(app.config['ML_MODELS_PATH'], exist_ok=True)

# Database setup
Base = declarative_base()
engine = create_engine(app.config['DATABASE_URL'])
SessionLocal = sessionmaker(bind=engine)

# Database Models
class CertificateRecord(Base):
    __tablename__ = 'certificates'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    file_hash = Column(String, index=True)
    name = Column(String)
    roll_no = Column(String)
    year = Column(String)
    institution = Column(String)
    course = Column(String)
    grade = Column(String)
    tampering_score = Column(Float)
    ml_authenticity_score = Column(Float)
    verdict = Column(String)
    ocr_confidence = Column(Float)
    image_quality = Column(String)
    processing_time = Column(Float)
    raw_text = Column(Text)
    features_json = Column(Text)
    ml_features_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    forensic_features = relationship("ForensicFeature", back_populates="certificate")
    similarity_matches = relationship("SimilarityMatch", back_populates="certificate")

class ForensicFeature(Base):
    __tablename__ = 'forensic_features'
    
    id = Column(Integer, primary_key=True)
    certificate_id = Column(Integer, ForeignKey('certificates.id'))
    feature_type = Column(String)  # 'texture', 'edge', 'color', 'noise'
    feature_vector = Column(Text)  # JSON serialized feature vector
    confidence = Column(Float)
    anomaly_score = Column(Float)
    
    certificate = relationship("CertificateRecord", back_populates="forensic_features")

class SimilarityMatch(Base):
    __tablename__ = 'similarity_matches'
    
    id = Column(Integer, primary_key=True)
    certificate_id = Column(Integer, ForeignKey('certificates.id'))
    matched_certificate_id = Column(Integer)
    similarity_score = Column(Float)
    match_type = Column(String)  # 'text', 'visual', 'combined'
    
    certificate = relationship("CertificateRecord", back_populates="similarity_matches")

class KnownInstitution(Base):
    __tablename__ = 'known_institutions'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    aliases = Column(Text)  # JSON array of alternative names
    country = Column(String)
    verification_patterns = Column(Text)  # JSON patterns for verification
    authentic_samples = Column(Integer, default=0)
    suspicious_samples = Column(Integer, default=0)

# Create tables
Base.metadata.create_all(bind=engine)

# In-memory storage with session expiry
sessions = {}
SESSION_TIMEOUT = 1800  # 30 minutes

@dataclass
class MLFeatures:
    """Data class for ML features extracted from certificates"""
    texture_features: np.ndarray
    edge_features: np.ndarray
    color_features: np.ndarray
    text_features: np.ndarray
    geometric_features: np.ndarray
    metadata_features: Dict

class CertificateMLAnalyzer:
    """ML-powered certificate analysis system"""
    
    def __init__(self, models_path: str):
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
        
        # Initialize models
        self.authenticity_model = None
        self.copy_move_model = None
        self.text_similarity_model = None
        self.anomaly_detector = None
        self.feature_scaler = None
        
        # Load pre-trained models if available
        self._load_models()
        
        # Initialize sentence transformer for text similarity
        if SKLEARN_AVAILABLE:
            try:
                self.text_similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.text_similarity_model = None

    def _load_models(self):
        """Load pre-trained ML models"""
        try:
            # Load authenticity classifier
            auth_model_path = self.models_path / "authenticity_model.pkl"
            if auth_model_path.exists():
                self.authenticity_model = joblib.load(auth_model_path)
                logger.info("Loaded authenticity model")
            
            # Load copy-move detection model
            copy_model_path = self.models_path / "copy_move_model.h5"
            if copy_model_path.exists() and ML_AVAILABLE:
                self.copy_move_model = load_model(copy_model_path)
                logger.info("Loaded copy-move detection model")
            
            # Load anomaly detector
            anomaly_path = self.models_path / "anomaly_detector.pkl"
            if anomaly_path.exists():
                self.anomaly_detector = joblib.load(anomaly_path)
                logger.info("Loaded anomaly detector")
            
            # Load feature scaler
            scaler_path = self.models_path / "feature_scaler.pkl"
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def extract_deep_features(self, image_path: str) -> MLFeatures:
        """Extract comprehensive ML features from certificate image"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image")
            
            # Convert to different color spaces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Extract texture features using LBP and HOG
            texture_features = self._extract_texture_features(gray)
            
            # Extract edge features
            edge_features = self._extract_edge_features(gray)
            
            # Extract color features
            color_features = self._extract_color_features(img, hsv)
            
            # Extract geometric features
            geometric_features = self._extract_geometric_features(gray)
            
            # Extract metadata features
            metadata_features = self._extract_metadata_features(image_path)
            
            # Placeholder for text features (to be filled by OCR analysis)
            text_features = np.zeros(100)
            
            return MLFeatures(
                texture_features=texture_features,
                edge_features=edge_features,
                color_features=color_features,
                text_features=text_features,
                geometric_features=geometric_features,
                metadata_features=metadata_features
            )
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return default features
            return MLFeatures(
                texture_features=np.zeros(50),
                edge_features=np.zeros(30),
                color_features=np.zeros(20),
                text_features=np.zeros(100),
                geometric_features=np.zeros(15),
                metadata_features={}
            )

    def _extract_texture_features(self, gray_img: np.ndarray) -> np.ndarray:
        """Extract texture features using LBP and other methods"""
        features = []
        
        if ADVANCED_ANALYSIS:
            # Local Binary Pattern
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
            
            # LBP histogram
            n_bins = n_points + 2
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, 
                                     range=(0, n_bins), density=True)
            features.extend(lbp_hist)
            
            # HOG features
            try:
                hog_features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), block_norm='L2-Hys',
                                 feature_vector=True)
                # Take first 20 HOG features to keep feature vector manageable
                features.extend(hog_features[:20])
            except Exception as e:
                logger.debug(f"HOG extraction failed: {e}")
                features.extend(np.zeros(20))
        
        # Fallback: use basic statistical features
        if len(features) < 30:
            features = [
                np.mean(gray_img), np.std(gray_img), np.var(gray_img),
                np.median(gray_img), np.min(gray_img), np.max(gray_img)
            ]
            features.extend(np.zeros(max(0, 50 - len(features))))
        
        return np.array(features[:50])

    def _extract_edge_features(self, gray_img: np.ndarray) -> np.ndarray:
        """Extract edge-based features"""
        features = []
        
        # Canny edges
        edges = cv2.Canny(gray_img, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Sobel gradients
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.max(gradient_magnitude)
        ])
        
        # Laplacian
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        features.extend([
            np.mean(np.abs(laplacian)),
            np.var(laplacian)
        ])
        
        # Pad to 30 features
        while len(features) < 30:
            features.append(0.0)
        
        return np.array(features[:30])

    def _extract_color_features(self, img_bgr: np.ndarray, img_hsv: np.ndarray) -> np.ndarray:
        """Extract color distribution features"""
        features = []
        
        # Color moments for each channel
        for channel in range(3):
            channel_data = img_bgr[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.mean((channel_data - np.mean(channel_data))**3)  # skewness approximation
            ])
        
        # HSV color features
        for channel in range(3):
            channel_data = img_hsv[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data)
            ])
        
        # Color histogram features
        hist_b = cv2.calcHist([img_bgr], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([img_bgr], [1], None, [8], [0, 256])
        features.extend([np.argmax(hist_b), np.argmax(hist_g)])
        
        return np.array(features[:20])

    def _extract_geometric_features(self, gray_img: np.ndarray) -> np.ndarray:
        """Extract geometric and structural features"""
        features = []
        
        # Image dimensions
        h, w = gray_img.shape
        features.extend([h, w, h/w])  # height, width, aspect ratio
        
        # Connected components
        _, labels = cv2.connectedComponents(cv2.threshold(gray_img, 0, 255, 
                                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        num_components = len(np.unique(labels)) - 1  # subtract background
        features.append(num_components)
        
        # Contours
        contours, _ = cv2.findContours(cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1], 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features.append(len(contours))  # number of contours
        
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            features.extend([
                np.mean(areas),
                np.std(areas),
                np.max(areas) if areas else 0
            ])
        else:
            features.extend([0, 0, 0])
        
        # Pad to 15 features
        while len(features) < 15:
            features.append(0.0)
        
        return np.array(features[:15])

    def _extract_metadata_features(self, image_path: str) -> Dict:
        """Extract metadata features from image"""
        metadata = {}
        
        try:
            with Image.open(image_path) as img:
                # Basic image info
                metadata['width'] = img.width
                metadata['height'] = img.height
                metadata['mode'] = img.mode
                metadata['format'] = img.format
                
                # EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        if isinstance(value, (int, float, str)) and len(str(value)) < 100:
                            metadata[f'exif_{tag}'] = value
                
                # File size
                metadata['file_size'] = os.path.getsize(image_path)
                
        except Exception as e:
            logger.debug(f"Metadata extraction failed: {e}")
        
        return metadata

    def detect_copy_move_forgery(self, image_path: str) -> Tuple[float, Dict]:
        """Detect copy-move forgery using advanced techniques"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0, {"error": "Could not load image"}
            
            # Block-based copy-move detection
            block_size = 16
            threshold = 0.8
            
            h, w = img.shape
            blocks = []
            positions = []
            
            # Extract overlapping blocks
            for i in range(0, h - block_size + 1, block_size // 2):
                for j in range(0, w - block_size + 1, block_size // 2):
                    block = img[i:i+block_size, j:j+block_size]
                    blocks.append(block.flatten())
                    positions.append((i, j))
            
            blocks = np.array(blocks)
            suspicious_pairs = []
            
            # Compare blocks using correlation
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    correlation = np.corrcoef(blocks[i], blocks[j])[0, 1]
                    if correlation > threshold:
                        # Check if blocks are not adjacent (to avoid natural similarities)
                        pos1, pos2 = positions[i], positions[j]
                        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        if distance > block_size * 2:  # Not adjacent
                            suspicious_pairs.append({
                                'correlation': correlation,
                                'distance': distance,
                                'pos1': pos1,
                                'pos2': pos2
                            })
            
            # Calculate forgery score
            if suspicious_pairs:
                max_correlation = max([pair['correlation'] for pair in suspicious_pairs])
                forgery_score = min(max_correlation * 100, 100)
            else:
                forgery_score = 0.0
            
            analysis = {
                'suspicious_pairs': len(suspicious_pairs),
                'max_correlation': max([pair['correlation'] for pair in suspicious_pairs]) if suspicious_pairs else 0,
                'method': 'block_based_correlation'
            }
            
            return forgery_score, analysis
            
        except Exception as e:
            logger.error(f"Copy-move detection failed: {e}")
            return 0.0, {"error": str(e)}

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        try:
            if not self.text_similarity_model or not text1.strip() or not text2.strip():
                return 0.0
            
            # Generate embeddings
            embeddings = self.text_similarity_model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Text similarity calculation failed: {e}")
            return 0.0

    def predict_authenticity(self, features: MLFeatures) -> Tuple[float, Dict]:
        """Predict certificate authenticity using ML models"""
        try:
            # Combine all features into a single vector
            feature_vector = np.concatenate([
                features.texture_features,
                features.edge_features,
                features.color_features,
                features.geometric_features
            ])
            
            # Scale features if scaler is available
            if self.feature_scaler:
                feature_vector = self.feature_scaler.transform([feature_vector])[0]
            
            predictions = {}
            
            # Authenticity prediction
            if self.authenticity_model and SKLEARN_AVAILABLE:
                try:
                    auth_prob = self.authenticity_model.predict_proba([feature_vector])[0]
                    predictions['authenticity_probability'] = float(auth_prob[1])  # Probability of being authentic
                except Exception as e:
                    logger.debug(f"Authenticity prediction failed: {e}")
                    predictions['authenticity_probability'] = 0.5
            
            # Anomaly detection
            if self.anomaly_detector:
                try:
                    anomaly_score = self.anomaly_detector.decision_function([feature_vector])[0]
                    predictions['anomaly_score'] = float(anomaly_score)
                    predictions['is_anomaly'] = anomaly_score < 0
                except Exception as e:
                    logger.debug(f"Anomaly detection failed: {e}")
                    predictions['anomaly_score'] = 0.0
                    predictions['is_anomaly'] = False
            
            # Calculate combined ML authenticity score
            if 'authenticity_probability' in predictions:
                ml_score = predictions['authenticity_probability'] * 100
            else:
                ml_score = 50.0  # Neutral score if no model available
            
            return ml_score, predictions
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return 50.0, {"error": str(e)}

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self):
        self.session = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
    
    def store_certificate(self, session_id: str, analysis_result: Dict, 
                         ml_features: MLFeatures, file_hash: str) -> int:
        """Store certificate analysis results in database"""
        try:
            # Create certificate record
            cert = CertificateRecord(
                session_id=session_id,
                file_hash=file_hash,
                name=analysis_result.get('name'),
                roll_no=analysis_result.get('roll_no'),
                year=analysis_result.get('year'),
                institution=analysis_result.get('institution'),
                course=analysis_result.get('course'),
                grade=analysis_result.get('grade'),
                tampering_score=analysis_result.get('tampering_score', 0),
                ml_authenticity_score=analysis_result.get('ml_authenticity_score', 0),
                verdict=analysis_result.get('verdict'),
                ocr_confidence=analysis_result.get('ocr_confidence', 0),
                image_quality=analysis_result.get('image_quality'),
                processing_time=analysis_result.get('processing_time', 0),
                raw_text=analysis_result.get('raw_text', ''),
                features_json=json.dumps(analysis_result.get('features', {})),
                ml_features_json=self._serialize_ml_features(ml_features)
            )
            
            self.session.add(cert)
            self.session.commit()
            
            logger.info(f"Stored certificate record with ID: {cert.id}")
            return cert.id
            
        except Exception as e:
            logger.error(f"Failed to store certificate: {e}")
            self.session.rollback()
            return -1
    
    def _serialize_ml_features(self, features: MLFeatures) -> str:
        """Serialize ML features to JSON"""
        try:
            return json.dumps({
                'texture_features': features.texture_features.tolist(),
                'edge_features': features.edge_features.tolist(),
                'color_features': features.color_features.tolist(),
                'text_features': features.text_features.tolist(),
                'geometric_features': features.geometric_features.tolist(),
                'metadata_features': features.metadata_features
            })
        except Exception as e:
            logger.error(f"ML feature serialization failed: {e}")
            return "{}"
    
    def find_similar_certificates(self, file_hash: str, features: MLFeatures, 
                                threshold: float = 0.8) -> List[Dict]:
        """Find similar certificates in database"""
        try:
            # Simple hash-based duplicate check
            duplicates = self.session.query(CertificateRecord).filter(
                CertificateRecord.file_hash == file_hash
            ).all()
            
            results = []
            for cert in duplicates:
                results.append({
                    'id': cert.id,
                    'similarity_score': 1.0,
                    'match_type': 'exact_duplicate',
                    'verdict': cert.verdict,
                    'created_at': cert.created_at.isoformat()
                })
            
            # TODO: Implement feature-based similarity search
            # This would require more sophisticated vector similarity search
            
            return results
            
        except Exception as e:
            logger.error(f"Similar certificate search failed: {e}")
            return []
    
    def get_institution_stats(self, institution_name: str) -> Dict:
        """Get statistics for a specific institution"""
        try:
            certs = self.session.query(CertificateRecord).filter(
                CertificateRecord.institution.ilike(f'%{institution_name}%')
            ).all()
            
            if not certs:
                return {"total": 0}
            
            authentic_count = sum(1 for cert in certs if cert.tampering_score < 30)
            suspicious_count = sum(1 for cert in certs if 30 <= cert.tampering_score < 70)
            forgery_count = sum(1 for cert in certs if cert.tampering_score >= 70)
            
            return {
                "total": len(certs),
                "authentic": authentic_count,
                "suspicious": suspicious_count,
                "likely_forgery": forgery_count,
                "avg_tampering_score": np.mean([cert.tampering_score or 0 for cert in certs]),
                "avg_ml_score": np.mean([cert.ml_authenticity_score or 0 for cert in certs])
            }
            
        except Exception as e:
            logger.error(f"Institution stats failed: {e}")
            return {"error": str(e)}

# Enhanced analysis functions
def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Hash calculation failed: {e}")
        return ""

def enhanced_analyze_certificate(file_path: str, session_id: str, 
                               ml_analyzer: CertificateMLAnalyzer) -> Dict:
    """Enhanced certificate analysis with ML and database integration"""
    try:
        logger.info(f"Starting enhanced analysis for session {session_id}")
        start_time = time.time()
        
        # Update session status
        if session_id in sessions:
            sessions[session_id]['status'] = 'processing'
        
        # Calculate file hash for duplicate detection
        file_hash = calculate_file_hash(file_path)
        
        # Handle PDF conversion
        image_path = file_path
        if file_path.lower().endswith('.pdf'):
            logger.info("Converting PDF to image...")
            converted_path = convert_pdf_to_image(file_path)
            if converted_path:
                image_path = converted_path
            else:
                raise Exception("Failed to convert PDF to image")
        
        # Extract ML features
        logger.info("Extracting ML features...")
        ml_features = ml_analyzer.extract_deep_features(image_path)
        
        # Calculate image quality metrics
        logger.info("Analyzing image quality...")
        quality_metrics = calculate_image_quality_metrics(image_path)
        
        # Enhance image for better OCR
        logger.info("Enhancing image for OCR...")
        enhanced_path, processed_img = enhance_image_quality(image_path)
        ocr_image = enhanced_path if enhanced_path else image_path
        
        # Advanced OCR extraction
        logger.info("Performing OCR extraction...")
        raw_text, ocr_confidence = advanced_ocr_extraction(ocr_image)
        
        # Update text features in ML features
        if raw_text and ml_analyzer.text_similarity_model:
            try:
                text_embedding = ml_analyzer.text_similarity_model.encode(raw_text)
                ml_features.text_features = text_embedding[:100]  # Take first 100 features
            except Exception as e:
                logger.debug(f"Text embedding failed: {e}")
        
        # QR code detection
        logger.info("Scanning for QR codes...")
        qr_data = scan_for_qr_codes(image_path)
        
        # Extract structured fields
        logger.info("Extracting certificate fields...")
        extracted_fields = extract_certificate_fields(raw_text)
        
        # Copy-move forgery detection
        logger.info("Detecting copy-move forgery...")
        copy_move_score, copy_move_analysis = ml_analyzer.detect_copy_move_forgery(image_path)
        
        # ML authenticity prediction
        logger.info("Predicting authenticity with ML...")
        ml_authenticity_score, ml_predictions = ml_analyzer.predict_authenticity(ml_features)
        
        # Traditional tampering detection
        logger.info("Performing traditional tampering analysis...")
        tampering_score, verdict, notes = advanced_tampering_detection(
            image_path, raw_text, ocr_confidence, qr_data, extracted_fields, quality_metrics
        )
        
        # Database operations
        with DatabaseManager() as db:
            # Check for similar certificates
            logger.info("Searching for similar certificates...")
            similar_certs = db.find_similar_certificates(file_hash, ml_features)
            
            # Get institution statistics if institution is detected
            institution_stats = {}
            if extracted_fields.get('institution'):
                institution_stats = db.get_institution_stats(extracted_fields['institution'])
        
        # Combine traditional and ML scores
        combined_score = (tampering_score * 0.6) + ((100 - ml_authenticity_score) * 0.4)
        combined_score = min(max(combined_score, 0), 100)
        
        # Adjust verdict based on combined analysis
        if combined_score < 25:
            final_verdict = "authentic"
        elif combined_score < 45:
            final_verdict = "likely_authentic"
        elif combined_score < 70:
            final_verdict = "suspicious"
        else:
            final_verdict = "likely_forgery"
        
        # Enhanced notes with ML insights
        enhanced_notes = notes
        if copy_move_score > 20:
            enhanced_notes += f" • Copy-move forgery detected (score: {copy_move_score:.1f})"
        
        if 'is_anomaly' in ml_predictions and ml_predictions['is_anomaly']:
            enhanced_notes += f" • ML anomaly detected (score: {ml_predictions.get('anomaly_score', 0):.2f})"
        
        if similar_certs:
            enhanced_notes += f" • {len(similar_certs)} similar certificate(s) found in database"
        
        # Compile comprehensive result
        processing_time = round(time.time() - start_time, 2)
        
        result = {
            # Basic extracted information
            'name': extracted_fields.get('name'),
            'roll_no': extracted_fields.get('roll_no'),
            'year': extracted_fields.get('year'),
            'institution': extracted_fields.get('institution'),
            'course': extracted_fields.get('course'),
            'grade': extracted_fields.get('grade'),
            
            # Scoring
            'tampering_score': round(combined_score, 1),
            'traditional_tampering_score': round(tampering_score, 1),
            'ml_authenticity_score': round(ml_authenticity_score, 1),
            'copy_move_score': round(copy_move_score, 1),
            'verdict': final_verdict,
            
            # Technical metrics
            'ocr_confidence': round(ocr_confidence, 1),
            'text_length': len(raw_text),
            'qr_codes_found': len(qr_data),
            'image_quality': quality_metrics['quality'],
            
            # Enhanced analysis
            'notes': enhanced_notes,
            'ml_predictions': ml_predictions,
            'copy_move_analysis': copy_move_analysis,
            'similar_certificates': len(similar_certs),
            'institution_stats': institution_stats,
            
            # Metadata
            'processing_time': processing_time,
            'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_hash': file_hash[:16],  # Show partial hash for reference
            'raw_text': raw_text[:500],  # Truncate for display
            
            # Internal data for storage
            '_full_raw_text': raw_text,
            '_ml_features': ml_features,
            '_file_hash': file_hash
        }
        
        # Store in database
        with DatabaseManager() as db:
            db.store_certificate(session_id, result, ml_features, file_hash)
        
        # Clean up temporary files
        try:
            if enhanced_path and os.path.exists(enhanced_path):
                os.remove(enhanced_path)
            if image_path != file_path and os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            logger.debug(f"Cleanup warning: {e}")
        
        logger.info(f"Enhanced analysis completed in {processing_time}s: {final_verdict} (combined score: {combined_score:.1f})")
        return result
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        raise

# Keep all original functions but add enhanced ones
def convert_pdf_to_image(pdf_path):
    """Convert PDF first page to image"""
    try:
        if PDF_SUPPORT:
            images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
            if images:
                img_path = pdf_path.replace('.pdf', '_converted.jpg')
                images[0].save(img_path, 'JPEG', quality=95)
                return img_path
        else:
            logger.warning("pdf2image not available, trying alternative method")
        
        with Image.open(pdf_path) as img:
            img_path = pdf_path.replace('.pdf', '_converted.jpg')
            img.convert('RGB').save(img_path, 'JPEG', quality=95)
            return img_path
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        return None

def enhance_image_quality(image_path):
    """Advanced image preprocessing for better OCR"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        enhanced_path = image_path.replace('.', '_enhanced.')
        cv2.imwrite(enhanced_path, cleaned)
        
        return enhanced_path, gray
        
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        return None, None

def calculate_image_quality_metrics(image_path):
    """Calculate image quality metrics"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"quality": "unknown", "sharpness": 0, "contrast": 0}
        
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        contrast = np.std(img)
        
        if laplacian_var > 500 and contrast > 50:
            quality = "excellent"
        elif laplacian_var > 200 and contrast > 30:
            quality = "good"
        elif laplacian_var > 100 and contrast > 20:
            quality = "fair"
        else:
            quality = "poor"
            
        return {
            "quality": quality,
            "sharpness": round(laplacian_var, 2),
            "contrast": round(contrast, 2)
        }
        
    except Exception as e:
        logger.error(f"Quality calculation failed: {e}")
        return {"quality": "unknown", "sharpness": 0, "contrast": 0}

def advanced_ocr_extraction(image_path):
    """Advanced OCR with multiple configurations"""
    results = []
    
    configs = [
        '--oem 3 --psm 6',
        '--oem 3 --psm 4',
        '--oem 3 --psm 3',
        '--oem 1 --psm 6'
    ]
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(image_path, config=config)
            if text.strip():
                data = pytesseract.image_to_data(image_path, config=config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = np.mean(confidences) if confidences else 0
                
                results.append({
                    'text': text.strip(),
                    'confidence': avg_confidence,
                    'config': config,
                    'word_count': len([w for w in text.split() if w.strip()])
                })
        except Exception as e:
            logger.error(f"OCR config {config} failed: {e}")
            continue
    
    if results:
        best_result = max(results, key=lambda x: (x['confidence'], x['word_count']))
        return best_result['text'], best_result['confidence']
    
    return "", 0

def scan_for_qr_codes(image_path):
    """Enhanced QR code detection"""
    try:
        img_original = cv2.imread(image_path)
        if img_original is None:
            return []
            
        all_qr_data = []
        
        decoded_objects = pyzbar.decode(img_original)
        for obj in decoded_objects:
            all_qr_data.append({
                'data': obj.data.decode('utf-8', errors='ignore'),
                'type': obj.type,
                'method': 'original'
            })
        
        gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        decoded_objects = pyzbar.decode(gray)
        for obj in decoded_objects:
            data_str = obj.data.decode('utf-8', errors='ignore')
            if not any(qr['data'] == data_str for qr in all_qr_data):
                all_qr_data.append({
                    'data': data_str,
                    'type': obj.type,
                    'method': 'grayscale'
                })
        
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        decoded_objects = pyzbar.decode(enhanced)
        for obj in decoded_objects:
            data_str = obj.data.decode('utf-8', errors='ignore')
            if not any(qr['data'] == data_str for qr in all_qr_data):
                all_qr_data.append({
                    'data': data_str,
                    'type': obj.type,
                    'method': 'enhanced'
                })
        
        return all_qr_data
        
    except Exception as e:
        logger.error(f"QR scanning failed: {e}")
        return []

def extract_certificate_fields(raw_text):
    """Advanced field extraction with multiple patterns"""
    fields = {
        'name': None,
        'roll_no': None,
        'year': None,
        'institution': None,
        'course': None,
        'grade': None
    }
    
    text = raw_text.upper().replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    # Name extraction patterns
    name_patterns = [
        r'(?:NAME|CANDIDATE\s*NAME|STUDENT\s*NAME|FULL\s*NAME)\s*[:\-]?\s*([A-Z][A-Z\s\.]{2,50})',
        r'(?:THIS\s*IS\s*TO\s*CERTIFY\s*THAT)\s+([A-Z][A-Z\s\.]{5,50})',
        r'(?:MR\.?|MS\.?|MISS)\s+([A-Z][A-Z\s\.]{5,50})',
        r'NAME\s*:\s*([A-Z][A-Z\s\.]{2,50})',
        r'(?:CANDIDATE|STUDENT)\s*:\s*([A-Z][A-Z\s\.]{2,50})'
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            name = match.strip()
            if len(name.split()) >= 2 and 5 <= len(name) <= 50:
                if not any(word in name for word in ['UNIVERSITY', 'COLLEGE', 'BOARD', 'EXAMINATION', 'CERTIFICATE']):
                    fields['name'] = name.title()
                    break
        if fields['name']:
            break
    
    # Roll number patterns
    roll_patterns = [
        r'(?:ROLL\s*(?:NO|NUMBER)|REG\s*(?:NO|NUMBER)|REGISTRATION\s*(?:NO|NUMBER)|ENROLLMENT\s*(?:NO|NUMBER))\s*[:\-]?\s*([A-Z0-9\-\/]{3,25})',
        r'(?:ROLL|REG|REGISTRATION|ENROLLMENT)\s*[:\-]?\s*([A-Z0-9\-\/]{3,25})',
        r'(?:ID|STUDENT\s*ID)\s*[:\-]?\s*([A-Z0-9\-\/]{3,25})',
        r'([A-Z]{1,3}\d{4,10})',
        r'(\d{4,12})',
    ]
    
    for pattern in roll_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            roll = match.strip()
            if 3 <= len(roll) <= 25 and re.search(r'\d', roll):
                fields['roll_no'] = roll
                break
        if fields['roll_no']:
            break
    
    # Year extraction
    year_patterns = [
        r'(?:YEAR|PASSING\s*YEAR|PASSED\s*IN|ACADEMIC\s*YEAR)\s*[:\-]?\s*(20\d{2}|19[89]\d)',
        r'(?:MAY|JUNE|APRIL|NOVEMBER|DECEMBER|MARCH)\s*(20\d{2})',
        r'(20[0-2]\d)(?:\s*-\s*20[0-2]\d)?',
        r'(19[89]\d)',
    ]
    
    current_year = datetime.now().year
    found_years = []
    
    for pattern in year_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1]
            try:
                year = int(match)
                if 1980 <= year <= current_year + 1:
                    found_years.append(year)
            except ValueError:
                continue
    
    if found_years:
        fields['year'] = str(max(found_years))
    
    # Institution extraction
    institution_patterns = [
        r'(?:UNIVERSITY|COLLEGE|INSTITUTE|BOARD)\s*(?:OF\s*)?([A-Z][A-Z\s]{5,50})',
        r'([A-Z][A-Z\s]{5,50})\s*(?:UNIVERSITY|COLLEGE|INSTITUTE|BOARD)',
        r'(?:ISSUED\s*BY|AWARDED\s*BY)\s*([A-Z][A-Z\s]{5,40})',
    ]
    
    for pattern in institution_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            inst = match.strip()
            if 5 <= len(inst) <= 50 and inst not in ['THE', 'OF', 'AND']:
                fields['institution'] = inst.title()
                break
        if fields['institution']:
            break
    
    return fields

def advanced_tampering_detection(image_path, raw_text, ocr_confidence, qr_data, extracted_fields, quality_metrics):
    """Comprehensive tampering detection using multiple techniques"""
    tampering_score = 0
    issues = []
    
    # Field Completeness Analysis
    required_fields = ['name', 'roll_no', 'year']
    missing_fields = [field for field in required_fields if not extracted_fields.get(field)]
    
    if missing_fields:
        score_penalty = len(missing_fields) * 15
        tampering_score += score_penalty
        issues.append(f"Missing critical fields: {', '.join(missing_fields)} (+{score_penalty})")
    
    # OCR Confidence Analysis
    if ocr_confidence < 60:
        score_penalty = int((60 - ocr_confidence) / 2)
        tampering_score += score_penalty
        issues.append(f"Low OCR confidence: {ocr_confidence:.1f}% (+{score_penalty})")
    
    # Image Quality Analysis
    if quality_metrics['quality'] in ['poor', 'fair']:
        score_penalty = 15 if quality_metrics['quality'] == 'poor' else 8
        tampering_score += score_penalty
        issues.append(f"Poor image quality: {quality_metrics['quality']} (+{score_penalty})")
    
    # Date Consistency Checks
    if extracted_fields.get('year'):
        try:
            year = int(extracted_fields['year'])
            current_year = datetime.now().year
            
            if year > current_year:
                tampering_score += 25
                issues.append(f"Future year detected: {year} (+25)")
            elif year < 1980:
                tampering_score += 20
                issues.append(f"Unreasonably old year: {year} (+20)")
            elif year > current_year - 1:
                tampering_score += 5
                issues.append(f"Very recent certificate: {year} (+5)")
        except ValueError:
            pass
    
    # Text Pattern Analysis
    text_length = len(raw_text.strip())
    if text_length < 50:
        score_penalty = 15
        tampering_score += score_penalty
        issues.append(f"Insufficient text extracted: {text_length} chars (+{score_penalty})")
    elif text_length > 5000:
        score_penalty = 5
        tampering_score += score_penalty
        issues.append(f"Unusually verbose certificate (+{score_penalty})")
    
    # Image Metadata Analysis
    try:
        with Image.open(image_path) as img:
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == 'Software' and value:
                        software = value.lower()
                        if any(editor in software for editor in ['photoshop', 'gimp', 'paint.net', 'canva']):
                            tampering_score += 12
                            issues.append(f"Image editing software detected: {value} (+12)")
                            break
            
            width, height = img.size
            if width < 800 or height < 600:
                tampering_score += 8
                issues.append(f"Low resolution image: {width}x{height} (+8)")
    except Exception as e:
        logger.debug(f"Metadata analysis failed: {e}")
    
    # Advanced Image Analysis
    if ADVANCED_ANALYSIS:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                regions = [
                    img[0:h//2, 0:w//2],
                    img[0:h//2, w//2:w],
                    img[h//2:h, 0:w//2],
                    img[h//2:h, w//2:w]
                ]
                
                texture_variations = []
                for region in regions:
                    if region.size > 0:
                        lbp = local_binary_pattern(region, 8, 1.5, method='uniform')
                        hist, _ = np.histogram(lbp.ravel(), bins=10)
                        texture_variations.append(np.std(hist))
                
                if len(texture_variations) > 1:
                    texture_inconsistency = np.std(texture_variations)
                    if texture_inconsistency > 100:
                        score_penalty = min(int(texture_inconsistency / 20), 15)
                        tampering_score += score_penalty
                        issues.append(f"Texture inconsistencies detected (+{score_penalty})")
        
        except Exception as e:
            logger.debug(f"Advanced image analysis failed: {e}")
    
    # Name Validation
    if extracted_fields.get('name'):
        name = extracted_fields['name']
        if re.search(r'\d', name):
            tampering_score += 8
            issues.append("Numbers found in name field (+8)")
        elif len(name.split()) < 2:
            tampering_score += 5
            issues.append("Incomplete name detected (+5)")
        elif len(name) > 40:
            tampering_score += 3
            issues.append("Unusually long name (+3)")
    
    # QR Code Analysis
    if qr_data:
        issues.append(f"QR codes found: {len(qr_data)}")
        tampering_score = max(0, tampering_score - 2)
    
    tampering_score = min(tampering_score, 100)
    
    if tampering_score < 20:
        verdict = "authentic"
    elif tampering_score < 40:
        verdict = "likely_authentic"
    elif tampering_score < 70:
        verdict = "suspicious"
    else:
        verdict = "likely_forgery"
    
    if not issues:
        issues.append("All standard verification checks passed")
    
    notes = " • ".join(issues)
    
    return tampering_score, verdict, notes

def cleanup_expired_sessions():
    """Remove expired sessions"""
    current_time = datetime.now()
    expired_sessions = [
        sid for sid, data in sessions.items()
        if current_time > data.get('expires_at', current_time)
    ]
    for sid in expired_sessions:
        del sessions[sid]
    return len(expired_sessions)

def generate_qr_code(data):
    """Generate high-quality QR code"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode()

# Initialize ML analyzer
ml_analyzer = CertificateMLAnalyzer(app.config['ML_MODELS_PATH'])

# Enhanced HTML Templates with ML insights
ENHANCED_VERIFIER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Acad Enhanced Certificate Verifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #0d0d0d 100%);
            min-height: 100vh; padding: 20px; color: #e0e0e0;
            overflow-x: hidden;
        }
        .container { 
            max-width: 1000px; margin: 0 auto; 
            background: linear-gradient(145deg, #1a1a1a, #0f0f0f);
            border-radius: 25px; 
            box-shadow: 0 25px 50px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.05);
            overflow: hidden; backdrop-filter: blur(10px);
        }
        .header { 
            background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
            color: #ffffff; padding: 40px; text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            position: relative; overflow: hidden;
        }
        .header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.03), transparent);
            animation: shimmer 3s ease-in-out infinite;
        }
        @keyframes shimmer {
            0%, 100% { transform: translateX(-100%); }
            50% { transform: translateX(100%); }
        }
        .header h1 { 
            font-size: 3.2em; margin-bottom: 15px; 
            background: linear-gradient(45deg, #888, #ccc, #888);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: textShine 2s ease-in-out infinite alternate;
            position: relative; z-index: 1;
        }
        @keyframes textShine {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }
        .header p { opacity: 0.8; font-size: 1.3em; position: relative; z-index: 1; }
        .badge { 
            display: inline-block; padding: 5px 12px; margin: 5px;
            border-radius: 15px; font-size: 0.8em; font-weight: bold;
            background: linear-gradient(145deg, #333, #222);
            border: 1px solid #444;
        }
        .badge.ml { background: linear-gradient(145deg, #1a2a3a, #0f1a2a); color: #7ab8ff; }
        .badge.db { background: linear-gradient(145deg, #2a1a3a, #1a0f2a); color: #b77aff; }
        .content { padding: 50px; background: #111111; }
        .qr-section { 
            text-align: center; margin: 40px 0; padding: 40px;
            background: linear-gradient(145deg, #1e1e1e, #0a0a0a);
            border-radius: 20px; 
            border: 2px solid #333333;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.5), 0 5px 20px rgba(0,0,0,0.3);
        }
        .qr-section h2 { 
            color: #888; margin-bottom: 25px; font-size: 2em; 
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }
        .qr-code { 
            display: inline-block; padding: 25px; 
            background: linear-gradient(145deg, #f8f8f8, #e0e0e0);
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.7), inset 0 1px 2px rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        .qr-code:hover { transform: scale(1.05); }
        .session-info { 
            margin: 25px 0; padding: 20px; 
            background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
            border-radius: 10px; font-family: 'Courier New', monospace;
            border: 1px solid #333; color: #aaa;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.3);
        }
        .status { 
            padding: 25px; margin: 25px 0; border-radius: 15px; 
            font-weight: bold; text-align: center; font-size: 1.2em;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        .pending { 
            background: linear-gradient(145deg, #2a2a2a, #1a1a1a); 
            border: 2px solid #444; color: #ccc;
        }
        .processing { 
            background: linear-gradient(145deg, #1a2a3a, #0f1a2a); 
            border: 2px solid #2a4a6a; color: #7ab8ff;
        }
        .success { 
            background: linear-gradient(145deg, #1a2a1a, #0f1a0f); 
            border: 2px solid #2a4a2a; color: #7aff7a;
        }
        .error { 
            background: linear-gradient(145deg, #2a1a1a, #1a0f0f); 
            border: 2px solid #4a2a2a; color: #ff7a7a;
        }
        .result-container { margin-top: 40px; }
        .result-box { 
            background: linear-gradient(145deg, #1a1a1a, #0f0f0f); 
            padding: 30px; margin: 20px 0; border-radius: 20px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.5), inset 0 1px 2px rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .authentic { 
            border-left: 5px solid #4CAF50;
            box-shadow: 0 10px 30px rgba(76, 175, 80, 0.2), inset 0 1px 2px rgba(255,255,255,0.05);
        }
        .suspicious { 
            border-left: 5px solid #FF9800;
            box-shadow: 0 10px 30px rgba(255, 152, 0, 0.2), inset 0 1px 2px rgba(255,255,255,0.05);
        }
        .forgery { 
            border-left: 5px solid #F44336;
            box-shadow: 0 10px 30px rgba(244, 67, 54, 0.2), inset 0 1px 2px rgba(255,255,255,0.05);
        }
        .score-section {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin: 25px 0;
        }
        .score-item {
            text-align: center; padding: 20px;
            background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
            border-radius: 15px; border: 1px solid #333;
        }
        .score-label { font-size: 0.9em; color: #888; margin-bottom: 10px; }
        .score-value { font-size: 2em; font-weight: bold; }
        .field-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px; margin: 25px 0;
        }
        .field-item { 
            padding: 20px; 
            background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
            border-radius: 12px;
            border-left: 4px solid #555;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.3), 0 2px 10px rgba(0,0,0,0.2);
            transition: transform 0.2s ease;
        }
        .field-item:hover { transform: translateY(-2px); }
        .field-label { font-weight: bold; color: #888; font-size: 0.9em; margin-bottom: 5px; }
        .field-value { font-size: 1.1em; color: #ccc; word-break: break-word; }
        .score-bar { 
            width: 100%; height: 25px; 
            background: linear-gradient(145deg, #0a0a0a, #1a1a1a);
            border-radius: 12px;
            overflow: hidden; margin: 15px 0;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.5);
        }
        .score-fill { 
            height: 100%; transition: width 0.8s ease;
            background: linear-gradient(90deg, #4CAF50, #FF9800, #F44336);
            position: relative; overflow: hidden;
        }
        .score-fill::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(45deg, 
                transparent 40%, 
                rgba(255,255,255,0.1) 50%, 
                transparent 60%);
            animation: scoreShimmer 2s linear infinite;
        }
        @keyframes scoreShimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        .ml-section {
            background: linear-gradient(145deg, #1a1a2a, #0f0f1a);
            border: 1px solid #2a4a6a;
            margin: 20px 0; padding: 25px; border-radius: 15px;
        }
        .db-section {
            background: linear-gradient(145deg, #2a1a2a, #1a0f1a);
            border: 1px solid #4a2a6a;
            margin: 20px 0; padding: 25px; border-radius: 15px;
        }
        .loading-spinner {
            border: 4px solid #222;
            border-top: 4px solid #666;
            border-radius: 50%;
            width: 50px; height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            box-shadow: 0 5px 15px rgba(0,0,0,0.5);
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        h3, h4 { 
            color: #bbb; margin: 20px 0 15px 0; 
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        hr {
            border: none; height: 1px; 
            background: linear-gradient(90deg, transparent, #333, transparent);
            margin: 30px 0;
        }
        @media (max-width: 768px) {
            .content { padding: 25px; }
            .header h1 { font-size: 2.5em; }
            .field-grid { grid-template-columns: 1fr; }
            .qr-section { padding: 25px; }
        }
        
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { 
            background: linear-gradient(145deg, #333, #555);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover { background: linear-gradient(145deg, #444, #666); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ACAD</h1>
            <p>Enhanced AI-Powered Certificate Verification</p>
            <div>
                <span class="badge ml">ML Powered</span>
                <span class="badge db">Database Integrated</span>
                <span class="badge">Forensic Analysis</span>
            </div>
        </div>
        
        <div class="content">
            <div class="qr-section">
                <h2>Scan to Upload Certificate</h2>
                <div class="qr-code">
                    {% if qr_code %}
                        <img src="data:image/png;base64,{{ qr_code }}" alt="QR Code" style="max-width: 200px; height: auto;">
                    {% else %}
                        <div style="width: 200px; height: 200px; background: #f0f0f0; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #333; font-size: 14px; text-align: center; border: 2px dashed #ccc;">
                            QR Code Loading...
                        </div>
                    {% endif %}
                </div>
                <div class="session-info">
                    <strong>Session ID:</strong> <span id="session-id">{{ session_id }}</span><br>
                    <strong>Created:</strong> <span id="created-time">{{ created_time }}</span><br>
                    <strong>Expires:</strong> <span id="expires-time">{{ expires_time }}</span>
                </div>
            </div>
            
            <div id="status" class="status pending">
                <div class="loading-spinner"></div>
                <strong>Status:</strong> Waiting for certificate upload
            </div>
            
            <div id="result" class="result-container" style="display: none;">
                <h3>Enhanced Verification Results</h3>
                <div id="result-content" class="result-box"></div>
            </div>
        </div>
    </div>

    <script>
        let sessionId = '{{ session_id }}';
        let pollInterval = null;
        let isProcessing = false;
        let retryCount = 0;
        const maxRetries = 3;

        function startPolling() {
            if (pollInterval) {
                clearInterval(pollInterval);
            }
            pollInterval = setInterval(checkResults, 2000);
        }

        function stopPolling() {
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }
        }

        function showError(message) {
            const statusDiv = document.getElementById('status');
            statusDiv.className = 'status error';
            statusDiv.innerHTML = `<strong>Error:</strong> ${message}`;
        }

        function sanitizeHTML(str) {
            const div = document.createElement('div');
            div.textContent = str || '';
            return div.innerHTML;
        }

        async function checkResults() {
            try {
                const response = await fetch(`/results/${encodeURIComponent(sessionId)}`, {
                    method: 'GET',
                    headers: {
                        'Accept': 'application/json',
                        'Cache-Control': 'no-cache'
                    }
                });

                if (!response.ok) {
                    if (response.status === 404) {
                        return;
                    }
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                
                const statusDiv = document.getElementById('status');
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('result-content');

                if (data.status === 'processing' && !isProcessing) {
                    isProcessing = true;
                    statusDiv.className = 'status processing';
                    statusDiv.innerHTML = `
                        <div class="loading-spinner"></div>
                        <strong>Processing with AI/ML Analysis...</strong>
                    `;
                } else if (data.status === 'done' && data.result) {
                    stopPolling();
                    statusDiv.className = 'status success';
                    statusDiv.innerHTML = '<strong>Enhanced Analysis Completed</strong>';
                    
                    displayEnhancedResults(data.result, resultDiv, resultContent);
                } else if (data.status === 'error') {
                    stopPolling();
                    showError(sanitizeHTML(data.error || 'Analysis failed'));
                }

                retryCount = 0;

            } catch (error) {
                console.error('Error checking results:', error);
                retryCount++;
                
                if (retryCount >= maxRetries) {
                    stopPolling();
                    showError(`Failed to check results after ${maxRetries} attempts. Please refresh the page.`);
                }
            }
        }

        function displayEnhancedResults(result, resultDiv, resultContent) {
            resultDiv.style.display = 'block';
            
            let verdictClass = 'authentic';
            let verdictIcon = '✓ PASS';
            let verdictText = 'AUTHENTIC';
            let verdictColor = '#4CAF50';
            
            const tamperingScore = Math.max(0, Math.min(100, Number(result.tampering_score) || 0));
            const mlScore = Math.max(0, Math.min(100, Number(result.ml_authenticity_score) || 50));
            const copyMoveScore = Math.max(0, Math.min(100, Number(result.copy_move_score) || 0));
            
            if (tamperingScore >= 70) {
                verdictClass = 'forgery';
                verdictIcon = '✗ FAIL';
                verdictText = 'LIKELY FORGERY';
                verdictColor = '#F44336';
            } else if (tamperingScore >= 45) {
                verdictClass = 'suspicious';
                verdictIcon = '⚠ RISK';
                verdictText = 'SUSPICIOUS';
                verdictColor = '#FF9800';
            }
            
            resultContent.className = `result-box ${verdictClass}`;
            resultContent.innerHTML = `
                <div style="text-align: center; margin-bottom: 30px;">
                    <h2 style="font-size: 2.5em; margin-bottom: 20px; color: ${verdictColor};">${verdictIcon}</h2>
                    <h3 style="font-size: 2em; margin-bottom: 20px;">${verdictText}</h3>
                    <div class="score-section">
                        <div class="score-item">
                            <div class="score-label">Combined Risk Score</div>
                            <div class="score-value" style="color: ${verdictColor};">${tamperingScore}/100</div>
                        </div>
                        <div class="score-item">
                            <div class="score-label">ML Authenticity</div>
                            <div class="score-value" style="color: #7ab8ff;">${mlScore}/100</div>
                        </div>
                        <div class="score-item">
                            <div class="score-label">Copy-Move Detection</div>
                            <div class="score-value" style="color: ${copyMoveScore > 20 ? '#FF6B6B' : '#4ECDC4'};">${copyMoveScore}/100</div>
                        </div>
                    </div>
                    <div style="margin: 20px 0;">
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${tamperingScore}%"></div>
                        </div>
                    </div>
                </div>
                
                <h4>📋 Extracted Information</h4>
                <div class="field-grid">
                    <div class="field-item">
                        <div class="field-label">Full Name</div>
                        <div class="field-value">${sanitizeHTML(result.name || 'Not detected')}</div>
                    </div>
                    <div class="field-item">
                        <div class="field-label">Roll/Registration Number</div>
                        <div class="field-value">${sanitizeHTML(result.roll_no || 'Not detected')}</div>
                    </div>
                    <div class="field-item">
                        <div class="field-label">Year/Date</div>
                        <div class="field-value">${sanitizeHTML(result.year || 'Not detected')}</div>
                    </div>
                    <div class="field-item">
                        <div class="field-label">Institution/Board</div>
                        <div class="field-value">${sanitizeHTML(result.institution || 'Not detected')}</div>
                    </div>
                </div>
                
                <h4>🔬 Technical Analysis</h4>
                <div class="field-grid">
                    <div class="field-item">
                        <div class="field-label">OCR Confidence</div>
                        <div class="field-value">${Math.max(0, Math.min(100, Number(result.ocr_confidence) || 0))}%</div>
                    </div>
                    <div class="field-item">
                        <div class="field-label">Text Extracted</div>
                        <div class="field-value">${Math.max(0, Number(result.text_length) || 0)} characters</div>
                    </div>
                    <div class="field-item">
                        <div class="field-label">QR Codes Found</div>
                        <div class="field-value">${Math.max(0, Number(result.qr_codes_found) || 0)}</div>
                    </div>
                    <div class="field-item">
                        <div class="field-label">Image Quality</div>
                        <div class="field-value">${sanitizeHTML(result.image_quality || 'Unknown')}</div>
                    </div>
                </div>
                
                <div class="ml-section">
                    <h4>🤖 AI/ML Analysis</h4>
                    <div class="field-grid">
                        <div class="field-item">
                            <div class="field-label">ML Authenticity Score</div>
                            <div class="field-value">${mlScore}%</div>
                        </div>
                        <div class="field-item">
                            <div class="field-label">Copy-Move Forgery</div>
                            <div class="field-value">${copyMoveScore > 20 ? 'Detected' : 'Not Detected'}</div>
                        </div>
                        <div class="field-item">
                            <div class="field-label">Anomaly Detection</div>
                            <div class="field-value">${result.ml_predictions && result.ml_predictions.is_anomaly ? 'Anomalous' : 'Normal'}</div>
                        </div>
                        <div class="field-item">
                            <div class="field-label">Processing Method</div>
                            <div class="field-value">Deep Learning + Forensics</div>
                        </div>
                    </div>
                </div>
                
                ${result.institution_stats && result.institution_stats.total > 0 ? `
                <div class="db-section">
                    <h4>📊 Database Insights</h4>
                    <div class="field-grid">
                        <div class="field-item">
                            <div class="field-label">Similar Certificates</div>
                            <div class="field-value">${result.similar_certificates || 0}</div>
                        </div>
                        <div class="field-item">
                            <div class="field-label">Institution Records</div>
                            <div class="field-value">${result.institution_stats.total}</div>
                        </div>
                        <div class="field-item">
                            <div class="field-label">Institution Authenticity</div>
                            <div class="field-value">${Math.round((result.institution_stats.authentic / result.institution_stats.total) * 100)}%</div>
                        </div>
                        <div class="field-item">
                            <div class="field-label">File Hash (partial)</div>
                            <div class="field-value">${sanitizeHTML(result.file_hash || 'N/A')}</div>
                        </div>
                    </div>
                </div>
                ` : ''}
                
                <h4>🔍 Forensic Analysis Report</h4>
                <div style="background: linear-gradient(145deg, #0f0f0f, #1a1a1a); padding: 20px; border-radius: 12px; margin-top: 15px; border: 1px solid #333; color: #ccc; line-height: 1.6;">
                    ${sanitizeHTML(result.notes || 'No additional notes available.')}
                </div>
                
                <hr>
                <div style="text-align: center; color: #888; font-size: 0.95em;">
                    <div style="margin: 10px 0;">
                        <strong>Processed:</strong> ${sanitizeHTML(result.processed_at || new Date().toLocaleString())}<br>
                        <strong>Processing Time:</strong> ${result.processing_time || 0}s<br>
                        <strong>Analysis Engine:</strong> Traditional + ML + Database
                    </div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 15px;">
                        This analysis combines traditional forensic techniques with machine learning models<br>
                        and database insights for comprehensive certificate verification.
                    </div>
                </div>
            `;
        }

        document.addEventListener('DOMContentLoaded', function() {
            startPolling();
        });

        window.addEventListener('beforeunload', function() {
            stopPolling();
        });

        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                stopPolling();
            } else if (!isProcessing) {
                startPolling();
            }
        });
    </script>
</body>
</html>"""

ENHANCED_UPLOAD_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Upload Certificate - Enhanced Acad</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #0d0d0d 100%);
            min-height: 100vh; padding: 20px; color: #e0e0e0;
        }
        .container { 
            max-width: 800px; margin: 0 auto; 
            background: linear-gradient(145deg, #1a1a1a, #0f0f0f);
            border-radius: 25px; 
            box-shadow: 0 25px 50px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.05);
            overflow: hidden; backdrop-filter: blur(10px);
        }
        .header { 
            background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
            color: #ffffff; padding: 40px; text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            position: relative; overflow: hidden;
        }
        .header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.03), transparent);
            animation: shimmer 3s ease-in-out infinite;
        }
        @keyframes shimmer {
            0%, 100% { transform: translateX(-100%); }
            50% { transform: translateX(100%); }
        }
        .header h1 { 
            font-size: 2.8em; margin-bottom: 10px; 
            background: linear-gradient(45deg, #888, #ccc, #888);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: textShine 2s ease-in-out infinite alternate;
            position: relative; z-index: 1;
        }
        @keyframes textShine {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }
        .badge { 
            display: inline-block; padding: 5px 12px; margin: 5px;
            border-radius: 15px; font-size: 0.8em; font-weight: bold;
            background: linear-gradient(145deg, #333, #222);
            border: 1px solid #444;
        }
        .badge.ml { background: linear-gradient(145deg, #1a2a3a, #0f1a2a); color: #7ab8ff; }
        .content { padding: 50px; background: #111111; }
        .upload-area { 
            border: 3px dashed #444; padding: 80px 30px; text-align: center; 
            margin: 40px 0; 
            background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
            border-radius: 20px; cursor: pointer; transition: all 0.4s ease;
            box-shadow: inset 0 4px 10px rgba(0,0,0,0.5), 0 5px 20px rgba(0,0,0,0.3);
            position: relative; overflow: hidden;
        }
        .upload-area::before {
            content: '';
            position: absolute;
            top: 0; left: -100%; width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
            transition: left 0.5s ease;
        }
        .upload-area:hover::before {
            left: 100%;
        }
        .upload-area:hover { 
            border-color: #666; 
            background: linear-gradient(145deg, #1a1a1a, #0f0f0f);
            transform: translateY(-5px);
            box-shadow: inset 0 4px 15px rgba(0,0,0,0.6), 0 10px 30px rgba(0,0,0,0.5);
        }
        .upload-area.dragover { 
            border-color: #888; 
            background: linear-gradient(145deg, #1a1a2a, #0f0f1a);
            transform: scale(1.02) translateY(-5px);
            box-shadow: inset 0 4px 20px rgba(0,50,100,0.3), 0 15px 40px rgba(0,0,0,0.7);
        }
        .upload-icon { 
            font-size: 5em; margin-bottom: 25px; color: #666; 
            transition: all 0.3s ease;
            text-shadow: 0 5px 15px rgba(0,0,0,0.5);
        }
        .upload-area:hover .upload-icon {
            color: #888;
            transform: scale(1.1);
        }
        .file-input-wrapper { position: relative; display: inline-block; }
        .file-input { 
            position: absolute; opacity: 0; width: 100%; height: 100%; cursor: pointer;
        }
        .file-input-label { 
            background: linear-gradient(145deg, #333, #222);
            color: #ccc; padding: 18px 35px; 
            border-radius: 30px; cursor: pointer; display: inline-block;
            transition: all 0.3s ease; font-weight: bold;
            box-shadow: 0 5px 20px rgba(0,0,0,0.5), inset 0 1px 2px rgba(255,255,255,0.1);
            border: 1px solid #444;
        }
        .file-input-label:hover { 
            background: linear-gradient(145deg, #444, #333);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.6), inset 0 1px 2px rgba(255,255,255,0.15);
        }
        .submit-btn { 
            background: linear-gradient(145deg, #2a2a2a, #1a1a1a);
            color: #ccc; padding: 18px 45px; border: 2px solid #444; border-radius: 30px; 
            cursor: pointer; font-size: 1.2em; font-weight: bold; width: 100%;
            margin: 25px 0; transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(0,0,0,0.5), inset 0 1px 2px rgba(255,255,255,0.05);
        }
        .submit-btn:hover { 
            background: linear-gradient(145deg, #333, #222);
            transform: translateY(-3px); 
            box-shadow: 0 10px 30px rgba(0,0,0,0.6), inset 0 1px 2px rgba(255,255,255,0.1);
            border-color: #555;
        }
        .submit-btn:disabled { 
            background: linear-gradient(145deg, #1a1a1a, #0f0f0f); 
            cursor: not-allowed; transform: none;
            color: #666; border-color: #333;
        }
        .status { 
            padding: 25px; margin: 25px 0; border-radius: 15px; text-align: center;
            font-weight: bold; font-size: 1.1em;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }
        .success { 
            background: linear-gradient(145deg, #1a2a1a, #0f1a0f); 
            border: 2px solid #2a4a2a; color: #7aff7a;
        }
        .error { 
            background: linear-gradient(145deg, #2a1a1a, #1a0f0f); 
            border: 2px solid #4a2a2a; color: #ff7a7a;
        }
        .processing { 
            background: linear-gradient(145deg, #1a2a3a, #0f1a2a); 
            border: 2px solid #2a4a6a; color: #7ab8ff;
        }
        .file-info { 
            background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
            padding: 20px; margin: 20px 0; border-radius: 12px;
            border-left: 4px solid #444;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.3), 0 2px 10px rgba(0,0,0,0.2);
            color: #ccc;
        }
        .progress-bar { 
            width: 100%; height: 12px; 
            background: linear-gradient(145deg, #0a0a0a, #1a1a1a);
            border-radius: 6px;
            overflow: hidden; margin: 15px 0;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.5);
        }
        .progress-fill { 
            height: 100%; 
            background: linear-gradient(90deg, #7ab8ff, #333, #666);
            transition: width 0.3s ease; width: 0%;
            position: relative; overflow: hidden;
        }
        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(45deg, 
                transparent 40%, 
                rgba(255,255,255,0.2) 50%, 
                transparent 60%);
            animation: progressShimmer 1.5s linear infinite;
        }
        @keyframes progressShimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        .session-display {
            text-align: center; margin-bottom: 25px; padding: 15px;
            background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
            border-radius: 10px; border: 1px solid #333;
            font-family: 'Courier New', monospace;
            color: #888;
        }
        .loading-spinner {
            border: 4px solid #222;
            border-top: 4px solid #666;
            border-radius: 50%;
            width: 40px; height: 40px;
            animation: spin 1s linear infinite;
            margin: 15px auto;
            box-shadow: 0 5px 15px rgba(0,0,0,0.5);
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        @media (max-width: 768px) {
            .content { padding: 30px; }
            .header h1 { font-size: 2.2em; }
            .upload-area { padding: 60px 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ACAD</h1>
            <p>Enhanced AI-Powered Certificate Upload</p>
            <div>
                <span class="badge ml">ML Analysis</span>
                <span class="badge">Forensic Detection</span>
            </div>
        </div>
        
        <div class="content">
            <div class="session-display">
                <strong>Session:</strong> <code>{{ session_id }}</code>
            </div>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📄</div>
                    <p style="font-size: 1.4em; margin: 20px 0; color: #ccc;"><strong>Drop certificate here for AI analysis</strong></p>
                    <p style="margin: 20px 0; color: #888;">or</p>
                    <div class="file-input-wrapper">
                        <input type="file" name="file" id="fileInput" class="file-input" accept="image/*,application/pdf" required>
                        <label for="fileInput" class="file-input-label">Choose File</label>
                    </div>
                </div>
                
                <div id="fileInfo" class="file-info" style="display: none;"></div>
                
                <button type="submit" id="submitBtn" class="submit-btn">
                    Upload & Analyze with AI/ML
                </button>
            </form>
            
            <div id="status" style="display: none;"></div>
        </div>
    </div>

    <script>
        const uploadForm = document.getElementById('uploadForm');
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const submitBtn = document.getElementById('submitBtn');
        const statusDiv = document.getElementById('status');
        const fileInfoDiv = document.getElementById('fileInfo');
        const sessionId = '{{ session_id }}';

        uploadArea.addEventListener('click', () => fileInput.click());
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            uploadArea.classList.add('dragover');
        }

        function unhighlight() {
            uploadArea.classList.remove('dragover');
        }

        uploadArea.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect(files[0]);
            }
        }

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        function handleFileSelect(file) {
            const maxSize = 50 * 1024 * 1024; // 50MB
            
            if (file.size > maxSize) {
                showStatus('error', 'File too large. Maximum size is 50MB.');
                return;
            }
            
            const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf'];
            if (!validTypes.includes(file.type)) {
                showStatus('error', 'Invalid file type. Please upload JPG, PNG, or PDF.');
                return;
            }
            
            fileInfoDiv.style.display = 'block';
            fileInfoDiv.innerHTML = `
                <strong>Selected File:</strong><br>
                <strong>Name:</strong> ${file.name}<br>
                <strong>Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB<br>
                <strong>Type:</strong> ${file.type}<br>
                <strong>Analysis:</strong> AI/ML + Traditional Forensics
            `;
            
            uploadArea.innerHTML = `
                <div class="upload-icon">✅</div>
                <p style="color: #7aff7a; font-size: 1.2em;"><strong>${file.name}</strong></p>
                <p style="color: #7ab8ff;">Ready for AI analysis</p>
            `;
        }

        function showStatus(type, message) {
            statusDiv.style.display = 'block';
            statusDiv.className = `status ${type}`;
            statusDiv.innerHTML = message;
        }

        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            if (!fileInput.files[0]) {
                showStatus('error', 'Please select a file first.');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('session_id', sessionId);
            
            submitBtn.disabled = true;
            submitBtn.textContent = 'Analyzing with AI/ML...';
            
            showStatus('processing', `
                <div class="loading-spinner"></div>
                <strong>Processing with Enhanced AI Analysis...</strong><br>
                <small>This may take 30-60 seconds for comprehensive analysis</small>
                <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
            `);
            
            // Enhanced progress simulation
            let progress = 0;
            const progressFill = document.getElementById('progressFill');
            const progressSteps = [
                { delay: 1000, progress: 15, text: 'Preprocessing image...' },
                { delay: 2000, progress: 35, text: 'Extracting ML features...' },
                { delay: 3000, progress: 55, text: 'Running OCR analysis...' },
                { delay: 4000, progress: 75, text: 'Detecting forgery patterns...' },
                { delay: 1000, progress: 90, text: 'Database comparison...' }
            ];
            
            let stepIndex = 0;
            const progressInterval = setInterval(() => {
                if (stepIndex < progressSteps.length) {
                    const step = progressSteps[stepIndex];
                    progress = step.progress;
                    progressFill.style.width = progress + '%';
                    
                    const statusText = statusDiv.innerHTML;
                    if (statusText.includes('Processing with Enhanced')) {
                        statusDiv.innerHTML = statusText.replace(
                            'This may take 30-60 seconds for comprehensive analysis',
                            step.text
                        );
                    }
                    stepIndex++;
                } else {
                    progress = Math.min(progress + Math.random() * 5, 95);
                    progressFill.style.width = progress + '%';
                }
            }, 800);
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                clearInterval(progressInterval);
                progressFill.style.width = '100%';
                
                const result = await response.json();
                
                if (response.ok) {
                    showStatus('success', `
                        <h3>🎉 Enhanced Analysis Complete!</h3>
                        <p style="margin: 15px 0;"><strong>AI/ML processing completed successfully</strong></p>
                        <p style="color: #7ab8ff;">You can now close this page and return to the verification screen.</p>
                    `);
                } else {
                    showStatus('error', result.error || 'Enhanced analysis failed');
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Upload & Analyze with AI/ML';
                }
            } catch (error) {
                clearInterval(progressInterval);
                showStatus('error', 'Network error: ' + error.message);
                submitBtn.disabled = false;
                submitBtn.textContent = 'Upload & Analyze with AI/ML';
            }
        });
    </script>
</body>
</html>"""

# Flask Routes with enhanced functionality

@app.route('/')
def index():
    cleanup_expired_sessions()
    return redirect(url_for('start_verification'))

@app.route('/start_verification')
def start_verification():
    """Generate new verification session with enhanced features"""
    cleanup_expired_sessions()
    
    session_id = str(uuid.uuid4())
    created_time = datetime.now()
    expires_time = created_time + timedelta(seconds=SESSION_TIMEOUT)
    
    upload_url = request.url_root + f"upload/{session_id}"
    qr_code_b64 = generate_qr_code(upload_url)
    
    sessions[session_id] = {
        'status': 'pending',
        'result': None,
        'created_at': created_time,
        'expires_at': expires_time
    }
    
    logger.info(f"Created enhanced session {session_id}, expires at {expires_time}")
    
    return render_template_string(ENHANCED_VERIFIER_TEMPLATE,
                         session_id=session_id, 
                         qr_code=qr_code_b64,
                         created_time=created_time.strftime('%H:%M:%S'),
                         expires_time=expires_time.strftime('%H:%M:%S'))

@app.route('/upload/<session_id>')
def upload_page(session_id):
    """Enhanced upload page"""
    cleanup_expired_sessions()
    
    if session_id not in sessions:
        return "Session expired or invalid", 404
    
    session_data = sessions[session_id]
    if datetime.now() > session_data['expires_at']:
        del sessions[session_id]
        return "Session expired", 408
    
    return render_template_string(ENHANCED_UPLOAD_TEMPLATE, session_id=session_id)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Enhanced certificate analysis with ML and database integration"""
    try:
        session_id = request.form.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid or expired session'}), 400
        
        session_data = sessions[session_id]
        if datetime.now() > session_data['expires_at']:
            del sessions[session_id]
            return jsonify({'error': 'Session expired'}), 408
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        allowed_types = {'image/jpeg', 'image/png', 'image/jpg', 'application/pdf'}
        if file.content_type not in allowed_types:
            return jsonify({'error': 'Invalid file type. Only JPG, PNG, and PDF are supported.'}), 400
        
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        safe_filename = f"{session_id}_{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        file.save(file_path)
        logger.info(f"File saved for enhanced analysis: {file_path} ({os.path.getsize(file_path)} bytes)")
        
        # Run enhanced analysis with ML and database integration
        result = enhanced_analyze_certificate(file_path, session_id, ml_analyzer)
        
        sessions[session_id]['status'] = 'done'
        sessions[session_id]['result'] = result
        
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up {file_path}: {e}")
        
        logger.info(f"Enhanced analysis complete for {session_id}: {result['verdict']} (combined: {result['tampering_score']})")
        
        return jsonify({
            'status': 'success',
            'message': 'Enhanced AI/ML analysis completed successfully',
            'session_id': session_id,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {e}")
        if session_id and session_id in sessions:
            sessions[session_id]['status'] = 'error'
            sessions[session_id]['error'] = str(e)
        return jsonify({'error': f'Enhanced analysis failed: {str(e)}'}), 500

@app.route('/results/<session_id>')
def get_results(session_id):
    """Get enhanced analysis results"""
    cleanup_expired_sessions()
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid or expired session'}), 404
    
    session_data = sessions[session_id]
    
    if datetime.now() > session_data['expires_at']:
        del sessions[session_id]
        return jsonify({'error': 'Session expired'}), 408
    
    response_data = {
        'status': session_data['status'],
        'result': session_data.get('result'),
        'session_id': session_id
    }
    
    if session_data['status'] == 'error':
        response_data['error'] = session_data.get('error', 'Unknown error occurred')
    
    return jsonify(response_data)

@app.route('/health')
def health_check():
    """Enhanced system health check"""
    cleanup_expired_sessions()
    
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        tesseract_status = "available"
    except Exception as e:
        tesseract_version = None
        tesseract_status = f"error: {e}"
    
    # Check database connection
    try:
        with DatabaseManager() as db:
            db_status = "connected"
            total_certs = db.session.query(CertificateRecord).count()
    except Exception as e:
        db_status = f"error: {e}"
        total_certs = 0
    
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(sessions),
        'tesseract_version': str(tesseract_version) if tesseract_version else None,
        'tesseract_status': tesseract_status,
        'database_status': db_status,
        'total_certificates': total_certs,
        'ml_available': ML_AVAILABLE,
        'sklearn_available': SKLEARN_AVAILABLE,
        'advanced_analysis': ADVANCED_ANALYSIS,
        'pdf_support': PDF_SUPPORT,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'timestamp': datetime.now().isoformat(),
        'session_timeout': SESSION_TIMEOUT
    })

@app.route('/stats')
def get_stats():
    """Get system statistics"""
    try:
        with DatabaseManager() as db:
            total_certs = db.session.query(CertificateRecord).count()
            authentic_certs = db.session.query(CertificateRecord).filter(
                CertificateRecord.tampering_score < 30
            ).count()
            suspicious_certs = db.session.query(CertificateRecord).filter(
                CertificateRecord.tampering_score.between(30, 70)
            ).count()
            forgery_certs = db.session.query(CertificateRecord).filter(
                CertificateRecord.tampering_score >= 70
            ).count()
            
            # Get recent activity
            recent_certs = db.session.query(CertificateRecord).filter(
                CertificateRecord.created_at >= datetime.now() - timedelta(days=7)
            ).count()
            
            return jsonify({
                'total_certificates': total_certs,
                'authentic': authentic_certs,
                'suspicious': suspicious_certs,
                'likely_forgery': forgery_certs,
                'recent_activity': recent_certs,
                'active_sessions': len(sessions)
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Acad Enhanced Certificate Verification System - Starting...")
    
    # System checks
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract OCR detected: {tesseract_version}")
    except Exception as e:
        print(f"✗ Tesseract OCR not found: {e}")
        print("\nInstallation Instructions:")
        print("  Ubuntu/Debian: sudo apt update && sudo apt install tesseract-ocr poppler-utils")
        print("  macOS: brew install tesseract poppler")
        print("  Windows: Download from https://github.com/tesseract-ocr/tesseract")
        sys.exit(1)
    
    # Check ML dependencies
    if ML_AVAILABLE:
        print("✓ TensorFlow available for deep learning")
    else:
        print("⚠ TensorFlow not available - ML features limited")
    
    if SKLEARN_AVAILABLE:
        print("✓ Scikit-learn available for ML algorithms")
    else:
        print("⚠ Scikit-learn not available - ML features limited")
    
    if PDF_SUPPORT:
        print("✓ PDF2Image available for PDF processing")
    else:
        print("⚠ PDF2Image not available - PDF support limited")
    
    if ADVANCED_ANALYSIS:
        print("✓ Scikit-image available for advanced forensics")
    else:
        print("⚠ Scikit-image not available - advanced analysis limited")
    
    # Database setup
    try:
        with DatabaseManager() as db:
            total_records = db.session.query(CertificateRecord).count()
            print(f"✓ Database connected ({total_records} existing records)")
    except Exception as e:
        print(f"⚠ Database warning: {e}")
    
    # Create directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['ML_MODELS_PATH'], exist_ok=True)
    print(f"✓ Directories ready")
    
    print(f"\n🚀 Enhanced Server starting...")
    print(f"📱 Open http://127.0.0.1:5000 to begin verification")
    print(f"⏱️  Session timeout: {SESSION_TIMEOUT} seconds")
    print(f"🤖 AI/ML: {'Enabled' if ML_AVAILABLE and SKLEARN_AVAILABLE else 'Limited'}")
    print(f"🗄️  Database: {'Connected' if 'Database connected' in locals() else 'Available'}")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)