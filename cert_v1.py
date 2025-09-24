#!/usr/bin/env python3
"""
Production-Ready Certificate Authenticity Verification System
Single-file Flask app with advanced OCR + tampering detection

Requirements:
pip install flask pytesseract opencv-python pillow pyzbar qrcode[pil] numpy scikit-image pdf2image

System Requirements:
- Install Tesseract OCR on your system:
  Ubuntu: sudo apt install tesseract-ocr poppler-utils
  macOS: brew install tesseract poppler
  Windows: Download from https://github.com/tesseract-ocr/tesseract and install poppler

Usage:
python cert_verifier.py
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
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
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

# Try to import optional dependencies
try:
    from skimage import feature, measure, morphology
    from skimage.filters import threshold_otsu
    from skimage.feature import local_binary_pattern
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

# In-memory storage with session expiry
sessions = {}  # session_id -> {status, result, created_at, expires_at}

# Session cleanup interval (5 minutes)
SESSION_TIMEOUT = 300

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

# HTML Templates with improved UI
VERIFIER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Certificate Authenticity Verifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px; color: #333;
        }
        .container { 
            max-width: 900px; margin: 0 auto; background: white; 
            border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
            color: white; padding: 30px; text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 1.1em; }
        .content { padding: 40px; }
        .qr-section { 
            text-align: center; margin: 30px 0; padding: 30px;
            background: #f8f9ff; border-radius: 15px; border: 2px dashed #4facfe;
        }
        .qr-section h2 { color: #4facfe; margin-bottom: 20px; font-size: 1.8em; }
        .qr-code { 
            display: inline-block; padding: 20px; background: white;
            border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .session-info { 
            margin: 20px 0; padding: 15px; background: #e8f4fd;
            border-radius: 8px; font-family: monospace;
        }
        .status { 
            padding: 20px; margin: 20px 0; border-radius: 10px; 
            font-weight: bold; text-align: center; font-size: 1.1em;
        }
        .pending { background: linear-gradient(45deg, #fff3cd, #ffeaa7); border: 1px solid #ffc107; }
        .processing { background: linear-gradient(45deg, #cce7ff, #74b9ff); border: 1px solid #0084ff; }
        .success { background: linear-gradient(45deg, #d4edda, #55efc4); border: 1px solid #28a745; }
        .error { background: linear-gradient(45deg, #f8d7da, #fd79a8); border: 1px solid #dc3545; }
        .result-container { margin-top: 30px; }
        .result-box { 
            background: white; padding: 25px; margin: 15px 0; border-radius: 15px; 
            box-shadow: 0 5px 20px rgba(0,0,0,0.08); border-left: 5px solid;
        }
        .authentic { border-left-color: #28a745; }
        .suspicious { border-left-color: #ffc107; }
        .forgery { border-left-color: #dc3545; }
        .field-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px; margin: 20px 0;
        }
        .field-item { 
            padding: 15px; background: #f8f9fa; border-radius: 8px;
            border-left: 4px solid #4facfe;
        }
        .field-label { font-weight: bold; color: #666; font-size: 0.9em; }
        .field-value { font-size: 1.1em; margin-top: 5px; }
        .score-bar { 
            width: 100%; height: 20px; background: #e9ecef; border-radius: 10px;
            overflow: hidden; margin: 10px 0;
        }
        .score-fill { 
            height: 100%; transition: width 0.5s ease;
            background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
        }
        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4facfe;
            border-radius: 50%;
            width: 40px; height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .instructions { 
            background: #e3f2fd; padding: 20px; border-radius: 10px; margin: 20px 0;
            border-left: 4px solid #2196f3;
        }
        .instructions h3 { color: #1976d2; margin-bottom: 15px; }
        .step { margin: 10px 0; padding-left: 25px; position: relative; }
        .step:before { 
            content: counter(step-counter); counter-increment: step-counter;
            position: absolute; left: 0; top: 0; 
            background: #2196f3; color: white; border-radius: 50%;
            width: 20px; height: 20px; text-align: center; font-size: 12px; line-height: 20px;
        }
        .steps { counter-reset: step-counter; }
        @media (max-width: 768px) {
            .content { padding: 20px; }
            .header h1 { font-size: 2em; }
            .field-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Certificate Verifier</h1>
            <p>Advanced OCR-based document authenticity verification</p>
        </div>
        
        <div class="content">
            <div class="instructions">
                <h3>How to Verify a Certificate</h3>
                <div class="steps">
                    <div class="step">Scan the QR code below with your mobile device</div>
                    <div class="step">Upload the certificate image or PDF</div>
                    <div class="step">Wait for AI-powered analysis to complete</div>
                    <div class="step">Review the detailed verification results</div>
                </div>
            </div>
            
            <div class="qr-section">
                <h2>Scan to Upload Certificate</h2>
                <div class="qr-code">
                    <img src="data:image/png;base64,{{ qr_code }}" alt="Verification QR Code">
                </div>
                <div class="session-info">
                    <strong>Session ID:</strong> {{ session_id }}<br>
                    <strong>Created:</strong> {{ created_time }}<br>
                    <strong>Expires:</strong> {{ expires_time }}
                </div>
            </div>
            
            <div id="status" class="status pending">
                <div class="loading-spinner"></div>
                <strong>Status:</strong> Waiting for certificate upload...
            </div>
            
            <div id="result" class="result-container" style="display: none;">
                <h3>Verification Results</h3>
                <div id="result-content" class="result-box"></div>
            </div>
        </div>
    </div>

    <script>
        const sessionId = '{{ session_id }}';
        let pollInterval;
        let isProcessing = false;

        function startPolling() {
            pollInterval = setInterval(checkResults, 1500);
        }

        function stopPolling() {
            if (pollInterval) {
                clearInterval(pollInterval);
            }
        }

        async function checkResults() {
            try {
                const response = await fetch(`/results/${sessionId}`);
                const data = await response.json();
                
                const statusDiv = document.getElementById('status');
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('result-content');

                if (data.status === 'processing' && !isProcessing) {
                    isProcessing = true;
                    statusDiv.className = 'status processing';
                    statusDiv.innerHTML = `
                        <div class="loading-spinner"></div>
                        <strong>Status:</strong> Analyzing certificate with AI-powered OCR...
                    `;
                } else if (data.status === 'done') {
                    stopPolling();
                    statusDiv.className = 'status success';
                    statusDiv.innerHTML = '<strong>Analysis Complete!</strong> Results are ready.';
                    
                    resultDiv.style.display = 'block';
                    const result = data.result;
                    
                    let verdictClass = 'authentic';
                    let verdictIcon = 'PASS';
                    let verdictText = 'AUTHENTIC';
                    
                    if (result.tampering_score >= 70) {
                        verdictClass = 'forgery';
                        verdictIcon = 'FAIL';
                        verdictText = 'LIKELY FORGERY';
                    } else if (result.tampering_score >= 35) {
                        verdictClass = 'suspicious';
                        verdictIcon = 'WARN';
                        verdictText = 'SUSPICIOUS';
                    }
                    
                    const scorePercent = Math.min(result.tampering_score, 100);
                    
                    resultContent.className = `result-box ${verdictClass}`;
                    resultContent.innerHTML = `
                        <div style="text-align: center; margin-bottom: 25px;">
                            <h2>${verdictIcon} - ${verdictText}</h2>
                            <div style="margin: 15px 0;">
                                <div class="score-bar">
                                    <div class="score-fill" style="width: ${scorePercent}%"></div>
                                </div>
                                <div>Tampering Risk Score: <strong>${result.tampering_score}/100</strong></div>
                            </div>
                        </div>
                        
                        <h4>Extracted Information</h4>
                        <div class="field-grid">
                            <div class="field-item">
                                <div class="field-label">Full Name</div>
                                <div class="field-value">${result.name || 'Not detected'}</div>
                            </div>
                            <div class="field-item">
                                <div class="field-label">Roll/Registration Number</div>
                                <div class="field-value">${result.roll_no || 'Not detected'}</div>
                            </div>
                            <div class="field-item">
                                <div class="field-label">Year/Date</div>
                                <div class="field-value">${result.year || 'Not detected'}</div>
                            </div>
                            <div class="field-item">
                                <div class="field-label">Institution/Board</div>
                                <div class="field-value">${result.institution || 'Not detected'}</div>
                            </div>
                        </div>
                        
                        <h4>Technical Analysis</h4>
                        <div class="field-grid">
                            <div class="field-item">
                                <div class="field-label">OCR Confidence</div>
                                <div class="field-value">${result.ocr_confidence}%</div>
                            </div>
                            <div class="field-item">
                                <div class="field-label">Text Extracted</div>
                                <div class="field-value">${result.text_length} characters</div>
                            </div>
                            <div class="field-item">
                                <div class="field-label">QR Codes Found</div>
                                <div class="field-value">${result.qr_codes_found}</div>
                            </div>
                            <div class="field-item">
                                <div class="field-label">Image Quality</div>
                                <div class="field-value">${result.image_quality}</div>
                            </div>
                        </div>
                        
                        <h4>Analysis Notes</h4>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 10px;">
                            ${result.notes}
                        </div>
                        
                        <hr style="margin: 20px 0;">
                        <div style="text-align: center; color: #666; font-size: 0.9em;">
                            <strong>Processed:</strong> ${result.processed_at}<br>
                            <em>This is an automated analysis. For official verification, contact the issuing authority.</em>
                        </div>
                    `;
                } else if (data.status === 'error') {
                    stopPolling();
                    statusDiv.className = 'status error';
                    statusDiv.innerHTML = '<strong>Error:</strong> ' + (data.error || 'Analysis failed');
                }
            } catch (error) {
                console.error('Error checking results:', error);
            }
        }

        // Start polling when page loads
        startPolling();
        
        // Clean up on page unload
        window.addEventListener('beforeunload', stopPolling);
    </script>
</body>
</html>
"""

UPLOAD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Upload Certificate - Verifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px; color: #333;
        }
        .container { 
            max-width: 600px; margin: 0 auto; background: white; 
            border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
            color: white; padding: 30px; text-align: center;
        }
        .header h1 { font-size: 2.2em; margin-bottom: 10px; }
        .content { padding: 40px; }
        .upload-area { 
            border: 3px dashed #4facfe; padding: 60px 20px; text-align: center; 
            margin: 30px 0; background: linear-gradient(45deg, #f8f9ff, #e8f4fd);
            border-radius: 15px; cursor: pointer; transition: all 0.3s ease;
        }
        .upload-area:hover { border-color: #2196f3; background: linear-gradient(45deg, #e8f4fd, #d1edff); }
        .upload-area.dragover { 
            border-color: #0066cc; background: linear-gradient(45deg, #cce7ff, #bbdefb);
            transform: scale(1.02);
        }
        .upload-icon { font-size: 4em; margin-bottom: 20px; color: #4facfe; }
        .file-input-wrapper { position: relative; display: inline-block; }
        .file-input { 
            position: absolute; opacity: 0; width: 100%; height: 100%; cursor: pointer;
        }
        .file-input-label { 
            background: #4facfe; color: white; padding: 15px 30px; 
            border-radius: 25px; cursor: pointer; display: inline-block;
            transition: background 0.3s ease; font-weight: bold;
        }
        .file-input-label:hover { background: #2196f3; }
        .submit-btn { 
            background: linear-gradient(45deg, #4facfe, #00f2fe);
            color: white; padding: 15px 40px; border: none; border-radius: 25px; 
            cursor: pointer; font-size: 1.1em; font-weight: bold; width: 100%;
            margin: 20px 0; transition: transform 0.3s ease;
        }
        .submit-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4); }
        .submit-btn:disabled { 
            background: #6c757d; cursor: not-allowed; transform: none;
        }
        .status { 
            padding: 20px; margin: 20px 0; border-radius: 10px; text-align: center;
            font-weight: bold;
        }
        .success { background: linear-gradient(45deg, #d4edda, #55efc4); border: 1px solid #28a745; color: #155724; }
        .error { background: linear-gradient(45deg, #f8d7da, #fd79a8); border: 1px solid #dc3545; color: #721c24; }
        .processing { background: linear-gradient(45deg, #cce7ff, #74b9ff); border: 1px solid #0084ff; color: #004085; }
        .file-info { 
            background: #e8f4fd; padding: 15px; margin: 15px 0; border-radius: 8px;
            border-left: 4px solid #4facfe;
        }
        .progress-bar { 
            width: 100%; height: 6px; background: #e9ecef; border-radius: 3px;
            overflow: hidden; margin: 10px 0;
        }
        .progress-fill { 
            height: 100%; background: linear-gradient(90deg, #4facfe, #00f2fe);
            transition: width 0.3s ease; width: 0%;
        }
        .tips { 
            background: #fff3cd; padding: 20px; border-radius: 10px; margin: 20px 0;
            border-left: 4px solid #ffc107;
        }
        .tips h3 { color: #856404; margin-bottom: 15px; }
        .tips ul { margin-left: 20px; }
        .tips li { margin: 8px 0; color: #856404; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Upload Certificate</h1>
            <p>AI-powered document verification</p>
        </div>
        
        <div class="content">
            <div style="text-align: center; margin-bottom: 20px;">
                <strong>Session:</strong> <code>{{ session_id }}</code>
            </div>
            
            <div class="tips">
                <h3>Upload Tips for Best Results</h3>
                <ul>
                    <li>Use high-resolution images (min 1200px width)</li>
                    <li>Ensure good lighting and avoid shadows</li>
                    <li>Keep the certificate flat and fully visible</li>
                    <li>Supported formats: JPG, PNG, PDF</li>
                </ul>
            </div>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📄</div>
                    <p style="font-size: 1.2em; margin: 15px 0;"><strong>Drop certificate here</strong></p>
                    <p style="margin: 15px 0;">or</p>
                    <div class="file-input-wrapper">
                        <input type="file" name="file" id="fileInput" class="file-input" accept="image/*,application/pdf" required>
                        <label for="fileInput" class="file-input-label">Choose File</label>
                    </div>
                </div>
                
                <div id="fileInfo" class="file-info" style="display: none;"></div>
                
                <button type="submit" id="submitBtn" class="submit-btn">
                    Upload & Analyze Certificate
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

        // Drag and drop functionality
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
                <strong>Type:</strong> ${file.type}
            `;
            
            uploadArea.innerHTML = `
                <div class="upload-icon">✅</div>
                <p><strong>${file.name}</strong></p>
                <p>Ready to upload</p>
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
            submitBtn.textContent = 'Uploading & Analyzing...';
            
            showStatus('processing', `
                Processing your certificate...<br>
                <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
                This may take 30-60 seconds for detailed analysis.
            `);
            
            // Simulate progress
            let progress = 0;
            const progressFill = document.getElementById('progressFill');
            const progressInterval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                progressFill.style.width = progress + '%';
            }, 500);
            
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
                        <h3>Analysis Complete!</h3>
                        <p>Your certificate has been successfully processed.</p>
                        <p><strong>You can now close this page.</strong></p>
                        <p>The verifier will see the results automatically.</p>
                    `);
                } else {
                    showStatus('error', result.error || 'Upload failed');
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Upload & Analyze Certificate';
                }
            } catch (error) {
                clearInterval(progressInterval);
                showStatus('error', 'Network error: ' + error.message);
                submitBtn.disabled = false;
                submitBtn.textContent = 'Upload & Analyze Certificate';
            }
        });
    </script>
</body>
</html>
"""

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

def convert_pdf_to_image(pdf_path):
    """Convert PDF first page to image"""
    try:
        if PDF_SUPPORT:
            # Try using pdf2image
            images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
            if images:
                img_path = pdf_path.replace('.pdf', '_converted.jpg')
                images[0].save(img_path, 'JPEG', quality=95)
                return img_path
        else:
            logger.warning("pdf2image not available, trying alternative method")
        
        # Alternative using PIL for simple PDFs
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
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None, None
            
        # Convert to RGB for PIL operations
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Enhance contrast and sharpness
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        # Convert back to OpenCV format
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Save enhanced image
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
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # Calculate contrast using standard deviation
        contrast = np.std(img)
        
        # Determine overall quality
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
    
    # Multiple OCR configurations for better accuracy
    configs = [
        '--oem 3 --psm 6',  # Uniform block of text
        '--oem 3 --psm 4',  # Single column of text
        '--oem 3 --psm 3',  # Fully automatic page segmentation
        '--oem 1 --psm 6'   # Neural nets LSTM engine
    ]
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(image_path, config=config)
            if text.strip():
                # Get confidence data
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
    
    # Return the result with highest confidence and reasonable word count
    if results:
        best_result = max(results, key=lambda x: (x['confidence'], x['word_count']))
        return best_result['text'], best_result['confidence']
    
    return "", 0

def scan_for_qr_codes(image_path):
    """Enhanced QR code detection"""
    try:
        # Try multiple image preprocessing approaches
        img_original = cv2.imread(image_path)
        if img_original is None:
            return []
            
        all_qr_data = []
        
        # Method 1: Original image
        decoded_objects = pyzbar.decode(img_original)
        for obj in decoded_objects:
            all_qr_data.append({
                'data': obj.data.decode('utf-8', errors='ignore'),
                'type': obj.type,
                'method': 'original'
            })
        
        # Method 2: Grayscale
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
        
        # Method 3: Enhanced contrast
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
    
    # Clean and normalize text
    text = raw_text.upper().replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    
    # Enhanced Name extraction patterns
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
            # Validate name (should have at least 2 words, reasonable length)
            if len(name.split()) >= 2 and 5 <= len(name) <= 50:
                # Remove common false positives
                if not any(word in name for word in ['UNIVERSITY', 'COLLEGE', 'BOARD', 'EXAMINATION', 'CERTIFICATE']):
                    fields['name'] = name.title()
                    break
        if fields['name']:
            break
    
    # Enhanced Roll/Registration number patterns
    roll_patterns = [
        r'(?:ROLL\s*(?:NO|NUMBER)|REG\s*(?:NO|NUMBER)|REGISTRATION\s*(?:NO|NUMBER)|ENROLLMENT\s*(?:NO|NUMBER))\s*[:\-]?\s*([A-Z0-9\-\/]{3,25})',
        r'(?:ROLL|REG|REGISTRATION|ENROLLMENT)\s*[:\-]?\s*([A-Z0-9\-\/]{3,25})',
        r'(?:ID|STUDENT\s*ID)\s*[:\-]?\s*([A-Z0-9\-\/]{3,25})',
        r'([A-Z]{1,3}\d{4,10})',  # Pattern like AB123456
        r'(\d{4,12})',  # Pure numeric patterns
    ]
    
    for pattern in roll_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            roll = match.strip()
            # Validate roll number
            if 3 <= len(roll) <= 25 and re.search(r'\d', roll):
                fields['roll_no'] = roll
                break
        if fields['roll_no']:
            break
    
    # Enhanced Year extraction
    year_patterns = [
        r'(?:YEAR|PASSING\s*YEAR|PASSED\s*IN|ACADEMIC\s*YEAR)\s*[:\-]?\s*(20\d{2}|19[89]\d)',
        r'(?:MAY|JUNE|APRIL|NOVEMBER|DECEMBER|MARCH)\s*(20\d{2})',
        r'(20[0-2]\d)(?:\s*-\s*20[0-2]\d)?',  # Year ranges like 2020-2021
        r'(19[89]\d)',  # 1980s-1990s
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
        # Prefer most recent reasonable year
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
    
    # 1. Field Completeness Analysis (0-40 points)
    required_fields = ['name', 'roll_no', 'year']
    missing_fields = [field for field in required_fields if not extracted_fields.get(field)]
    
    if missing_fields:
        score_penalty = len(missing_fields) * 15
        tampering_score += score_penalty
        issues.append(f"Missing critical fields: {', '.join(missing_fields)} (+{score_penalty})")
    
    # 2. OCR Confidence Analysis (0-25 points)
    if ocr_confidence < 60:
        score_penalty = int((60 - ocr_confidence) / 2)
        tampering_score += score_penalty
        issues.append(f"Low OCR confidence: {ocr_confidence:.1f}% (+{score_penalty})")
    
    # 3. Image Quality Analysis (0-20 points)
    if quality_metrics['quality'] in ['poor', 'fair']:
        score_penalty = 15 if quality_metrics['quality'] == 'poor' else 8
        tampering_score += score_penalty
        issues.append(f"Poor image quality: {quality_metrics['quality']} (+{score_penalty})")
    
    # 4. Date Consistency Checks (0-25 points)
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
                # Very recent certificates might be suspicious
                tampering_score += 5
                issues.append(f"Very recent certificate: {year} (+5)")
        except ValueError:
            pass
    
    # 5. Text Pattern Analysis (0-15 points)
    text_length = len(raw_text.strip())
    if text_length < 50:
        score_penalty = 15
        tampering_score += score_penalty
        issues.append(f"Insufficient text extracted: {text_length} chars (+{score_penalty})")
    elif text_length > 5000:
        score_penalty = 5
        tampering_score += score_penalty
        issues.append(f"Unusually verbose certificate (+{score_penalty})")
    
    # 6. Image Metadata Analysis (0-15 points)
    try:
        with Image.open(image_path) as img:
            # Check for editing software in EXIF data
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
            
            # Check image dimensions
            width, height = img.size
            if width < 800 or height < 600:
                tampering_score += 8
                issues.append(f"Low resolution image: {width}x{height} (+8)")
    except Exception as e:
        logger.debug(f"Metadata analysis failed: {e}")
    
    # 7. Advanced Image Analysis (0-20 points)
    if ADVANCED_ANALYSIS:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Check for copy-paste artifacts using texture analysis
                # Calculate local binary patterns for texture analysis
                
                # Divide image into regions and analyze texture consistency
                h, w = img.shape
                regions = [
                    img[0:h//2, 0:w//2],      # Top-left
                    img[0:h//2, w//2:w],      # Top-right
                    img[h//2:h, 0:w//2],      # Bottom-left
                    img[h//2:h, w//2:w]       # Bottom-right
                ]
                
                texture_variations = []
                for region in regions:
                    if region.size > 0:
                        lbp = local_binary_pattern(region, 8, 1.5, method='uniform')
                        hist, _ = np.histogram(lbp.ravel(), bins=10)
                        texture_variations.append(np.std(hist))
                
                if len(texture_variations) > 1:
                    texture_inconsistency = np.std(texture_variations)
                    if texture_inconsistency > 100:  # Threshold for suspicious texture variation
                        score_penalty = min(int(texture_inconsistency / 20), 15)
                        tampering_score += score_penalty
                        issues.append(f"Texture inconsistencies detected (+{score_penalty})")
        
        except Exception as e:
            logger.debug(f"Advanced image analysis failed: {e}")
    else:
        logger.debug("Advanced analysis not available - scikit-image not installed")
    
    # 8. Name Validation (0-10 points)
    if extracted_fields.get('name'):
        name = extracted_fields['name']
        # Check for unusual patterns in names
        if re.search(r'\d', name):  # Numbers in name
            tampering_score += 8
            issues.append("Numbers found in name field (+8)")
        elif len(name.split()) < 2:  # Single word name
            tampering_score += 5
            issues.append("Incomplete name detected (+5)")
        elif len(name) > 40:  # Very long name
            tampering_score += 3
            issues.append("Unusually long name (+3)")
    
    # 9. QR Code Analysis
    if qr_data:
        issues.append(f"QR codes found: {len(qr_data)}")
        # For now, presence of QR codes is neutral to slightly positive
        tampering_score = max(0, tampering_score - 2)
    
    # Cap the score at 100
    tampering_score = min(tampering_score, 100)
    
    # Determine final verdict
    if tampering_score < 20:
        verdict = "authentic"
    elif tampering_score < 40:
        verdict = "likely_authentic"
    elif tampering_score < 70:
        verdict = "suspicious"
    else:
        verdict = "likely_forgery"
    
    # Compile notes
    if not issues:
        issues.append("All standard verification checks passed")
    
    notes = " • ".join(issues)
    
    return tampering_score, verdict, notes

def analyze_certificate(file_path, session_id):
    """Main comprehensive certificate analysis pipeline"""
    try:
        logger.info(f"Starting comprehensive analysis for session {session_id}")
        start_time = time.time()
        
        # Update session status to processing
        if session_id in sessions:
            sessions[session_id]['status'] = 'processing'
        
        # Handle PDF conversion
        image_path = file_path
        if file_path.lower().endswith('.pdf'):
            logger.info("Converting PDF to image...")
            converted_path = convert_pdf_to_image(file_path)
            if converted_path:
                image_path = converted_path
            else:
                raise Exception("Failed to convert PDF to image")
        
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
        logger.info(f"OCR extracted {len(raw_text)} characters with {ocr_confidence:.1f}% confidence")
        
        # QR code detection
        logger.info("Scanning for QR codes...")
        qr_data = scan_for_qr_codes(image_path)
        
        # Extract structured fields
        logger.info("Extracting certificate fields...")
        extracted_fields = extract_certificate_fields(raw_text)
        logger.info(f"Extracted fields: {extracted_fields}")
        
        # Comprehensive tampering analysis
        logger.info("Performing tampering analysis...")
        tampering_score, verdict, notes = advanced_tampering_detection(
            image_path, raw_text, ocr_confidence, qr_data, extracted_fields, quality_metrics
        )
        
        # Compile final result
        processing_time = round(time.time() - start_time, 2)
        
        result = {
            'name': extracted_fields.get('name'),
            'roll_no': extracted_fields.get('roll_no'),
            'year': extracted_fields.get('year'),
            'institution': extracted_fields.get('institution'),
            'course': extracted_fields.get('course'),
            'grade': extracted_fields.get('grade'),
            'tampering_score': tampering_score,
            'verdict': verdict,
            'notes': notes,
            'ocr_confidence': round(ocr_confidence, 1),
            'text_length': len(raw_text),
            'qr_codes_found': len(qr_data),
            'image_quality': quality_metrics['quality'],
            'processing_time': processing_time,
            'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Clean up temporary files
        try:
            if enhanced_path and os.path.exists(enhanced_path):
                os.remove(enhanced_path)
            if image_path != file_path and os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            logger.debug(f"Cleanup warning: {e}")
        
        logger.info(f"Analysis completed in {processing_time}s: {verdict} (score: {tampering_score})")
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

# Flask Routes

@app.route('/')
def index():
    cleanup_expired_sessions()
    return redirect(url_for('start_verification'))

@app.route('/start_verification')
def start_verification():
    """Generate new verification session with expiry"""
    cleanup_expired_sessions()
    
    session_id = str(uuid.uuid4())
    created_time = datetime.now()
    expires_time = created_time + timedelta(seconds=SESSION_TIMEOUT)
    
    # Create upload URL for QR code
    upload_url = request.url_root + f"upload/{session_id}"
    
    # Generate QR code
    qr_code_b64 = generate_qr_code(upload_url)
    
    # Store session with expiry
    sessions[session_id] = {
        'status': 'pending',
        'result': None,
        'created_at': created_time,
        'expires_at': expires_time
    }
    
    logger.info(f"Created session {session_id}, expires at {expires_time}")
    
    return render_template_string(VERIFIER_TEMPLATE, 
                                session_id=session_id, 
                                qr_code=qr_code_b64,
                                created_time=created_time.strftime('%H:%M:%S'),
                                expires_time=expires_time.strftime('%H:%M:%S'))

@app.route('/upload/<session_id>')
def upload_page(session_id):
    """Upload page with session validation"""
    cleanup_expired_sessions()
    
    if session_id not in sessions:
        return "Session expired or invalid", 404
    
    session_data = sessions[session_id]
    if datetime.now() > session_data['expires_at']:
        del sessions[session_id]
        return "Session expired", 408
    
    return render_template_string(UPLOAD_TEMPLATE, session_id=session_id)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process uploaded certificate with comprehensive analysis"""
    try:
        session_id = request.form.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid or expired session'}), 400
        
        # Check session expiry
        session_data = sessions[session_id]
        if datetime.now() > session_data['expires_at']:
            del sessions[session_id]
            return jsonify({'error': 'Session expired'}), 408
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type and size
        allowed_types = {'image/jpeg', 'image/png', 'image/jpg', 'application/pdf'}
        if file.content_type not in allowed_types:
            return jsonify({'error': 'Invalid file type. Only JPG, PNG, and PDF are supported.'}), 400
        
        # Save uploaded file securely
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        safe_filename = f"{session_id}_{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        file.save(file_path)
        logger.info(f"File saved: {file_path} ({os.path.getsize(file_path)} bytes)")
        
        # Run comprehensive analysis
        result = analyze_certificate(file_path, session_id)
        
        # Update session with results
        sessions[session_id]['status'] = 'done'
        sessions[session_id]['result'] = result
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up {file_path}: {e}")
        
        logger.info(f"Analysis complete for {session_id}: {result['verdict']}")
        
        return jsonify({
            'status': 'success', 
            'message': 'Analysis completed successfully',
            'session_id': session_id,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        if session_id and session_id in sessions:
            sessions[session_id]['status'] = 'error'
            sessions[session_id]['error'] = str(e)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/results/<session_id>')
def get_results(session_id):
    """Get analysis results with session validation"""
    cleanup_expired_sessions()
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid or expired session'}), 404
    
    session_data = sessions[session_id]
    
    # Check if session expired
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
    """System health check"""
    cleanup_expired_sessions()
    
    # Check Tesseract availability
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        tesseract_status = "available"
    except Exception as e:
        tesseract_version = None
        tesseract_status = f"error: {e}"
    
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(sessions),
        'tesseract_version': str(tesseract_version) if tesseract_version else None,
        'tesseract_status': tesseract_status,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'timestamp': datetime.now().isoformat(),
        'session_timeout': SESSION_TIMEOUT,
        'advanced_analysis': ADVANCED_ANALYSIS,
        'pdf_support': PDF_SUPPORT
    })

@app.route('/cleanup')
def manual_cleanup():
    """Manual cleanup endpoint for testing"""
    expired_count = cleanup_expired_sessions()
    return jsonify({
        'message': f'Cleaned up {expired_count} expired sessions',
        'active_sessions': len(sessions)
    })

if __name__ == '__main__':
    # Comprehensive system checks
    print("Certificate Verification System - Starting...")
    
    # Check Tesseract installation
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract OCR detected: {tesseract_version}")
    except Exception as e:
        print(f"Tesseract OCR not found: {e}")
        print("\nInstallation Instructions:")
        print("  Ubuntu/Debian: sudo apt update && sudo apt install tesseract-ocr poppler-utils")
        print("  macOS: brew install tesseract poppler")
        print("  Windows: Download from https://github.com/tesseract-ocr/tesseract")
        sys.exit(1)
    
    # Check optional dependencies
    if PDF_SUPPORT:
        print("PDF2Image available for PDF processing")
    else:
        print("PDF2Image not available - PDF support limited")
        print("   Install with: pip install pdf2image")
    
    if ADVANCED_ANALYSIS:
        print("Scikit-image available for advanced analysis")
    else:
        print("Scikit-image not available - advanced analysis limited")
        print("   Install with: pip install scikit-image")
    
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print(f"Upload directory ready: {app.config['UPLOAD_FOLDER']}")
    
    print(f"\nServer starting...")
    print(f"Open http://127.0.0.1:5000 to begin verification")
    print(f"Health check: http://127.0.0.1:5000/health")
    print(f"Session timeout: {SESSION_TIMEOUT} seconds")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)