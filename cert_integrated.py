#!/usr/bin/env python3
"""
ACAD - Enhanced Certificate Verification System
Improved extraction accuracy with advanced validation
"""

from flask import Flask, render_template_string, request, jsonify, redirect, url_for
import os
import uuid
import base64
import io
import re
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import pytesseract
import qrcode
from werkzeug.utils import secure_filename
import tempfile
import traceback
from collections import defaultdict, Counter
from difflib import SequenceMatcher

# Optional ML imports
HAS_TORCH = False
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    HAS_TORCH = True
except ImportError:
    print("Info: PyTorch/Transformers not available - ML features disabled")

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['SECRET_KEY'] = os.urandom(24)

# Session storage
sessions = {}
SESSION_TIMEOUT = 600

# Configuration
DEVICE = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

# Model instances
_trocr_processor = None
_trocr_model = None

# =====================
# ENHANCED PATTERNS
# =====================

INSTITUTION_PATTERNS = [
    r'TAMIL\s+NADU\s+(?:DR\.?)?\s*(?:M\.?G\.?R\.?)?\s*TEACHERS?\s+EDUCATION\s+UNIVERSITY',
    r'TAMIL\s+NADU.*?EDUCATION.*?UNIVERSITY',
    r'(?:ANNA|BHARATHIAR|MADRAS|BANGALORE|MUMBAI|DELHI)\s+UNIVERSITY',
    r'(?:IIT|NIT|IIIT)\s+(?:MADRAS|BOMBAY|DELHI|KANPUR)',
]

DEGREE_FULL_NAMES = {
    'B.ED': ['BACHELOR OF EDUCATION', 'B.ED', 'B ED', 'BED'],
    'B.TECH': ['BACHELOR OF TECHNOLOGY', 'B.TECH', 'BTECH'],
    'B.E': ['BACHELOR OF ENGINEERING', 'B.E', 'BE'],
    'M.ED': ['MASTER OF EDUCATION', 'M.ED', 'M ED', 'MED'],
    'M.TECH': ['MASTER OF TECHNOLOGY', 'M.TECH', 'MTECH'],
    'MBA': ['MASTER OF BUSINESS ADMINISTRATION', 'MBA'],
}

NAME_STOPWORDS = [
    'UNIVERSITY', 'COLLEGE', 'INSTITUTE', 'BACHELOR', 'MASTER', 'DOCTOR',
    'DEGREE', 'CERTIFICATE', 'AWARDED', 'PRESENTED', 'CERTIFY', 'DIPLOMA',
    'EDUCATION', 'EXAMINATION', 'PROVISIONAL', 'THIS', 'THAT', 'HEREBY',
    'QUALIFIED', 'PASSED', 'CLASS', 'OBTAINED', 'FIRST', 'SECOND', 'THIRD'
]

# =====================
# HTML TEMPLATES
# =====================

UPLOAD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Upload Certificate</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a0a; min-height: 100vh; padding: 20px; }
        .container { max-width: 700px; margin: 0 auto; background: #1a1a1a; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.8); overflow: hidden; border: 1px solid #2a2a2a; }
        .header { background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); padding: 30px; text-align: center; color: #e0e0e0; border-bottom: 2px solid #3a3a3a; }
        .header h1 { font-size: 1.8em; color: #ffffff; }
        .header p { font-size: 0.9em; color: #999; margin-top: 5px; }
        .content { padding: 40px; }
        .upload-area { border: 3px dashed #4a4a4a; padding: 60px 30px; text-align: center; margin: 30px 0; background: #0f0f0f; border-radius: 15px; cursor: pointer; transition: all 0.3s; position: relative; }
        .upload-area:hover { border-color: #6a6a6a; transform: translateY(-3px); box-shadow: 0 5px 15px rgba(100,100,100,0.3); background: #151515; }
        .upload-icon { font-size: 4em; margin-bottom: 20px; }
        .file-input { position: absolute; opacity: 0; width: 100%; height: 100%; cursor: pointer; top: 0; left: 0; }
        .file-input-label { background: #2a2a2a; color: #e0e0e0; padding: 15px 30px; border-radius: 25px; cursor: pointer; display: inline-block; transition: all 0.3s; font-weight: 500; border: 1px solid #3a3a3a; }
        .file-input-label:hover { background: #3a3a3a; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(100,100,100,0.4); }
        .submit-btn { background: #2a2a2a; color: #e0e0e0; padding: 15px 40px; border: 1px solid #3a3a3a; border-radius: 25px; cursor: pointer; font-size: 1.1em; font-weight: 600; width: 100%; margin: 20px 0; transition: all 0.3s; }
        .submit-btn:hover:not(:disabled) { background: #3a3a3a; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(100,100,100,0.4); }
        .submit-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .status { padding: 20px; margin: 20px 0; border-radius: 10px; text-align: center; font-weight: 500; display: none; }
        .success { background: #1a2f1a; border: 2px solid #2d5f2d; color: #5fb85f; }
        .error { background: #2f1a1a; border: 2px solid #5f2d2d; color: #f77; }
        .processing { background: #2f2a1a; border: 2px solid #5f542d; color: #ffb74d; }
        .file-name { margin: 15px 0; padding: 12px; background: #0f0f0f; border-radius: 8px; color: #9db4d4; display: none; font-size: 0.95em; border: 1px solid #2a2a2a; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>&#127891; Upload Certificate</h1>
            <p>Enhanced Accuracy Verification</p>
        </div>
        <div class="content">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area" id="uploadArea">
                    <input type="file" name="file" id="fileInput" class="file-input" accept="image/*" required>
                    <div class="upload-icon">&#127891;</div>
                    <p style="font-size:1.2em; margin:15px 0; color:#ccc;"><strong>Drop certificate here</strong></p>
                    <p style="margin:15px 0; color:#888;">or</p>
                    <label for="fileInput" class="file-input-label">Choose File</label>
                </div>
                <div id="fileName" class="file-name"></div>
                <button type="submit" id="submitBtn" class="submit-btn">Upload & Analyze</button>
            </form>
            <div id="status" class="status"></div>
        </div>
    </div>
    <script>
        const form = document.getElementById('uploadForm');
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const submitBtn = document.getElementById('submitBtn');
        const statusDiv = document.getElementById('status');
        const fileNameDiv = document.getElementById('fileName');
        const sessionId = '{{ session_id }}';
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const fileName = e.target.files[0].name;
                const fileSize = (e.target.files[0].size / 1024 / 1024).toFixed(2);
                fileNameDiv.innerHTML = `Selected: <strong>${fileName}</strong> (${fileSize} MB)`;
                fileNameDiv.style.display = 'block';
            }
        });
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => { e.preventDefault(); e.stopPropagation(); }, false);
        });
        
        uploadArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                const fileName = files[0].name;
                const fileSize = (files[0].size / 1024 / 1024).toFixed(2);
                fileNameDiv.innerHTML = `Selected: <strong>${fileName}</strong> (${fileSize} MB)`;
                fileNameDiv.style.display = 'block';
            }
        });
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!fileInput.files || fileInput.files.length === 0) {
                alert('Please select a file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('session_id', sessionId);
            
            submitBtn.disabled = true;
            submitBtn.textContent = 'Analyzing...';
            statusDiv.className = 'status processing';
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = '<strong>&#128270; Analyzing certificate...</strong>';
            
            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const result = await response.json();
                
                if (response.ok) {
                    statusDiv.className = 'status success';
                    statusDiv.innerHTML = '<h3>&#10004; Analysis Complete!</h3><p style="margin-top:10px;">Check the main screen for results.</p>';
                    submitBtn.textContent = 'Analysis Complete';
                } else {
                    throw new Error(result.error || 'Upload failed');
                }
            } catch (error) {
                statusDiv.className = 'status error';
                statusDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
                submitBtn.disabled = false;
                submitBtn.textContent = 'Upload & Analyze';
            }
        });
    </script>
</body>
</html>"""

VERIFIER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ACAD - Certificate Verifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a0a; min-height: 100vh; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; background: #1a1a1a; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.8); overflow: hidden; border: 1px solid #2a2a2a; }
        .header { background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); color: #e0e0e0; padding: 40px; text-align: center; border-bottom: 2px solid #3a3a3a; }
        .header h1 { font-size: 2.3em; margin-bottom: 10px; color: #ffffff; }
        .header p { opacity: 0.8; font-size: 1.05em; }
        .content { padding: 40px; }
        .qr-section { text-align: center; margin: 30px 0; padding: 30px; background: #0f0f0f; border-radius: 15px; border: 1px solid #2a2a2a; }
        .qr-code { display: inline-block; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
        .session-info { margin: 20px 0; padding: 15px; background: #1a1a1a; border-radius: 10px; font-family: monospace; color: #888; font-size: 0.9em; border: 1px solid #2a2a2a; }
        .status { padding: 20px; margin: 20px 0; border-radius: 10px; text-align: center; font-weight: 500; }
        .pending { background: #1a2a3a; border: 2px solid #2d4d6d; color: #6db4f7; }
        .processing { background: #2f2a1a; border: 2px solid #5f542d; color: #ffb74d; }
        .success { background: #1a2f1a; border: 2px solid #2d5f2d; color: #5fb85f; }
        .error { background: #2f1a1a; border: 2px solid #5f2d2d; color: #f77; }
        .result-box { background: #0f0f0f; padding: 30px; margin: 25px 0; border-radius: 15px; border: 1px solid #2a2a2a; }
        .authentic { border-left: 5px solid #5fb85f; }
        .suspicious { border-left: 5px solid #ffb74d; }
        .forgery { border-left: 5px solid #f77; }
        .field-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }
        .field-item { padding: 15px; background: #1a1a1a; border-radius: 10px; border: 1px solid #2a2a2a; position: relative; }
        .field-item.high-confidence { border-left: 3px solid #5fb85f; }
        .field-item.medium-confidence { border-left: 3px solid #ffb74d; }
        .field-item.low-confidence { border-left: 3px solid #f77; }
        .field-label { font-weight: 600; color: #888; font-size: 0.85em; margin-bottom: 5px; }
        .field-value { font-size: 1.05em; color: #ccc; word-wrap: break-word; }
        .confidence-badge { position: absolute; top: 10px; right: 10px; font-size: 0.75em; padding: 3px 8px; border-radius: 10px; background: #2a2a2a; color: #888; }
        .tampering-badge { display: inline-block; padding: 10px 20px; border-radius: 20px; font-weight: 600; font-size: 1em; }
        .badge-authentic { background: #2d5f2d; color: #5fb85f; }
        .badge-suspicious { background: #5f542d; color: #ffb74d; }
        .badge-forgery { background: #5f2d2d; color: #f77; }
        .spinner { border: 3px solid #2a2a2a; border-top: 3px solid #6a6a6a; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 15px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        h4 { color: #ccc; margin: 25px 0 15px 0; font-size: 1.15em; }
        .tech-info { background: #1a2a3a; padding: 15px; border-radius: 10px; margin: 15px 0; font-size: 0.9em; color: #9db4d4; border: 1px solid #2d4d6d; }
        h2 { color: #e0e0e0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>&#127891; ACAD Certificate Verifier</h1>
            <p>Enhanced Accuracy Analysis</p>
        </div>
        <div class="content">
            <div class="qr-section">
                <h2 style="color: #888; margin-bottom: 20px;">Scan QR Code to Upload</h2>
                <div class="qr-code">
                    <img src="data:image/png;base64,{{ qr_code }}" alt="QR Code" style="max-width: 180px;">
                </div>
                <div class="session-info">
                    <strong>Session ID:</strong> {{ session_id }}<br>
                    <strong>Expires:</strong> {{ expires_time }}
                </div>
            </div>
            <div id="status" class="status pending">
                <div class="spinner"></div>
                <strong>Waiting for certificate upload...</strong>
            </div>
            <div id="result" style="display: none;"></div>
        </div>
    </div>
    <script>
        const sessionId = '{{ session_id }}';
        let pollInterval = setInterval(checkResults, 1500);
        
        async function checkResults() {
            try {
                const response = await fetch(`/results/${sessionId}`);
                if (!response.ok) return;
                
                const data = await response.json();
                const statusDiv = document.getElementById('status');
                const resultDiv = document.getElementById('result');
                
                if (data.status === 'processing') {
                    statusDiv.className = 'status processing';
                    statusDiv.innerHTML = '<div class="spinner"></div><strong>&#128270; Analyzing certificate...</strong>';
                } else if (data.status === 'done' && data.result) {
                    clearInterval(pollInterval);
                    statusDiv.className = 'status success';
                    statusDiv.innerHTML = '<strong>&#10004; Analysis Complete!</strong>';
                    displayResults(data.result, resultDiv);
                } else if (data.status === 'error') {
                    clearInterval(pollInterval);
                    statusDiv.className = 'status error';
                    statusDiv.innerHTML = '<strong>Error:</strong> ' + (data.error || 'Analysis failed');
                }
            } catch (error) {
                console.error('Polling error:', error);
            }
        }
        
        function getConfidenceClass(conf) {
            if (conf >= 85) return 'high-confidence';
            if (conf >= 65) return 'medium-confidence';
            return 'low-confidence';
        }
        
        function displayResults(result, resultDiv) {
            resultDiv.style.display = 'block';
            
            let verdictClass = result.tampering_score < 20 ? 'authentic' : result.tampering_score < 70 ? 'suspicious' : 'forgery';
            let badgeClass = result.tampering_score < 20 ? 'badge-authentic' : result.tampering_score < 70 ? 'badge-suspicious' : 'badge-forgery';
            let verdictText = (result.tampering_verdict || 'unknown').toUpperCase().replace(/_/g, ' ');
            
            let html = `<div class="result-box ${verdictClass}">
                <h2 style="text-align:center; margin-bottom:20px; color:#e0e0e0;">Certificate Analysis Results</h2>
                <div style="text-align:center; margin: 25px 0;">
                    <span class="tampering-badge ${badgeClass}">${verdictText}</span>
                    <div style="font-size:1.5em; margin-top:15px; color:#e0e0e0;">
                        Authenticity Score: <strong>${100 - result.tampering_score}/100</strong>
                    </div>
                    <div style="margin-top:10px; color:#888;">Overall Confidence: ${result.overall_confidence}%</div>
                </div>
                <h4>Extracted Information</h4>
                <div class="field-grid">
                    <div class="field-item ${getConfidenceClass(result.field_confidences.name)}">
                        <span class="confidence-badge">${result.field_confidences.name}%</span>
                        <div class="field-label">Student Name</div>
                        <div class="field-value">${result.name || 'Not detected'}</div>
                    </div>
                    <div class="field-item ${getConfidenceClass(result.field_confidences.roll_no)}">
                        <span class="confidence-badge">${result.field_confidences.roll_no}%</span>
                        <div class="field-label">Registration Number</div>
                        <div class="field-value">${result.roll_no || 'Not detected'}</div>
                    </div>
                    <div class="field-item ${getConfidenceClass(result.field_confidences.degree)}">
                        <span class="confidence-badge">${result.field_confidences.degree}%</span>
                        <div class="field-label">Degree</div>
                        <div class="field-value">${result.degree || 'Not detected'}</div>
                    </div>
                    <div class="field-item ${getConfidenceClass(result.field_confidences.year)}">
                        <span class="confidence-badge">${result.field_confidences.year}%</span>
                        <div class="field-label">Year</div>
                        <div class="field-value">${result.year || 'Not detected'}</div>
                    </div>
                    <div class="field-item ${getConfidenceClass(result.field_confidences.institution)}">
                        <span class="confidence-badge">${result.field_confidences.institution}%</span>
                        <div class="field-label">Institution</div>
                        <div class="field-value">${result.institution || 'Not detected'}</div>
                    </div>
                </div>
                <h4>Technical Analysis</h4>
                <div class="tech-info">
                    <strong>OCR Confidence:</strong> ${result.ocr_confidence}%<br>
                    <strong>Processing Time:</strong> ${result.processing_time}s<br>
                    <strong>Image Quality:</strong> ${result.image_quality}<br>
                    <strong>OCR Passes:</strong> ${result.ocr_attempts}
                </div>
                <h4>Analysis Notes</h4>
                <div style="background:#1a1a1a; padding:20px; border-radius:10px; color:#888; border:1px solid #2a2a2a;">
                    ${result.analysis_notes || 'All validation checks passed'}
                </div>
            </div>`;
            
            resultDiv.innerHTML = html;
        }
    </script>
</body>
</html>"""

# =====================
# UTILITY FUNCTIONS
# =====================

def similarity_score(str1, str2):
    """Calculate similarity between two strings"""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1.upper(), str2.upper()).ratio()

def normalize_text(text):
    """Advanced text normalization"""
    if not text:
        return ""
    
    # Remove common OCR artifacts
    text = re.sub(r'[|\u00a6\u00c2\u20ac\u2039\u201a\u00ac\u00c5]', '', text)
    
    # Fix common character issues
    replacements = {
        '\uff10': '0', '\uff11': '1', '\uff12': '2', '\uff13': '3', '\uff14': '4',
        '\uff15': '5', '\uff16': '6', '\uff17': '7', '\uff18': '8', '\uff19': '9',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def clean_name(name):
    """Clean extracted name"""
    if not name:
        return None
    
    # Remove common artifacts
    name = re.sub(r'[^A-Za-z\s\.]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Remove stopwords
    words = name.split()
    cleaned_words = [w for w in words if w.upper() not in NAME_STOPWORDS and len(w) > 1]
    
    if cleaned_words:
        return ' '.join(cleaned_words)
    return None

# =====================
# VALIDATION FUNCTIONS
# =====================

def validate_name(name):
    """Strict name validation"""
    if not name or len(name) < 3 or len(name) > 70:
        return False
    
    words = name.split()
    if len(words) < 2 or len(words) > 6:
        return False
    
    # Check for minimum word length
    if any(len(w) < 2 for w in words):
        return False
    
    # Check alpha ratio
    alpha_count = sum(c.isalpha() for c in name)
    if len(name) > 0 and alpha_count / len(name) < 0.7:
        return False
    
    # Check for too many digits
    digit_count = sum(c.isdigit() for c in name)
    if digit_count > 2:
        return False
    
    # Exclude stopwords
    upper_name = name.upper()
    for stopword in NAME_STOPWORDS:
        if stopword in upper_name:
            return False
    
    return True

def validate_roll(roll):
    """Enhanced roll number validation"""
    if not roll or len(roll) < 4 or len(roll) > 25:
        return False
    
    # Exclude pure years
    if re.match(r'^(19|20)\d{2}$', roll):
        return False
    
    # Must have letters AND numbers
    has_letter = bool(re.search(r'[A-Z]', roll.upper()))
    has_number = bool(re.search(r'\d', roll))
    
    return has_letter and has_number

# =====================
# ENHANCED EXTRACTION
# =====================

def extract_name_enhanced(lines, full_text):
    """Enhanced name extraction with better accuracy"""
    candidates = []
    
    logger.info("=== Name Extraction ===")
    
    # Strategy 1: "This is to certify that" pattern
    certify_pattern = r'(?:THIS\s+IS\s+TO\s+CERTIFY\s+THAT|CERTIFY\s+THAT)[:\s]+([A-Z][A-Za-z\s\.]{3,60})'
    matches = re.finditer(certify_pattern, full_text, re.IGNORECASE)
    
    for match in matches:
        raw_name = match.group(1).strip()
        # Stop at common keywords
        raw_name = re.split(r'\s+(?:HAS|QUALIFIED|PASSED|OBTAINED|COMPLETED)', raw_name, maxsplit=1)[0].strip()
        
        cleaned = clean_name(raw_name)
        if cleaned and validate_name(cleaned):
            candidates.append(('certify_pattern', cleaned, 98))
            logger.info(f"Found via certify pattern: {cleaned} [98%]")
    
    # Strategy 2: Lines after "This is to certify"
    for i, line in enumerate(lines):
        if re.search(r'THIS\s+IS\s+TO\s+CERTIFY|CERTIFY\s+THAT', line, re.IGNORECASE):
            # Check next 3 lines
            for offset in range(1, 4):
                if i + offset >= len(lines):
                    break
                
                candidate_line = lines[i + offset].strip()
                
                # Skip lines with common keywords
                if re.search(r'QUALIFIED|DEGREE|BACHELOR|MASTER|UNIVERSITY|EXAMINATION', candidate_line, re.IGNORECASE):
                    continue
                
                # Skip lines with too many digits
                if len(candidate_line) > 0 and sum(c.isdigit() for c in candidate_line) / len(candidate_line) > 0.3:
                    continue
                
                cleaned = clean_name(candidate_line)
                if cleaned and validate_name(cleaned):
                    conf = 95 - (offset * 3)
                    candidates.append((f'after_certify_{offset}', cleaned, conf))
                    logger.info(f"Found after certify (+{offset}): {cleaned} [{conf}%]")
    
    # Strategy 3: Look for proper names in top 40% of document
    for i in range(5, min(len(lines), int(len(lines) * 0.4))):
        line = lines[i].strip()
        
        # Skip short lines
        if len(line) < 5 or len(line) > 70:
            continue
        
        # Skip lines with keywords
        if re.search(r'UNIVERSITY|COLLEGE|DEGREE|BACHELOR|EXAMINATION|CERTIFICATE', line, re.IGNORECASE):
            continue
        
        # Check if mostly alphabetic
        alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
        if alpha_ratio < 0.7:
            continue
        
        cleaned = clean_name(line)
        if cleaned and validate_name(cleaned):
            conf = max(70 - i, 50)
            candidates.append((f'structural_{i}', cleaned, conf))
            logger.info(f"Found structural: {cleaned} [{conf}%]")
    
    # Merge and rank candidates
    if candidates:
        # Sort by confidence
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]
        logger.info(f"Best name: {best[1]} [{best[2]}%]")
        return best[1], best[2]
    
    logger.warning("Name not found")
    return None, 0

def extract_roll_enhanced(lines, full_text):
    """Enhanced roll number extraction"""
    candidates = []
    
    logger.info("=== Roll Number Extraction ===")
    
    # Strategy 1: Labeled patterns with context
    label_patterns = [
        r'REGISTER\s*(?:NO|NUMBER|NUM)?\.?\s*[:\-]?\s*([A-Z0-9]{6,20})',
        r'REGISTRATION\s*(?:NO|NUMBER)?\.?\s*[:\-]?\s*([A-Z0-9]{6,20})',
        r'REG\.?\s*(?:NO|NUMBER)?\.?\s*[:\-]?\s*([A-Z0-9]{6,20})',
        r'ROLL\s*(?:NO|NUMBER)?\.?\s*[:\-]?\s*([A-Z0-9]{6,20})',
    ]
    
    for pattern in label_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            roll = match.group(1).strip().upper()
            roll = re.sub(r'\s+', '', roll)  # Remove spaces
            
            if validate_roll(roll):
                candidates.append(('labeled', roll, 98))
                logger.info(f"Found via label: {roll} [98%]")
    
    # Strategy 2: Format-based patterns
    # Look for patterns like: 10411BBD083, 19BCE1234, etc.
    format_patterns = [
        r'\b(\d{5,6}[A-Z]{2,4}\d{3,5})\b',  # 10411BBD083
        r'\b([A-Z]{2,4}\d{6,10})\b',         # ABC1234567
        r'\b(\d{2}[A-Z]{3}\d{4})\b',         # 19BCE1234
    ]
    
    for pattern in format_patterns:
        matches = re.finditer(pattern, full_text)
        for match in matches:
            roll = match.group(1).strip().upper()
            
            # Skip if it looks like a year - FIXED SYNTAX ERROR
            if re.match(r'^(19|20)\d{2}$', roll):
                continue
            
            if validate_roll(roll):
                # Boost confidence if near "register" or "roll"
                context_start = max(0, match.start() - 100)
                context_end = min(len(full_text), match.end() + 100)
                context = full_text[context_start:context_end].upper()
                
                conf = 85
                if 'REGIST' in context or 'ROLL' in context:
                    conf = 92
                
                candidates.append(('format', roll, conf))
                logger.info(f"Found via format: {roll} [{conf}%]")
    
    # Strategy 3: Look in structured areas (top section)
    for i in range(min(15, len(lines))):
        line = lines[i].upper()
        
        # Look for alphanumeric codes
        codes = re.findall(r'\b([A-Z0-9]{6,20})\b', line)
        for code in codes:
            code = re.sub(r'\s+', '', code)
            
            if validate_roll(code):
                conf = 75
                if 'REGIST' in line or 'ROLL' in line:
                    conf = 90
                
                candidates.append((f'structural_{i}', code, conf))
                logger.info(f"Found structural: {code} [{conf}%]")
    
    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]
        logger.info(f"Best roll: {best[1]} [{best[2]}%]")
        return best[1], best[2]
    
    logger.warning("Roll number not found")
    return None, 0

def extract_degree_enhanced(lines, full_text):
    """Enhanced degree extraction"""
    candidates = []
    
    logger.info("=== Degree Extraction ===")
    
    # Strategy 1: Full degree names
    for standard, variations in DEGREE_FULL_NAMES.items():
        for variation in variations:
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, full_text, re.IGNORECASE):
                candidates.append(('full_name', standard, 96))
                logger.info(f"Found degree: {standard} [96%]")
                break
    
    # Strategy 2: "Degree of" pattern
    degree_pattern = r'DEGREE\s+OF\s+([A-Z][A-Za-z\s]{5,50})'
    matches = re.finditer(degree_pattern, full_text, re.IGNORECASE)
    
    for match in matches:
        degree = match.group(1).strip().upper()
        # Stop at common keywords
        degree = re.split(r'\s*[,\.]|\s+(?:HE|SHE|WITH|FROM)', degree)[0].strip()
        
        if len(degree) > 5:
            candidates.append(('degree_of_pattern', degree, 98))
            logger.info(f"Found via 'degree of': {degree} [98%]")
    
    # Strategy 3: Look for "BACHELOR/MASTER OF"
    full_degree_pattern = r'((?:BACHELOR|MASTER)\s+OF\s+[A-Z][A-Za-z\s]{5,50})'
    matches = re.finditer(full_degree_pattern, full_text, re.IGNORECASE)
    
    for match in matches:
        degree = match.group(1).strip().upper()
        degree = re.split(r'\s*[,\.]|\s+(?:HE|SHE|WITH)', degree)[0].strip()
        
        candidates.append(('full_pattern', degree, 97))
        logger.info(f"Found full degree: {degree} [97%]")
    
    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]
        logger.info(f"Best degree: {best[1]} [{best[2]}%]")
        return best[1], best[2]
    
    logger.warning("Degree not found")
    return None, 0

def extract_year_enhanced(lines, full_text):
    """Enhanced year extraction"""
    candidates = []
    current_year = datetime.now().year
    
    logger.info("=== Year Extraction ===")
    
    # Strategy 1: Date patterns (DD-MM-YYYY)
    date_patterns = [
        r'DATE[:\s]+(\d{1,2})[-/](\d{1,2})[-/](20\d{2})',
        r'(\d{1,2})[-/](\d{1,2})[-/](20\d{2})',
    ]
    
    for pattern in date_patterns:
        matches = re.finditer(pattern, full_text)
        for match in matches:
            year = match.group(3) if len(match.groups()) == 3 else match.group(1)
            year_int = int(year)
            
            if 2000 <= year_int <= current_year + 1:
                candidates.append(('date', year, 98))
                logger.info(f"Found via date: {year} [98%]")
    
    # Strategy 2: Context-based
    context_patterns = [
        (r'(?:PASSED|COMPLETED|EXAMINATION)\s+(?:IN|HELD\s+IN)\s+[A-Z][a-z]+[/\s]*(20\d{2})', 96),
        (r'(?:MAY|JUNE|APRIL|NOVEMBER|DECEMBER)[/\s]+(20\d{2})', 94),
    ]
    
    for pattern, conf in context_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            year = match.group(1)
            year_int = int(year)
            
            if 2000 <= year_int <= current_year + 1:
                candidates.append(('context', year, conf))
                logger.info(f"Found via context: {year} [{conf}%]")
    
    # Strategy 3: Frequency analysis
    all_years = re.findall(r'\b(20[0-2]\d)\b', full_text)
    year_freq = Counter(all_years)
    
    for year, count in year_freq.most_common(3):
        year_int = int(year)
        
        if 2000 <= year_int <= current_year + 1:
            conf = 70 + min(count * 10, 25)
            candidates.append(('frequency', year, conf))
            logger.info(f"Found via frequency: {year} [{conf}%] (count: {count})")
    
    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]
        logger.info(f"Best year: {best[1]} [{best[2]}%]")
        return best[1], best[2]
    
    logger.warning("Year not found")
    return None, 0

def extract_institution_enhanced(lines, full_text):
    """Enhanced institution extraction"""
    candidates = []
    
    logger.info("=== Institution Extraction ===")
    
    # Strategy 1: Top lines (header)
    for i in range(min(8, len(lines))):
        line = lines[i].strip()
        
        if re.search(r'UNIVERSITY|COLLEGE|INSTITUTE', line, re.IGNORECASE):
            if 10 < len(line) < 200:
                conf = 98 - (i * 2)
                candidates.append((f'header_{i}', line, conf))
                logger.info(f"Found in header: {line} [{conf}%]")
    
    # Strategy 2: Known patterns
    for pattern in INSTITUTION_PATTERNS:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            inst = match.group(0).strip()
            inst = re.sub(r'\s+', ' ', inst)
            candidates.append(('known_pattern', inst, 97))
            logger.info(f"Found via known pattern: {inst} [97%]")
    
    # Strategy 3: "UNIVERSITY" pattern
    uni_pattern = r'([A-Z][A-Za-z\s\.&,\-]{10,100})\s+UNIVERSITY'
    matches = re.finditer(uni_pattern, full_text, re.IGNORECASE)
    
    for match in matches:
        inst = match.group(0).strip()
        inst = re.sub(r'\s+', ' ', inst)
        
        if len(inst) > 15:
            candidates.append(('university_pattern', inst, 95))
            logger.info(f"Found university: {inst} [95%]")
    
    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]
        logger.info(f"Best institution: {best[1]} [{best[2]}%]")
        return best[1], best[2]
    
    logger.warning("Institution not found")
    return None, 0

# =====================
# IMAGE PREPROCESSING
# =====================

def preprocess_image(image_path):
    """Enhanced image preprocessing"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Failed to load image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Denoising + Adaptive threshold
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Method 2: Otsu's thresholding
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 3: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Method 4: Sharpening
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        return {
            'binary': binary,
            'otsu': otsu,
            'enhanced': enhanced,
            'sharpened': sharpened,
            'gray': gray,
            'original': image
        }
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return None

# =====================
# OCR PROCESSING
# =====================

def perform_ocr_enhanced(image_path):
    """Enhanced multi-pass OCR"""
    ocr_results = []
    
    processed = preprocess_image(image_path)
    if processed is None:
        return [], 0
    
    # Pass 1: Binary image
    try:
        config = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text1 = pytesseract.image_to_string(processed['binary'], lang='eng', config=config)
        conf_data = pytesseract.image_to_data(processed['binary'], lang='eng', config=config, output_type=pytesseract.Output.DICT)
        avg_conf = np.mean([int(c) for c in conf_data['conf'] if int(c) > 0]) if conf_data['conf'] else 0
        
        ocr_results.append({'text': text1, 'confidence': avg_conf, 'method': 'binary'})
        logger.info(f"OCR Pass 1 (binary): {avg_conf:.1f}%")
    except Exception as e:
        logger.error(f"OCR Pass 1 failed: {e}")
    
    # Pass 2: Otsu threshold
    try:
        config = '--oem 3 --psm 3'
        text2 = pytesseract.image_to_string(processed['otsu'], lang='eng', config=config)
        conf_data = pytesseract.image_to_data(processed['otsu'], lang='eng', config=config, output_type=pytesseract.Output.DICT)
        avg_conf = np.mean([int(c) for c in conf_data['conf'] if int(c) > 0]) if conf_data['conf'] else 0
        
        ocr_results.append({'text': text2, 'confidence': avg_conf, 'method': 'otsu'})
        logger.info(f"OCR Pass 2 (otsu): {avg_conf:.1f}%")
    except Exception as e:
        logger.error(f"OCR Pass 2 failed: {e}")
    
    # Pass 3: Enhanced (CLAHE)
    try:
        config = '--oem 3 --psm 6'
        text3 = pytesseract.image_to_string(processed['enhanced'], lang='eng', config=config)
        conf_data = pytesseract.image_to_data(processed['enhanced'], lang='eng', config=config, output_type=pytesseract.Output.DICT)
        avg_conf = np.mean([int(c) for c in conf_data['conf'] if int(c) > 0]) if conf_data['conf'] else 0
        
        ocr_results.append({'text': text3, 'confidence': avg_conf, 'method': 'enhanced'})
        logger.info(f"OCR Pass 3 (enhanced): {avg_conf:.1f}%")
    except Exception as e:
        logger.error(f"OCR Pass 3 failed: {e}")
    
    # Pass 4: Sharpened
    try:
        config = '--oem 3 --psm 6'
        text4 = pytesseract.image_to_string(processed['sharpened'], lang='eng', config=config)
        conf_data = pytesseract.image_to_data(processed['sharpened'], lang='eng', config=config, output_type=pytesseract.Output.DICT)
        avg_conf = np.mean([int(c) for c in conf_data['conf'] if int(c) > 0]) if conf_data['conf'] else 0
        
        ocr_results.append({'text': text4, 'confidence': avg_conf, 'method': 'sharpened'})
        logger.info(f"OCR Pass 4 (sharpened): {avg_conf:.1f}%")
    except Exception as e:
        logger.error(f"OCR Pass 4 failed: {e}")
    
    overall_conf = max(r['confidence'] for r in ocr_results) if ocr_results else 0
    
    return ocr_results, overall_conf

def merge_ocr_results(ocr_results):
    """Merge OCR results intelligently"""
    if not ocr_results:
        return "", []
    
    # Combine all text
    all_text = "\n\n".join([r['text'] for r in ocr_results])
    
    # Get unique lines
    all_lines = []
    for result in ocr_results:
        lines = [line.strip() for line in result['text'].split('\n') if line.strip()]
        all_lines.extend(lines)
    
    # Remove duplicates
    unique_lines = []
    for line in all_lines:
        is_duplicate = False
        for existing in unique_lines:
            if similarity_score(line, existing) > 0.90:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_lines.append(line)
    
    return all_text, unique_lines

# =====================
# TAMPERING DETECTION
# =====================

def detect_tampering(image_path):
    """Simplified tampering detection"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return 30, "unknown"
        
        # Basic ELA
        ela_score = compute_ela_simple(image_path)
        
        if ela_score < 20:
            verdict = "authentic"
        elif ela_score < 60:
            verdict = "suspicious"
        else:
            verdict = "likely_forgery"
        
        logger.info(f"Tampering score: {ela_score} - {verdict}")
        return ela_score, verdict
        
    except Exception as e:
        logger.error(f"Tampering detection error: {e}")
        return 30, "unknown"

def compute_ela_simple(image_path):
    """Simplified ELA"""
    try:
        img = Image.open(image_path)
        temp_path = tempfile.mktemp(suffix='.jpg')
        img.save(temp_path, 'JPEG', quality=95)
        
        original = cv2.imread(str(image_path))
        resaved = cv2.imread(temp_path)
        
        try:
            os.unlink(temp_path)
        except:
            pass
        
        if original is None or resaved is None:
            return 30
        
        diff = cv2.absdiff(original, resaved)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        std_error = np.std(diff_gray)
        
        if std_error < 5:
            return 10
        elif std_error < 15:
            return 25
        elif std_error < 30:
            return 50
        else:
            return 80
            
    except Exception as e:
        logger.error(f"ELA error: {e}")
        return 30

# =====================
# MAIN ANALYSIS
# =====================

def analyze_certificate(image_path):
    """Main certificate analysis with enhanced extraction"""
    start_time = time.time()
    
    try:
        logger.info("=" * 80)
        logger.info("STARTING ENHANCED CERTIFICATE ANALYSIS")
        logger.info("=" * 80)
        
        # Tampering detection
        logger.info("\n[1/3] Tampering Detection...")
        tamper_score, tamper_verdict = detect_tampering(image_path)
        
        # OCR
        logger.info("\n[2/3] Multi-Pass OCR...")
        ocr_results, ocr_confidence = perform_ocr_enhanced(image_path)
        
        if not ocr_results:
            raise ValueError("OCR failed")
        
        merged_text, unique_lines = merge_ocr_results(ocr_results)
        
        logger.info(f"Extracted {len(unique_lines)} unique lines")
        logger.info(f"OCR Confidence: {ocr_confidence:.1f}%")
        
        # Normalize
        normalized_text = normalize_text(merged_text)
        normalized_lines = [normalize_text(line) for line in unique_lines]
        
        # Field extraction
        logger.info("\n[3/3] Enhanced Field Extraction...")
        
        name, name_conf = extract_name_enhanced(normalized_lines, normalized_text)
        roll, roll_conf = extract_roll_enhanced(normalized_lines, normalized_text)
        degree, degree_conf = extract_degree_enhanced(normalized_lines, normalized_text)
        year, year_conf = extract_year_enhanced(normalized_lines, normalized_text)
        institution, inst_conf = extract_institution_enhanced(normalized_lines, normalized_text)
        
        # Log results
        logger.info("\n" + "=" * 80)
        logger.info("EXTRACTION RESULTS:")
        logger.info("=" * 80)
        logger.info(f"Name:        {name or 'NOT FOUND'} ({name_conf}%)")
        logger.info(f"Roll No:     {roll or 'NOT FOUND'} ({roll_conf}%)")
        logger.info(f"Degree:      {degree or 'NOT FOUND'} ({degree_conf}%)")
        logger.info(f"Year:        {year or 'NOT FOUND'} ({year_conf}%)")
        logger.info(f"Institution: {institution or 'NOT FOUND'} ({inst_conf}%)")
        logger.info("=" * 80)
        
        # Calculate overall confidence
        field_confs = [name_conf, roll_conf, degree_conf, year_conf, inst_conf]
        non_zero_confs = [c for c in field_confs if c > 0]
        overall_conf = int(sum(non_zero_confs) / len(non_zero_confs)) if non_zero_confs else 0
        
        processing_time = round(time.time() - start_time, 2)
        
        # Image quality
        image = cv2.imread(str(image_path))
        if image is not None:
            h, w = image.shape[:2]
            if h * w > 2000000:
                quality = "Excellent"
            elif h * w > 1000000:
                quality = "Good"
            else:
                quality = "Fair"
        else:
            quality = "Unknown"
        
        # Analysis notes
        notes = []
        if overall_conf >= 85:
            notes.append("High confidence extraction")
        elif overall_conf >= 70:
            notes.append("Good extraction with medium confidence")
        else:
            notes.append("Some fields have low confidence")
        
        if tamper_score < 20:
            notes.append("Document appears authentic")
        elif tamper_score < 60:
            notes.append("Some irregularities detected")
        else:
            notes.append("Significant tampering indicators")
        
        result = {
            'name': name,
            'roll_no': roll,
            'degree': degree,
            'year': year,
            'institution': institution,
            'field_confidences': {
                'name': name_conf,
                'roll_no': roll_conf,
                'degree': degree_conf,
                'year': year_conf,
                'institution': inst_conf,
            },
            'overall_confidence': overall_conf,
            'ocr_confidence': int(ocr_confidence),
            'tampering_score': tamper_score,
            'tampering_verdict': tamper_verdict,
            'processing_time': processing_time,
            'image_quality': quality,
            'ocr_attempts': len(ocr_results),
            'analysis_notes': ' | '.join(notes),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"\nAnalysis complete - Overall: {overall_conf}%")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise

# =====================
# FLASK ROUTES
# =====================

@app.route('/')
def index():
    """Main verifier page"""
    session_id = str(uuid.uuid4())[:8]
    expires = datetime.now() + timedelta(seconds=SESSION_TIMEOUT)
    
    sessions[session_id] = {
        'status': 'pending',
        'expires': expires,
        'result': None
    }
    
    upload_url = request.url_root.rstrip('/') + url_for('upload_page', session_id=session_id)
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(upload_url)
    qr.make(fit=True)
    
    qr_image = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    qr_image.save(buffer, format='PNG')
    qr_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return render_template_string(
        VERIFIER_HTML,
        session_id=session_id,
        qr_code=qr_base64,
        expires_time=expires.strftime('%I:%M:%S %p')
    )

@app.route('/upload/<session_id>')
def upload_page(session_id):
    """Upload page"""
    if session_id not in sessions:
        return "Invalid or expired session", 404
    
    session = sessions[session_id]
    if datetime.now() > session['expires']:
        return "Session expired", 410
    
    return render_template_string(UPLOAD_HTML, session_id=session_id)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded certificate"""
    try:
        session_id = request.form.get('session_id')
        
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        session = sessions[session_id]
        if datetime.now() > session['expires']:
            return jsonify({'error': 'Session expired'}), 410
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / f"{session_id}_{filename}"
        file.save(str(filepath))
        
        logger.info(f"File uploaded: {filepath}")
        
        session['status'] = 'processing'
        session['filepath'] = filepath
        
        try:
            result = analyze_certificate(filepath)
            session['status'] = 'done'
            session['result'] = result
            
            return jsonify({
                'status': 'success',
                'message': 'Analysis complete',
                'session_id': session_id
            })
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            logger.error(traceback.format_exc())
            session['status'] = 'error'
            session['error'] = str(e)
            
            return jsonify({
                'status': 'error',
                'error': f'Analysis failed: {str(e)}'
            }), 500
            
        finally:
            try:
                if filepath.exists():
                    filepath.unlink()
            except Exception as e:
                logger.warning(f"Could not delete file: {e}")
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<session_id>')
def get_results(session_id):
    """Get analysis results"""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 404
    
    session = sessions[session_id]
    
    if datetime.now() > session['expires']:
        return jsonify({'error': 'Session expired'}), 410
    
    response = {
        'status': session['status'],
        'session_id': session_id
    }
    
    if session['status'] == 'done' and session.get('result'):
        response['result'] = session['result']
    elif session['status'] == 'error':
        response['error'] = session.get('error', 'Unknown error')
    
    return jsonify(response)

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'device': DEVICE,
        'torch_available': HAS_TORCH
    })

# =====================
# SESSION CLEANUP
# =====================

def cleanup_old_sessions():
    """Clean up expired sessions"""
    now = datetime.now()
    expired = [sid for sid, session in sessions.items() if now > session['expires']]
    
    for sid in expired:
        try:
            if 'filepath' in sessions[sid]:
                filepath = sessions[sid]['filepath']
                if filepath.exists():
                    filepath.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning up session {sid}: {e}")
        
        del sessions[sid]
    
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")

def periodic_cleanup():
    """Periodic cleanup task"""
    import threading
    
    def cleanup_loop():
        while True:
            time.sleep(60)
            try:
                cleanup_old_sessions()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()

# =====================
# APPLICATION ENTRY POINT
# =====================

def main():
    """Main application entry point"""
    print("=" * 80)
    print("ACAD - Enhanced Certificate Verification System")
    print("Improved Accuracy Version")
    print("=" * 80)
    print()
    
    # Start cleanup thread
    print("Starting background cleanup...")
    periodic_cleanup()
    print()
    
    # Configuration info
    print("Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  PyTorch: {'Available' if HAS_TORCH else 'Not Available'}")
    print(f"  Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"  Session timeout: {SESSION_TIMEOUT}s")
    print()
    
    # Start Flask app
    print("=" * 80)
    print("Starting server...")
    print("=" * 80)
    print()
    print("Access at: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        print("Cleaning up...")
        
        try:
            import shutil
            if Path(app.config['UPLOAD_FOLDER']).exists():
                shutil.rmtree(app.config['UPLOAD_FOLDER'])
            print("Cleanup complete")
        except Exception as e:
            print(f"Cleanup warning: {e}")
        
        print("Goodbye!")

if __name__ == '__main__':
    main()