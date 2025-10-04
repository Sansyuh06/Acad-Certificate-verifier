#!/usr/bin/env python3
"""
Complete Setup Checker and Assistant for ACAD Certificate Verification System
Run this script first to verify all dependencies and guide you through setup
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}→ {text}{Colors.END}")

def check_python_version():
    """Check Python version"""
    print_info("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} - Need Python 3.7+")
        return False

def check_pip():
    """Check if pip is available"""
    print_info("Checking pip...")
    try:
        result = subprocess.run(['pip', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_success(f"pip installed: {result.stdout.split()[1]}")
            return True
    except:
        pass
    print_error("pip not found")
    return False

def check_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def install_packages(packages):
    """Install missing packages"""
    if not packages:
        return True
    
    print_info(f"Installing {len(packages)} packages...")
    print(f"  Packages: {', '.join(packages)}")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        print_success("All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to install some packages")
        return False

def check_python_packages():
    """Check all required Python packages"""
    print_info("Checking Python packages...")
    
    packages = {
        'flask': 'flask',
        'PIL': 'pillow',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pytesseract': 'pytesseract',
        'pyzbar': 'pyzbar',
        'qrcode': 'qrcode[pil]',
        'skimage': 'scikit-image',
        'pdf2image': 'pdf2image',
        'torch': 'torch',
        'transformers': 'transformers',
        'timm': 'timm',
        'tqdm': 'tqdm',
        'pandas': 'pandas'
    }
    
    missing = []
    installed = []
    
    for import_name, package_name in packages.items():
        if check_package(import_name):
            installed.append(package_name)
        else:
            missing.append(package_name)
    
    print_success(f"{len(installed)}/{len(packages)} packages installed")
    
    if missing:
        print_warning(f"Missing packages: {', '.join(missing)}")
        response = input("\nInstall missing packages now? (y/n): ").lower()
        if response == 'y':
            return install_packages(missing)
        else:
            print_warning("Skipping package installation")
            return False
    
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print_info("Checking Tesseract OCR...")
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print_success(f"Tesseract OCR {version} found")
        return True
    except Exception as e:
        print_error("Tesseract OCR not found or not configured")
        print("\n  Installation instructions:")
        print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Download installer and add to PATH")
        print("  Or set in Python: pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
        return False

def check_cuda():
    """Check if CUDA is available"""
    print_info("Checking CUDA/GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print_success(f"CUDA available: {device_name}")
            return True
        else:
            print_warning("CUDA not available - will use CPU (slower training)")
            return False
    except:
        print_warning("PyTorch not installed - cannot check CUDA")
        return False

def check_project_structure():
    """Check project folder structure"""
    print_info("Checking project structure...")
    
    current_dir = Path.cwd()
    print(f"  Current directory: {current_dir}")
    
    required_files = ['cert_integrated.py', 'organize_datasets.py', 'train_tamper_model.py']
    missing_files = []
    
    for file in required_files:
        if (current_dir / file).exists():
            print_success(f"Found: {file}")
        else:
            print_error(f"Missing: {file}")
            missing_files.append(file)
    
    models_dir = current_dir / 'models'
    if models_dir.exists():
        print_success(f"Found: models/ directory")
        
        # Check for Kaggle datasets
        datasets = [
            'defacto-copymove',
            'OCR Document Text Recognition Dataset',
            'Original and Tampered Image Dataset',
            'Professional Certification Programs',
            'standard OCR dataset',
            'TextOCR - Text Extraction from Images Dataset'
        ]
        
        found_datasets = []
        for ds in datasets:
            if (models_dir / ds).exists():
                found_datasets.append(ds)
        
        print_success(f"Found {len(found_datasets)}/{len(datasets)} Kaggle datasets")
        
    else:
        print_error("Missing: models/ directory")
        print("  Create it with: mkdir models")
    
    return len(missing_files) == 0

def check_organized_dataset():
    """Check if dataset has been organized"""
    print_info("Checking organized dataset...")
    
    org_dir = Path('models/organized_dataset')
    
    if not org_dir.exists():
        print_warning("Dataset not organized yet")
        return False
    
    authentic_dir = org_dir / 'authentic'
    tampered_dir = org_dir / 'tampered'
    
    authentic_count = len(list(authentic_dir.glob('*'))) if authentic_dir.exists() else 0
    tampered_count = len(list(tampered_dir.glob('*'))) if tampered_dir.exists() else 0
    
    print(f"  Authentic images: {authentic_count}")
    print(f"  Tampered images: {tampered_count}")
    
    if authentic_count < 50 or tampered_count < 50:
        print_warning("Not enough images for training (need 50+ each)")
        return False
    
    print_success("Dataset organized and ready")
    return True

def check_trained_model():
    """Check if model has been trained"""
    print_info("Checking trained model...")
    
    model_path = Path('models/tamper_model.pth')
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print_success(f"Trained model found: {size_mb:.1f} MB")
        return True
    else:
        print_warning("Model not trained yet")
        return False

def provide_next_steps(checks):
    """Provide guidance on next steps"""
    print_header("NEXT STEPS")
    
    if not checks['python_version']:
        print_error("CRITICAL: Update Python to 3.7 or higher")
        return
    
    if not checks['pip']:
        print_error("CRITICAL: Install pip")
        return
    
    if not checks['packages']:
        print_warning("Install missing Python packages first")
        print("  Run: pip install flask pytesseract opencv-python pillow pyzbar qrcode[pil] numpy scikit-image pdf2image torch transformers timm tqdm pandas")
        return
    
    if not checks['tesseract']:
        print_warning("Install Tesseract OCR")
        print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        return
    
    if not checks['project_structure']:
        print_warning("Ensure all script files are in the current directory")
        return
    
    if not checks['organized_dataset']:
        print("\n" + "="*70)
        print("STEP 1: Organize your datasets")
        print("="*70)
        print("Run: python organize_datasets.py")
        print("\nThis will create organized_dataset/ with authentic and tampered folders")
        return
    
    if not checks['trained_model']:
        print("\n" + "="*70)
        print("STEP 2: Train the tampering detection model")
        print("="*70)
        print('Run: python train_tamper_model.py --data_dir "models\\organized_dataset" --epochs 30 --batch_size 16')
        print("\nTraining will take 1-3 hours depending on your hardware")
        return
    
    # Everything is ready!
    print_success("ALL SETUP COMPLETE!")
    print("\n" + "="*70)
    print("You can now run the application!")
    print("="*70)
    print("\nRun: python cert_integrated.py")
    print("\nThen open: http://127.0.0.1:5000")

def main():
    print_header("ACAD CERTIFICATE VERIFICATION SYSTEM - SETUP CHECKER")
    
    checks = {}
    
    # Run all checks
    checks['python_version'] = check_python_version()
    checks['pip'] = check_pip()
    checks['packages'] = check_python_packages()
    checks['tesseract'] = check_tesseract()
    checks['cuda'] = check_cuda()
    checks['project_structure'] = check_project_structure()
    checks['organized_dataset'] = check_organized_dataset()
    checks['trained_model'] = check_trained_model()
    
    # Summary
    print_header("SETUP STATUS SUMMARY")
    
    total = len(checks)
    passed = sum(1 for v in checks.values() if v)
    
    status_symbols = {
        True: f"{Colors.GREEN}✓{Colors.END}",
        False: f"{Colors.RED}✗{Colors.END}"
    }
    
    print(f"Python Version:      {status_symbols[checks['python_version']]}")
    print(f"pip:                 {status_symbols[checks['pip']]}")
    print(f"Python Packages:     {status_symbols[checks['packages']]}")
    print(f"Tesseract OCR:       {status_symbols[checks['tesseract']]}")
    print(f"CUDA/GPU:            {status_symbols[checks['cuda']]} (optional)")
    print(f"Project Structure:   {status_symbols[checks['project_structure']]}")
    print(f"Organized Dataset:   {status_symbols[checks['organized_dataset']]}")
    print(f"Trained Model:       {status_symbols[checks['trained_model']]}")
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} checks passed{Colors.END}")
    
    # Provide guidance
    provide_next_steps(checks)
    
    print("\n" + "="*70)
    print("For detailed help, check the documentation or re-run this script")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()