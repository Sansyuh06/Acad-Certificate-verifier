#!/usr/bin/env python3
"""
Manual Dataset Organization Helper
Helps you organize your Kaggle datasets into the structure needed for training

Expected Output Structure:
    organized_dataset/
        authentic/     # Real, unmodified certificates
        tampered/      # Forged, edited, or manipulated certificates
"""

import os
import shutil
from pathlib import Path
from PIL import Image

def safe_copy(src, dest_folder, prefix=""):
    """Safely copy image file"""
    try:
        img = Image.open(src)
        img.verify()  # Verify it's a valid image
        
        dest_name = f"{prefix}{src.stem}{src.suffix}"
        dest_path = dest_folder / dest_name
        
        # Avoid overwriting
        counter = 1
        while dest_path.exists():
            dest_name = f"{prefix}{src.stem}_{counter}{src.suffix}"
            dest_path = dest_folder / dest_name
            counter += 1
        
        shutil.copy2(src, dest_path)
        return True
    except Exception as e:
        print(f"  ✗ Error with {src.name}: {e}")
        return False

def organize_datasets(base_dir):
    """Organize all datasets"""
    base_dir = Path(base_dir)
    output_dir = base_dir / "organized_dataset"
    
    authentic_dir = output_dir / "authentic"
    tampered_dir = output_dir / "tampered"
    
    # Create directories
    authentic_dir.mkdir(parents=True, exist_ok=True)
    tampered_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ORGANIZING DATASETS FOR TAMPERING DETECTION")
    print("=" * 70)
    
    authentic_count = 0
    tampered_count = 0
    
    # 1. Professional Certification Programs (AUTHENTIC)
    print("\n[1/6] Processing: Professional Certification Programs")
    cert_dir = base_dir / "Professional Certification Programs"
    if cert_dir.exists():
        print(f"  Source: {cert_dir}")
        image_files = list(cert_dir.rglob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"  Found: {len(image_files)} images")
        
        for img_file in image_files:
            if safe_copy(img_file, authentic_dir, "cert_"):
                authentic_count += 1
        print(f"  ✓ Copied {authentic_count} authentic certificates")
    else:
        print(f"  ✗ Directory not found")
    
    # 2. Original and Tampered Image Dataset
    print("\n[2/6] Processing: Original and Tampered Image Dataset")
    orig_tamp_dir = base_dir / "Original and Tampered Image Dataset"
    if orig_tamp_dir.exists():
        print(f"  Source: {orig_tamp_dir}")
        
        # Look for authentic/original subfolders
        original_found = False
        for subfolder_name in ['original', 'Original', 'authentic', 'Authentic', 
                              'real', 'Real', 'pristine', 'Pristine']:
            subfolder = orig_tamp_dir / subfolder_name
            if subfolder.exists():
                print(f"  Found authentic folder: {subfolder_name}/")
                original_found = True
                images = [f for f in subfolder.rglob("*") 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                print(f"    Images: {len(images)}")
                
                before = authentic_count
                for img_file in images:
                    if safe_copy(img_file, authentic_dir, "orig_"):
                        authentic_count += 1
                print(f"    ✓ Copied {authentic_count - before} authentic images")
        
        # Look for tampered subfolders
        tampered_found = False
        for subfolder_name in ['tampered', 'Tampered', 'forged', 'Forged', 
                              'fake', 'Fake', 'modified', 'Modified']:
            subfolder = orig_tamp_dir / subfolder_name
            if subfolder.exists():
                print(f"  Found tampered folder: {subfolder_name}/")
                tampered_found = True
                images = [f for f in subfolder.rglob("*") 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                print(f"    Images: {len(images)}")
                
                before = tampered_count
                for img_file in images:
                    if safe_copy(img_file, tampered_dir, "tamp_"):
                        tampered_count += 1
                print(f"    ✓ Copied {tampered_count - before} tampered images")
        
        if not original_found and not tampered_found:
            print("  ✗ No recognized subfolders found")
            print("  Available folders:")
            for item in orig_tamp_dir.iterdir():
                if item.is_dir():
                    print(f"    - {item.name}/")
    else:
        print(f"  ✗ Directory not found")
    
    # 3. defacto-copymove (TAMPERED - copy-move forgery)
    print("\n[3/6] Processing: defacto-copymove")
    copymove_dir = base_dir / "defacto-copymove"
    if copymove_dir.exists():
        print(f"  Source: {copymove_dir}")
        images = [f for f in copymove_dir.rglob("*") 
                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"  Found: {len(images)} images (copy-move forgeries)")
        
        before = tampered_count
        for img_file in images:
            if safe_copy(img_file, tampered_dir, "copymove_"):
                tampered_count += 1
        print(f"  ✓ Copied {tampered_count - before} tampered images")
    else:
        print(f"  ✗ Directory not found")
    
    # 4. OCR Document Text Recognition Dataset (AUTHENTIC - clean documents)
    print("\n[4/6] Processing: OCR Document Text Recognition Dataset")
    ocr_doc_dir = base_dir / "OCR Document Text Recognition Dataset"
    if ocr_doc_dir.exists():
        print(f"  Source: {ocr_doc_dir}")
        images = [f for f in ocr_doc_dir.rglob("*") 
                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"  Found: {len(images)} document images")
        
        before = authentic_count
        for img_file in images:
            if safe_copy(img_file, authentic_dir, "doc_"):
                authentic_count += 1
        print(f"  ✓ Copied {authentic_count - before} authentic documents")
    else:
        print(f"  ✗ Directory not found")
    
    # 5. standard OCR dataset (AUTHENTIC)
    print("\n[5/6] Processing: standard OCR dataset")
    std_ocr_dir = base_dir / "standard OCR dataset"
    if std_ocr_dir.exists():
        print(f"  Source: {std_ocr_dir}")
        images = [f for f in std_ocr_dir.rglob("*") 
                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"  Found: {len(images)} images")
        
        before = authentic_count
        for img_file in images:
            if safe_copy(img_file, authentic_dir, "std_"):
                authentic_count += 1
        print(f"  ✓ Copied {authentic_count - before} authentic documents")
    else:
        print(f"  ✗ Directory not found")
    
    # 6. TextOCR (AUTHENTIC - text extraction)
    print("\n[6/6] Processing: TextOCR")
    textocr_dir = base_dir / "TextOCR - Text Extraction from Images Dataset"
    if textocr_dir.exists():
        print(f"  Source: {textocr_dir}")
        images = [f for f in textocr_dir.rglob("*") 
                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"  Found: {len(images)} images")
        
        # Limit to avoid overwhelming the dataset
        if len(images) > 1000:
            print(f"  Limiting to 1000 images (too many)")
            images = images[:1000]
        
        before = authentic_count
        for img_file in images:
            if safe_copy(img_file, authentic_dir, "textocr_"):
                authentic_count += 1
        print(f"  ✓ Copied {authentic_count - before} authentic documents")
    else:
        print(f"  ✗ Directory not found")
    
    # Summary
    print("\n" + "=" * 70)
    print("ORGANIZATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"\nDataset Summary:")
    print(f"  Authentic images: {authentic_count}")
    print(f"  Tampered images:  {tampered_count}")
    print(f"  Total images:     {authentic_count + tampered_count}")
    
    if tampered_count == 0:
        print("\n⚠️  WARNING: No tampered images found!")
        print("    You need tampered/forged images to train the model.")
        print("    Check the 'Original and Tampered Image Dataset' folder structure.")
    elif authentic_count < 100 or tampered_count < 100:
        print("\n⚠️  WARNING: Small dataset detected!")
        print("    For good results, you need at least 100-200 images per class.")
    elif abs(authentic_count - tampered_count) > authentic_count * 0.5:
        print("\n⚠️  WARNING: Imbalanced dataset!")
        print("    Try to have similar numbers of authentic and tampered images.")
    else:
        print("\n✓ Dataset looks good! Ready for training.")
    
    print(f"\nNext step:")
    print(f"  python train_tamper_model.py --data_dir {output_dir} --epochs 30")
    
    return output_dir

if __name__ == "__main__":
    base_dir = r"D:\fyeshi\project\certificate\models"
    
    print("Dataset Organization Script")
    print(f"Base directory: {base_dir}\n")
    
    if not Path(base_dir).exists():
        print(f"ERROR: Directory not found: {base_dir}")
        print("Please update the base_dir path in the script.")
        exit(1)
    
    organize_datasets(base_dir)