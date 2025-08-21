#!/usr/bin/env python3
import os
import kaggle

def download_pcos_data():
    """Download PCOS dataset from Kaggle"""
    print("📥 Downloading PCOS dataset...")
    
    # Create data directory
    os.makedirs('/app/data/raw', exist_ok=True)
    
    # Download dataset
    kaggle.api.dataset_download_files(
        'anaghachoudhari/pcos-detection-using-ultrasound-images',
        path='/app/data/raw',
        unzip=True
    )
    
    print("✅ Dataset downloaded successfully!")

if __name__ == "__main__":
    download_pcos_data() 