# Technical Task: PCOS Detection PoC - Model Training & Comparison

## Project Overview
Create a **simple Proof of Concept** to train and compare **EfficientNet-V1-B3** and **ResNet-50** models for PCOS detection using the Anagha Choudhari dataset. Focus on rapid development, training, and comparison - no production infrastructure needed.

## Simple Setup

### **Repository**: https://github.com/Kenigs-glitch/PCOS_detection
### **GPU Server**: `bober@192.168.100.11` (full access)
### **Environment**: Docker for consistency

## Project Structure (Minimal)
```
PCOS_detection/
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── compare.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── training_comparison.ipynb
│   └── results_analysis.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── efficientnet_b3/
│   └── resnet50/
├── results/
│   ├── plots/
│   └── metrics/
├── scripts/
│   ├── download_data.py
│   ├── train_models.py
│   └── compare_models.py
├── config.yaml
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Docker Setup (Simple)

### **Dockerfile**
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Basic dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command
CMD ["bash"]
```

### **requirements.txt**
```text
tensorflow==2.15.0
torch==2.1.0
torchvision==0.16.0
scikit-learn==1.3.0
numpy==1.24.0
pandas==2.1.0
matplotlib==3.7.0
seaborn==0.12.0
opencv-python==4.8.0
pillow==10.0.0
jupyter==1.0.0
kaggle==1.5.16
tqdm==4.66.0
pyyaml==6.0
```

### **docker-compose.yml**
```yaml
version: '3.8'

services:
  pcos-dev:
    build: 
      context: .
      dockerfile: docker/Dockerfile
    container_name: pcos_poc
    volumes:
      - ./src:/app/src
      - ./notebooks:/app/notebooks
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
      - ./scripts:/app/scripts
      - ./config.yaml:/app/config.yaml
    ports:
      - "8888:8888"  # Jupyter
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      bash -c "
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' &
        tail -f /dev/null
      "
```

## Core Implementation

### **Configuration (config.yaml)**
```yaml
# Simple configuration
data:
  source: "anaghachoudhari/pcos-detection-using-ultrasound-images"
  image_size: [300, 300]
  batch_size: 32
  validation_split: 0.2

models:
  efficientnet_b3:
    name: "EfficientNetB3"
    epochs: 50
    learning_rate: 0.001
    
  resnet50:
    name: "ResNet50"  
    epochs: 50
    learning_rate: 0.001

training:
  patience: 10
  save_best_only: true
```

### **Data Loader (src/data_loader.py)**
```python
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import yaml

class PCOSDataLoader:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_dataset(self):
        """Load and split PCOS dataset"""
        data_dir = "/app/data/raw/train"
        
        # Load dataset
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            image_size=self.config['data']['image_size'],
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            seed=42
        )
        
        # Split into train/validation
        val_size = int(len(dataset) * self.config['data']['validation_split'])
        train_ds = dataset.skip(val_size)
        val_ds = dataset.take(val_size)
        
        # Optimize performance
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def get_class_names(self):
        """Get class names"""
        return ['Normal', 'PCOS']
```

### **Models (src/models.py)**
```python
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, ResNet50

class ModelFactory:
    @staticmethod
    def create_efficientnet_b3(input_shape=(300, 300, 3)):
        """Create EfficientNet-B3 model"""
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def create_resnet50(input_shape=(300, 300, 3)):
        """Create ResNet-50 model"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        return model
```

### **Training Script (src/train.py)**
```python
import tensorflow as tf
import yaml
import os
from data_loader import PCOSDataLoader
from models import ModelFactory

class ModelTrainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def train_model(self, model_name):
        """Train a specific model"""
        print(f"🚀 Training {model_name}")
        
        # Load data
        data_loader = PCOSDataLoader()
        train_ds, val_ds = data_loader.load_dataset()
        
        # Create model
        if model_name == 'efficientnet_b3':
            model = ModelFactory.create_efficientnet_b3()
        elif model_name == 'resnet50':
            model = ModelFactory.create_resnet50()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['models'][model_name]['learning_rate']
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.config['training']['patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'/app/models/{model_name}/best_model.h5',
                save_best_only=self.config['training']['save_best_only']
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5
            )
        ]
        
        # Train model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config['models'][model_name]['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        os.makedirs(f'/app/models/{model_name}', exist_ok=True)
        model.save(f'/app/models/{model_name}/final_model.h5')
        
        print(f"✅ {model_name} training completed")
        return history, model

if __name__ == "__main__":
    import sys
    trainer = ModelTrainer()
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        trainer.train_model(model_name)
    else:
        # Train both models
        trainer.train_model('efficientnet_b3')
        trainer.train_model('resnet50')
```

### **Evaluation & Comparison (src/compare.py)**
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

class ModelComparator:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_models(self):
        """Load trained models"""
        model_paths = {
            'efficientnet_b3': '/app/models/efficientnet_b3/final_model.h5',
            'resnet50': '/app/models/resnet50/final_model.h5'
        }
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                self.models[name] = tf.keras.models.load_model(path)
                print(f"✅ Loaded {name}")
            else:
                print(f"❌ Model not found: {path}")
    
    def evaluate_models(self, test_ds):
        """Evaluate all models"""
        for name, model in self.models.items():
            print(f"📊 Evaluating {name}")
            
            # Get predictions
            y_pred = model.predict(test_ds)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Get true labels
            y_true = []
            for _, labels in test_ds:
                y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_true = np.array(y_true)
            
            # Calculate metrics
            accuracy = np.mean(y_pred_classes == y_true)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'y_true': y_true,
                'y_pred': y_pred_classes,
                'y_pred_proba': y_pred
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
    
    def plot_comparison(self):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        
        axes[0, 0].bar(models, accuracies, color=['blue', 'green'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Add accuracy values on bars
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
        
        # Confusion matrices
        for i, (name, results) in enumerate(self.results.items()):
            row = (i + 1) // 2
            col = (i + 1) % 2
            
            cm = confusion_matrix(results['y_true'], results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[row, col])
            axes[row, col].set_title(f'{name} - Confusion Matrix')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('/app/results/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate detailed comparison report"""
        report = "# PCOS Detection Model Comparison Report\n\n"
        
        for name, results in self.results.items():
            report += f"## {name.upper()}\n"
            report += f"- **Accuracy**: {results['accuracy']:.4f}\n"
            
            # Classification report
            class_report = classification_report(
                results['y_true'], 
                results['y_pred'],
                target_names=['Normal', 'PCOS']
            )
            report += f"```\n{class_report}\n```\n\n"
        
        # Save report
        with open('/app/results/comparison_report.md', 'w') as f:
            f.write(report)
        
        print("📄 Report saved to /app/results/comparison_report.md")
        return report

if __name__ == "__main__":
    # Load test data
    from data_loader import PCOSDataLoader
    data_loader = PCOSDataLoader()
    _, test_ds = data_loader.load_dataset()  # Using validation as test for simplicity
    
    # Compare models
    comparator = ModelComparator()
    comparator.load_models()
    comparator.evaluate_models(test_ds)
    comparator.plot_comparison()
    comparator.generate_report()
```

## Simple Scripts

### **Download Data (scripts/download_data.py)**
```python
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
```

### **Training Script (scripts/train_models.py)**
```python
#!/usr/bin/env python3
import sys
sys.path.append('/app/src')

from train import ModelTrainer

def main():
    trainer = ModelTrainer()
    
    print("🚀 Starting PCOS Model Training")
    
    # Train EfficientNet-B3
    print("\n" + "="*50)
    trainer.train_model('efficientnet_b3')
    
    # Train ResNet-50
    print("\n" + "="*50)
    trainer.train_model('resnet50')
    
    print("\n✅ All models trained successfully!")

if __name__ == "__main__":
    main()
```

### **Comparison Script (scripts/compare_models.py)**
```python
#!/usr/bin/env python3
import sys
sys.path.append('/app/src')

from compare import ModelComparator
from data_loader import PCOSDataLoader

def main():
    print("📊 Starting Model Comparison")
    
    # Load test data
    data_loader = PCOSDataLoader()
    _, test_ds = data_loader.load_dataset()
    
    # Compare models
    comparator = ModelComparator()
    comparator.load_models()
    comparator.evaluate_models(test_ds)
    comparator.plot_comparison()
    comparator.generate_report()
    
    print("✅ Model comparison completed!")

if __name__ == "__main__":
    main()
```

## Simple Deployment

### **Deploy to GPU Server (simple)**
```bash
#!/bin/bash
# scripts/deploy.sh

SERVER="bober@192.168.100.11"
PROJECT_DIR="/home/bober/PCOS_detection"

echo "🚀 Deploying to GPU server..."

# Create project directory on server
ssh $SERVER "mkdir -p $PROJECT_DIR"

# Copy project files
rsync -avz --exclude='.git' --exclude='__pycache__' . $SERVER:$PROJECT_DIR/

# Setup and run on server
ssh $SERVER << EOF
    cd $PROJECT_DIR
    
    # Build container
    docker-compose build
    
    # Download data
    docker-compose run --rm pcos-dev python scripts/download_data.py
    
    # Train models
    docker-compose run --rm pcos-dev python scripts/train_models.py
    
    # Compare models
    docker-compose run --rm pcos-dev python scripts/compare_models.py
    
    echo "✅ Training and comparison completed!"
EOF

echo "✅ Deployment completed!"
```

## Usage

### **Local Development**
```bash
# 1. Clone repository
git clone https://github.com/Kenigs-glitch/PCOS_detection.git
cd PCOS_detection

# 2. Setup Kaggle credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Build and start
docker-compose up -d

# 4. Download data
docker-compose exec pcos-dev python scripts/download_data.py

# 5. Train models
docker-compose exec pcos-dev python scripts/train_models.py

# 6. Compare results
docker-compose exec pcos-dev python scripts/compare_models.py

# 7. Access Jupyter (optional)
# Open: http://localhost:8888
```

### **GPU Server Deployment**
```bash
# Simple one-command deployment
./scripts/deploy.sh

# Or manual deployment
rsync -avz . bober@192.168.100.11:/home/bober/PCOS_detection/
ssh bober@192.168.100.11 "cd /home/bober/PCOS_detection && docker-compose run --rm pcos-dev python scripts/train_models.py"
```

## Expected Results

### **Training Time (RTX 3090)**
- **EfficientNet-B3**: ~45 minutes
- **ResNet-50**: ~60 minutes  
- **Total PoC**: ~2 hours

### **Deliverables**
- ✅ Two trained models (EfficientNet-B3, ResNet-50)
- ✅ Performance comparison report
- ✅ Visualization plots
- ✅ Model accuracy metrics
- ✅ Simple deployment setup

This simplified approach focuses on **rapid proof-of-concept development** without production complexity - perfect for demonstrating model performance and getting quick results! 🚀