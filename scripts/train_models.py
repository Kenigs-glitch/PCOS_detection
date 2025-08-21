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