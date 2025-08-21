#!/usr/bin/env python3
"""
Test script to verify PCOS detection setup
"""
import sys
import os
sys.path.append('/app/src')

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from src.data_loader import PCOSDataLoader
        print("✅ PCOSDataLoader imported")
    except ImportError as e:
        print(f"❌ PCOSDataLoader import failed: {e}")
        return False
    
    try:
        from src.models import ModelFactory
        print("✅ ModelFactory imported")
    except ImportError as e:
        print(f"❌ ModelFactory import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if models can be created"""
    print("\n🔍 Testing model creation...")
    
    try:
        from src.models import ModelFactory
        
        # Test EfficientNet-B3
        efficientnet = ModelFactory.create_efficientnet_b3()
        print(f"✅ EfficientNet-B3 created: {efficientnet.count_params():,} parameters")
        
        # Test ResNet-50
        resnet = ModelFactory.create_resnet50()
        print(f"✅ ResNet-50 created: {resnet.count_params():,} parameters")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_config():
    """Test if configuration can be loaded"""
    print("\n🔍 Testing configuration...")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration loaded successfully")
        print(f"   Image size: {config['data']['image_size']}")
        print(f"   Batch size: {config['data']['batch_size']}")
        print(f"   Models: {list(config['models'].keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_gpu():
    """Test GPU availability"""
    print("\n🔍 Testing GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   {gpu.name}")
        else:
            print("⚠️  No GPU detected - training will use CPU")
        
        return True
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 PCOS Detection Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_config,
        test_gpu
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Setup is ready for training.")
        return True
    else:
        print("❌ Some tests failed. Please check the setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 