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