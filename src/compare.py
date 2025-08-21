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