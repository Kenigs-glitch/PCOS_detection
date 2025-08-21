"""
Comprehensive metrics for PCOS detection model evaluation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import os
import time


class PCOSMetrics(Callback):
    """Custom callback for comprehensive PCOS detection metrics"""
    
    def __init__(self, log_dir='/app/results/metrics'):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1_score': [],
            'auc_score': [], 'val_auc_score': [],
            'specificity': [], 'val_specificity': [],
            'sensitivity': [], 'val_sensitivity': []
        }
        self.start_time = time.time()
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Store basic metrics
        for metric, value in logs.items():
            if metric in self.metrics_history:
                self.metrics_history[metric].append(value)
        
        # Calculate additional metrics if we have predictions
        if len(self.y_true) > 0 and len(self.y_pred) > 0:
            y_true = np.array(self.y_true)
            y_pred = np.array(self.y_pred)
            y_pred_proba = np.array(self.y_pred_proba)
            
            # Calculate comprehensive metrics
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # AUC score (for binary classification)
            if len(np.unique(y_true)) == 2:
                auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                auc_score = 0.0
            
            # Specificity and Sensitivity (for binary classification)
            if len(np.unique(y_true)) == 2:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                specificity = 0.0
                sensitivity = 0.0
            
            # Store calculated metrics
            self.metrics_history['precision'].append(precision)
            self.metrics_history['recall'].append(recall)
            self.metrics_history['f1_score'].append(f1)
            self.metrics_history['auc_score'].append(auc_score)
            self.metrics_history['specificity'].append(specificity)
            self.metrics_history['sensitivity'].append(sensitivity)
        
        # Calculate training time
        elapsed_time = time.time() - self.start_time
        
        # Print comprehensive progress
        print(f"\n📊 Epoch {epoch + 1} Metrics:")
        print(f"   Training - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
        if 'precision' in self.metrics_history and len(self.metrics_history['precision']) > 0:
            print(f"   Training - Precision: {self.metrics_history['precision'][-1]:.4f}, Recall: {self.metrics_history['recall'][-1]:.4f}, F1: {self.metrics_history['f1_score'][-1]:.4f}")
        print(f"   Validation - Loss: {logs['val_loss']:.4f}, Accuracy: {logs['val_accuracy']:.4f}")
        if 'val_precision' in self.metrics_history and len(self.metrics_history['val_precision']) > 0:
            print(f"   Validation - Precision: {self.metrics_history['val_precision'][-1]:.4f}, Recall: {self.metrics_history['val_recall'][-1]:.4f}, F1: {self.metrics_history['val_f1_score'][-1]:.4f}")
        print(f"   Time elapsed: {elapsed_time/60:.1f} minutes")
        
        # Save metrics to file
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(f'{self.log_dir}/training_metrics.csv', index=False)
        
        # Create real-time plot
        self.plot_metrics()
    
    def plot_metrics(self):
        """Create comprehensive real-time metrics visualization"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Loss plot
        if len(self.metrics_history['loss']) > 0:
            axes[0, 0].plot(self.metrics_history['loss'], label='Training Loss', color='blue')
            axes[0, 0].plot(self.metrics_history['val_loss'], label='Validation Loss', color='red')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy plot
        if len(self.metrics_history['accuracy']) > 0:
            axes[0, 1].plot(self.metrics_history['accuracy'], label='Training Accuracy', color='blue')
            axes[0, 1].plot(self.metrics_history['val_accuracy'], label='Validation Accuracy', color='red')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Precision plot
        if len(self.metrics_history['precision']) > 0:
            axes[1, 0].plot(self.metrics_history['precision'], label='Training Precision', color='blue')
            axes[1, 0].plot(self.metrics_history['val_precision'], label='Validation Precision', color='red')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall plot
        if len(self.metrics_history['recall']) > 0:
            axes[1, 1].plot(self.metrics_history['recall'], label='Training Recall', color='blue')
            axes[1, 1].plot(self.metrics_history['val_recall'], label='Validation Recall', color='red')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # F1 Score plot
        if len(self.metrics_history['f1_score']) > 0:
            axes[2, 0].plot(self.metrics_history['f1_score'], label='Training F1-Score', color='blue')
            axes[2, 0].plot(self.metrics_history['val_f1_score'], label='Validation F1-Score', color='red')
            axes[2, 0].set_title('Model F1-Score')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('F1-Score')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
        
        # AUC Score plot
        if len(self.metrics_history['auc_score']) > 0:
            axes[2, 1].plot(self.metrics_history['auc_score'], label='Training AUC', color='blue')
            axes[2, 1].plot(self.metrics_history['val_auc_score'], label='Validation AUC', color='red')
            axes[2, 1].set_title('Model AUC Score')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('AUC Score')
            axes[2, 1].legend()
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()


def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None, class_names=None):
    """
    Calculate comprehensive metrics for PCOS detection
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        class_names: Names of classes (optional)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    if class_names is None:
        class_names = ['Normal', 'PCOS']
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Binary classification specific metrics
    if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
        auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        # Calculate specificity and sensitivity
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate positive and negative predictive values
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Calculate balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Calculate Matthews Correlation Coefficient
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
    else:
        auc_score = 0.0
        specificity = 0.0
        sensitivity = 0.0
        ppv = 0.0
        npv = 0.0
        balanced_accuracy = 0.0
        mcc = 0.0
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'ppv': ppv,  # Positive Predictive Value
        'npv': npv,  # Negative Predictive Value
        'balanced_accuracy': balanced_accuracy,
        'matthews_correlation': mcc,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics


def plot_comprehensive_metrics(metrics, save_path='/app/results/plots/'):
    """
    Create comprehensive visualization of model metrics
    
    Args:
        metrics: Dictionary containing metrics from calculate_comprehensive_metrics
        save_path: Path to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'PCOS'], 
                yticklabels=['Normal', 'PCOS'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. Metrics Bar Chart
    plt.subplot(2, 3, 2)
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Balanced Acc']
    metric_values = [
        metrics['accuracy'], metrics['precision'], metrics['recall'],
        metrics['f1_score'], metrics['auc_score'], metrics['balanced_accuracy']
    ]
    bars = plt.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Sensitivity vs Specificity
    plt.subplot(2, 3, 3)
    plt.bar(['Sensitivity', 'Specificity'], 
            [metrics['sensitivity'], metrics['specificity']], 
            color=['green', 'blue'])
    plt.title('Sensitivity vs Specificity')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels
    plt.text(0, metrics['sensitivity'] + 0.01, f'{metrics["sensitivity"]:.3f}', ha='center')
    plt.text(1, metrics['specificity'] + 0.01, f'{metrics["specificity"]:.3f}', ha='center')
    
    # 4. PPV vs NPV
    plt.subplot(2, 3, 4)
    plt.bar(['PPV', 'NPV'], [metrics['ppv'], metrics['npv']], color=['orange', 'red'])
    plt.title('Positive vs Negative Predictive Value')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels
    plt.text(0, metrics['ppv'] + 0.01, f'{metrics["ppv"]:.3f}', ha='center')
    plt.text(1, metrics['npv'] + 0.01, f'{metrics["npv"]:.3f}', ha='center')
    
    # 5. Matthews Correlation Coefficient
    plt.subplot(2, 3, 5)
    mcc = metrics['matthews_correlation']
    plt.bar(['MCC'], [mcc], color='purple')
    plt.title('Matthews Correlation Coefficient')
    plt.ylabel('Score')
    plt.ylim(-1, 1)
    
    # Add value label
    plt.text(0, mcc + 0.01 if mcc >= 0 else mcc - 0.01, f'{mcc:.3f}', ha='center')
    
    # 6. Summary Statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = f"""
    Model Performance Summary
    
    Overall Accuracy: {metrics['accuracy']:.3f}
    Precision: {metrics['precision']:.3f}
    Recall: {metrics['recall']:.3f}
    F1-Score: {metrics['f1_score']:.3f}
    AUC Score: {metrics['auc_score']:.3f}
    
    Sensitivity: {metrics['sensitivity']:.3f}
    Specificity: {metrics['specificity']:.3f}
    PPV: {metrics['ppv']:.3f}
    NPV: {metrics['npv']:.3f}
    
    Balanced Accuracy: {metrics['balanced_accuracy']:.3f}
    Matthews CC: {metrics['matthews_correlation']:.3f}
    """
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/comprehensive_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_metrics_report(metrics, save_path='/app/results/'):
    """
    Generate a comprehensive metrics report in markdown format
    
    Args:
        metrics: Dictionary containing metrics from calculate_comprehensive_metrics
        save_path: Path to save the report
    """
    os.makedirs(save_path, exist_ok=True)
    
    report_content = f"""# PCOS Detection Model Performance Report

## Executive Summary

The PCOS detection model achieved the following performance metrics:

- **Overall Accuracy**: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)
- **Precision**: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)
- **Recall**: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)
- **F1-Score**: {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)
- **AUC Score**: {metrics['auc_score']:.3f} ({metrics['auc_score']*100:.1f}%)

## Detailed Metrics

### Binary Classification Metrics

- **Sensitivity (True Positive Rate)**: {metrics['sensitivity']:.3f} ({metrics['sensitivity']*100:.1f}%)
- **Specificity (True Negative Rate)**: {metrics['specificity']:.3f} ({metrics['specificity']*100:.1f}%)
- **Positive Predictive Value (PPV)**: {metrics['ppv']:.3f} ({metrics['ppv']*100:.1f}%)
- **Negative Predictive Value (NPV)**: {metrics['npv']:.3f} ({metrics['npv']*100:.1f}%)
- **Balanced Accuracy**: {metrics['balanced_accuracy']:.3f} ({metrics['balanced_accuracy']*100:.1f}%)
- **Matthews Correlation Coefficient**: {metrics['matthews_correlation']:.3f}

### Model Quality Assessment

- **AUC Score**: {metrics['auc_score']:.3f} - {'Excellent' if metrics['auc_score'] >= 0.9 else 'Good' if metrics['auc_score'] >= 0.8 else 'Fair' if metrics['auc_score'] >= 0.7 else 'Poor'} discrimination ability
- **Balanced Accuracy**: {metrics['balanced_accuracy']:.3f} - {'Excellent' if metrics['balanced_accuracy'] >= 0.9 else 'Good' if metrics['balanced_accuracy'] >= 0.8 else 'Fair' if metrics['balanced_accuracy'] >= 0.7 else 'Poor'} performance on imbalanced data
- **Matthews Correlation**: {metrics['matthews_correlation']:.3f} - {'Strong' if abs(metrics['matthews_correlation']) >= 0.7 else 'Moderate' if abs(metrics['matthews_correlation']) >= 0.5 else 'Weak'} correlation between predictions and actual values

## Clinical Relevance

### For PCOS Detection:
- **Sensitivity**: {metrics['sensitivity']:.3f} - The model correctly identifies {metrics['sensitivity']*100:.1f}% of actual PCOS cases
- **Specificity**: {metrics['specificity']:.3f} - The model correctly identifies {metrics['specificity']*100:.1f}% of normal cases
- **PPV**: {metrics['ppv']:.3f} - When the model predicts PCOS, it is correct {metrics['ppv']*100:.1f}% of the time
- **NPV**: {metrics['npv']:.3f} - When the model predicts normal, it is correct {metrics['npv']*100:.1f}% of the time

## Recommendations

Based on the performance metrics:

1. **Model Performance**: {'The model shows excellent performance' if metrics['accuracy'] >= 0.9 else 'The model shows good performance' if metrics['accuracy'] >= 0.8 else 'The model shows fair performance' if metrics['accuracy'] >= 0.7 else 'The model needs improvement'} for PCOS detection.

2. **Clinical Application**: {'Suitable for clinical use' if metrics['auc_score'] >= 0.8 and metrics['balanced_accuracy'] >= 0.8 else 'May need additional validation before clinical use'}.

3. **Areas for Improvement**: {'Consider data augmentation' if metrics['recall'] < metrics['precision'] else 'Consider addressing class imbalance' if metrics['balanced_accuracy'] < metrics['accuracy'] else 'Model performance is well-balanced'}.

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(f'{save_path}/comprehensive_metrics_report.md', 'w') as f:
        f.write(report_content)
    
    return report_content 