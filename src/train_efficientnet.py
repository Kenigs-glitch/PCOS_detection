import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import Callback
import pandas as pd

class MetricsMonitor(Callback):
    """Custom callback for real-time metrics monitoring"""
    def __init__(self, log_dir='/app/results/metrics'):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': []
        }
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        # Store metrics
        for metric, value in logs.items():
            if metric in self.metrics_history:
                self.metrics_history[metric].append(value)
        
        # Calculate training time
        elapsed_time = time.time() - self.start_time
        
        # Print progress
        print(f"\n📊 Epoch {epoch + 1} Metrics:")
        print(f"   Training - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
        print(f"   Validation - Loss: {logs['val_loss']:.4f}, Accuracy: {logs['val_accuracy']:.4f}")
        print(f"   Time elapsed: {elapsed_time/60:.1f} minutes")
        
        # Save metrics to file
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(f'{self.log_dir}/training_metrics.csv', index=False)
        
        # Create real-time plot
        self.plot_metrics()
    
    def plot_metrics(self):
        """Create real-time metrics visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.metrics_history['loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(self.metrics_history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.metrics_history['accuracy'], label='Training Accuracy', color='blue')
        axes[0, 1].plot(self.metrics_history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision plot
        axes[1, 0].plot(self.metrics_history['precision'], label='Training Precision', color='blue')
        axes[1, 0].plot(self.metrics_history['val_precision'], label='Validation Precision', color='red')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall plot
        axes[1, 1].plot(self.metrics_history['recall'], label='Training Recall', color='blue')
        axes[1, 1].plot(self.metrics_history['val_recall'], label='Validation Recall', color='red')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()

class GradCAM:
    """Grad-CAM implementation for model interpretability"""
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    self.layer_name = layer.name
                    break
    
    def generate_cam(self, image, class_index=None):
        """Generate Grad-CAM for a given image"""
        # Create a model that outputs the last conv layer and predictions
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            class_channel = predictions[:, class_index]
        
        # Compute gradients of the class with respect to the output feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by corresponding gradients
        conv_outputs = conv_outputs[0]
        cam = conv_outputs @ pooled_grads[..., tf.newaxis]
        cam = tf.squeeze(cam)
        
        # Apply ReLU and normalize
        cam = tf.maximum(cam, 0) / tf.math.reduce_max(cam)
        
        return cam.numpy()

class EfficientNetTrainer:
    """Focused EfficientNet-B3 trainer with comprehensive analytics"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.history = None
        self.test_predictions = None
        self.test_labels = None
        
    def create_model(self):
        """Create EfficientNet-B3 model with custom head"""
        print("🏗️ Creating EfficientNet-B3 model...")
        
        # Base model
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(300, 300, 3),
            include_preprocessing=True
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Custom classification head
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(2, activation='softmax', name='predictions')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['models']['efficientnet_b3']['learning_rate']
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        self.model = model
        return model
    
    def load_data(self):
        """Load and preprocess dataset"""
        print("📥 Loading PCOS dataset...")
        
        # Load dataset
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            '/app/data/raw/train',
            labels='inferred',
            label_mode='categorical',
            image_size=(300, 300),
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            seed=42,
            color_mode='rgb'
        )
        
        # Split into train/validation
        val_size = int(len(dataset) * self.config['data']['validation_split'])
        train_ds = dataset.skip(val_size)
        val_ds = dataset.take(val_size)
        
        # Optimize performance
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        
        print(f"✅ Dataset loaded - Train: {len(train_ds)}, Validation: {len(val_ds)}")
        return train_ds, val_ds
    
    def train(self):
        """Train the EfficientNet-B3 model"""
        print("🚀 Starting EfficientNet-B3 training...")
        
        # Create model and load data
        self.create_model()
        train_ds, val_ds = self.load_data()
        
        # Setup callbacks
        callbacks = [
            MetricsMonitor(),
            tf.keras.callbacks.EarlyStopping(
                patience=self.config['training']['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                '/app/models/efficientnet_b3/best_model.h5',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config['models']['efficientnet_b3']['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        os.makedirs('/app/models/efficientnet_b3', exist_ok=True)
        self.model.save('/app/models/efficientnet_b3/final_model.h5')
        
        print("✅ Training completed!")
        return self.history
    
    def evaluate_model(self, test_ds):
        """Evaluate model and generate comprehensive analytics"""
        print("📊 Evaluating model performance...")
        
        # Get predictions
        self.test_predictions = self.model.predict(test_ds)
        self.test_labels = []
        
        for _, labels in test_ds:
            self.test_labels.extend(np.argmax(labels.numpy(), axis=1))
        
        self.test_labels = np.array(self.test_labels)
        pred_classes = np.argmax(self.test_predictions, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(pred_classes == self.test_labels)
        
        print(f"📈 Model Accuracy: {accuracy:.4f}")
        
        # Generate comprehensive analytics
        self.generate_confusion_matrix(pred_classes)
        self.generate_classification_report(pred_classes)
        self.analyze_failures(pred_classes)
        self.generate_grad_cams(test_ds)
        
        return accuracy
    
    def generate_confusion_matrix(self, pred_classes):
        """Generate and save confusion matrix"""
        print("📋 Generating confusion matrix...")
        
        cm = confusion_matrix(self.test_labels, pred_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'PCOS'],
                   yticklabels=['Normal', 'PCOS'])
        plt.title('EfficientNet-B3 Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('/app/results/plots/efficientnet_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Confusion matrix saved")
    
    def generate_classification_report(self, pred_classes):
        """Generate detailed classification report"""
        print("📄 Generating classification report...")
        
        report = classification_report(
            self.test_labels, pred_classes,
            target_names=['Normal', 'PCOS'],
            output_dict=True
        )
        
        # Save as markdown
        with open('/app/results/efficientnet_classification_report.md', 'w') as f:
            f.write("# EfficientNet-B3 Classification Report\n\n")
            f.write(f"## Overall Accuracy: {report['accuracy']:.4f}\n\n")
            
            for class_name in ['Normal', 'PCOS']:
                f.write(f"## {class_name}\n")
                f.write(f"- Precision: {report[class_name]['precision']:.4f}\n")
                f.write(f"- Recall: {report[class_name]['recall']:.4f}\n")
                f.write(f"- F1-Score: {report[class_name]['f1-score']:.4f}\n")
                f.write(f"- Support: {report[class_name]['support']}\n\n")
        
        print("✅ Classification report saved")
    
    def analyze_failures(self, pred_classes):
        """Analyze model failures and misclassifications"""
        print("🔍 Analyzing model failures...")
        
        # Find misclassifications
        misclassified = pred_classes != self.test_labels
        
        if np.sum(misclassified) > 0:
            failure_rate = np.mean(misclassified)
            print(f"❌ Failure Rate: {failure_rate:.4f} ({np.sum(misclassified)} misclassifications)")
            
            # Analyze failure patterns
            failure_analysis = {
                'total_samples': len(self.test_labels),
                'misclassifications': int(np.sum(misclassified)),
                'failure_rate': float(failure_rate),
                'false_positives': int(np.sum((pred_classes == 1) & (self.test_labels == 0))),
                'false_negatives': int(np.sum((pred_classes == 0) & (self.test_labels == 1)))
            }
            
            # Save failure analysis
            with open('/app/results/efficientnet_failure_analysis.md', 'w') as f:
                f.write("# EfficientNet-B3 Failure Analysis\n\n")
                f.write(f"- Total Samples: {failure_analysis['total_samples']}\n")
                f.write(f"- Misclassifications: {failure_analysis['misclassifications']}\n")
                f.write(f"- Failure Rate: {failure_analysis['failure_rate']:.4f}\n")
                f.write(f"- False Positives: {failure_analysis['false_positives']}\n")
                f.write(f"- False Negatives: {failure_analysis['false_negatives']}\n")
            
            print("✅ Failure analysis saved")
        else:
            print("🎉 No misclassifications found!")
    
    def generate_grad_cams(self, test_ds, num_samples=10):
        """Generate Grad-CAM visualizations for model interpretability"""
        print("🎯 Generating Grad-CAM visualizations...")
        
        # Create GradCAM instance
        grad_cam = GradCAM(self.model)
        
        # Get sample images
        sample_images = []
        sample_labels = []
        sample_predictions = []
        
        for images, labels in test_ds.take(1):
            sample_images = images[:num_samples]
            sample_labels = np.argmax(labels[:num_samples].numpy(), axis=1)
            sample_predictions = np.argmax(self.model.predict(sample_images), axis=1)
            break
        
        # Generate Grad-CAM for each sample
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(sample_images[i].numpy().astype('uint8'))
            axes[0, i].set_title(f'True: {["Normal", "PCOS"][sample_labels[i]]}\nPred: {["Normal", "PCOS"][sample_predictions[i]]}')
            axes[0, i].axis('off')
            
            # Grad-CAM
            cam = grad_cam.generate_cam(sample_images[i:i+1])
            cam_resized = cv2.resize(cam, (300, 300))
            
            # Overlay Grad-CAM on original image
            img = sample_images[i].numpy().astype('uint8')
            cam_colored = cv2.applyColorMap((cam_resized * 255).astype('uint8'), cv2.COLORMAP_JET)
            cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
            
            # Blend images
            alpha = 0.6
            blended = cv2.addWeighted(img, 1-alpha, cam_colored, alpha, 0)
            
            axes[1, i].imshow(blended)
            axes[1, i].set_title('Grad-CAM')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('/app/results/plots/efficientnet_gradcam.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Grad-CAM visualizations saved")

def main():
    """Main training function"""
    print("🎯 EfficientNet-B3 PCOS Detection Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = EfficientNetTrainer()
    
    # Train model
    history = trainer.train()
    
    # Evaluate model
    train_ds, val_ds = trainer.load_data()
    accuracy = trainer.evaluate_model(val_ds)
    
    print("\n🎉 Training and evaluation completed!")
    print(f"📊 Final Model Accuracy: {accuracy:.4f}")
    print("\n📁 Results saved in:")
    print("   - /app/results/plots/")
    print("   - /app/results/metrics/")
    print("   - /app/models/efficientnet_b3/")

if __name__ == "__main__":
    main() 