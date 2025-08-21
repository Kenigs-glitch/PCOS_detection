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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.callbacks import Callback
import pandas as pd
from metrics import PCOSMetrics, calculate_comprehensive_metrics, plot_comprehensive_metrics, generate_metrics_report
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU devices found: {len(gpus)}")
        print(f"GPU devices: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠️ No GPU devices found, using CPU")

class GraphConvolutionLayer(tf.keras.layers.Layer):
    """Graph Convolution Layer for GNN"""
    def __init__(self, output_dim, activation='relu', dropout_rate=0.1, **kwargs):
        super(GraphConvolutionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # Input shape: (batch_size, num_nodes, input_dim)
        self.W = self.add_weight(
            name='weight',
            shape=(input_shape[-1], self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, inputs, adj_matrix):
        # Graph convolution: H' = σ(D^(-1/2) * A * D^(-1/2) * H * W + b)
        # inputs: node features (batch_size, num_nodes, input_dim)
        # adj_matrix: adjacency matrix (batch_size, num_nodes, num_nodes)
        
        # Normalize adjacency matrix
        D = tf.reduce_sum(adj_matrix, axis=-1, keepdims=True)  # Degree matrix
        D_inv_sqrt = tf.math.rsqrt(tf.maximum(D, 1e-8))  # D^(-1/2)
        adj_norm = adj_matrix * D_inv_sqrt * tf.transpose(D_inv_sqrt, [0, 2, 1])
        
        # Graph convolution
        support = tf.matmul(inputs, self.W)  # H * W
        output = tf.matmul(adj_norm, support)  # A_norm * (H * W)
        output = output + self.b  # Add bias
        output = self.activation(output)  # Apply activation
        output = self.dropout(output)  # Apply dropout
        
        return output

class GraphNeuralNetwork(tf.keras.Model):
    """Graph Neural Network for PCOS Detection"""
    
    def __init__(self, num_classes=2, hidden_dims=[128, 64, 32], dropout_rate=0.3, **kwargs):
        super(GraphNeuralNetwork, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Graph convolution layers
        self.gcn_layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            self.gcn_layers.append(
                GraphConvolutionLayer(
                    output_dim=hidden_dim,
                    activation='relu',
                    dropout_rate=dropout_rate,
                    name=f'gcn_layer_{i}'
                )
            )
        
        # Global pooling and classification
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])
        
    def call(self, inputs, training=None):
        node_features, adj_matrix = inputs
        
        # Apply graph convolution layers
        x = node_features
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, adj_matrix)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Classification
        output = self.classifier(x, training=training)
        
        return output

class GraphDataGenerator:
    """Generate graph data from ultrasound images"""
    
    def __init__(self, k_neighbors=8, feature_dim=512):
        self.k_neighbors = k_neighbors
        self.feature_dim = feature_dim
        self.feature_extractor = self._create_feature_extractor()
        self.scaler = StandardScaler()
        
    def _create_feature_extractor(self):
        """Create CNN feature extractor"""
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        return base_model
    
    def extract_features(self, images):
        """Extract features from images"""
        # Resize images to 224x224 for EfficientNet
        resized_images = tf.image.resize(images, (224, 224))
        
        # Preprocess for EfficientNet
        preprocessed_images = tf.keras.applications.efficientnet.preprocess_input(resized_images)
        
        # Extract features
        features = self.feature_extractor(preprocessed_images)
        
        return features
    
    def build_graph(self, features, labels):
        """Build graph from features"""
        # Convert to numpy for sklearn
        features_np = features.numpy()
        
        # Build k-nearest neighbors graph
        adj_matrix = kneighbors_graph(
            features_np, 
            n_neighbors=self.k_neighbors, 
            mode='connectivity',
            include_self=True
        ).toarray()
        
        # Normalize adjacency matrix
        adj_matrix = adj_matrix.astype(np.float32)
        
        # Add self-loops
        adj_matrix += np.eye(adj_matrix.shape[0])
        
        # Normalize by degree
        degree = np.sum(adj_matrix, axis=1, keepdims=True)
        adj_matrix = adj_matrix / np.maximum(degree, 1e-8)
        
        return adj_matrix
    
    def generate_graph_data(self, images, labels):
        """Generate graph data from images"""
        print("🔍 Extracting features from images...")
        features = self.extract_features(images)
        
        print("🕸️ Building graph structure...")
        adj_matrix = self.build_graph(features, labels)
        
        # Normalize features
        features_np = features.numpy()
        features_normalized = self.scaler.fit_transform(features_np)
        
        return features_normalized.astype(np.float32), adj_matrix.astype(np.float32)

class MetricsMonitor(Callback):
    """Custom callback for real-time metrics monitoring"""
    
    def __init__(self, log_dir='/app/results/metrics'):
        super(MetricsMonitor, self).__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_file = os.path.join(log_dir, 'gnn_training_metrics.csv')
        self.metrics_data = []
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # Add epoch number
        logs['epoch'] = epoch + 1
        logs['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to CSV
        df = pd.DataFrame([logs])
        if os.path.exists(self.metrics_file):
            df.to_csv(self.metrics_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.metrics_file, index=False)
        
        # Print progress
        print(f"Epoch {epoch+1}: loss={logs.get('loss', 0):.4f}, "
              f"accuracy={logs.get('accuracy', 0):.4f}, "
              f"val_loss={logs.get('val_loss', 0):.4f}, "
              f"val_accuracy={logs.get('val_accuracy', 0):.4f}")

class GNNTrainer:
    """Graph Neural Network trainer for PCOS detection"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.history = None
        self.data_generator = GraphDataGenerator()
        
    def load_data(self):
        """Load and prepare graph data"""
        print("📊 Loading PCOS dataset...")
        
        # Load images using existing data loader
        data_dir = "/app/data/raw/data/train"
        
        # Load dataset
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=None,  # Load all images
            image_size=(300, 300),
            shuffle=True,
            seed=42
        )
        
        # Convert to numpy arrays
        images = []
        labels = []
        
        for image, label in dataset:
            images.append(image.numpy())
            labels.append(label.numpy())
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"✅ Loaded {len(images)} images with shape {images.shape}")
        print(f"📊 Class distribution: {np.sum(labels, axis=0)}")
        
        # Generate graph data
        features, adj_matrix = self.data_generator.generate_graph_data(images, labels)
        
        # Split data
        split_idx = int(0.8 * len(features))
        train_features = features[:split_idx]
        train_adj = adj_matrix[:split_idx, :split_idx]
        train_labels = labels[:split_idx]
        
        val_features = features[split_idx:]
        val_adj = adj_matrix[split_idx:, split_idx:]
        val_labels = labels[split_idx:]
        
        print(f"📊 Train: {len(train_features)}, Validation: {len(val_features)}")
        
        return (train_features, train_adj, train_labels), (val_features, val_adj, val_labels)
    
    def create_model(self):
        """Create GNN model"""
        print("🏗️ Creating Graph Neural Network model...")
        
        self.model = GraphNeuralNetwork(
            num_classes=2,
            hidden_dims=[128, 64, 32],
            dropout_rate=0.3
        )
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ GNN model created successfully")
        
    def train(self):
        """Train the GNN model"""
        print("🚀 Starting Graph Neural Network training...")
        
        # Create model and load data
        self.create_model()
        (train_features, train_adj, train_labels), (val_features, val_adj, val_labels) = self.load_data()
        
        # Setup callbacks
        callbacks = [
            PCOSMetrics(),
            MetricsMonitor(),
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                verbose=1,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                '/app/models/gnn/best_model.h5',
                save_best_only=True,
                verbose=1,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                min_lr=1e-6,
                verbose=1,
                monitor='val_accuracy'
            )
        ]
        
        # Create data generators
        train_dataset = tf.data.Dataset.from_tensor_slices((
            (train_features, train_adj),
            train_labels
        )).batch(32).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            (val_features, val_adj),
            val_labels
        )).batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Train model
        print("🎯 Training GNN model...")
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=50,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        os.makedirs('/app/models/gnn', exist_ok=True)
        self.model.save('/app/models/gnn/final_model.h5')
        
        print("✅ GNN training completed!")
        return self.history
    
    def evaluate_model(self, val_dataset):
        """Evaluate the trained model"""
        print("📊 Evaluating GNN model...")
        
        # Evaluate on validation set
        results = self.model.evaluate(val_dataset, verbose=1)
        
        # Generate predictions
        predictions = self.model.predict(val_dataset)
        y_pred = np.argmax(predictions, axis=1)
        
        # Get true labels
        y_true = []
        for _, labels in val_dataset:
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_true = np.array(y_true)
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(y_true, y_pred, predictions)
        
        # Save results
        os.makedirs('/app/results/plots', exist_ok=True)
        plot_comprehensive_metrics(metrics, '/app/results/plots/gnn_metrics.png')
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('GNN Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('/app/results/plots/gnn_confusion_matrix.png')
        plt.close()
        
        # Save detailed report
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('/app/results/metrics/gnn_classification_report.csv')
        
        print("✅ GNN evaluation completed!")
        return results[1]  # Return accuracy
    
    def generate_visualizations(self):
        """Generate training visualizations"""
        if self.history is None:
            print("⚠️ No training history available")
            return
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('/app/results/plots/gnn_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ GNN visualizations saved")

def main():
    """Main training function"""
    print("🎯 Graph Neural Network PCOS Detection Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = GNNTrainer()
    
    # Train model
    history = trainer.train()
    
    # Load data for evaluation
    (_, _, _), (val_features, val_adj, val_labels) = trainer.load_data()
    val_dataset = tf.data.Dataset.from_tensor_slices((
        (val_features, val_adj),
        val_labels
    )).batch(32)
    
    # Evaluate model
    accuracy = trainer.evaluate_model(val_dataset)
    
    # Generate visualizations
    trainer.generate_visualizations()
    
    print("\n🎉 GNN Training and evaluation completed!")
    print(f"📊 Final Model Accuracy: {accuracy:.4f}")
    print("\n📁 Results saved in:")
    print("   - /app/results/plots/")
    print("   - /app/results/metrics/")
    print("   - /app/models/gnn/")

if __name__ == "__main__":
    main() 