import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import torch_geometric.nn as gnn
from torchvision import transforms
from PIL import Image
import cv2
from skimage.feature import canny
from skimage.filters import sobel
from skimage.measure import regionprops
from scipy.spatial import Delaunay
import networkx as nx
import warnings
import yaml
from datetime import datetime
import seaborn as sns
from metrics import calculate_comprehensive_metrics, plot_comprehensive_metrics, generate_metrics_report

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configure GPU memory growth for TensorFlow compatibility
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU devices found: {len(gpus)}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

class GraphDataProcessor:
    """Process ultrasound images and convert them to graph representations"""
    
    def __init__(self, data_dir="/app/data/raw/data/train"):
        self.data_dir = data_dir
        
    def convert_image_to_graph(self, image_path, threshold=50):
        """
        Convert an ultrasound image to a graph representation.
        Uses edge detection and Delaunay triangulation to create a graph.
        """
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize image if needed
        img = cv2.resize(img, (224, 224))
        
        # Apply preprocessing
        # Enhance image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Edge detection using Canny
        edges = canny(img, sigma=1, low_threshold=10, high_threshold=50)
        
        # Find keypoints based on edges
        keypoints = np.argwhere(edges > 0)
        
        # If too many keypoints, sample a subset
        if len(keypoints) > 300:
            indices = np.random.choice(len(keypoints), 300, replace=False)
            keypoints = keypoints[indices]
        
        # If too few keypoints, add some random points
        if len(keypoints) < 30:
            # Add points from Sobel edge detection as backup
            sobel_edges = sobel(img) > threshold
            extra_points = np.argwhere(sobel_edges)
            if len(extra_points) > 0:
                if len(extra_points) > 50:
                    indices = np.random.choice(len(extra_points), 50, replace=False)
                    extra_points = extra_points[indices]
                keypoints = np.vstack([keypoints, extra_points])
        
        # If still too few keypoints, add random points
        if len(keypoints) < 30:
            random_points = np.random.randint(0, 224, size=(50, 2))
            keypoints = np.vstack([keypoints, random_points])
        
        # Extract node features
        node_features = []
        for point in keypoints:
            y, x = point
            
            # Get pixel intensity at point
            intensity = img[y, x]
            
            # Get local gradient
            if 0 < y < 223 and 0 < x < 223:
                dx = int(img[y, x+1]) - int(img[y, x-1])
                dy = int(img[y+1, x]) - int(img[y-1, x])
            else:
                dx, dy = 0, 0
            
            # Create feature vector for this node
            node_feat = [
                intensity / 255.0,  # Normalize intensity
                dx / 255.0,         # Normalized gradient in x direction
                dy / 255.0,         # Normalized gradient in y direction
                x / 224.0,          # Normalized x position
                y / 224.0           # Normalized y position
            ]
            node_features.append(node_feat)
        
        # Convert to numpy array
        node_features = np.array(node_features, dtype=np.float32)
        
        # Create edges using Delaunay triangulation
        if len(keypoints) > 3:  # Delaunay requires at least 4 points
            try:
                # Create Delaunay triangulation
                tri = Delaunay(keypoints)
                
                # Extract edges from triangulation
                edges = set()
                for simplex in tri.simplices:
                    edges.add((simplex[0], simplex[1]))
                    edges.add((simplex[1], simplex[2]))
                    edges.add((simplex[2], simplex[0]))
                    
                edge_index = np.array(list(edges), dtype=np.int64).T
                
                # Add reverse edges for undirected graph
                edge_index_reverse = edge_index[[1, 0], :]
                edge_index = np.concatenate([edge_index, edge_index_reverse], axis=1)
                
                # Calculate edge features based on distance between nodes
                edge_features = []
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[0, i], edge_index[1, i]
                    src_point = keypoints[src]
                    dst_point = keypoints[dst]
                    
                    # Euclidean distance
                    dist = np.sqrt(np.sum((src_point - dst_point) ** 2))
                    
                    # Intensity difference
                    int_diff = abs(img[src_point[0], src_point[1]] - img[dst_point[0], dst_point[1]]) / 255.0
                    
                    edge_features.append([dist / 224.0, int_diff])
                
                edge_features = np.array(edge_features, dtype=np.float32)
                
            except Exception as e:
                # Fallback if Delaunay fails
                print(f"Delaunay failed: {e}. Using kNN graph instead.")
                G = nx.Graph()
                for i, point in enumerate(keypoints):
                    G.add_node(i, pos=point)
                
                # Connect k nearest neighbors
                for i, point1 in enumerate(keypoints):
                    distances = []
                    for j, point2 in enumerate(keypoints):
                        if i != j:
                            dist = np.sqrt(np.sum((point1 - point2) ** 2))
                            distances.append((j, dist))
                    
                    # Sort by distance and connect to k nearest
                    distances.sort(key=lambda x: x[1])
                    for j, dist in distances[:5]:  # k=5
                        G.add_edge(i, j, weight=dist)
                
                # Extract edge index
                edge_list = list(G.edges())
                if not edge_list:
                    # Create a fully connected graph as a last resort
                    edge_list = [(i, j) for i in range(len(keypoints)) for j in range(len(keypoints)) if i != j]
                
                edge_index = np.array(edge_list, dtype=np.int64).T
                
                # Calculate edge features
                edge_features = []
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[0, i], edge_index[1, i]
                    src_point = keypoints[src]
                    dst_point = keypoints[dst]
                    
                    # Euclidean distance
                    dist = np.sqrt(np.sum((src_point - dst_point) ** 2))
                    
                    # Intensity difference
                    int_diff = abs(img[src_point[0], src_point[1]] - img[dst_point[0], dst_point[1]]) / 255.0
                    
                    edge_features.append([dist / 224.0, int_diff])
                
                edge_features = np.array(edge_features, dtype=np.float32)
        else:
            # Create a simple chain graph if too few points
            edge_index = np.array([[i, i+1] for i in range(len(keypoints)-1)], dtype=np.int64).T
            if edge_index.size == 0:  # Handle case with only one node
                edge_index = np.zeros((2, 0), dtype=np.int64)
                edge_features = np.zeros((0, 2), dtype=np.float32)
            else:
                # Calculate edge features
                edge_features = []
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[0, i], edge_index[1, i]
                    src_point = keypoints[src]
                    dst_point = keypoints[dst]
                    
                    # Euclidean distance
                    dist = np.sqrt(np.sum((src_point - dst_point) ** 2))
                    
                    # Intensity difference
                    int_diff = abs(img[src_point[0], src_point[1]] - img[dst_point[0], dst_point[1]]) / 255.0
                    
                    edge_features.append([dist / 224.0, int_diff])
                
                edge_features = np.array(edge_features, dtype=np.float32)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float)
        )
        
        # Add positional information (optional)
        data.pos = torch.tensor(keypoints, dtype=torch.float) / 224.0
        
        return data
    
    def load_data(self):
        """Load and convert images to graphs"""
        print("Loading and converting images to graphs...")
        train_graphs = []
        train_labels = []
        
        # Process training data
        for class_name in ['infected', 'notinfected']:
            class_dir = os.path.join(self.data_dir, class_name)
            label = 1 if class_name == 'infected' else 0
            
            print(f"Processing {class_name} class...")
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        graph = self.convert_image_to_graph(img_path)
                        train_graphs.append(graph)
                        train_labels.append(label)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        # Split training data into train and validation
        train_idx, val_idx = train_test_split(
            range(len(train_graphs)), 
            test_size=0.2, 
            stratify=train_labels,
            random_state=42
        )
        
        # Create train and validation datasets
        train_dataset = [train_graphs[i] for i in train_idx]
        val_dataset = [train_graphs[i] for i in val_idx]
        
        # Add labels
        for i, idx in enumerate(train_idx):
            train_dataset[i].y = torch.tensor([train_labels[idx]], dtype=torch.long)
        
        for i, idx in enumerate(val_idx):
            val_dataset[i].y = torch.tensor([train_labels[idx]], dtype=torch.long)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        print(f"Dataset created with {len(train_dataset)} training and {len(val_dataset)} validation samples")
        
        return train_loader, val_loader, train_dataset, val_dataset

class PCOSGraphNN(torch.nn.Module):
    """Graph Neural Network for PCOS Detection"""
    
    def __init__(self, num_node_features, num_edge_features, hidden_channels=64):
        super(PCOSGraphNN, self).__init__()
        
        # GNN layers
        self.conv1 = gnn.GCNConv(num_node_features, hidden_channels)
        self.conv2 = gnn.GATConv(hidden_channels, hidden_channels, heads=4, dropout=0.1)
        self.conv3 = gnn.GraphConv(hidden_channels * 4, hidden_channels)
        
        # Edge feature integration layer
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_edge_features, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Graph pooling layers
        self.pool1 = gnn.global_mean_pool
        self.pool2 = gnn.global_max_pool
        
        # Final classification layers
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 32)
        self.fc3 = nn.Linear(32, 2)  # Binary classification
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Edge weight computation using edge features
        if edge_attr is not None:
            edge_weight = self.edge_mlp(edge_attr).squeeze()
        else:
            edge_weight = None
        
        # First graph convolution layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second graph attention layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third graph convolution layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Global pooling (graph-level representation)
        x1 = self.pool1(x, batch)  # Mean pooling
        x2 = self.pool2(x, batch)  # Max pooling
        
        # Concatenate different pooling results
        x = torch.cat([x1, x2], dim=1)
        
        # Final classification MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class MetricsMonitor:
    """Monitor training metrics and save to files"""
    
    def __init__(self, log_dir='/app/results/metrics'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_file = os.path.join(log_dir, 'gnn_pytorch_metrics.csv')
        self.metrics_data = []
        
    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, val_f1, val_kappa, val_auc):
        """Log metrics to CSV file"""
        metrics = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_kappa': val_kappa,
            'val_auc': val_auc
        }
        
        # Save to CSV
        df = pd.DataFrame([metrics])
        if os.path.exists(self.metrics_file):
            df.to_csv(self.metrics_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.metrics_file, index=False)
        
        # Print progress
        print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
              f"val_f1={val_f1:.4f}, val_kappa={val_kappa:.4f}, val_auc={val_auc:.4f}")

class GNNTrainer:
    """PyTorch Geometric GNN trainer for PCOS detection"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.history = None
        self.data_processor = GraphDataProcessor()
        self.metrics_monitor = MetricsMonitor()
        
    def train_model(self, model, train_loader, val_loader, num_epochs=50, lr=0.001, weight_decay=5e-4):
        """Train the GNN model"""
        print("Starting model training...")
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_kappa': [],
            'val_auc': []
        }
        
        # Early stopping
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = criterion(output, data.y.view(-1))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Compute training metrics
                train_loss += loss.item() * data.num_graphs
                pred = output.argmax(dim=1)
                train_correct += (pred == data.y.view(-1)).sum().item()
                train_total += data.num_graphs
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_targets = []
            val_probs = []
            
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    
                    # Forward pass
                    output = model(data)
                    loss = criterion(output, data.y.view(-1))
                    
                    # Compute validation metrics
                    val_loss += loss.item() * data.num_graphs
                    pred = output.argmax(dim=1)
                    val_correct += (pred == data.y.view(-1)).sum().item()
                    val_total += data.num_graphs
                    
                    # Save predictions and targets for F1, Kappa, AUC
                    val_preds.extend(pred.cpu().numpy())
                    val_targets.extend(data.y.view(-1).cpu().numpy())
                    val_probs.extend(F.softmax(output, dim=1)[:, 1].cpu().numpy())
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Calculate other metrics
            val_f1 = f1_score(val_targets, val_preds)
            val_kappa = cohen_kappa_score(val_targets, val_preds)
            val_auc = roc_auc_score(val_targets, val_probs)
            
            # Update learning rate scheduler
            scheduler.step(val_acc)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_kappa'].append(val_kappa)
            history['val_auc'].append(val_auc)
            
            # Log metrics
            self.metrics_monitor.log_metrics(
                epoch, train_loss, train_acc, val_loss, val_acc, val_f1, val_kappa, val_auc
            )
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                
                # Save best model
                os.makedirs('/app/models/gnn_pytorch', exist_ok=True)
                torch.save(best_model_state, '/app/models/gnn_pytorch/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return model, history
    
    def evaluate_model(self, model, data_loader):
        """Evaluate the trained model"""
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                # Compute metrics
                correct += (pred == data.y.view(-1)).sum().item()
                total += data.num_graphs
                
                # Save predictions and targets
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(data.y.view(-1).cpu().numpy())
                all_probs.extend(F.softmax(output, dim=1)[:, 1].cpu().numpy())
        
        # Calculate metrics
        accuracy = correct / total
        f1 = f1_score(all_targets, all_preds)
        kappa = cohen_kappa_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'kappa': kappa,
            'auc': auc,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
        
        return metrics
    
    def generate_visualizations(self, history, test_metrics):
        """Generate training visualizations"""
        os.makedirs('/app/results/plots', exist_ok=True)
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training and validation loss
        axes[0, 0].plot(history['train_loss'], label='Training Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot training and validation accuracy
        axes[0, 1].plot(history['train_acc'], label='Training Accuracy')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot validation F1 score and Kappa
        axes[1, 0].plot(history['val_f1'], label='F1 Score')
        axes[1, 0].plot(history['val_kappa'], label='Kappa')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Validation F1 Score and Kappa')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot validation AUC
        axes[1, 1].plot(history['val_auc'], label='AUC-ROC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('Validation AUC-ROC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('/app/results/plots/gnn_pytorch_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(test_metrics['targets'], test_metrics['probabilities'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {test_metrics["auc"]:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('PCOS Detection ROC Curve (Validation Set)')
        plt.legend()
        plt.grid(True)
        plt.savefig('/app/results/plots/gnn_pytorch_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot confusion matrix
        cm = confusion_matrix(test_metrics['targets'], test_metrics['predictions'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('GNN PyTorch Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('/app/results/plots/gnn_pytorch_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ GNN PyTorch visualizations saved")
    
    def train(self):
        """Main training function"""
        print("🚀 Starting PyTorch Geometric GNN training...")
        
        # Load data
        train_loader, val_loader, train_dataset, val_dataset = self.data_processor.load_data()
        
        # Get number of features
        sample_data = train_dataset[0]
        num_node_features = sample_data.x.shape[1]
        num_edge_features = sample_data.edge_attr.shape[1] if sample_data.edge_attr is not None else 0
        
        print(f"Node features: {num_node_features}, Edge features: {num_edge_features}")
        
        # Initialize model
        self.model = PCOSGraphNN(num_node_features, num_edge_features).to(device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Train model
        self.model, self.history = self.train_model(
            self.model, 
            train_loader, 
            val_loader, 
            num_epochs=50,
            lr=0.001,
            weight_decay=5e-4
        )
        
        # Evaluate on validation set
        test_metrics = self.evaluate_model(self.model, val_loader)
        
        # Print final metrics
        print("\n🎉 Final Validation Metrics:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"F1 Score: {test_metrics['f1_score']:.4f}")
        print(f"Kappa: {test_metrics['kappa']:.4f}")
        print(f"AUC-ROC: {test_metrics['auc']:.4f}")
        
        # Generate visualizations
        self.generate_visualizations(self.history, test_metrics)
        
        # Save final model
        torch.save(self.model.state_dict(), '/app/models/gnn_pytorch/final_model.pth')
        print("✅ Final model saved to /app/models/gnn_pytorch/final_model.pth")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Score', 'Kappa', 'AUC-ROC'],
            'Validation': [test_metrics['accuracy'], test_metrics['f1_score'], test_metrics['kappa'], test_metrics['auc']]
        })
        
        metrics_df.to_csv('/app/results/metrics/gnn_pytorch_final_metrics.csv', index=False)
        print("✅ Final metrics saved to /app/results/metrics/gnn_pytorch_final_metrics.csv")
        
        return self.history, test_metrics

def main():
    """Main execution function"""
    print("🎯 PyTorch Geometric Graph Neural Network PCOS Detection Training")
    print("=" * 70)
    
    # Initialize trainer
    trainer = GNNTrainer()
    
    # Train model
    history, test_metrics = trainer.train()
    
    print("\n🎉 PyTorch Geometric GNN Training completed!")
    print(f"📊 Final Model Accuracy: {test_metrics['accuracy']:.4f}")
    print("\n📁 Results saved in:")
    print("   - /app/results/plots/")
    print("   - /app/results/metrics/")
    print("   - /app/models/gnn_pytorch/")

if __name__ == "__main__":
    main() 