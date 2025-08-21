import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
from torch_geometric.data import Data
import cv2

class GNNInterpretability:
    """GNN-specific interpretability methods to replace Grad-CAM"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def generate_node_importance_heatmap(self, graph, pred_class=None, save_path=None):
        """Generate node importance visualization (GNN equivalent of Grad-CAM)"""
        graph = graph.to(self.device)
        graph.requires_grad_(True)
        
        # Forward pass
        output = self.model(graph)
        
        # Get prediction class if not provided
        if pred_class is None:
            pred_class = output.argmax(dim=1).item()
        
        # Backward pass to get gradients
        output[0, pred_class].backward()
        
        # Calculate node importance
        node_importance = torch.mean(torch.abs(graph.x.grad), dim=1)
        node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-8)
        
        # Create heatmap using node positions
        heatmap = np.zeros((224, 224))
        
        if hasattr(graph, 'pos') and graph.pos is not None:
            pos = graph.pos.cpu().numpy()
            for i, (x, y) in enumerate(pos):
                x_coord = int(x * 224)
                y_coord = int(y * 224)
                if 0 <= x_coord < 224 and 0 <= y_coord < 224:
                    heatmap[y_coord, x_coord] = node_importance[i].item()
        
        # Apply Gaussian blur for smoother visualization
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original graph with node importance
        if hasattr(graph, 'pos') and graph.pos is not None:
            scatter = ax1.scatter(graph.pos[:, 0].cpu().numpy() * 224, 
                                graph.pos[:, 1].cpu().numpy() * 224,
                                c=node_importance.cpu().numpy(), 
                                cmap='hot', s=50, alpha=0.7)
            ax1.set_xlim(0, 224)
            ax1.set_ylim(0, 224)
            ax1.invert_yaxis()
        else:
            ax1.scatter(range(len(node_importance)), range(len(node_importance)),
                       c=node_importance.cpu().numpy(), cmap='hot', s=50, alpha=0.7)
        
        ax1.set_title(f'Node Importance\nPrediction Class: {pred_class}')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        
        # Heatmap visualization
        im = ax2.imshow(heatmap, cmap='hot', alpha=0.8)
        ax2.set_title('Node Importance Heatmap\n(GNN equivalent of Grad-CAM)')
        ax2.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Node importance heatmap saved to: {save_path}")
        
        plt.show()
        return heatmap, node_importance
    
    def analyze_feature_importance(self, graph, pred_class=None, save_path=None):
        """Analyze which node features are most important"""
        graph = graph.to(self.device)
        graph.requires_grad_(True)
        
        # Forward pass
        output = self.model(graph)
        
        if pred_class is None:
            pred_class = output.argmax(dim=1).item()
        
        # Backward pass
        output[0, pred_class].backward()
        
        # Get gradients for each feature
        feature_gradients = graph.x.grad  # Shape: [num_nodes, num_features]
        
        # Calculate feature importance
        feature_importance = torch.mean(torch.abs(feature_gradients), dim=0)
        feature_names = ['Intensity', 'Gradient X', 'Gradient Y', 'Position X', 'Position Y']
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        bars = plt.bar(feature_names, feature_importance.cpu().numpy())
        plt.title('Node Feature Importance Analysis')
        plt.ylabel('Average Gradient Magnitude')
        plt.xlabel('Node Features')
        
        # Add value labels on bars
        for bar, value in zip(bars, feature_importance.cpu().numpy()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance analysis saved to: {save_path}")
        
        plt.show()
        return feature_importance.cpu().numpy()
    
    def visualize_edge_importance(self, graph, save_path=None):
        """Visualize which edges (connections) are most important"""
        graph = graph.to(self.device)
        graph.requires_grad_(True)
        
        # Forward pass
        output = self.model(graph)
        pred_class = output.argmax(dim=1).item()
        
        # Backward pass
        output[0, pred_class].backward()
        
        # Calculate edge importance based on edge features
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            edge_importance = torch.mean(torch.abs(graph.edge_attr.grad), dim=1)
        else:
            # Fallback: use edge index to calculate importance
            edge_importance = torch.ones(graph.edge_index.size(1))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Edge importance histogram
        ax1.hist(edge_importance.cpu().numpy(), bins=20, alpha=0.7, color='skyblue')
        ax1.set_title('Edge Importance Distribution')
        ax1.set_xlabel('Edge Importance Score')
        ax1.set_ylabel('Number of Edges')
        
        # Top important edges
        top_k = min(10, len(edge_importance))
        top_indices = torch.topk(edge_importance, top_k).indices
        
        top_importance = edge_importance[top_indices].cpu().numpy()
        ax2.bar(range(top_k), top_importance, color='red', alpha=0.7)
        ax2.set_title(f'Top {top_k} Most Important Edges')
        ax2.set_xlabel('Edge Index')
        ax2.set_ylabel('Importance Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Edge importance analysis saved to: {save_path}")
        
        plt.show()
        return edge_importance.cpu().numpy()
    
    def generate_confidence_analysis(self, data_loader, save_path=None):
        """Analyze prediction confidence distribution"""
        self.model.eval()
        confidences = []
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = F.softmax(output, dim=1)
                
                # Get confidence (max probability)
                confidence = torch.max(probs, dim=1)[0]
                pred = output.argmax(dim=1)
                
                confidences.extend(confidence.cpu().numpy())
                predictions.extend(pred.cpu().numpy())
                targets.extend(data.y.view(-1).cpu().numpy())
        
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Confidence distribution
        ax1.hist(confidences, bins=20, alpha=0.7, color='green')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Number of Predictions')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax1.legend()
        
        # Confidence vs accuracy
        correct = (predictions == targets)
        ax2.scatter(confidences, correct, alpha=0.5, color='blue')
        ax2.set_title('Confidence vs Correctness')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Correct Prediction (1=Yes, 0=No)')
        
        # Confidence by class
        for class_idx in [0, 1]:
            class_mask = targets == class_idx
            ax3.hist(confidences[class_mask], bins=15, alpha=0.7, 
                    label=f'Class {class_idx}', density=True)
        ax3.set_title('Confidence Distribution by Class')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Density')
        ax3.legend()
        
        # High confidence predictions
        high_conf_mask = confidences > 0.9
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = np.mean(correct[high_conf_mask])
            ax4.bar(['High Confidence (>0.9)', 'All Predictions'], 
                   [high_conf_accuracy, np.mean(correct)], 
                   color=['orange', 'blue'], alpha=0.7)
            ax4.set_title('Accuracy Comparison')
            ax4.set_ylabel('Accuracy')
            ax4.set_ylim(0, 1)
        else:
            ax4.text(0.5, 0.5, 'No high confidence predictions', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('High Confidence Analysis')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence analysis saved to: {save_path}")
        
        plt.show()
        
        return {
            'confidences': confidences,
            'predictions': predictions,
            'targets': targets,
            'mean_confidence': np.mean(confidences),
            'high_conf_accuracy': np.mean(correct[confidences > 0.9]) if np.sum(confidences > 0.9) > 0 else 0
        }
    
    def create_comprehensive_interpretability_report(self, data_loader, num_samples=5, save_dir='/app/results/plots/gnn_interpretability'):
        """Create comprehensive interpretability report with all visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("🔍 Generating comprehensive GNN interpretability report...")
        
        # Generate node importance heatmaps for sample graphs
        sample_count = 0
        for data in data_loader:
            if sample_count >= num_samples:
                break
            
            for i in range(data.num_graphs):
                if sample_count >= num_samples:
                    break
                
                # Extract single graph
                graph_idx = (data.batch == i).nonzero(as_tuple=True)[0]
                single_graph = Data(
                    x=data.x[graph_idx],
                    edge_index=data.edge_index[:, (data.batch[data.edge_index[0]] == i) & (data.batch[data.edge_index[1]] == i)],
                    edge_attr=data.edge_attr[(data.batch[data.edge_index[0]] == i) & (data.batch[data.edge_index[1]] == i)] if data.edge_attr is not None else None,
                    pos=data.pos[graph_idx] if hasattr(data, 'pos') else None,
                    y=data.y[i:i+1]
                )
                
                # Generate visualizations
                heatmap_path = os.path.join(save_dir, f'node_importance_sample_{sample_count}.png')
                self.generate_node_importance_heatmap(single_graph, save_path=heatmap_path)
                
                feature_path = os.path.join(save_dir, f'feature_importance_sample_{sample_count}.png')
                self.analyze_feature_importance(single_graph, save_path=feature_path)
                
                edge_path = os.path.join(save_dir, f'edge_importance_sample_{sample_count}.png')
                self.visualize_edge_importance(single_graph, save_path=edge_path)
                
                sample_count += 1
        
        # Generate confidence analysis
        confidence_path = os.path.join(save_dir, 'confidence_analysis.png')
        confidence_results = self.generate_confidence_analysis(data_loader, save_path=confidence_path)
        
        # Create summary report
        report_path = os.path.join(save_dir, 'interpretability_summary.txt')
        with open(report_path, 'w') as f:
            f.write("GNN Interpretability Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("Generated Visualizations:\n")
            f.write(f"- Node importance heatmaps: {num_samples} samples\n")
            f.write(f"- Feature importance analysis: {num_samples} samples\n")
            f.write(f"- Edge importance analysis: {num_samples} samples\n")
            f.write(f"- Confidence analysis: Complete dataset\n\n")
            f.write("Key Findings:\n")
            f.write(f"- Mean prediction confidence: {confidence_results['mean_confidence']:.3f}\n")
            f.write(f"- High confidence accuracy: {confidence_results['high_conf_accuracy']:.3f}\n")
            f.write(f"- Overall accuracy: {np.mean(confidence_results['predictions'] == confidence_results['targets']):.3f}\n\n")
            f.write("Note: These visualizations are GNN-specific and provide\n")
            f.write("better interpretability than traditional Grad-CAM for\n")
            f.write("graph-structured data like ultrasound images.\n")
        
        print(f"✅ Comprehensive interpretability report saved to: {save_dir}")
        return save_dir

def main():
    """Example usage of GNN interpretability"""
    print("GNN Interpretability Module")
    print("This module provides GNN-specific interpretability methods")
    print("to replace traditional Grad-CAM visualizations.")
    print()
    print("Available methods:")
    print("1. generate_node_importance_heatmap() - GNN equivalent of Grad-CAM")
    print("2. analyze_feature_importance() - Which features matter most")
    print("3. visualize_edge_importance() - Which connections are important")
    print("4. generate_confidence_analysis() - Prediction confidence analysis")
    print("5. create_comprehensive_interpretability_report() - All visualizations")

if __name__ == "__main__":
    main() 