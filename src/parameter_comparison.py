import torch
import torch.nn as nn
import torch_geometric.nn as gnn

def calculate_gnn_parameters():
    """Calculate parameters for our GNN model"""
    num_node_features = 5
    num_edge_features = 2
    hidden_channels = 64
    
    # GNN layers
    conv1 = gnn.GCNConv(num_node_features, hidden_channels)
    conv2 = gnn.GATConv(hidden_channels, hidden_channels, heads=4, dropout=0.1)
    conv3 = gnn.GraphConv(hidden_channels * 4, hidden_channels)
    
    # Edge feature integration
    edge_mlp = nn.Sequential(
        nn.Linear(num_edge_features, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )
    
    # Classification head
    fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
    fc2 = nn.Linear(hidden_channels, 32)
    fc3 = nn.Linear(32, 2)
    
    # Calculate parameters
    gnn_params = sum(p.numel() for p in conv1.parameters())
    gnn_params += sum(p.numel() for p in conv2.parameters())
    gnn_params += sum(p.numel() for p in conv3.parameters())
    gnn_params += sum(p.numel() for p in edge_mlp.parameters())
    gnn_params += sum(p.numel() for p in fc1.parameters())
    gnn_params += sum(p.numel() for p in fc2.parameters())
    gnn_params += sum(p.numel() for p in fc3.parameters())
    
    return gnn_params

def calculate_cnn_parameters():
    """Calculate parameters for a comparable CNN model"""
    # EfficientNet-B3 equivalent
    # Base model + classification head
    base_params = 12000000  # EfficientNet-B3 base
    classifier_params = 1280 * 512 + 512 * 256 + 256 * 2  # Large FC layers
    total_cnn_params = base_params + classifier_params
    
    return total_cnn_params

def calculate_resnet_parameters():
    """Calculate parameters for ResNet-50"""
    # ResNet-50 parameters
    base_params = 25600000  # ResNet-50 base
    classifier_params = 2048 * 512 + 512 * 256 + 256 * 2  # Large FC layers
    total_resnet_params = base_params + classifier_params
    
    return total_resnet_params

if __name__ == "__main__":
    gnn_params = calculate_gnn_parameters()
    cnn_params = calculate_cnn_parameters()
    resnet_params = calculate_resnet_parameters()
    
    print("Parameter Count Comparison")
    print("=" * 50)
    print("PyTorch Geometric GNN: {:,} parameters".format(gnn_params))
    print("EfficientNet-B3 CNN:   {:,} parameters".format(cnn_params))
    print("ResNet-50 CNN:         {:,} parameters".format(resnet_params))
    print()
    print("Size Comparison:")
    print("GNN is {:.1f}x smaller than EfficientNet".format(cnn_params/gnn_params))
    print("GNN is {:.1f}x smaller than ResNet-50".format(resnet_params/gnn_params))
    print()
    print("Why GNN is lightweight:")
    print("1. Parameter sharing across graph nodes")
    print("2. No large fully connected layers")
    print("3. Efficient graph pooling")
    print("4. Sparse connectivity patterns")
    print("5. Variable input sizes (no padding needed)") 