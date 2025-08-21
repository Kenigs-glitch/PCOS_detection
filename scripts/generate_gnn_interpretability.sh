#!/bin/bash

echo "🔍 Generating GNN Interpretability Visualizations"
echo "=================================================="

# Check if container is running
if ! docker compose ps | grep -q "pcos_poc"; then
    echo "🚀 Starting Docker container..."
    docker compose up -d
    sleep 10
fi

# Check if trained model exists
if [ ! -f "/app/models/gnn_pytorch/final_model.pth" ]; then
    echo "❌ No trained GNN model found. Please train the model first:"
    echo "   docker compose exec pcos-dev python src/train_gnn_pytorch.py"
    exit 1
fi

echo "📊 Generating GNN-specific interpretability visualizations..."
echo "Note: These replace traditional Grad-CAM for graph neural networks"
echo ""

# Create interpretability visualizations
docker compose exec pcos-dev python -c "
import sys
sys.path.append('/app/src')

from gnn_interpretability import GNNInterpretability
import torch
from torch_geometric.data import DataLoader
from train_gnn_pytorch import GraphDataProcessor, PCOSGraphNN

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PCOSGraphNN(5, 2, 64).to(device)
model.load_state_dict(torch.load('/app/models/gnn_pytorch/final_model.pth', map_location=device))
model.eval()

# Load data
data_processor = GraphDataProcessor()
train_loader, val_loader, train_dataset, val_dataset = data_processor.load_data()

# Create interpretability analyzer
interpreter = GNNInterpretability(model, device)

# Generate comprehensive report
print('Generating interpretability visualizations...')
save_dir = interpreter.create_comprehensive_interpretability_report(val_loader, num_samples=5)

print(f'\\n✅ Interpretability report generated successfully!')
print(f'📁 Results saved in: {save_dir}')
print('\\n📋 Generated visualizations:')
print('- Node importance heatmaps (GNN equivalent of Grad-CAM)')
print('- Feature importance analysis')
print('- Edge importance analysis')
print('- Confidence analysis')
print('\\n💡 These visualizations provide better interpretability than')
print('   traditional Grad-CAM for graph-structured data.')
"

echo ""
echo "✅ GNN Interpretability visualizations completed!"
echo ""
echo "📁 Results available in:"
echo "   - /app/results/plots/gnn_interpretability/"
echo ""
echo "📋 What was generated:"
echo "   1. Node importance heatmaps (GNN equivalent of Grad-CAM)"
echo "   2. Feature importance analysis"
echo "   3. Edge importance analysis"
echo "   4. Confidence analysis"
echo "   5. Comprehensive interpretability report"
echo ""
echo "💡 These visualizations are specifically designed for GNNs and"
echo "   provide better interpretability than traditional Grad-CAM for"
echo "   graph-structured data like ultrasound images." 