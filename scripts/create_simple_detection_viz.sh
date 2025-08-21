#!/bin/bash

echo "🎯 Creating Simple PCOS Detection Visualizations"
echo "================================================"
echo "For non-technical audience - quick and visual!"
echo ""

# Check if container is running
if ! docker compose ps | grep -q "pcos_poc"; then
    echo "🚀 Starting Docker container..."
    docker compose up -d
    sleep 10
fi

# Check if trained model exists
if [ ! -f "/app/models/gnn_pytorch/final_model.pth" ]; then
    echo "❌ No trained model found. Please train first:"
    echo "   docker compose exec pcos-dev python src/train_gnn_pytorch.py"
    exit 1
fi

echo "📊 Generating simple detection visualizations..."

# Create simple visualizations
docker compose exec pcos-dev python -c "
import sys
sys.path.append('/app/src')

from simple_visual_detection import create_simple_detection_visualization, create_client_presentation
from train_gnn_pytorch import GraphDataProcessor

# Load data
print('Loading data...')
data_processor = GraphDataProcessor()
train_loader, val_loader, train_dataset, val_dataset = data_processor.load_data()

# Create visualizations
print('Creating simple detection visualizations...')
save_dir = create_simple_detection_visualization(
    '/app/models/gnn_pytorch/final_model.pth', 
    val_loader
)

# Create client presentation
create_client_presentation(save_dir)

print(f'\\n✅ Simple visualizations created!')
print(f'📁 Results in: {save_dir}')
print('\\n📋 What was created:')
print('- 3 example detection visualizations')
print('- Detection process summary')
print('- Performance summary')
print('- Client presentation summary')
"

echo ""
echo "✅ Simple detection visualizations completed!"
echo ""
echo "📁 Results available in:"
echo "   - /app/results/plots/simple_detection/"
echo ""
echo "📋 Created for non-technical audience:"
echo "   1. Example detection visualizations (3 samples)"
echo "   2. Simple 4-step process explanation"
echo "   3. Performance summary chart"
echo "   4. Client presentation summary"
echo ""
echo "💡 These show how the AI detects PCOS visually"
echo "   without technical jargon - perfect for presentations!" 