#!/bin/bash

echo "🎯 Starting PyTorch Geometric Graph Neural Network PCOS Detection Training"
echo "========================================================================"

# Check if container is running
if ! docker compose ps | grep -q "pcos_poc"; then
    echo "🚀 Starting Docker container..."
    docker compose up -d
    sleep 10
fi

# Check if data exists
if [ ! -d "/app/data/raw/train" ]; then
    echo "📥 Downloading dataset..."
    docker compose exec pcos-dev python scripts/download_data.py
fi

# Start training
echo "🚀 Starting PyTorch Geometric GNN training..."
echo "📊 Monitor progress at: /app/results/metrics/gnn_pytorch_metrics.csv"
echo "📈 Real-time plots at: /app/results/plots/gnn_pytorch_training_curves.png"
echo ""

docker compose exec pcos-dev python src/train_gnn_pytorch.py

echo ""
echo "✅ PyTorch Geometric GNN Training completed!"
echo "📁 Results available in:"
echo "   - /app/results/plots/"
echo "   - /app/results/metrics/"
echo "   - /app/models/gnn_pytorch/"
echo ""
echo "🔍 Run analysis notebook:"
echo "   docker compose exec pcos-dev jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''" 