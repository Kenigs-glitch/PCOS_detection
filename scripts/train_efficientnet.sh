#!/bin/bash

echo "🎯 Starting EfficientNet-B3 PCOS Detection Training"
echo "=================================================="

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
echo "🚀 Starting EfficientNet-B3 training..."
echo "📊 Monitor progress at: /app/results/metrics/training_metrics.csv"
echo "📈 Real-time plots at: /app/results/metrics/training_progress.png"
echo ""

docker compose exec pcos-dev python src/train_efficientnet.py

echo ""
echo "✅ Training completed!"
echo "📁 Results available in:"
echo "   - /app/results/plots/"
echo "   - /app/results/metrics/"
echo "   - /app/models/efficientnet_b3/"
echo ""
echo "🔍 Run analysis notebook:"
echo "   docker compose exec pcos-dev jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''" 