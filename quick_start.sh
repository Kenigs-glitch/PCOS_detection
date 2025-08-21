#!/bin/bash

echo "🚀 PCOS Detection - Quick Start"
echo "================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if kaggle.json exists
if [ ! -f "kaggle.json" ]; then
    echo "❌ kaggle.json not found. Please ensure your Kaggle credentials are in place."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Build and start container
echo "🔨 Building and starting Docker container..."
docker-compose up -d --build

# Wait for container to be ready
echo "⏳ Waiting for container to be ready..."
sleep 10

# Test setup
echo "🧪 Testing setup..."
docker-compose exec pcos-dev python scripts/test_setup.py

if [ $? -ne 0 ]; then
    echo "❌ Setup test failed. Please check the container logs."
    exit 1
fi

echo "✅ Setup test passed"

# Download dataset
echo "📥 Downloading PCOS dataset..."
docker-compose exec pcos-dev python scripts/download_data.py

if [ $? -ne 0 ]; then
    echo "❌ Dataset download failed. Please check your Kaggle credentials."
    exit 1
fi

echo "✅ Dataset downloaded"

# Ask user if they want to proceed with training
echo ""
echo "🎯 Ready to start training!"
echo "This will train both EfficientNet-B3 and ResNet-50 models."
echo "Expected time: ~2 hours on GPU"
echo ""
read -p "Do you want to proceed with training? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting model training..."
    docker-compose exec pcos-dev python scripts/train_models.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Training completed successfully!"
        
        echo "📊 Running model comparison..."
        docker-compose exec pcos-dev python scripts/compare_models.py
        
        echo ""
        echo "🎉 All done! Check the results in the /results directory."
        echo "📈 Access Jupyter Lab at: http://localhost:8888"
    else
        echo "❌ Training failed. Check the logs for details."
        exit 1
    fi
else
    echo "⏸️  Training skipped. You can run it later with:"
    echo "   docker-compose exec pcos-dev python scripts/train_models.py"
fi

echo ""
echo "📋 Next steps:"
echo "1. Access Jupyter Lab: http://localhost:8888"
echo "2. View results: ./results/"
echo "3. Stop container: docker-compose down"
echo ""
echo "Happy training! 🎯" 