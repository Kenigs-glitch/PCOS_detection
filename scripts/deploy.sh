#!/bin/bash

SERVER="bober@192.168.100.11"
PROJECT_DIR="/home/bober/PCOS_detection"

echo "🚀 Deploying to GPU server..."

# Create project directory on server
ssh $SERVER "mkdir -p $PROJECT_DIR"

# Copy project files
rsync -avz --exclude='.git' --exclude='__pycache__' . $SERVER:$PROJECT_DIR/

# Setup and run on server
ssh $SERVER << EOF
    cd $PROJECT_DIR
    
    # Build container
    docker-compose build
    
    # Download data
    docker-compose run --rm pcos-dev python scripts/download_data.py
    
    # Train models
    docker-compose run --rm pcos-dev python scripts/train_models.py
    
    # Compare models
    docker-compose run --rm pcos-dev python scripts/compare_models.py
    
    echo "✅ Training and comparison completed!"
EOF

echo "✅ Deployment completed!" 