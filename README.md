# PCOS Detection - Model Training & Comparison

A **Proof of Concept** project for training and comparing **EfficientNet-V1-B3** and **ResNet-50** models for PCOS detection using ultrasound images from the Anagha Choudhari dataset.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- Kaggle API credentials

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Kenigs-glitch/PCOS_detection.git
cd PCOS_detection
```

2. **Setup Kaggle credentials**
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. **Build and start the container**
```bash
docker-compose up -d
```

4. **Download the dataset**
```bash
docker-compose exec pcos-dev python scripts/download_data.py
```

5. **Train the models**
```bash
docker-compose exec pcos-dev python scripts/train_models.py
```

6. **Compare model performance**
```bash
docker-compose exec pcos-dev python scripts/compare_models.py
```

## 📁 Project Structure

```
PCOS_detection/
├── docker/
│   ├── Dockerfile              # Container configuration
│   └── requirements.txt        # Python dependencies
├── src/
│   ├── data_loader.py          # Dataset loading and preprocessing
│   ├── models.py               # Model definitions (EfficientNet-B3, ResNet-50)
│   ├── train.py                # Training pipeline
│   └── compare.py              # Model evaluation and comparison
├── notebooks/
│   ├── data_exploration.ipynb  # Dataset analysis
│   └── training_comparison.ipynb # Training visualization
├── scripts/
│   ├── download_data.py        # Dataset download script
│   ├── train_models.py         # Training execution script
│   ├── compare_models.py       # Comparison execution script
│   └── deploy.sh               # GPU server deployment
├── data/                       # Dataset storage
├── models/                     # Trained model storage
├── results/                    # Results and visualizations
├── config.yaml                 # Configuration file
├── docker-compose.yml          # Container orchestration
└── README.md                   # This file
```

## 🏗️ Architecture

### Models
- **EfficientNet-B3**: Lightweight, efficient architecture
- **ResNet-50**: Deep residual network with skip connections

### Training Strategy
- Transfer learning with ImageNet pre-trained weights
- Frozen base layers for feature extraction
- Custom classification head with dropout regularization
- Early stopping and learning rate reduction

### Data Pipeline
- Image resizing to 300x300 pixels
- Data augmentation (built into TensorFlow)
- Train/validation split (80/20)
- Batch processing with prefetching

## 📊 Expected Results

### Training Time (RTX 3090)
- **EfficientNet-B3**: ~45 minutes
- **ResNet-50**: ~60 minutes
- **Total PoC**: ~2 hours

### Performance Metrics
- Accuracy comparison
- Precision, Recall, F1-Score
- Confusion matrices
- Training curves

## 🖥️ GPU Server Deployment

### Automated Deployment
```bash
# Deploy to GPU server (bober@192.168.100.11)
./scripts/deploy.sh
```

### Manual Deployment
```bash
# Copy files to server
rsync -avz . bober@192.168.100.11:/home/bober/PCOS_detection/

# Execute training on server
ssh bober@192.168.100.11 "cd /home/bober/PCOS_detection && docker-compose run --rm pcos-dev python scripts/train_models.py"
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Image size and batch size
- Learning rates and epochs
- Model-specific parameters
- Training callbacks

## 📈 Monitoring

### Jupyter Lab Access
```bash
# Access Jupyter Lab at http://localhost:8888
docker-compose up -d
```

### Training Progress
- Real-time training metrics
- Early stopping with patience
- Model checkpointing
- Learning rate scheduling

## 📋 Deliverables

- ✅ Two trained models (EfficientNet-B3, ResNet-50)
- ✅ Performance comparison report (`/results/comparison_report.md`)
- ✅ Visualization plots (`/results/plots/`)
- ✅ Model accuracy metrics
- ✅ Docker deployment setup
- ✅ Jupyter notebooks for analysis

## 🛠️ Development

### Local Development
```bash
# Start development environment
docker-compose up -d

# Access container
docker-compose exec pcos-dev bash

# Run individual components
python src/train.py efficientnet_b3
python src/compare.py
```

### Testing
```bash
# Test data loading
docker-compose exec pcos-dev python -c "from src.data_loader import PCOSDataLoader; PCOSDataLoader().load_dataset()"

# Test model creation
docker-compose exec pcos-dev python -c "from src.models import ModelFactory; ModelFactory.create_efficientnet_b3()"
```

## 📝 Notes

- **Minimalistic approach**: Focus on rapid PoC development
- **Docker environment**: Ensures consistency across platforms
- **GPU optimization**: CUDA support for faster training
- **Modular design**: Easy to extend and modify

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

---

**Built with ❤️ for PCOS detection research** 