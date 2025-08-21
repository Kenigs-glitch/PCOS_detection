import tensorflow as tf
import yaml
import os
from data_loader import PCOSDataLoader
from models import ModelFactory

class ModelTrainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def train_model(self, model_name):
        """Train a specific model"""
        print(f"🚀 Training {model_name}")
        
        # Load data
        data_loader = PCOSDataLoader()
        train_ds, val_ds = data_loader.load_dataset()
        
        # Create model
        if model_name == 'efficientnet_b3':
            model = ModelFactory.create_efficientnet_b3()
        elif model_name == 'resnet50':
            model = ModelFactory.create_resnet50()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['models'][model_name]['learning_rate']
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.config['training']['patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'/app/models/{model_name}/best_model.h5',
                save_best_only=self.config['training']['save_best_only']
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5
            )
        ]
        
        # Train model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config['models'][model_name]['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        os.makedirs(f'/app/models/{model_name}', exist_ok=True)
        model.save(f'/app/models/{model_name}/final_model.h5')
        
        print(f"✅ {model_name} training completed")
        return history, model

if __name__ == "__main__":
    import sys
    trainer = ModelTrainer()
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        trainer.train_model(model_name)
    else:
        # Train both models
        trainer.train_model('efficientnet_b3')
        trainer.train_model('resnet50') 