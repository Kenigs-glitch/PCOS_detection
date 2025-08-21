import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import yaml

class PCOSDataLoader:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_dataset(self):
        """Load and split PCOS dataset"""
        data_dir = "/app/data/raw/train"
        
        # Load dataset
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            image_size=self.config['data']['image_size'],
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            seed=42
        )
        
        # Split into train/validation
        val_size = int(len(dataset) * self.config['data']['validation_split'])
        train_ds = dataset.skip(val_size)
        val_ds = dataset.take(val_size)
        
        # Optimize performance
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def get_class_names(self):
        """Get class names"""
        return ['Normal', 'PCOS'] 