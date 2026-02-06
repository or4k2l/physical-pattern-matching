"""Scalable data loaders for vision datasets."""

from typing import Optional, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class ScalableDataLoader:
    """
    Scalable data loader using TensorFlow Datasets.
    
    Features:
    - Handles multiple datasets (CIFAR-10, CIFAR-100, ImageNet, etc.)
    - Automatic batching and prefetching
    - Data augmentation
    - Caching for performance
    """
    
    def __init__(
        self,
        dataset_name: str = "cifar10",
        batch_size: int = 32,
        image_size: Tuple[int, int] = (32, 32),
        cache: bool = True,
        prefetch: bool = True,
        augment: bool = False
    ):
        """
        Initialize data loader.
        
        Args:
            dataset_name: Name of TFDS dataset
            batch_size: Batch size
            image_size: Target image size (height, width)
            cache: Whether to cache dataset
            prefetch: Whether to prefetch batches
            augment: Whether to apply data augmentation
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.cache = cache
        self.prefetch = prefetch
        self.augment = augment
        
    def preprocess(self, example, training: bool = False):
        """
        Preprocess a single example.
        
        Args:
            example: Dictionary with 'image' and 'label' keys
            training: Whether in training mode
            
        Returns:
            Preprocessed (image, label) tuple
        """
        image = example['image']
        label = example['label']
        
        # Resize if needed
        if image.shape[:2] != self.image_size:
            image = tf.image.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # Data augmentation for training
        if training and self.augment:
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            
            # Random brightness
            image = tf.image.random_brightness(image, 0.1)
            
            # Random contrast
            image = tf.image.random_contrast(image, 0.9, 1.1)
        
        return image, label
    
    def load_dataset(self, split: str):
        """
        Load dataset split.
        
        Args:
            split: Dataset split ('train', 'test', etc.)
            
        Returns:
            TensorFlow dataset
        """
        ds = tfds.load(
            self.dataset_name,
            split=split,
            as_supervised=False,
            shuffle_files=(split == 'train')
        )
        
        return ds
    
    def create_dataset(self, split: str, repeat: bool = False):
        """
        Create preprocessed and batched dataset.
        
        Args:
            split: Dataset split ('train' or 'test')
            repeat: Whether to repeat the dataset infinitely
            
        Returns:
            Batched TensorFlow dataset
        """
        ds = self.load_dataset(split)
        
        # Shuffle for training
        if split == 'train':
            ds = ds.shuffle(10000)
        
        # Preprocess
        training = (split == 'train')
        ds = ds.map(
            lambda x: self.preprocess(x, training=training),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Cache before batching for better performance
        if self.cache:
            ds = ds.cache()
        
        # Repeat if requested
        if repeat:
            ds = ds.repeat()
        
        # Batch
        ds = ds.batch(self.batch_size)
        
        # Prefetch
        if self.prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    def to_numpy_iterator(self, dataset):
        """
        Convert TensorFlow dataset to numpy iterator.
        
        Args:
            dataset: TensorFlow dataset
            
        Returns:
            Iterator yielding dict with 'image' and 'label' keys
        """
        for images, labels in dataset:
            yield {
                'image': images.numpy(),
                'label': labels.numpy()
            }
    
    def get_train_loader(self):
        """Get a fresh training data iterator."""
        train_ds = self.create_dataset('train', repeat=False)
        return self.to_numpy_iterator(train_ds)
    
    def get_test_loader(self):
        """Get a fresh validation/test data iterator."""
        test_ds = self.create_dataset('test', repeat=False)
        return self.to_numpy_iterator(test_ds)
