"""Noise library for robustness testing and augmentation."""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp


class NoiseLibrary:
    """
    Library of noise functions for robustness testing.
    
    Supports:
    - Gaussian noise
    - Salt and pepper noise
    - Fog/haze effects
    - Occlusion (random patches)
    - Mixup augmentation
    """
    
    def __init__(self, rng_key: Optional[jax.random.PRNGKey] = None):
        """
        Initialize noise library.
        
        Args:
            rng_key: JAX random key for reproducibility
        """
        self.rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(42)
    
    def gaussian_noise(
        self,
        images: jnp.ndarray,
        std: float = 0.1,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Add Gaussian noise to images.
        
        Args:
            images: Input images [batch, height, width, channels] or [batch, height, width]
            std: Standard deviation of noise
            rng_key: Random key (uses self.rng_key if None)
            
        Returns:
            Noisy images clipped to [0, 1]
        """
        if rng_key is None:
            self.rng_key, rng_key = jax.random.split(self.rng_key)
        
        noise = jax.random.normal(rng_key, images.shape) * std
        noisy = images + noise
        return jnp.clip(noisy, 0.0, 1.0)
    
    def salt_pepper_noise(
        self,
        images: jnp.ndarray,
        prob: float = 0.05,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Add salt and pepper noise to images.
        
        Args:
            images: Input images
            prob: Probability of noise for each pixel
            rng_key: Random key
            
        Returns:
            Noisy images
        """
        if rng_key is None:
            self.rng_key, rng_key = jax.random.split(self.rng_key)
        
        key1, key2 = jax.random.split(rng_key)
        
        # Salt (white pixels)
        salt_mask = jax.random.uniform(key1, images.shape) < prob / 2
        
        # Pepper (black pixels)
        pepper_mask = jax.random.uniform(key2, images.shape) < prob / 2
        
        noisy = jnp.where(salt_mask, 1.0, images)
        noisy = jnp.where(pepper_mask, 0.0, noisy)
        
        return noisy
    
    def fog_noise(
        self,
        images: jnp.ndarray,
        intensity: float = 0.3,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Add fog/haze effect to images.
        
        Args:
            images: Input images
            intensity: Fog intensity (0 = no fog, 1 = complete white)
            rng_key: Random key
            
        Returns:
            Foggy images
        """
        if rng_key is None:
            self.rng_key, rng_key = jax.random.split(self.rng_key)
        
        # Create fog layer (white with some variation)
        fog = jax.random.uniform(rng_key, images.shape) * 0.1 + 0.9
        
        # Blend with original image
        foggy = (1 - intensity) * images + intensity * fog
        return jnp.clip(foggy, 0.0, 1.0)
    
    def occlusion_noise(
        self,
        images: jnp.ndarray,
        patch_size: int = 8,
        num_patches: int = 3,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Add occlusion patches to images.
        
        Args:
            images: Input images [batch, height, width, channels] or [batch, height, width]
            patch_size: Size of occlusion patches
            num_patches: Number of patches per image
            rng_key: Random key
            
        Returns:
            Occluded images
        """
        if rng_key is None:
            self.rng_key, rng_key = jax.random.split(self.rng_key)
        
        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]
        
        occluded = images.copy()
        
        for _ in range(num_patches):
            # Random positions for patches
            rng_key, key1, key2 = jax.random.split(rng_key, 3)
            y_positions = jax.random.randint(key1, (batch_size,), 0, height - patch_size)
            x_positions = jax.random.randint(key2, (batch_size,), 0, width - patch_size)
            
            # Apply occlusion (set to black)
            for i in range(batch_size):
                y = int(y_positions[i])
                x = int(x_positions[i])
                occluded = occluded.at[i, y:y+patch_size, x:x+patch_size].set(0.0)
        
        return occluded
    
    def apply_noise(
        self,
        images: jnp.ndarray,
        noise_type: str,
        severity: float,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Apply specified noise type with given severity.
        
        Args:
            images: Input images
            noise_type: One of ['gaussian', 'salt_pepper', 'fog', 'occlusion']
            severity: Noise severity (0 = clean, higher = more severe)
            rng_key: Random key
            
        Returns:
            Noisy images
        """
        if severity == 0.0:
            return images
        
        if rng_key is None:
            self.rng_key, rng_key = jax.random.split(self.rng_key)
        
        if noise_type == "gaussian":
            return self.gaussian_noise(images, std=severity, rng_key=rng_key)
        elif noise_type == "salt_pepper":
            return self.salt_pepper_noise(images, prob=severity, rng_key=rng_key)
        elif noise_type == "fog":
            return self.fog_noise(images, intensity=severity, rng_key=rng_key)
        elif noise_type == "occlusion":
            # Scale num_patches based on severity
            num_patches = max(1, int(severity * 10))
            return self.occlusion_noise(images, patch_size=8, num_patches=num_patches, rng_key=rng_key)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    @staticmethod
    def mixup(
        images: jnp.ndarray,
        labels: jnp.ndarray,
        alpha: float = 1.0,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jax.random.PRNGKey]:
        """
        Apply mixup augmentation.
        
        Mixup creates virtual training examples by mixing pairs of examples and their labels.
        
        Args:
            images: Input images [batch, ...]
            labels: Input labels [batch]
            alpha: Mixup interpolation strength (higher = more mixing)
            rng_key: Random key
            
        Returns:
            Tuple of (mixed_images, labels1, labels2, lambda, new_rng_key)
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        
        batch_size = images.shape[0]
        
        # Sample mixing coefficient
        rng_key, lambda_key = jax.random.split(rng_key)
        lam = jax.random.beta(lambda_key, alpha, alpha)
        
        # Random shuffle indices
        rng_key, perm_key = jax.random.split(rng_key)
        indices = jax.random.permutation(perm_key, batch_size)
        
        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * images[indices]
        labels2 = labels[indices]
        
        return mixed_images, labels, labels2, lam, rng_key
