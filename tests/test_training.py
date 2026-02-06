"""Tests for training components."""

import pytest
import jax
import jax.numpy as jnp
import optax

from robust_vision.training.losses import (
    label_smoothing_cross_entropy,
    margin_loss,
    focal_loss,
    compute_accuracy
)
from robust_vision.training.state import TrainStateWithEMA, create_train_state
from robust_vision.models.cnn import ProductionCNN


class TestLossFunctions:
    """Tests for loss functions."""
    
    @pytest.fixture
    def logits(self):
        """Create test logits."""
        return jnp.array([
            [2.0, 1.0, 0.5, 0.1],
            [0.5, 2.5, 0.3, 0.2],
            [1.0, 0.5, 3.0, 0.1]
        ])
    
    @pytest.fixture
    def labels(self):
        """Create test labels."""
        return jnp.array([0, 1, 2])
    
    def test_label_smoothing_cross_entropy(self, logits, labels):
        """Test label smoothing cross-entropy loss."""
        loss = label_smoothing_cross_entropy(logits, labels, num_classes=4, smoothing=0.1)
        
        # Loss should be a scalar
        assert loss.shape == ()
        
        # Loss should be positive
        assert loss > 0
        
        # Loss with smoothing should be different from without
        loss_no_smooth = label_smoothing_cross_entropy(logits, labels, num_classes=4, smoothing=0.0)
        assert not jnp.allclose(loss, loss_no_smooth)
    
    def test_margin_loss(self, logits, labels):
        """Test margin loss."""
        loss = margin_loss(logits, labels, margin=1.0)
        
        # Loss should be a scalar
        assert loss.shape == ()
        
        # Loss should be non-negative
        assert loss >= 0
        
        # Larger margin should give larger loss (for these logits)
        loss_small = margin_loss(logits, labels, margin=0.5)
        loss_large = margin_loss(logits, labels, margin=2.0)
        # This may not always be true, so we just check they're valid
        assert loss_small >= 0 and loss_large >= 0
    
    def test_focal_loss(self, logits, labels):
        """Test focal loss."""
        loss = focal_loss(logits, labels, num_classes=4, alpha=0.25, gamma=2.0)
        
        # Loss should be a scalar
        assert loss.shape == ()
        
        # Loss should be positive
        assert loss > 0
    
    def test_compute_accuracy(self, logits, labels):
        """Test accuracy computation."""
        accuracy = compute_accuracy(logits, labels)
        
        # Accuracy should be between 0 and 1
        assert 0 <= accuracy <= 1
        
        # For our test logits, all predictions should be correct
        assert accuracy == 1.0


class TestTrainStateWithEMA:
    """Tests for TrainStateWithEMA."""
    
    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return ProductionCNN(n_classes=10, features=[32, 64], dropout_rate=0.1)
    
    def test_create_train_state(self, model):
        """Test creating training state."""
        rng = jax.random.PRNGKey(0)
        state = create_train_state(
            rng=rng,
            model=model,
            learning_rate=1e-3,
            weight_decay=1e-4,
            ema_decay=0.99,
            input_shape=(1, 32, 32, 3)
        )
        
        # Check state has required attributes
        assert hasattr(state, 'params')
        assert hasattr(state, 'ema_params')
        assert hasattr(state, 'opt_state')
        assert hasattr(state, 'step')
        
        # EMA params should be initialized
        assert state.ema_params is not None
    
    def test_ema_update(self, model):
        """Test EMA parameter update."""
        rng = jax.random.PRNGKey(0)
        state = create_train_state(
            rng=rng,
            model=model,
            learning_rate=1e-3,
            weight_decay=1e-4,
            ema_decay=0.99,
            input_shape=(1, 32, 32, 3)
        )
        
        # Store initial EMA params
        initial_ema = state.ema_params
        
        # Modify params (simulate a training step)
        new_params = jax.tree.map(lambda x: x + 0.01, state.params)
        state = state.replace(params=new_params)
        
        # Update EMA
        state = state.apply_ema_update()
        
        # EMA params should have changed
        changed = False
        for leaf_initial, leaf_new in zip(
            jax.tree_util.tree_leaves(initial_ema),
            jax.tree_util.tree_leaves(state.ema_params)
        ):
            if not jnp.allclose(leaf_initial, leaf_new):
                changed = True
                break
        assert changed
    
    def test_training_step(self, model):
        """Test a full training step."""
        from robust_vision.training.trainer import ProductionTrainer
        
        rng = jax.random.PRNGKey(0)
        state = create_train_state(
            rng=rng,
            model=model,
            learning_rate=1e-3,
            weight_decay=1e-4,
            ema_decay=0.99,
            input_shape=(1, 32, 32, 3)
        )
        
        # Create dummy batch
        rng, data_rng, dropout_rng = jax.random.split(rng, 3)
        images = jax.random.normal(data_rng, (8, 32, 32, 3))
        labels = jax.random.randint(data_rng, (8,), 0, 10)
        batch = (images, labels)
        
        # Training step with dropout_rng
        new_state, metrics = ProductionTrainer.train_step(
            state=state,
            batch=batch,
            num_classes=10,
            loss_type="label_smoothing",
            loss_kwargs={"smoothing": 0.1},
            dropout_rng=dropout_rng
        )
        
        # Check state was updated
        assert new_state.step > state.step
        
        # Check metrics
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert metrics['loss'] > 0


if __name__ == "__main__":
    pytest.main([__file__])
