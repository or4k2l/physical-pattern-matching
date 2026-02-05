# ==============================================================================
# PHYSICALLY-INSPIRED ROBUST PATTERN MATCHING: SYSTEMATIC KITTI BENCHMARK
# Architect: Yahya Akbay (or4k2l) - Enhanced Scientific Version
# Dataset: KITTI LiDAR-based Depth Images (via KaggleHub)
#
# This script systematically tests whether memristive crossbar arrays with
# physics-inspired learning rules show superior robustness compared to
# standard digital approaches under various noise conditions.
# ==============================================================================

import kagglehub
import os
import glob
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
from jax import jit, random
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict
import pandas as pd

# ==============================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# ==============================================================================

print("=" * 80)
print("DOWNLOADING KITTI DATASET")
print("=" * 80)

path = kagglehub.dataset_download("ahmedfawzyelaraby/kitti-lidar-based-2d-depth-images")
print(f"Dataset downloaded to: {path}\n")


def load_kitti_images(dataset_path: str, resolution: Tuple[int, int] = (64, 64),
                      n_samples: int = 20) -> Tuple[np.ndarray, List[str]]:
    """
    Load multiple KITTI images for robust testing.

    Args:
        dataset_path: Path to KITTI dataset
        resolution: Target resolution (hardware constraint simulation)
        n_samples: Number of images to load for testing

    Returns:
        images: Array of shape [n_samples, height * width]
        filenames: List of source filenames
    """
    image_files = glob.glob(f"{dataset_path}/**/*.png", recursive=True)

    if not image_files:
        raise ValueError("No images found in dataset!")

    # Sample diverse images
    indices = np.linspace(0, len(image_files) - 1, n_samples, dtype=int)

    images = []
    filenames = []

    for idx in indices:
        img_path = image_files[idx]
        img = Image.open(img_path).convert("L")
        img = img.resize(resolution)
        img_array = np.array(img) / 255.0
        images.append(img_array.flatten())
        filenames.append(os.path.basename(img_path))

    return np.array(images), filenames


# ==============================================================================
# SECTION 2: PHYSICS-INSPIRED CROSSBAR MODEL
# ==============================================================================

# JIT-compiled helper functions (functional approach for JAX compatibility)
@jit
def _crossbar_inference(weights: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
    """Crossbar inference: I = G^T . V"""
    return jnp.dot(weights.T, inputs)


@jit
def _hebbian_update(weights: jnp.ndarray, inputs: jnp.ndarray,
                   target: jnp.ndarray, lr: float,
                   g_min: float, g_max: float) -> jnp.ndarray:
    """Hebbian plasticity: Delta G proportional to post and pre."""
    correlation = jnp.outer(inputs, target)
    new_weights = weights + lr * correlation
    return jnp.clip(new_weights, g_min, g_max)


@jit
def _digital_inference(weights: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
    """Standard matrix multiplication - same as physical crossbar."""
    # Both physical and digital use the same computation: I = G^T . V
    # The only difference is the learning rule
    return jnp.dot(weights.T, inputs)


@jit
def _sgd_update(weights: jnp.ndarray, inputs: jnp.ndarray,
                target: jnp.ndarray, lr: float) -> jnp.ndarray:
    """Standard SGD with MSE loss (unconstrained)."""
    output = jnp.dot(weights.T, inputs)
    error = output - target
    gradient = jnp.outer(inputs, error)
    # No clipping - this is the key difference from physical crossbar
    return weights - lr * gradient


@dataclass
class CrossbarConfig:
    """Configuration for memristive crossbar array."""
    n_inputs: int
    n_outputs: int
    conductance_range: Tuple[float, float] = (0.0, 1.0)
    learning_rate: float = 0.2
    digital_learning_rate: float = 0.01  # Lower LR for unconstrained optimization
    training_iterations: int = 30


class PhysicalCrossbar:
    """
    Simulates a memristive crossbar array with physically-motivated learning.

    Key properties:
    - Conductance-based computation (I = G . V)
    - Hebbian plasticity
    - Saturation constraints (mimicking physical device limits)
    """

    def __init__(self, config: CrossbarConfig, seed: int = 42):
        self.config = config
        self.key = random.PRNGKey(seed)

        # Initialize conductances near zero (fresh memristors)
        self.weights = random.uniform(
            self.key,
            (config.n_inputs, config.n_outputs),
            minval=0.0,
            maxval=0.01
        )

    def inference(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Crossbar inference: I = G^T . V

        Args:
            inputs: Voltage vector [n_inputs]

        Returns:
            currents: Output currents [n_outputs]
        """
        return _crossbar_inference(self.weights, inputs)

    def train(self, image: jnp.ndarray, target_signal: jnp.ndarray):
        """Train crossbar on single image with target pattern."""
        for _ in range(self.config.training_iterations):
            self.weights = _hebbian_update(
                self.weights,
                image,
                target_signal,
                self.config.learning_rate,
                self.config.conductance_range[0],
                self.config.conductance_range[1]
            )

    def get_memory_trace(self, neuron_idx: int = 0) -> np.ndarray:
        """Extract learned conductance pattern for visualization."""
        return np.array(self.weights[:, neuron_idx])


# ==============================================================================
# SECTION 3: BASELINE - STANDARD DIGITAL NETWORK
# ==============================================================================

class DigitalBaseline:
    """
    Standard digital neural network for comparison.
    Same architecture, but without physical constraints.
    """

    def __init__(self, config: CrossbarConfig, seed: int = 42):
        self.config = config
        key = random.PRNGKey(seed)

        # Initialize similarly to physical crossbar (small positive values)
        # This ensures fair comparison - both start from similar initial conditions
        self.weights = random.uniform(
            key,
            (config.n_inputs, config.n_outputs),
            minval=0.0,
            maxval=0.01
        )

    def inference(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Standard matrix multiplication."""
        return _digital_inference(self.weights, inputs)

    def train(self, image: jnp.ndarray, target_signal: jnp.ndarray):
        """Train with gradient descent (unconstrained but with lower LR for stability)."""
        for _ in range(self.config.training_iterations):
            self.weights = _sgd_update(
                self.weights,
                image,
                target_signal,
                self.config.digital_learning_rate  # Use lower LR
            )


# ==============================================================================
# SECTION 4: NOISE MODELS
# ==============================================================================


def inject_gaussian_noise(clean: jnp.ndarray, noise_level: float,
                          seed: int) -> jnp.ndarray:
    """Additive Gaussian noise (sensor thermal noise)."""
    key = random.PRNGKey(seed)
    noise = random.normal(key, clean.shape) * noise_level
    return jnp.clip(clean + noise, 0.0, 1.0)


def inject_salt_pepper_noise(clean: jnp.ndarray, noise_level: float,
                             seed: int) -> jnp.ndarray:
    """Salt and pepper noise (pixel dropouts, LiDAR occlusions)."""
    key = random.PRNGKey(seed)
    mask = random.uniform(key, clean.shape)

    noisy = clean.copy()
    noisy = jnp.where(mask < noise_level / 2, 0.0, noisy)  # Pepper
    noisy = jnp.where(mask > 1.0 - noise_level / 2, 1.0, noisy)  # Salt
    return noisy


def inject_structured_occlusion(clean: jnp.ndarray, resolution: int,
                                occlusion_ratio: float, seed: int) -> jnp.ndarray:
    """Block occlusion (simulates object obstruction)."""
    key = random.PRNGKey(seed)
    img_2d = clean.reshape(resolution, resolution)

    # Random occlusion block
    block_size = int(resolution * np.sqrt(occlusion_ratio))
    start_x = random.randint(key, (), 0, resolution - block_size)
    start_y = random.randint(random.split(key)[0], (), 0, resolution - block_size)

    img_2d = img_2d.at[start_x:start_x + block_size, start_y:start_y + block_size].set(0.0)
    return img_2d.flatten()


def inject_occlusion_wrapper(clean: jnp.ndarray, noise_level: float,
                            seed: int, resolution: int = 64) -> jnp.ndarray:
    """Wrapper for occlusion with consistent signature."""
    return inject_structured_occlusion(clean, resolution, noise_level, seed)


NOISE_TYPES = {
    "gaussian": inject_gaussian_noise,
    "salt_pepper": inject_salt_pepper_noise,
    "occlusion": inject_occlusion_wrapper
}


# ==============================================================================
# SECTION 5: SYSTEMATIC BENCHMARK
# ==============================================================================


def run_benchmark(images: np.ndarray, filenames: List[str],
                 noise_levels: List[float], resolution: int = 64) -> pd.DataFrame:
    """
    Systematic comparison: Physical Crossbar vs Digital Baseline

    Metrics:
    - Signal-to-Noise Ratio (SNR): target_output / control_output
    - Classification Accuracy: percent correct detections above threshold
    - Robustness Index: Area under SNR curve across noise levels

    Returns:
        DataFrame with comparative results
    """
    n_inputs = resolution * resolution
    config = CrossbarConfig(n_inputs=n_inputs, n_outputs=2)
    target_signal = jnp.array([1.0, 0.0])  # Neuron 0 = target, Neuron 1 = control

    results = []

    print("\n" + "=" * 80)
    print("RUNNING SYSTEMATIC BENCHMARK")
    print("=" * 80)

    for img_idx, (clean_img, filename) in enumerate(zip(images, filenames)):
        print(f"\nTesting Image {img_idx + 1}/{len(images)}: {filename}")

        clean_img = jnp.array(clean_img)

        # Train both models on clean image
        physical = PhysicalCrossbar(config, seed=42 + img_idx)
        digital = DigitalBaseline(config, seed=42 + img_idx)

        physical.train(clean_img, target_signal)
        digital.train(clean_img, target_signal)

        # Test across noise types and levels
        for noise_type_name, noise_func in NOISE_TYPES.items():
            for noise_level in tqdm(noise_levels, desc=f"  {noise_type_name.capitalize()} noise"):

                # Generate noisy input
                noisy_img = noise_func(clean_img, noise_level, seed=99 + img_idx)

                # Inference
                physical_output = physical.inference(noisy_img)
                digital_output = digital.inference(noisy_img)

                # Metrics
                physical_snr = physical_output[0] / (physical_output[1] + 1e-6)
                digital_snr = digital_output[0] / (digital_output[1] + 1e-6)

                physical_correct = float(physical_output[0] > physical_output[1])
                digital_correct = float(digital_output[0] > digital_output[1])

                results.append({
                    "image": filename,
                    "noise_type": noise_type_name,
                    "noise_level": noise_level,
                    "physical_snr": float(physical_snr),
                    "digital_snr": float(digital_snr),
                    "physical_correct": physical_correct,
                    "digital_correct": digital_correct,
                    "physical_output_0": float(physical_output[0]),
                    "physical_output_1": float(physical_output[1]),
                    "digital_output_0": float(digital_output[0]),
                    "digital_output_1": float(digital_output[1])
                })

    return pd.DataFrame(results)


# ==============================================================================
# SECTION 6: VISUALIZATION & ANALYSIS
# ==============================================================================


def plot_comprehensive_results(df: pd.DataFrame, sample_images: np.ndarray):
    """Generate publication-quality figures."""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # --- PLOT 1: SNR Comparison Across Noise Levels ---
    ax1 = fig.add_subplot(gs[0, :2])

    for noise_type in df["noise_type"].unique():
        subset = df[df["noise_type"] == noise_type]
        grouped = subset.groupby("noise_level").agg({
            "physical_snr": "mean",
            "digital_snr": "mean"
        }).reset_index()

        ax1.plot(grouped["noise_level"], grouped["physical_snr"],
                marker="o", linewidth=2, label=f"Physical ({noise_type})")
        ax1.plot(grouped["noise_level"], grouped["digital_snr"],
                marker="s", linewidth=2, linestyle="--", label=f"Digital ({noise_type})")

    ax1.set_xlabel("Noise Level", fontsize=12)
    ax1.set_ylabel("Signal-to-Noise Ratio (log scale)", fontsize=12)
    ax1.set_yscale("log")
    ax1.set_title("Robustness Comparison: Physical vs Digital", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    # --- PLOT 2: Accuracy Comparison ---
    ax2 = fig.add_subplot(gs[0, 2:])

    accuracy_data = df.groupby("noise_type").agg({
        "physical_correct": "mean",
        "digital_correct": "mean"
    }).reset_index()

    x = np.arange(len(accuracy_data))
    width = 0.35

    ax2.bar(x - width / 2, accuracy_data["physical_correct"] * 100, width,
           label="Physical Crossbar", color="#2ecc71", alpha=0.8)
    ax2.bar(x + width / 2, accuracy_data["digital_correct"] * 100, width,
           label="Digital Baseline", color="#3498db", alpha=0.8)

    ax2.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax2.set_title("Average Accuracy by Noise Type", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(accuracy_data["noise_type"])
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim([0, 105])

    # --- PLOT 3: Example Clean Image ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(sample_images[0].reshape(64, 64), cmap="inferno")
    ax3.set_title("Ground Truth (KITTI)", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # --- PLOT 4: Gaussian Noise Example ---
    ax4 = fig.add_subplot(gs[1, 1])
    noisy = inject_gaussian_noise(jnp.array(sample_images[0]), 0.5, 99)
    ax4.imshow(noisy.reshape(64, 64), cmap="inferno")
    ax4.set_title("Gaussian Noise (50%)", fontsize=11, fontweight="bold")
    ax4.axis("off")

    # --- PLOT 5: Salt and Pepper Example ---
    ax5 = fig.add_subplot(gs[1, 2])
    noisy = inject_salt_pepper_noise(jnp.array(sample_images[0]), 0.3, 99)
    ax5.imshow(noisy.reshape(64, 64), cmap="inferno")
    ax5.set_title("Salt and Pepper (30%)", fontsize=11, fontweight="bold")
    ax5.axis("off")

    # --- PLOT 6: Occlusion Example ---
    ax6 = fig.add_subplot(gs[1, 3])
    noisy = inject_structured_occlusion(jnp.array(sample_images[0]), 64, 0.25, 99)
    ax6.imshow(noisy.reshape(64, 64), cmap="inferno")
    ax6.set_title("Occlusion (25%)", fontsize=11, fontweight="bold")
    ax6.axis("off")

    # --- PLOT 7: Statistical Significance Test ---
    ax7 = fig.add_subplot(gs[2, :2])

    # Violin plot comparing distributions
    physical_snrs = df["physical_snr"].values
    digital_snrs = df["digital_snr"].values

    ax7.violinplot([physical_snrs, digital_snrs],
                   positions=[1, 2],
                   showmeans=True,
                   showmedians=True)

    ax7.set_xticks([1, 2])
    ax7.set_xticklabels(["Physical\nCrossbar", "Digital\nBaseline"])
    ax7.set_ylabel("SNR Distribution", fontsize=12)
    ax7.set_title("Statistical Distribution of Robustness", fontsize=14, fontweight="bold")
    ax7.grid(True, axis="y", alpha=0.3)
    ax7.set_yscale("log")

    # --- PLOT 8: Summary Statistics ---
    ax8 = fig.add_subplot(gs[2, 2:])
    ax8.axis("off")

    # Calculate key metrics
    physical_mean_snr = df["physical_snr"].mean()
    digital_mean_snr = df["digital_snr"].mean()
    physical_acc = df["physical_correct"].mean() * 100
    digital_acc = df["digital_correct"].mean() * 100

    improvement = ((physical_mean_snr - digital_mean_snr) / digital_mean_snr) * 100

    summary_text = f"""
    +------------------------------------------+
    |           BENCHMARK SUMMARY              |
    +------------------------------------------+
    |                                          |
    |  Physical Crossbar:                      |
    |    Mean SNR: {physical_mean_snr:.2f}                   |
    |    Accuracy: {physical_acc:.1f}%                    |
    |                                          |
    |  Digital Baseline:                       |
    |    Mean SNR: {digital_mean_snr:.2f}                    |
    |    Accuracy: {digital_acc:.1f}%                     |
    |                                          |
    |  Relative Improvement:                   |
    |    SNR: {improvement:+.1f}%                          |
    |                                          |
    |  Total Tests: {len(df)}                        |
    |  Images: {df['image'].nunique()}                            |
    |  Noise Types: {df['noise_type'].nunique()}                      |
    +------------------------------------------+
    """

    ax8.text(0.1, 0.5, summary_text, fontsize=11, fontfamily="monospace",
            verticalalignment="center", bbox=dict(boxstyle="round",
            facecolor="wheat", alpha=0.3))

    plt.suptitle("Physically-Inspired Pattern Matching: Systematic KITTI Benchmark",
                fontsize=16, fontweight="bold", y=0.98)

    return fig



def plot_detailed_single_case(clean_img: np.ndarray, physical: PhysicalCrossbar,
                              digital: DigitalBaseline, resolution: int = 64):
    """Detailed analysis of a single test case."""

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Physical Crossbar Analysis
    axes[0, 0].imshow(clean_img.reshape(resolution, resolution), cmap="inferno")
    axes[0, 0].set_title("Training Image (Clean)", fontweight="bold")
    axes[0, 0].axis("off")

    noisy = inject_gaussian_noise(jnp.array(clean_img), 0.6, 99)
    axes[0, 1].imshow(noisy.reshape(resolution, resolution), cmap="inferno")
    axes[0, 1].set_title("Test Input (60% Noise)", fontweight="bold")
    axes[0, 1].axis("off")

    memory = physical.get_memory_trace(0).reshape(resolution, resolution)
    im1 = axes[0, 2].imshow(memory, cmap="hot")
    axes[0, 2].set_title("Learned Conductance Map\n(Physical Memory)", fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)

    phys_out = physical.inference(noisy)
    axes[0, 3].bar(["Target\nNeuron", "Control\nNeuron"], phys_out, color=["green", "gray"])
    axes[0, 3].set_title(f"Physical Output\nSNR: {phys_out[0] / (phys_out[1] + 1e-6):.1f}",
                        fontweight="bold")
    axes[0, 3].set_ylabel("Current (I)")
    axes[0, 3].grid(True, axis="y", alpha=0.3)

    # Row 2: Digital Baseline Analysis
    axes[1, 0].imshow(clean_img.reshape(resolution, resolution), cmap="inferno")
    axes[1, 0].set_title("Training Image (Clean)", fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(noisy.reshape(resolution, resolution), cmap="inferno")
    axes[1, 1].set_title("Test Input (60% Noise)", fontweight="bold")
    axes[1, 1].axis("off")

    dig_memory = digital.weights[:, 0].reshape(resolution, resolution)
    im2 = axes[1, 2].imshow(dig_memory, cmap="hot")
    axes[1, 2].set_title("Learned Weight Map\n(Digital Memory)", fontweight="bold")
    axes[1, 2].axis("off")
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

    dig_out = digital.inference(noisy)
    axes[1, 3].bar(["Target\nNeuron", "Control\nNeuron"], dig_out, color=["blue", "gray"])
    axes[1, 3].set_title(f"Digital Output\nSNR: {dig_out[0] / (dig_out[1] + 1e-6):.1f}",
                        fontweight="bold")
    axes[1, 3].set_ylabel("Activation")
    axes[1, 3].grid(True, axis="y", alpha=0.3)

    fig.text(0.01, 0.75, "PHYSICAL\nCROSSBAR", fontsize=14, fontweight="bold",
            rotation=90, va="center")
    fig.text(0.01, 0.25, "DIGITAL\nBASELINE", fontsize=14, fontweight="bold",
            rotation=90, va="center")

    plt.tight_layout()
    return fig


# ==============================================================================
# SECTION 7: MAIN EXECUTION
# ==============================================================================


def main():
    """Run complete benchmark suite."""

    # Configuration
    resolution = 64
    n_images = 10  # Use more images for robust statistics
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Create output directory if it does not exist
    output_dir = "/mnt/user-data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n" + "=" * 80)
    print("LOADING KITTI DATASET")
    print("=" * 80)
    images, filenames = load_kitti_images(path, (resolution, resolution), n_images)
    print(f"Loaded {len(images)} images at {resolution}x{resolution} resolution")

    # Run benchmark
    results_df = run_benchmark(images, filenames, noise_levels, resolution)

    # Save results
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    fig1 = plot_comprehensive_results(results_df, images)
    fig1_path = os.path.join(output_dir, "comprehensive_benchmark.png")
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig1_path}")

    # Detailed single case analysis
    config = CrossbarConfig(n_inputs=resolution * resolution, n_outputs=2)
    target = jnp.array([1.0, 0.0])

    physical = PhysicalCrossbar(config)
    digital = DigitalBaseline(config)

    test_img = jnp.array(images[0])
    physical.train(test_img, target)
    digital.train(test_img, target)

    fig2 = plot_detailed_single_case(images[0], physical, digital, resolution)
    fig2_path = os.path.join(output_dir, "detailed_case_study.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig2_path}")

    plt.show()

    # Final statistical summary
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    print("\nMean SNR by approach:")
    print(results_df.groupby("noise_type")[["physical_snr", "digital_snr"]].mean())

    print("\nAccuracy by noise type:")
    print(results_df.groupby("noise_type")[["physical_correct", "digital_correct"]].mean() * 100)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"1. Check {fig1_path} for cross-noise comparison")
    print(f"2. Check {fig2_path} for single-image analysis")
    print(f"3. Check {csv_path} for raw data")
    print("\nInterpretation Guide:")
    print("- Higher SNR = Better signal separation")
    print("- Physical constraints may provide implicit regularization")
    print("- Real-world advantage depends on hardware implementation")
    print("=" * 80)


if __name__ == "__main__":
    main()
