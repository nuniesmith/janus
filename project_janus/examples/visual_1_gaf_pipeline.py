#!/usr/bin/env python3
"""
Visual 1: GAF (Gramian Angular Field) Transformation Pipeline
==============================================================

Three-panel visualization demonstrating the complete transformation from
raw price time series to GAF texture representation.

Panel A: Learnable Normalization
Panel B: Polar Clock Representation
Panel C: GASF vs GADF Comparison

Part of Project JANUS visualization specification.

Usage:
    python visual_1_gaf_pipeline.py --input prices.csv --output gaf_pipeline.png
    python visual_1_gaf_pipeline.py --synthetic --window 100 --show
    python visual_1_gaf_pipeline.py --example trending --save-all

Dependencies:
    pip install numpy matplotlib torch

Author: Project JANUS Team
License: MIT
"""

import argparse
import warnings
from pathlib import Path
from typing import Literal, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

# Optional: PyTorch for learnable normalization
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Using NumPy normalization fallback.")

# ============================================================================
# SYNTHETIC DATA GENERATORS
# ============================================================================


def generate_synthetic_prices(
    pattern: Literal["trending", "mean_reverting", "volatile", "quiet"] = "trending",
    length: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic price time series with different market regimes.

    Args:
        pattern: Market regime to simulate
        length: Number of timesteps
        seed: Random seed for reproducibility

    Returns:
        Price series array of shape [length]
    """
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, length)

    if pattern == "trending":
        # Upward trend with noise
        prices = 100 + 0.5 * t + 3 * np.random.randn(length)

    elif pattern == "mean_reverting":
        # Oscillating around mean
        prices = 100 + 10 * np.sin(t) + 2 * np.random.randn(length)

    elif pattern == "volatile":
        # High variance, choppy
        prices = 100 + 15 * np.random.randn(length)
        prices = np.cumsum(prices - 100) / 10 + 100

    elif pattern == "quiet":
        # Low variance, stable
        prices = 100 + 0.5 * np.sin(t) + 0.5 * np.random.randn(length)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return prices


# ============================================================================
# GAF TRANSFORMATION CORE
# ============================================================================


class LearnableNormalizer:
    """
    Learnable affine normalization with tanh wrapper.

    Implements: x_norm = tanh(gamma * (x - mu) / sigma + beta)
    """

    def __init__(self, use_torch: bool = False):
        self.use_torch = use_torch and TORCH_AVAILABLE

        if self.use_torch:
            self.gamma = torch.ones(1)
            self.beta = torch.zeros(1)
        else:
            self.gamma = 1.0
            self.beta = 0.0

        self.running_mean = None
        self.running_std = None

    def fit(self, x: np.ndarray):
        """Compute running statistics from data."""
        self.running_mean = np.mean(x)
        self.running_std = np.std(x) + 1e-8

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply learnable normalization."""
        if self.running_mean is None:
            self.fit(x)

        # Standardize
        x_std = (x - self.running_mean) / self.running_std

        # Learnable affine
        if self.use_torch:
            x_affine = self.gamma.item() * x_std + self.beta.item()
        else:
            x_affine = self.gamma * x_std + self.beta

        # Tanh wrapper ensures domain [-1, 1]
        x_norm = np.tanh(x_affine)

        return x_norm

    def get_saturated_indices(
        self, x_norm: np.ndarray, threshold: float = 0.95
    ) -> np.ndarray:
        """Detect saturated (fat-tail) events."""
        return np.abs(x_norm) > threshold


def compute_gaf(
    x_norm: np.ndarray, method: Literal["gasf", "gadf"] = "gasf"
) -> np.ndarray:
    """
    Compute Gramian Angular Field from normalized time series.

    Args:
        x_norm: Normalized time series in range [-1, 1]
        method: "gasf" (summation) or "gadf" (difference)

    Returns:
        GAF matrix of shape [len(x_norm), len(x_norm)]
    """
    # Ensure domain constraint
    x_norm = np.clip(x_norm, -1, 1)

    # Polar coordinate transformation
    phi = np.arccos(x_norm)  # Angular coordinate

    # Compute trigonometric components
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    if method == "gasf":
        # Gramian Angular Summation Field
        # cos(phi_i + phi_j) = cos(phi_i)cos(phi_j) - sin(phi_i)sin(phi_j)
        gaf = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)

    elif method == "gadf":
        # Gramian Angular Difference Field
        # sin(phi_i - phi_j) = sin(phi_i)cos(phi_j) - cos(phi_i)sin(phi_j)
        gaf = np.outer(sin_phi, cos_phi) - np.outer(cos_phi, sin_phi)

    else:
        raise ValueError(f"Unknown method: {method}")

    return gaf


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def visualize_gaf_pipeline(
    price_series: np.ndarray, save_path: Optional[str] = None, dpi: int = 300
) -> plt.Figure:
    """
    Generate Visual 1: Three-panel GAF transformation pipeline.

    Implements specification from Section 2.1.1.

    Args:
        price_series: Raw price time series [T]
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    # Initialize normalizer
    normalizer = LearnableNormalizer(use_torch=False)
    x_norm = normalizer.normalize(price_series)

    # Detect saturation
    saturated = normalizer.get_saturated_indices(x_norm)

    # Compute GAF matrices
    gasf = compute_gaf(x_norm, method="gasf")
    gadf = compute_gaf(x_norm, method="gadf")

    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 0.05], hspace=0.3, wspace=0.4)

    # ========================================================================
    # PANEL A: Learnable Normalization
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0:2])

    time = np.arange(len(price_series))

    # Raw series (gray, semi-transparent)
    ax_a.plot(
        time,
        price_series,
        color="gray",
        alpha=0.6,
        label="Raw Price",
        linewidth=1.5,
        zorder=1,
    )

    # Normalized series (blue)
    ax_a_twin = ax_a.twinx()
    ax_a_twin.plot(
        time, x_norm, color="#003366", label="Normalized", linewidth=2.5, zorder=2
    )

    # Saturation bounds
    ax_a_twin.axhline(
        1,
        color="red",
        linestyle="--",
        alpha=0.5,
        linewidth=1.5,
        label="Saturation bound",
        zorder=0,
    )
    ax_a_twin.axhline(
        -1, color="red", linestyle="--", alpha=0.5, linewidth=1.5, zorder=0
    )

    # Highlight saturated regions
    if np.any(saturated):
        ax_a_twin.scatter(
            time[saturated],
            x_norm[saturated],
            color="red",
            s=80,
            zorder=5,
            marker="X",
            edgecolors="darkred",
            linewidths=1.5,
            label="Saturation (fat tail)",
        )

    # Formatting
    ax_a.set_xlabel("Time", fontsize=11)
    ax_a.set_ylabel("Raw Price", fontsize=11, color="gray")
    ax_a_twin.set_ylabel("Normalized Value", fontsize=11, color="#003366")
    ax_a.set_title(
        "Panel A: Learnable Normalization\n"
        + r"$\tilde{x}_t = \tanh(\gamma \cdot \frac{x_t - \mu}{\sigma} + \beta)$",
        fontweight="bold",
        fontsize=12,
        pad=10,
    )
    ax_a.tick_params(axis="y", labelcolor="gray")
    ax_a_twin.tick_params(axis="y", labelcolor="#003366")
    ax_a.grid(True, alpha=0.3)
    ax_a_twin.set_ylim(-1.2, 1.2)

    # Combined legend
    lines1, labels1 = ax_a.get_legend_handles_labels()
    lines2, labels2 = ax_a_twin.get_legend_handles_labels()
    ax_a.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # ========================================================================
    # PANEL B: Polar Clock Representation
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 2], projection="polar")

    # Compute polar coordinates
    phi = np.arccos(np.clip(x_norm, -1, 1))  # Angular coordinate [0, π]
    r = np.linspace(0, 1, len(time))  # Radial coordinate [0, 1]

    # Plot spiral
    ax_b.plot(phi, r, color="#003366", linewidth=2.5, alpha=0.8)

    # Mark start and end points
    ax_b.scatter(
        phi[0],
        r[0],
        color="green",
        s=150,
        marker="o",
        zorder=5,
        edgecolors="darkgreen",
        linewidths=2,
        label="Start",
    )
    ax_b.scatter(
        phi[-1],
        r[-1],
        color="red",
        s=150,
        marker="o",
        zorder=5,
        edgecolors="darkred",
        linewidths=2,
        label="End",
    )

    # Mean price angle (dashed radial line)
    mean_angle = np.arccos(
        np.clip(np.tanh(0), -1, 1)
    )  # Angle at mean (normalized to 0)
    ax_b.plot(
        [mean_angle, mean_angle],
        [0, 1],
        "k--",
        alpha=0.5,
        linewidth=1.5,
        label="Mean price",
    )

    # Formatting
    ax_b.set_title(
        "Panel B: Polar Clock\n" + r"$\phi_t = \arccos(\tilde{x}_t), r_t = t/T$",
        fontweight="bold",
        fontsize=12,
        pad=20,
    )
    ax_b.set_theta_zero_location("N")
    ax_b.set_theta_direction(-1)
    ax_b.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax_b.grid(True, alpha=0.3)

    # ========================================================================
    # PANEL C: GASF vs GADF
    # ========================================================================

    # GASF subplot
    ax_c1 = fig.add_subplot(gs[0, 3:5])
    im1 = ax_c1.imshow(
        gasf, cmap="Blues", vmin=-1, vmax=1, aspect="auto", interpolation="nearest"
    )
    ax_c1.set_title(
        "GASF (Correlation)\n" + r"$\cos(\phi_i + \phi_j)$",
        fontweight="bold",
        fontsize=11,
    )
    ax_c1.set_xlabel("Time Index j", fontsize=10)
    ax_c1.set_ylabel("Time Index i", fontsize=10)
    ax_c1.tick_params(labelsize=9)

    # GADF subplot
    ax_c2 = fig.add_subplot(gs[0, 5])
    im2 = ax_c2.imshow(
        gadf, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest"
    )
    ax_c2.set_title(
        "GADF (Flux)\n" + r"$\sin(\phi_i - \phi_j)$", fontweight="bold", fontsize=11
    )
    ax_c2.set_xlabel("Time Index j", fontsize=10)
    ax_c2.set_yticklabels([])
    ax_c2.tick_params(labelsize=9)

    # Colorbars
    cbar1_ax = fig.add_subplot(gs[1, 3:5])
    cbar1 = plt.colorbar(im1, cax=cbar1_ax, orientation="horizontal")
    cbar1.set_label("Correlation Value", fontsize=10)

    cbar2_ax = fig.add_subplot(gs[1, 5])
    cbar2 = plt.colorbar(im2, cax=cbar2_ax, orientation="horizontal")
    cbar2.set_label("Flux Value", fontsize=10)

    # ========================================================================
    # Overall title
    # ========================================================================
    plt.suptitle(
        "Visual 1: GAF Transformation Pipeline\n"
        + "From Time Series to Spatiotemporal Manifold",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")

    return fig


def visualize_all_patterns(
    output_dir: str = "outputs", window: int = 100, dpi: int = 300
) -> None:
    """
    Generate GAF pipeline visualizations for all market patterns.

    Args:
        output_dir: Directory to save outputs
        window: Time window length
        dpi: Resolution for saved figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    patterns = ["trending", "mean_reverting", "volatile", "quiet"]

    for pattern in patterns:
        print(f"Generating {pattern.upper()} market visualization...")

        prices = generate_synthetic_prices(pattern, length=window)
        save_file = output_path / f"visual_1_gaf_{pattern}.png"

        fig = visualize_gaf_pipeline(prices, save_path=str(save_file), dpi=dpi)
        plt.close(fig)

    print(f"\n✓ All visualizations saved to {output_dir}/")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Parse arguments and generate visualization."""
    parser = argparse.ArgumentParser(
        description="Generate GAF transformation pipeline visualization (Visual 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from synthetic trending market
  %(prog)s --synthetic --example trending --output trending.png

  # Generate all patterns
  %(prog)s --save-all --window 150

  # Display interactively
  %(prog)s --synthetic --show

  # Load from CSV (first column = prices)
  %(prog)s --input prices.csv --output result.png
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i", type=str, help="Path to CSV file with price data"
    )
    input_group.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data generator"
    )
    input_group.add_argument(
        "--save-all", action="store_true", help="Generate all 4 market patterns"
    )

    # Synthetic data options
    parser.add_argument(
        "--example",
        choices=["trending", "mean_reverting", "volatile", "quiet"],
        default="trending",
        help="Market pattern to generate (default: trending)",
    )
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        default=100,
        help="Time window length (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Output options
    parser.add_argument(
        "--output", "-o", type=str, help="Output file path (e.g., gaf_pipeline.png)"
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plot interactively"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for --save-all (default: outputs/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for saved images (default: 300)",
    )

    args = parser.parse_args()

    # Generate visualization(s)
    if args.save_all:
        visualize_all_patterns(
            output_dir=args.output_dir, window=args.window, dpi=args.dpi
        )
    else:
        # Load or generate price data
        if args.input:
            # Load from CSV
            try:
                data = np.loadtxt(args.input, delimiter=",")
                if data.ndim > 1:
                    prices = data[:, 0]  # First column
                else:
                    prices = data
                print(f"Loaded {len(prices)} prices from {args.input}")
            except Exception as e:
                print(f"Error loading {args.input}: {e}")
                return
        else:
            # Generate synthetic
            prices = generate_synthetic_prices(
                pattern=args.example, length=args.window, seed=args.seed
            )
            print(f"Generated {args.example} market with {len(prices)} timesteps")

        # Visualize
        fig = visualize_gaf_pipeline(prices, save_path=args.output, dpi=args.dpi)

        if args.show:
            plt.show()
        elif not args.output:
            print("Warning: No output path specified and --show not set.")
            print("Use --output or --show to view results.")


if __name__ == "__main__":
    main()
