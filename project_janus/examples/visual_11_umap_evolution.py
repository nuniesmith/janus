#!/usr/bin/env python3
"""
Visual 11: UMAP Schema Manifold Evolution

This module implements the UMAP projection visualization for Project JANUS's
neocortical memory system, showing how high-dimensional market schemas evolve
and cluster over time across three memory timescales.

Theoretical Foundation:
- UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
- Three-timescale memory architecture: Short-term (seconds), Mid-term (hours), Long-term (days)
- Manifold learning reveals emergent market regime clustering
- Topology preservation validates learned representations

Reference:
McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold
Approximation and Projection for Dimension Reduction. arXiv:1802.03426

Author: Project JANUS Visualization Team
License: MIT
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

warnings.filterwarnings("ignore")

# Try to import UMAP (optional dependency)
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Using PCA fallback for demonstration.")
    print("Install with: pip install umap-learn")


class SchemaMemory:
    """
    Simulates the neocortical schema memory system with three timescales.

    This class generates synthetic high-dimensional market state embeddings
    and tracks their evolution as the system learns market regimes.
    """

    def __init__(self, n_features: int = 64, n_regimes: int = 3, seed: int = 42):
        """
        Initialize schema memory system.

        Parameters
        ----------
        n_features : int
            Dimensionality of schema embeddings
        n_regimes : int
            Number of distinct market regimes
        seed : int
            Random seed for reproducibility
        """
        self.n_features = n_features
        self.n_regimes = n_regimes
        self.seed = seed
        np.random.seed(seed)

        # Define market regimes
        self.regime_names = ["Bullish", "Bearish", "Choppy"][:n_regimes]
        self.regime_colors = {
            "Bullish": "#2E8B57",  # SeaGreen
            "Bearish": "#DC143C",  # Crimson
            "Choppy": "#FFB347",  # Pastel Orange
        }

    def generate_schemas(
        self, n_samples: int, time_step: int, convergence_rate: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate schema embeddings at a specific training time step.

        Parameters
        ----------
        n_samples : int
            Number of schema samples to generate
        time_step : int
            Training iteration (controls cluster separation)
        convergence_rate : float
            Rate at which clusters separate over time

        Returns
        -------
        embeddings : np.ndarray, shape (n_samples, n_features)
            High-dimensional schema embeddings
        labels : np.ndarray, shape (n_samples,)
            Regime labels (0, 1, 2, ...)
        """
        # Cluster separation increases with training
        cluster_std = 3.0 * np.exp(-convergence_rate * time_step) + 0.5

        # Generate regime-specific schema clusters
        embeddings, labels = make_blobs(
            n_samples=n_samples,
            n_features=self.n_features,
            centers=self.n_regimes,
            cluster_std=cluster_std,
            random_state=self.seed + time_step,
        )

        return embeddings, labels


def compute_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    """
    Project high-dimensional embeddings to 2D using UMAP.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, n_features)
        High-dimensional data
    n_neighbors : int
        UMAP parameter: local neighborhood size
    min_dist : float
        UMAP parameter: minimum distance between points in 2D
    metric : str
        Distance metric ('euclidean', 'cosine', etc.)
    random_state : int
        Random seed

    Returns
    -------
    projection : np.ndarray, shape (n_samples, 2)
        2D coordinates
    """
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric=metric,
            random_state=random_state,
        )
        projection = reducer.fit_transform(embeddings)
    else:
        # Fallback to PCA if UMAP not available
        reducer = PCA(n_components=2, random_state=random_state)
        projection = reducer.fit_transform(embeddings)

    return projection


def compute_trustworthiness(
    X_high: np.ndarray, X_low: np.ndarray, k: int = 15
) -> float:
    """
    Compute trustworthiness metric T(k).

    Measures how well k-nearest neighbors are preserved from high-D to low-D.

    Parameters
    ----------
    X_high : np.ndarray, shape (n_samples, n_features_high)
        High-dimensional data
    X_low : np.ndarray, shape (n_samples, n_features_low)
        Low-dimensional projection
    k : int
        Number of neighbors to consider

    Returns
    -------
    T : float
        Trustworthiness score in [0, 1] (higher is better)
    """
    n = X_high.shape[0]

    # Compute pairwise distances
    D_high = pairwise_distances(X_high)
    D_low = pairwise_distances(X_low)

    # Get k-nearest neighbors in each space
    nn_high = np.argsort(D_high, axis=1)[:, 1 : k + 1]  # Exclude self
    nn_low = np.argsort(D_low, axis=1)[:, 1 : k + 1]

    # Count preserved neighbors
    preserved = 0
    for i in range(n):
        preserved += len(set(nn_high[i]) & set(nn_low[i]))

    T = preserved / (n * k)
    return T


def compute_distortion_ratio(X_high: np.ndarray, X_low: np.ndarray) -> np.ndarray:
    """
    Compute per-point distortion ratio.

    For each point, compute mean(d_low / d_high) across all neighbors.
    Ratio ≈ 1 indicates faithful projection.

    Parameters
    ----------
    X_high : np.ndarray, shape (n_samples, n_features_high)
        High-dimensional data
    X_low : np.ndarray, shape (n_samples, n_features_low)
        Low-dimensional projection

    Returns
    -------
    ratios : np.ndarray, shape (n_samples,)
        Distortion ratio per point
    """
    n = X_high.shape[0]
    D_high = pairwise_distances(X_high)
    D_low = pairwise_distances(X_low)

    ratios = np.zeros(n)
    for i in range(n):
        # Avoid division by zero
        mask = D_high[i] > 1e-10
        if mask.sum() > 0:
            ratios[i] = np.mean(D_low[i, mask] / D_high[i, mask])
        else:
            ratios[i] = 1.0

    return ratios


def plot_umap_panel(
    ax: plt.Axes,
    projection: np.ndarray,
    labels: np.ndarray,
    regime_names: List[str],
    regime_colors: Dict[str, str],
    time_step: int,
    title: str,
):
    """
    Plot a single UMAP panel.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    projection : np.ndarray, shape (n_samples, 2)
        2D UMAP coordinates
    labels : np.ndarray, shape (n_samples,)
        Regime labels
    regime_names : list of str
        Names of regimes
    regime_colors : dict
        Color mapping for each regime
    time_step : int
        Training iteration number
    title : str
        Panel title
    """
    # Plot each regime separately for legend
    for regime_idx, regime_name in enumerate(regime_names):
        mask = labels == regime_idx
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            c=regime_colors[regime_name],
            s=30,
            alpha=0.6,
            edgecolors="white",
            linewidths=0.5,
            label=regime_name,
        )

    ax.set_xlabel("UMAP Dimension 1", fontsize=10, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_aspect("equal", adjustable="box")

    # Add time step annotation
    ax.text(
        0.02,
        0.98,
        f"t = {time_step}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def plot_distortion_heatmap(
    ax: plt.Axes,
    projection: np.ndarray,
    distortion: np.ndarray,
    title: str = "Distortion Heatmap",
):
    """
    Plot UMAP projection with distortion ratio as color overlay.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    projection : np.ndarray, shape (n_samples, 2)
        2D coordinates
    distortion : np.ndarray, shape (n_samples,)
        Per-point distortion ratios
    title : str
        Panel title
    """
    # Clip distortion for visualization
    distortion_clipped = np.clip(distortion, 0.5, 2.0)

    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=distortion_clipped,
        s=40,
        cmap="RdYlBu_r",
        vmin=0.5,
        vmax=2.0,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Distortion Ratio\n(d_2D / d_highD)", fontsize=9, fontweight="bold")

    ax.set_xlabel("UMAP Dimension 1", fontsize=10, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_aspect("equal", adjustable="box")

    # Add interpretation guide
    ax.text(
        0.02,
        0.02,
        "Blue ≈ 1: Faithful\nRed > 1: Inflated",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def plot_trustworthiness_curve(
    ax: plt.Axes,
    X_high: np.ndarray,
    X_low: np.ndarray,
    k_values: List[int] = [5, 10, 15, 20, 30, 50],
):
    """
    Plot trustworthiness T(k) as a function of k.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    X_high : np.ndarray
        High-dimensional data
    X_low : np.ndarray
        Low-dimensional projection
    k_values : list of int
        Values of k to compute trustworthiness for
    """
    T_scores = [compute_trustworthiness(X_high, X_low, k) for k in k_values]

    ax.plot(k_values, T_scores, "o-", color="#4682B4", linewidth=2, markersize=8)

    # Threshold lines
    ax.axhline(
        y=0.8, color="green", linestyle="--", linewidth=1.5, label="Good (T > 0.8)"
    )
    ax.axhline(
        y=0.6,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="Acceptable (T > 0.6)",
    )

    ax.set_xlabel("Number of Neighbors (k)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Trustworthiness T(k)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Projection Quality: Trustworthiness", fontsize=12, fontweight="bold", pad=10
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Annotate final value
    final_T = T_scores[-1]
    ax.text(
        k_values[-1],
        final_T,
        f"  T({k_values[-1]}) = {final_T:.3f}",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#4682B4",
    )


def visualize_umap_evolution(
    memory: SchemaMemory,
    time_steps: List[int] = [0, 100, 500, 1000],
    n_samples: int = 500,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Create complete UMAP schema evolution visualization (V11).

    Parameters
    ----------
    memory : SchemaMemory
        Schema memory system
    time_steps : list of int
        Training iterations to visualize
    n_samples : int
        Number of schema samples per time step
    save_path : Path, optional
        Output path for PNG
    dpi : int
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.Figure
        The complete figure
    """
    n_panels = len(time_steps)

    # Create figure with 2 rows: evolution + quality metrics
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, n_panels, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

    # Row 1: UMAP evolution across time steps
    projections = []
    embeddings_list = []

    for col_idx, t in enumerate(time_steps):
        # Generate schemas at this time step
        embeddings, labels = memory.generate_schemas(n_samples, t)
        embeddings_list.append(embeddings)

        # Compute UMAP projection
        projection = compute_umap_projection(
            embeddings, n_neighbors=15, min_dist=0.1, random_state=memory.seed
        )
        projections.append(projection)

        # Plot panel
        ax = fig.add_subplot(gs[0, col_idx])

        if t == 0:
            title = f"t = {t}\n(Random Init)"
        elif t == time_steps[-1]:
            title = f"t = {t}\n(Converged)"
        else:
            title = f"t = {t}\n(Learning)"

        plot_umap_panel(
            ax, projection, labels, memory.regime_names, memory.regime_colors, t, title
        )

    # Row 2: Quality metrics for final time step
    final_embeddings = embeddings_list[-1]
    final_projection = projections[-1]

    # Panel 1: Distortion heatmap
    ax_dist = fig.add_subplot(gs[1, 0:2])
    distortion = compute_distortion_ratio(final_embeddings, final_projection)
    plot_distortion_heatmap(ax_dist, final_projection, distortion)

    # Panel 2: Trustworthiness curve
    ax_trust = fig.add_subplot(gs[1, 2:])
    plot_trustworthiness_curve(ax_trust, final_embeddings, final_projection)

    # Overall title
    method_name = "UMAP" if UMAP_AVAILABLE else "PCA (Fallback)"
    fig.suptitle(
        f"Visual 11: Schema Manifold Evolution ({method_name})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Metadata footer
    metadata = (
        f"Schema Dimensionality: {memory.n_features}D → 2D | "
        f"Regimes: {memory.n_regimes} | "
        f"Samples: {n_samples} per time step"
    )
    fig.text(
        0.5,
        0.01,
        metadata,
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
        color="gray",
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"✓ Saved: {save_path}")

    return fig


def main():
    """Main execution function with CLI."""
    parser = argparse.ArgumentParser(
        description="Visual 11: UMAP Schema Manifold Evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visual_11_umap_evolution.py --show
  python visual_11_umap_evolution.py --save-all --output-dir ../outputs
  python visual_11_umap_evolution.py --time-steps 0 200 500 1500 --samples 1000
        """,
    )

    parser.add_argument(
        "--show", action="store_true", help="Display figure interactively"
    )
    parser.add_argument(
        "--save-all", action="store_true", help="Save all output figures"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs"),
        help="Output directory for saved figures",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        nargs="+",
        default=[0, 100, 500, 1000],
        help="Training iterations to visualize",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of schema samples per time step",
    )
    parser.add_argument(
        "--n-features", type=int, default=64, help="Dimensionality of schema embeddings"
    )
    parser.add_argument(
        "--n-regimes", type=int, default=3, help="Number of market regimes"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for saved figures"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Visual 11: UMAP Schema Manifold Evolution")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Schema dimensionality: {args.n_features}D → 2D")
    print(f"  Market regimes: {args.n_regimes}")
    print(f"  Samples per time step: {args.samples}")
    print(f"  Time steps: {args.time_steps}")
    print(f"  Random seed: {args.seed}")

    if not UMAP_AVAILABLE:
        print("\n⚠ Warning: UMAP not installed, using PCA fallback")
        print("  For production use, install: pip install umap-learn")

    # Initialize schema memory
    print(f"\n[1/2] Initializing schema memory system...")
    memory = SchemaMemory(
        n_features=args.n_features, n_regimes=args.n_regimes, seed=args.seed
    )
    print(f"✓ Memory system ready with {args.n_regimes} regimes")

    # Create visualization
    print(f"\n[2/2] Computing UMAP projections and generating visualization...")

    save_path = None
    if args.save_all:
        save_path = args.output_dir / "visual_11_umap_evolution.png"

    fig = visualize_umap_evolution(
        memory=memory,
        time_steps=args.time_steps,
        n_samples=args.samples,
        save_path=save_path,
        dpi=args.dpi,
    )

    print(f"\n{'=' * 70}")
    print("✓ Visualization complete!")
    print(f"{'=' * 70}")

    if args.save_all:
        print(f"\nOutput saved to: {args.output_dir}/")

    print("\nInterpretation Guide:")
    print("  • t=0: Random initialization → schemas scattered")
    print("  • t=100-500: Learning phase → clusters emerge")
    print("  • t=1000: Convergence → distinct regime islands")
    print("  • Distortion heatmap: Blue points = faithful projection")
    print("  • Trustworthiness T(k) > 0.8 = high-quality projection")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
