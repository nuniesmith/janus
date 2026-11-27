#!/usr/bin/env python3
"""
Visual 8: Mahalanobis Ellipsoid - Correlation-Aware Anomaly Detection

This module implements the Mahalanobis distance visualization for Project JANUS's
Amygdala anomaly detection system, demonstrating why Euclidean distance is
insufficient for financial risk assessment.

Theoretical Foundation:
- Mahalanobis distance accounts for covariance structure between features
- Euclidean distance treats all dimensions equally (fails for correlated data)
- Financial markets exhibit strong feature correlations (spread vs volatility)
- Ellipsoid geometry reveals natural correlation-aligned decision boundaries

Key Insight:
Euclidean distance: "How far in straight line?"
Mahalanobis distance: "How many standard deviations along correlation structure?"

Reference:
Mahalanobis, P. C. (1936). On the generalized distance in statistics.
Proceedings of the National Institute of Sciences of India, 2(1), 49-55.

Author: Project JANUS Visualization Team
License: MIT
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import linalg
from scipy.stats import chi2

warnings.filterwarnings("ignore")


def generate_correlated_data(
    n_samples: int = 500,
    mean: np.ndarray = np.array([0.0, 0.0]),
    covariance: np.ndarray = np.array([[1.0, 0.7], [0.7, 1.0]]),
    seed: int = 42,
) -> np.ndarray:
    """
    Generate correlated 2D data from multivariate normal distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    mean : np.ndarray, shape (2,)
        Mean vector
    covariance : np.ndarray, shape (2, 2)
        Covariance matrix
    seed : int
        Random seed

    Returns
    -------
    data : np.ndarray, shape (n_samples, 2)
        Generated correlated data
    """
    np.random.seed(seed)
    data = np.random.multivariate_normal(mean, covariance, n_samples)
    return data


def inject_anomalies(
    data: np.ndarray, n_anomalies: int = 20, anomaly_type: str = "euclidean_false"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject synthetic anomalies into data.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, 2)
        Normal data
    n_anomalies : int
        Number of anomalies to inject
    anomaly_type : str
        Type of anomaly: 'euclidean_false', 'true_outlier', 'mixed'

    Returns
    -------
    data_with_anomalies : np.ndarray, shape (n_samples + n_anomalies, 2)
        Data with anomalies
    labels : np.ndarray, shape (n_samples + n_anomalies,)
        0 = normal, 1 = anomaly
    """
    np.random.seed(43)
    anomalies = []

    if anomaly_type == "euclidean_false":
        # Points close to origin in Euclidean sense but violating correlation
        # Example: Low spread + high volatility (unusual combination)
        for _ in range(n_anomalies):
            # Generate points perpendicular to correlation direction
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.5, 1.5)
            # Rotate to be perpendicular to main correlation axis
            x = radius * np.cos(angle + np.pi / 4)
            y = radius * np.sin(angle + np.pi / 4)
            anomalies.append([x, y])

    elif anomaly_type == "true_outlier":
        # Points far from center in all directions
        for _ in range(n_anomalies):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(3, 5)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            anomalies.append([x, y])

    else:  # mixed
        # Half euclidean-false, half true outliers
        for i in range(n_anomalies):
            if i < n_anomalies // 2:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0.5, 1.5)
                x = radius * np.cos(angle + np.pi / 4)
                y = radius * np.sin(angle + np.pi / 4)
            else:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(3, 5)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
            anomalies.append([x, y])

    anomalies = np.array(anomalies)
    data_with_anomalies = np.vstack([data, anomalies])
    labels = np.hstack([np.zeros(len(data)), np.ones(len(anomalies))])

    return data_with_anomalies, labels


def compute_mahalanobis_distance(
    data: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """
    Compute Mahalanobis distance for each point.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data points
    mean : np.ndarray, shape (n_features,)
        Distribution mean
    cov : np.ndarray, shape (n_features, n_features)
        Covariance matrix

    Returns
    -------
    distances : np.ndarray, shape (n_samples,)
        Mahalanobis distance for each point
    """
    # Add regularization for numerical stability
    cov_reg = cov + np.eye(cov.shape[0]) * 1e-6

    # Compute inverse covariance
    cov_inv = linalg.inv(cov_reg)

    # Compute distances
    diff = data - mean
    distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    return distances


def compute_euclidean_distance(data: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance for each point.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data points
    mean : np.ndarray, shape (n_features,)
        Distribution mean

    Returns
    -------
    distances : np.ndarray, shape (n_samples,)
        Euclidean distance for each point
    """
    diff = data - mean
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return distances


def get_ellipse_params(
    mean: np.ndarray, cov: np.ndarray, n_std: float = 2.0
) -> Tuple[float, float, float]:
    """
    Compute ellipse parameters from covariance matrix.

    Parameters
    ----------
    mean : np.ndarray, shape (2,)
        Ellipse center
    cov : np.ndarray, shape (2, 2)
        Covariance matrix
    n_std : float
        Number of standard deviations (confidence level)

    Returns
    -------
    width : float
        Ellipse width (2 * semi-major axis)
    height : float
        Ellipse height (2 * semi-minor axis)
    angle : float
        Rotation angle in degrees
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Compute ellipse parameters
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    return width, height, angle


def plot_comparison(
    ax: plt.Axes,
    data: np.ndarray,
    labels: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    distance_type: str = "mahalanobis",
    threshold: float = 2.0,
):
    """
    Plot either Euclidean or Mahalanobis distance comparison.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    data : np.ndarray
        Data points
    labels : np.ndarray
        True labels (0 = normal, 1 = anomaly)
    mean : np.ndarray
        Distribution mean
    cov : np.ndarray
        Covariance matrix
    distance_type : str
        'euclidean' or 'mahalanobis'
    threshold : float
        Distance threshold for anomaly detection
    """
    # Compute distances
    if distance_type == "mahalanobis":
        distances = compute_mahalanobis_distance(data, mean, cov)
    else:
        distances = compute_euclidean_distance(data, mean)

    # Classify based on threshold
    predicted_anomaly = distances > threshold

    # Plot normal points
    normal_mask = labels == 0
    ax.scatter(
        data[normal_mask, 0],
        data[normal_mask, 1],
        c="lightblue",
        s=30,
        alpha=0.6,
        edgecolors="steelblue",
        linewidths=0.5,
        label="Normal Data",
    )

    # Plot true anomalies
    anomaly_mask = labels == 1
    ax.scatter(
        data[anomaly_mask, 0],
        data[anomaly_mask, 1],
        c="orange",
        s=60,
        marker="^",
        alpha=0.9,
        edgecolors="darkorange",
        linewidths=1,
        label="True Anomaly",
    )

    # Highlight detection results
    # True positives (correctly detected anomalies)
    tp_mask = anomaly_mask & predicted_anomaly
    if tp_mask.any():
        ax.scatter(
            data[tp_mask, 0],
            data[tp_mask, 1],
            s=200,
            facecolors="none",
            edgecolors="green",
            linewidths=2,
            label="Detected (TP)",
        )

    # False positives (normal flagged as anomaly)
    fp_mask = normal_mask & predicted_anomaly
    if fp_mask.any():
        ax.scatter(
            data[fp_mask, 0],
            data[fp_mask, 1],
            s=200,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="False Alarm (FP)",
        )

    # False negatives (anomaly missed)
    fn_mask = anomaly_mask & ~predicted_anomaly
    if fn_mask.any():
        ax.scatter(
            data[fn_mask, 0],
            data[fn_mask, 1],
            s=200,
            marker="x",
            c="red",
            linewidths=2,
            label="Missed (FN)",
        )

    # Plot decision boundary
    if distance_type == "euclidean":
        # Circle
        circle = plt.Circle(
            mean,
            threshold,
            fill=False,
            edgecolor="purple",
            linewidth=2,
            linestyle="--",
            label="Decision Boundary",
        )
        ax.add_patch(circle)
    else:
        # Ellipse
        width, height, angle = get_ellipse_params(mean, cov, n_std=threshold)
        ellipse = Ellipse(
            mean,
            width,
            height,
            angle=angle,
            fill=False,
            edgecolor="green",
            linewidth=2,
            linestyle="--",
            label="Decision Boundary",
        )
        ax.add_patch(ellipse)

    # Plot mean
    ax.scatter(
        mean[0],
        mean[1],
        c="black",
        s=100,
        marker="X",
        edgecolors="white",
        linewidths=1,
        zorder=10,
        label="Mean",
    )

    # Compute metrics
    tp = tp_mask.sum()
    fp = fp_mask.sum()
    fn = fn_mask.sum()
    tn = (normal_mask & ~predicted_anomaly).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Title with metrics
    title = f"Panel {'B' if distance_type == 'mahalanobis' else 'A'}: "
    title += f"{distance_type.title()} Distance"
    if distance_type == "euclidean":
        title += " (Fallacy)"
    else:
        title += " (Insight)"

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    # Metrics box
    metrics_text = (
        f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}"
    )
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    # Labels
    ax.set_xlabel("Feature 1 (e.g., Spread)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Feature 2 (e.g., Volatility)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Set consistent axis limits
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)


def visualize_mahalanobis_ellipsoid(
    n_samples: int = 500,
    n_anomalies: int = 20,
    correlation: float = 0.7,
    threshold: float = 2.5,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Create complete Mahalanobis ellipsoid visualization (V8).

    Parameters
    ----------
    n_samples : int
        Number of normal samples
    n_anomalies : int
        Number of anomalies to inject
    correlation : float
        Correlation coefficient between features
    threshold : float
        Distance threshold (in standard deviations)
    save_path : Path, optional
        Output path for PNG
    dpi : int
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.Figure
        The complete figure
    """
    # Generate correlated data
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, correlation], [correlation, 1.0]])

    data = generate_correlated_data(n_samples, mean, cov)

    # Inject anomalies
    data_with_anomalies, labels = inject_anomalies(data, n_anomalies, "mixed")

    # Create figure
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    ax_euclidean = fig.add_subplot(gs[0, 0])
    ax_mahalanobis = fig.add_subplot(gs[0, 1])

    # Panel A: Euclidean distance
    plot_comparison(
        ax_euclidean,
        data_with_anomalies,
        labels,
        mean,
        cov,
        distance_type="euclidean",
        threshold=threshold,
    )

    # Panel B: Mahalanobis distance
    plot_comparison(
        ax_mahalanobis,
        data_with_anomalies,
        labels,
        mean,
        cov,
        distance_type="mahalanobis",
        threshold=threshold,
    )

    # Overall title
    fig.suptitle(
        "Visual 8: Mahalanobis Ellipsoid - Correlation-Aware Anomaly Detection",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Formula annotation
    formula_text = (
        f"Mahalanobis Distance: $D_M(\\mathbf{{s}}_t) = \\sqrt{{(\\mathbf{{s}}_t - \\boldsymbol{{\\mu}})^T \\boldsymbol{{\\Sigma}}^{{-1}} (\\mathbf{{s}}_t - \\boldsymbol{{\\mu}})}}$\n"
        f"Correlation: ρ = {correlation:.2f} | Threshold: {threshold:.1f}σ | Samples: {n_samples} + {n_anomalies} anomalies"
    )
    fig.text(
        0.5,
        0.01,
        formula_text,
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
        description="Visual 8: Mahalanobis Ellipsoid Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visual_8_mahalanobis.py --show
  python visual_8_mahalanobis.py --save-all --output-dir ../outputs
  python visual_8_mahalanobis.py --correlation 0.9 --threshold 3.0 --show
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
        "--samples", type=int, default=500, help="Number of normal samples"
    )
    parser.add_argument(
        "--anomalies", type=int, default=20, help="Number of anomalies to inject"
    )
    parser.add_argument(
        "--correlation",
        type=float,
        default=0.7,
        help="Correlation coefficient (0-1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.5,
        help="Detection threshold (standard deviations)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for saved figures"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Visual 8: Mahalanobis Ellipsoid - Correlation-Aware Anomaly Detection")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Normal samples: {args.samples}")
    print(f"  Anomalies: {args.anomalies}")
    print(f"  Correlation: {args.correlation:.2f}")
    print(f"  Threshold: {args.threshold:.1f}σ")

    # Create visualization
    print(f"\n[1/1] Creating Mahalanobis vs Euclidean comparison...")

    save_path = None
    if args.save_all:
        save_path = args.output_dir / "visual_8_mahalanobis.png"

    fig = visualize_mahalanobis_ellipsoid(
        n_samples=args.samples,
        n_anomalies=args.anomalies,
        correlation=args.correlation,
        threshold=args.threshold,
        save_path=save_path,
        dpi=args.dpi,
    )

    print(f"\n{'=' * 70}")
    print("✓ Visualization complete!")
    print(f"{'=' * 70}")

    if args.save_all:
        print(f"\nOutput saved to: {args.output_dir}/")

    print("\nInterpretation:")
    print("  • Panel A (Euclidean): Circular boundary ignores correlation")
    print("  • Panel B (Mahalanobis): Ellipse aligns with data structure")
    print("  • Green circles: True positives (correctly detected)")
    print("  • Red circles: False positives (false alarms)")
    print("  • Red X: False negatives (missed anomalies)")
    print("\nKey Insight:")
    print("  Mahalanobis distance understands that widening spread during")
    print("  high volatility is NORMAL, but widening spread during low")
    print("  volatility is a CRISIS signal. Euclidean distance cannot.")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
