#!/usr/bin/env python3
"""
Visual 3: ViViT Factorized Attention - Spatiotemporal Attention Maps

This module implements the Video Vision Transformer (ViViT) factorized attention
visualization for Project JANUS, showing how spatial and temporal attention
mechanisms process GAF texture sequences.

Theoretical Foundation:
- ViViT uses factorized encoder: spatial attention → temporal attention
- Spatial encoder: Attends within single frame (learns texture patterns)
- Temporal encoder: Attends across time (learns sequence dynamics)
- Complexity reduction: O(T^2 + (H*W)^2) instead of O((T*H*W)^2)

Key Insight:
Factorization separates "what is the pattern?" (spatial) from
"how does it evolve?" (temporal), enabling real-time trading performance.

Reference:
Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., & Schmid, C. (2021).
ViViT: A Video Vision Transformer. In ICCV 2021.

Author: Project JANUS Visualization Team
License: MIT
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# Try to import torch (optional)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Using synthetic attention maps.")
    print("Install with: pip install torch")


class SpatialAttentionHead(nn.Module):
    """
    Single head of spatial self-attention.

    Attends only within a single frame (no temporal cross-attention).
    """

    def __init__(self, dim: int = 64, num_patches: int = 16):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.scale = dim**-0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weight extraction.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, num_patches, dim)
            Input tokens from a single frame

        Returns
        -------
        output : torch.Tensor, shape (batch, num_patches, dim)
            Attention-weighted output
        attention_weights : torch.Tensor, shape (batch, num_patches, num_patches)
            Attention map
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        output = attn @ v

        return output, attn


class TemporalAttentionHead(nn.Module):
    """
    Single head of temporal self-attention.

    Attends only across frames (no spatial cross-attention).
    """

    def __init__(self, dim: int = 64, num_frames: int = 8):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.scale = dim**-0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weight extraction.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, num_frames, dim)
            CLS tokens from spatial encoder

        Returns
        -------
        output : torch.Tensor, shape (batch, num_frames, dim)
            Attention-weighted output
        attention_weights : torch.Tensor, shape (batch, num_frames, num_frames)
            Temporal attention map
        """
        B, T, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        output = attn @ v

        return output, attn


def generate_synthetic_spatial_attention(
    num_patches: int = 16, pattern: str = "diagonal", seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic spatial attention map.

    Parameters
    ----------
    num_patches : int
        Number of patches (e.g., 4x4 = 16)
    pattern : str
        Attention pattern: 'diagonal', 'local', 'global', 'learned'
    seed : int
        Random seed

    Returns
    -------
    attention : np.ndarray, shape (num_patches, num_patches)
        Synthetic attention weights (row-normalized)
    """
    np.random.seed(seed)

    if pattern == "diagonal":
        # Strong diagonal (self-attention)
        attention = np.eye(num_patches) * 0.5
        # Add weak off-diagonal (context)
        attention += np.random.rand(num_patches, num_patches) * 0.1
        attention += attention.T  # Symmetrize

    elif pattern == "local":
        # Local neighborhood attention
        attention = np.zeros((num_patches, num_patches))
        patch_size = int(np.sqrt(num_patches))
        for i in range(num_patches):
            row_i = i // patch_size
            col_i = i % patch_size
            for j in range(num_patches):
                row_j = j // patch_size
                col_j = j % patch_size
                # Manhattan distance
                dist = abs(row_i - row_j) + abs(col_i - col_j)
                if dist <= 1:
                    attention[i, j] = 1.0 - dist * 0.3
                else:
                    attention[i, j] = np.random.rand() * 0.05

    elif pattern == "global":
        # Uniform global attention
        attention = np.ones((num_patches, num_patches)) * 0.8
        attention += np.random.rand(num_patches, num_patches) * 0.2

    else:  # learned
        # Realistic learned pattern with structure
        attention = np.random.rand(num_patches, num_patches)
        # Add correlation structure
        for i in range(num_patches):
            attention[i, i] = np.random.rand() * 0.5 + 0.5  # Strong self-attention
            # Smooth with neighbors
            if i > 0:
                attention[i, i - 1] = attention[i - 1, i - 1] * 0.3
            if i < num_patches - 1:
                attention[i, i + 1] = attention[i, i + 1] * 0.3

    # Row-normalize to sum to 1
    attention = attention / attention.sum(axis=1, keepdims=True)

    return attention


def generate_synthetic_temporal_attention(
    num_frames: int = 8, pattern: str = "recent_bias", seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic temporal attention map.

    Parameters
    ----------
    num_frames : int
        Number of time frames
    pattern : str
        Attention pattern: 'recent_bias', 'causal', 'uniform', 'learned'
    seed : int
        Random seed

    Returns
    -------
    attention : np.ndarray, shape (num_frames, num_frames)
        Synthetic temporal attention weights (row-normalized)
    """
    np.random.seed(seed + 1)

    if pattern == "recent_bias":
        # Exponential decay from current frame
        attention = np.zeros((num_frames, num_frames))
        for i in range(num_frames):
            for j in range(num_frames):
                # Distance in time
                dist = abs(i - j)
                attention[i, j] = np.exp(-0.3 * dist)

    elif pattern == "causal":
        # Only attend to past and present (no future)
        attention = np.tril(np.ones((num_frames, num_frames)))
        # Add recency bias
        for i in range(num_frames):
            for j in range(i + 1):
                attention[i, j] *= np.exp(-0.2 * (i - j))

    elif pattern == "uniform":
        # Equal attention to all frames
        attention = np.ones((num_frames, num_frames))

    else:  # learned
        # Realistic learned temporal pattern
        attention = np.random.rand(num_frames, num_frames)
        # Add temporal structure (attend more to recent past)
        for i in range(num_frames):
            for j in range(num_frames):
                if j <= i:  # Past or present
                    attention[i, j] += np.exp(-0.1 * (i - j)) * 0.5
                else:  # Future (small attention for bidirectional models)
                    attention[i, j] *= 0.3

    # Row-normalize
    attention = attention / attention.sum(axis=1, keepdims=True)

    return attention


def plot_spatial_attention_heatmap(
    ax: plt.Axes,
    attention: np.ndarray,
    patch_size: int = 4,
    title: str = "Spatial Attention",
):
    """
    Plot spatial attention as heatmap.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    attention : np.ndarray, shape (num_patches, num_patches)
        Attention weights
    patch_size : int
        Grid size (e.g., 4 for 4x4 patches)
    title : str
        Plot title
    """
    im = ax.imshow(
        attention, cmap="viridis", aspect="auto", vmin=0, vmax=attention.max()
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=9, fontweight="bold")

    # Grid lines to show patch structure
    for i in range(patch_size):
        ax.axhline(i * patch_size - 0.5, color="white", linewidth=0.5, alpha=0.3)
        ax.axvline(i * patch_size - 0.5, color="white", linewidth=0.5, alpha=0.3)

    # Labels
    ax.set_xlabel("Key Patch Index", fontsize=10, fontweight="bold")
    ax.set_ylabel("Query Patch Index", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Ticks at patch boundaries
    ticks = [i * patch_size + patch_size // 2 for i in range(patch_size)]
    labels = [f"P{i}" for i in range(patch_size)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)


def plot_temporal_attention_heatmap(
    ax: plt.Axes,
    attention: np.ndarray,
    num_frames: int = 8,
    title: str = "Temporal Attention",
):
    """
    Plot temporal attention as heatmap.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    attention : np.ndarray, shape (num_frames, num_frames)
        Temporal attention weights
    num_frames : int
        Number of frames
    title : str
        Plot title
    """
    im = ax.imshow(
        attention, cmap="plasma", aspect="auto", vmin=0, vmax=attention.max()
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=9, fontweight="bold")

    # Diagonal line (self-attention)
    ax.plot([0, num_frames - 1], [0, num_frames - 1], "w--", linewidth=1, alpha=0.5)

    # Labels
    ax.set_xlabel("Key Frame (Time j)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Query Frame (Time i)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Frame labels
    ax.set_xticks(range(num_frames))
    ax.set_xticklabels(
        [
            f"t-{num_frames - 1 - i}" if i < num_frames - 1 else "t"
            for i in range(num_frames)
        ],
        fontsize=8,
    )
    ax.set_yticks(range(num_frames))
    ax.set_yticklabels(
        [
            f"t-{num_frames - 1 - i}" if i < num_frames - 1 else "t"
            for i in range(num_frames)
        ],
        fontsize=8,
    )


def plot_attention_flow_diagram(
    ax: plt.Axes, num_patches: int = 16, num_frames: int = 8
):
    """
    Plot architectural flow diagram showing factorization.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    num_patches : int
        Number of spatial patches per frame
    num_frames : int
        Number of temporal frames
    """
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Title
    ax.text(
        5,
        9.5,
        "Factorized Encoder Architecture",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )

    # Input layer
    ax.add_patch(
        plt.Rectangle(
            (0.5, 7.5), 2, 1, facecolor="#E8F4F8", edgecolor="black", linewidth=2
        )
    )
    ax.text(
        1.5,
        8,
        "GAF Video\n(T×H×W)",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    # Arrow to tubelet extraction
    ax.annotate(
        "",
        xy=(1.5, 6.8),
        xytext=(1.5, 7.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    )

    # Tubelet extraction
    ax.add_patch(
        plt.Rectangle(
            (0.5, 5.8), 2, 1, facecolor="#FFE6CC", edgecolor="black", linewidth=2
        )
    )
    ax.text(1.5, 6.3, "Tubelet\nEmbedding", ha="center", va="center", fontsize=9)

    # Split into spatial encoder (left) and temporal encoder (right)
    # Arrow to spatial encoder
    ax.annotate(
        "",
        xy=(3.5, 4.8),
        xytext=(2.5, 6.3),
        arrowprops=dict(arrowstyle="->", lw=2, color="green"),
    )

    # Spatial encoder
    ax.add_patch(
        plt.Rectangle(
            (3, 3.8), 2.5, 2, facecolor="#D4EDDA", edgecolor="green", linewidth=2
        )
    )
    ax.text(
        4.25,
        5.3,
        "Spatial Encoder",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="darkgreen",
    )
    ax.text(
        4.25,
        4.8,
        f"({num_patches} patches)",
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
    )
    ax.text(
        4.25, 4.3, "Attends within\nsingle frame", ha="center", va="center", fontsize=8
    )

    # Arrow from spatial to temporal
    ax.annotate(
        "",
        xy=(7, 4.8),
        xytext=(5.5, 4.8),
        arrowprops=dict(arrowstyle="->", lw=2, color="purple"),
    )
    ax.text(6.25, 5.2, "CLS tokens", ha="center", fontsize=8, style="italic")

    # Temporal encoder
    ax.add_patch(
        plt.Rectangle(
            (6.5, 3.8), 2.5, 2, facecolor="#F8D7DA", edgecolor="red", linewidth=2
        )
    )
    ax.text(
        7.75,
        5.3,
        "Temporal Encoder",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="darkred",
    )
    ax.text(
        7.75,
        4.8,
        f"({num_frames} frames)",
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
    )
    ax.text(
        7.75, 4.3, "Attends across\ntime sequence", ha="center", va="center", fontsize=8
    )

    # Arrow to output
    ax.annotate(
        "",
        xy=(7.75, 2.8),
        xytext=(7.75, 3.8),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    )

    # Output
    ax.add_patch(
        plt.Rectangle(
            (6.75, 1.8), 2, 1, facecolor="#D1C4E9", edgecolor="black", linewidth=2
        )
    )
    ax.text(
        7.75,
        2.3,
        "Output\n(Classification)",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    # Complexity annotation
    complexity_text = (
        "Complexity Reduction:\n"
        f"Full: O(({num_frames}×{num_patches})²) = O({(num_frames * num_patches) ** 2:,})\n"
        f"Factorized: O({num_frames}² + {num_patches}²) = O({num_frames**2 + num_patches**2:,})"
    )
    ax.text(
        0.5,
        2.5,
        complexity_text,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    # Key insight box
    insight_text = (
        "Key Insight:\nSpatial → What is the pattern?\nTemporal → How does it evolve?"
    )
    ax.text(
        9.5,
        2.5,
        insight_text,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )


def visualize_vivit_attention(
    num_patches: int = 16,
    num_frames: int = 8,
    spatial_pattern: str = "learned",
    temporal_pattern: str = "recent_bias",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Create complete ViViT factorized attention visualization (V3).

    Parameters
    ----------
    num_patches : int
        Number of spatial patches (e.g., 4x4 = 16)
    num_frames : int
        Number of temporal frames
    spatial_pattern : str
        Spatial attention pattern
    temporal_pattern : str
        Temporal attention pattern
    save_path : Path, optional
        Output path for PNG
    dpi : int
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.Figure
        The complete figure
    """
    # Generate attention maps
    spatial_attn = generate_synthetic_spatial_attention(
        num_patches, pattern=spatial_pattern
    )
    temporal_attn = generate_synthetic_temporal_attention(
        num_frames, pattern=temporal_pattern
    )

    # Create figure with custom grid
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Row 1: Architecture diagram (spans 2 columns)
    ax_arch = fig.add_subplot(gs[0, :2])
    plot_attention_flow_diagram(ax_arch, num_patches, num_frames)

    # Row 1, Col 3: Spatial attention example
    ax_spatial_example = fig.add_subplot(gs[0, 2])
    patch_size = int(np.sqrt(num_patches))
    plot_spatial_attention_heatmap(
        ax_spatial_example,
        spatial_attn,
        patch_size=patch_size,
        title="Panel A: Spatial Attention\n(Within Frame)",
    )

    # Row 2, Col 1: Spatial attention (full)
    ax_spatial = fig.add_subplot(gs[1, 0])
    plot_spatial_attention_heatmap(
        ax_spatial,
        spatial_attn,
        patch_size=patch_size,
        title="Spatial Attention Detail",
    )

    # Row 2, Col 2: Temporal attention
    ax_temporal = fig.add_subplot(gs[1, 1])
    plot_temporal_attention_heatmap(
        ax_temporal,
        temporal_attn,
        num_frames=num_frames,
        title="Panel B: Temporal Attention\n(Across Frames)",
    )

    # Row 2, Col 3: Attention statistics
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.axis("off")

    # Compute statistics
    spatial_entropy = -np.sum(
        spatial_attn * np.log(spatial_attn + 1e-10), axis=1
    ).mean()
    temporal_entropy = -np.sum(
        temporal_attn * np.log(temporal_attn + 1e-10), axis=1
    ).mean()

    spatial_max_attn = spatial_attn.max(axis=1).mean()
    temporal_max_attn = temporal_attn.max(axis=1).mean()

    stats_text = (
        "Attention Statistics\n"
        "=" * 30 + "\n\n"
        "Spatial Encoder:\n"
        f"  Entropy: {spatial_entropy:.3f}\n"
        f"  Max Attn: {spatial_max_attn:.3f}\n"
        f"  Patches: {num_patches}\n"
        f"  Pattern: {spatial_pattern}\n\n"
        "Temporal Encoder:\n"
        f"  Entropy: {temporal_entropy:.3f}\n"
        f"  Max Attn: {temporal_max_attn:.3f}\n"
        f"  Frames: {num_frames}\n"
        f"  Pattern: {temporal_pattern}\n\n"
        "Interpretation:\n"
        "• Low entropy → Focused attention\n"
        "• High entropy → Distributed attention\n"
        "• Max attn → Peak focus strength"
    )

    ax_stats.text(
        0.1,
        0.95,
        stats_text,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # Overall title
    method_name = "ViViT" if not TORCH_AVAILABLE else "ViViT (PyTorch)"
    fig.suptitle(
        f"Visual 3: {method_name} Factorized Attention - Spatiotemporal Learning",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Footer
    footer_text = (
        f"Factorization: Spatial ({patch_size}×{patch_size} patches) → Temporal ({num_frames} frames) | "
        f"Complexity: O({num_frames}² + {num_patches}²) = {num_frames**2 + num_patches**2:,} ops"
    )
    fig.text(
        0.5,
        0.01,
        footer_text,
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
        description="Visual 3: ViViT Factorized Attention Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visual_3_vivit_attention.py --show
  python visual_3_vivit_attention.py --save-all --output-dir ../outputs
  python visual_3_vivit_attention.py --patches 25 --frames 10 --show
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
        "--patches",
        type=int,
        default=16,
        help="Number of spatial patches (must be perfect square)",
    )
    parser.add_argument(
        "--frames", type=int, default=8, help="Number of temporal frames"
    )
    parser.add_argument(
        "--spatial-pattern",
        type=str,
        default="learned",
        choices=["diagonal", "local", "global", "learned"],
        help="Spatial attention pattern",
    )
    parser.add_argument(
        "--temporal-pattern",
        type=str,
        default="recent_bias",
        choices=["recent_bias", "causal", "uniform", "learned"],
        help="Temporal attention pattern",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for saved figures"
    )

    args = parser.parse_args()

    # Validate patches is perfect square
    patch_size = int(np.sqrt(args.patches))
    if patch_size * patch_size != args.patches:
        print(f"Error: --patches must be a perfect square (e.g., 16, 25, 36)")
        print(f"  Suggested: {patch_size**2} or {(patch_size + 1) ** 2}")
        return

    print("=" * 70)
    print("Visual 3: ViViT Factorized Attention - Spatiotemporal Learning")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Spatial patches: {args.patches} ({patch_size}×{patch_size})")
    print(f"  Temporal frames: {args.frames}")
    print(f"  Spatial pattern: {args.spatial_pattern}")
    print(f"  Temporal pattern: {args.temporal_pattern}")

    if not TORCH_AVAILABLE:
        print("\n⚠ Warning: PyTorch not installed, using synthetic attention maps")

    # Create visualization
    print(f"\n[1/1] Creating ViViT factorized attention visualization...")

    save_path = None
    if args.save_all:
        save_path = args.output_dir / "visual_3_vivit_attention.png"

    fig = visualize_vivit_attention(
        num_patches=args.patches,
        num_frames=args.frames,
        spatial_pattern=args.spatial_pattern,
        temporal_pattern=args.temporal_pattern,
        save_path=save_path,
        dpi=args.dpi,
    )

    print(f"\n{'=' * 70}")
    print("✓ Visualization complete!")
    print(f"{'=' * 70}")

    if args.save_all:
        print(f"\nOutput saved to: {args.output_dir}/")

    print("\nInterpretation:")
    print("  • Top row: Architecture showing factorization flow")
    print("  • Bottom left: Spatial attention (within-frame patterns)")
    print("  • Bottom center: Temporal attention (across-frame sequences)")
    print("  • Bottom right: Attention statistics and metrics")
    print("\nKey Insight:")
    print("  Factorization reduces complexity from O(T²×P²) to O(T² + P²),")
    print("  enabling real-time processing while preserving spatiotemporal")
    print("  understanding. Spatial learns 'what', temporal learns 'when'.")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
