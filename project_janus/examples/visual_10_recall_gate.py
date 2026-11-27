#!/usr/bin/env python3
"""
Visual 10: Recall Gate Comparator

Visualizes the gated consolidation mechanism that controls whether new memories
are integrated into neocortical schemas based on consistency with existing knowledge.

This visualization shows:
- New memory vs. reconstructed memory comparison
- Cosine similarity gating threshold
- Accept/reject decisions based on similarity
- Protection against catastrophic forgetting and noise overfitting

Author: Project JANUS Visualization Team
Date: December 2024
License: MIT
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Color palette (WCAG 2.1 AA compliant, colorblind-safe)
COLORS = {
    "new_memory": "#2E8B57",  # Sea green (new memory)
    "reconstructed": "#4169E1",  # Royal blue (reconstructed)
    "accept": "#32CD32",  # Lime green (accepted)
    "reject": "#DC143C",  # Crimson (rejected)
    "gate": "#FF8C00",  # Dark orange (gate mechanism)
    "threshold": "#9370DB",  # Medium purple (threshold)
    "edge": "#696969",  # Dim gray (connections)
    "background": "#F5F5F5",  # White smoke
    "grid": "#E0E0E0",  # Light gray
    "text": "#2F2F2F",  # Dark gray
}


class RecallGateComparator:
    """
    Recall-gated consolidation mechanism visualization.

    The visualization demonstrates how:
    1. New memories from hippocampus are compared to schema reconstructions
    2. Cosine similarity determines gate activation
    3. High similarity (consistent) → accept and update
    4. Low similarity (noise/outlier) → reject and discard
    5. Prevents catastrophic forgetting
    """

    def __init__(
        self,
        num_samples: int = 100,
        embedding_dim: int = 64,
        similarity_threshold: float = 0.7,
        random_seed: int = 42,
    ):
        """
        Initialize the recall gate comparator.

        Args:
            num_samples: Number of memory consolidation attempts
            embedding_dim: Dimensionality of memory embeddings
            similarity_threshold: Cosine similarity threshold for acceptance
            random_seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        self.threshold = similarity_threshold

        np.random.seed(random_seed)

        # Generate synthetic memory comparison data
        self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic memory comparison data."""
        # Generate new memories (from hippocampus)
        self.new_memories = np.random.randn(self.num_samples, self.embedding_dim)
        self.new_memories /= np.linalg.norm(
            self.new_memories, axis=1, keepdims=True
        )  # Normalize

        # Generate reconstructed memories (from neocortical schema)
        # Most reconstructions are similar to new memories (valid learning)
        # Some are dissimilar (noise or out-of-distribution)
        self.reconstructed_memories = self.new_memories.copy()

        # Add noise to reconstructions
        noise = np.random.randn(self.num_samples, self.embedding_dim) * 0.3
        self.reconstructed_memories += noise

        # Add some completely dissimilar memories (outliers)
        outlier_indices = np.random.choice(self.num_samples, size=15, replace=False)
        self.reconstructed_memories[outlier_indices] = np.random.randn(
            15, self.embedding_dim
        )

        # Normalize reconstructed memories
        self.reconstructed_memories /= np.linalg.norm(
            self.reconstructed_memories, axis=1, keepdims=True
        )

        # Compute cosine similarities
        self.similarities = np.sum(
            self.new_memories * self.reconstructed_memories, axis=1
        )

        # Gate decisions (accept if similarity > threshold)
        self.gate_open = self.similarities >= self.threshold

        # Compute update magnitudes (only for accepted memories)
        self.update_magnitudes = np.zeros(self.num_samples)
        self.update_magnitudes[self.gate_open] = 1.0 - self.similarities[self.gate_open]

        # Schema update strength over time
        self.schema_strength = np.cumsum(self.gate_open.astype(float)) / np.arange(
            1, self.num_samples + 1
        )

        # Generate example embeddings for visualization
        self._generate_example_embeddings()

    def _generate_example_embeddings(self) -> None:
        """Generate example embeddings for detailed comparison."""
        # Select representative examples
        accepted_idx = np.where(self.gate_open)[0][0] if any(self.gate_open) else 0
        rejected_idx = np.where(~self.gate_open)[0][0] if any(~self.gate_open) else 1

        self.example_accepted = {
            "new": self.new_memories[accepted_idx],
            "reconstructed": self.reconstructed_memories[accepted_idx],
            "similarity": self.similarities[accepted_idx],
            "index": accepted_idx,
        }

        self.example_rejected = {
            "new": self.new_memories[rejected_idx],
            "reconstructed": self.reconstructed_memories[rejected_idx],
            "similarity": self.similarities[rejected_idx],
            "index": rejected_idx,
        }

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def create_visualization(self) -> plt.Figure:
        """
        Create the complete recall gate comparator visualization.

        Returns:
            Matplotlib figure object
        """
        # Create figure with 4 panels
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.2, 1],
            height_ratios=[1, 1],
            hspace=0.35,
            wspace=0.3,
            left=0.08,
            right=0.95,
        )

        # Panel A: Gate comparator diagram
        ax_diagram = fig.add_subplot(gs[0, 0])
        self._plot_comparator_diagram(ax_diagram)

        # Panel B: Similarity distribution
        ax_dist = fig.add_subplot(gs[0, 1])
        self._plot_similarity_distribution(ax_dist)

        # Panel C: Decision timeline
        ax_timeline = fig.add_subplot(gs[1, 0])
        self._plot_decision_timeline(ax_timeline)

        # Panel D: Schema strength evolution
        ax_strength = fig.add_subplot(gs[1, 1])
        self._plot_schema_strength(ax_strength)

        # Overall title
        fig.suptitle(
            "Visual 10: Recall Gate Comparator\n"
            "Consistency-Based Memory Consolidation with Catastrophic Forgetting Prevention",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        return fig

    def _plot_comparator_diagram(self, ax: plt.Axes) -> None:
        """Plot the gate comparator block diagram."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis("off")
        ax.set_facecolor(COLORS["background"])

        # Title
        ax.text(
            5,
            7.8,
            "A. Recall Gate Comparator Architecture",
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=COLORS["text"],
        )

        # ===== INPUT: NEW MEMORY =====
        new_mem_x, new_mem_y = 1.5, 6.0
        new_box = FancyBboxPatch(
            (new_mem_x - 0.6, new_mem_y - 0.4),
            1.2,
            0.8,
            boxstyle="round,pad=0.08",
            facecolor=COLORS["new_memory"],
            edgecolor="white",
            linewidth=2.5,
            alpha=0.85,
        )
        ax.add_patch(new_box)
        ax.text(
            new_mem_x,
            new_mem_y,
            "New Memory\n(Hippocampus)",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )
        ax.text(
            new_mem_x,
            new_mem_y - 0.7,
            "h_new ∈ ℝ^d",
            ha="center",
            va="top",
            fontsize=8,
            family="monospace",
            style="italic",
            color=COLORS["text"],
        )

        # ===== SCHEMA RECONSTRUCTION =====
        recon_x, recon_y = 1.5, 4.0
        recon_box = FancyBboxPatch(
            (recon_x - 0.6, recon_y - 0.4),
            1.2,
            0.8,
            boxstyle="round,pad=0.08",
            facecolor=COLORS["reconstructed"],
            edgecolor="white",
            linewidth=2.5,
            alpha=0.85,
        )
        ax.add_patch(recon_box)
        ax.text(
            recon_x,
            recon_y,
            "Reconstructed\n(Schema)",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )
        ax.text(
            recon_x,
            recon_y - 0.7,
            "h_recon = Decode(z)",
            ha="center",
            va="top",
            fontsize=8,
            family="monospace",
            style="italic",
            color=COLORS["text"],
        )

        # ===== SIMILARITY COMPARATOR =====
        comp_x, comp_y = 4.5, 5.0
        comp_circle = Circle(
            (comp_x, comp_y),
            0.5,
            facecolor=COLORS["gate"],
            edgecolor="white",
            linewidth=3,
            alpha=0.9,
        )
        ax.add_patch(comp_circle)
        ax.text(
            comp_x,
            comp_y + 0.1,
            "Cosine",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
        ax.text(
            comp_x,
            comp_y - 0.15,
            "Similarity",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

        # Formula below
        ax.text(
            comp_x,
            comp_y - 0.85,
            "sim = (h_new · h_recon) /\n      (||h_new|| ||h_recon||)",
            ha="center",
            va="top",
            fontsize=8,
            family="monospace",
            color=COLORS["text"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.95),
        )

        # ===== THRESHOLD GATE =====
        gate_x, gate_y = 7.0, 5.0
        gate_box = FancyBboxPatch(
            (gate_x - 0.5, gate_y - 0.4),
            1.0,
            0.8,
            boxstyle="round,pad=0.08",
            facecolor=COLORS["threshold"],
            edgecolor="white",
            linewidth=2.5,
            alpha=0.85,
        )
        ax.add_patch(gate_box)
        ax.text(
            gate_x,
            gate_y + 0.1,
            "Gate",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )
        ax.text(
            gate_x,
            gate_y - 0.15,
            f"θ={self.threshold}",
            ha="center",
            va="center",
            fontsize=9,
            family="monospace",
            color="white",
        )

        # ===== DECISION OUTPUTS =====
        # Accept path
        accept_x, accept_y = 8.5, 6.5
        accept_box = FancyBboxPatch(
            (accept_x - 0.5, accept_y - 0.3),
            1.0,
            0.6,
            boxstyle="round,pad=0.08",
            facecolor=COLORS["accept"],
            edgecolor="white",
            linewidth=2,
            alpha=0.85,
        )
        ax.add_patch(accept_box)
        ax.text(
            accept_x,
            accept_y,
            "✓ ACCEPT\nUpdate Schema",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

        # Reject path
        reject_x, reject_y = 8.5, 3.5
        reject_box = FancyBboxPatch(
            (reject_x - 0.5, reject_y - 0.3),
            1.0,
            0.6,
            boxstyle="round,pad=0.08",
            facecolor=COLORS["reject"],
            edgecolor="white",
            linewidth=2,
            alpha=0.85,
        )
        ax.add_patch(reject_box)
        ax.text(
            reject_x,
            reject_y,
            "✗ REJECT\nDiscard",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

        # ===== CONNECTIONS =====
        # New memory → Comparator
        arrow1 = FancyArrowPatch(
            (new_mem_x + 0.65, new_mem_y - 0.2),
            (comp_x - 0.5, comp_y + 0.3),
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2.5,
            color=COLORS["new_memory"],
            alpha=0.7,
        )
        ax.add_patch(arrow1)

        # Reconstructed → Comparator
        arrow2 = FancyArrowPatch(
            (recon_x + 0.65, recon_y + 0.2),
            (comp_x - 0.5, comp_y - 0.3),
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2.5,
            color=COLORS["reconstructed"],
            alpha=0.7,
        )
        ax.add_patch(arrow2)

        # Comparator → Gate
        arrow3 = FancyArrowPatch(
            (comp_x + 0.5, comp_y),
            (gate_x - 0.55, gate_y),
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2.5,
            color=COLORS["gate"],
            alpha=0.7,
        )
        ax.add_patch(arrow3)

        # Gate → Accept (if sim >= θ)
        arrow4 = FancyArrowPatch(
            (gate_x + 0.5, gate_y + 0.3),
            (accept_x - 0.5, accept_y),
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2.5,
            color=COLORS["accept"],
            alpha=0.7,
        )
        ax.add_patch(arrow4)
        ax.text(
            (gate_x + accept_x) / 2,
            (gate_y + accept_y) / 2 + 0.3,
            "sim ≥ θ",
            fontsize=8,
            fontweight="bold",
            color=COLORS["accept"],
        )

        # Gate → Reject (if sim < θ)
        arrow5 = FancyArrowPatch(
            (gate_x + 0.5, gate_y - 0.3),
            (reject_x - 0.5, reject_y),
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2.5,
            color=COLORS["reject"],
            alpha=0.7,
        )
        ax.add_patch(arrow5)
        ax.text(
            (gate_x + reject_x) / 2,
            (gate_y + reject_y) / 2 - 0.3,
            "sim < θ",
            fontsize=8,
            fontweight="bold",
            color=COLORS["reject"],
        )

        # ===== UPDATE EQUATION =====
        ax.text(
            5,
            1.5,
            "Schema Update Rule:",
            fontsize=11,
            fontweight="bold",
            color=COLORS["text"],
        )
        ax.text(
            5,
            0.9,
            "z ← z + η · 𝟙[sim ≥ θ] · (h_new - z)",
            ha="center",
            fontsize=10,
            family="monospace",
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=COLORS["accept"],
                edgecolor="white",
                linewidth=2,
                alpha=0.3,
            ),
        )

        # ===== KEY INSIGHTS =====
        insights = [
            "Prevents catastrophic forgetting",
            "Filters out noise and outliers",
            "Maintains schema consistency",
        ]
        insight_text = "Key Benefits:\n" + "\n".join(f"• {i}" for i in insights)
        ax.text(
            0.5,
            2.5,
            insight_text,
            fontsize=9,
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=COLORS["threshold"],
                linewidth=2,
                alpha=0.95,
            ),
        )

    def _plot_similarity_distribution(self, ax: plt.Axes) -> None:
        """Plot distribution of similarity scores."""
        ax.set_facecolor(COLORS["background"])
        ax.set_title(
            "B. Similarity Score Distribution", fontsize=13, fontweight="bold", pad=15
        )

        # Histogram of similarities
        bins = np.linspace(-1, 1, 30)
        counts_accept, _, _ = ax.hist(
            self.similarities[self.gate_open],
            bins=bins,
            color=COLORS["accept"],
            alpha=0.7,
            label="Accepted",
            edgecolor="white",
            linewidth=1.5,
        )
        counts_reject, _, _ = ax.hist(
            self.similarities[~self.gate_open],
            bins=bins,
            color=COLORS["reject"],
            alpha=0.7,
            label="Rejected",
            edgecolor="white",
            linewidth=1.5,
        )

        # Threshold line
        ax.axvline(
            x=self.threshold,
            color=COLORS["threshold"],
            linestyle="--",
            linewidth=3,
            label=f"Threshold θ={self.threshold}",
            alpha=0.9,
        )

        # Shade regions
        ax.axvspan(-1, self.threshold, alpha=0.1, color=COLORS["reject"])
        ax.axvspan(self.threshold, 1, alpha=0.1, color=COLORS["accept"])

        ax.set_xlabel("Cosine Similarity", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count", fontsize=11, fontweight="bold")
        ax.set_xlim(-1, 1)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

        # Statistics
        accept_rate = self.gate_open.mean() * 100
        stats_text = (
            f"Acceptance Rate: {accept_rate:.1f}%\n"
            f"Mean (Accept): {self.similarities[self.gate_open].mean():.3f}\n"
            f"Mean (Reject): {self.similarities[~self.gate_open].mean():.3f}"
        )
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            family="monospace",
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=COLORS["text"],
                linewidth=1.5,
                alpha=0.95,
            ),
        )

    def _plot_decision_timeline(self, ax: plt.Axes) -> None:
        """Plot accept/reject decisions over time."""
        ax.set_facecolor(COLORS["background"])
        ax.set_title(
            "C. Gate Decision Timeline", fontsize=13, fontweight="bold", pad=15
        )

        t = np.arange(self.num_samples)

        # Plot similarity scores
        ax.plot(
            t,
            self.similarities,
            linewidth=1.5,
            color=COLORS["gate"],
            alpha=0.6,
            label="Similarity Score",
        )

        # Color-code points by decision
        accept_mask = self.gate_open
        reject_mask = ~self.gate_open

        ax.scatter(
            t[accept_mask],
            self.similarities[accept_mask],
            c=COLORS["accept"],
            s=30,
            alpha=0.8,
            label="Accepted",
            edgecolors="white",
            linewidths=0.5,
        )
        ax.scatter(
            t[reject_mask],
            self.similarities[reject_mask],
            c=COLORS["reject"],
            s=30,
            alpha=0.8,
            label="Rejected",
            edgecolors="white",
            linewidths=0.5,
        )

        # Threshold line
        ax.axhline(
            y=self.threshold,
            color=COLORS["threshold"],
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            label=f"Threshold θ={self.threshold}",
        )

        # Shade accept/reject regions
        ax.axhspan(-1, self.threshold, alpha=0.1, color=COLORS["reject"])
        ax.axhspan(self.threshold, 1, alpha=0.1, color=COLORS["accept"])

        ax.set_xlabel("Memory Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cosine Similarity", fontsize=11, fontweight="bold")
        ax.set_xlim(0, self.num_samples - 1)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95, ncol=2)

    def _plot_schema_strength(self, ax: plt.Axes) -> None:
        """Plot schema consolidation strength over time."""
        ax.set_facecolor(COLORS["background"])
        ax.set_title(
            "D. Schema Consolidation Strength", fontsize=13, fontweight="bold", pad=15
        )

        t = np.arange(self.num_samples)

        # Plot cumulative acceptance rate
        ax.plot(
            t,
            self.schema_strength,
            linewidth=2.5,
            color=COLORS["reconstructed"],
            label="Consolidation Strength",
            alpha=0.9,
        )

        # Fill area
        ax.fill_between(
            t, 0, self.schema_strength, color=COLORS["reconstructed"], alpha=0.2
        )

        # Target strength line
        target_strength = 0.8
        ax.axhline(
            y=target_strength,
            color=COLORS["threshold"],
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label=f"Target Strength ({target_strength})",
        )

        # Convergence point
        if (self.schema_strength >= target_strength).any():
            convergence_idx = np.where(self.schema_strength >= target_strength)[0][0]
            ax.plot(
                convergence_idx,
                self.schema_strength[convergence_idx],
                "o",
                color=COLORS["accept"],
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=2,
                label=f"Convergence (t={convergence_idx})",
            )

        ax.set_xlabel("Memory Index", fontsize=11, fontweight="bold")
        ax.set_ylabel(
            "Strength (Cumulative Accept Rate)", fontsize=11, fontweight="bold"
        )
        ax.set_xlim(0, self.num_samples - 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

        # Final strength annotation
        final_strength = self.schema_strength[-1]
        ax.text(
            0.98,
            0.05,
            f"Final Strength: {final_strength:.3f}\n"
            f"Total Accepted: {self.gate_open.sum()}/{self.num_samples}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=COLORS["reconstructed"],
                linewidth=2,
                alpha=0.95,
            ),
        )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate Visual 10: Recall Gate Comparator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display interactive visualization
  python visual_10_recall_gate.py --show

  # Save high-resolution output
  python visual_10_recall_gate.py --save-all --output-dir ../outputs

  # Custom threshold
  python visual_10_recall_gate.py --show --threshold 0.8
        """,
    )

    parser.add_argument(
        "--show", action="store_true", help="Display the visualization interactively"
    )
    parser.add_argument("--save-all", action="store_true", help="Save visualization")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../outputs",
        help="Output directory for saved figures (default: ../outputs)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for gate (default: 0.7)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of memory samples (default: 100)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for saved figures (default: 300)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Create output directory if saving
    if args.save_all:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path.absolute()}")

    # Create recall gate visualization
    print("Generating Recall Gate Comparator visualization...")
    gate = RecallGateComparator(
        num_samples=args.num_samples,
        embedding_dim=64,
        similarity_threshold=args.threshold,
        random_seed=args.seed,
    )

    # Generate visualization
    fig = gate.create_visualization()

    # Save if requested
    if args.save_all:
        output_file = output_path / "visual_10_recall_gate.png"
        print(f"Saving to {output_file}...")
        fig.savefig(
            output_file,
            dpi=args.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"✓ Saved: {output_file}")

    # Display if requested
    if args.show:
        print("Displaying visualization...")
        plt.show()
    elif not args.save_all:
        print("No output requested. Use --show or --save-all")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
