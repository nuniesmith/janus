#!/usr/bin/env python3
"""
Visual 9: Memory Consolidation Cycle

Visualizes the three-timescale memory hierarchy mimicking biological memory systems:
- Episodic Buffer (Hippocampus) for fast recording
- SWR Simulator (Sharp-Wave Ripples) for prioritized replay
- Neocortical Schema for slow integration

This visualization shows:
- Circular flow of memory consolidation
- Prioritized Experience Replay (PER) distribution
- Transfer from fast learning to slow generalization
- Power-law replay probabilities

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
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
    Wedge,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Color palette (WCAG 2.1 AA compliant, colorblind-safe)
COLORS = {
    "hippocampus": "#2E8B57",  # Sea green (episodic buffer)
    "swr": "#FF8C00",  # Dark orange (sharp-wave ripples)
    "neocortex": "#4169E1",  # Royal blue (schemas)
    "replay": "#DC143C",  # Crimson (replay events)
    "transfer": "#9370DB",  # Medium purple (consolidation)
    "edge": "#696969",  # Dim gray (connections)
    "background": "#F5F5F5",  # White smoke
    "grid": "#E0E0E0",  # Light gray
    "text": "#2F2F2F",  # Dark gray
    "surprise": "#FFD700",  # Gold (high TD error)
}


class MemoryConsolidationCycle:
    """
    Three-timescale memory hierarchy visualization.

    The visualization demonstrates how:
    1. Episodic buffer records raw experiences (FIFO)
    2. SWR simulator prioritizes high-error events for replay
    3. Neocortical schemas slowly integrate generalized patterns
    4. Power-law replay distribution focuses on rare events
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        num_schemas: int = 50,
        alpha: float = 0.6,
        random_seed: int = 42,
    ):
        """
        Initialize the memory consolidation system.

        Args:
            buffer_size: Size of episodic buffer
            num_schemas: Number of neocortical schemas
            alpha: PER exponent (higher = more prioritization)
            random_seed: Random seed for reproducibility
        """
        self.buffer_size = buffer_size
        self.num_schemas = num_schemas
        self.alpha = alpha

        np.random.seed(random_seed)

        # Generate synthetic memory data
        self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic memory and replay data."""
        # TD errors (used for prioritization)
        # Most experiences have low error, few have high error (power law)
        self.td_errors = np.random.exponential(scale=0.5, size=self.buffer_size)
        self.td_errors = np.clip(self.td_errors, 0.01, 10.0)

        # Add a few "black swan" events with very high error
        black_swan_indices = np.random.choice(self.buffer_size, size=5, replace=False)
        self.td_errors[black_swan_indices] = np.random.uniform(5, 10, size=5)

        # Compute prioritized replay probabilities
        priorities = np.abs(self.td_errors) ** self.alpha
        self.replay_probs = priorities / priorities.sum()

        # Sample replay events
        self.replay_samples = np.random.choice(
            self.buffer_size, size=500, p=self.replay_probs
        )

        # Neocortical schema strengths (how consolidated each pattern is)
        self.schema_strengths = np.random.beta(2, 5, size=self.num_schemas)

        # Memory age (for FIFO visualization)
        self.memory_ages = np.arange(self.buffer_size)

        # Consolidation events over time
        self.consolidation_timeline = self._generate_consolidation_timeline()

    def _generate_consolidation_timeline(self) -> np.ndarray:
        """Generate timeline of consolidation events."""
        timeline_length = 100
        timeline = np.zeros((timeline_length, 3))  # [hippocampus, swr, neocortex]

        # Hippocampus: constantly recording (high activity)
        timeline[:, 0] = 0.7 + 0.2 * np.random.rand(timeline_length)

        # SWR: bursty activity during "sleep" periods
        sleep_periods = [(20, 35), (60, 75)]
        for start, end in sleep_periods:
            timeline[start:end, 1] = 0.8 + 0.2 * np.random.rand(end - start)

        # Neocortex: slow, gradual increase
        timeline[:, 2] = np.linspace(0.1, 0.6, timeline_length)
        timeline[:, 2] += 0.1 * np.random.rand(timeline_length)

        return timeline

    def create_visualization(self) -> plt.Figure:
        """
        Create the complete memory consolidation visualization.

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

        # Panel A: Consolidation cycle diagram
        ax_cycle = fig.add_subplot(gs[0, 0])
        self._plot_consolidation_cycle(ax_cycle)

        # Panel B: PER distribution
        ax_per = fig.add_subplot(gs[0, 1])
        self._plot_per_distribution(ax_per)

        # Panel C: Replay frequency histogram
        ax_replay = fig.add_subplot(gs[1, 0])
        self._plot_replay_histogram(ax_replay)

        # Panel D: Consolidation timeline
        ax_timeline = fig.add_subplot(gs[1, 1])
        self._plot_consolidation_timeline(ax_timeline)

        # Overall title
        fig.suptitle(
            "Visual 9: Memory Consolidation Cycle\n"
            "Hippocampus → SWR Replay → Neocortical Schema Integration",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        return fig

    def _plot_consolidation_cycle(self, ax: plt.Axes) -> None:
        """Plot the circular consolidation cycle diagram."""
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axis("off")
        ax.set_facecolor(COLORS["background"])

        # Title
        ax.text(
            0,
            1.95,
            "A. Three-Timescale Memory Hierarchy",
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=COLORS["text"],
        )

        # Center positions for each component
        positions = {
            "hippocampus": (-1.0, 0.5),
            "swr": (0, -1.0),
            "neocortex": (1.0, 0.5),
        }

        # ===== HIPPOCAMPUS (Episodic Buffer) =====
        hipp_x, hipp_y = positions["hippocampus"]

        # Draw as circular buffer (ring)
        outer_circle = Circle(
            (hipp_x, hipp_y),
            0.5,
            facecolor=COLORS["hippocampus"],
            edgecolor="white",
            linewidth=3,
            alpha=0.8,
        )
        ax.add_patch(outer_circle)

        inner_circle = Circle(
            (hipp_x, hipp_y),
            0.3,
            facecolor=COLORS["background"],
            edgecolor="white",
            linewidth=2,
            alpha=1.0,
        )
        ax.add_patch(inner_circle)

        # FIFO arrow
        arc = Arc(
            (hipp_x, hipp_y),
            0.8,
            0.8,
            angle=0,
            theta1=30,
            theta2=330,
            color="white",
            linewidth=2.5,
            linestyle="-",
        )
        ax.add_patch(arc)

        # Arrow head
        arrow_angle = np.radians(30)
        arrow_x = hipp_x + 0.4 * np.cos(arrow_angle)
        arrow_y = hipp_y + 0.4 * np.sin(arrow_angle)
        ax.annotate(
            "",
            xy=(arrow_x, arrow_y),
            xytext=(arrow_x - 0.1, arrow_y + 0.05),
            arrowprops=dict(arrowstyle="->", color="white", lw=2.5),
        )

        ax.text(
            hipp_x,
            hipp_y + 0.85,
            "Episodic Buffer",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=COLORS["hippocampus"],
        )
        ax.text(
            hipp_x,
            hipp_y - 0.85,
            "Hippocampus\nFIFO Ring",
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color=COLORS["text"],
        )

        # Capacity label
        ax.text(
            hipp_x,
            hipp_y,
            f"N={self.buffer_size}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=COLORS["hippocampus"],
        )

        # ===== SWR SIMULATOR (Prioritized Replay) =====
        swr_x, swr_y = positions["swr"]

        # Draw as burst/spike pattern
        swr_box = FancyBboxPatch(
            (swr_x - 0.45, swr_y - 0.4),
            0.9,
            0.8,
            boxstyle="round,pad=0.1",
            facecolor=COLORS["swr"],
            edgecolor="white",
            linewidth=3,
            alpha=0.8,
        )
        ax.add_patch(swr_box)

        # Draw spike pattern inside
        spike_x = np.linspace(swr_x - 0.3, swr_x + 0.3, 7)
        spike_heights = [0.15, 0.35, 0.2, 0.4, 0.25, 0.3, 0.15]
        for i, (sx, sh) in enumerate(zip(spike_x, spike_heights)):
            ax.plot(
                [sx, sx],
                [swr_y - 0.2, swr_y - 0.2 + sh],
                color="white",
                linewidth=3,
                solid_capstyle="round",
            )

        ax.text(
            swr_x,
            swr_y + 0.6,
            "SWR Simulator",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=COLORS["swr"],
        )
        ax.text(
            swr_x,
            swr_y - 0.6,
            "Sharp-Wave\nRipples",
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color=COLORS["text"],
        )

        # Priority label
        ax.text(
            swr_x,
            swr_y + 0.35,
            f"P ∝ |δ|^{self.alpha}",
            ha="center",
            va="center",
            fontsize=8,
            family="monospace",
            color="white",
            fontweight="bold",
        )

        # ===== NEOCORTEX (Schema Storage) =====
        neo_x, neo_y = positions["neocortex"]

        # Draw as network/grid structure
        neo_circle = Circle(
            (neo_x, neo_y),
            0.5,
            facecolor=COLORS["neocortex"],
            edgecolor="white",
            linewidth=3,
            alpha=0.8,
        )
        ax.add_patch(neo_circle)

        # Grid pattern inside
        grid_points = [
            (neo_x - 0.2, neo_y + 0.2),
            (neo_x, neo_y + 0.2),
            (neo_x + 0.2, neo_y + 0.2),
            (neo_x - 0.2, neo_y),
            (neo_x, neo_y),
            (neo_x + 0.2, neo_y),
            (neo_x - 0.2, neo_y - 0.2),
            (neo_x, neo_y - 0.2),
            (neo_x + 0.2, neo_y - 0.2),
        ]
        for px, py in grid_points:
            ax.plot(px, py, "o", color="white", markersize=5)

        # Connect grid points
        for i in range(3):
            for j in range(2):
                idx = i * 3 + j
                ax.plot(
                    [grid_points[idx][0], grid_points[idx + 1][0]],
                    [grid_points[idx][1], grid_points[idx + 1][1]],
                    color="white",
                    linewidth=1,
                    alpha=0.6,
                )
        for i in range(2):
            for j in range(3):
                idx = i * 3 + j
                ax.plot(
                    [grid_points[idx][0], grid_points[idx + 3][0]],
                    [grid_points[idx][1], grid_points[idx + 3][1]],
                    color="white",
                    linewidth=1,
                    alpha=0.6,
                )

        ax.text(
            neo_x,
            neo_y + 0.85,
            "Neocortical\nSchemas",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=COLORS["neocortex"],
        )
        ax.text(
            neo_x,
            neo_y - 0.85,
            "Vector DB\n(Qdrant)",
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color=COLORS["text"],
        )

        # ===== FLOW ARROWS =====
        # Hippocampus → SWR (Experience sampling)
        arrow1 = FancyArrowPatch(
            (hipp_x + 0.2, hipp_y - 0.4),
            (swr_x - 0.3, swr_y + 0.3),
            arrowstyle="->",
            mutation_scale=25,
            linewidth=3,
            color=COLORS["replay"],
            alpha=0.7,
        )
        ax.add_patch(arrow1)
        ax.text(
            (hipp_x + swr_x) / 2 - 0.3,
            (hipp_y + swr_y) / 2 + 0.2,
            "Sample\nExperiences",
            ha="center",
            fontsize=8,
            color=COLORS["replay"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

        # SWR → Neocortex (Consolidation)
        arrow2 = FancyArrowPatch(
            (swr_x + 0.3, swr_y + 0.3),
            (neo_x - 0.2, neo_y - 0.4),
            arrowstyle="->",
            mutation_scale=25,
            linewidth=3,
            color=COLORS["transfer"],
            alpha=0.7,
        )
        ax.add_patch(arrow2)
        ax.text(
            (swr_x + neo_x) / 2 + 0.3,
            (swr_y + neo_y) / 2 + 0.2,
            "Consolidate\nPatterns",
            ha="center",
            fontsize=8,
            color=COLORS["transfer"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

        # Neocortex → Hippocampus (Feedback/Prior)
        arrow3 = FancyArrowPatch(
            (neo_x - 0.4, neo_y + 0.3),
            (hipp_x + 0.4, hipp_y + 0.3),
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2,
            color=COLORS["edge"],
            alpha=0.5,
            linestyle="dashed",
        )
        ax.add_patch(arrow3)
        ax.text(
            (hipp_x + neo_x) / 2,
            (hipp_y + neo_y) / 2 + 0.7,
            "Prior Knowledge",
            ha="center",
            fontsize=7,
            color=COLORS["edge"],
            style="italic",
        )

        # ===== TIMESCALE ANNOTATIONS =====
        timescales = [
            (hipp_x, hipp_y - 1.15, "τ₁ ~ ms", "Fast recording"),
            (swr_x, swr_y - 0.9, "τ₂ ~ s", "Replay bursts"),
            (neo_x, neo_y - 1.15, "τ₃ ~ hours", "Slow integration"),
        ]
        for x, y, tau, desc in timescales:
            ax.text(
                x,
                y,
                f"{tau}\n({desc})",
                ha="center",
                va="top",
                fontsize=8,
                color=COLORS["text"],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=COLORS["background"],
                    edgecolor=COLORS["text"],
                    linewidth=1,
                    alpha=0.95,
                ),
            )

    def _plot_per_distribution(self, ax: plt.Axes) -> None:
        """Plot Prioritized Experience Replay distribution."""
        ax.set_facecolor(COLORS["background"])
        ax.set_title(
            "B. Prioritized Replay Distribution", fontsize=13, fontweight="bold", pad=15
        )

        # Sort experiences by TD error
        sorted_indices = np.argsort(self.td_errors)[::-1]
        sorted_errors = self.td_errors[sorted_indices]
        sorted_probs = self.replay_probs[sorted_indices]

        # Plot on log-log scale to show power law
        x = np.arange(1, len(sorted_probs) + 1)

        # Replay probability
        ax.plot(
            x,
            sorted_probs,
            linewidth=2.5,
            color=COLORS["swr"],
            label=f"Replay Prob (α={self.alpha})",
            alpha=0.9,
        )

        # Highlight top 5% (most replayed)
        top_5_percent = int(0.05 * len(sorted_probs))
        ax.fill_between(
            x[:top_5_percent],
            0,
            sorted_probs[:top_5_percent],
            color=COLORS["surprise"],
            alpha=0.4,
            label="Top 5% (High Surprise)",
        )

        # Uniform baseline (for comparison)
        uniform_prob = 1.0 / len(sorted_probs)
        ax.axhline(
            y=uniform_prob,
            color=COLORS["edge"],
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label="Uniform (no priority)",
        )

        # Annotations
        ax.annotate(
            "Black Swans\n(Rare, High-Error)",
            xy=(5, sorted_probs[0]),
            xytext=(50, sorted_probs[0] * 1.5),
            fontsize=9,
            color=COLORS["surprise"],
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["surprise"], lw=2),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95),
        )

        ax.set_xlabel("Experience Rank (by TD Error)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Replay Probability", fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both", linestyle="--")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

        # Add power law annotation
        ax.text(
            0.05,
            0.05,
            f"Power Law: P(i) ∝ |δᵢ|^{self.alpha}\n"
            f"Heavy-tailed distribution\n"
            f"focuses on rare events",
            transform=ax.transAxes,
            fontsize=9,
            style="italic",
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=COLORS["background"],
                edgecolor=COLORS["swr"],
                linewidth=2,
                alpha=0.95,
            ),
        )

    def _plot_replay_histogram(self, ax: plt.Axes) -> None:
        """Plot histogram of replay frequency."""
        ax.set_facecolor(COLORS["background"])
        ax.set_title(
            "C. Replay Event Frequency", fontsize=13, fontweight="bold", pad=15
        )

        # Histogram of replayed experiences
        counts, bins, patches = ax.hist(
            self.replay_samples,
            bins=50,
            color=COLORS["replay"],
            alpha=0.7,
            edgecolor="white",
            linewidth=1.5,
        )

        # Color code by TD error magnitude
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i + 1]) / 2
            bin_idx = int(bin_center)
            if bin_idx < len(self.td_errors):
                error = self.td_errors[bin_idx]
                # Normalize error to [0, 1] for colormap
                normalized_error = min(error / 10.0, 1.0)
                color = plt.cm.YlOrRd(normalized_error)
                patch.set_facecolor(color)

        ax.set_xlabel("Experience Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("Replay Count", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

        # Color bar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=10)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("TD Error |δ|", fontsize=10, fontweight="bold")

        # Statistics
        most_replayed = np.bincount(self.replay_samples).argmax()
        max_count = counts.max()

        ax.annotate(
            f"Most Replayed:\nIndex {most_replayed}\n({int(max_count)} times)",
            xy=(most_replayed, max_count),
            xytext=(most_replayed + 150, max_count * 0.8),
            fontsize=9,
            color=COLORS["replay"],
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["replay"], lw=2),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95),
        )

    def _plot_consolidation_timeline(self, ax: plt.Axes) -> None:
        """Plot activity timeline across memory components."""
        ax.set_facecolor(COLORS["background"])
        ax.set_title(
            "D. Consolidation Activity Timeline", fontsize=13, fontweight="bold", pad=15
        )

        t = np.arange(len(self.consolidation_timeline))

        # Plot each component
        ax.plot(
            t,
            self.consolidation_timeline[:, 0],
            linewidth=2.5,
            color=COLORS["hippocampus"],
            label="Hippocampus (Recording)",
            alpha=0.9,
        )
        ax.plot(
            t,
            self.consolidation_timeline[:, 1],
            linewidth=2.5,
            color=COLORS["swr"],
            label="SWR (Replay)",
            alpha=0.9,
        )
        ax.plot(
            t,
            self.consolidation_timeline[:, 2],
            linewidth=2.5,
            color=COLORS["neocortex"],
            label="Neocortex (Integration)",
            alpha=0.9,
        )

        # Highlight sleep periods
        sleep_periods = [(20, 35), (60, 75)]
        for start, end in sleep_periods:
            ax.axvspan(start, end, alpha=0.2, color=COLORS["swr"])
            ax.text(
                (start + end) / 2,
                1.05,
                "Sleep\n(Replay)",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLORS["swr"],
                fontweight="bold",
            )

        ax.set_xlabel("Time (arbitrary units)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Activity Level", fontsize=11, fontweight="bold")
        ax.set_xlim(0, len(t) - 1)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

        # Key insight
        ax.text(
            0.98,
            0.05,
            "During sleep, hippocampal\nreplay transfers memories\nto neocortex",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            style="italic",
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=COLORS["transfer"],
                linewidth=2,
                alpha=0.95,
            ),
        )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate Visual 9: Memory Consolidation Cycle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display interactive visualization
  python visual_9_memory_consolidation.py --show

  # Save high-resolution output
  python visual_9_memory_consolidation.py --save-all --output-dir ../outputs

  # Custom parameters
  python visual_9_memory_consolidation.py --show --buffer-size 2000 --alpha 0.8
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
        "--buffer-size",
        type=int,
        default=1000,
        help="Episodic buffer size (default: 1000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="PER prioritization exponent (default: 0.6)",
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

    # Create memory consolidation visualization
    print("Generating Memory Consolidation Cycle visualization...")
    memory = MemoryConsolidationCycle(
        buffer_size=args.buffer_size,
        num_schemas=50,
        alpha=args.alpha,
        random_seed=args.seed,
    )

    # Generate visualization
    fig = memory.create_visualization()

    # Save if requested
    if args.save_all:
        output_file = output_path / "visual_9_memory_consolidation.png"
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
