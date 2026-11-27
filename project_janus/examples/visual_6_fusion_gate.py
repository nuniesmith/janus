#!/usr/bin/env python3
"""
Visual 6: Multimodal Fusion Gate

Visualizes the gated cross-attention mechanism that dynamically balances multiple
input modalities (Visual/GAF, Temporal/Price, Sentiment/Text) based on context.

This visualization shows:
- Three input streams (GAF video, price temporal, sentiment text)
- Fusion hub with learned gating mechanism
- Gate activations over time showing dynamic rebalancing
- Output fusion with modality contributions

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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Color palette (WCAG 2.1 AA compliant, colorblind-safe)
COLORS = {
    "visual": "#2E8B57",  # Sea green (GAF/Visual)
    "temporal": "#4169E1",  # Royal blue (Price/Temporal)
    "sentiment": "#8B4513",  # Saddle brown (Text/Sentiment)
    "fusion": "#9370DB",  # Medium purple (Fusion hub)
    "gate": "#FF8C00",  # Dark orange (Gate mechanism)
    "output": "#DC143C",  # Crimson (Output)
    "edge": "#696969",  # Dim gray (connections)
    "background": "#F5F5F5",  # White smoke
    "grid": "#E0E0E0",  # Light gray
    "text": "#2F2F2F",  # Dark gray
}


class MultimodalFusionGate:
    """
    Gated cross-attention mechanism for multimodal fusion.

    The visualization demonstrates how:
    1. Three modalities (Visual, Temporal, Sentiment) enter the system
    2. A learned gate network determines attention weights
    3. Gate activations change over time based on context
    4. Output represents weighted fusion of all modalities
    """

    def __init__(
        self,
        sequence_length: int = 100,
        hidden_dim: int = 256,
        random_seed: int = 42,
    ):
        """
        Initialize the multimodal fusion gate.

        Args:
            sequence_length: Length of temporal sequence
            hidden_dim: Dimensionality of hidden representations
            random_seed: Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        np.random.seed(random_seed)

        # Generate synthetic gate activations over time
        self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic gate activation patterns."""
        t = np.linspace(0, 4 * np.pi, self.sequence_length)

        # Simulate different market regimes affecting gate behavior
        # Regime 1 (0-25%): High volatility -> prioritize Visual (GAF patterns)
        # Regime 2 (25-50%): Trending -> prioritize Temporal (momentum)
        # Regime 3 (50-75%): News event -> prioritize Sentiment
        # Regime 4 (75-100%): Balanced -> all modalities

        # Visual gate (GAF): High during volatile periods
        visual_base = 0.5 + 0.3 * np.sin(t)
        visual_spike = np.exp(-((t - 2 * np.pi) ** 2) / 2)  # Spike in middle
        self.gate_visual = np.clip(visual_base + 0.4 * visual_spike, 0, 1)

        # Temporal gate (Price): Trending periods
        temporal_base = 0.4 + 0.2 * np.sin(t + np.pi / 4)
        temporal_ramp = 0.3 * (t / (4 * np.pi))  # Increasing trend
        self.gate_temporal = np.clip(temporal_base + temporal_ramp, 0, 1)

        # Sentiment gate (Text): Spiky during news events
        sentiment_base = 0.3 + 0.1 * np.sin(t + np.pi / 2)
        # Create sharp spikes at specific times (news events)
        news_events = [np.pi, 2.5 * np.pi, 3.5 * np.pi]
        sentiment_spikes = sum(
            0.6 * np.exp(-((t - event) ** 2) / 0.3) for event in news_events
        )
        self.gate_sentiment = np.clip(sentiment_base + sentiment_spikes, 0, 1)

        # Normalize gates to sum to 1 (softmax-like)
        total = self.gate_visual + self.gate_temporal + self.gate_sentiment
        self.gate_visual = self.gate_visual / total
        self.gate_temporal = self.gate_temporal / total
        self.gate_sentiment = self.gate_sentiment / total

        # Compute fusion output (weighted combination)
        self.fusion_output = (
            self.gate_visual * 0.8
            + self.gate_temporal * 0.6
            + self.gate_sentiment * 0.5
        )

        # Market regimes for annotation
        self.regimes = [
            (0, 25, "Volatile", COLORS["visual"]),
            (25, 50, "Trending", COLORS["temporal"]),
            (50, 75, "News Event", COLORS["sentiment"]),
            (75, 100, "Balanced", COLORS["fusion"]),
        ]

    def create_visualization(self) -> plt.Figure:
        """
        Create the complete multimodal fusion gate visualization.

        Returns:
            Matplotlib figure object
        """
        # Create figure with 3 panels
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            3, 1, height_ratios=[1.2, 1, 1], hspace=0.35, left=0.08, right=0.95
        )

        # Panel A: Schematic diagram of fusion architecture
        ax_arch = fig.add_subplot(gs[0])
        self._plot_architecture(ax_arch)

        # Panel B: Gate activations over time
        ax_gates = fig.add_subplot(gs[1])
        self._plot_gate_activations(ax_gates)

        # Panel C: Modality contributions (stacked area)
        ax_contrib = fig.add_subplot(gs[2])
        self._plot_contributions(ax_contrib)

        # Overall title
        fig.suptitle(
            "Visual 6: Multimodal Fusion Gate\n"
            "Dynamic Attention Rebalancing Across Visual, Temporal, and Sentiment Streams",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        return fig

    def _plot_architecture(self, ax: plt.Axes) -> None:
        """Plot the fusion architecture schematic."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_facecolor(COLORS["background"])

        # Title
        ax.text(
            5,
            5.8,
            "A. Gated Cross-Attention Architecture",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=COLORS["text"],
        )

        # ===== INPUT MODALITIES =====
        modalities = [
            ("Visual\n(GAF)", 1.5, 4.5, COLORS["visual"], "CNN"),
            ("Temporal\n(Price)", 5.0, 4.5, COLORS["temporal"], "LSTM"),
            ("Sentiment\n(Text)", 8.5, 4.5, COLORS["sentiment"], "BERT"),
        ]

        modality_positions = {}
        for name, x, y, color, encoder in modalities:
            modality_positions[name.split("\n")[0]] = (x, y)

            # Input box
            box = FancyBboxPatch(
                (x - 0.5, y - 0.35),
                1.0,
                0.7,
                boxstyle="round,pad=0.08",
                facecolor=color,
                edgecolor="white",
                linewidth=2.5,
                alpha=0.85,
            )
            ax.add_patch(box)

            # Label
            ax.text(
                x,
                y,
                name,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

            # Encoder annotation
            ax.text(
                x,
                y - 0.65,
                f"Encoder: {encoder}",
                ha="center",
                va="top",
                fontsize=8,
                style="italic",
                color=COLORS["text"],
            )

        # ===== FUSION HUB =====
        fusion_x, fusion_y = 5.0, 2.5

        # Large fusion circle
        fusion_circle = Circle(
            (fusion_x, fusion_y),
            0.6,
            facecolor=COLORS["fusion"],
            edgecolor="white",
            linewidth=3,
            alpha=0.9,
        )
        ax.add_patch(fusion_circle)

        ax.text(
            fusion_x,
            fusion_y + 0.15,
            "Fusion",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
        ax.text(
            fusion_x,
            fusion_y - 0.15,
            "Hub",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )

        # ===== GATE MECHANISM =====
        gate_x, gate_y = 5.0, 3.5

        # Gate box (smaller, inside pathway)
        gate_box = FancyBboxPatch(
            (gate_x - 0.35, gate_y - 0.25),
            0.7,
            0.5,
            boxstyle="round,pad=0.05",
            facecolor=COLORS["gate"],
            edgecolor="white",
            linewidth=2,
            alpha=0.9,
        )
        ax.add_patch(gate_box)

        ax.text(
            gate_x,
            gate_y,
            "Gate\nMLP→σ",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

        # Gate formula
        ax.text(
            gate_x + 1.2,
            gate_y,
            "g = σ(W·[v;t;s] + b)\ng ∈ [0,1]³",
            ha="left",
            va="center",
            fontsize=8,
            family="monospace",
            color=COLORS["text"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

        # ===== CONNECTIONS =====
        # Input → Gate
        for name, (mod_x, mod_y) in modality_positions.items():
            arrow = FancyArrowPatch(
                (mod_x, mod_y - 0.4),
                (gate_x, gate_y + 0.3),
                arrowstyle="-",
                mutation_scale=20,
                linewidth=2,
                color=COLORS["edge"],
                alpha=0.5,
            )
            ax.add_patch(arrow)

        # Gate → Fusion (with modulation)
        gate_outputs = [
            (gate_x - 0.4, gate_y - 0.3, fusion_x - 0.5, fusion_y + 0.4, "gᵥ"),
            (gate_x, gate_y - 0.3, fusion_x, fusion_y + 0.6, "gₜ"),
            (gate_x + 0.4, gate_y - 0.3, fusion_x + 0.5, fusion_y + 0.4, "gₛ"),
        ]

        colors_list = [COLORS["visual"], COLORS["temporal"], COLORS["sentiment"]]
        for i, (x1, y1, x2, y2, label) in enumerate(gate_outputs):
            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="->",
                mutation_scale=20,
                linewidth=2.5,
                color=colors_list[i],
                alpha=0.7,
            )
            ax.add_patch(arrow)

            # Gate value label
            ax.text(
                (x1 + x2) / 2 + 0.15,
                (y1 + y2) / 2,
                label,
                fontsize=9,
                fontweight="bold",
                style="italic",
                color=colors_list[i],
            )

        # ===== OUTPUT =====
        output_x, output_y = 5.0, 0.8

        output_box = FancyBboxPatch(
            (output_x - 0.6, output_y - 0.3),
            1.2,
            0.6,
            boxstyle="round,pad=0.08",
            facecolor=COLORS["output"],
            edgecolor="white",
            linewidth=2.5,
            alpha=0.85,
        )
        ax.add_patch(output_box)

        ax.text(
            output_x,
            output_y,
            "Fused Output",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )

        # Fusion → Output
        arrow = FancyArrowPatch(
            (fusion_x, fusion_y - 0.6),
            (output_x, output_y + 0.35),
            arrowstyle="->",
            mutation_scale=25,
            linewidth=3,
            color=COLORS["edge"],
            alpha=0.6,
        )
        ax.add_patch(arrow)

        # Output formula
        ax.text(
            output_x,
            output_y - 0.6,
            "h = gᵥ·vₑ + gₜ·tₑ + gₛ·sₑ",
            ha="center",
            va="top",
            fontsize=9,
            family="monospace",
            color=COLORS["text"],
        )

        # ===== ANNOTATIONS =====
        # Key insight box
        ax.text(
            0.5,
            1.8,
            "Key Insight:\nGate learns to suppress\nirrelevant modalities\nbased on context",
            fontsize=9,
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=COLORS["gate"],
                edgecolor="white",
                linewidth=2,
                alpha=0.3,
            ),
        )

        # Example scenario
        ax.text(
            9.5,
            1.8,
            "Example:\nHigh latency → gₜ↓\nNews event → gₛ↑\nVolatile → gᵥ↑",
            ha="right",
            fontsize=9,
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=COLORS["text"],
                linewidth=1.5,
                alpha=0.95,
            ),
        )

    def _plot_gate_activations(self, ax: plt.Axes) -> None:
        """Plot gate activation values over time."""
        t = np.arange(self.sequence_length)

        ax.set_facecolor(COLORS["background"])
        ax.set_title(
            "B. Gate Activations Over Time (Market Regimes)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        # Plot gate values
        ax.plot(
            t,
            self.gate_visual,
            linewidth=2.5,
            color=COLORS["visual"],
            label="Visual Gate (gᵥ)",
            alpha=0.9,
        )
        ax.plot(
            t,
            self.gate_temporal,
            linewidth=2.5,
            color=COLORS["temporal"],
            label="Temporal Gate (gₜ)",
            alpha=0.9,
        )
        ax.plot(
            t,
            self.gate_sentiment,
            linewidth=2.5,
            color=COLORS["sentiment"],
            label="Sentiment Gate (gₛ)",
            alpha=0.9,
        )

        # Background shading for regimes
        for start, end, label, color in self.regimes:
            ax.axvspan(start, end, alpha=0.15, color=color)
            # Regime label
            ax.text(
                (start + end) / 2,
                1.05,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

        # Styling
        ax.set_xlabel("Time Step", fontsize=11, fontweight="bold")
        ax.set_ylabel("Gate Activation", fontsize=11, fontweight="bold")
        ax.set_xlim(0, self.sequence_length - 1)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

        # Add horizontal reference line
        ax.axhline(y=1.0, color="k", linestyle="--", linewidth=1, alpha=0.3)

        # Annotations for key events
        annotations = [
            (25, 0.75, "Volatility\nSpike", COLORS["visual"]),
            (50, 0.85, "News\nEvent", COLORS["sentiment"]),
            (75, 0.5, "Regime\nShift", COLORS["fusion"]),
        ]

        for x, y, text, color in annotations:
            ax.annotate(
                text,
                xy=(x, y),
                xytext=(x + 5, y + 0.15),
                fontsize=8,
                color=color,
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.5,
                    alpha=0.7,
                    shrinkA=0,
                    shrinkB=5,
                ),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            )

    def _plot_contributions(self, ax: plt.Axes) -> None:
        """Plot modality contributions as stacked area chart."""
        t = np.arange(self.sequence_length)

        ax.set_facecolor(COLORS["background"])
        ax.set_title(
            "C. Modality Contributions (Normalized)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        # Stacked area chart
        ax.fill_between(
            t,
            0,
            self.gate_visual,
            color=COLORS["visual"],
            alpha=0.7,
            label="Visual (GAF)",
        )
        ax.fill_between(
            t,
            self.gate_visual,
            self.gate_visual + self.gate_temporal,
            color=COLORS["temporal"],
            alpha=0.7,
            label="Temporal (Price)",
        )
        ax.fill_between(
            t,
            self.gate_visual + self.gate_temporal,
            1.0,  # Should sum to 1
            color=COLORS["sentiment"],
            alpha=0.7,
            label="Sentiment (Text)",
        )

        # Add separation lines
        ax.plot(t, self.gate_visual, color="white", linewidth=1.5, alpha=0.8)
        ax.plot(
            t,
            self.gate_visual + self.gate_temporal,
            color="white",
            linewidth=1.5,
            alpha=0.8,
        )

        # Styling
        ax.set_xlabel("Time Step", fontsize=11, fontweight="bold")
        ax.set_ylabel("Contribution (Σ=1)", fontsize=11, fontweight="bold")
        ax.set_xlim(0, self.sequence_length - 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linestyle="--", axis="x")
        ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

        # Background shading for regimes (lighter)
        for start, end, label, color in self.regimes:
            ax.axvspan(start, end, alpha=0.08, color=color)

        # Summary statistics box
        stats_text = (
            f"Mean Contributions:\n"
            f"Visual:    {self.gate_visual.mean():.2f}\n"
            f"Temporal:  {self.gate_temporal.mean():.2f}\n"
            f"Sentiment: {self.gate_sentiment.mean():.2f}"
        )
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=COLORS["text"],
                linewidth=1.5,
                alpha=0.95,
            ),
        )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate Visual 6: Multimodal Fusion Gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display interactive visualization
  python visual_6_fusion_gate.py --show

  # Save high-resolution output
  python visual_6_fusion_gate.py --save-all --output-dir ../outputs

  # Custom sequence length
  python visual_6_fusion_gate.py --show --sequence-length 200
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
        "--sequence-length",
        type=int,
        default=100,
        help="Length of temporal sequence (default: 100)",
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

    # Create fusion gate visualization
    print("Generating Multimodal Fusion Gate visualization...")
    fusion = MultimodalFusionGate(
        sequence_length=args.sequence_length,
        hidden_dim=256,
        random_seed=args.seed,
    )

    # Generate visualization
    fig = fusion.create_visualization()

    # Save if requested
    if args.save_all:
        output_file = output_path / "visual_6_fusion_gate.png"
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
