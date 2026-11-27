#!/usr/bin/env python3
"""
Visual 4: Logic Tensor Network (LTN) Grounding Graph

Visualizes the bipartite mapping from symbolic concepts to neural embeddings,
demonstrating how discrete logical predicates are "grounded" in continuous
vector spaces for differentiable reasoning.

This visualization shows:
- Symbolic layer (concepts, predicates, rules)
- Neural layer (embeddings, MLPs, fuzzy logic operations)
- Grounding mappings between layers
- Łukasiewicz T-norm operations (AND, OR, IMPLIES)

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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Color palette (WCAG 2.1 AA compliant, colorblind-safe)
COLORS = {
    "concept": "#2E8B57",  # Sea green (concepts)
    "predicate": "#4169E1",  # Royal blue (predicates)
    "rule": "#8B4513",  # Saddle brown (rules)
    "embedding": "#9370DB",  # Medium purple (embeddings)
    "mlp": "#FF8C00",  # Dark orange (MLPs)
    "fuzzy_op": "#DC143C",  # Crimson (fuzzy operations)
    "edge": "#696969",  # Dim gray (connections)
    "background": "#F5F5F5",  # White smoke
    "grid": "#E0E0E0",  # Light gray
    "text": "#2F2F2F",  # Dark gray
}


class LTNGroundingGraph:
    """
    Bipartite graph showing symbolic-to-neural grounding in Logic Tensor Networks.

    The visualization demonstrates how:
    1. Concepts (e.g., "Bullish", "HighVolume") map to vector embeddings
    2. Predicates (e.g., Buy(x), Sell(x)) map to neural networks (MLPs)
    3. Logical operations (AND, OR, IMPLIES) use Łukasiewicz T-norms
    4. Rules combine these elements for differentiable reasoning
    """

    def __init__(
        self,
        num_concepts: int = 5,
        num_predicates: int = 4,
        num_rules: int = 3,
        embedding_dim: int = 64,
        random_seed: int = 42,
    ):
        """
        Initialize the LTN grounding graph.

        Args:
            num_concepts: Number of symbolic concepts
            num_predicates: Number of predicates
            num_rules: Number of logical rules
            embedding_dim: Dimensionality of concept embeddings
            random_seed: Random seed for reproducibility
        """
        self.num_concepts = num_concepts
        self.num_predicates = num_predicates
        self.num_rules = num_rules
        self.embedding_dim = embedding_dim

        np.random.seed(random_seed)

        # Define symbolic concepts
        self.concepts = [
            "Bullish",
            "Bearish",
            "HighVolume",
            "LowLiquidity",
            "Trending",
        ][:num_concepts]

        # Define predicates
        self.predicates = ["Buy(x)", "Sell(x)", "Hold(x)", "Risk(x)"][:num_predicates]

        # Define logical rules
        self.rules = [
            "Bullish(x) ∧ HighVolume(x) → Buy(x)",
            "Bearish(x) ∨ LowLiquidity(x) → Sell(x)",
            "¬Risk(x) → Hold(x)",
        ][:num_rules]

        # Generate synthetic embeddings (for visualization)
        self.embeddings = {
            concept: np.random.randn(embedding_dim) for concept in self.concepts
        }

        # MLP architectures for predicates
        self.mlp_configs = {
            pred: [embedding_dim, 128, 64, 1]  # Input → Hidden → Output
            for pred in self.predicates
        }

    def lukasiewicz_and(self, u: float, v: float) -> float:
        """Łukasiewicz T-norm AND operation."""
        return max(0.0, u + v - 1.0)

    def lukasiewicz_or(self, u: float, v: float) -> float:
        """Łukasiewicz S-norm OR operation."""
        return min(1.0, u + v)

    def lukasiewicz_implies(self, u: float, v: float) -> float:
        """Łukasiewicz implication."""
        return min(1.0, 1.0 - u + v)

    def lukasiewicz_not(self, u: float) -> float:
        """Łukasiewicz negation."""
        return 1.0 - u

    def create_visualization(
        self, show_embeddings: bool = True, show_gradients: bool = True
    ) -> plt.Figure:
        """
        Create the complete LTN grounding graph visualization.

        Args:
            show_embeddings: Whether to show embedding vector representations
            show_gradients: Whether to show gradient flow arrows

        Returns:
            Matplotlib figure object
        """
        # Create figure with 2 panels
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)

        # Panel A: Bipartite grounding graph
        ax_graph = fig.add_subplot(gs[0])
        self._plot_grounding_graph(ax_graph, show_embeddings, show_gradients)

        # Panel B: Fuzzy logic operations
        ax_fuzzy = fig.add_subplot(gs[1])
        self._plot_fuzzy_operations(ax_fuzzy)

        # Overall title
        fig.suptitle(
            "Visual 4: Logic Tensor Network (LTN) Grounding Graph\n"
            "Symbolic Concepts → Neural Embeddings → Differentiable Reasoning",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        return fig

    def _plot_grounding_graph(
        self, ax: plt.Axes, show_embeddings: bool, show_gradients: bool
    ) -> None:
        """Plot the bipartite grounding graph."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")
        ax.set_facecolor(COLORS["background"])

        # Title
        ax.text(
            5,
            11.5,
            "A. Symbolic-Neural Grounding",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=COLORS["text"],
        )

        # ===== SYMBOLIC LAYER (Top) =====
        ax.text(
            0.5,
            10.5,
            "Symbolic Layer",
            fontsize=12,
            fontweight="bold",
            color=COLORS["text"],
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=COLORS["text"]),
        )

        # Concepts
        concept_y = 9.5
        concept_positions = {}
        for i, concept in enumerate(self.concepts):
            x = 1 + i * 1.8
            y = concept_y
            concept_positions[concept] = (x, y)

            # Draw concept box
            box = FancyBboxPatch(
                (x - 0.4, y - 0.25),
                0.8,
                0.5,
                boxstyle="round,pad=0.05",
                facecolor=COLORS["concept"],
                edgecolor="white",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(box)

            # Concept label
            ax.text(
                x,
                y,
                concept,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

        # Predicates
        predicate_y = 8.5
        predicate_positions = {}
        for i, predicate in enumerate(self.predicates):
            x = 1.5 + i * 2.0
            y = predicate_y
            predicate_positions[predicate] = (x, y)

            # Draw predicate box
            box = FancyBboxPatch(
                (x - 0.35, y - 0.25),
                0.7,
                0.5,
                boxstyle="round,pad=0.05",
                facecolor=COLORS["predicate"],
                edgecolor="white",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(box)

            # Predicate label
            ax.text(
                x,
                y,
                predicate,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

        # Rules
        rule_y = 7.5
        for i, rule in enumerate(self.rules):
            x = 2 + i * 2.5
            y = rule_y

            # Draw rule box
            box = FancyBboxPatch(
                (x - 0.8, y - 0.25),
                1.6,
                0.5,
                boxstyle="round,pad=0.05",
                facecolor=COLORS["rule"],
                edgecolor="white",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(box)

            # Rule label (shortened for display)
            short_rule = rule.split("→")[0][:20] + "→..."
            ax.text(
                x,
                y,
                short_rule,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )

        # ===== GROUNDING SEPARATOR =====
        ax.plot([0.5, 9.5], [6.8, 6.8], "k--", linewidth=2, alpha=0.5)
        ax.text(
            5,
            6.9,
            "Grounding Function 𝒢",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            style="italic",
            color=COLORS["text"],
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        # ===== NEURAL LAYER (Bottom) =====
        ax.text(
            0.5,
            6.3,
            "Neural Layer",
            fontsize=12,
            fontweight="bold",
            color=COLORS["text"],
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=COLORS["text"]),
        )

        # Embeddings
        embedding_y = 5.3
        embedding_positions = {}
        for i, concept in enumerate(self.concepts):
            x = 1 + i * 1.8
            y = embedding_y
            embedding_positions[concept] = (x, y)

            # Draw embedding representation
            if show_embeddings:
                # Show as mini-vector
                for j in range(5):  # Show first 5 dimensions
                    val = self.embeddings[concept][j]
                    color_intensity = (val + 3) / 6  # Normalize to [0,1]
                    rect = mpatches.Rectangle(
                        (x - 0.3 + j * 0.12, y - 0.15),
                        0.1,
                        0.3,
                        facecolor=plt.cm.viridis(color_intensity),
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    ax.add_patch(rect)

                # Dimension label
                ax.text(
                    x,
                    y - 0.45,
                    f"ℝ^{self.embedding_dim}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    style="italic",
                    color=COLORS["text"],
                )
            else:
                # Simple circle representation
                circle = Circle(
                    (x, y),
                    0.15,
                    facecolor=COLORS["embedding"],
                    edgecolor="white",
                    linewidth=2,
                    alpha=0.8,
                )
                ax.add_patch(circle)

        # MLPs for predicates
        mlp_y = 4.0
        mlp_positions = {}
        for i, predicate in enumerate(self.predicates):
            x = 1.5 + i * 2.0
            y = mlp_y
            mlp_positions[predicate] = (x, y)

            # Draw MLP layers
            config = self.mlp_configs[predicate]
            layer_width = 0.6
            layer_spacing = 0.3
            total_width = len(config) * layer_spacing
            start_x = x - total_width / 2

            for j, layer_size in enumerate(config):
                layer_x = start_x + j * layer_spacing
                height = 0.1 + (layer_size / 200) * 0.4  # Scale by size

                rect = mpatches.Rectangle(
                    (layer_x - 0.08, y - height / 2),
                    0.16,
                    height,
                    facecolor=COLORS["mlp"],
                    edgecolor="white",
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax.add_patch(rect)

                # Connect layers
                if j < len(config) - 1:
                    ax.plot(
                        [layer_x + 0.08, layer_x + layer_spacing - 0.08],
                        [y, y],
                        "k-",
                        linewidth=1,
                        alpha=0.4,
                    )

            # MLP label
            ax.text(
                x,
                y - 0.45,
                "MLP→σ",
                ha="center",
                va="top",
                fontsize=8,
                style="italic",
                color=COLORS["text"],
            )

        # Fuzzy logic operations
        fuzzy_y = 2.5
        operations = ["AND (∧)", "OR (∨)", "IMPLIES (→)"]
        for i, op in enumerate(operations):
            x = 2.5 + i * 2.0
            y = fuzzy_y

            # Draw operation node
            circle = Circle(
                (x, y),
                0.25,
                facecolor=COLORS["fuzzy_op"],
                edgecolor="white",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(circle)

            # Operation label
            ax.text(
                x,
                y,
                op.split()[0],
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )

            # Formula below
            formulas = ["max(0, u+v-1)", "min(1, u+v)", "min(1, 1-u+v)"]
            ax.text(
                x,
                y - 0.45,
                formulas[i],
                ha="center",
                va="top",
                fontsize=7,
                family="monospace",
                color=COLORS["text"],
            )

        # ===== GROUNDING ARROWS =====
        # Concepts → Embeddings
        for concept in self.concepts:
            x1, y1 = concept_positions[concept]
            x2, y2 = embedding_positions[concept]

            arrow = FancyArrowPatch(
                (x1, y1 - 0.3),
                (x2, y2 + 0.2),
                arrowstyle="->",
                mutation_scale=15,
                linewidth=1.5,
                color=COLORS["edge"],
                alpha=0.6,
            )
            ax.add_patch(arrow)

        # Predicates → MLPs
        for predicate in self.predicates:
            x1, y1 = predicate_positions[predicate]
            x2, y2 = mlp_positions[predicate]

            arrow = FancyArrowPatch(
                (x1, y1 - 0.3),
                (x2, y2 + 0.5),
                arrowstyle="->",
                mutation_scale=15,
                linewidth=1.5,
                color=COLORS["edge"],
                alpha=0.6,
            )
            ax.add_patch(arrow)

        # Gradient flow (if enabled)
        if show_gradients:
            # Show backprop from fuzzy ops to embeddings
            for i in range(min(2, len(self.concepts))):
                concept = self.concepts[i]
                x1, y1 = embedding_positions[concept]
                x2, y2 = 2.5, fuzzy_y

                arrow = FancyArrowPatch(
                    (x2, y2),
                    (x1, y1),
                    arrowstyle="<-",
                    mutation_scale=12,
                    linewidth=2,
                    color="red",
                    alpha=0.4,
                    linestyle="dashed",
                )
                ax.add_patch(arrow)

            # Gradient label
            ax.text(
                1.5,
                3.5,
                "∇ Gradient Flow",
                fontsize=9,
                style="italic",
                color="red",
                alpha=0.7,
            )

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=COLORS["concept"], label="Concepts"),
            mpatches.Patch(facecolor=COLORS["predicate"], label="Predicates"),
            mpatches.Patch(facecolor=COLORS["rule"], label="Rules"),
            mpatches.Patch(facecolor=COLORS["embedding"], label="Embeddings"),
            mpatches.Patch(facecolor=COLORS["mlp"], label="MLPs"),
            mpatches.Patch(facecolor=COLORS["fuzzy_op"], label="Fuzzy Ops"),
        ]
        ax.legend(
            handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.95
        )

    def _plot_fuzzy_operations(self, ax: plt.Axes) -> None:
        """Plot Łukasiewicz fuzzy logic operations."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 3.5)
        ax.axis("off")
        ax.set_facecolor(COLORS["background"])

        # Title
        ax.text(
            0.5,
            3.3,
            "B. Łukasiewicz T-norms",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=COLORS["text"],
        )

        # Create mini-plots for each operation
        operations = [
            ("AND (∧)", self.lukasiewicz_and, 2.3, "max(0, u+v-1)"),
            ("OR (∨)", self.lukasiewicz_or, 1.4, "min(1, u+v)"),
            ("IMPLIES (→)", self.lukasiewicz_implies, 0.5, "min(1, 1-u+v)"),
        ]

        for op_name, op_func, y_center, formula in operations:
            # Create sample data
            u_vals = np.linspace(0, 1, 50)
            v_fixed = 0.6  # Fix v for 1D plot
            results = [op_func(u, v_fixed) for u in u_vals]

            # Mini-plot area
            plot_width = 0.35
            plot_height = 0.6
            plot_left = 0.05
            plot_bottom = y_center - plot_height / 2

            # Create inset axes for the plot
            inset = ax.inset_axes(
                [plot_left, plot_bottom, plot_width, plot_height],
                transform=ax.transData,
            )

            # Plot operation
            inset.plot(
                u_vals,
                results,
                linewidth=2.5,
                color=COLORS["fuzzy_op"],
                label=f"{op_name}",
            )
            inset.axhline(
                y=v_fixed, linestyle="--", color=COLORS["text"], alpha=0.3, linewidth=1
            )

            inset.set_xlim(0, 1)
            inset.set_ylim(0, 1)
            inset.set_xlabel("Truth(P)", fontsize=8)
            inset.set_ylabel("Result", fontsize=8)
            inset.tick_params(labelsize=7)
            inset.grid(True, alpha=0.3)

            # Operation label
            ax.text(
                0.45,
                y_center + 0.25,
                op_name,
                fontsize=11,
                fontweight="bold",
                color=COLORS["fuzzy_op"],
            )

            # Formula
            ax.text(
                0.45,
                y_center,
                f"f(u,v) = {formula}",
                fontsize=9,
                family="monospace",
                color=COLORS["text"],
            )

            # Example values
            example_u = 0.7
            example_v = 0.6
            example_result = op_func(example_u, example_v)

            ax.text(
                0.45,
                y_center - 0.15,
                f"Example: f({example_u:.1f}, {example_v:.1f}) = {example_result:.2f}",
                fontsize=8,
                style="italic",
                color=COLORS["text"],
                alpha=0.7,
            )

            # Differentiability note
            if y_center > 2.0:  # Only on first one
                ax.text(
                    0.45,
                    y_center - 0.30,
                    "✓ Continuous & Differentiable",
                    fontsize=8,
                    color=COLORS["concept"],
                    fontweight="bold",
                )

        # Comparison note
        ax.text(
            0.5,
            0.1,
            "Unlike Boolean logic (step functions),\n"
            "Łukasiewicz logic provides smooth gradients\n"
            "enabling backpropagation through logic.",
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=COLORS["concept"],
                linewidth=2,
                alpha=0.9,
            ),
        )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate Visual 4: LTN Grounding Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display interactive visualization
  python visual_4_ltn_grounding.py --show

  # Save high-resolution output
  python visual_4_ltn_grounding.py --save-all --output-dir ../outputs

  # Minimal visualization (no embeddings/gradients)
  python visual_4_ltn_grounding.py --show --no-embeddings --no-gradients
        """,
    )

    parser.add_argument(
        "--show", action="store_true", help="Display the visualization interactively"
    )
    parser.add_argument(
        "--save-all", action="store_true", help="Save all visualization variants"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../outputs",
        help="Output directory for saved figures (default: ../outputs)",
    )
    parser.add_argument(
        "--no-embeddings", action="store_true", help="Hide embedding vector details"
    )
    parser.add_argument(
        "--no-gradients", action="store_true", help="Hide gradient flow arrows"
    )
    parser.add_argument(
        "--num-concepts",
        type=int,
        default=5,
        help="Number of concepts to display (default: 5)",
    )
    parser.add_argument(
        "--num-predicates",
        type=int,
        default=4,
        help="Number of predicates to display (default: 4)",
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

    # Create LTN grounding graph
    print("Generating LTN Grounding Graph...")
    ltn = LTNGroundingGraph(
        num_concepts=args.num_concepts,
        num_predicates=args.num_predicates,
        num_rules=3,
        embedding_dim=64,
        random_seed=args.seed,
    )

    # Generate visualization
    fig = ltn.create_visualization(
        show_embeddings=not args.no_embeddings, show_gradients=not args.no_gradients
    )

    # Save if requested
    if args.save_all:
        output_file = output_path / "visual_4_ltn_grounding.png"
        print(f"Saving to {output_file}...")
        fig.savefig(
            output_file,
            dpi=args.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"✓ Saved: {output_file}")

        # Save minimal version (no embeddings/gradients)
        fig_minimal = ltn.create_visualization(
            show_embeddings=False, show_gradients=False
        )
        output_file_minimal = output_path / "visual_4_ltn_grounding_minimal.png"
        print(f"Saving minimal version to {output_file_minimal}...")
        fig_minimal.savefig(
            output_file_minimal,
            dpi=args.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"✓ Saved: {output_file_minimal}")
        plt.close(fig_minimal)

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
