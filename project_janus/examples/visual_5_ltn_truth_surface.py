#!/usr/bin/env python3
"""
Visual 5: Łukasiewicz T-Norm Truth Surface Visualization
=========================================================

Reference implementation demonstrating differentiable logic landscapes.
This visualization shows why Łukasiewicz fuzzy logic is suitable for
gradient-based optimization, unlike discrete Boolean logic.

Part of Project JANUS visualization specification.

Usage:
    python visual_5_ltn_truth_surface.py --operation and --output truth_and.png
    python visual_5_ltn_truth_surface.py --operation or --save-all
    python visual_5_ltn_truth_surface.py --operation implies --show-gradients

Dependencies:
    pip install numpy matplotlib

Author: Project JANUS Team
License: MIT
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# ŁUKASIEWICZ LOGIC OPERATORS
# ============================================================================


def lukasiewicz_and(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Łukasiewicz T-norm (AND operation).

    Formula: p ∧ q = max(0, p + q - 1)

    This is differentiable with respect to both p and q, enabling
    backpropagation through logical constraints.

    Args:
        p: Truth value(s) of first predicate, range [0, 1]
        q: Truth value(s) of second predicate, range [0, 1]

    Returns:
        Truth value of p AND q, range [0, 1]
    """
    return np.maximum(0, p + q - 1)


def lukasiewicz_or(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Łukasiewicz T-conorm (OR operation).

    Formula: p ∨ q = min(1, p + q)

    Args:
        p: Truth value(s) of first predicate
        q: Truth value(s) of second predicate

    Returns:
        Truth value of p OR q
    """
    return np.minimum(1, p + q)


def lukasiewicz_implies(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Łukasiewicz implication.

    Formula: p ⇒ q = min(1, 1 - p + q)

    Args:
        p: Truth value(s) of antecedent
        q: Truth value(s) of consequent

    Returns:
        Truth value of p IMPLIES q
    """
    return np.minimum(1, 1 - p + q)


def lukasiewicz_not(p: np.ndarray) -> np.ndarray:
    """
    Łukasiewicz negation.

    Formula: ¬p = 1 - p

    Args:
        p: Truth value(s) of predicate

    Returns:
        Truth value of NOT p
    """
    return 1 - p


# ============================================================================
# BOOLEAN LOGIC OPERATORS (FOR COMPARISON)
# ============================================================================


def boolean_and(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Standard Boolean AND: p ∧ q = p · q"""
    return p * q


def boolean_or(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Standard Boolean OR: p ∨ q = p + q - p·q"""
    return p + q - p * q


def boolean_implies(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Standard Boolean IMPLIES: p ⇒ q = (p ≤ q)"""
    return np.where(p <= q, 1.0, 0.0)


# ============================================================================
# GRADIENT COMPUTATION
# ============================================================================


def compute_gradient_lukasiewicz_and(
    p: float, q: float, delta: float = 0.01
) -> Tuple[float, float]:
    """
    Numerically compute gradient of Łukasiewicz AND.

    Args:
        p: Point p-coordinate
        q: Point q-coordinate
        delta: Finite difference step size

    Returns:
        (grad_p, grad_q): Partial derivatives
    """
    # ∂(p∧q)/∂p
    grad_p = (lukasiewicz_and(p + delta, q) - lukasiewicz_and(p, q)) / delta

    # ∂(p∧q)/∂q
    grad_q = (lukasiewicz_and(p, q + delta) - lukasiewicz_and(p, q)) / delta

    return grad_p, grad_q


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def visualize_truth_surface(
    operation: str = "and",
    save_path: Optional[str] = None,
    show_gradients: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    """
    Generate 3D surface plot comparing Łukasiewicz and Boolean logic.

    Implements Visual 5 specification from JANUS documentation.

    Args:
        operation: Logic operation - 'and', 'or', or 'implies'
        save_path: If provided, save figure to this path
        show_gradients: If True, overlay gradient vectors
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    # Create meshgrid for truth values
    resolution = 50
    p = np.linspace(0, 1, resolution)
    q = np.linspace(0, 1, resolution)
    P, Q = np.meshgrid(p, q)

    # Compute truth values for selected operation
    if operation == "and":
        Z_lukasiewicz = lukasiewicz_and(P, Q)
        Z_boolean = boolean_and(P, Q)
        title = "Łukasiewicz AND (p ∧ q)"
        formula = r"$\max(0, p + q - 1)$"
        bool_label = r"Boolean: $p \cdot q$"
    elif operation == "or":
        Z_lukasiewicz = lukasiewicz_or(P, Q)
        Z_boolean = boolean_or(P, Q)
        title = "Łukasiewicz OR (p ∨ q)"
        formula = r"$\min(1, p + q)$"
        bool_label = r"Boolean: $p + q - pq$"
    elif operation == "implies":
        Z_lukasiewicz = lukasiewicz_implies(P, Q)
        Z_boolean = boolean_implies(P, Q)
        title = "Łukasiewicz IMPLIES (p ⇒ q)"
        formula = r"$\min(1, 1 - p + q)$"
        bool_label = "Boolean: step function"
    else:
        raise ValueError(
            f"Unknown operation: {operation}. Use 'and', 'or', or 'implies'"
        )

    # Create figure with two 3D subplots
    fig = plt.figure(figsize=(16, 6))

    # ========================================================================
    # LEFT PANEL: Łukasiewicz Logic (Differentiable)
    # ========================================================================
    ax1 = fig.add_subplot(121, projection="3d")

    # Plot smooth surface
    surf1 = ax1.plot_surface(
        P,
        Q,
        Z_lukasiewicz,
        cmap=cm.viridis,
        alpha=0.85,
        edgecolor="none",
        antialiased=True,
    )

    # Add gradient vectors if requested
    if show_gradients and operation == "and":
        sample_points = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]

        for p_val, q_val in sample_points:
            grad_p, grad_q = compute_gradient_lukasiewicz_and(p_val, q_val)
            z_val = lukasiewicz_and(p_val, q_val)

            # Scale gradients for visibility
            scale = 0.15
            ax1.quiver(
                p_val,
                q_val,
                z_val,
                grad_p * scale,
                grad_q * scale,
                0.05,
                color="red",
                arrow_length_ratio=0.3,
                linewidth=2.5,
                alpha=0.9,
            )

    # Formatting
    ax1.set_xlabel("Truth(P)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Truth(Q)", fontsize=11, labelpad=8)
    ax1.set_zlabel("Truth(P ∧ Q)", fontsize=11, labelpad=8)
    ax1.set_title(
        f"{title}\n{formula}\n(Differentiable ✓)",
        fontweight="bold",
        fontsize=12,
        pad=15,
    )
    ax1.view_init(elev=25, azim=45)

    # Set axis limits
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)

    # Add colorbar
    cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
    cbar1.set_label("Truth Value", fontsize=10)

    # ========================================================================
    # RIGHT PANEL: Boolean Logic (Non-differentiable)
    # ========================================================================
    ax2 = fig.add_subplot(122, projection="3d")

    # Plot as wireframe to emphasize discrete nature
    surf2 = ax2.plot_wireframe(
        P, Q, Z_boolean, color="gray", alpha=0.6, linewidth=0.8, rstride=2, cstride=2
    )

    # Formatting
    ax2.set_xlabel("Truth(P)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Truth(Q)", fontsize=11, labelpad=8)
    ax2.set_zlabel("Truth(P ∧ Q)", fontsize=11, labelpad=8)
    ax2.set_title(
        f"{bool_label}\n(Non-differentiable ✗)", fontweight="bold", fontsize=12, pad=15
    )
    ax2.view_init(elev=25, azim=45)

    # Set axis limits
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_zlim(0, 1)

    # ========================================================================
    # OVERALL TITLE
    # ========================================================================
    gradient_note = (
        "\nGradient vectors (red arrows) enable backpropagation"
        if show_gradients
        else ""
    )
    plt.suptitle(
        f"Visual 5: Łukasiewicz T-Norm Landscape{gradient_note}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")

    return fig


def visualize_all_operations(
    output_dir: str = "outputs", show_gradients: bool = True, dpi: int = 300
) -> None:
    """
    Generate all three logic operation visualizations.

    Args:
        output_dir: Directory to save outputs
        show_gradients: Whether to show gradient vectors
        dpi: Resolution for saved figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    operations = ["and", "or", "implies"]

    for op in operations:
        save_file = output_path / f"visual_5_ltn_{op}.png"
        print(f"Generating {op.upper()} visualization...")

        fig = visualize_truth_surface(
            operation=op,
            save_path=str(save_file),
            show_gradients=(show_gradients and op == "and"),  # Only AND gets gradients
            dpi=dpi,
        )
        plt.close(fig)

    print(f"\n✓ All visualizations saved to {output_dir}/")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Parse arguments and generate visualization."""
    parser = argparse.ArgumentParser(
        description="Generate Łukasiewicz logic truth surface visualizations (Visual 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --operation and --output truth_and.png
  %(prog)s --operation implies --show
  %(prog)s --save-all --output-dir results/
        """,
    )

    parser.add_argument(
        "--operation",
        choices=["and", "or", "implies"],
        default="and",
        help="Logic operation to visualize (default: and)",
    )

    parser.add_argument(
        "--output", "-o", type=str, help="Output file path (e.g., truth_surface.png)"
    )

    parser.add_argument(
        "--show", action="store_true", help="Display plot interactively"
    )

    parser.add_argument(
        "--show-gradients",
        action="store_true",
        default=True,
        help="Overlay gradient vectors (default: True for AND operation)",
    )

    parser.add_argument(
        "--no-gradients", action="store_true", help="Disable gradient vector overlay"
    )

    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Generate all three operations (AND, OR, IMPLIES)",
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

    # Determine gradient display
    show_gradients = args.show_gradients and not args.no_gradients

    # Generate visualization(s)
    if args.save_all:
        visualize_all_operations(
            output_dir=args.output_dir, show_gradients=show_gradients, dpi=args.dpi
        )
    else:
        fig = visualize_truth_surface(
            operation=args.operation,
            save_path=args.output,
            show_gradients=show_gradients,
            dpi=args.dpi,
        )

        if args.show:
            plt.show()
        elif not args.output:
            print(
                "Warning: No output path specified and --show not set. Use --output or --show."
            )
            plt.show()


if __name__ == "__main__":
    main()
