#!/usr/bin/env python3
"""
Visual 7: OpAL Decision Engine - Dual Pathway Visualization

This module implements the Opponent Actor Learning (OpAL) decision engine
visualization, showing both the circuit diagram (Direct/Indirect pathways)
and the decision space with historical trajectory.

Theoretical Foundation:
- OpAL models the basal ganglia's dual pathway architecture
- Direct pathway (D1 SPNs): "Go" signal, potentiates action
- Indirect pathway (D2 SPNs): "No-Go" signal, suppresses action
- Dopamine (RPE) modulates both pathways with opposing plasticity

Reference:
Collins, A. G., & Frank, M. J. (2014). Opponent actor learning (OpAL):
Modeling interactive effects of striatal dopamine on reinforcement learning
and choice incentive. Psychological Review, 121(3), 337.

Author: Project JANUS Visualization Team
License: MIT
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

warnings.filterwarnings("ignore")


class OpALDecisionEngine:
    """
    Opponent Actor Learning decision engine with visualization capabilities.

    This class simulates the basal ganglia's dual pathway architecture for
    decision making under uncertainty.
    """

    def __init__(
        self,
        theta_buy: float = 0.2,
        theta_sell: float = -0.2,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ):
        """
        Initialize OpAL decision engine.

        Parameters
        ----------
        theta_buy : float
            Decision threshold for BUY action (G - N > theta_buy)
        theta_sell : float
            Decision threshold for SELL action (G - N < theta_sell)
        learning_rate : float
            Plasticity rate for pathway weight updates
        momentum : float
            Temporal smoothing factor (exponential moving average)
        """
        self.theta_buy = theta_buy
        self.theta_sell = theta_sell
        self.lr = learning_rate
        self.momentum = momentum

        # Pathway activations (smoothed)
        self.G = 0.0  # Direct pathway ("Go")
        self.N = 0.0  # Indirect pathway ("No-Go")

        # History tracking
        self.history: List[Dict] = []

    def compute_pathways(
        self, state: np.ndarray, W_direct: np.ndarray, W_indirect: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute Direct and Indirect pathway activations.

        Parameters
        ----------
        state : np.ndarray, shape (n_features,)
            Current market state representation
        W_direct : np.ndarray, shape (n_features,)
            Direct pathway weights
        W_indirect : np.ndarray, shape (n_features,)
            Indirect pathway weights

        Returns
        -------
        G : float
            Direct pathway activation (Go signal)
        N : float
            Indirect pathway activation (No-Go signal)
        """
        # ReLU activation for biological plausibility
        G_raw = np.maximum(0, np.dot(W_direct, state))
        N_raw = np.maximum(0, np.dot(W_indirect, state))

        # Normalize to [0, 1]
        G_raw = np.clip(G_raw, 0, 1)
        N_raw = np.clip(N_raw, 0, 1)

        # Apply momentum (exponential moving average)
        self.G = self.momentum * self.G + (1 - self.momentum) * G_raw
        self.N = self.momentum * self.N + (1 - self.momentum) * N_raw

        return self.G, self.N

    def make_decision(self, G: float, N: float) -> str:
        """
        Make trading decision based on pathway competition.

        Parameters
        ----------
        G : float
            Direct pathway activation
        N : float
            Indirect pathway activation

        Returns
        -------
        decision : str
            One of 'BUY', 'SELL', 'HOLD'
        """
        delta = G - N

        if delta > self.theta_buy:
            return "BUY"
        elif delta < self.theta_sell:
            return "SELL"
        else:
            return "HOLD"

    def update_weights(
        self,
        W_direct: np.ndarray,
        W_indirect: np.ndarray,
        state: np.ndarray,
        rpe: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update pathway weights based on reward prediction error (RPE).

        Opposing plasticity rules:
        - Positive RPE: Strengthen Direct, Weaken Indirect
        - Negative RPE: Weaken Direct, Strengthen Indirect

        Parameters
        ----------
        W_direct : np.ndarray
            Direct pathway weights
        W_indirect : np.ndarray
            Indirect pathway weights
        state : np.ndarray
            Current state
        rpe : float
            Reward prediction error (dopamine signal)

        Returns
        -------
        W_direct : np.ndarray
            Updated direct pathway weights
        W_indirect : np.ndarray
            Updated indirect pathway weights
        """
        # Opposing plasticity
        W_direct = W_direct + self.lr * rpe * state
        W_indirect = W_indirect - self.lr * rpe * state

        return W_direct, W_indirect

    def add_history(
        self,
        G: float,
        N: float,
        decision: str,
        rpe: float,
        profit: Optional[float] = None,
    ):
        """Add current state to history for visualization."""
        self.history.append(
            {"G": G, "N": N, "decision": decision, "rpe": rpe, "profit": profit}
        )


def generate_synthetic_trajectory(
    n_steps: int = 1000, regime: str = "volatile", seed: int = 42
) -> OpALDecisionEngine:
    """
    Generate synthetic OpAL trajectory for demonstration.

    Parameters
    ----------
    n_steps : int
        Number of time steps to simulate
    regime : str
        Market regime: 'volatile', 'trending', 'choppy'
    seed : int
        Random seed for reproducibility

    Returns
    -------
    engine : OpALDecisionEngine
        Populated engine with history
    """
    np.random.seed(seed)

    engine = OpALDecisionEngine(theta_buy=0.2, theta_sell=-0.2)

    # Initialize weights
    n_features = 10
    W_direct = np.random.randn(n_features) * 0.1
    W_indirect = np.random.randn(n_features) * 0.1

    # Regime-specific parameters
    if regime == "volatile":
        noise_scale = 0.5
        trend = 0.0
    elif regime == "trending":
        noise_scale = 0.2
        trend = 0.002
    else:  # choppy
        noise_scale = 0.3
        trend = 0.0

    price = 100.0

    for t in range(n_steps):
        # Generate synthetic market state
        price_return = trend + np.random.randn() * noise_scale
        price *= 1 + price_return

        # Create state vector (simplified)
        state = np.random.randn(n_features) * 0.5
        state[0] = price_return  # Recent return
        state[1] = np.random.randn() * noise_scale  # Volatility proxy

        # Compute pathways
        G, N = engine.compute_pathways(state, W_direct, W_indirect)

        # Make decision
        decision = engine.make_decision(G, N)

        # Simulate outcome and RPE
        if decision == "BUY":
            profit = price_return * 100  # Simplified P&L
        elif decision == "SELL":
            profit = -price_return * 100
        else:
            profit = 0.0

        # Compute RPE (simplified TD error)
        expected_value = G - N
        rpe = profit - expected_value

        # Update weights
        W_direct, W_indirect = engine.update_weights(W_direct, W_indirect, state, rpe)

        # Record history
        engine.add_history(G, N, decision, rpe, profit)

    return engine


def plot_circuit_diagram(ax: plt.Axes, G: float, N: float, rpe: float):
    """
    Plot the basal ganglia circuit diagram (Panel A).

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    G : float
        Current direct pathway activation
    N : float
        Current indirect pathway activation
    rpe : float
        Current reward prediction error (dopamine)
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_aspect("equal")

    # Define node positions
    nodes = {
        "Cortex": (5, 9),
        "Striatum_D1": (2, 6),
        "Striatum_D2": (8, 6),
        "SNc": (5, 3),
        "STN_GPe": (8, 4),
        "Thalamus": (5, 1),
    }

    # Color coding
    color_direct = "#2E8B57"  # SeaGreen
    color_indirect = "#DC143C"  # Crimson
    color_dopamine = "#FFD700"  # Gold

    # Draw nodes
    for name, (x, y) in nodes.items():
        if name == "Cortex":
            color = "#4682B4"  # SteelBlue
            label = "Cortex\n(State Input)"
        elif name == "Striatum_D1":
            color = color_direct
            label = f"Striatum D1\n(Direct)\nG={G:.2f}"
        elif name == "Striatum_D2":
            color = color_indirect
            label = f"Striatum D2\n(Indirect)\nN={N:.2f}"
        elif name == "SNc":
            color = color_dopamine
            label = f"SNc\n(Dopamine)\nδ={rpe:.2f}"
        elif name == "STN_GPe":
            color = "#8B7D6B"  # Gray
            label = "STN/GPe"
        else:  # Thalamus
            color = "#9370DB"  # MediumPurple
            label = "Thalamus\n(Action Output)"

        circle = Circle((x, y), 0.8, color=color, ec="black", lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    # Draw pathways
    # Direct pathway: Cortex -> D1 -> Thalamus
    arrow_direct = FancyArrowPatch(
        nodes["Cortex"],
        nodes["Striatum_D1"],
        arrowstyle="->",
        mutation_scale=30,
        lw=3 + G * 5,
        color=color_direct,
        zorder=2,
        alpha=0.7,
    )
    ax.add_patch(arrow_direct)

    arrow_direct2 = FancyArrowPatch(
        nodes["Striatum_D1"],
        nodes["Thalamus"],
        arrowstyle="->",
        mutation_scale=30,
        lw=3 + G * 5,
        color=color_direct,
        zorder=2,
        alpha=0.7,
    )
    ax.add_patch(arrow_direct2)

    # Indirect pathway: Cortex -> D2 -> STN/GPe -> Thalamus (inhibitory)
    arrow_indirect = FancyArrowPatch(
        nodes["Cortex"],
        nodes["Striatum_D2"],
        arrowstyle="->",
        mutation_scale=30,
        lw=3 + N * 5,
        color=color_indirect,
        zorder=2,
        alpha=0.7,
    )
    ax.add_patch(arrow_indirect)

    arrow_indirect2 = FancyArrowPatch(
        nodes["Striatum_D2"],
        nodes["STN_GPe"],
        arrowstyle="->",
        mutation_scale=30,
        lw=3 + N * 5,
        color=color_indirect,
        zorder=2,
        alpha=0.7,
    )
    ax.add_patch(arrow_indirect2)

    arrow_indirect3 = FancyArrowPatch(
        nodes["STN_GPe"],
        nodes["Thalamus"],
        arrowstyle="-|",
        mutation_scale=30,
        lw=3 + N * 5,
        color=color_indirect,
        zorder=2,
        alpha=0.7,
        linestyle="--",
    )
    ax.add_patch(arrow_indirect3)

    # Dopamine modulation (bidirectional)
    arrow_dopa_d1 = FancyArrowPatch(
        nodes["SNc"],
        nodes["Striatum_D1"],
        arrowstyle="<->",
        mutation_scale=20,
        lw=2,
        color=color_dopamine,
        zorder=1,
        alpha=0.5,
        linestyle=":",
    )
    ax.add_patch(arrow_dopa_d1)

    arrow_dopa_d2 = FancyArrowPatch(
        nodes["SNc"],
        nodes["Striatum_D2"],
        arrowstyle="<->",
        mutation_scale=20,
        lw=2,
        color=color_dopamine,
        zorder=1,
        alpha=0.5,
        linestyle=":",
    )
    ax.add_patch(arrow_dopa_d2)

    # Plasticity annotations
    if rpe > 0:
        plasticity_text = "δ > 0:\n↑ Direct\n↓ Indirect"
        text_color = color_direct
    elif rpe < 0:
        plasticity_text = "δ < 0:\n↓ Direct\n↑ Indirect"
        text_color = color_indirect
    else:
        plasticity_text = "δ = 0:\nNo Change"
        text_color = "gray"

    ax.text(
        0.5,
        5,
        plasticity_text,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=text_color, lw=2),
        color=text_color,
        fontweight="bold",
    )

    ax.set_title(
        "Panel A: Basal Ganglia Circuit", fontsize=14, fontweight="bold", pad=10
    )


def plot_decision_space(
    ax: plt.Axes, engine: OpALDecisionEngine, current_idx: Optional[int] = None
):
    """
    Plot the decision space with historical trajectory (Panel B).

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    engine : OpALDecisionEngine
        Engine with history
    current_idx : int, optional
        Index of current state to highlight (default: last state)
    """
    if current_idx is None:
        current_idx = len(engine.history) - 1

    # Extract history
    G_history = [h["G"] for h in engine.history]
    N_history = [h["N"] for h in engine.history]
    decisions = [h["decision"] for h in engine.history]
    profits = [h.get("profit", 0) for h in engine.history]

    # Decision regions
    G_range = np.linspace(0, 1, 100)
    N_range = np.linspace(0, 1, 100)
    G_grid, N_grid = np.meshgrid(G_range, N_range)

    # Compute decision for each point
    decision_grid = np.zeros_like(G_grid)
    for i in range(len(G_range)):
        for j in range(len(N_range)):
            delta = G_grid[j, i] - N_grid[j, i]
            if delta > engine.theta_buy:
                decision_grid[j, i] = 1  # BUY
            elif delta < engine.theta_sell:
                decision_grid[j, i] = -1  # SELL
            else:
                decision_grid[j, i] = 0  # HOLD

    # Plot decision regions
    buy_mask = decision_grid == 1
    sell_mask = decision_grid == -1
    hold_mask = decision_grid == 0

    ax.contourf(
        G_grid,
        N_grid,
        buy_mask.astype(float),
        levels=[0.5, 1.5],
        colors=["#90EE90"],
        alpha=0.3,
    )
    ax.contourf(
        G_grid,
        N_grid,
        sell_mask.astype(float),
        levels=[0.5, 1.5],
        colors=["#FFB6C1"],
        alpha=0.3,
    )
    ax.contourf(
        G_grid,
        N_grid,
        hold_mask.astype(float),
        levels=[0.5, 1.5],
        colors=["#D3D3D3"],
        alpha=0.3,
    )

    # Decision boundaries
    ax.plot(
        [0, 1],
        [engine.theta_buy, 1 + engine.theta_buy],
        "g--",
        lw=2,
        label=f"BUY boundary (Δ={engine.theta_buy})",
    )
    ax.plot(
        [0, 1],
        [engine.theta_sell, 1 + engine.theta_sell],
        "r--",
        lw=2,
        label=f"SELL boundary (Δ={engine.theta_sell})",
    )

    # Historical trajectory (colored by profit)
    profits_array = np.array(profits)
    # Normalize profits for color mapping
    if len(profits_array) > 0 and profits_array.std() > 0:
        profits_norm = (profits_array - profits_array.mean()) / profits_array.std()
        profits_norm = np.clip(profits_norm, -2, 2)
    else:
        profits_norm = np.zeros_like(profits_array)

    # Color map: green for profit, red for loss
    colors = plt.cm.RdYlGn((profits_norm + 2) / 4)  # Map [-2, 2] to [0, 1]

    ax.scatter(G_history, N_history, c=colors, s=10, alpha=0.4, edgecolors="none")

    # Current state
    if 0 <= current_idx < len(engine.history):
        current_G = engine.history[current_idx]["G"]
        current_N = engine.history[current_idx]["N"]
        current_decision = engine.history[current_idx]["decision"]

        ax.scatter(
            current_G,
            current_N,
            s=200,
            c="blue",
            marker="*",
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label="Current State",
        )

        # Decision annotation
        if current_decision == "BUY":
            dec_color = "green"
        elif current_decision == "SELL":
            dec_color = "red"
        else:
            dec_color = "gray"

        ax.text(
            0.95,
            0.05,
            f"Decision: {current_decision}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=dec_color,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor=dec_color,
                lw=2,
                alpha=0.9,
            ),
        )

    ax.set_xlabel("Direct Pathway Activation (G)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Indirect Pathway Activation (N)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(
        "Panel B: Decision Space & Historical Trajectory",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )


def visualize_opal_decision(
    engine: OpALDecisionEngine,
    current_idx: Optional[int] = None,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Create complete OpAL decision visualization (V7).

    Parameters
    ----------
    engine : OpALDecisionEngine
        Engine with trajectory history
    current_idx : int, optional
        Time step to visualize (default: last)
    save_path : Path, optional
        Output path for PNG
    dpi : int
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.Figure
        The complete figure
    """
    if current_idx is None:
        current_idx = len(engine.history) - 1

    if current_idx < 0 or current_idx >= len(engine.history):
        raise ValueError(f"Invalid current_idx: {current_idx}")

    # Extract current state
    current = engine.history[current_idx]
    G = current["G"]
    N = current["N"]
    rpe = current["rpe"]

    # Create figure
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Panel A: Circuit diagram
    plot_circuit_diagram(ax1, G, N, rpe)

    # Panel B: Decision space
    plot_decision_space(ax2, engine, current_idx)

    # Overall title
    fig.suptitle(
        "Visual 7: OpAL Decision Engine - Dual Pathway Architecture",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Metadata footer
    metadata = (
        f"Time Step: {current_idx + 1}/{len(engine.history)} | "
        f"Thresholds: θ_buy={engine.theta_buy}, θ_sell={engine.theta_sell}"
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
        description="Visual 7: OpAL Decision Engine Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visual_7_opal_decision.py --show
  python visual_7_opal_decision.py --save-all --output-dir ../outputs
  python visual_7_opal_decision.py --regime trending --steps 2000 --show
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
        "--regime",
        type=str,
        default="volatile",
        choices=["volatile", "trending", "choppy"],
        help="Market regime for simulation",
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of time steps to simulate"
    )
    parser.add_argument(
        "--current-step",
        type=int,
        default=None,
        help="Time step to visualize (default: last)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for saved figures"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Visual 7: OpAL Decision Engine - Dual Pathway Visualization")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Market regime: {args.regime}")
    print(f"  Time steps: {args.steps}")
    print(f"  Random seed: {args.seed}")

    # Generate synthetic trajectory
    print(f"\n[1/2] Generating synthetic OpAL trajectory...")
    engine = generate_synthetic_trajectory(
        n_steps=args.steps, regime=args.regime, seed=args.seed
    )
    print(f"✓ Generated {len(engine.history)} time steps")

    # Statistics
    decisions = [h["decision"] for h in engine.history]
    buy_count = decisions.count("BUY")
    sell_count = decisions.count("SELL")
    hold_count = decisions.count("HOLD")

    print(f"\nDecision Statistics:")
    print(f"  BUY:  {buy_count:4d} ({100 * buy_count / len(decisions):5.1f}%)")
    print(f"  SELL: {sell_count:4d} ({100 * sell_count / len(decisions):5.1f}%)")
    print(f"  HOLD: {hold_count:4d} ({100 * hold_count / len(decisions):5.1f}%)")

    # Determine current step
    current_idx = (
        args.current_step if args.current_step is not None else len(engine.history) - 1
    )

    # Create visualization
    print(f"\n[2/2] Creating OpAL visualization (step {current_idx + 1})...")

    save_path = None
    if args.save_all:
        save_path = args.output_dir / f"visual_7_opal_decision_{args.regime}.png"

    fig = visualize_opal_decision(
        engine=engine, current_idx=current_idx, save_path=save_path, dpi=args.dpi
    )

    print(f"\n{'=' * 70}")
    print("✓ Visualization complete!")
    print(f"{'=' * 70}")

    if args.save_all:
        print(f"\nOutput saved to: {args.output_dir}/")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
