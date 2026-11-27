#!/usr/bin/env python3
"""
Visual 2: LOB vs GAF Comparison - Microstructure Fusion View

This module implements the side-by-side comparison of Limit Order Book (LOB)
heatmap and Gramian Angular Field (GAF) texture, demonstrating why multimodal
fusion is necessary for Project JANUS.

Theoretical Foundation:
- LOB shows market microstructure: where orders are resting (intent)
- GAF shows price kinematics: how price is actually moving (dynamics)
- Fusion layer combines both for complete market understanding

Key Insight:
LOB = Static snapshot of supply/demand
GAF = Dynamic signature of price trajectory
Together = Intent + Execution = Complete market state

Reference:
Sirignano, J., & Cont, R. (2019). Universal features of price formation in
financial markets: perspectives from deep learning. Quantitative Finance,
19(9), 1449-1459.

Author: Project JANUS Visualization Team
License: MIT
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")


class LOBSimulator:
    """
    Simulates Limit Order Book for visualization purposes.

    In production, this would connect to actual exchange data feed.
    """

    def __init__(self, mid_price: float = 100.0, tick_size: float = 0.01):
        """
        Initialize LOB simulator.

        Parameters
        ----------
        mid_price : float
            Current mid-market price
        tick_size : float
            Minimum price increment
        """
        self.mid_price = mid_price
        self.tick_size = tick_size

    def generate_lob_snapshot(
        self, n_levels: int = 20, regime: str = "normal"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic LOB snapshot.

        Parameters
        ----------
        n_levels : int
            Number of price levels on each side (bid/ask)
        regime : str
            Market regime: 'normal', 'imbalanced', 'wide_spread'

        Returns
        -------
        bid_prices : np.ndarray, shape (n_levels,)
            Bid price levels (descending)
        bid_volumes : np.ndarray, shape (n_levels,)
            Volume at each bid level
        ask_prices : np.ndarray, shape (n_levels,)
            Ask price levels (ascending)
        ask_volumes : np.ndarray, shape (n_levels,)
            Volume at each ask level
        """
        # Generate price levels
        if regime == "wide_spread":
            spread = 0.5
        else:
            spread = 0.05

        bid_prices = np.array(
            [self.mid_price - spread / 2 - i * self.tick_size for i in range(n_levels)]
        )
        ask_prices = np.array(
            [self.mid_price + spread / 2 + i * self.tick_size for i in range(n_levels)]
        )

        # Generate volumes based on regime
        np.random.seed(42)

        if regime == "imbalanced":
            # Heavy buy pressure
            bid_volumes = np.random.exponential(200, n_levels)
            ask_volumes = np.random.exponential(50, n_levels)
        elif regime == "wide_spread":
            # Sparse liquidity
            bid_volumes = np.random.exponential(30, n_levels)
            ask_volumes = np.random.exponential(30, n_levels)
        else:  # normal
            # Balanced
            bid_volumes = np.random.exponential(100, n_levels)
            ask_volumes = np.random.exponential(100, n_levels)

        # Add depth decay (volume decreases away from mid)
        depth_decay = np.exp(-0.1 * np.arange(n_levels))
        bid_volumes *= depth_decay
        ask_volumes *= depth_decay

        return bid_prices, bid_volumes, ask_prices, ask_volumes


def generate_gaf_from_price(prices: np.ndarray, method: str = "GASF") -> np.ndarray:
    """
    Generate Gramian Angular Field from price series.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Price time series
    method : str
        'GASF' or 'GADF'

    Returns
    -------
    gaf : np.ndarray, shape (T, T)
        Gramian Angular Field matrix
    """
    # Normalize to [-1, 1]
    prices_min = prices.min()
    prices_max = prices.max()

    if prices_max - prices_min < 1e-10:
        # Constant price - add small noise
        prices_norm = np.zeros_like(prices)
    else:
        prices_norm = 2 * (prices - prices_min) / (prices_max - prices_min) - 1

    # Clip to valid arccos domain
    prices_norm = np.clip(prices_norm, -1, 1)

    # Convert to polar coordinates
    phi = np.arccos(prices_norm)

    # Compute GAF
    n = len(phi)
    gaf = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if method == "GASF":
                # Gramian Angular Summation Field
                gaf[i, j] = np.cos(phi[i] + phi[j])
            else:  # GADF
                # Gramian Angular Difference Field
                gaf[i, j] = np.sin(phi[i] - phi[j])

    return gaf


def generate_synthetic_price_series(
    n_steps: int = 100, regime: str = "trending", seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic price series.

    Parameters
    ----------
    n_steps : int
        Number of time steps
    regime : str
        'trending', 'mean_reverting', 'volatile', 'quiet'
    seed : int
        Random seed

    Returns
    -------
    prices : np.ndarray, shape (n_steps,)
        Synthetic price series
    """
    np.random.seed(seed)

    if regime == "trending":
        drift = 0.002
        volatility = 0.005
        returns = np.random.randn(n_steps) * volatility + drift
    elif regime == "mean_reverting":
        # Ornstein-Uhlenbeck process
        mean = 0
        theta = 0.15
        sigma = 0.01
        returns = np.zeros(n_steps)
        x = 0
        for i in range(n_steps):
            dx = theta * (mean - x) + sigma * np.random.randn()
            x += dx
            returns[i] = x
    elif regime == "volatile":
        drift = 0
        volatility = 0.015
        returns = np.random.randn(n_steps) * volatility + drift
    else:  # quiet
        drift = 0
        volatility = 0.002
        returns = np.random.randn(n_steps) * volatility + drift

    # Convert returns to price
    prices = 100 * np.exp(np.cumsum(returns))

    return prices


def plot_lob_heatmap(
    ax: plt.Axes,
    bid_prices: np.ndarray,
    bid_volumes: np.ndarray,
    ask_prices: np.ndarray,
    ask_volumes: np.ndarray,
    mid_price: float,
):
    """
    Plot LOB as a horizontal heatmap.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    bid_prices : np.ndarray
        Bid price levels
    bid_volumes : np.ndarray
        Bid volumes
    ask_prices : np.ndarray
        Ask price levels
    ask_volumes : np.ndarray
        Ask volumes
    mid_price : float
        Mid-market price for reference line
    """
    n_levels = len(bid_prices)

    # Create grid for heatmap
    max_volume = max(bid_volumes.max(), ask_volumes.max())

    # Plot bid side (green)
    for i, (price, volume) in enumerate(zip(bid_prices, bid_volumes)):
        color_intensity = volume / max_volume
        ax.barh(
            i,
            volume,
            left=0,
            height=0.8,
            color=(0, 0.5 + 0.5 * (1 - color_intensity), 0),
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        # Price label
        ax.text(
            -max_volume * 0.05,
            i,
            f"${price:.2f}",
            ha="right",
            va="center",
            fontsize=7,
            color="darkgreen",
        )

    # Plot ask side (red)
    for i, (price, volume) in enumerate(zip(ask_prices, ask_volumes)):
        color_intensity = volume / max_volume
        ax.barh(
            n_levels + i,
            volume,
            left=0,
            height=0.8,
            color=(0.5 + 0.5 * (1 - color_intensity), 0, 0),
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        # Price label
        ax.text(
            -max_volume * 0.05,
            n_levels + i,
            f"${price:.2f}",
            ha="right",
            va="center",
            fontsize=7,
            color="darkred",
        )

    # Mid-price line
    ax.axhline(
        n_levels - 0.5, color="black", linewidth=2, linestyle="--", label="Mid-Price"
    )

    # Spread annotation
    spread = ask_prices[0] - bid_prices[0]
    ax.text(
        max_volume * 0.5,
        n_levels - 0.5,
        f"Spread: ${spread:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    # Labels
    ax.set_xlabel("Volume (Contracts)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Price Level", fontsize=11, fontweight="bold")
    ax.set_ylim(-1, 2 * n_levels)
    ax.set_xlim(0, max_volume * 1.1)

    # Y-axis labels
    ax.set_yticks([n_levels / 2 - 0.5, n_levels - 0.5, 1.5 * n_levels - 0.5])
    ax.set_yticklabels(["Bids\n(Support)", "Mid", "Asks\n(Resistance)"])

    ax.set_title(
        "Panel A: Limit Order Book (Intent)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, axis="x", linestyle=":")

    # Legend box
    legend_text = (
        "LOB shows WHERE orders rest\n"
        "• Green: Buy-side liquidity\n"
        "• Red: Sell-side liquidity\n"
        "• Depth = Market resilience"
    )
    ax.text(
        0.98,
        0.02,
        legend_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )


def plot_gaf_texture(ax: plt.Axes, prices: np.ndarray, gaf_method: str = "GASF"):
    """
    Plot GAF texture heatmap.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    prices : np.ndarray
        Price time series
    gaf_method : str
        'GASF' or 'GADF'
    """
    # Generate GAF
    gaf = generate_gaf_from_price(prices, method=gaf_method)

    # Plot heatmap
    if gaf_method == "GASF":
        cmap = "YlGnBu"
        label = "Correlation"
    else:
        cmap = "RdYlBu"
        label = "Flux"

    im = ax.imshow(gaf, cmap=cmap, aspect="auto", interpolation="bilinear")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, fontsize=10, fontweight="bold")

    # Labels
    ax.set_xlabel("Time Index (j)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Time Index (i)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Panel B: GAF Texture ({gaf_method}) (Dynamics)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    # Time series overlay (top)
    ax2 = ax.twiny()
    ax2.plot(
        np.arange(len(prices)),
        prices,
        "k-",
        linewidth=1,
        alpha=0.5,
        label="Price",
    )
    ax2.set_xlabel("Price Series", fontsize=9, style="italic")
    ax2.tick_params(labelsize=7)
    ax2.set_xlim(0, len(prices) - 1)

    # Legend box
    legend_text = (
        "GAF shows HOW price moves\n"
        f"• {gaf_method}: Temporal correlation\n"
        "• Diagonal: Time autocorrelation\n"
        "• Texture = Market regime"
    )
    ax.text(
        0.02,
        0.98,
        legend_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )


def visualize_lob_gaf_comparison(
    prices: np.ndarray,
    regime: str = "normal",
    gaf_method: str = "GASF",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Create complete LOB vs GAF comparison visualization (V2).

    Parameters
    ----------
    prices : np.ndarray
        Price time series
    regime : str
        LOB regime ('normal', 'imbalanced', 'wide_spread')
    gaf_method : str
        GAF method ('GASF' or 'GADF')
    save_path : Path, optional
        Output path for PNG
    dpi : int
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.Figure
        The complete figure
    """
    # Generate LOB snapshot
    lob_sim = LOBSimulator(mid_price=prices[-1])
    bid_prices, bid_volumes, ask_prices, ask_volumes = lob_sim.generate_lob_snapshot(
        n_levels=20, regime=regime
    )

    # Create figure
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    ax_lob = fig.add_subplot(gs[0, 0])
    ax_gaf = fig.add_subplot(gs[0, 1])

    # Panel A: LOB heatmap
    plot_lob_heatmap(
        ax_lob, bid_prices, bid_volumes, ask_prices, ask_volumes, prices[-1]
    )

    # Panel B: GAF texture
    plot_gaf_texture(ax_gaf, prices, gaf_method)

    # Overall title
    fig.suptitle(
        "Visual 2: LOB vs GAF Comparison - Microstructure Fusion View",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Fusion explanation footer
    fusion_text = (
        "Multimodal Fusion Necessity: LOB (static intent) + GAF (dynamic execution) → Complete Market Understanding\n"
        f"LOB Regime: {regime.replace('_', ' ').title()} | "
        f"GAF Method: {gaf_method} | "
        f"Price Window: {len(prices)} steps"
    )
    fig.text(
        0.5,
        0.01,
        fusion_text,
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
        description="Visual 2: LOB vs GAF Comparison Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visual_2_lob_gaf_comparison.py --show
  python visual_2_lob_gaf_comparison.py --save-all --output-dir ../outputs
  python visual_2_lob_gaf_comparison.py --regime imbalanced --price-regime trending --show
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
        default="normal",
        choices=["normal", "imbalanced", "wide_spread"],
        help="LOB regime for simulation",
    )
    parser.add_argument(
        "--price-regime",
        type=str,
        default="trending",
        choices=["trending", "mean_reverting", "volatile", "quiet"],
        help="Price series regime",
    )
    parser.add_argument(
        "--gaf-method",
        type=str,
        default="GASF",
        choices=["GASF", "GADF"],
        help="GAF method (Summation or Difference)",
    )
    parser.add_argument(
        "--window", type=int, default=100, help="Price series window size"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for saved figures"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Visual 2: LOB vs GAF Comparison - Microstructure Fusion View")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  LOB regime: {args.regime}")
    print(f"  Price regime: {args.price_regime}")
    print(f"  GAF method: {args.gaf_method}")
    print(f"  Window size: {args.window}")
    print(f"  Random seed: {args.seed}")

    # Generate synthetic price series
    print(f"\n[1/2] Generating synthetic price series...")
    prices = generate_synthetic_price_series(
        n_steps=args.window, regime=args.price_regime, seed=args.seed
    )
    print(f"✓ Generated {len(prices)} price points")
    print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"  Final price: ${prices[-1]:.2f}")

    # Create visualization
    print(f"\n[2/2] Creating LOB vs GAF comparison...")

    if args.save_all:
        # Generate all combinations
        combinations = [
            ("normal", "GASF"),
            ("imbalanced", "GASF"),
            ("wide_spread", "GASF"),
            (args.regime, "GADF"),
        ]

        for lob_regime, gaf_method in combinations:
            save_path = (
                args.output_dir
                / f"visual_2_lob_{lob_regime}_gaf_{gaf_method.lower()}_{args.price_regime}.png"
            )
            fig = visualize_lob_gaf_comparison(
                prices=prices,
                regime=lob_regime,
                gaf_method=gaf_method,
                save_path=save_path,
                dpi=args.dpi,
            )
            plt.close(fig)
    else:
        save_path = None
        if not args.show:
            save_path = (
                args.output_dir
                / f"visual_2_lob_{args.regime}_gaf_{args.gaf_method.lower()}_{args.price_regime}.png"
            )

        fig = visualize_lob_gaf_comparison(
            prices=prices,
            regime=args.regime,
            gaf_method=args.gaf_method,
            save_path=save_path,
            dpi=args.dpi,
        )

    print(f"\n{'=' * 70}")
    print("✓ Visualization complete!")
    print(f"{'=' * 70}")

    if args.save_all:
        print(f"\nOutput saved to: {args.output_dir}/")
    elif save_path:
        print(f"\nOutput: {save_path}")

    print("\nInterpretation:")
    print("  • LOB (Left): Shows market microstructure - WHERE orders rest")
    print("  • GAF (Right): Shows price dynamics - HOW price moves")
    print("  • Fusion combines both: Intent (LOB) + Execution (GAF)")
    print("  • Green depth = Buy-side support")
    print("  • Red depth = Sell-side resistance")
    print("  • GAF texture = Market regime signature")

    if args.show:
        plt.show()
    elif not args.save_all:
        plt.close(fig)


if __name__ == "__main__":
    main()
