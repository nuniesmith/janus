#!/usr/bin/env python3
"""
Visual 12: Runtime Topology

Visualizes the "two minds" of Rust concurrency in Project JANUS:
- Tokio async runtime for I/O-bound Forward Service (event loop)
- Rayon parallel runtime for CPU-bound Backward Service (work stealing)

This visualization shows:
- Tokio: Single-threaded event loop with many hollow tasks (awaiting I/O)
- Rayon: Multi-threaded work stealing with full CPU utilization
- Comparison of execution patterns and resource usage
- Justification for the "Async Sandwich" architecture

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
    "tokio": "#2E8B57",  # Sea green (async/await)
    "rayon": "#4169E1",  # Royal blue (parallel)
    "io_wait": "#FFD700",  # Gold (I/O waiting)
    "cpu_work": "#DC143C",  # Crimson (CPU working)
    "idle": "#E0E0E0",  # Light gray (idle)
    "thread": "#9370DB",  # Medium purple (threads)
    "edge": "#696969",  # Dim gray (connections)
    "background": "#F5F5F5",  # White smoke
    "grid": "#E0E0E0",  # Light gray
    "text": "#2F2F2F",  # Dark gray
}


class RuntimeTopology:
    """
    Visualization of Tokio vs Rayon concurrency models.

    The visualization demonstrates:
    1. Tokio event loop with lightweight async tasks
    2. Rayon work-stealing thread pool
    3. Resource utilization patterns
    4. Why each model suits its workload
    """

    def __init__(
        self,
        num_tokio_tasks: int = 20,
        num_rayon_threads: int = 8,
        timeline_length: int = 100,
        random_seed: int = 42,
    ):
        """
        Initialize runtime topology visualization.

        Args:
            num_tokio_tasks: Number of async tasks in Tokio
            num_rayon_threads: Number of threads in Rayon pool
            timeline_length: Length of execution timeline
            random_seed: Random seed for reproducibility
        """
        self.num_tokio_tasks = num_tokio_tasks
        self.num_rayon_threads = num_rayon_threads
        self.timeline_length = timeline_length

        np.random.seed(random_seed)

        # Generate synthetic execution traces
        self._generate_execution_traces()

    def _generate_execution_traces(self) -> None:
        """Generate synthetic execution traces for both runtimes."""
        # Tokio: Mostly I/O wait, brief CPU bursts
        self.tokio_trace = np.zeros((self.timeline_length, self.num_tokio_tasks))
        for i in range(self.num_tokio_tasks):
            # Random I/O events
            io_events = np.random.choice(
                self.timeline_length,
                size=int(self.timeline_length * 0.1),
                replace=False,
            )
            self.tokio_trace[io_events, i] = 1  # CPU work
            # Rest is waiting (0)

        # Rayon: Full CPU utilization across all threads
        self.rayon_trace = np.zeros((self.timeline_length, self.num_rayon_threads))
        for i in range(self.num_rayon_threads):
            # High CPU utilization with occasional gaps
            work_periods = np.random.rand(self.timeline_length) > 0.15
            self.rayon_trace[:, i] = work_periods.astype(float)

        # Compute utilization
        self.tokio_utilization = self.tokio_trace.mean(axis=1)
        self.rayon_utilization = self.rayon_trace.mean(axis=1)

    def create_visualization(self) -> plt.Figure:
        """
        Create the complete runtime topology visualization.

        Returns:
            Matplotlib figure object
        """
        # Create figure with 4 panels (2x2 grid)
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3, left=0.08, right=0.95
        )

        # Panel A: Tokio architecture diagram
        ax_tokio_arch = fig.add_subplot(gs[0, 0])
        self._plot_tokio_architecture(ax_tokio_arch)

        # Panel B: Rayon architecture diagram
        ax_rayon_arch = fig.add_subplot(gs[0, 1])
        self._plot_rayon_architecture(ax_rayon_arch)

        # Panel C: Tokio execution trace
        ax_tokio_trace = fig.add_subplot(gs[1, 0])
        self._plot_tokio_trace(ax_tokio_trace)

        # Panel D: Rayon execution trace
        ax_rayon_trace = fig.add_subplot(gs[1, 1])
        self._plot_rayon_trace(ax_rayon_trace)

        # Overall title
        fig.suptitle(
            "Visual 12: Runtime Topology — Tokio vs Rayon\n"
            "Forward Service (Async I/O) vs Backward Service (Parallel Compute)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        return fig

    def _plot_tokio_architecture(self, ax: plt.Axes) -> None:
        """Plot Tokio async runtime architecture."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_facecolor(COLORS["background"])

        # Title
        ax.text(
            5,
            9.7,
            "A. Tokio Runtime (Forward Service)",
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=COLORS["tokio"],
        )
        ax.text(
            5,
            9.2,
            'The "Wait" Engine — Event Loop for I/O-bound tasks',
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            color=COLORS["text"],
        )

        # ===== EVENT LOOP (REACTOR) =====
        reactor_x, reactor_y = 5.0, 6.5
        reactor_circle = Circle(
            (reactor_x, reactor_y),
            1.2,
            facecolor=COLORS["tokio"],
            edgecolor="white",
            linewidth=3,
            alpha=0.8,
        )
        ax.add_patch(reactor_circle)

        # Circular arrow inside (event loop)
        from matplotlib.patches import Arc

        arc = Arc(
            (reactor_x, reactor_y),
            1.8,
            1.8,
            angle=0,
            theta1=30,
            theta2=330,
            color="white",
            linewidth=3,
            linestyle="-",
        )
        ax.add_patch(arc)

        # Arrow head
        arrow_angle = np.radians(30)
        arrow_x = reactor_x + 0.9 * np.cos(arrow_angle)
        arrow_y = reactor_y + 0.9 * np.sin(arrow_angle)
        ax.annotate(
            "",
            xy=(arrow_x, arrow_y),
            xytext=(arrow_x - 0.2, arrow_y + 0.1),
            arrowprops=dict(arrowstyle="->", color="white", lw=3),
        )

        ax.text(
            reactor_x,
            reactor_y,
            "Event\nLoop",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )

        ax.text(
            reactor_x,
            reactor_y - 1.6,
            "Single-threaded Reactor\n(epoll/kqueue)",
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color=COLORS["text"],
        )

        # ===== ASYNC TASKS (LIGHTWEIGHT) =====
        task_positions = [
            (1.5, 8.5, "WS\nRecv", "hollow"),
            (3.0, 8.0, "Parse\nData", "hollow"),
            (8.5, 8.5, "Send\nOrder", "hollow"),
            (7.0, 8.0, "Log\nEvent", "hollow"),
            (1.8, 5.5, "DB\nQuery", "hollow"),
            (8.2, 5.5, "HTTP\nReq", "hollow"),
        ]

        for x, y, label, style in task_positions:
            if style == "hollow":
                # Hollow rectangle (awaiting I/O)
                rect = Rectangle(
                    (x - 0.35, y - 0.3),
                    0.7,
                    0.6,
                    facecolor=COLORS["background"],
                    edgecolor=COLORS["io_wait"],
                    linewidth=2.5,
                    linestyle="--",
                    alpha=0.9,
                )
            else:
                # Solid rectangle (CPU work)
                rect = Rectangle(
                    (x - 0.35, y - 0.3),
                    0.7,
                    0.6,
                    facecolor=COLORS["cpu_work"],
                    edgecolor="white",
                    linewidth=2,
                    alpha=0.8,
                )
            ax.add_patch(rect)

            ax.text(
                x, y, label, ha="center", va="center", fontsize=7, fontweight="bold"
            )

            # Arrow to/from reactor
            if y > reactor_y:
                arrow = FancyArrowPatch(
                    (x, y - 0.35),
                    (reactor_x, reactor_y + 1.0),
                    arrowstyle="<->",
                    mutation_scale=15,
                    linewidth=1.5,
                    color=COLORS["edge"],
                    alpha=0.4,
                    linestyle="dashed",
                )
            else:
                arrow = FancyArrowPatch(
                    (x, y + 0.35),
                    (reactor_x, reactor_y - 1.0),
                    arrowstyle="<->",
                    mutation_scale=15,
                    linewidth=1.5,
                    color=COLORS["edge"],
                    alpha=0.4,
                    linestyle="dashed",
                )
            ax.add_patch(arrow)

        # ===== KEY METRICS =====
        metrics_box = FancyBboxPatch(
            (0.3, 1.0),
            4.0,
            2.5,
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=COLORS["tokio"],
            linewidth=2,
            alpha=0.95,
        )
        ax.add_patch(metrics_box)

        metrics_text = (
            "Characteristics:\n"
            "• Single thread, many tasks\n"
            "• Tasks are lightweight (~KB)\n"
            "• CPU idle while awaiting I/O\n"
            "• Latency from network, not CPU\n"
            "• Perfect for WebSocket/TCP"
        )
        ax.text(
            2.3,
            2.25,
            metrics_text,
            fontsize=9,
            color=COLORS["text"],
            verticalalignment="center",
        )

        # ===== WORKLOAD EXAMPLE =====
        ax.text(
            5.5, 1.8, "Workload:", fontsize=10, fontweight="bold", color=COLORS["text"]
        )
        ax.text(
            5.5,
            1.3,
            "Market data ingestion\n(99% I/O wait, 1% CPU)",
            fontsize=8,
            style="italic",
            color=COLORS["text"],
        )

    def _plot_rayon_architecture(self, ax: plt.Axes) -> None:
        """Plot Rayon parallel runtime architecture."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_facecolor(COLORS["background"])

        # Title
        ax.text(
            5,
            9.7,
            "B. Rayon Runtime (Backward Service)",
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=COLORS["rayon"],
        )
        ax.text(
            5,
            9.2,
            'The "Think" Engine — Work Stealing for CPU-bound tasks',
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            color=COLORS["text"],
        )

        # ===== THREAD POOL =====
        pool_x, pool_y = 5.0, 6.5
        num_threads_display = 8

        # Draw thread pool as grid
        thread_size = 0.8
        spacing = 1.0
        cols = 4
        rows = 2

        for i in range(num_threads_display):
            row = i // cols
            col = i % cols
            x = pool_x - (cols * spacing) / 2 + col * spacing + spacing / 2
            y = pool_y + 0.5 - row * spacing

            # Solid rectangle (full CPU utilization)
            rect = Rectangle(
                (x - thread_size / 2, y - thread_size / 2),
                thread_size,
                thread_size,
                facecolor=COLORS["cpu_work"],
                edgecolor="white",
                linewidth=2.5,
                alpha=0.85,
            )
            ax.add_patch(rect)

            ax.text(
                x,
                y,
                f"T{i}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

        ax.text(
            pool_x,
            pool_y - 1.5,
            "Thread Pool (Work Stealing)\nN = num_cpus",
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color=COLORS["text"],
        )

        # ===== WORK QUEUE =====
        queue_y = 8.5
        queue_box = FancyBboxPatch(
            (pool_x - 1.5, queue_y - 0.3),
            3.0,
            0.6,
            boxstyle="round,pad=0.08",
            facecolor=COLORS["rayon"],
            edgecolor="white",
            linewidth=2.5,
            alpha=0.8,
        )
        ax.add_patch(queue_box)

        ax.text(
            pool_x,
            queue_y,
            "Work Queue (Tasks)",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

        # Arrow from queue to threads
        arrow = FancyArrowPatch(
            (pool_x, queue_y - 0.35),
            (pool_x, pool_y + 0.9),
            arrowstyle="->",
            mutation_scale=25,
            linewidth=3,
            color=COLORS["edge"],
            alpha=0.6,
        )
        ax.add_patch(arrow)

        # ===== WORK ITEMS (CPU-BOUND) =====
        work_items = [
            (1.5, 4.5, "UMAP\nProject"),
            (3.5, 4.5, "Gradient\nCalc"),
            (6.5, 4.5, "Matrix\nMul"),
            (8.5, 4.5, "Replay\nBatch"),
        ]

        for x, y, label in work_items:
            # Solid rectangles (CPU work)
            rect = Rectangle(
                (x - 0.4, y - 0.3),
                0.8,
                0.6,
                facecolor=COLORS["thread"],
                edgecolor="white",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(rect)

            ax.text(
                x, y, label, ha="center", va="center", fontsize=7, fontweight="bold"
            )

        # ===== KEY METRICS =====
        metrics_box = FancyBboxPatch(
            (0.3, 1.0),
            4.0,
            2.5,
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=COLORS["rayon"],
            linewidth=2,
            alpha=0.95,
        )
        ax.add_patch(metrics_box)

        metrics_text = (
            "Characteristics:\n"
            "• Multi-threaded, parallel\n"
            "• All cores 100% utilized\n"
            "• CPU-bound computation\n"
            "• Work stealing balances load\n"
            "• Perfect for batch compute"
        )
        ax.text(
            2.3,
            2.25,
            metrics_text,
            fontsize=9,
            color=COLORS["text"],
            verticalalignment="center",
        )

        # ===== WORKLOAD EXAMPLE =====
        ax.text(
            5.5, 1.8, "Workload:", fontsize=10, fontweight="bold", color=COLORS["text"]
        )
        ax.text(
            5.5,
            1.3,
            "Memory consolidation\n(1% I/O wait, 99% CPU)",
            fontsize=8,
            style="italic",
            color=COLORS["text"],
        )

    def _plot_tokio_trace(self, ax: plt.Axes) -> None:
        """Plot Tokio execution trace."""
        ax.set_facecolor(COLORS["background"])
        ax.set_title(
            "C. Tokio Execution Trace (Sparse CPU Usage)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        # Heatmap of task execution
        im = ax.imshow(
            self.tokio_trace.T,
            aspect="auto",
            cmap="YlGn",
            interpolation="nearest",
            extent=[0, self.timeline_length, 0, self.num_tokio_tasks],
        )

        ax.set_xlabel("Time (ms)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Async Task ID", fontsize=11, fontweight="bold")

        # Color bar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("CPU Activity", fontsize=10, fontweight="bold")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Awaiting", "Working"])

        # Utilization overlay
        ax2 = ax.twinx()
        ax2.plot(
            np.arange(self.timeline_length),
            self.tokio_utilization * 100,
            color=COLORS["tokio"],
            linewidth=2,
            alpha=0.7,
            label="CPU Utilization",
        )
        ax2.set_ylabel(
            "CPU Usage (%)", fontsize=11, fontweight="bold", color=COLORS["tokio"]
        )
        ax2.tick_params(axis="y", labelcolor=COLORS["tokio"])
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper right", fontsize=9)

        # Average utilization annotation
        avg_util = self.tokio_utilization.mean() * 100
        ax.text(
            0.02,
            0.98,
            f"Avg CPU: {avg_util:.1f}%\n(Mostly idle)",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color=COLORS["tokio"],
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95),
        )

    def _plot_rayon_trace(self, ax: plt.Axes) -> None:
        """Plot Rayon execution trace."""
        ax.set_facecolor(COLORS["background"])
        ax.set_title(
            "D. Rayon Execution Trace (Full CPU Saturation)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        # Heatmap of thread execution
        im = ax.imshow(
            self.rayon_trace.T,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
            extent=[0, self.timeline_length, 0, self.num_rayon_threads],
        )

        ax.set_xlabel("Time (ms)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Thread ID", fontsize=11, fontweight="bold")

        # Color bar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("CPU Activity", fontsize=10, fontweight="bold")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Idle", "Working"])

        # Utilization overlay
        ax2 = ax.twinx()
        ax2.plot(
            np.arange(self.timeline_length),
            self.rayon_utilization * 100,
            color=COLORS["rayon"],
            linewidth=2,
            alpha=0.7,
            label="CPU Utilization",
        )
        ax2.set_ylabel(
            "CPU Usage (%)", fontsize=11, fontweight="bold", color=COLORS["rayon"]
        )
        ax2.tick_params(axis="y", labelcolor=COLORS["rayon"])
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper right", fontsize=9)

        # Average utilization annotation
        avg_util = self.rayon_utilization.mean() * 100
        ax.text(
            0.02,
            0.98,
            f"Avg CPU: {avg_util:.1f}%\n(Near saturation)",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color=COLORS["rayon"],
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95),
        )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate Visual 12: Runtime Topology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display interactive visualization
  python visual_12_runtime_topology.py --show

  # Save high-resolution output
  python visual_12_runtime_topology.py --save-all --output-dir ../outputs

  # Custom parameters
  python visual_12_runtime_topology.py --show --num-threads 16
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
        "--num-threads",
        type=int,
        default=8,
        help="Number of Rayon threads (default: 8)",
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

    # Create runtime topology visualization
    print("Generating Runtime Topology visualization...")
    runtime = RuntimeTopology(
        num_tokio_tasks=20,
        num_rayon_threads=args.num_threads,
        timeline_length=100,
        random_seed=args.seed,
    )

    # Generate visualization
    fig = runtime.create_visualization()

    # Save if requested
    if args.save_all:
        output_file = output_path / "visual_12_runtime_topology.png"
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
