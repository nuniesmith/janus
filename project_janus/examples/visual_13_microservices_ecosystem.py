#!/usr/bin/env python3
"""
Visual 13: Microservices Ecosystem

Visualizes the complete Project JANUS deployment architecture using C4 container
diagram principles, showing:
- Python Gateway (model training & ONNX export)
- Forward Pod (Rust, real-time inference)
- Backward Pod (Rust, memory consolidation)
- Qdrant vector database (persistent schemas)
- External data sources and outputs
- Feedback loops and data flows

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
    "python": "#3776AB",  # Python blue
    "rust": "#CE412B",  # Rust orange
    "database": "#4169E1",  # Royal blue (databases)
    "external": "#2E8B57",  # Sea green (external systems)
    "data_flow": "#696969",  # Dim gray (data connections)
    "feedback": "#9370DB",  # Medium purple (feedback loops)
    "critical": "#DC143C",  # Crimson (critical path)
    "background": "#F5F5F5",  # White smoke
    "grid": "#E0E0E0",  # Light gray
    "text": "#2F2F2F",  # Dark gray
    "container_bg": "#FFFFFF",  # White (container backgrounds)
}


class MicroservicesEcosystem:
    """
    C4 Container diagram for Project JANUS microservices architecture.

    The visualization demonstrates:
    1. Python training gateway (PyTorch, HuggingFace)
    2. Rust forward pod (real-time inference, Tokio)
    3. Rust backward pod (memory consolidation, Rayon)
    4. Qdrant vector database (persistent storage)
    5. Data flows and feedback loops
    6. External interfaces (market data, order execution)
    """

    def __init__(self):
        """Initialize the microservices ecosystem visualization."""
        # Define containers
        self.containers = {
            "python_gateway": {
                "name": "Python Gateway",
                "tech": "PyTorch, HuggingFace, ONNX",
                "desc": "Model training & export",
                "color": COLORS["python"],
                "pos": (2, 8),
                "size": (2.5, 1.5),
            },
            "forward_pod": {
                "name": "Forward Pod",
                "tech": "Rust, Tokio, tract-onnx",
                "desc": "Real-time inference & trading",
                "color": COLORS["rust"],
                "pos": (6, 8),
                "size": (2.5, 1.5),
            },
            "backward_pod": {
                "name": "Backward Pod",
                "tech": "Rust, Rayon, ndarray",
                "desc": "Memory consolidation",
                "color": COLORS["rust"],
                "pos": (6, 4.5),
                "size": (2.5, 1.5),
            },
            "qdrant": {
                "name": "Qdrant",
                "tech": "Vector Database",
                "desc": "Persistent schemas",
                "color": COLORS["database"],
                "pos": (10.5, 6.5),
                "size": (2.0, 1.2),
            },
        }

        # Define external systems
        self.external_systems = {
            "market_data": {
                "name": "Market Data",
                "tech": "WebSocket/TCP",
                "pos": (6, 11),
                "size": (2.0, 0.8),
            },
            "order_execution": {
                "name": "Order Execution",
                "tech": "FIX Protocol",
                "pos": (10, 10),
                "size": (2.0, 0.8),
            },
            "monitoring": {
                "name": "Monitoring",
                "tech": "Prometheus/Grafana",
                "pos": (1, 4),
                "size": (2.0, 0.8),
            },
        }

        # Define data flows
        self.data_flows = [
            # Training → Forward (ONNX model)
            {
                "from": "python_gateway",
                "to": "forward_pod",
                "label": "model.onnx",
                "style": "solid",
                "color": COLORS["python"],
            },
            # Market Data → Forward
            {
                "from": "market_data",
                "to": "forward_pod",
                "label": "Price/LOB stream",
                "style": "solid",
                "color": COLORS["critical"],
            },
            # Forward → Order Execution
            {
                "from": "forward_pod",
                "to": "order_execution",
                "label": "Trading orders",
                "style": "solid",
                "color": COLORS["critical"],
            },
            # Forward → Backward (Experiences)
            {
                "from": "forward_pod",
                "to": "backward_pod",
                "label": "Experiences\n(s,a,r,s')",
                "style": "solid",
                "color": COLORS["data_flow"],
            },
            # Backward → Qdrant (Write schemas)
            {
                "from": "backward_pod",
                "to": "qdrant",
                "label": "Write schemas",
                "style": "solid",
                "color": COLORS["data_flow"],
            },
            # Qdrant → Backward (Read schemas)
            {
                "from": "qdrant",
                "to": "backward_pod",
                "label": "Read schemas",
                "style": "dashed",
                "color": COLORS["data_flow"],
            },
            # Backward → Forward (Feedback loop - updated schemas)
            {
                "from": "backward_pod",
                "to": "forward_pod",
                "label": "Updated\nschemas\n(hot reload)",
                "style": "dashed",
                "color": COLORS["feedback"],
            },
            # Monitoring connections
            {
                "from": "forward_pod",
                "to": "monitoring",
                "label": "Metrics",
                "style": "dotted",
                "color": COLORS["data_flow"],
            },
            {
                "from": "backward_pod",
                "to": "monitoring",
                "label": "Metrics",
                "style": "dotted",
                "color": COLORS["data_flow"],
            },
        ]

    def create_visualization(self) -> plt.Figure:
        """
        Create the complete microservices ecosystem visualization.

        Returns:
            Matplotlib figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 13)
        ax.axis("off")
        ax.set_facecolor(COLORS["background"])

        # Title and subtitle
        fig.suptitle(
            "Visual 13: Project JANUS Microservices Ecosystem\n"
            "C4 Container Diagram — Production Deployment Architecture",
            fontsize=16,
            fontweight="bold",
            y=0.97,
        )

        # Draw external systems first (background layer)
        for ext_name, ext_info in self.external_systems.items():
            self._draw_external_system(ax, ext_name, ext_info)

        # Draw containers (main layer)
        for container_name, container_info in self.containers.items():
            self._draw_container(ax, container_name, container_info)

        # Draw data flows (foreground layer)
        for flow in self.data_flows:
            self._draw_data_flow(ax, flow)

        # Add legend
        self._add_legend(ax)

        # Add system boundary
        self._add_system_boundary(ax)

        # Add deployment notes
        self._add_deployment_notes(ax)

        return fig

    def _draw_container(self, ax: plt.Axes, name: str, info: Dict) -> None:
        """Draw a container (microservice)."""
        x, y = info["pos"]
        w, h = info["size"]

        # Container box with shadow
        shadow = FancyBboxPatch(
            (x - w / 2 + 0.05, y - h / 2 - 0.05),
            w,
            h,
            boxstyle="round,pad=0.15",
            facecolor="gray",
            alpha=0.3,
        )
        ax.add_patch(shadow)

        # Main container box
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.15",
            facecolor=COLORS["container_bg"],
            edgecolor=info["color"],
            linewidth=3,
            alpha=1.0,
        )
        ax.add_patch(box)

        # Technology badge
        tech_badge = FancyBboxPatch(
            (x - w / 2 + 0.1, y + h / 2 - 0.45),
            w - 0.2,
            0.35,
            boxstyle="round,pad=0.05",
            facecolor=info["color"],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.add_patch(tech_badge)

        # Container name
        ax.text(
            x,
            y + h / 2 - 0.27,
            info["name"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )

        # Technology stack
        ax.text(
            x,
            y,
            info["tech"],
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
            color=info["color"],
        )

        # Description
        ax.text(
            x,
            y - h / 2 + 0.25,
            info["desc"],
            ha="center",
            va="center",
            fontsize=8,
            color=COLORS["text"],
        )

    def _draw_external_system(self, ax: plt.Axes, name: str, info: Dict) -> None:
        """Draw an external system (outside JANUS boundary)."""
        x, y = info["pos"]
        w, h = info["size"]

        # External system box (dashed border)
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=COLORS["background"],
            edgecolor=COLORS["external"],
            linewidth=2.5,
            linestyle="--",
            alpha=0.9,
        )
        ax.add_patch(box)

        # Name
        ax.text(
            x,
            y + 0.15,
            info["name"],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=COLORS["external"],
        )

        # Technology
        ax.text(
            x,
            y - 0.15,
            info["tech"],
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
            color=COLORS["text"],
        )

    def _draw_data_flow(self, ax: plt.Axes, flow: Dict) -> None:
        """Draw a data flow arrow between components."""
        # Get source and destination positions
        if flow["from"] in self.containers:
            from_info = self.containers[flow["from"]]
        else:
            from_info = self.external_systems[flow["from"]]

        if flow["to"] in self.containers:
            to_info = self.containers[flow["to"]]
        else:
            to_info = self.external_systems[flow["to"]]

        from_x, from_y = from_info["pos"]
        to_x, to_y = to_info["pos"]

        # Adjust start/end points to container edges
        from_w, from_h = from_info["size"]
        to_w, to_h = to_info["size"]

        # Calculate connection points
        dx = to_x - from_x
        dy = to_y - from_y
        angle = np.arctan2(dy, dx)

        # Start point (exit from source)
        if abs(dx) > abs(dy):
            # Horizontal connection
            if dx > 0:
                start_x = from_x + from_w / 2
                start_y = from_y
            else:
                start_x = from_x - from_w / 2
                start_y = from_y
        else:
            # Vertical connection
            if dy > 0:
                start_x = from_x
                start_y = from_y + from_h / 2
            else:
                start_x = from_x
                start_y = from_y - from_h / 2

        # End point (enter to destination)
        if abs(dx) > abs(dy):
            # Horizontal connection
            if dx > 0:
                end_x = to_x - to_w / 2
                end_y = to_y
            else:
                end_x = to_x + to_w / 2
                end_y = to_y
        else:
            # Vertical connection
            if dy > 0:
                end_x = to_x
                end_y = to_y - to_h / 2
            else:
                end_x = to_x
                end_y = to_y + to_h / 2

        # Draw arrow
        arrow_style = "->" if flow["style"] != "dotted" else "-"
        arrow = FancyArrowPatch(
            (start_x, start_y),
            (end_x, end_y),
            arrowstyle=arrow_style,
            mutation_scale=20,
            linewidth=2.5 if flow["color"] == COLORS["critical"] else 2,
            color=flow["color"],
            linestyle=flow["style"] if flow["style"] != "solid" else "-",
            alpha=0.8,
        )
        ax.add_patch(arrow)

        # Label
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2

        # Offset label slightly from arrow
        offset = 0.3
        if abs(dx) > abs(dy):
            label_y = mid_y + offset
            label_x = mid_x
        else:
            label_x = mid_x + offset
            label_y = mid_y

        ax.text(
            label_x,
            label_y,
            flow["label"],
            ha="center",
            va="center",
            fontsize=7,
            color=flow["color"],
            fontweight="bold" if flow["color"] == COLORS["critical"] else "normal",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=flow["color"],
                linewidth=1,
                alpha=0.95,
            ),
        )

    def _add_system_boundary(self, ax: plt.Axes) -> None:
        """Draw system boundary around JANUS components."""
        # Large boundary box
        boundary = Rectangle(
            (1.5, 3),
            10.5,
            7.5,
            facecolor="none",
            edgecolor=COLORS["rust"],
            linewidth=3,
            linestyle="-",
            alpha=0.6,
        )
        ax.add_patch(boundary)

        # System label
        ax.text(
            1.7,
            10.3,
            "Project JANUS System Boundary",
            fontsize=11,
            fontweight="bold",
            color=COLORS["rust"],
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor=COLORS["rust"],
                linewidth=2,
                alpha=0.95,
            ),
        )

    def _add_legend(self, ax: plt.Axes) -> None:
        """Add legend explaining symbols."""
        legend_elements = [
            mpatches.Patch(
                facecolor=COLORS["python"],
                edgecolor="white",
                label="Python Service",
            ),
            mpatches.Patch(
                facecolor=COLORS["rust"], edgecolor="white", label="Rust Service"
            ),
            mpatches.Patch(
                facecolor=COLORS["database"],
                edgecolor="white",
                label="Database",
            ),
            mpatches.Patch(
                facecolor=COLORS["background"],
                edgecolor=COLORS["external"],
                linestyle="--",
                label="External System",
            ),
            mpatches.Patch(
                facecolor=COLORS["critical"],
                edgecolor="white",
                label="Critical Path",
            ),
            mpatches.Patch(
                facecolor=COLORS["feedback"],
                edgecolor="white",
                label="Feedback Loop",
            ),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=9,
            framealpha=0.95,
            title="Legend",
            title_fontsize=10,
        )

    def _add_deployment_notes(self, ax: plt.Axes) -> None:
        """Add deployment and infrastructure notes."""
        notes_text = (
            "Deployment Notes:\n"
            "• Forward Pod: Low-latency Tier 1 (<100ms)\n"
            "• Backward Pod: Batch Tier 3 (hourly)\n"
            "• Qdrant: Persistent, replicated storage\n"
            "• All services: Docker + Kubernetes\n"
            "• Monitoring: Real-time telemetry\n"
            "• Hot reload: Zero-downtime updates"
        )

        ax.text(
            1,
            1.8,
            notes_text,
            fontsize=8,
            color=COLORS["text"],
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=COLORS["database"],
                linewidth=2,
                alpha=0.95,
            ),
        )

        # Performance SLA box
        sla_text = (
            "Performance SLA:\n"
            "Forward:  p99 < 50ms\n"
            "Backward: p99 < 30s\n"
            "Uptime:   99.9%"
        )

        ax.text(
            13,
            2.5,
            sla_text,
            ha="right",
            fontsize=8,
            family="monospace",
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=COLORS["critical"],
                edgecolor="white",
                linewidth=2,
                alpha=0.2,
            ),
        )

        # Technology stack summary
        stack_text = (
            "Tech Stack:\n"
            "Training:   PyTorch, ONNX\n"
            "Runtime:    Rust, Tokio, Rayon\n"
            "Storage:    Qdrant (vectors)\n"
            "Infra:      K8s, Docker, Prometheus"
        )

        ax.text(
            13,
            1,
            stack_text,
            ha="right",
            fontsize=8,
            family="monospace",
            color=COLORS["text"],
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
        description="Generate Visual 13: Microservices Ecosystem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display interactive visualization
  python visual_13_microservices_ecosystem.py --show

  # Save high-resolution output
  python visual_13_microservices_ecosystem.py --save-all --output-dir ../outputs
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
        "--dpi", type=int, default=300, help="DPI for saved figures (default: 300)"
    )

    args = parser.parse_args()

    # Create output directory if saving
    if args.save_all:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path.absolute()}")

    # Create microservices ecosystem visualization
    print("Generating Microservices Ecosystem visualization...")
    ecosystem = MicroservicesEcosystem()

    # Generate visualization
    fig = ecosystem.create_visualization()

    # Save if requested
    if args.save_all:
        output_file = output_path / "visual_13_microservices_ecosystem.png"
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
