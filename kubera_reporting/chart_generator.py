"""Chart generation utilities using matplotlib."""

import io

import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")


def generate_allocation_chart(allocation: dict[str, float]) -> bytes:
    """Generate pie chart image for asset allocation.

    Args:
        allocation: Dictionary with allocation percentages

    Returns:
        PNG image as bytes
    """
    # Define colors for common categories (both generic and custom names)
    colors = {
        # Generic categories
        "Stocks": "#4CAF50",  # Green
        "Bonds": "#2196F3",  # Blue
        "Crypto": "#FF9800",  # Orange
        "Real Estate": "#9C27B0",  # Purple
        "Cash": "#00BCD4",  # Cyan
        "Other": "#795548",  # Brown
        # Common custom categories (case-insensitive matching)
        "stock positions": "#4CAF50",  # Green
        "credit investments": "#2196F3",  # Blue
        "debt investments": "#2196F3",  # Blue
        "fixed income": "#2196F3",  # Blue
        "cash": "#00BCD4",  # Cyan
        "cash equivalents": "#00BCD4",  # Cyan
        "alternatives": "#FF9800",  # Orange
        "private equity": "#9C27B0",  # Purple
        "real estate": "#E91E63",  # Pink
        "commodities": "#FFC107",  # Amber
    }

    labels = list(allocation.keys())
    values = list(allocation.values())
    
    # Match colors case-insensitively, use a vibrant color palette if no match
    default_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4", "#E91E63", "#FFC107", "#795548"]
    chart_colors = []
    for i, label in enumerate(labels):
        color = colors.get(label) or colors.get(label.lower())
        if not color:
            color = default_colors[i % len(default_colors)]
        chart_colors.append(color)

    # Create figure with better layout
    fig, ax = plt.subplots(figsize=(10, 7))

    # Function to only show percentages for slices >= 5%
    def autopct_format(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= 5 else ""

    # Create pie chart with conditional percentages
    pie_result = ax.pie(
        values,
        labels=None,  # No labels on pie - use legend instead
        colors=chart_colors,
        autopct=autopct_format,
        startangle=90,
        textprops={"fontsize": 14, "weight": "bold"},
        pctdistance=0.75,  # Move percentages closer to center
    )
    wedges, texts, autotexts = pie_result  # type: ignore[misc]

    # Make percentage text white for better visibility
    for autotext in autotexts:
        autotext.set_color("white")

    # Add legend below the chart with percentages
    legend_labels = [f"{label} ({value:.1f}%)" for label, value in zip(labels, values)]
    ax.legend(
        wedges,
        legend_labels,
        loc="center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fontsize=12,
        frameon=False,
    )

    ax.axis("equal")

    # Save to bytes buffer with extra padding for legend
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    buf.seek(0)

    return buf.read()
