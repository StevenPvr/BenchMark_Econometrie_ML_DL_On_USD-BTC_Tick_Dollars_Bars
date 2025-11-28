"""Visualization utilities for feature analysis.

This module provides plotting functions using both:
- Plotly: Interactive HTML visualizations (zoomable, hoverable)
- Matplotlib: Static PNG images

Each plot function saves both formats when both output paths are provided.

Plot types:
- Heatmaps: Correlation matrices
- Bar charts: VIF scores, feature importance
- Dendrograms: Hierarchical clustering
- Scatter plots: Feature embeddings (t-SNE, UMAP)
- Line plots: Rolling correlations, ACF/PACF
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib  # type: ignore[import-untyped]
from matplotlib.figure import Figure  # type: ignore[import-untyped]
matplotlib.use("Agg")  # Non-interactive backend

from scipy.cluster import hierarchy  # type: ignore[import-untyped]

from src.analyse_features.config import (
    COLORSCALE_CORRELATION,
    COLORSCALE_SEQUENTIAL,
    FIGSIZE_ACF,
    FIGSIZE_BAR,
    FIGSIZE_DENDROGRAM,
    FIGSIZE_HEATMAP,
    FIGSIZE_SCATTER,
    PNG_DPI,
    PLOTS_HTML_DIR,
    PLOTS_PNG_DIR,
    TOP_N_FEATURES,
    ensure_directories,
)
from src.config_logging import get_logger

logger = get_logger(__name__)


def _save_plotly(fig: Any, filename: str) -> Path:
    """Save Plotly figure to HTML."""
    ensure_directories()
    path = PLOTS_HTML_DIR / f"{filename}.html"
    fig.write_html(str(path))
    logger.info("Saved interactive plot: %s", path)
    return path


def _save_matplotlib(fig: Figure, filename: str) -> Path:
    """Save Matplotlib figure to PNG."""
    ensure_directories()
    path = PLOTS_PNG_DIR / f"{filename}.png"
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved static plot: %s", path)
    return path


# ============================================================================
# CORRELATION HEATMAPS
# ============================================================================

def plot_correlation_heatmap_plotly(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    filename: str = "correlation_heatmap",
) -> Path:
    """Create interactive correlation heatmap with Plotly.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        title: Plot title.
        filename: Output filename (without extension).

    Returns:
        Path to saved HTML file.
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=COLORSCALE_CORRELATION,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
    ))

    fig.update_layout(
        title=title,
        width=1200,
        height=1000,
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
    )

    return _save_plotly(fig, filename)


def plot_correlation_heatmap_matplotlib(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    filename: str = "correlation_heatmap",
) -> Path:
    """Create static correlation heatmap with Matplotlib.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        title: Plot title.
        filename: Output filename.

    Returns:
        Path to saved PNG file.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)

    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))

    # Only show labels if not too many
    if len(corr_matrix.columns) <= 50:
        ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(corr_matrix.index, fontsize=6)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.set_title(title)

    return _save_matplotlib(fig, filename)


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    filename: str = "correlation_heatmap",
) -> tuple[Path, Path]:
    """Create both interactive and static correlation heatmaps.

    Args:
        corr_matrix: Correlation matrix.
        title: Plot title.
        filename: Base filename.

    Returns:
        Tuple of (html_path, png_path).
    """
    html_path = plot_correlation_heatmap_plotly(corr_matrix, title, filename)
    png_path = plot_correlation_heatmap_matplotlib(corr_matrix, title, filename)
    return html_path, png_path


# ============================================================================
# BAR CHARTS
# ============================================================================

def plot_bar_chart_plotly(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    filename: str,
    top_n: int = TOP_N_FEATURES,
    color_col: str | None = None,
) -> Path:
    """Create interactive bar chart with Plotly.

    Args:
        df: DataFrame with data.
        x_col: Column for x-axis (categories).
        y_col: Column for y-axis (values).
        title: Plot title.
        filename: Output filename.
        top_n: Number of top items to show.
        color_col: Column for color scale (optional).

    Returns:
        Path to saved HTML.
    """
    import plotly.express as px  # type: ignore[import-untyped]

    # Take top N
    plot_df = df.head(top_n).copy()

    if color_col and color_col in plot_df.columns:
        fig = px.bar(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            color_continuous_scale=COLORSCALE_SEQUENTIAL,
        )
    else:
        fig = px.bar(plot_df, x=x_col, y=y_col, title=title)

    fig.update_layout(
        xaxis_tickangle=45,
        height=600,
        width=1000,
    )

    return _save_plotly(fig, filename)


def plot_bar_chart_matplotlib(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    filename: str,
    top_n: int = TOP_N_FEATURES,
) -> Path:
    """Create static bar chart with Matplotlib.

    Args:
        df: DataFrame with data.
        x_col: Column for x-axis.
        y_col: Column for y-axis.
        title: Plot title.
        filename: Output filename.
        top_n: Number of items.

    Returns:
        Path to saved PNG.
    """
    plot_df = df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)

    cmap = plt.cm.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(plot_df)))
    y_values = np.asarray(plot_df[y_col].values, dtype=np.float64)
    ax.barh(range(len(plot_df)), y_values, color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df[x_col].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(y_col)
    ax.set_title(title)

    return _save_matplotlib(fig, filename)


def plot_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    filename: str,
    top_n: int = TOP_N_FEATURES,
) -> tuple[Path, Path]:
    """Create both interactive and static bar charts."""
    html_path = plot_bar_chart_plotly(df, x_col, y_col, title, filename, top_n)
    png_path = plot_bar_chart_matplotlib(df, x_col, y_col, title, filename, top_n)
    return html_path, png_path


# ============================================================================
# VIF PLOTS
# ============================================================================

def plot_vif_scores(
    vif_df: pd.DataFrame,
    filename: str = "vif_scores",
    top_n: int = TOP_N_FEATURES,
) -> tuple[Path, Path]:
    """Plot VIF scores for features.

    Args:
        vif_df: DataFrame with 'feature' and 'vif' columns.
        filename: Base filename.
        top_n: Number of features to show.

    Returns:
        Tuple of paths.
    """
    return plot_bar_chart(
        vif_df,
        x_col="feature",
        y_col="vif",
        title=f"Top {top_n} Features by VIF",
        filename=filename,
        top_n=top_n,
    )


# ============================================================================
# DENDROGRAM
# ============================================================================

def plot_dendrogram_plotly(
    distance_matrix: np.ndarray,
    labels: list[str],
    title: str = "Feature Clustering Dendrogram",
    filename: str = "dendrogram",
) -> Path:
    """Create interactive dendrogram with Plotly.

    Args:
        distance_matrix: Distance matrix (n_features x n_features).
        labels: Feature names.
        title: Plot title.
        filename: Output filename.

    Returns:
        Path to saved HTML.
    """
    import plotly.figure_factory as ff  # type: ignore[import-untyped]

    # Plotly's create_dendrogram expects a data matrix, not linkage matrix
    # It computes linkage internally using the distance matrix
    fig = ff.create_dendrogram(
        distance_matrix,
        labels=labels,
        orientation="left",
    )

    fig.update_layout(
        title=title,
        width=1200,
        height=max(800, len(labels) * 15),
        yaxis=dict(tickfont=dict(size=8)),
    )

    return _save_plotly(fig, filename)


def plot_dendrogram_matplotlib(
    linkage_matrix: np.ndarray,
    labels: list[str],
    title: str = "Feature Clustering Dendrogram",
    filename: str = "dendrogram",
) -> Path:
    """Create static dendrogram with Matplotlib.

    Args:
        linkage_matrix: Linkage matrix.
        labels: Feature names.
        title: Plot title.
        filename: Output filename.

    Returns:
        Path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_DENDROGRAM)

    hierarchy.dendrogram(
        linkage_matrix,
        labels=labels,
        orientation="left",
        ax=ax,
        leaf_font_size=6,
    )

    ax.set_title(title)

    return _save_matplotlib(fig, filename)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[str],
    title: str = "Feature Clustering Dendrogram",
    filename: str = "dendrogram",
    distance_matrix: np.ndarray | None = None,
) -> tuple[Path | None, Path]:
    """Create both interactive and static dendrograms.

    Args:
        linkage_matrix: Linkage matrix for matplotlib dendrogram.
        labels: Feature names.
        title: Plot title.
        filename: Output filename.
        distance_matrix: Optional distance matrix for plotly dendrogram.
            If not provided, plotly dendrogram is skipped.

    Returns:
        Tuple of (html_path, png_path). html_path may be None if no distance_matrix.
    """
    html_path: Path | None = None
    if distance_matrix is not None:
        html_path = plot_dendrogram_plotly(distance_matrix, labels, title, filename)
    png_path = plot_dendrogram_matplotlib(linkage_matrix, labels, title, filename)
    return html_path, png_path


# ============================================================================
# SCATTER PLOTS (Embeddings)
# ============================================================================

def plot_embedding_plotly(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    color_col: str | None = None,
    title: str = "Feature Embedding",
    filename: str = "embedding",
) -> Path:
    """Create interactive scatter plot for embeddings.

    Args:
        df: DataFrame with embedding coordinates.
        x_col: X coordinate column.
        y_col: Y coordinate column.
        label_col: Column for point labels (hover).
        color_col: Column for color coding (optional).
        title: Plot title.
        filename: Output filename.

    Returns:
        Path to saved HTML.
    """
    import plotly.express as px  # type: ignore[import-untyped]

    if color_col and color_col in df.columns:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            hover_name=label_col,
            title=title,
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            hover_name=label_col,
            title=title,
        )

    fig.update_traces(textposition="top center")
    fig.update_layout(
        width=1000,
        height=800,
    )

    return _save_plotly(fig, filename)


def plot_embedding_matplotlib(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    color_col: str | None = None,
    title: str = "Feature Embedding",
    filename: str = "embedding",
) -> Path:
    """Create static scatter plot for embeddings.

    Args:
        df: DataFrame with embedding coordinates.
        x_col: X coordinate column.
        y_col: Y coordinate column.
        label_col: Column for labels.
        color_col: Column for colors.
        title: Plot title.
        filename: Output filename.

    Returns:
        Path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)

    if color_col and color_col in df.columns:
        scatter = ax.scatter(
            df[x_col],
            df[y_col],
            c=df[color_col],
            cmap="viridis",
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax, label=color_col)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.7)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)

    return _save_matplotlib(fig, filename)


def plot_embedding(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    color_col: str | None = None,
    title: str = "Feature Embedding",
    filename: str = "embedding",
) -> tuple[Path, Path]:
    """Create both interactive and static embedding plots."""
    html_path = plot_embedding_plotly(df, x_col, y_col, label_col, color_col, title, filename)
    png_path = plot_embedding_matplotlib(df, x_col, y_col, label_col, color_col, title, filename)
    return html_path, png_path


# ============================================================================
# ACF/PACF PLOTS
# ============================================================================

def plot_acf_pacf(
    acf_values: np.ndarray,
    pacf_values: np.ndarray,
    feature_name: str,
    filename: str | None = None,
) -> Path:
    """Plot ACF and PACF for a single feature.

    Args:
        acf_values: ACF values.
        pacf_values: PACF values.
        feature_name: Feature name for title.
        filename: Output filename.

    Returns:
        Path to saved PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_ACF)

    lags = range(len(acf_values))

    # ACF plot
    ax1.bar(lags, acf_values, width=0.8, color="steelblue")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axhline(y=1.96 / np.sqrt(len(acf_values) * 10), color="red", linestyle="--", alpha=0.5)
    ax1.axhline(y=-1.96 / np.sqrt(len(acf_values) * 10), color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Lag")
    ax1.set_ylabel("ACF")
    ax1.set_title(f"ACF - {feature_name}")

    # PACF plot
    ax2.bar(lags, pacf_values, width=0.8, color="darkorange")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axhline(y=1.96 / np.sqrt(len(pacf_values) * 10), color="red", linestyle="--", alpha=0.5)
    ax2.axhline(y=-1.96 / np.sqrt(len(pacf_values) * 10), color="red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("PACF")
    ax2.set_title(f"PACF - {feature_name}")

    plt.tight_layout()

    filename = filename or f"acf_pacf_{feature_name}"
    return _save_matplotlib(fig, filename)


# ============================================================================
# STATIONARITY SUMMARY PLOT
# ============================================================================

def plot_stationarity_summary(
    result_df: pd.DataFrame,
    filename: str = "stationarity_summary",
) -> tuple[Path, Path]:
    """Plot summary of stationarity test results.

    Args:
        result_df: DataFrame from stationarity analysis.
        filename: Output filename.

    Returns:
        Tuple of paths.
    """
    import plotly.express as px  # type: ignore[import-untyped]

    # Count by conclusion
    counts = result_df["stationarity_conclusion"].value_counts().reset_index()
    counts.columns = ["conclusion", "count"]

    # Plotly pie chart
    fig = px.pie(
        counts,
        values="count",
        names="conclusion",
        title="Stationarity Test Results",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    html_path = _save_plotly(fig, filename)

    # Matplotlib bar chart
    fig_mpl, ax = plt.subplots(figsize=(8, 6))

    colors = {"stationary": "green", "non_stationary": "red",
              "trend_stationary": "orange", "uncertain": "gray",
              "insufficient_data": "lightgray"}

    bar_colors = [colors.get(c, "blue") for c in counts["conclusion"]]
    ax.bar(counts["conclusion"], counts["count"], color=bar_colors)
    ax.set_xlabel("Conclusion")
    ax.set_ylabel("Number of Features")
    ax.set_title("Stationarity Test Results")
    plt.xticks(rotation=45, ha="right")

    png_path = _save_matplotlib(fig_mpl, filename)

    return html_path, png_path


# ============================================================================
# TARGET CORRELATION PLOTS
# ============================================================================

def plot_target_correlations(
    target_df: pd.DataFrame,
    filename: str = "target_correlations",
    top_n: int = TOP_N_FEATURES,
) -> tuple[Path, Path]:
    """Plot feature-target correlations.

    Args:
        target_df: DataFrame with correlation metrics.
        filename: Output filename.
        top_n: Number of features to show.

    Returns:
        Tuple of paths.
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    plot_df = target_df.head(top_n).copy()

    # Plotly with dual bars
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Bar(
        name="Pearson",
        x=plot_df["feature"],
        y=plot_df["abs_pearson"],
        marker_color="steelblue",
    ))

    fig.add_trace(go.Bar(
        name="Spearman",
        x=plot_df["feature"],
        y=plot_df["abs_spearman"],
        marker_color="darkorange",
    ))

    fig.update_layout(
        title=f"Top {top_n} Features by Target Correlation",
        barmode="group",
        xaxis_tickangle=45,
        height=600,
        width=1200,
    )

    html_path = _save_plotly(fig, filename)

    # Matplotlib
    fig_mpl, ax = plt.subplots(figsize=FIGSIZE_BAR)

    x = np.arange(len(plot_df))
    width = 0.35

    ax.barh(x - width / 2, plot_df["abs_pearson"], width, label="Pearson", color="steelblue")
    ax.barh(x + width / 2, plot_df["abs_spearman"], width, label="Spearman", color="darkorange")

    ax.set_yticks(x)
    ax.set_yticklabels(plot_df["feature"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Absolute Correlation")
    ax.set_title(f"Top {top_n} Features by Target Correlation")
    ax.legend()

    png_path = _save_matplotlib(fig_mpl, filename)

    return html_path, png_path
