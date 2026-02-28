"""
Dashboard visualizations for Inspect AI evaluation results.

Creates interactive visualizations using matplotlib and plotly
for comprehensive analysis of model benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    import pandas as pd
    from .log_analyzer import LogAnalyzer


# =============================================================================
# Pass@k Computation
# =============================================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k estimator (unbiased, Chen et al. 2021).
    
    Args:
        n: Total number of samples
        c: Number of correct samples  
        k: k value for pass@k
    
    Returns:
        Estimated pass@k probability
    """
    if n - c < k:
        return 1.0
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def set_style():
    """Set consistent plot style with larger fonts for publication."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'


def plot_accuracy_comparison(df: pd.DataFrame, output_dir: Path | None = None) -> plt.Figure:
    """Bar chart comparing accuracy across models with error bars."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by accuracy
    df_sorted = df.sort_values("accuracy", ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_sorted)))
    
    bars = ax.barh(
        df_sorted["model"],
        df_sorted["accuracy"],
        xerr=df_sorted["stderr"],
        color=colors,
        capsize=5,
        edgecolor='black',
        linewidth=1.2
    )
    
    # Add value labels
    for bar, acc in zip(bars, df_sorted["accuracy"]):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f'{acc:.1%}',
            va='center',
            fontsize=12,
            fontweight='bold'
        )
    
    ax.set_xlabel("Accuracy", fontsize=14, fontweight='bold')
    ax.set_title("Model Accuracy Comparison (HumanEval-Rel)", fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline', linewidth=2)
    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(axis='y', labelsize=11)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "accuracy_comparison.png", dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / "accuracy_comparison.pdf", bbox_inches='tight')
    
    return fig


def plot_token_usage(df: pd.DataFrame, output_dir: Path | None = None) -> plt.Figure:
    """Bar chart of token usage per model."""
    set_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df_sorted = df.sort_values("total_tokens", ascending=True)
    
    # Total tokens (in thousands)
    ax1 = axes[0]
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(df_sorted)))
    bars1 = ax1.barh(df_sorted["model"], df_sorted["total_tokens"] / 1000, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel("Total Tokens (thousands)", fontsize=14, fontweight='bold')
    ax1.set_title("Total Token Consumption", fontsize=15, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.tick_params(axis='y', labelsize=11)
    for bar, val in zip(bars1, df_sorted["total_tokens"]):
        ax1.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f'{val/1000:.0f}K', va='center', fontsize=11, fontweight='bold')
    
    # Tokens per sample
    ax2 = axes[1]
    df_sorted2 = df.sort_values("tokens_per_sample", ascending=True)
    colors2 = plt.cm.Blues(np.linspace(0.3, 0.9, len(df_sorted2)))
    bars2 = ax2.barh(df_sorted2["model"], df_sorted2["tokens_per_sample"], color=colors2, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel("Tokens per Sample", fontsize=14, fontweight='bold')
    ax2.set_title("Average Token Usage per Sample", fontsize=15, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='y', labelsize=11)
    for bar, val in zip(bars2, df_sorted2["tokens_per_sample"]):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f}', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "token_usage.png", dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / "token_usage.pdf", bbox_inches='tight')
    
    return fig


def plot_error_breakdown(error_df: pd.DataFrame, output_dir: Path | None = None) -> plt.Figure:
    """Stacked bar chart showing error type breakdown per model."""
    set_style()
    
    models = error_df["model"].tolist()
    
    def _plot_on_ax(ax, df, title, exclude_types=None, exclude_first_color = False):
        # Prepare data - exclude 'model' column and excluded types
        error_types = [c for c in df.columns if c != "model"]
        if exclude_types:
            error_types = [et for et in error_types if et not in exclude_types]
        
        # Sort error types by total occurrence
        totals = {et: df[et].sum() for et in error_types}
        error_types = sorted(error_types, key=lambda x: totals[x], reverse=True)
        
        # Take top N error types
        top_n = 10
        plot_df = df.copy()
        if len(error_types) > top_n:
            other_types = error_types[top_n:]
            error_types = error_types[:top_n]
            # Sum others
            plot_df["Other"] = plot_df[other_types].sum(axis=1)
            error_types.append("Other")
        
        # Create stacked bars
        x = np.arange(len(models))
        width = 0.7
        

        colors = plt.cm.Set3(np.linspace(0, 1, len(error_types)))
        if exclude_first_color:
            colors = plt.cm.Set3(np.linspace(0, 1, len(error_types)+1))[1:]
        
        bottom = np.zeros(len(models))
        for i, error_type in enumerate(error_types):
            if error_type in plot_df.columns:
                values = plot_df[error_type].values
                ax.bar(x, values, width, bottom=bottom, label=error_type, color=colors[i])
                bottom += values
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel("Error Count")
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=9, ncol=2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    _plot_on_ax(ax1, error_df, "Error Type Breakdown by Model")
    _plot_on_ax(ax2, error_df, "Error Type Breakdown (excluding AssertionError)", exclude_types=["AssertionError"], exclude_first_color=True)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "error_breakdown.png", dpi=150, bbox_inches='tight')
    
    return fig


def plot_error_pie_chart(error_df: pd.DataFrame, output_dir: Path | None = None) -> plt.Figure:
    """Pie chart showing overall error distribution."""
    set_style()
    
    # Sum errors across all models
    error_types = [c for c in error_df.columns if c != "model"]
    totals = {et: error_df[et].sum() for et in error_types}
    
    # Sort and group small slices
    sorted_errors = sorted(totals.items(), key=lambda x: -x[1])
    
    labels = []
    sizes = []
    rare_count = 0
    total_errors = sum(totals.values())
    
    for error_type, count in sorted_errors:
        pct = count / total_errors * 100 if total_errors > 0 else 0
        if pct >= 2.0:  # Show errors >= 2%
            labels.append(error_type)
            sizes.append(count)
        else:
            rare_count += count
    
    if rare_count > 0:
        labels.append("Rare Errors")
        sizes.append(rare_count)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    def make_autopct(sizes):
        def autopct(pct):
            total = sum(sizes)
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}%\n({val:,})'
        return autopct
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=make_autopct(sizes),
        colors=colors,
        startangle=90,
        pctdistance=0.75,
    )
    
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
    
    ax.set_title(f'Error Type Distribution\n(n={total_errors:,} total errors)', fontsize=14)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "error_pie_chart.png", dpi=150, bbox_inches='tight')
    
    return fig


def plot_accuracy_vs_tokens(df: pd.DataFrame, output_dir: Path | None = None) -> plt.Figure:
    """Scatter plot of accuracy vs token usage."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(
        df["tokens_per_sample"],
        df["accuracy"],
        s=100,
        c=range(len(df)),
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add labels
    for _, row in df.iterrows():
        ax.annotate(
            row["model"],
            (row["tokens_per_sample"], row["accuracy"]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    ax.set_xlabel("Tokens per Sample")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Token Usage")
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "accuracy_vs_tokens.png", dpi=150, bbox_inches='tight')
    
    return fig


def natural_sort_key(s):
    """Sort key for natural sorting of strings with numbers (Q1, Q2, Q10, not Q1, Q10, Q2)."""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def plot_heatmap_per_question(
    per_q_df: pd.DataFrame,
    output_dir: Path | None = None,
    max_questions: int = 50
) -> plt.Figure:
    """Heatmap of pass rates per question per model."""
    # Don't use seaborn style for heatmaps - it adds unwanted grid
    plt.rcParams['figure.figsize'] = (20, 8)
    plt.rcParams['font.size'] = 12
    
    # Pivot to get model x question matrix
    pivot_df = per_q_df.pivot(index="model", columns="question_id", values="pass_rate")
    
    # Limit questions if too many
    if pivot_df.shape[1] > max_questions:
        # Select questions with most variance
        variance = pivot_df.var()
        top_questions = variance.nlargest(max_questions).index
        pivot_df = pivot_df[top_questions]
    
    # Sort columns using natural sort (Q0, Q1, Q2, ..., Q10, not Q0, Q1, Q10, Q2)
    sorted_columns = sorted(pivot_df.columns, key=natural_sort_key)
    pivot_df = pivot_df.reindex(sorted_columns, axis=1)
    
    # Fill NaN with 0 for display
    pivot_df = pivot_df.fillna(0)
    
    # Adjust figure width based on number of questions
    fig_width = max(18, len(pivot_df.columns) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    # Use pcolormesh for cleaner rendering without grid artifacts
    im = ax.pcolormesh(
        pivot_df.values, 
        cmap='RdYlGn', 
        vmin=0, 
        vmax=1,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Y-axis labels (models) - center in cells
    ax.set_yticks(np.arange(len(pivot_df.index)) + 0.5)
    ax.set_yticklabels(pivot_df.index, fontsize=12, fontweight='bold')
    
    # X-axis labels - center in cells
    ax.set_xticks(np.arange(len(pivot_df.columns)) + 0.5)
    ax.set_xticklabels(
        pivot_df.columns,
        rotation=45, ha='right', fontsize=9
    )
    
    ax.set_title("Pass Rate by Individual Question and Model", fontsize=16, fontweight='bold')
    ax.set_xlabel("Question ID", fontsize=14, fontweight='bold')
    ax.set_ylabel("Model", fontsize=14, fontweight='bold')
    
    # Invert y-axis so first model is at top
    ax.invert_yaxis()
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Pass Rate", shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Pass Rate", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "question_heatmap.png", dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / "question_heatmap.pdf", bbox_inches='tight')
    
    return fig


def plot_heatmap_by_category(
    per_q_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> plt.Figure:
    """Heatmap of pass rates grouped by question category (aggregated).
    
    This provides a clearer view by grouping similar questions together.
    Models sorted by strength (top to bottom), categories sorted by difficulty (hardest left, easiest right).
    """
    # Don't use seaborn style for heatmaps - it adds unwanted grid
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    
    # Extract category from question_id 
    # Format is now "category[Qx]" e.g., "reliability_metrics_and_evaluation[Q0]"
    per_q_df = per_q_df.copy()
    per_q_df['category'] = per_q_df['question_id'].apply(
        lambda x: x.split('[')[0] if '[' in x else x
    )
    
    # Aggregate by category and model
    category_df = per_q_df.groupby(['model', 'category']).agg({
        'pass_rate': 'mean',
        'total': 'sum',
        'passed': 'sum'
    }).reset_index()
    
    # Pivot to get model x category matrix
    pivot_df = category_df.pivot(index="model", columns="category", values="pass_rate")
    
    # Sort columns by category difficulty (average pass rate): easiest on left, hardest on right
    category_avgs = pivot_df.mean(axis=0).sort_values(ascending=False)
    pivot_df = pivot_df[category_avgs.index]
    
    # Sort rows by model strength (average pass rate): strongest at top (will be inverted in display)
    model_strength = pivot_df.mean(axis=1).sort_values(ascending=False)
    pivot_df = pivot_df.loc[model_strength.index]
    
    # Fill NaN with 0 for display
    pivot_df = pivot_df.fillna(0)
    
    # Compute category averages (across all models) - already sorted by difficulty
    category_avgs = pivot_df.mean(axis=0)
    
    # Compute model averages (across all categories)
    model_avgs = pivot_df.mean(axis=1)
    
    # Add model averages as a column in the pivot table
    pivot_df_with_avg = pivot_df.copy()
    pivot_df_with_avg['Average'] = model_avgs
    
    # Create figure with appropriate size
    n_categories = len(pivot_df.columns)
    n_models = len(pivot_df.index)
    fig_width = max(12, n_categories * 1.5 + 3.5)
    fig_height = max(8, n_models * 0.9 + 2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Use pcolormesh for cleaner rendering
    im = ax.pcolormesh(
        pivot_df_with_avg.values, 
        cmap='RdYlGn', 
        vmin=0, 
        vmax=1,
        edgecolors='white',
        linewidth=1.5
    )
    
    # Y-axis labels (models) - center in cells
    ax.set_yticks(np.arange(len(pivot_df_with_avg.index)) + 0.5)
    ax.set_yticklabels(pivot_df_with_avg.index, fontsize=12, fontweight='bold')
    
    # X-axis labels (categories + Average) - center in cells
    ax.set_xticks(np.arange(len(pivot_df_with_avg.columns)) + 0.5)
    # Clean up category names for display
    category_labels = [c.replace('_', ' ').title() for c in pivot_df.columns]
    category_labels.append('Average')
    ax.set_xticklabels(category_labels, rotation=25, ha='right', fontsize=11, fontweight='bold')
    
    ax.set_title("")
    ax.set_xlabel("", fontsize=14, fontweight='bold')
    ax.set_ylabel("Model", fontsize=13, fontweight='bold')
    
    # Invert y-axis so first model is at top
    ax.invert_yaxis()
    
    # Add value annotations in each cell
    for i in range(len(pivot_df_with_avg.index)):
        for j in range(len(pivot_df_with_avg.columns)):
            value = pivot_df_with_avg.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if value < 0.4 or value > 0.7 else 'black'
                ax.text(j + 0.5, i + 0.5, f'{value:.0%}', ha='center', va='center',
                       fontsize=11, fontweight='bold', color=text_color)
    
    # Adjust margins to show content properly
    ax.set_ylim(len(pivot_df_with_avg.index), -0.5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Pass Rate", shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Pass Rate", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "category_heatmap.png", dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / "category_heatmap.pdf", bbox_inches='tight')
    
    return fig


def plot_duration_comparison(df: pd.DataFrame, output_dir: Path | None = None) -> plt.Figure:
    """Bar chart comparing evaluation duration."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    df_sorted = df.sort_values("duration_minutes", ascending=True)
    
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(df_sorted)))
    bars = ax.barh(df_sorted["model"], df_sorted["duration_minutes"], color=colors)
    
    for bar, val in zip(bars, df_sorted["duration_minutes"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f} min', va='center', fontsize=9)
    
    ax.set_xlabel("Duration (minutes)")
    ax.set_title("Evaluation Duration by Model")
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "duration_comparison.png", dpi=150, bbox_inches='tight')
    
    return fig


def plot_name_error_token_limit(samples_df: pd.DataFrame, output_dir: Path | None = None) -> plt.Figure:
    """Plot count of No Output cases that hit the token limit."""
    set_style()
    
    # Filter for No Output and token limit (10000)
    limit = 10000
    filtered = samples_df[
        (samples_df['error_type'] == 'No Output') & 
        (samples_df['total_tokens'] >= limit)
    ]
    
    # Count per model
    counts = filtered.groupby('model').size().reset_index(name='count')
    
    # Ensure all models are present even if count is 0
    all_models = samples_df['model'].unique()
    counts = counts.set_index('model').reindex(all_models, fill_value=0).reset_index()
    counts = counts.sort_values('count', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(counts)))
    bars = ax.bar(counts['model'], counts['count'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"No Output Cases at Token Limit (>= {limit} tokens)")
    ax.set_xticks(range(len(counts['model'])))
    ax.set_xticklabels(counts['model'], rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "name_error_token_limit.png", dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# Pass@k Analysis Functions (NEW)
# =============================================================================

def compute_pass_at_k_dataframe(
    per_q_df: pd.DataFrame,
    k_values: list[int] = None
) -> pd.DataFrame:
    """
    Compute pass@k for all k values for each model-question pair.
    
    Args:
        per_q_df: DataFrame with columns [model, question_id, total, passed]
        k_values: List of k values to compute (default: 1-20)
        
    Returns:
        DataFrame with pass@k columns for each k value
    """
    if k_values is None:
        k_values = [1,2,3,5,10,15,20]
    
    rows = []
    for _, row in per_q_df.iterrows():
        n = int(row['total'])
        c = int(row['passed'])
        result = {
            'model': row['model'],
            'question_id': row['question_id'],
            'n': n,
            'c': c,
        }
        for k in k_values:
            if k <= n:
                result[f'pass_at_{k}'] = pass_at_k(n, c, k)
            else:
                result[f'pass_at_{k}'] = np.nan
        rows.append(result)
    return pd.DataFrame(rows)


def plot_pass_at_k_curves(
    per_q_df: pd.DataFrame,
    output_dir: Path | None = None,
    k_values: list[int] = None
) -> plt.Figure:
    """
    Plot pass@k curves showing how performance improves with more attempts.
    Uses log scale for k and shows 95% confidence intervals.
    
    Args:
        per_q_df: DataFrame with columns [model, question_id, total, passed]
        output_dir: Directory to save figures
        k_values: List of k values for x-axis (default: 1-20)
        
    Returns:
        matplotlib Figure
    """
    set_style()
    
    # Default k values from 1 to 20
    if k_values is None:
        k_values = [1, 2, 3, 5, 10, 15, 20]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute pass@k for each model
    models = per_q_df['model'].unique()
    
    # Color map for models - sorted by pass@1 performance
    model_pass1 = {}
    for model in models:
        model_data = per_q_df[per_q_df['model'] == model]
        avg_pass1 = model_data.apply(
            lambda row: pass_at_k(int(row['total']), int(row['passed']), 1), 
            axis=1
        ).mean()
        model_pass1[model] = avg_pass1
    
    sorted_models = sorted(models, key=lambda m: model_pass1[m], reverse=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_models)))
    
    for idx, model in enumerate(sorted_models):
        model_data = per_q_df[per_q_df['model'] == model]
        pass_at_k_means = []
        
        for k in k_values:
            # Compute pass@k for each question
            question_pass_at_k = model_data.apply(
                lambda row, k=k: pass_at_k(int(row['total']), int(row['passed']), k) 
                    if k <= row['total'] else np.nan,
                axis=1
            ).dropna()
            
            if len(question_pass_at_k) == 0:
                pass_at_k_means.append(np.nan)
                continue
            
            # Mean pass@k across questions
            mean_pass_k = question_pass_at_k.mean()
            pass_at_k_means.append(mean_pass_k)
        
        # Plot line with markers
        ax.plot(k_values, pass_at_k_means, marker='o', markersize=5, 
                label=model, color=colors[idx], linewidth=2.5)
    
    # Use log scale for x-axis
    ax.set_xscale('log')
    # Custom tick positions for log scale
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.minorticks_off()
    
    ax.set_xlabel("k (number of attempts)", fontsize=14, fontweight='bold')
    ax.set_ylabel("pass@k", fontsize=14, fontweight='bold')
    ax.set_title("pass@k Curves by Model", fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.9, 22)
    ax.grid(True, alpha=0.3, which='major')
    ax.tick_params(axis='both', labelsize=12)
    
    # Add horizontal line at 50%
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "pass_at_k_curves.png", dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / "pass_at_k_curves.pdf", bbox_inches='tight')
    
    return fig


def plot_pass_at_k_distribution(
    per_q_df: pd.DataFrame,
    output_dir: Path | None = None,
    k: int = 1
) -> plt.Figure:
    """
    Plot distribution of pass@k values across questions (avg and max over models).
    
    Args:
        per_q_df: DataFrame with columns [model, question_id, total, passed]
        output_dir: Directory to save figures  
        k: The k value to analyze
        
    Returns:
        matplotlib Figure
    """
    set_style()
    
    # Compute pass@k for each model-question pair
    per_q_df = per_q_df.copy()
    per_q_df['pass_at_k'] = per_q_df.apply(
        lambda row: pass_at_k(int(row['total']), int(row['passed']), k),
        axis=1
    )
    
    # Compute avg and max pass@k per question (across models)
    question_stats = per_q_df.groupby('question_id').agg(
        avg_pass_k=('pass_at_k', 'mean'),
        max_pass_k=('pass_at_k', 'max'),
        min_pass_k=('pass_at_k', 'min'),
    ).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Average pass@k distribution
    ax1 = axes[0]
    bins = np.linspace(0, 1, 21)
    ax1.hist(question_stats['avg_pass_k'], bins=bins, edgecolor='black', 
             color='steelblue', alpha=0.7, linewidth=1.2)
    ax1.axvline(question_stats['avg_pass_k'].mean(), color='red', 
                linestyle='--', linewidth=2.5, label=f"Mean: {question_stats['avg_pass_k'].mean():.1%}")
    ax1.set_xlabel(f"Average pass@{k} (across models)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Number of Questions", fontsize=14, fontweight='bold')
    ax1.set_title(f"Distribution of Average pass@{k}", fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.tick_params(axis='both', labelsize=12)
    
    # Count never-solved and always-solved
    never_solved = (question_stats['max_pass_k'] == 0).sum()
    always_solved = (question_stats['min_pass_k'] == 1).sum()
    ax1.text(0.02, 0.98, f"Never solved (max=0): {never_solved}\nAlways solved (min=1): {always_solved}", 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Max pass@k distribution (benchmark solvability ceiling)
    ax2 = axes[1]
    ax2.hist(question_stats['max_pass_k'], bins=bins, edgecolor='black',
             color='forestgreen', alpha=0.7, linewidth=1.2)
    ax2.axvline(question_stats['max_pass_k'].mean(), color='red',
                linestyle='--', linewidth=2.5, label=f"Mean: {question_stats['max_pass_k'].mean():.1%}")
    ax2.set_xlabel(f"Max pass@{k} (best model)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Number of Questions", fontsize=14, fontweight='bold')
    ax2.set_title(f"Distribution of Max pass@{k} (Solvability Ceiling)", fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"pass_at_{k}_distribution.png", dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / f"pass_at_{k}_distribution.pdf", bbox_inches='tight')
    
    return fig


def plot_efficiency_analysis(
    summary_df: pd.DataFrame,
    output_dir: Path | None = None
) -> plt.Figure:
    """
    Plot efficiency analysis: tokens per correct answer and accuracy vs efficiency.
    
    Args:
        summary_df: DataFrame with columns [model, accuracy, passed, total_tokens, tokens_per_sample]
        output_dir: Directory to save figures
        
    Returns:
        matplotlib Figure
    """
    set_style()
    
    # Compute tokens per correct answer
    df = summary_df.copy()
    df['tokens_per_correct'] = df['total_tokens'] / df['passed']
    df['tokens_per_correct'] = df['tokens_per_correct'].replace([np.inf, -np.inf], np.nan)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Tokens per correct answer (lower is better)
    ax1 = axes[0]
    df_sorted = df.sort_values('tokens_per_correct', ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted)))
    
    bars = ax1.barh(df_sorted['model'], df_sorted['tokens_per_correct'] / 1000, 
                    color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars, df_sorted['tokens_per_correct']):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val/1000:.1f}K', va='center', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel("Tokens per Correct Answer (thousands)", fontsize=14, fontweight='bold')
    ax1.set_title("Efficiency: Tokens per Correct Answer\n(lower is better)", fontsize=15, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.tick_params(axis='y', labelsize=11)
    
    # Right: Accuracy vs Efficiency scatter
    ax2 = axes[1]
    
    # Compute efficiency score (higher is better): accuracy / (tokens_per_sample / 1000)
    df['efficiency_score'] = df['accuracy'] / (df['tokens_per_sample'] / 1000)
    
    scatter = ax2.scatter(
        df['tokens_per_sample'] / 1000,
        df['accuracy'],
        s=180,
        c=df['efficiency_score'],
        cmap='RdYlGn',
        edgecolors='black',
        linewidth=1.5
    )
    
    # Add labels
    for _, row in df.iterrows():
        ax2.annotate(
            row['model'],
            (row['tokens_per_sample'] / 1000, row['accuracy']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )
    
    ax2.set_xlabel("Tokens per Sample (thousands)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
    ax2.set_title("Accuracy vs Token Usage\n(color = efficiency score)", fontsize=15, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='both', labelsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2, label="Efficiency Score")
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Efficiency Score", fontsize=12, fontweight='bold')
    
    # Add iso-efficiency lines
    for eff in [0.2, 0.4, 0.6]:
        x_range = np.linspace(0.5, 8, 100)
        y_iso = eff * x_range
        ax2.plot(x_range, y_iso, '--', alpha=0.3, color='gray')
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "efficiency_analysis.png", dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / "efficiency_analysis.pdf", bbox_inches='tight')
    
    return fig


def compute_paper_statistics(
    summary_df: pd.DataFrame,
    per_q_df: pd.DataFrame,
    output_dir: Path | None = None
) -> dict:
    """
    Compute all statistics needed for the paper.
    
    Returns:
        Dictionary with computed statistics
    """
    stats_dict = {}
    
    # Basic statistics
    stats_dict['n_models'] = len(summary_df)
    stats_dict['n_questions'] = per_q_df['question_id'].nunique()
    stats_dict['samples_per_question'] = int(per_q_df['total'].iloc[0])
    stats_dict['total_samples'] = int(summary_df['total_samples'].sum())
    
    # Accuracy statistics
    stats_dict['mean_accuracy'] = summary_df['accuracy'].mean()
    stats_dict['std_accuracy'] = summary_df['accuracy'].std()
    stats_dict['min_accuracy'] = summary_df['accuracy'].min()
    stats_dict['max_accuracy'] = summary_df['accuracy'].max()
    stats_dict['best_model'] = summary_df.loc[summary_df['accuracy'].idxmax(), 'model']
    stats_dict['worst_model'] = summary_df.loc[summary_df['accuracy'].idxmin(), 'model']
    
    # Compute max pass@1 per question (best model performance)
    per_q_df = per_q_df.copy()
    per_q_df['pass_at_1'] = per_q_df.apply(
        lambda row: pass_at_k(int(row['total']), int(row['passed']), 1),
        axis=1
    )
    
    question_max = per_q_df.groupby('question_id')['pass_at_1'].max()
    
    # Never solved questions (max pass@1 = 0 across all models)
    never_solved = (question_max == 0).sum()
    stats_dict['never_solved_count'] = never_solved
    stats_dict['never_solved_pct'] = never_solved / len(question_max)
    stats_dict['never_solved_questions'] = question_max[question_max == 0].index.tolist()
    
    # Always solved questions (min pass@1 = 1 across all models)  
    question_min = per_q_df.groupby('question_id')['pass_at_1'].min()
    always_solved = (question_min == 1).sum()
    stats_dict['always_solved_count'] = always_solved
    stats_dict['always_solved_pct'] = always_solved / len(question_min)
    
    # Token statistics
    stats_dict['mean_tokens_per_sample'] = summary_df['tokens_per_sample'].mean()
    stats_dict['total_tokens'] = summary_df['total_tokens'].sum()
    
    # Efficiency
    summary_df = summary_df.copy()
    summary_df['tokens_per_correct'] = summary_df['total_tokens'] / summary_df['passed']
    stats_dict['mean_tokens_per_correct'] = summary_df['tokens_per_correct'].mean()
    stats_dict['most_efficient_model'] = summary_df.loc[summary_df['tokens_per_correct'].idxmin(), 'model']
    
    # Save to file
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "paper_statistics.txt", 'w') as f:
            f.write("HumanEval-Rel Paper Statistics\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("BASIC COUNTS:\n")
            f.write(f"  Models evaluated: {stats_dict['n_models']}\n")
            f.write(f"  Questions: {stats_dict['n_questions']}\n")
            f.write(f"  Samples per question: {stats_dict['samples_per_question']}\n")
            f.write(f"  Total samples: {stats_dict['total_samples']:,}\n\n")
            
            f.write("ACCURACY:\n")
            f.write(f"  Mean pass@1: {stats_dict['mean_accuracy']:.1%}\n")
            f.write(f"  Std pass@1: {stats_dict['std_accuracy']:.1%}\n")
            f.write(f"  Range: {stats_dict['min_accuracy']:.1%} - {stats_dict['max_accuracy']:.1%}\n")
            f.write(f"  Best model: {stats_dict['best_model']}\n")
            f.write(f"  Worst model: {stats_dict['worst_model']}\n\n")
            
            f.write("QUESTION DIFFICULTY:\n")
            f.write(f"  Never solved (0% by all models): {stats_dict['never_solved_count']} ({stats_dict['never_solved_pct']:.1%})\n")
            f.write(f"  Always solved (100% by all models): {stats_dict['always_solved_count']} ({stats_dict['always_solved_pct']:.1%})\n\n")
            
            if stats_dict['never_solved_questions']:
                f.write("  Never-solved questions:\n")
                for q in stats_dict['never_solved_questions']:
                    f.write(f"    - {q}\n")
                f.write("\n")
            
            f.write("TOKEN USAGE:\n")
            f.write(f"  Mean tokens per sample: {stats_dict['mean_tokens_per_sample']:,.0f}\n")
            f.write(f"  Total tokens: {stats_dict['total_tokens']:,.0f}\n")
            f.write(f"  Mean tokens per correct: {stats_dict['mean_tokens_per_correct']:,.0f}\n")
            f.write(f"  Most efficient model: {stats_dict['most_efficient_model']}\n")
        
        print(f"  ✓ paper_statistics.txt")
    
    return stats_dict


def compute_category_statistics(
    per_q_df: pd.DataFrame,
    output_dir: Path | None = None
) -> pd.DataFrame:
    """
    Compute statistics broken down by question category.
    
    Returns:
        DataFrame with per-category statistics
    """
    # Extract category from question_id
    per_q_df = per_q_df.copy()
    per_q_df['category'] = per_q_df['question_id'].apply(
        lambda x: x.split('[')[0] if '[' in x else x
    )
    
    # Compute pass@1 for each row
    per_q_df['pass_at_1'] = per_q_df.apply(
        lambda row: pass_at_k(int(row['total']), int(row['passed']), 1),
        axis=1
    )
    
    # Aggregate by category
    category_stats = per_q_df.groupby('category').agg(
        n_questions=('question_id', 'nunique'),
        mean_pass_1=('pass_at_1', 'mean'),
        std_pass_1=('pass_at_1', 'std'),
        min_pass_1=('pass_at_1', 'min'),
        max_pass_1=('pass_at_1', 'max'),
    ).reset_index()
    
    # Also compute per-model per-category
    model_category = per_q_df.groupby(['model', 'category']).agg(
        mean_pass_1=('pass_at_1', 'mean'),
    ).reset_index()
    
    # Pivot for model comparison by category
    pivot = model_category.pivot(index='model', columns='category', values='mean_pass_1')
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        category_stats.to_csv(output_dir / "category_statistics.csv", index=False)
        pivot.to_csv(output_dir / "model_by_category.csv")
        print(f"  ✓ category_statistics.csv")
        print(f"  ✓ model_by_category.csv")
    
    return category_stats


def generate_html_dashboard(
    analyzer: "LogAnalyzer",
    output_path: str | Path = "dashboard.html"
) -> None:
    """Generate an interactive HTML dashboard with all visualizations."""
    import base64
    from io import BytesIO
    
    output_path = Path(output_path)
    
    # Get dataframes
    summary_df= analyzer.get_summary_dataframe()
    error_df = analyzer.get_error_breakdown_dataframe()
    per_q_df = analyzer.get_per_question_dataframe()
    samples_df = analyzer.get_samples_dataframe()
    
    # Generate plots and convert to base64
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    plots = {}
    plots["accuracy"] = fig_to_base64(plot_accuracy_comparison(summary_df))
    plots["tokens"] = fig_to_base64(plot_token_usage(summary_df))
    plots["errors"] = fig_to_base64(plot_error_breakdown(error_df))
    plots["error_pie"] = fig_to_base64(plot_error_pie_chart(error_df))
    plots["accuracy_vs_tokens"] = fig_to_base64(plot_accuracy_vs_tokens(summary_df))
    plots["heatmap_category"] = fig_to_base64(plot_heatmap_by_category(per_q_df))
    plots["heatmap_questions"] = fig_to_base64(plot_heatmap_per_question(per_q_df))
    plots["duration"] = fig_to_base64(plot_duration_comparison(summary_df))
    plots["name_error_limit"] = fig_to_base64(plot_name_error_token_limit(samples_df))
    
    # Generate summary table HTML
    summary_html = summary_df.loc[:, ["model", "accuracy", "stderr", "tokens_per_sample"]].to_html(
        classes='dataframe',
        index=False,
        float_format=lambda x: f'{x:.4f}' if abs(x) < 100 else f'{x:,.0f}'
    )
    
    # Load template
    template_path = Path(__file__).parent / "templates" / "dashboard.html"
    if not template_path.exists():
        # Fallback if template not found (or create it on the fly if needed, but better to error out or use simple string)
        print(f"Warning: Template not found at {template_path}")
        return

    template_content = template_path.read_text(encoding='utf-8')
    
    # Prepare replacements
    replacements = {
        "{{TIMESTAMP}}": __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "{{METRIC_MODELS}}": str(len(summary_df)),
        "{{METRIC_BEST_ACCURACY}}": f"{summary_df['accuracy'].max():.1%}",
        "{{METRIC_AVG_ACCURACY}}": f"{summary_df['accuracy'].mean():.1%}",
        "{{METRIC_TOTAL_TOKENS}}": f"{summary_df['total_tokens'].sum() / 1_000_000:.1f}M",
        "{{METRIC_TOTAL_SAMPLES}}": f"{summary_df['total_samples'].sum():,}",
        "{{PLOT_ACCURACY}}": plots["accuracy"],
        "{{PLOT_ACCURACY_VS_TOKENS}}": plots["accuracy_vs_tokens"],
        "{{PLOT_TOKENS}}": plots["tokens"],
        "{{PLOT_ERRORS}}": plots["errors"],
        "{{PLOT_ERROR_PIE}}": plots["error_pie"],
        "{{PLOT_NAME_ERROR_LIMIT}}": plots["name_error_limit"],
        "{{PLOT_HEATMAP_CATEGORY}}": plots["heatmap_category"],
        "{{PLOT_HEATMAP_QUESTIONS}}": plots["heatmap_questions"],
        "{{SUMMARY_TABLE}}": summary_html,
    }
    
    # Replace placeholders
    html_content = template_content
    for key, value in replacements.items():
        html_content = html_content.replace(key, value)
    
    output_path.write_text(html_content, encoding='utf-8')
    print(f"✓ Dashboard saved to {output_path}")
    
    # Copy logo if it exists
    logo_path = template_path.parent / "logo.png"
    if logo_path.exists():
        import shutil
        dest_logo = output_path.parent / "logo.png"
        shutil.copy2(logo_path, dest_logo)
        print(f"✓ Logo copied to {dest_logo}")


def main():
    """Generate dashboard from results directory."""
    from .log_analyzer import LogAnalyzer
    
    # Create output directory
    output_dir = Path("assets/dashboard")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and analyze
    analyzer = LogAnalyzer("results")
    analyzer.load_logs()
    analyzer.analyze_all()
    
    # Get dataframes
    summary_df = analyzer.get_summary_dataframe()
    error_df = analyzer.get_error_breakdown_dataframe()
    per_q_df = analyzer.get_per_question_dataframe()
    samples_df = analyzer.get_samples_dataframe()
    
    # Generate individual plots
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(summary_df, output_dir)
    plot_token_usage(summary_df, output_dir)
    plot_error_breakdown(error_df, output_dir)
    plot_error_pie_chart(error_df, output_dir)
    plot_accuracy_vs_tokens(summary_df, output_dir)
    plot_heatmap_per_question(per_q_df, output_dir)
    plot_duration_comparison(summary_df, output_dir)
    plot_name_error_token_limit(samples_df, output_dir)
    
    # Generate HTML dashboard
    generate_html_dashboard(analyzer, "dashboard.html")
    
    # # Export parquet for inspect-viz
    # analyzer.export_to_parquet(output_dir / "eval_results.parquet")
    
    print("\n✓ All visualizations generated!")
    print(f"  - Individual plots: {output_dir}/")
    print("  - HTML Dashboard: dashboard.html")
    # print(f"  - Parquet for inspect-viz: {output_dir}/eval_results.parquet")


if __name__ == "__main__":
    main()
