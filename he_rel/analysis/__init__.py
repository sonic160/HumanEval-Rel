"""
Analysis module for HumanEval-Rel evaluation results.

This module provides tools for analyzing Inspect AI evaluation logs,
computing metrics, and generating visualizations.
"""

from .log_analyzer import (
    LogAnalyzer,
    ModelMetrics,
    SampleAnalysis,
    analyze_log,
    analyze_samples_detailed,
    pass_at_k,
    compute_pass_at_k_per_question,
    categorize_error,
    extract_missing_module,
)

from .dashboard import (
    plot_accuracy_comparison,
    plot_token_usage,
    plot_error_breakdown,
    plot_error_pie_chart,
    plot_accuracy_vs_tokens,
    plot_heatmap_per_question,
    plot_heatmap_by_category,
    plot_duration_comparison,
    generate_html_dashboard,
)

__all__ = [
    # Analyzer
    "LogAnalyzer",
    "ModelMetrics", 
    "SampleAnalysis",
    "analyze_log",
    "analyze_samples_detailed",
    "pass_at_k",
    "compute_pass_at_k_per_question",
    "categorize_error",
    "extract_missing_module",
    # Dashboard
    "plot_accuracy_comparison",
    "plot_token_usage",
    "plot_error_breakdown",
    "plot_error_pie_chart",
    "plot_accuracy_vs_tokens",
    "plot_heatmap_per_question",
    "plot_heatmap_by_category",
    "plot_duration_comparison",
    "generate_html_dashboard",
]
