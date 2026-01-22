#!/usr/bin/env python
"""
Generate evaluation dashboard from Inspect AI logs.

Usage:
    python -m he_rel.analysis.run_analysis [--results-dir RESULTS_DIR] [--output-dir OUTPUT_DIR]
    
Example:
    python -m he_rel.analysis.run_analysis --results-dir results --output-dir assets/dashboard
"""

import argparse
from pathlib import Path

from .log_analyzer import LogAnalyzer
from .dashboard import (
    plot_accuracy_comparison,
    plot_token_usage,
    plot_error_breakdown,
    plot_error_pie_chart,
    plot_accuracy_vs_tokens,
    plot_heatmap_per_question,
    plot_heatmap_by_category,
    plot_duration_comparison,
    plot_name_error_token_limit,
    generate_html_dashboard,
    # New pass@k and efficiency functions
    plot_pass_at_k_curves,
    plot_pass_at_k_distribution,
    plot_efficiency_analysis,
    compute_paper_statistics,
    compute_category_statistics,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation dashboard from Inspect AI logs"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing .eval log files (default: results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/dashboard",
        help="Directory for output files (default: assets/dashboard)"
    )
    parser.add_argument(
        "--dashboard-path",
        type=str,
        default="dashboard.html",
        help="Path for HTML dashboard (default: dashboard.html)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating individual plot images"
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip exporting CSV files"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable log loading cache"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("HumanEval-Rel Evaluation Analysis")
    print("=" * 60)
    
    # Load and analyze
    print(f"\n📂 Loading logs from: {results_dir}")
    analyzer = LogAnalyzer(results_dir, use_cache=not args.no_cache)
    analyzer.load_logs()
    
    if not analyzer.logs:
        print("❌ No evaluation logs found!")
        return 1
    
    print(f"\n📊 Analyzing {len(analyzer.logs)} logs...")
    analyzer.analyze_all()
    
    # Get dataframes
    summary_df = analyzer.get_summary_dataframe()
    error_df = analyzer.get_error_breakdown_dataframe()
    per_q_df = analyzer.get_per_question_dataframe()
    samples_df = analyzer.get_samples_dataframe()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 RESULTS SUMMARY")
    print("=" * 60)
    
    for _, row in summary_df.sort_values("accuracy", ascending=False).iterrows():
        print(f"  {row['model']:35} accuracy={row['accuracy']:.1%} ± {row['stderr']:.1%}")
    
    # Generate plots
    if not args.no_plots:
        print(f"\n🎨 Generating visualizations to: {output_dir}")
        plot_accuracy_comparison(summary_df, output_dir)
        print("  ✓ accuracy_comparison.png")
        
        plot_token_usage(summary_df, output_dir)
        print("  ✓ token_usage.png")
        
        plot_error_breakdown(error_df, output_dir)
        print("  ✓ error_breakdown.png")
        
        plot_error_pie_chart(error_df, output_dir)
        print("  ✓ error_pie_chart.png")
        
        plot_accuracy_vs_tokens(summary_df, output_dir)
        print("  ✓ accuracy_vs_tokens.png")
        
        plot_heatmap_by_category(per_q_df, output_dir)
        print("  ✓ category_heatmap.png")
        
        plot_heatmap_per_question(per_q_df, output_dir)
        print("  ✓ question_heatmap.png")
        
        plot_duration_comparison(summary_df, output_dir)
        print("  ✓ duration_comparison.png")
        
        plot_name_error_token_limit(samples_df, output_dir)
        print("  ✓ name_error_token_limit.png")
        
        # New pass@k and efficiency plots
        plot_pass_at_k_curves(per_q_df, output_dir)
        print("  ✓ pass_at_k_curves.png/pdf")
        
        plot_pass_at_k_distribution(per_q_df, output_dir, k=1)
        print("  ✓ pass_at_1_distribution.png/pdf")
        
        plot_efficiency_analysis(summary_df, output_dir)
        print("  ✓ efficiency_analysis.png/pdf")
    
    # Export CSVs
    if not args.no_csv:
        print(f"\n📁 Exporting CSV files to: {output_dir}")
        summary_df.to_csv(output_dir / "model_summary.csv", index=False)
        print("  ✓ model_summary.csv")
        
        error_df.to_csv(output_dir / "error_breakdown.csv", index=False)
        print("  ✓ error_breakdown.csv")
        
        per_q_df.to_csv(output_dir / "per_question_scores.csv", index=False)
        print("  ✓ per_question_scores.csv")
        
        # New statistics exports
        compute_paper_statistics(summary_df, per_q_df, output_dir)
        compute_category_statistics(per_q_df, output_dir)
    
    # Export parquet for inspect-viz
    print(f"\n📦 Exporting parquet for inspect-viz...")
    analyzer.export_to_parquet(output_dir / "eval_results.parquet")
    
    # Generate HTML dashboard
    print(f"\n🌐 Generating HTML dashboard: {args.dashboard_path}")
    generate_html_dashboard(analyzer, args.dashboard_path)
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)
    print(f"\n  Dashboard:  {args.dashboard_path}")
    print(f"  Plots:      {output_dir}/*.png")
    print(f"  Data:       {output_dir}/*.csv")
    print(f"  Parquet:    {output_dir}/eval_results.parquet")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
