"""
Analyze Inspect AI evaluation logs.

Extracts metrics from .eval log files:
- Pass rates and accuracy scores
- Token usage statistics
- Error type breakdowns
- Per-sample analysis
- Model comparison metrics
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from inspect_ai.log import EvalLog, EvalLogInfo, list_eval_logs, read_eval_log

import pandas as pd


# Global cache for entry points
_ENTRY_POINTS: dict[str, str] | None = None


def load_entry_points(questions_path: Path | None = None) -> dict[str, str]:
    """Load entry points from questions.jsonl (cached)."""
    global _ENTRY_POINTS
    if _ENTRY_POINTS is not None:
        return _ENTRY_POINTS
    
    if questions_path is None:
        # Try common locations
        for path in [Path("dataset/questions.jsonl"), Path("../dataset/questions.jsonl")]:
            if path.exists():
                questions_path = path
                break
    
    if questions_path is None or not questions_path.exists():
        _ENTRY_POINTS = {}
        return _ENTRY_POINTS
    
    entry_points = {}
    with open(questions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for q in data:
            task_id = q['task_id']
            entry_point = q['entry_point']
            entry_points[task_id] = entry_point
    
    _ENTRY_POINTS = entry_points
    return _ENTRY_POINTS


@dataclass
class ModelMetrics:
    """Metrics for a single model evaluation."""
    
    model_name: str
    model_id: str
    task_name: str
    task_id: str
    
    # Overall metrics
    accuracy: float
    stderr: float
    total_samples: int
    passed_samples: int
    failed_samples: int
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    
    # Timing
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    
    # Error breakdown
    error_types: dict[str, int] = field(default_factory=dict)
    
    # Per-task scores (question_id -> list of scores)
    per_question_scores: dict[str, list[str]] = field(default_factory=dict)
    
    # Per-question token counts
    per_question_tokens: dict[str, list[int]] = field(default_factory=dict)


@dataclass 
class SampleAnalysis:
    """Analysis of a single evaluation sample."""
    
    sample_id: str
    question_id: str
    category: str
    passed: bool
    score_value: str
    error_type: str | None = None
    error_message: str | None = None
    code_extracted: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


def categorize_error(explanation: str | None, question_id: str | None = None) -> str:
    """Categorize the type of error from score explanation.
    
    For NameError, distinguishes between:
    - 'No Output': The entry point function was not defined (model didn't output code)
    - 'NameError': A helper function was called without being defined (lazy coding)
    """
    if not explanation:
        return "NoErrorInfo"
    
    # Try to extract actual Python exception name
    match = re.search(
        r'\b([A-Z][a-zA-Z]*Error|[A-Z][a-zA-Z]*Exception|[A-Z][a-zA-Z]*Warning)\b',
        explanation
    )
    if match:
        error_type = match.group(1)
        
        # Special handling for NameError: check if it's the entry point
        if error_type == "NameError" and question_id:
            entry_points = load_entry_points()
            entry_point = entry_points.get(question_id, "")
            
            # Check if the undefined name is the entry point
            name_match = re.search(r"name ['\"]([^'\"]+)['\"] is not defined", explanation)
            if name_match:
                undefined_name = name_match.group(1)
                if undefined_name == entry_point:
                    return "No Output"  # Model didn't define the required function
        
        return error_type
    
    error_lower = explanation.lower()
    
    if "timeout" in error_lower:
        return "Timeout"
    elif "no code extracted" in error_lower:
        return "NoCodeExtracted"
    elif "assertion" in error_lower and ("failed" in error_lower or "error" in error_lower):
        return "AssertionError"
    elif "passed" in error_lower or "success" in error_lower:
        return "Success"
    
    return "Unknown"


def extract_missing_module(error: str | None) -> str | None:
    """Extract the missing module name from an import error message."""
    if not error:
        return None
    
    # Pattern: "No module named 'xxx'" or "No module named xxx"
    match = re.search(r"No module named ['\"]?([^'\">\s]+)['\"]?", error)
    if match:
        return match.group(1)
    
    # Pattern: "cannot import name 'xxx' from 'yyy'"
    match = re.search(
        r"cannot import name ['\"]?([^'\"]+)['\"]? from ['\"]?([^'\"]+)['\"]?",
        error
    )
    if match:
        return f"{match.group(2)}.{match.group(1)}"
    
    return None


def parse_datetime(dt_str: str) -> float:
    """Parse ISO datetime string to timestamp."""
    from datetime import datetime, timezone
    try:
        # Handle ISO format with timezone
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.timestamp()
    except (ValueError, AttributeError):
        return 0.0


def analyze_log(log: EvalLog) -> ModelMetrics:
    """Analyze a single evaluation log and extract all metrics."""
    
    # Extract model name (clean up the full path)
    model_id = log.eval.model
    model_name = model_id.split("/")[-1].replace(":nitro", "").replace("-", "_")
    
    # Get scores
    score_info = log.results.scores[0] if log.results.scores else None
    accuracy = 0.0
    stderr = 0.0
    
    if score_info:
        accuracy = score_info.metrics.get("accuracy", type("", (), {"value": 0})).value
        stderr = score_info.metrics.get("stderr", type("", (), {"value": 0})).value
    
    # Token usage from stats
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cache_read = 0
    cache_write = 0
    reasoning = 0
    
    if log.stats and log.stats.model_usage:
        for model, usage in log.stats.model_usage.items():
            input_tokens += usage.input_tokens or 0
            output_tokens += usage.output_tokens or 0
            total_tokens += usage.total_tokens or 0
            cache_read += usage.input_tokens_cache_read or 0
            cache_write += usage.input_tokens_cache_write or 0
            reasoning += usage.reasoning_tokens or 0
    
    # Timing
    started_at = log.stats.started_at if log.stats else ""
    completed_at = log.stats.completed_at if log.stats else ""
    
    duration = 0.0
    if started_at and completed_at:
        duration = parse_datetime(completed_at) - parse_datetime(started_at)
    
    # Analyze samples
    error_types: Counter[str] = Counter()
    per_question_scores: dict[str, list[str]] = defaultdict(list)
    per_question_tokens: dict[str, list[int]] = defaultdict(list)
    passed_count = 0
    failed_count = 0
    
    samples = log.samples or []
    
    for sample in samples:
        # Get score from the new 'scores' dict
        score = None
        if hasattr(sample, 'scores') and sample.scores:
            score = list(sample.scores.values())[0]  # Get first scorer's result
        elif hasattr(sample, 'score') and sample.score:
            score = sample.score
            
        score_value = score.value if score else "I"
        explanation = score.explanation if score else ""
        
        # Extract sample_id (used for entry point lookup)
        sample_id = sample.id or ""
        
        # Extract question_id from brackets [Q0] -> Q0
        bracket_match = re.search(r'\[([^\]]+)\]', sample_id)
        question_id = bracket_match.group(1) if bracket_match else sample_id
        
        # Track pass/fail
        is_passed = score_value == "C"  # C = Correct in this benchmark
        if is_passed:
            passed_count += 1
        else:
            failed_count += 1
            # Categorize error (passing sample_id to distinguish No Output vs NameError)
            error_type = categorize_error(explanation, sample_id)
            error_types[error_type] += 1
        
        # Extract category from sample ID
        # Format: "0_reliability_metrics_and_evaluation[Q0]" 
        #   -> question_id = "Q0", category = "reliability_metrics_and_evaluation"
        
        # Extract category (part between first _ and [)
        parts = sample_id.split("_", 1)
        if len(parts) > 1:
            category = parts[1].split("[")[0]
        else:
            category = "unknown"
        
        # Use category_questionid as the key for per-question tracking
        full_question_key = f"{category}[{question_id}]"
        per_question_scores[full_question_key].append(score_value)
        
        # Track tokens per sample
        sample_tokens = 0
        if hasattr(sample, 'model_usage') and sample.model_usage:
            for usage in sample.model_usage.values():
                sample_tokens += usage.total_tokens or 0
        per_question_tokens[full_question_key].append(sample_tokens)
    
    return ModelMetrics(
        model_name=model_name,
        model_id=model_id,
        task_name=log.eval.task,
        task_id=log.eval.task_id,
        accuracy=accuracy,
        stderr=stderr,
        total_samples=len(samples),
        passed_samples=passed_count,
        failed_samples=failed_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration,
        error_types=dict(error_types),
        per_question_scores=dict(per_question_scores),
        per_question_tokens=dict(per_question_tokens),
    )


def analyze_samples_detailed(log: EvalLog) -> list[SampleAnalysis]:
    """Get detailed analysis for each sample in a log."""
    results = []
    
    for sample in (log.samples or []):
        # Get score
        score = None
        if hasattr(sample, 'scores') and sample.scores:
            score = list(sample.scores.values())[0]
        elif hasattr(sample, 'score') and sample.score:
            score = sample.score
        
        score_value = score.value if score else "I"
        explanation = score.explanation if score else ""
        answer = score.answer if score else ""
        
        # Extract sample_id (used for entry point lookup)
        sample_id = sample.id or ""
        
        # Extract question_id from brackets [Q0] -> Q0
        bracket_match = re.search(r'\[([^\]]+)\]', sample_id)
        question_id = bracket_match.group(1) if bracket_match else sample_id
        
        is_passed = score_value == "C"
        error_type = None if is_passed else categorize_error(explanation, sample_id)
        
        # Extract category (part between first _ and [)
        parts = sample_id.split("_", 1)
        category = parts[1].split("[")[0] if len(parts) > 1 else "unknown"
        
        # Extract tokens
        input_tokens = 0
        output_tokens = 0
        if hasattr(sample, 'model_usage') and sample.model_usage:
            for usage in sample.model_usage.values():
                input_tokens += usage.input_tokens or 0
                output_tokens += usage.output_tokens or 0
        
        results.append(SampleAnalysis(
            sample_id=sample_id,
            question_id=question_id,
            category=category,
            passed=is_passed,
            score_value=score_value,
            error_type=error_type,
            error_message=explanation[:500] if explanation else None,
            code_extracted=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ))
    
    return results


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k estimator.
    
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


def compute_pass_at_k_per_question(
    per_question_scores: dict[str, list[str]],
    k_values: list[int] = [1, 5, 10]
) -> dict[str, dict[int, float]]:
    """Compute pass@k for each question."""
    results = {}
    
    for question_id, scores in per_question_scores.items():
        n = len(scores)
        c = sum(1 for s in scores if s == "C")
        
        results[question_id] = {}
        for k in k_values:
            if k <= n:
                results[question_id][k] = pass_at_k(n, c, k)
    
    return results


class LogAnalyzer:
    """Main analyzer class for processing multiple evaluation logs."""
    
    def __init__(self, results_dir: str | Path, use_cache: bool = True):
        self.results_dir = Path(results_dir)
        self.use_cache = use_cache
        self.cache_dir = self.results_dir / ".cache"
        self.logs: list[EvalLog] = []
        self.metrics: list[ModelMetrics] = []
        
    def load_logs(self) -> None:
        """Load all evaluation logs from the results directory."""
        log_files = list(self.results_dir.glob("*.eval"))
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Found {len(log_files)} evaluation logs in {self.results_dir}")
        
        for log_file in sorted(log_files):
            log = None
            cache_file = self.cache_dir / f"{log_file.name}.pkl" if self.use_cache else None
            
            # Try loading from cache
            if self.use_cache and cache_file.exists():
                # Check if cache is newer than source
                if cache_file.stat().st_mtime > log_file.stat().st_mtime:
                    try:
                        with open(cache_file, "rb") as f:
                            log = pickle.load(f)
                        print(f"  ✓ Loaded from cache: {log_file.name}")
                    except Exception as e:
                        print(f"  ! Cache error for {log_file.name}: {e}")
            
            if log is None:
                try:
                    log = read_eval_log(str(log_file))
                    if log.status == "success":
                        print(f"  ✓ Loaded: {log_file.name}")
                        # Save to cache
                        if self.use_cache:
                            try:
                                with open(cache_file, "wb") as f:
                                    pickle.dump(log, f)
                            except Exception as e:
                                print(f"  ! Failed to cache {log_file.name}: {e}")
                    else:
                        print(f"  ✗ Skipped (status={log.status}): {log_file.name}")
                        continue
                except Exception as e:
                    print(f"  ✗ Error loading {log_file.name}: {e}")
                    continue
            
            if log:
                self.logs.append(log)
    
    def analyze_all(self) -> list[ModelMetrics]:
        """Analyze all loaded logs and return metrics."""
        self.metrics = []
        
        for log in self.logs:
            metrics = analyze_log(log)
            self.metrics.append(metrics)
            print(f"  Analyzed: {metrics.model_name} - accuracy={metrics.accuracy:.4f}")
        
        return self.metrics
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get a pandas DataFrame with summary metrics for all models."""
        
        data = []
        for m in self.metrics:
            data.append({
                "model": m.model_name,
                "model_id": m.model_id,
                "accuracy": m.accuracy,
                "stderr": m.stderr,
                "total_samples": m.total_samples,
                "passed": m.passed_samples,
                "failed": m.failed_samples,
                "pass_rate": m.passed_samples / m.total_samples if m.total_samples > 0 else 0,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "total_tokens": m.total_tokens,
                "tokens_per_sample": m.total_tokens / m.total_samples if m.total_samples > 0 else 0,
                "duration_seconds": m.duration_seconds,
                "duration_minutes": m.duration_seconds / 60,
            })
        
        return pd.DataFrame(data)
    
    def get_error_breakdown_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame with error type breakdown per model."""
        import pandas as pd
        
        # Collect all error types
        all_error_types = set()
        for m in self.metrics:
            all_error_types.update(m.error_types.keys())
        
        data = []
        for m in self.metrics:
            row = {"model": m.model_name}
            for error_type in all_error_types:
                row[error_type] = m.error_types.get(error_type, 0)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_per_question_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame with per-question pass rates."""
        import pandas as pd
        
        # Collect all questions
        all_questions = set()
        for m in self.metrics:
            all_questions.update(m.per_question_scores.keys())
        
        data = []
        for m in self.metrics:
            for question_id in all_questions:
                scores = m.per_question_scores.get(question_id, [])
                if scores:
                    n = len(scores)
                    c = sum(1 for s in scores if s == "C")
                    data.append({
                        "model": m.model_name,
                        "question_id": question_id,
                        "total": n,
                        "passed": c,
                        "pass_rate": c / n if n > 0 else 0,
                        "pass_at_1": pass_at_k(n, c, 1) if n >= 1 else None,
                    })
        
        return pd.DataFrame(data)

    def get_samples_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame with detailed analysis for all samples across all logs."""
        import pandas as pd
        
        all_samples = []
        for log in self.logs:
            # Extract model name (clean up the full path)
            model_id = log.eval.model
            model_name = model_id.split("/")[-1].replace(":nitro", "").replace("-", "_")
            
            samples = analyze_samples_detailed(log)
            for s in samples:
                sample_dict = {
                    "model": model_name,
                    "sample_id": s.sample_id,
                    "question_id": s.question_id,
                    "category": s.category,
                    "passed": s.passed,
                    "score_value": s.score_value,
                    "error_type": s.error_type,
                    "error_message": s.error_message,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "total_tokens": s.input_tokens + s.output_tokens
                }
                all_samples.append(sample_dict)
        
        return pd.DataFrame(all_samples)
    
    def export_to_parquet(self, output_path: str | Path) -> None:
        """Export metrics to parquet format for use with inspect-viz."""
        import pandas as pd
        
        output_path = Path(output_path)
        
        # Create format expected by inspect-viz
        data = []
        for m in self.metrics:
            data.append({
                "task_name": m.task_name,
                "model": m.model_name,
                "model_organization_name": m.model_id.split("/")[0] if "/" in m.model_id else "unknown",
                "model_release_date": None,  # Could be populated from metadata
                "score_headline_value": m.accuracy,
                "score_headline_stderr": m.stderr,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "total_tokens": m.total_tokens,
                "duration_seconds": m.duration_seconds,
            })
        
        df = pd.DataFrame(data)
        df.to_parquet(output_path)
        print(f"Exported to {output_path}")


def main():
    """Example usage of the log analyzer."""
    analyzer = LogAnalyzer("results")
    analyzer.load_logs()
    analyzer.analyze_all()
    
    # Print summary
    df = analyzer.get_summary_dataframe()
    print("\n=== Model Summary ===")
    print(df.to_string())
    
    # Export for inspect-viz
    analyzer.export_to_parquet("analysis_results.parquet")


if __name__ == "__main__":
    main()
