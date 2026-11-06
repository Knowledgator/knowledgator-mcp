"""Markdown report generation for training metrics and dataset analytics."""

from typing import Any

from knowledgator_mcp.utils import build_frontend_url


def _format_loss_chart(loss_metrics: list[dict[str, Any]]) -> str:
    if not loss_metrics:
        return "*No loss data available.*"

    values = [m.get("value", 0) for m in loss_metrics]
    steps = list(range(1, len(values) + 1))

    min_v = min(values)
    max_v = max(values)
    padding = (max_v - min_v) * 0.1 or 0.0001
    y_min = max(0, min_v - padding)
    y_max = max_v + padding

    steps_str = ", ".join(str(s) for s in steps)
    values_str = ", ".join(f"{v:.6f}" for v in values)

    return f"""```mermaid
xychart-beta
    title "Training Loss"
    x-axis "Step" [{steps_str}]
    y-axis "Loss" {y_min:.6f} --> {y_max:.6f}
    line [{values_str}]
```"""


def _format_eval_metrics_table(metrics: dict[str, float]) -> str:
    if not metrics:
        return "*No evaluation metrics available.*"

    rows = ["| Metric | Score |", "|--------|-------|"]
    for name, value in metrics.items():
        pct = value * 100 if value <= 1 else value
        label = name.replace("_", " ").title()
        rows.append(f"| {label} | {pct:.1f}% |")

    return "\n".join(rows)


def _format_label_distribution(labels: list[dict[str, Any]]) -> str:
    if not labels:
        return ""

    total = sum(item.get("count", 0) for item in labels)
    if total == 0:
        return ""

    rows = ["| Label | Count | % |", "|-------|-------|---|"]
    for item in labels:
        name = item.get("name", item.get("label", "Unknown"))
        count = item.get("count", 0)
        pct = (count / total) * 100
        rows.append(f"| {name} | {count} | {pct:.1f}% |")

    return "### Label Distribution\n\n" + "\n".join(rows)


def _format_quality_radar(quality: dict[str, float], overall_score: float | None = None) -> str:
    dims = ["balance", "quality", "diversity", "complexity", "consistency"]
    available = {k: quality.get(k, 0) for k in dims if k in quality}

    if not available:
        return ""

    values = [available.get(d, 0) * 100 if available.get(d, 0) <= 1 else available.get(d, 0) for d in dims if d in available]
    labels = [d.title() for d in dims if d in available]

    if overall_score is not None:
        overall = overall_score
    else:
        overall = sum(values) / len(values) if values else 0
    overall_pct = overall * 100 if overall <= 1 else overall

    labels_str = ", ".join(labels)
    values_str = ", ".join(f"{v:.0f}" for v in values)

    return f"""### Quality Profile (Overall: {overall_pct:.0f}%)

```mermaid
xychart-beta
    title "Quality Dimensions"
    x-axis [{labels_str}]
    y-axis "Score" 0 --> 100
    bar [{values_str}]
```"""


def generate_training_metrics_report(
    training_id: str,
    training_info: dict[str, Any] | None = None,
    loss_metrics: list[dict[str, Any]] | None = None,
    eval_metrics: dict[str, float] | None = None,
) -> str:
    parts = []

    status = training_info.get("status", "unknown") if training_info else "unknown"
    name = training_info.get("name", "") if training_info else ""
    url = build_frontend_url("trainings", training_id)

    header = f"## Training: {training_id}"
    if name:
        header += f" ({name})"
    parts.append(header)
    parts.append(f"\n[View in Platform]({url})")
    parts.append(f"\n**Status:** {status}\n")

    parts.append("### Training Loss\n")
    parts.append(_format_loss_chart(loss_metrics or []))

    parts.append("\n### Evaluation Metrics\n")
    parts.append(_format_eval_metrics_table(eval_metrics or {}))

    return "\n".join(parts)


def generate_dataset_analytics_report(
    dataset_id: str,
    statistics: dict[str, Any],
) -> str:
    parts = []
    url = build_frontend_url("datasets", dataset_id)

    parts.append(f"## Dataset Analytics: {dataset_id}")
    parts.append(f"\n[View in Platform]({url})\n")

    data = statistics.get("data", statistics)

    quality_profile = data.get("quality_profile", {})
    quality_dims = quality_profile.get("dimensions", quality_profile)
    overall_score = quality_profile.get("overall_score")
    if quality_dims:
        parts.append(_format_quality_radar(quality_dims, overall_score))

    label_dist = data.get("label_distribution", {})
    labels = label_dist.get("distribution", label_dist if isinstance(label_dist, list) else [])
    if labels:
        parts.append("\n" + _format_label_distribution(labels))

    text_stats = data.get("text_statistics", {})
    if text_stats:
        parts.append("\n### Text Statistics\n")
        rows = ["| Metric | Min | Max | Mean | Median |", "|--------|-----|-----|------|--------|"]
        for metric, vals in text_stats.items():
            if isinstance(vals, dict):
                rows.append(f"| {metric.title()} | {vals.get('min', 0):.0f} | {vals.get('max', 0):.0f} | {vals.get('mean', 0):.1f} | {vals.get('median', 0):.0f} |")
        parts.append("\n".join(rows))

    key_metrics = data.get("key_metrics", {})
    if key_metrics:
        parts.append("\n### Key Metrics\n")
        parts.append(f"- **Examples:** {key_metrics.get('examples', 'N/A')}")
        parts.append(f"- **File Size:** {key_metrics.get('file_size', 'N/A')}")
        parts.append(f"- **Avg Tokens:** {key_metrics.get('avg_tokens', 'N/A')}")
        parts.append(f"- **Total Tokens:** {key_metrics.get('total_tokens', 'N/A')}")

    return "\n".join(parts)
