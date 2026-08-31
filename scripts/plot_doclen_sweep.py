#!/usr/bin/env python3
import json
import sys
from pathlib import Path

CASE_LABELS = {
    "cascade_tile": "cascade",
    "hot_cold_sharing": "hot/cold",
    "branching_prefix_sharing": "branching",
}
COLORS = {
    "baseline": "#5577aa",
    "v1": "#d28a43",
    "green": "#5f9f73",
    "purple": "#9b6fb0",
    "grid": "#d8d8d8",
    "text": "#222222",
}


def stat(case, mode, key):
    value = case["modes"][mode][key]
    if isinstance(value, dict):
        return value["mean"]
    return value


def fmt(value):
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.2f}"


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_grouped_bars(labels, series, title, ylabel, path):
    width, height = 1180, 470
    left, right, top, bottom = 78, 24, 48, 94
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_values = [v for _, values, _ in series for v in values]
    ymax = max(all_values) * 1.12 if all_values else 1.0
    ymax = ymax or 1.0
    group_w = plot_w / len(labels)
    bar_w = min(28, group_w * 0.34)
    gap = bar_w * 0.12
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-family="Arial" font-size="18" fill="{COLORS["text"]}">{esc(title)}</text>',
        f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13" fill="{COLORS["text"]}">{esc(ylabel)}</text>',
    ]
    for i in range(5):
        y = top + plot_h - plot_h * i / 4
        val = ymax * i / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="{COLORS["text"]}">{fmt(val)}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>')
    n = len(series)
    for gi, label in enumerate(labels):
        cx = left + group_w * (gi + 0.5)
        for si, (_, values, color) in enumerate(series):
            x = cx - (n * bar_w + (n - 1) * gap) / 2 + si * (bar_w + gap)
            h = plot_h * values[gi] / ymax
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        first, second = label.split("\n", 1)
        parts.append(f'<text x="{cx:.1f}" y="{top+plot_h+20}" text-anchor="middle" font-family="Arial" font-size="11" fill="{COLORS["text"]}">{esc(first)}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{top+plot_h+36}" text-anchor="middle" font-family="Arial" font-size="11" fill="{COLORS["text"]}">{esc(second)}</text>')
    lx = left + 8
    for i, (name, _, color) in enumerate(series):
        x = lx + i * 150
        parts.append(f'<rect x="{x}" y="{height-28}" width="13" height="13" fill="{color}"/>')
        parts.append(f'<text x="{x+18}" y="{height-17}" font-family="Arial" font-size="12" fill="{COLORS["text"]}">{esc(name)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n")


def labels_for(summary):
    labels = []
    case_refs = []
    for doc_key in sorted(summary, key=lambda x: int(x.split("_")[1])):
        doc_len = int(doc_key.split("_")[1])
        for case_name in CASE_LABELS:
            if case_name not in summary[doc_key]:
                continue
            labels.append(f"{doc_len}\n{CASE_LABELS[case_name]}")
            case_refs.append(summary[doc_key][case_name])
    return labels, case_refs


def comparison_values(cases, key):
    return [case["comparison"][key] for case in cases]


def metric_values(cases, mode, key):
    return [stat(case, mode, key) for case in cases]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: plot_doclen_sweep.py EXP_ROOT")
    root = Path(sys.argv[1])
    summary = json.loads((root / "summary.json").read_text())
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    labels, cases = labels_for(summary)

    grouped = [
        ("ttft_latency_median", "seconds", "Median TTFT", "ttft_median_baseline_vs_v1.svg"),
        ("request_latency_median", "seconds", "Median request latency", "request_latency_median_baseline_vs_v1.svg"),
        ("prefill_step_time_sec", "seconds", "Measured prefill step time", "prefill_time_baseline_vs_v1.svg"),
        ("decode_step_time_sec", "seconds", "Measured decode step time", "decode_time_baseline_vs_v1.svg"),
        ("queueing_latency_avg", "seconds", "Queueing latency avg", "queueing_avg_baseline_vs_v1.svg"),
        ("queueing_latency_max", "seconds", "Queueing latency max", "queueing_max_baseline_vs_v1.svg"),
    ]
    for key, ylabel, title, filename in grouped:
        svg_grouped_bars(
            labels,
            [
                ("baseline", metric_values(cases, "baseline", key), COLORS["baseline"]),
                ("V1", metric_values(cases, "v1", key), COLORS["v1"]),
            ],
            title,
            ylabel,
            fig_dir / filename,
        )

    svg_grouped_bars(
        labels,
        [
            ("prefill", comparison_values(cases, "prefill_time_speedup_v1_over_baseline"), COLORS["baseline"]),
            ("TTFT med", comparison_values(cases, "ttft_median_speedup_v1_over_baseline"), COLORS["v1"]),
            ("req med", comparison_values(cases, "request_latency_median_speedup_v1_over_baseline"), COLORS["green"]),
            ("queue avg", comparison_values(cases, "queueing_avg_speedup_v1_over_baseline"), COLORS["purple"]),
        ],
        "V1 over baseline speedups",
        "speedup, higher is better",
        fig_dir / "doclen_speedups.svg",
    )

    svg_grouped_bars(
        labels,
        [
            ("baseline recompute", metric_values(cases, "baseline", "document_recomputed_tokens_est"), COLORS["baseline"]),
            ("V1 CPU restore", metric_values(cases, "v1", "cpu_prefix_cache_restored_token_count"), COLORS["v1"]),
            ("GPU reuse", metric_values(cases, "baseline", "prefix_cache_reused_token_count"), COLORS["green"]),
        ],
        "Token accounting",
        "tokens",
        fig_dir / "tokens_recompute_restore_reuse.svg",
    )


if __name__ == "__main__":
    main()
