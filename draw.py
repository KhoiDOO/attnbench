#!/usr/bin/env python3
"""Create high-quality, multi-panel PDF plots from benchmark summary CSV data."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCENARIOS = ["non_causal", "causal", "sliding_window", "module_fast_path"]
PASSES = ["fwd", "fwd_bwd"]
BACKENDS = ["sdpa", "flash", "xformers", "hffa2", "hffa3", "hffa4"]
STATUSES = ["success", "unsupported", "error"]
COLORS = {
    "sdpa": "#0072B2",
    "flash": "#D55E00",
    "xformers": "#009E73",
    "hffa2": "#CC79A7",
    "hffa3": "#E69F00",
    "hffa4": "#56B4E9",
}
STATUS_COLORS = {"success": "#009E73", "unsupported": "#E69F00", "error": "#D55E00"}
SCENARIO_LABELS = {
    "non_causal": "Non-causal",
    "causal": "Causal",
    "sliding_window": "Sliding window",
    "module_fast_path": "Module fast path",
}
PASS_LABELS = {"fwd": "Forward", "fwd_bwd": "Forward + backward"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw multi-panel PDF figures from benchmark summary CSV data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("summary.csv"))
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def successful_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "success"]


def aggregate_by_sequence(rows: Iterable[dict[str, str]], metric: str) -> dict[tuple[str, str, str], list[tuple[int, float]]]:
    values: defaultdict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in successful_rows(rows):
        if not row.get(metric):
            continue
        key = (row["scenario"], row["pass"], row["backend"], int(row["seq_len"]))
        values[key].append(float(row[metric]))

    aggregated: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for (scenario, execution_pass, backend, seq_len), samples in values.items():
        aggregated[(scenario, execution_pass, backend)].append((seq_len, median(samples)))
    for points in aggregated.values():
        points.sort()
    return aggregated


def kernel_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COLORS[backend], marker="o", linewidth=2, label=backend)
        for backend in BACKENDS
    ]


def save_metric_figure(rows: list[dict[str, str]], metric: str, ylabel: str, output_path: Path) -> None:
    aggregated = aggregate_by_sequence(rows, metric)
    figure, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    axes = axes.reshape(4, 2)

    for row_index, scenario in enumerate(SCENARIOS):
        for column_index, execution_pass in enumerate(PASSES):
            axis = axes[row_index, column_index]
            for backend in BACKENDS:
                points = aggregated.get((scenario, execution_pass, backend), [])
                if points:
                    x_values, y_values = zip(*points)
                    axis.plot(
                        x_values,
                        y_values,
                        color=COLORS[backend],
                        marker="o",
                        linewidth=2,
                        markersize=5,
                        label=backend,
                    )
            axis.set_xscale("log", base=2)
            axis.set_xticks([512, 1024, 2048, 4096, 8192])
            axis.set_xticklabels(["512", "1K", "2K", "4K", "8K"])
            axis.set_title(f"{SCENARIO_LABELS[scenario]} | {PASS_LABELS[execution_pass]}")
            axis.set_xlabel("Sequence length, L")
            axis.set_ylabel(ylabel)
            axis.grid(True, which="major", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        f"Attention kernel {ylabel.lower()}\nMedian across batch size, heads, head dimension, and dtype",
        fontsize=17,
        fontweight="bold",
    )
    figure.legend(handles=kernel_legend(), loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def save_status_figure(rows: list[dict[str, str]], output_path: Path) -> None:
    counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["scenario"], row["backend"], row["status"])] += 1

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    for axis, scenario in zip(axes.flat, SCENARIOS):
        positions = list(range(len(BACKENDS)))
        bottoms = [0] * len(BACKENDS)
        for status in STATUSES:
            heights = [counts[(scenario, backend, status)] for backend in BACKENDS]
            axis.bar(
                positions,
                heights,
                bottom=bottoms,
                color=STATUS_COLORS[status],
                label=status,
                width=0.72,
            )
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
        axis.set_title(SCENARIO_LABELS[scenario])
        axis.set_xticks(positions, BACKENDS)
        axis.set_ylabel("Number of benchmark rows")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("Backend availability by attention scenario", fontsize=17, fontweight="bold")
    figure.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def save_precision_figure(rows: list[dict[str, str]], output_path: Path) -> None:
    filtered = [
        row
        for row in successful_rows(rows)
        if row["scenario"] == "non_causal" and row["seq_len"] == "8192"
    ]
    values: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in filtered:
        values[(row["pass"], row["backend"], row["dtype"])].append(float(row["median_ms"]))

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x_positions = list(range(len(BACKENDS)))
    bar_width = 0.36
    for axis, execution_pass in zip(axes, PASSES):
        for offset, dtype in enumerate(["float16", "bfloat16"]):
            heights = [
                median(values[(execution_pass, backend, dtype)])
                if values[(execution_pass, backend, dtype)]
                else 0
                for backend in BACKENDS
            ]
            axis.bar(
                [position + (offset - 0.5) * bar_width for position in x_positions],
                heights,
                width=bar_width,
                label=dtype,
                color="#4C78A8" if dtype == "float16" else "#F58518",
            )
        axis.set_title(PASS_LABELS[execution_pass])
        axis.set_xticks(x_positions, BACKENDS)
        axis.set_xlabel("Kernel backend")
        axis.set_ylabel("Median latency (ms)")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("Precision comparison at L = 8,192, non-causal", fontsize=17, fontweight="bold")
    figure.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0, 0.08, 1, 0.93))
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_metric_figure(rows, "median_ms", "Median latency (ms)", args.output_dir / "kernel_speed.pdf")
    save_metric_figure(rows, "tflops", "Throughput (TFLOP/s)", args.output_dir / "kernel_throughput.pdf")
    save_metric_figure(rows, "peak_memory_mb", "Peak memory (MB)", args.output_dir / "kernel_memory.pdf")
    save_status_figure(rows, args.output_dir / "backend_status.pdf")
    save_precision_figure(rows, args.output_dir / "precision_comparison.pdf")
    print(f"Wrote 5 PDF figures to {args.output_dir}")


if __name__ == "__main__":
    main()