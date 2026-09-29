#!/usr/bin/env python3
"""Convert attention benchmark JSON results into an analysis-ready CSV."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RESULT_COLUMNS = [
    "backend",
    "scenario",
    "pass",
    "batch_size",
    "seq_len",
    "num_heads",
    "head_dim",
    "dtype",
    "window_size",
    "api",
    "skip_check",
    "status",
    "error_message",
    "median_ms",
    "mean_ms",
    "std_ms",
    "min_ms",
    "max_ms",
    "p95_ms",
    "peak_memory_mb",
    "tflops",
]

CONTEXT_COLUMNS = [
    "benchmark_timestamp",
    "python_version",
    "torch_version",
    "cuda_available",
    "cuda_version",
    "device_name",
    "device_count",
    "compute_capability",
    "total_memory_gb",
    "suite",
    "warmup",
    "repeat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten benchmark JSON results into a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("bench_results.json"),
        help="Benchmark JSON file to summarize.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("summary.csv"),
        help="Output CSV file.",
    )
    return parser.parse_args()


def csv_value(value: Any) -> Any:
    """Represent structured values as compact JSON for a single CSV cell."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def flatten_results(data: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    metadata = data.get("metadata", {})
    config = data.get("config", {})
    context = {
        "benchmark_timestamp": metadata.get("timestamp"),
        "python_version": metadata.get("python_version"),
        "torch_version": metadata.get("torch_version"),
        "cuda_available": metadata.get("cuda_available"),
        "cuda_version": metadata.get("cuda_version"),
        "device_name": metadata.get("device_name"),
        "device_count": metadata.get("device_count"),
        "compute_capability": csv_value(metadata.get("compute_capability")),
        "total_memory_gb": metadata.get("total_memory_gb"),
        "suite": config.get("suite"),
        "warmup": config.get("warmup"),
        "repeat": config.get("repeat"),
    }

    rows = []
    for result in data.get("results", []):
        row = {**context}
        row.update({column: csv_value(result.get(column)) for column in RESULT_COLUMNS})
        rows.append(row)
    return CONTEXT_COLUMNS + RESULT_COLUMNS, rows


def write_summary(input_path: Path, output_path: Path) -> int:
    with input_path.open(encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Input must be a benchmark JSON object with a results list")

    columns, rows = flatten_results(data)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    args = parse_args()
    row_count = write_summary(args.input, args.output)
    print(f"Wrote {row_count} rows to {args.output}")


if __name__ == "__main__":
    main()