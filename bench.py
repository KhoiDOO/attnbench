#!/usr/bin/env python3
"""Comprehensive Attention Benchmark Suite.

Benchmarks all installed attention kernel backends across different scales
(batch sizes, sequence lengths, head dimensions, attention heads), data types (fp16, bf16),
and execution scenarios (non-causal, causal, sliding-window, module fast-path vs functional).
Measures latency (median/mean/std/min/max/p95), computational throughput (TFLOP/s),
and peak memory usage, saving results to a structured JSON file.
"""

import argparse
import datetime
import gc
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

import functional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark attention backends across scales and scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="bench_results.json",
        help="Path to save the JSON benchmark results.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Specific backends to benchmark. Default: all available backends.",
    )
    parser.add_argument(
        "--batch-sizes",
        "-b",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Batch sizes to evaluate.",
    )
    parser.add_argument(
        "--seq-lens",
        "-l",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
        help="Sequence lengths to evaluate.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=[8, 16],
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--head-dims",
        "-d",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Head dimensions.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["float16", "bfloat16"],
        choices=["float16", "bfloat16", "float32"],
        help="Data types to test.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["non_causal", "causal", "sliding_window", "module_fast_path"],
        choices=["non_causal", "causal", "sliding_window", "module_fast_path"],
        help="Attention scenarios to benchmark.",
    )
    parser.add_argument(
        "--passes",
        nargs="+",
        default=["fwd", "fwd_bwd"],
        choices=["fwd", "fwd_bwd"],
        help="Execution passes to measure: fwd (inference) and/or fwd_bwd (training).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--repeat",
        "-r",
        type=int,
        default=30,
        help="Number of timed benchmark repetitions.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test run with reduced scales and repetitions.",
    )
    parser.add_argument(
        "--suite",
        choices=["sweep", "report"],
        default="sweep",
        help="Benchmark suite: 'sweep' for Cartesian grid, 'report' for comprehensive scaling comparison reported in README.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append/update results in existing JSON file instead of overwriting.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print Markdown comparison tables matching README format after benchmarking.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[dtype_str.lower()]


def compute_flops(b: int, l: int, h: int, d: int, causal: bool = False, exec_pass: str = "fwd") -> float:
    """Computes theoretical floating point operations for attention.

    - QK^T: 2 * B * H * L * L * D
    - Softmax * V: 2 * B * H * L * L * D
    Total forward non-causal: 4 * B * H * L^2 * D.
    Causal attention masks upper triangle, halving arithmetic operations.
    Backward pass evaluates dQ, dK, dV, requiring approx 2.5x forward FLOPs.
    """
    fwd_factor = 2.0 if causal else 4.0
    fwd_flops = fwd_factor * b * h * (l**2) * d
    if exec_pass == "fwd_bwd":
        return fwd_flops + 2.5 * fwd_flops
    elif exec_pass == "bwd":
        return 2.5 * fwd_flops
    return fwd_flops


def get_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            }
        )
    info["available_backends"] = functional.available_backends()
    return info


def benchmark_call(
    call_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exec_pass: str = "fwd",
    warmup: int = 10,
    repeat: int = 30,
) -> Dict[str, Any]:
    """Measures execution latency using hardware CUDA events, peak VRAM, and stats."""
    # Warmup
    for _ in range(warmup):
        if exec_pass == "fwd":
            with torch.no_grad():
                out = call_fn(q, k, v)
        else:
            q_b = q.detach().clone().requires_grad_(True)
            k_b = k.detach().clone().requires_grad_(True)
            v_b = v.detach().clone().requires_grad_(True)
            out = call_fn(q_b, k_b, v_b)
            loss = out.sum()
            loss.backward()
    torch.cuda.synchronize()

    # Profile memory
    torch.cuda.reset_peak_memory_stats()

    latencies_ms: List[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(repeat):
        if exec_pass == "fwd":
            with torch.no_grad():
                start_event.record()
                out = call_fn(q, k, v)
                end_event.record()
        elif exec_pass == "bwd":
            q_b = q.detach().clone().requires_grad_(True)
            k_b = k.detach().clone().requires_grad_(True)
            v_b = v.detach().clone().requires_grad_(True)
            out = call_fn(q_b, k_b, v_b)
            loss = out.sum()
            start_event.record()
            loss.backward()
            end_event.record()
        elif exec_pass == "fwd_bwd":
            q_b = q.detach().clone().requires_grad_(True)
            k_b = k.detach().clone().requires_grad_(True)
            v_b = v.detach().clone().requires_grad_(True)
            start_event.record()
            out = call_fn(q_b, k_b, v_b)
            loss = out.sum()
            loss.backward()
            end_event.record()

        torch.cuda.synchronize()
        latencies_ms.append(start_event.elapsed_time(end_event))

    latencies_ms.sort()
    n = len(latencies_ms)
    median_ms = latencies_ms[n // 2] if n % 2 != 0 else (latencies_ms[n // 2 - 1] + latencies_ms[n // 2]) / 2.0
    mean_ms = sum(latencies_ms) / n
    variance = sum((x - mean_ms) ** 2 for x in latencies_ms) / n
    std_ms = variance**0.5
    min_ms = latencies_ms[0]
    max_ms = latencies_ms[-1]
    p95_idx = int(0.95 * n)
    p95_ms = latencies_ms[min(p95_idx, n - 1)]

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    return {
        "median_ms": round(median_ms, 4),
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "min_ms": round(min_ms, 4),
        "max_ms": round(max_ms, 4),
        "p95_ms": round(p95_ms, 4),
        "peak_memory_mb": round(peak_memory_mb, 2),
    }


def print_table_header():
    header = (
        f"{'Backend':<10} | {'Scenario':<15} | {'Pass':<7} | {'Shape (B,L,H,D)':<20} | {'Dtype':<8} | "
        f"{'Latency (ms)':<12} | {'TFLOP/s':<9} | {'VRAM (MB)':<9} | {'Status':<10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)


def print_table_row(
    backend: str,
    scenario: str,
    exec_pass: str,
    shape_str: str,
    dtype_str: str,
    latency_str: str,
    tflops_str: str,
    vram_str: str,
    status_str: str,
):
    print(
        f"{backend:<10} | {scenario:<15} | {exec_pass:<7} | {shape_str:<20} | {dtype_str:<8} | "
        f"{latency_str:<12} | {tflops_str:<9} | {vram_str:<9} | {status_str:<10}"
    )


def generate_report_markdown(records: List[Dict[str, Any]]) -> str:
    """Generates structured GitHub-style Markdown tables matching README report format."""
    idx = {}
    for r in records:
        key = (
            r.get("backend"),
            r.get("scenario"),
            r.get("pass"),
            r.get("batch_size"),
            r.get("seq_len"),
            r.get("num_heads"),
            r.get("head_dim"),
            r.get("dtype"),
        )
        idx[key] = r

    def get_val(backend, scenario, exec_pass, b, l, h, d, dtype):
        return idx.get((backend, scenario, exec_pass, b, l, h, d, dtype))

    def fmt_cell(r):
        if not r:
            return "-", "-", "-"
        if r.get("status") == "unsupported":
            return "*unsupported*", "-", "-"
        if r.get("status") != "success":
            return f"*{r.get('status')}*", "-", "-"
        ms = f"{r['median_ms']:.3f} ms"
        tflops = f"{r['tflops']:.1f}"
        vram = f"{r['peak_memory_mb']:.1f} MB"
        if r.get("peak_memory_mb", 0) >= 1024:
            vram = f"{r['peak_memory_mb'] / 1024:.2f} GB"
        return ms, tflops, vram

    md = []

    # 1. Long-Context Sequence Length Scaling (B=2, H=32, D=128, bfloat16, non_causal)
    md.append("### 1. Long-Context Sequence Length Scaling ($L = 2048 \\to 16384$)")
    md.append("*Configuration: Batch Size $B=2$, Heads $H=32$, Head Dimension $D=128$, `bfloat16`, Scenario: `non_causal`*\n")
    md.append("| Sequence Length ($L$) | Pass | SDPA Latency | SDPA TFLOP/s | Flash Latency | Flash TFLOP/s | HFFA2 Latency | HFFA2 TFLOP/s | HFFA4 Latency | HFFA4 TFLOP/s | Peak VRAM |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for l in [2048, 4096, 8192, 16384]:
        for ep in ["fwd", "fwd_bwd"]:
            rs = {b: get_val(b, "non_causal", ep, 2, l, 32, 128, "bfloat16") for b in ["sdpa", "flash", "hffa2", "hffa4"]}
            cells = {b: fmt_cell(rs[b]) for b in ["sdpa", "flash", "hffa2", "hffa4"]}
            lat_floats = [(b, rs[b]["median_ms"]) for b in rs if rs[b] and rs[b].get("status") == "success"]
            min_lat = min((f for _, f in lat_floats), default=None)
            lat_strs = {}
            for b in ["sdpa", "flash", "hffa2", "hffa4"]:
                ms = cells[b][0]
                if min_lat is not None and rs[b] and rs[b].get("status") == "success" and abs(rs[b]["median_ms"] - min_lat) < 1e-4:
                    lat_strs[b] = f"**{ms}**"
                else:
                    lat_strs[b] = ms
            vrams = [rs[b].get("peak_memory_mb", 0) for b in rs if rs[b]]
            max_vram = max(vrams) if vrams else 0
            vram_str = f"{max_vram:.1f} MB" if max_vram < 1024 else f"{max_vram / 1024:.2f} GB"
            md.append(f"| **{l:,}** | `{ep}` | {lat_strs['sdpa']} | {cells['sdpa'][1]} | {lat_strs['flash']} | {cells['flash'][1]} | {lat_strs['hffa2']} | {cells['hffa2'][1]} | {lat_strs['hffa4']} | {cells['hffa4'][1]} | {vram_str} |")

    # 2. Large Head Dimension Scaling (B=2, L=8192, H=32, bfloat16, non_causal)
    md.append("\n---\n")
    md.append("### 2. Large Head Dimension Scaling ($D = 64 \\to 256$)")
    md.append("*Configuration: Batch Size $B=2$, Sequence Length $L=8192$, Heads $H=32$, `bfloat16`, Scenario: `non_causal`*\n")
    md.append("| Head Dimension ($D$) | Pass | SDPA Latency | SDPA TFLOP/s | Flash Latency | Flash TFLOP/s | HFFA2 Latency | HFFA2 TFLOP/s | HFFA4 Latency | HFFA4 TFLOP/s | Peak VRAM |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for d in [64, 128, 256]:
        for ep in ["fwd", "fwd_bwd"]:
            rs = {b: get_val(b, "non_causal", ep, 2, 8192, 32, d, "bfloat16") for b in ["sdpa", "flash", "hffa2", "hffa4"]}
            cells = {b: fmt_cell(rs[b]) for b in ["sdpa", "flash", "hffa2", "hffa4"]}
            lat_floats = [(b, rs[b]["median_ms"]) for b in rs if rs[b] and rs[b].get("status") == "success"]
            min_lat = min((f for _, f in lat_floats), default=None)
            lat_strs = {}
            for b in ["sdpa", "flash", "hffa2", "hffa4"]:
                ms = cells[b][0]
                if min_lat is not None and rs[b] and rs[b].get("status") == "success" and abs(rs[b]["median_ms"] - min_lat) < 1e-4:
                    lat_strs[b] = f"**{ms}**"
                else:
                    lat_strs[b] = ms
            vrams = [rs[b].get("peak_memory_mb", 0) for b in rs if rs[b]]
            max_vram = max(vrams) if vrams else 0
            vram_str = f"{max_vram:.1f} MB" if max_vram < 1024 else f"{max_vram / 1024:.2f} GB"
            md.append(f"| **{d}** | `{ep}` | {lat_strs['sdpa']} | {cells['sdpa'][1]} | {lat_strs['flash']} | {cells['flash'][1]} | {lat_strs['hffa2']} | {cells['hffa2'][1]} | {lat_strs['hffa4']} | {cells['hffa4'][1]} | {vram_str} |")

    # 3. Multi-Batch Scaling (B = 1 -> 4)
    md.append("\n---\n")
    md.append("### 3. Multi-Batch Scaling ($B = 1 \\to 4$)")
    md.append("*Configuration: Sequence Length $L=8192$, Heads $H=32$, Head Dimension $D=128$, `bfloat16`, Scenario: `non_causal`*\n")
    md.append("| Batch Size ($B$) | Pass | SDPA Latency | SDPA TFLOP/s | Flash Latency | Flash TFLOP/s | HFFA2 Latency | HFFA2 TFLOP/s | HFFA4 Latency | HFFA4 TFLOP/s | Peak VRAM |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for b in [1, 2, 4]:
        for ep in ["fwd", "fwd_bwd"]:
            rs = {bk: get_val(bk, "non_causal", ep, b, 8192, 32, 128, "bfloat16") for bk in ["sdpa", "flash", "hffa2", "hffa4"]}
            cells = {bk: fmt_cell(rs[bk]) for bk in ["sdpa", "flash", "hffa2", "hffa4"]}
            lat_floats = [(bk, rs[bk]["median_ms"]) for bk in rs if rs[bk] and rs[bk].get("status") == "success"]
            min_lat = min((f for _, f in lat_floats), default=None)
            lat_strs = {}
            for bk in ["sdpa", "flash", "hffa2", "hffa4"]:
                ms = cells[bk][0]
                if min_lat is not None and rs[bk] and rs[bk].get("status") == "success" and abs(rs[bk]["median_ms"] - min_lat) < 1e-4:
                    lat_strs[bk] = f"**{ms}**"
                else:
                    lat_strs[bk] = ms
            vrams = [rs[bk].get("peak_memory_mb", 0) for bk in rs if rs[bk]]
            max_vram = max(vrams) if vrams else 0
            vram_str = f"{max_vram:.1f} MB" if max_vram < 1024 else f"{max_vram / 1024:.2f} GB"
            md.append(f"| **{b}** | `{ep}` | {lat_strs['sdpa']} | {cells['sdpa'][1]} | {lat_strs['flash']} | {cells['flash'][1]} | {lat_strs['hffa2']} | {cells['hffa2'][1]} | {lat_strs['hffa4']} | {cells['hffa4'][1]} | {vram_str} |")

    # 4. Extreme Scale Stress Test (B=4, L=16384, H=32, D=256)
    md.append("\n---\n")
    md.append("### 4. Extreme Scale Stress Test ($B=4, L=16384, H=32, D=256$)")
    md.append("*Evaluation of maximal scale: $4 \\times 16,384 \\times 32 \\times 256$ (134 million elements per Q/K/V tensor)*\n")
    md.append("| Scenario | Pass | SDPA Latency | Flash Latency | HFFA2 Latency | HFFA4 Latency | Best Effective TFLOP/s | Peak VRAM |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    scenarios_meta = [
        ("non_causal", "**Non-Causal**"),
        ("causal", "**Causal**"),
        ("sliding_window", "**Sliding Window** ($w=512$)"),
    ]
    for sc, sc_title in scenarios_meta:
        for ep in ["fwd", "fwd_bwd"]:
            rs = {bk: get_val(bk, sc, ep, 4, 16384, 32, 256, "bfloat16") for bk in ["sdpa", "flash", "hffa2", "hffa4"]}
            cells = {bk: fmt_cell(rs[bk]) for bk in ["sdpa", "flash", "hffa2", "hffa4"]}
            lat_floats = [(bk, rs[bk]["median_ms"]) for bk in rs if rs[bk] and rs[bk].get("status") == "success"]
            min_lat = min((f for _, f in lat_floats), default=None)
            lat_strs = {}
            for bk in ["sdpa", "flash", "hffa2", "hffa4"]:
                ms = cells[bk][0]
                if min_lat is not None and rs[bk] and rs[bk].get("status") == "success" and abs(rs[bk]["median_ms"] - min_lat) < 1e-4:
                    lat_strs[bk] = f"**{ms}**"
                else:
                    lat_strs[bk] = ms
            tflops_floats = [rs[bk]["tflops"] for bk in rs if rs[bk] and rs[bk].get("tflops") is not None and rs[bk].get("status") == "success"]
            best_tflops = f"**{max(tflops_floats):.1f} TFLOP/s**" if tflops_floats else "-"
            vrams = [rs[bk].get("peak_memory_mb", 0) for bk in rs if rs[bk]]
            max_vram = max(vrams) if vrams else 0
            vram_str = f"{max_vram:.1f} MB" if max_vram < 1024 else f"{max_vram / 1024:.2f} GB"
            md.append(f"| {sc_title} | `{ep}` | {lat_strs['sdpa']} | {lat_strs['flash']} | {lat_strs['hffa2']} | {lat_strs['hffa4']} | {best_tflops} | {vram_str} |")

    # 5. Attention Scenario Breakdown (B=2, L=8192, H=32, D=128)
    md.append("\n---\n")
    md.append("### 5. Attention Scenario Breakdown ($B=2, L=8192, H=32, D=128$)")
    md.append("\n| Scenario | Pass | SDPA Latency | SDPA TFLOP/s | Flash Latency | Flash TFLOP/s | HFFA2 Latency | HFFA2 TFLOP/s | HFFA4 Latency | HFFA4 TFLOP/s |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    sc_list = [
        ("non_causal", "**Non-Causal**"),
        ("causal", "**Causal**"),
        ("sliding_window", "**Sliding Window** ($w=512$)"),
        ("module_fast_path", "**Module Fast-Path**"),
    ]
    for sc, sc_title in sc_list:
        for ep in ["fwd", "fwd_bwd"]:
            rs = {bk: get_val(bk, sc, ep, 2, 8192, 32, 128, "bfloat16") for bk in ["sdpa", "flash", "hffa2", "hffa4"]}
            cells = {bk: fmt_cell(rs[bk]) for bk in ["sdpa", "flash", "hffa2", "hffa4"]}
            lat_floats = [(bk, rs[bk]["median_ms"]) for bk in rs if rs[bk] and rs[bk].get("status") == "success"]
            min_lat = min((f for _, f in lat_floats), default=None)
            lat_strs = {}
            for bk in ["sdpa", "flash", "hffa2", "hffa4"]:
                ms = cells[bk][0]
                if min_lat is not None and rs[bk] and rs[bk].get("status") == "success" and abs(rs[bk]["median_ms"] - min_lat) < 1e-4:
                    lat_strs[bk] = f"**{ms}**"
                else:
                    lat_strs[bk] = ms
            md.append(f"| {sc_title} | `{ep}` | {lat_strs['sdpa']} | {cells['sdpa'][1]} | {lat_strs['flash']} | {cells['flash'][1]} | {lat_strs['hffa2']} | {cells['hffa2'][1]} | {lat_strs['hffa4']} | {cells['hffa4'][1]} |")

    # 6. Precision Comparison: float16 vs bfloat16
    md.append("\n---\n")
    md.append("### 6. Precision Comparison: `float16` vs `bfloat16`")
    md.append("*Configuration: Batch Size $B=2$, Sequence Length $L=8192$, Heads $H=32$, Head Dimension $D=128$, `non_causal`*\n")
    md.append("| Precision | Pass | SDPA Latency | Flash Latency | HFFA2 Latency | HFFA4 Latency | Best TFLOP/s |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for dt, dt_title in [("float16", "**`float16`**"), ("bfloat16", "**`bfloat16`**")]:
        for ep in ["fwd", "fwd_bwd"]:
            rs = {bk: get_val(bk, "non_causal", ep, 2, 8192, 32, 128, dt) for bk in ["sdpa", "flash", "hffa2", "hffa4"]}
            cells = {bk: fmt_cell(rs[bk]) for bk in ["sdpa", "flash", "hffa2", "hffa4"]}
            lat_floats = [(bk, rs[bk]["median_ms"]) for bk in rs if rs[bk] and rs[bk].get("status") == "success"]
            min_lat = min((f for _, f in lat_floats), default=None)
            lat_strs = {}
            for bk in ["sdpa", "flash", "hffa2", "hffa4"]:
                ms = cells[bk][0]
                if min_lat is not None and rs[bk] and rs[bk].get("status") == "success" and abs(rs[bk]["median_ms"] - min_lat) < 1e-4:
                    lat_strs[bk] = f"**{ms}**"
                else:
                    lat_strs[bk] = ms
            tflops_floats = [rs[bk]["tflops"] for bk in rs if rs[bk] and rs[bk].get("tflops") is not None and rs[bk].get("status") == "success"]
            best_tflops = f"**{max(tflops_floats):.1f} TFLOP/s**" if tflops_floats else "-"
            md.append(f"| {dt_title} | `{ep}` | {lat_strs['sdpa']} | {lat_strs['flash']} | {lat_strs['hffa2']} | {lat_strs['hffa4']} | {best_tflops} |")

    return "\n".join(md)


def run_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark attention kernels.")

    # Apply quick settings if requested
    if args.quick:
        args.batch_sizes = [1]
        args.seq_lens = [512, 2048]
        args.head_dims = [64]
        args.heads = [8]
        args.dtypes = ["float16"]
        args.warmup = 5
        args.repeat = 10

    available = functional.available_backends()
    backends = args.backends if args.backends else available
    backends = [b for b in backends if b in available]

    print("\n" + "=" * 90)
    print("  ATTNBENCH: HIGH-PRECISION ATTENTION SPEED & SCALE BENCHMARK")
    print("=" * 90)
    print(f"Device:               {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability:   {torch.cuda.get_device_capability(0)}")
    print(f"Active Backends:      {', '.join(backends)}")
    print(f"Suite:                {args.suite}")
    print(f"Passes:               {', '.join(args.passes)}")
    print(f"Warmup / Repeats:     {args.warmup} / {args.repeat}")
    print(f"Output File:          {args.output}")
    print("=" * 90 + "\n")

    # Warmup / preload all backends upfront
    functional.preload_all()

    system_info = get_system_info()
    records: List[Dict[str, Any]] = []

    print_table_header()

    if args.suite == "report":
        # Targeted configurations covering all 6 report sections in README
        configs = [
            # 1. Sequence Length Scaling (B=2, H=32, D=128, bfloat16, non_causal)
            (2, 2048, 32, 128, "bfloat16", "non_causal"),
            (2, 4096, 32, 128, "bfloat16", "non_causal"),
            (2, 8192, 32, 128, "bfloat16", "non_causal"),
            (2, 16384, 32, 128, "bfloat16", "non_causal"),
            # 2. Large Head Dimension Scaling (B=2, L=8192, H=32, bfloat16, non_causal)
            (2, 8192, 32, 64, "bfloat16", "non_causal"),
            (2, 8192, 32, 256, "bfloat16", "non_causal"),
            # 3. Multi-Batch Scaling (L=8192, H=32, D=128, bfloat16, non_causal)
            (1, 8192, 32, 128, "bfloat16", "non_causal"),
            (4, 8192, 32, 128, "bfloat16", "non_causal"),
            # 4. Extreme Scale Stress Test (B=4, L=16384, H=32, D=256, bfloat16)
            (4, 16384, 32, 256, "bfloat16", "non_causal"),
            (4, 16384, 32, 256, "bfloat16", "causal"),
            (4, 16384, 32, 256, "bfloat16", "sliding_window"),
            # 5. Attention Scenario Breakdown (B=2, L=8192, H=32, D=128, bfloat16)
            (2, 8192, 32, 128, "bfloat16", "causal"),
            (2, 8192, 32, 128, "bfloat16", "sliding_window"),
            (2, 8192, 32, 128, "bfloat16", "module_fast_path"),
            # 6. Precision Comparison (B=2, L=8192, H=32, D=128, float16)
            (2, 8192, 32, 128, "float16", "non_causal"),
            (2, 8192, 32, 128, "float16", "causal"),
            (2, 8192, 32, 128, "float16", "sliding_window"),
            (2, 8192, 32, 128, "float16", "module_fast_path"),
        ]
        seen = set()
        eval_tuples = []
        for c in configs:
            if c not in seen:
                seen.add(c)
                eval_tuples.append(c)
    else:
        eval_tuples = [
            (b, l, h, d, dtype_str, scenario)
            for b in args.batch_sizes
            for l in args.seq_lens
            for h in args.heads
            for d in args.head_dims
            for dtype_str in args.dtypes
            for scenario in args.scenarios
        ]

    for (b, l, h, d, dtype_str, scenario) in eval_tuples:
        shape_str = f"({b},{l},{h},{d})"
        dtype = get_torch_dtype(dtype_str)
        causal = scenario == "causal"
        window_size = (min(512, l // 2), min(512, l // 2)) if scenario == "sliding_window" else None

        for backend in backends:
            # Check backend sliding window support
            if scenario == "sliding_window" and backend in ("sdpa", "xformers", "xformer"):
                print_table_row(
                    backend, scenario, "-", shape_str, dtype_str, "-", "-", "-", "unsupported"
                )
                records.append(
                    {
                        "backend": backend,
                        "scenario": scenario,
                        "pass": "-",
                        "batch_size": b,
                        "seq_len": l,
                        "num_heads": h,
                        "head_dim": d,
                        "dtype": dtype_str,
                        "window_size": window_size,
                        "api": "functional",
                        "status": "unsupported",
                    }
                )
                continue

            for exec_pass in args.passes:
                flops = compute_flops(b, l, h, d, causal=causal, exec_pass=exec_pass)

                # Instantiate test inputs
                try:
                    q = torch.randn(b, l, h, d, device="cuda", dtype=dtype)
                    k = torch.randn(b, l, h, d, device="cuda", dtype=dtype)
                    v = torch.randn(b, l, h, d, device="cuda", dtype=dtype)
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    print_table_row(
                        backend, scenario, exec_pass, shape_str, dtype_str, "-", "-", "-", "OOM"
                    )
                    records.append(
                        {
                            "backend": backend,
                            "scenario": scenario,
                            "pass": exec_pass,
                            "batch_size": b,
                            "seq_len": l,
                            "num_heads": h,
                            "head_dim": d,
                            "dtype": dtype_str,
                            "status": "oom",
                        }
                    )
                    continue

                # Setup call function
                try:
                    if scenario == "module_fast_path":
                        mod = functional.Attention(
                            backend=backend,
                            causal=causal,
                            window_size=window_size,
                            skip_check=True,
                        )
                        call_fn = lambda q_in, k_in, v_in: mod(q_in, k_in, v_in)
                        api_type = "module"
                    else:
                        call_fn = lambda q_in, k_in, v_in: functional.attention(
                            q_in,
                            k_in,
                            v_in,
                            backend=backend,
                            causal=causal,
                            window_size=window_size,
                            skip_check=True,
                        )
                        api_type = "functional"

                    # Benchmark
                    stats = benchmark_call(
                        call_fn,
                        q,
                        k,
                        v,
                        exec_pass=exec_pass,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )

                    # Compute TFLOP/s
                    tflops = round((flops / (stats["median_ms"] * 1e-3)) / 1e12, 2)
                    stats["tflops"] = tflops

                    rec = {
                        "backend": backend,
                        "scenario": scenario,
                        "pass": exec_pass,
                        "batch_size": b,
                        "seq_len": l,
                        "num_heads": h,
                        "head_dim": d,
                        "dtype": dtype_str,
                        "window_size": window_size,
                        "api": api_type,
                        "skip_check": True,
                        "status": "success",
                        **stats,
                    }
                    records.append(rec)

                    print_table_row(
                        backend,
                        scenario,
                        exec_pass,
                        shape_str,
                        dtype_str,
                        f"{stats['median_ms']:.3f}",
                        f"{tflops:.1f}",
                        f"{stats['peak_memory_mb']:.1f}",
                        "PASS",
                    )

                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    print_table_row(
                        backend, scenario, exec_pass, shape_str, dtype_str, "-", "-", "-", "OOM"
                    )
                    records.append(
                        {
                            "backend": backend,
                            "scenario": scenario,
                            "pass": exec_pass,
                            "batch_size": b,
                            "seq_len": l,
                            "num_heads": h,
                            "head_dim": d,
                            "dtype": dtype_str,
                            "status": "oom",
                        }
                    )
                except Exception as exc:
                    err_msg = str(exc).split("\n")[0][:25]
                    print_table_row(
                        backend,
                        scenario,
                        exec_pass,
                        shape_str,
                        dtype_str,
                        "-",
                        "-",
                        "-",
                        f"ERR:{err_msg}",
                    )
                    records.append(
                        {
                            "backend": backend,
                            "scenario": scenario,
                            "pass": exec_pass,
                            "batch_size": b,
                            "seq_len": l,
                            "num_heads": h,
                            "head_dim": d,
                            "dtype": dtype_str,
                            "status": "error",
                            "error_message": str(exc),
                        }
                    )
                finally:
                    del q, k, v
                    gc.collect()
                    torch.cuda.empty_cache()

    output_path = os.path.abspath(args.output)
    final_records = records
    if args.append and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            rec_map = {}
            for r in existing_data.get("results", []):
                key = (
                    r.get("backend"),
                    r.get("scenario"),
                    r.get("pass"),
                    r.get("batch_size"),
                    r.get("seq_len"),
                    r.get("num_heads"),
                    r.get("head_dim"),
                    r.get("dtype"),
                )
                rec_map[key] = r
            for r in records:
                key = (
                    r.get("backend"),
                    r.get("scenario"),
                    r.get("pass"),
                    r.get("batch_size"),
                    r.get("seq_len"),
                    r.get("num_heads"),
                    r.get("head_dim"),
                    r.get("dtype"),
                )
                rec_map[key] = r
            final_records = list(rec_map.values())
        except Exception as exc:
            print(f"Warning: could not merge with existing results ({exc}). Overwriting.")

    output_data = {
        "metadata": system_info,
        "config": {
            "suite": args.suite,
            "batch_sizes": args.batch_sizes,
            "seq_lens": args.seq_lens,
            "heads": args.heads,
            "head_dims": args.head_dims,
            "dtypes": args.dtypes,
            "scenarios": args.scenarios,
            "passes": args.passes,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "results": final_records,
    }

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 90)
    print(f"Benchmark completed successfully! Total evaluated cases: {len(final_records)}")
    print(f"Results saved to: {output_path}")
    print("=" * 90 + "\n")

    if args.markdown:
        md_report = generate_report_markdown(final_records)
        print("\n" + "=" * 90)
        print("  GENERATED MARKDOWN REPORT TABLES")
        print("=" * 90 + "\n")
        print(md_report)
        print("\n" + "=" * 90 + "\n")

    return output_data


if __name__ == "__main__":
    cli_args = parse_args()
    run_benchmarks(cli_args)
