# attnbench: Scaled Attention Kernel Benchmark Suite

A high-performance benchmark suite and unified interface for scaled dot-product attention kernels across modern GPU architectures, scales, and execution scenarios.

---

## Benchmark Hardware & Environment

All measurements in this report were performed on the following system configuration:

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (1 GPU)
- **Architecture / Compute Capability**: Blackwell `sm_120` (CC 12.0)
- **Total VRAM**: 94.96 GB
- **CUDA Version**: 13.2 (`cuda-toolkit=13.2.2`)
- **PyTorch**: 2.13.0+cu132
- **Python**: 3.12.14
- **Evaluated Test Cases**: 3,680 total benchmark rows recorded in [`summary.csv`](summary.csv), generated from [`bench_results.json`](bench_results.json)

---

## Supported Attention Backends & Status

| Backend | Name / Hub Source | Supported Features | Blackwell (CC 12.0) Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| `sdpa` | PyTorch `torch.nn.functional.scaled_dot_product_attention` | Non-Causal, Causal, GQA/MQA, Dropout | **PASS** (480), **UNSUPPORTED** (80) | The unsupported rows are sliding-window cases. |
| `flash` | Dao-AILab `flash-attn` (v2.x) | Non-Causal, Causal, Sliding Window, Dropout | **PASS** (640) | Full pass across every tested scenario and execution pass. |
| `hffa2` | Hugging Face Kernel Hub `kernels-community/flash-attn2` | Non-Causal, Causal, Sliding Window, Dropout | **PASS** (640) | Full pass across every tested scenario and execution pass. |
| `hffa3` | Hugging Face Kernel Hub `kernels-community/flash-attn3` | FlashAttention-3 | **ERROR** (640) | Requires Hopper compute capability 9.x; the test GPU is Blackwell 12.0. |
| `hffa4` | Hugging Face Kernel Hub `kernels-community/flash-attn4` | FlashAttention-4 (CuTe DSL) | **PASS** (640) | Fully operational in this run, including sliding-window and backward cases. |
| `xformers` | Meta `xformers.ops.memory_efficient_attention` | Memory-efficient Cutlass attention | **ERROR** (480), **UNSUPPORTED** (80) | The installed package lacks `xformers.ops.memory_efficient_attention` / `mslk`. |

---

## CSV-derived Benchmark Results

The current `sweep` contains 3,680 rows covering `B={1,2}`, `L={512, 1024, 2048, 4096, 8192}`, `H={8,16}`, `D={64,128}`, `float16` and `bfloat16`, four scenarios, and forward or forward-plus-backward execution. The tables below report medians pooled across batch size, heads, head dimension, and dtype; use `summary.csv` for the individual measurements.

### Backend status

| Backend | Successful | Unsupported | Errors | Result |
| :--- | ---: | ---: | ---: | :--- |
| `sdpa` | 480 | 80 | 0 | Sliding-window attention is unsupported. |
| `flash` | 640 | 0 | 0 | Fully operational. |
| `hffa2` | 640 | 0 | 0 | Fully operational. |
| `hffa3` | 0 | 0 | 640 | Requires Hopper compute capability 9.x. |
| `hffa4` | 640 | 0 | 0 | Fully operational, including sliding-window cases. |
| `xformers` | 0 | 80 | 480 | Attention operator is unavailable because the `mslk` runtime is missing. |

### Median latency by scenario

| Scenario | Pass | SDPA | Flash | HFFA2 | HFFA3 | HFFA4 | xFormers |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Non-causal | `fwd` | 0.0955 ms | 0.1061 ms | 0.1136 ms | n/a | 0.1071 ms | n/a |
| Non-causal | `fwd_bwd` | 0.3825 ms | 0.5302 ms | 0.5511 ms | n/a | 0.4642 ms | n/a |
| Causal | `fwd` | 0.08145 ms | 0.0881 ms | 0.0954 ms | n/a | 0.1015 ms | n/a |
| Causal | `fwd_bwd` | 0.3484 ms | 0.4974 ms | 0.5426 ms | n/a | **0.4354 ms** | n/a |
| Sliding window | `fwd` | n/a | **0.07745 ms** | 0.09115 ms | n/a | 0.1163 ms | n/a |
| Sliding window | `fwd_bwd` | n/a | **0.4543 ms** | 0.4799 ms | n/a | 0.4880 ms | n/a |
| Module fast path | `fwd` | 0.0961 ms | 0.1061 ms | 0.1145 ms | n/a | 0.1080 ms | n/a |
| Module fast path | `fwd_bwd` | 0.4001 ms | 0.5223 ms | 0.5596 ms | n/a | **0.4684 ms** | n/a |

### Median throughput by scenario

| Scenario | Pass | SDPA | Flash | HFFA2 | HFFA3 | HFFA4 | xFormers |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Non-causal | `fwd` | **236.6** | 223.2 | 206.4 | n/a | 211.7 | n/a |
| Non-causal | `fwd_bwd` | **239.1** | 186.1 | 175.3 | n/a | 216.0 | n/a |
| Causal | `fwd` | **150.7** | 132.7 | 123.5 | n/a | 122.7 | n/a |
| Causal | `fwd_bwd` | **159.8** | 113.0 | 99.45 | n/a | 125.2 | n/a |
| Sliding window | `fwd` | n/a | **299.9** | 253.6 | n/a | 202.9 | n/a |
| Sliding window | `fwd_bwd` | n/a | 224.2 | **229.9** | n/a | 211.5 | n/a |
| Module fast path | `fwd` | **237.0** | 220.3 | 205.1 | n/a | 210.3 | n/a |
| Module fast path | `fwd_bwd` | **242.3** | 181.8 | 164.7 | n/a | 221.4 | n/a |

## Visualizations

The benchmark summary is also available as high-quality, vector PDF figures. The latency, throughput, and memory plots use separate panels for each attention scenario and execution pass, while preserving a consistent color for every backend.

- [Kernel speed](figures/kernel_speed.pdf): median latency versus sequence length for all kernels.
- [Kernel throughput](figures/kernel_throughput.pdf): effective TFLOP/s versus sequence length.
- [Kernel memory](figures/kernel_memory.pdf): peak GPU memory versus sequence length.
- [Backend status](figures/backend_status.pdf): successful, unsupported, and failed benchmark rows by scenario.
- [Precision comparison](figures/precision_comparison.pdf): `float16` versus `bfloat16` at `L = 8192` for non-causal attention.

Regenerate the figures from the CSV summary with:

```bash
python draw.py summary.csv --output-dir figures
```

---

## Architectural Findings & Analysis

1. **Current Dense-Attention Leader**:
   - Across the pooled CSV measurements, SDPA has the lowest median latency for non-causal and causal attention and the highest median throughput for those dense scenarios.
   - The median non-causal forward latency is 0.0955 ms for SDPA, 0.1061 ms for `flash`, 0.1136 ms for `hffa2`, and 0.1071 ms for `hffa4`.
2. **Sliding-Window Attention**:
   - SDPA and xFormers do not provide a usable sliding-window result in this run.
   - `flash` is fastest for sliding-window forward and forward-plus-backward execution, with median latencies of 0.07745 ms and 0.4543 ms respectively.
3. **FlashAttention-3 Architecture Locking**:
   - `hffa3` reports errors for all 640 rows because FlashAttention-3 requires Hopper compute capability 9.x, while this GPU is Blackwell compute capability 12.0.
   - [functional.py](functional.py) dynamically queries device compute capability and safely intercepts unsupported architectures.
4. **FlashAttention-4 Availability**:
   - `hffa4` completes all 640 tested rows, including causal, sliding-window, module fast-path, and backward cases.
   - It is the fastest measured backend for causal and module-fast-path forward-plus-backward medians in this sweep, at 0.4354 ms and 0.4684 ms respectively.
5. **xFormers Runtime Dependency**:
   - xFormers reports 480 errors because `xformers.ops.memory_efficient_attention` is unavailable, likely due to the missing `mslk` runtime, and 80 unsupported sliding-window rows.

---

## Quickstart & Usage

### 1. Functional Attention API

```python
import torch
import functional

# Setup inputs (B, L, H, D)
q = torch.randn(2, 4096, 32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 4096, 32, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 4096, 32, 128, device="cuda", dtype=torch.bfloat16)

# Automatic backend resolution
out = functional.attention(q, k, v)

# Explicit backend selection
out_flash = functional.attention(q, k, v, backend="flash")
out_hffa2 = functional.attention(q, k, v, backend="hffa2")

# Sliding window local attention (left=512, right=512)
out_window = functional.attention(q, k, v, backend="flash", window_size=(512, 512))

# Module wrapper with pre-bound fast-path dispatch
attn_block = functional.Attention(backend="flash", causal=True, skip_check=True)
out_block = attn_block(q, k, v)
```

### 2. Running Benchmarks

```bash
# Run a quick smoke benchmark across all installed backends
python3 bench.py --quick

# Run a custom scale sweep
python3 bench.py \
  --batch-sizes 1 2 4 \
  --seq-lens 2048 4096 8192 16384 \
  --heads 16 32 \
  --head-dims 64 128 256 \
  --dtypes bfloat16 float16 \
  --scenarios non_causal causal sliding_window module_fast_path \
  --passes fwd fwd_bwd \
  --warmup 3 \
  --repeat 7 \
  --output bench_results.json
```

---

## Environment Setup & Installation Guide

```bash
# Create conda environment
conda create -c conda-forge -c nvidia -n attnbench python=3.12 cuda-toolkit=13.2.2
conda activate attnbench

# Install PyTorch with CUDA 13.2
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
pip install ninja wheel packaging

# Setup CUDA build environment
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch; print(';'.join(sorted(list({f'{v[0]}.{v[1]}' for v in [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]}))))")

# Install FlashAttention-2
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Install Hugging Face Kernel Hub
pip install kernels
pip install apache-tvm-ffi
pip install nvidia-cutlass-dsl
pip install torch-c-dlpack-ext  # compatibility package for torch <= 2.9
```