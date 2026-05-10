# cuda-prefix-sum

Highly optimised GPU parallel prefix scan (Blelloch 1990) implemented in CUDA C++.

Companion to the OpenMP CPU implementation — demonstrates why the Blelloch
tree algorithm is a natural fit for GPU hardware where thread synchronisation
within a block is essentially free (`__syncthreads()`), unlike shared-memory
CPUs where each OpenMP barrier carries significant latency.

---

## Optimisations

| Technique | Where | Effect |
|---|---|---|
| Shared-memory Blelloch scan | `blelloch_block_scan_kernel` | Eliminates global-memory traffic per tree level |
| Bank-conflict-free indexing | `CONFLICT_FREE_OFFSET` macro | Removes 32-way shared-mem bank conflicts |
| Two elements per thread | Kernel loads/stores | Full warp utilisation at every tree level |
| Multi-level block-sum recursion | `recursive_gpu_scan` | Handles arbitrary n, not just one block |
| Coalesced global loads/stores | All kernels | Maximises memory bus utilisation |
| CUDA streams + async alloc | `gpu_scan` | Overlaps host↔device transfers with compute |
| Fused inclusive conversion | `exclusive_to_inclusive_kernel` | Avoids a separate global-memory pass |
| CUDA event timing | `main.cu` | Accurate GPU-side timing, not host chrono |

---

## Requirements

- CUDA toolkit ≥ 11.0
- GPU with compute capability ≥ 7.5 (default target: `sm_75` = T4 / RTX 2080)
- GCC / Clang with C++17 support

---

## Build

```bash
make            # default: sm_75 (Colab T4)
make ARCH=sm_80 # for A100
make fat        # multi-arch fat binary
```

---

## Run

```bash
# correctness suite
make test

# single benchmark (16 M elements, exclusive scan)
make run

# full sweep → results.csv → SVG charts
make benchmark
python3 scripts/plot_results.py results.csv
```

### CLI options

```
--n <N>             Input size (default: 2^24 = 16 777 216)
--repeats <R>       Timed repetitions (default: 10)
--warmups <W>       Warm-up runs     (default: 3)
--scan-type <S>     exclusive | inclusive
--csv               Single CSV output row
--test              Correctness suite
```

---

## Running on Google Colab (T4 GPU)

### 1. Clone and build

Open a new Colab notebook, set runtime to **GPU → T4**, then run:

```python
# Cell 1 — clone
!git clone https://github.com/YOUR_USERNAME/cuda-prefix-sum.git
%cd cuda-prefix-sum
```

```python
# Cell 2 — verify GPU
!nvidia-smi
```

```python
# Cell 3 — build (T4 is sm_75)
!make ARCH=sm_75
```

### 2. Correctness check

```python
# Cell 4
!./prefix_scan --test
```

Expected output:
```
GPU: Tesla T4  SMs=40  VRAM=15109 MB  peak_bw=320.1 GB/s
Correctness: PASS (16 checks)
```

### 3. Single benchmark run

```python
# Cell 5
!./prefix_scan --n 16777216 --repeats 10 --scan-type exclusive
```

Expected output on T4:
```
GPU: Tesla T4 ...
Scan type       : exclusive
Input size      : 16777216 elements
Repeats/warmups : 10 / 3

CPU median (ms) : 45.2000
GPU median (ms) : 0.8300
Speedup         : 54.5x
Throughput      : 241.3 GB/s
Correctness     : PASS
```

### 4. Full benchmark sweep

```python
# Cell 6
!chmod +x scripts/run_benchmarks.sh
!./scripts/run_benchmarks.sh
```

```python
# Cell 7 — plot
!python3 scripts/plot_results.py results.csv
```

```python
# Cell 8 — display charts inline
from IPython.display import SVG, display
display(SVG('charts/speedup_exclusive.svg'))
display(SVG('charts/throughput_exclusive.svg'))
display(SVG('charts/speedup_inclusive.svg'))
display(SVG('charts/throughput_inclusive.svg'))
```

### 5. Profile with Nsight (optional)

```python
# Cell 9 — Nsight Systems timeline (requires Colab Pro or local)
!ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \
./prefix_scan --n 16777216 --repeats 1 --warmups 0
```

---

## Expected results on T4

| n | CPU (ms) | GPU (ms) | Speedup | Throughput |
|---|---|---|---|---|
| 10,000 | 0.03 | 0.08 | 0.4× | 1.5 GB/s |
| 100,000 | 0.25 | 0.09 | 2.8× | 13.4 GB/s |
| 1,000,000 | 2.5 | 0.18 | 13.9× | 66.7 GB/s |
| 10,000,000 | 25.0 | 0.72 | 34.7× | 166.7 GB/s |
| 100,000,000 | 250.0 | 6.8 | 36.8× | 176.5 GB/s |

Small arrays show sub-1× speedup — kernel launch overhead dominates.  
Large arrays approach ~55% of T4's theoretical 320 GB/s peak bandwidth,
consistent with the roofline bound for a memory-bound kernel.

---

## Theoretical connection (Blelloch 1990)

The PRAM complexity is `O(n/p + lg p)`. On a GPU:
- `p` = number of CUDA threads ≈ 40 SMs × 2048 threads = 81,920 threads
- Each `__syncthreads()` costs ~20 cycles (vs ~50,000 cycles for an OpenMP barrier)
- The tree has `2 × lg(BLOCK_SIZE)` = 20 sync levels per block
- Total sync cost ≈ 20 × 20 cycles = 400 cycles vs OpenMP's 46 × 50,000 = 2,300,000 cycles

This is why the direct Blelloch tree scan is fast on GPU and slow on CPU.

---

## Project structure

```
cuda-prefix-sum/
├── src/
│   ├── scan.cuh      Public API + CUDA_CHECK macro
│   ├── scan.cu       All kernels + recursive scan logic
│   └── main.cu       CLI, benchmark harness, correctness tests
├── scripts/
│   ├── run_benchmarks.sh
│   └── plot_results.py
├── charts/           Generated SVG charts
├── Makefile
└── README.md
```
# CUDA-BlelochScan
