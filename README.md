# cuda-prefix-sum-v2

**Highly optimised GPU prefix scan — single-pass decoupled look-back.**  
Research implementation based on Blelloch (1990) + Merrill & Garland (2016).

---

## Why v1 Was Slow — The Diagnosis

v1 used **host-side recursion** between kernel launches:

```
launch kernel (scan each block)        ← GPU kernel 1
CPU: cudaMalloc block sums             ← HOST ROUND-TRIP  ~1.3 ms overhead
launch kernel (scan block sums)        ← GPU kernel 2
CPU: cudaMalloc again...               ← another round-trip
launch kernel (add offsets)            ← GPU kernel 3
... repeat log(n) times
```

For n = 100M: log₂(100M/2048) ≈ 16 → **~20 ms just in host overhead**.  
The T4 peak bandwidth is 320 GB/s.  
100M × 4 bytes = 400 MB → **theoretical minimum: 1.25 ms**.  
v1 measured: 207 ms. The extra 206 ms is pure overhead.

---

## The Fix — Decoupled Look-Back (Merrill & Garland, 2016)

**Paper:** "Single-pass Parallel Prefix Scan with Decoupled Look-back"  
This is the algorithm used inside **NVIDIA CUB** and **Thrust** today.

### Algorithm per tile (GPU thread block):

```
1. Compute LOCAL prefix sum (Blelloch in shared memory, bank-conflict free)
2. Publish LOCAL AGGREGATE to global status array → mark PARTIAL
3. LOOK-BACK: scan predecessor tiles' status entries (in device memory):
     - Found STATUS_PREFIX tile? → add its value and STOP
     - Found STATUS_PARTIAL tile? → accumulate and keep looking
4. Once own prefix is known → add to all local elements
5. Publish own INCLUSIVE PREFIX → mark PREFIX (so successors can proceed)
```

**Result:**  
- **ONE kernel launch** for ANY array size  
- **Zero host round-trips** between levels  
- Tiles communicate through device memory — no CPU involvement  
- O(1) amortised look-back per element

### Optimisations on top of Blelloch:

| Optimisation | Description | Benefit |
|---|---|---|
| Single-pass | One kernel launch, no host recursion | Eliminates ~1.3 ms × log(n) overhead |
| Warp shuffles | `__shfl_up_sync` for intra-warp scan | No smem bank conflicts for first 5 levels |
| 8 items/thread | Each thread processes ITEMS_PT=8 items | Better arithmetic intensity, fewer __syncthreads |
| `__ldg()` loads | Read-only cache hint for input array | L2 cache reuse on repeated benchmark runs |
| Persistent workspace | Status/counter arrays allocated once, reused | No per-call cudaMalloc latency |
| Fused inclusive | Inclusive result computed inside same kernel | No second pass / extra kernel launch |

---

## Three Algorithms Compared

| Algorithm | Kernel launches (n=100M) | Expected speedup vs CPU |
|---|---|---|
| `blelloch` (v1) | ~16 launches + host mallocs | 0.4× (slower than CPU!) |
| `single_pass` (v2) | **1 launch** | 10–30× |
| `cub` (reference) | 1 launch (production CUB) | 15–40× (ceiling) |

---

## Quick Start on Colab (T4)

### Step 1 — Get a T4 GPU
`Runtime → Change runtime type → T4 GPU`

### Step 2 — Clone and build
```python
!git clone https://github.com/YOUR_USERNAME/cuda-prefix-sum-v2.git
%cd cuda-prefix-sum-v2
!make ARCH=sm_75    # T4 is sm_75
```

### Step 3 — Correctness check
```python
!./prefix_scan_v2 --test
```
Expected: `Correctness: PASS (N checks, 0 failures)`

### Step 4 — Compare all algorithms at 16M elements
```python
!./prefix_scan_v2 --algo all --n 16777216 --scan-type exclusive
```

### Step 5 — Full benchmark sweep
```python
!chmod +x scripts/run_benchmarks.sh
!./scripts/run_benchmarks.sh
```

### Step 6 — Plot results
```python
!pip install matplotlib -q
!python3 scripts/plot_results.py results_v2.csv

from IPython.display import SVG, display
display(SVG('charts/speedup_exclusive.svg'))
display(SVG('charts/throughput_exclusive.svg'))
display(SVG('charts/improvement_exclusive.svg'))
```

---

## Expected Results on T4

| n | Blelloch (v1) | Single-Pass (v2) | CUB | v2 improvement |
|---|---|---|---|---|
| 10K | ~1.3 ms (0.003×) | ~0.05 ms | ~0.05 ms | **26×** |
| 100K | ~1.4 ms (0.04×) | ~0.08 ms | ~0.07 ms | **18×** |
| 1M | ~3.3 ms (0.2×) | ~0.3 ms | ~0.25 ms | **11×** |
| 10M | ~22 ms (0.5×) | ~1.5 ms (7×) | ~1.2 ms | **15×** |
| 100M | ~207 ms (0.4×) | ~8 ms (10×) | ~6 ms | **26×** |

---

## Research Context

### Theoretical basis (Blelloch 1990)
- Optimal parallel scan: O(n/p + log p) on PRAM
- PRAM assumes **zero communication cost** (barriers are free)
- Real hardware: each barrier costs ~100 µs on CPU (OpenMP), ~1.3 ms host round-trip on GPU (naive CUDA)

### Gap analysis
```
v1 gap:  kernel launch overhead dominates → O(log n) × 1.3 ms
v2 fix:  decoupled look-back → O(1) launches → gap reduced to memory bandwidth
CUB gap: remaining overhead vs peak bandwidth (pipeline latency, address generation)
```

### Why this matters
Both our OpenMP (v0) and v1 CUDA implementations were slower than sequential.  
Both confirm the same thesis: **the PRAM zero-cost synchronization assumption
fails in real hardware** — on CPU via barrier latency, on GPU via kernel launch overhead.

The single-pass algorithm (v2) eliminates the GPU-specific bottleneck by
keeping all inter-tile communication on the device. This is why production
libraries (CUB, Thrust, rocPRIM) all use this approach.

---

## File Structure

```
cuda-prefix-sum-v2/
├── src/
│   ├── scan.cuh    ← API header, three Algorithm variants
│   ├── scan.cu     ← All implementations (Blelloch, SinglePass, CUB)
│   └── main.cu     ← CLI harness, benchmarking, correctness tests
├── scripts/
│   ├── run_benchmarks.sh   ← Full sweep → results_v2.csv
│   └── plot_results.py     ← Comparison charts
├── charts/             ← Output SVGs (after running plot_results.py)
├── Makefile
└── README.md
```
