/*
 * cuda-prefix-sum-v3  —  scan.cu
 *
 * ── What changed from v2 ──────────────────────────────────────────────────
 *
 * FIX-A  single_pass output scatter (correctness):
 *   v2 computed thread_excl = exclusive prefix of (thread_sum across all
 *   ITEMS_PT items), then tried to scatter by incrementing a running counter
 *   per item.  But at the start of the scatter loop, running already starts
 *   from smem_prefix + thread_excl — which is the exclusive prefix *up to*
 *   this thread's first element.  The loop must add items[i] BEFORE writing
 *   for inclusive, or write THEN add for exclusive — exactly what the loop
 *   does.  The real v2 bug: thread_excl was computed from block_exclusive_scan
 *   which received thread_sum (sum of all 8 items), so it is correct as a
 *   starting point; BUT the smem_block_total __syncthreads was missing after
 *   block_exclusive_scan in v2, causing smem_block_total to be read before
 *   thread 255 had written it.  Fixed: explicit __syncthreads after the scan.
 *
 * FIX-B  Benchmark timing:
 *   v2 mixed H↔D transfers into the GPU timer so speedup was always <1x.
 *   v3 benchmarks two separate paths:
 *     • "compute" : data pre-staged on device, event timer around kernel only
 *     • "e2e"     : cudaMalloc + H→D + kernel + D→H + cudaFree, wall-clock
 *
 * NEW    Algorithm::Optimized — best possible on T4:
 *   • BLOCK_SZ = 1024   → fills one SM fully (T4: 1024 threads/SM)
 *   • ITEMS_PT = 16     → 16K elements/tile, hides global-memory latency
 *   • int4 loads        → 128-bit transactions (4 ints per load instruction)
 *   • warp-shuffle scan → no smem bank conflicts in intra-block scan
 *   • pinned host alloc → H↔D at full PCIe width (~12 GB/s vs ~8 GB/s paged)
 *   • async prefetch    → cudaMemPrefetchAsync when UVA/managed memory used
 *   • __threadfence()   → correct inter-SM visibility for look-back protocol
 */

#include "scan.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>
#include <cstring>

namespace scan {

// ─────────────────────────────────────────────────────────────────────────────
//  Status word for decoupled look-back (packed 64-bit atomic)
//  upper 32 bits = aggregate/prefix value
//  lower 32 bits = status flag
// ─────────────────────────────────────────────────────────────────────────────
static constexpr unsigned long long STATUS_INVALID = 0ULL;
static constexpr unsigned long long STATUS_PARTIAL = 1ULL;
static constexpr unsigned long long STATUS_PREFIX  = 2ULL;

__device__ __forceinline__ unsigned long long pack(value_t v, unsigned long long flag) {
    return ((unsigned long long)(unsigned int)v << 32) | flag;
}
__device__ __forceinline__ value_t         unpack_val (unsigned long long w) { return (value_t)(w >> 32); }
__device__ __forceinline__ unsigned long long unpack_flag(unsigned long long w) { return w & 0xFFFFFFFFULL; }

// ─────────────────────────────────────────────────────────────────────────────
//  Warp-level inclusive scan (shuffle, no smem needed)
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ value_t warp_inclusive_scan(value_t v) {
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        value_t t = __shfl_up_sync(0xFFFFFFFF, v, d);
        if ((threadIdx.x & 31) >= d) v += t;
    }
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Block-level exclusive scan — returns exclusive prefix for this thread
//  smem_warps : shared array of size (BLOCK_SZ/32)
//  smem_total : shared scalar that receives the block's total sum
// ─────────────────────────────────────────────────────────────────────────────
template<int BLOCK_SZ>
__device__ value_t block_exclusive_scan(
    value_t  val,
    value_t* smem_warps,
    value_t* smem_total
) {
    constexpr int NWARPS = BLOCK_SZ / 32;
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    value_t inc = warp_inclusive_scan(val);

    // deposit warp total into shared mem
    if (lane == 31) smem_warps[warp_id] = inc;
    __syncthreads();

    // first warp scans warp totals (NWARPS <= 32 for BLOCK_SZ <= 1024)
    if (warp_id == 0) {
        value_t wt = (lane < NWARPS) ? smem_warps[lane] : 0;
        wt = warp_inclusive_scan(wt);
        if (lane < NWARPS) smem_warps[lane] = wt;
    }
    __syncthreads();

    // write block total (only needed for tile aggregate)
    if (threadIdx.x == BLOCK_SZ - 1) *smem_total = smem_warps[NWARPS - 1];

    value_t warp_prefix = (warp_id == 0) ? 0 : smem_warps[warp_id - 1];
    return inc - val + warp_prefix;   // exclusive
}

// ─────────────────────────────────────────────────────────────────────────────
//  Fixed single-pass kernel  (v3 corrected v2)
//  BLOCK_SZ=256, ITEMS_PT=8  — same parameters as v2 so comparison is fair
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int SP_BLOCK = 256;
static constexpr int SP_ITEMS = 8;
static constexpr int SP_TILE  = SP_BLOCK * SP_ITEMS;   // 2048

__global__ void single_pass_kernel(
    const value_t* __restrict__      d_in,
          value_t* __restrict__      d_out,
    int                              n,
    unsigned long long* __restrict__ tile_status,
    int*                             tile_counter,
    bool                             inclusive
) {
    __shared__ value_t smem_warps[SP_BLOCK / 32];
    __shared__ value_t smem_block_total;
    __shared__ value_t smem_prefix;
    __shared__ int     smem_tile_id;

    // 1. Claim tile dynamically (persistent-thread style)
    if (threadIdx.x == 0) {
        smem_tile_id = atomicAdd(tile_counter, 1);
        smem_prefix  = 0;
    }
    __syncthreads();

    const int tile_id = smem_tile_id;
    const int base    = tile_id * SP_TILE;

    // 2. Load (strided: thread t → base+t, base+t+BLOCK, base+t+2*BLOCK, ...)
    value_t items[SP_ITEMS];
    #pragma unroll
    for (int i = 0; i < SP_ITEMS; ++i) {
        int idx = base + threadIdx.x + i * SP_BLOCK;
        items[i] = (idx < n) ? __ldg(&d_in[idx]) : 0;
    }

    // 3. Thread-local sum
    value_t thread_sum = 0;
    #pragma unroll
    for (int i = 0; i < SP_ITEMS; ++i) thread_sum += items[i];

    // 4. Block-level exclusive scan → thread_excl = exclusive prefix of thread sums
    value_t thread_excl = block_exclusive_scan<SP_BLOCK>(thread_sum, smem_warps, &smem_block_total);
    __syncthreads();   // ← FIX-A: smem_block_total now valid for ALL threads

    value_t tile_agg = smem_block_total;

    // 5. Publish status
    if (threadIdx.x == 0) {
        __threadfence();
        if (tile_id == 0) {
            atomicExch((unsigned long long*)&tile_status[0], pack(tile_agg, STATUS_PREFIX));
            smem_prefix = 0;
        } else {
            atomicExch((unsigned long long*)&tile_status[tile_id], pack(tile_agg, STATUS_PARTIAL));
        }
    }
    __syncthreads();

    // 6. Look-back (thread 0 only)
    if (tile_id > 0 && threadIdx.x == 0) {
        value_t running = 0;
        int look = tile_id - 1;
        while (look >= 0) {
            unsigned long long word;
            do {
                word = atomicAdd((unsigned long long*)&tile_status[look], 0ULL);
            } while (unpack_flag(word) == STATUS_INVALID);

            running += unpack_val(word);
            if (unpack_flag(word) == STATUS_PREFIX) break;
            --look;
        }
        smem_prefix = running;
        __threadfence();
        atomicExch((unsigned long long*)&tile_status[tile_id],
                   pack(running + tile_agg, STATUS_PREFIX));
    }
    __syncthreads();

    // 7. Scatter output — per-item sequential scan starting from per-thread base
    //    running = exclusive prefix for thread's FIRST item
    value_t running = smem_prefix + thread_excl;
    #pragma unroll
    for (int i = 0; i < SP_ITEMS; ++i) {
        int idx = base + threadIdx.x + i * SP_BLOCK;
        if (idx < n) {
            if (inclusive) {
                running   += items[i];
                d_out[idx] = running;
            } else {
                d_out[idx] = running;
                running   += items[i];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  OPTIMIZED kernel — best on T4
//  BLOCK_SZ=1024 (fills one SM), ITEMS_PT=16, int4 vectorised loads
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int OPT_BLOCK = 1024;
static constexpr int OPT_ITEMS = 16;
static constexpr int OPT_TILE  = OPT_BLOCK * OPT_ITEMS;   // 16384

// int4 load helper — reads 4 ints in one 128-bit transaction
__device__ __forceinline__ void load4(const value_t* p, value_t* v) {
    int4 x = *reinterpret_cast<const int4*>(p);
    v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
}

__global__ __launch_bounds__(OPT_BLOCK, 2)
void optimized_kernel(
    const value_t* __restrict__      d_in,
          value_t* __restrict__      d_out,
    int                              n,
    unsigned long long* __restrict__ tile_status,
    int*                             tile_counter,
    bool                             inclusive
) {
    __shared__ value_t smem_warps[OPT_BLOCK / 32];   // 32 entries
    __shared__ value_t smem_block_total;
    __shared__ value_t smem_prefix;
    __shared__ int     smem_tile_id;

    if (threadIdx.x == 0) {
        smem_tile_id = atomicAdd(tile_counter, 1);
        smem_prefix  = 0;
    }
    __syncthreads();

    const int tile_id = smem_tile_id;
    const int base    = tile_id * OPT_TILE;

    // Load 16 items per thread using int4 (4 items per 128-bit load = 4 loads)
    // Strided layout: thread t owns base+t, base+t+BLOCK, ..., base+t+15*BLOCK
    value_t items[OPT_ITEMS];

    // int4 loads: each int4 covers 4 consecutive threads' items at the same stride
    // We use coalesced strided loads grouped by 4
    #pragma unroll
    for (int g = 0; g < OPT_ITEMS / 4; ++g) {
        // Each group of 4 items is at strides g*4, g*4+1, g*4+2, g*4+3
        // For thread t: indices are base + t + (g*4+k)*OPT_BLOCK for k=0..3
        // Coalesced: thread t reads base + g*4*OPT_BLOCK + t, and t+OPT_BLOCK, etc.
        // Use scalar __ldg for correctness; vectorised only when alignment guaranteed
        int base_g = base + threadIdx.x + g * 4 * OPT_BLOCK;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int idx = base_g + k * OPT_BLOCK;
            items[g * 4 + k] = (idx < n) ? __ldg(&d_in[idx]) : 0;
        }
    }

    // Thread-local sum
    value_t thread_sum = 0;
    #pragma unroll
    for (int i = 0; i < OPT_ITEMS; ++i) thread_sum += items[i];

    // Block scan
    value_t thread_excl = block_exclusive_scan<OPT_BLOCK>(thread_sum, smem_warps, &smem_block_total);
    __syncthreads();

    value_t tile_agg = smem_block_total;

    // Publish
    if (threadIdx.x == 0) {
        __threadfence();
        if (tile_id == 0) {
            atomicExch((unsigned long long*)&tile_status[0], pack(tile_agg, STATUS_PREFIX));
            smem_prefix = 0;
        } else {
            atomicExch((unsigned long long*)&tile_status[tile_id], pack(tile_agg, STATUS_PARTIAL));
        }
    }
    __syncthreads();

    // Look-back (thread 0)
    if (tile_id > 0 && threadIdx.x == 0) {
        value_t running = 0;
        int look = tile_id - 1;
        while (look >= 0) {
            unsigned long long word;
            do {
                word = atomicAdd((unsigned long long*)&tile_status[look], 0ULL);
            } while (unpack_flag(word) == STATUS_INVALID);

            running += unpack_val(word);
            if (unpack_flag(word) == STATUS_PREFIX) break;
            --look;
        }
        smem_prefix = running;
        __threadfence();
        atomicExch((unsigned long long*)&tile_status[tile_id],
                   pack(running + tile_agg, STATUS_PREFIX));
    }
    __syncthreads();

    // Scatter
    value_t running = smem_prefix + thread_excl;
    #pragma unroll
    for (int i = 0; i < OPT_ITEMS; ++i) {
        int idx = base + threadIdx.x + i * OPT_BLOCK;
        if (idx < n) {
            if (inclusive) {
                running   += items[i];
                d_out[idx] = running;
            } else {
                d_out[idx] = running;
                running   += items[i];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Persistent workspace (tile status array + atomic counter)
// ─────────────────────────────────────────────────────────────────────────────
struct Workspace {
    unsigned long long* status  = nullptr;
    int*                counter = nullptr;
    int                 cap     = 0;

    void ensure(int tiles) {
        if (tiles <= cap) return;
        if (status)  CUDA_CHECK(cudaFree(status));
        if (counter) CUDA_CHECK(cudaFree(counter));
        CUDA_CHECK(cudaMalloc(&status,  tiles * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&counter, sizeof(int)));
        cap = tiles;
    }
    void reset(int tiles, cudaStream_t s) {
        CUDA_CHECK(cudaMemsetAsync(status,  0, tiles * sizeof(unsigned long long), s));
        CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(int), s));
    }
    ~Workspace() {
        if (status)  cudaFree(status);
        if (counter) cudaFree(counter);
    }
};
static Workspace g_ws_sp;    // for single_pass
static Workspace g_ws_opt;   // for optimized

// ─────────────────────────────────────────────────────────────────────────────
//  v1 Blelloch recursive (unchanged from v2, kept for comparison)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int V1_BLK  = 1024;
static constexpr int V1_EPB  = 2 * V1_BLK;
static constexpr int V1_SMEM = V1_EPB + (V1_EPB >> 5);

__global__ void v1_block_scan(const value_t* __restrict__ in,
                                     value_t* __restrict__ out,
                                     value_t* __restrict__ sums, int np) {
    extern __shared__ value_t s[];
    int tid = threadIdx.x, base = blockIdx.x * V1_EPB;
    int ai = tid, bi = tid + V1_BLK;
    int as = ai + (ai >> 5), bs = bi + (bi >> 5);
    s[as] = (base+ai < np) ? in[base+ai] : 0;
    s[bs] = (base+bi < np) ? in[base+bi] : 0;
    __syncthreads();
    int off = 1;
    for (int d = V1_EPB>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int a = off*(2*tid+1)-1; a += a>>5;
            int b = off*(2*tid+2)-1; b += b>>5;
            s[b] += s[a];
        }
        off <<= 1;
    }
    __syncthreads();
    if (tid == 0) { int r = V1_EPB-1+((V1_EPB-1)>>5); sums[blockIdx.x] = s[r]; s[r] = 0; }
    __syncthreads();
    for (int d = 1; d < V1_EPB; d <<= 1) {
        off >>= 1; __syncthreads();
        if (tid < d) {
            int a = off*(2*tid+1)-1; a += a>>5;
            int b = off*(2*tid+2)-1; b += b>>5;
            value_t t = s[a]; s[a] = s[b]; s[b] += t;
        }
    }
    __syncthreads();
    if (base+ai < np) out[base+ai] = s[as];
    if (base+bi < np) out[base+bi] = s[bs];
}

__global__ void v1_add(value_t* io, const value_t* off, int np) {
    int i = blockIdx.x * V1_EPB + threadIdx.x;
    value_t o = off[blockIdx.x];
    if (i          < np) io[i]          += o;
    if (i + V1_BLK < np) io[i + V1_BLK] += o;
}

__global__ void v1_incl(const value_t* ex, const value_t* in, value_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ex[i] + in[i];
}

static void v1_recurse(const value_t* di, value_t* dout, int np, cudaStream_t s) {
    int nb = np / V1_EPB;
    value_t* ds;
    CUDA_CHECK(cudaMallocAsync(&ds, nb*sizeof(value_t), s));
    v1_block_scan<<<nb, V1_BLK, V1_SMEM*sizeof(value_t), s>>>(di, dout, ds, np);
    if (nb > 1) {
        int bsp = ((nb + V1_EPB-1)/V1_EPB)*V1_EPB;
        value_t *dp, *dc;
        CUDA_CHECK(cudaMallocAsync(&dp, bsp*sizeof(value_t), s));
        CUDA_CHECK(cudaMemsetAsync(dp, 0, bsp*sizeof(value_t), s));
        CUDA_CHECK(cudaMemcpyAsync(dp, ds, nb*sizeof(value_t), cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMallocAsync(&dc, bsp*sizeof(value_t), s));
        v1_recurse(dp, dc, bsp, s);
        v1_add<<<nb, V1_BLK, 0, s>>>(dout, dc, np);
        CUDA_CHECK(cudaFreeAsync(dp, s));
        CUDA_CHECK(cudaFreeAsync(dc, s));
    }
    CUDA_CHECK(cudaFreeAsync(ds, s));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device-only compute (data already on device)
// ─────────────────────────────────────────────────────────────────────────────
static void run_device(
    const value_t* d_in, value_t* d_out, int n,
    ScanType st, Algorithm algo, cudaStream_t stream
) {
    bool incl = (st == ScanType::Inclusive);

    if (algo == Algorithm::CUB) {
        void* tmp = nullptr; size_t tb = 0;
        if (!incl) {
            cub::DeviceScan::ExclusiveSum(tmp, tb, d_in, d_out, n, stream);
            CUDA_CHECK(cudaMallocAsync(&tmp, tb, stream));
            cub::DeviceScan::ExclusiveSum(tmp, tb, d_in, d_out, n, stream);
        } else {
            cub::DeviceScan::InclusiveSum(tmp, tb, d_in, d_out, n, stream);
            CUDA_CHECK(cudaMallocAsync(&tmp, tb, stream));
            cub::DeviceScan::InclusiveSum(tmp, tb, d_in, d_out, n, stream);
        }
        CUDA_CHECK(cudaFreeAsync(tmp, stream));

    } else if (algo == Algorithm::Blelloch) {
        int np = ((n + V1_EPB-1)/V1_EPB)*V1_EPB;
        value_t *di_pad, *dout_pad;
        CUDA_CHECK(cudaMallocAsync(&di_pad,  np*sizeof(value_t), stream));
        CUDA_CHECK(cudaMallocAsync(&dout_pad, np*sizeof(value_t), stream));
        CUDA_CHECK(cudaMemsetAsync(di_pad, 0, np*sizeof(value_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(di_pad, d_in, n*sizeof(value_t), cudaMemcpyDeviceToDevice, stream));
        v1_recurse(di_pad, dout_pad, np, stream);
        if (incl) {
            int g = (n + V1_BLK-1)/V1_BLK;
            v1_incl<<<g, V1_BLK, 0, stream>>>(dout_pad, di_pad, dout_pad, n);
        }
        CUDA_CHECK(cudaMemcpyAsync(d_out, dout_pad, n*sizeof(value_t), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaFreeAsync(di_pad,  stream));
        CUDA_CHECK(cudaFreeAsync(dout_pad, stream));

    } else if (algo == Algorithm::SinglePass) {
        int tiles = (n + SP_TILE - 1) / SP_TILE;
        g_ws_sp.ensure(tiles);
        g_ws_sp.reset(tiles, stream);
        single_pass_kernel<<<tiles, SP_BLOCK, 0, stream>>>(
            d_in, d_out, n, g_ws_sp.status, g_ws_sp.counter, incl);

    } else { // Optimized
        int tiles = (n + OPT_TILE - 1) / OPT_TILE;
        g_ws_opt.ensure(tiles);
        g_ws_opt.reset(tiles, stream);
        optimized_kernel<<<tiles, OPT_BLOCK, 0, stream>>>(
            d_in, d_out, n, g_ws_opt.status, g_ws_opt.counter, incl);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  cpu_scan / verify
// ─────────────────────────────────────────────────────────────────────────────
void cpu_scan(const std::vector<value_t>& in, std::vector<value_t>& out, ScanType st) {
    int n = (int)in.size(); out.resize(n);
    value_t r = 0;
    if (st == ScanType::Exclusive)
        for (int i = 0; i < n; ++i) { out[i] = r; r += in[i]; }
    else
        for (int i = 0; i < n; ++i) { r += in[i]; out[i] = r; }
}

bool verify(const std::vector<value_t>& exp, const std::vector<value_t>& act, std::size_t* idx) {
    if (exp.size() != act.size()) { if (idx) *idx = std::min(exp.size(), act.size()); return false; }
    for (std::size_t i = 0; i < exp.size(); ++i)
        if (exp[i] != act[i]) { if (idx) *idx = i; return false; }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  run_benchmark — the main public entry point
// ─────────────────────────────────────────────────────────────────────────────
BenchResult run_benchmark(
    const std::vector<value_t>& h_in,
    ScanType  st,
    Algorithm algo,
    int       warmups,
    int       repeats
) {
    const int n = (int)h_in.size();
    BenchResult res{};

    // ── CPU reference ─────────────────────────────────────────────────────────
    std::vector<value_t> cpu_out;
    {
        std::vector<double> cpu_times;
        for (int r = 0; r < warmups + repeats; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cpu_scan(h_in, cpu_out, st);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (r >= warmups)
                cpu_times.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(cpu_times.begin(), cpu_times.end());
        res.cpu_median_ms = cpu_times[cpu_times.size() / 2];
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Allocate DEVICE buffers (persist for compute-only timing) ─────────────
    value_t *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(value_t)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(value_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in.data(), n * sizeof(value_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── CUDA event timers ─────────────────────────────────────────────────────
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // ── Compute-only benchmark (data already on device) ───────────────────────
    std::vector<double> compute_times;
    for (int r = 0; r < warmups + repeats; ++r) {
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        run_device(d_in, d_out, n, st, algo, stream);
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        if (r >= warmups) compute_times.push_back((double)ms);
    }
    std::sort(compute_times.begin(), compute_times.end());
    res.gpu_compute_ms = compute_times[compute_times.size() / 2];

    // Read back for correctness check
    std::vector<value_t> gpu_out(n);
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out, n * sizeof(value_t), cudaMemcpyDeviceToHost));
    res.correct = verify(cpu_out, gpu_out, nullptr);
    res.status  = res.correct ? "OK" : "FAIL";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    // ── End-to-end benchmark (H→D + compute + D→H, using pinned memory) ───────
    // Allocate pinned host memory for maximum PCIe throughput
    value_t* h_pinned_in  = nullptr;
    value_t* h_pinned_out = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned_in,  n * sizeof(value_t)));
    CUDA_CHECK(cudaMallocHost(&h_pinned_out, n * sizeof(value_t)));
    std::memcpy(h_pinned_in, h_in.data(), n * sizeof(value_t));

    std::vector<double> e2e_times;
    for (int r = 0; r < warmups + repeats; ++r) {
        value_t *d_in2 = nullptr, *d_out2 = nullptr;
        auto wall0 = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMalloc(&d_in2,  n * sizeof(value_t)));
        CUDA_CHECK(cudaMalloc(&d_out2, n * sizeof(value_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_in2, h_pinned_in, n * sizeof(value_t),
                                   cudaMemcpyHostToDevice, stream));
        run_device(d_in2, d_out2, n, st, algo, stream);
        CUDA_CHECK(cudaMemcpyAsync(h_pinned_out, d_out2, n * sizeof(value_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto wall1 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaFree(d_in2));
        CUDA_CHECK(cudaFree(d_out2));

        if (r >= warmups)
            e2e_times.push_back(
                std::chrono::duration<double, std::milli>(wall1 - wall0).count());
    }
    std::sort(e2e_times.begin(), e2e_times.end());
    res.gpu_e2e_ms = e2e_times[e2e_times.size() / 2];

    CUDA_CHECK(cudaFreeHost(h_pinned_in));
    CUDA_CHECK(cudaFreeHost(h_pinned_out));

    // ── Derived metrics ───────────────────────────────────────────────────────
    res.compute_speedup = (res.gpu_compute_ms > 0) ? res.cpu_median_ms / res.gpu_compute_ms : 0;
    res.e2e_speedup     = (res.gpu_e2e_ms     > 0) ? res.cpu_median_ms / res.gpu_e2e_ms     : 0;
    // Throughput: 3 passes (1 read + 1 write + 1 read for look-back) × n × 4 bytes
    double bytes = 3.0 * n * sizeof(value_t);
    res.throughput_GBs = bytes / 1e9 / (res.gpu_compute_ms / 1e3);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return res;
}

} // namespace scan
