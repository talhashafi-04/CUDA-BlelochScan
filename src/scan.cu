/*
 * cuda-prefix-sum-v2  —  scan.cu  (fixed)
 *
 * Fixes vs previous version:
 *  FIX 1 — tile_agg broadcast: __shfl_sync only works within a warp (32 threads).
 *           For BLOCK_SZ=512, thread BLOCK_SZ-1 is in warp 15, not warp 0.
 *           The correct broadcast uses shared memory, not __shfl_sync across warps.
 *
 *  FIX 2 — look-back deadlock: if the GPU schedules tile N before tile N-1 has
 *           published STATUS_PARTIAL, the spin loop waits forever.
 *           Fix: use __threadfence() after writing tile_status so the store is
 *           visible to other SMs before the spinning tile reads it.
 *
 *  FIX 3 — smem_items array was sized with padding slots but indexed without
 *           them in the final store loop. Simplified: no padding (bank conflicts
 *           are acceptable; correctness is the priority).
 */

#include "scan.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace scan {

// ─────────────────────────────────────────────────────────────────────────────
//  Tuning
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int BLOCK_SZ = 256;
static constexpr int ITEMS_PT = 8;
static constexpr int TILE_SZ  = BLOCK_SZ * ITEMS_PT;  // 2048 per tile

// ─────────────────────────────────────────────────────────────────────────────
//  Status flags packed into a 64-bit atomic word
//  upper 32 bits = value,  lower 32 bits = flag
// ─────────────────────────────────────────────────────────────────────────────
static constexpr unsigned long long FLAG_INVALID = 0ULL;
static constexpr unsigned long long FLAG_PARTIAL = 1ULL;
static constexpr unsigned long long FLAG_PREFIX  = 2ULL;

__device__ __forceinline__
unsigned long long pack(value_t val, unsigned long long flag) {
    return ((unsigned long long)(unsigned int)val << 32) | flag;
}
__device__ __forceinline__ value_t        unpack_val (unsigned long long w) { return (value_t)(unsigned int)(w >> 32); }
__device__ __forceinline__ unsigned long long unpack_flag(unsigned long long w) { return w & 0xFFFFFFFFULL; }

// ─────────────────────────────────────────────────────────────────────────────
//  Warp-level inclusive scan (shuffle, no smem, no __syncthreads)
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
//  Block-level exclusive scan
//  smem_warps : shared scratch, size = BLOCK_SZ/32
//  smem_total : shared scalar set to the block's total sum
//  Returns exclusive prefix for this thread within the block.
// ─────────────────────────────────────────────────────────────────────────────
__device__ value_t block_exclusive_scan(
    value_t  val,
    value_t* smem_warps,
    value_t* smem_total
) {
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int nwarps  = BLOCK_SZ >> 5;

    value_t inc = warp_inclusive_scan(val);

    if (lane == 31) smem_warps[warp_id] = inc;
    __syncthreads();

    // First warp scans warp totals (nwarps <= 32 for BLOCK_SZ <= 1024)
    if (warp_id == 0) {
        value_t wt = (lane < nwarps) ? smem_warps[lane] : 0;
        wt = warp_inclusive_scan(wt);
        if (lane < nwarps) smem_warps[lane] = wt;
    }
    __syncthreads();

    if (threadIdx.x == BLOCK_SZ - 1) *smem_total = smem_warps[nwarps - 1];

    value_t warp_prefix = (warp_id == 0) ? 0 : smem_warps[warp_id - 1];
    return inc - val + warp_prefix;  // exclusive
}

// ─────────────────────────────────────────────────────────────────────────────
//  Single-pass decoupled look-back kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void single_pass_kernel(
    const value_t* __restrict__      d_in,
          value_t* __restrict__      d_out,
    int                              n,
    unsigned long long* __restrict__ tile_status,
    int*                             tile_counter,
    bool                             inclusive
) {
    __shared__ value_t smem_warps[BLOCK_SZ / 32];
    __shared__ value_t smem_block_total;
    __shared__ value_t smem_prefix;
    __shared__ int     smem_tile_id;

    // ── 1. Claim tile ─────────────────────────────────────────────────────────
    if (threadIdx.x == 0) {
        smem_tile_id = atomicAdd(tile_counter, 1);
        smem_prefix  = 0;
    }
    __syncthreads();

    const int tile_id = smem_tile_id;
    const int base    = tile_id * TILE_SZ;

    // ── 2. Load data (strided layout: thread t owns indices base+t, base+t+BLOCK_SZ, ...) ──
    value_t items[ITEMS_PT];
    #pragma unroll
    for (int i = 0; i < ITEMS_PT; ++i) {
        int idx = base + threadIdx.x + i * BLOCK_SZ;
        items[i] = (idx < n) ? __ldg(&d_in[idx]) : 0;
    }

    // ── 3. Thread-local sum ───────────────────────────────────────────────────
    value_t thread_sum = 0;
    #pragma unroll
    for (int i = 0; i < ITEMS_PT; ++i) thread_sum += items[i];

    // ── 4. Block scan → smem_block_total holds tile aggregate after __syncthreads ──
    value_t thread_excl = block_exclusive_scan(thread_sum, smem_warps, &smem_block_total);
    __syncthreads();  // smem_block_total is now valid

    value_t tile_agg = smem_block_total;

    // ── 5. Publish status ─────────────────────────────────────────────────────
    if (threadIdx.x == 0) {
        if (tile_id == 0) {
            __threadfence();
            atomicExch((unsigned long long*)&tile_status[0],
                       pack(tile_agg, FLAG_PREFIX));
            smem_prefix = 0;
        } else {
            __threadfence();
            atomicExch((unsigned long long*)&tile_status[tile_id],
                       pack(tile_agg, FLAG_PARTIAL));
        }
    }
    __syncthreads();

    // ── 6. Look-back (only thread 0 does this) ────────────────────────────────
    if (tile_id > 0 && threadIdx.x == 0) {
        value_t running = 0;
        int look = tile_id - 1;

        while (look >= 0) {
            unsigned long long word;
            // spin until predecessor publishes at least PARTIAL
            do {
                word = atomicAdd((unsigned long long*)&tile_status[look], 0ULL);
            } while (unpack_flag(word) == FLAG_INVALID);

            value_t agg = unpack_val(word);
            if (unpack_flag(word) == FLAG_PREFIX) {
                running += agg;
                break;
            } else {
                running += agg;
                --look;
            }
        }

        smem_prefix = running;

        // Publish own complete prefix
        __threadfence();
        atomicExch((unsigned long long*)&tile_status[tile_id],
                   pack(running + tile_agg, FLAG_PREFIX));
    }
    __syncthreads();

    // ── 7. Write output ───────────────────────────────────────────────────────
    value_t running = smem_prefix + thread_excl;

    #pragma unroll
    for (int i = 0; i < ITEMS_PT; ++i) {
        int idx = base + threadIdx.x + i * BLOCK_SZ;
        if (idx < n) {
            if (inclusive) {
                running    += items[i];
                d_out[idx]  = running;
            } else {
                d_out[idx]  = running;
                running    += items[i];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Persistent workspace
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
static Workspace g_ws;

// ─────────────────────────────────────────────────────────────────────────────
//  v1 Blelloch recursive (kept for comparison)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int V1_BLK  = 1024;
static constexpr int V1_EPB  = 2 * V1_BLK;
static constexpr int V1_SMEM = V1_EPB + (V1_EPB >> 5);

__global__ void v1_block_scan(
    const value_t* __restrict__ in,
          value_t* __restrict__ out,
          value_t* __restrict__ sums, int np)
{
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
    if (tid == 0) {
        int r = V1_EPB-1+((V1_EPB-1)>>5);
        sums[blockIdx.x] = s[r];
        s[r] = 0;
    }
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
//  Public API
// ─────────────────────────────────────────────────────────────────────────────
void gpu_scan(
    const std::vector<value_t>& input,
    std::vector<value_t>&       output,
    ScanType  st,
    Algorithm algo
) {
    const int n = (int)input.size();
    output.resize(n);
    if (n == 0) return;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (algo == Algorithm::CUB) {
        value_t *di, *dout;
        CUDA_CHECK(cudaMallocAsync(&di,   n*sizeof(value_t), stream));
        CUDA_CHECK(cudaMallocAsync(&dout, n*sizeof(value_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(di, input.data(), n*sizeof(value_t),
                                   cudaMemcpyHostToDevice, stream));
        void* tmp = nullptr; size_t tb = 0;
        if (st == ScanType::Exclusive) {
            cub::DeviceScan::ExclusiveSum(tmp, tb, di, dout, n, stream);
            CUDA_CHECK(cudaMallocAsync(&tmp, tb, stream));
            cub::DeviceScan::ExclusiveSum(tmp, tb, di, dout, n, stream);
        } else {
            cub::DeviceScan::InclusiveSum(tmp, tb, di, dout, n, stream);
            CUDA_CHECK(cudaMallocAsync(&tmp, tb, stream));
            cub::DeviceScan::InclusiveSum(tmp, tb, di, dout, n, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(output.data(), dout, n*sizeof(value_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeAsync(di,  stream));
        CUDA_CHECK(cudaFreeAsync(dout, stream));
        if (tmp) CUDA_CHECK(cudaFreeAsync(tmp, stream));

    } else if (algo == Algorithm::Blelloch) {
        int np = ((n + V1_EPB-1)/V1_EPB)*V1_EPB;
        value_t *di, *dout;
        CUDA_CHECK(cudaMallocAsync(&di,   np*sizeof(value_t), stream));
        CUDA_CHECK(cudaMallocAsync(&dout, np*sizeof(value_t), stream));
        CUDA_CHECK(cudaMemsetAsync(di, 0, np*sizeof(value_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(di, input.data(), n*sizeof(value_t),
                                   cudaMemcpyHostToDevice, stream));
        v1_recurse(di, dout, np, stream);
        if (st == ScanType::Inclusive) {
            int g = (n + V1_BLK-1)/V1_BLK;
            v1_incl<<<g, V1_BLK, 0, stream>>>(dout, di, dout, n);
        }
        CUDA_CHECK(cudaMemcpyAsync(output.data(), dout, n*sizeof(value_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeAsync(di,  stream));
        CUDA_CHECK(cudaFreeAsync(dout, stream));

    } else {
        // ── single-pass look-back ──────────────────────────────────────────
        int num_tiles = (n + TILE_SZ - 1) / TILE_SZ;
        g_ws.ensure(num_tiles);
        g_ws.reset(num_tiles, stream);

        value_t *di, *dout;
        CUDA_CHECK(cudaMallocAsync(&di,   n*sizeof(value_t), stream));
        CUDA_CHECK(cudaMallocAsync(&dout, n*sizeof(value_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(di, input.data(), n*sizeof(value_t),
                                   cudaMemcpyHostToDevice, stream));

        single_pass_kernel<<<num_tiles, BLOCK_SZ, 0, stream>>>(
            di, dout, n,
            g_ws.status, g_ws.counter,
            (st == ScanType::Inclusive)
        );

        CUDA_CHECK(cudaMemcpyAsync(output.data(), dout, n*sizeof(value_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeAsync(di,  stream));
        CUDA_CHECK(cudaFreeAsync(dout, stream));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

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

} // namespace scan
