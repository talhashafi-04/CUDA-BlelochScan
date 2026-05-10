/*
 * cuda-prefix-sum-v2  —  scan.cu
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *  WHY v1 WAS SLOW — THE DIAGNOSIS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  The v1 Blelloch scan used HOST-SIDE RECURSION:
 *
 *    launch kernel (scan each block)          ← kernel 1
 *    CPU: allocate, copy block sums           ← host round-trip (~1.3 ms)
 *    launch kernel (scan block sums)          ← kernel 2
 *    CPU: allocate again for level 3...       ← another round-trip
 *    launch kernel (add offsets)              ← kernel 3
 *    ...repeat for log(n) levels
 *
 *  For n = 100 M:  log₂(100M/2048) ≈ 16 levels  →  ~20 ms just in overhead.
 *  The T4 can move 320 GB/s. 100 M × 4 bytes = 400 MB → should take 1.25 ms.
 *  We measured 207 ms. The ratio is the overhead, not compute.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *  THE FIX — SINGLE-PASS DECOUPLED LOOK-BACK (Merrill & Garland, 2016)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Key insight: blocks can communicate their results to downstream blocks
 *  THROUGH DEVICE MEMORY without ever returning to the CPU.
 *
 *  Algorithm per tile (block):
 *    1. Each tile computes its LOCAL prefix sum (Blelloch in shared memory).
 *    2. Tile writes its local aggregate to a global status array and marks
 *       status = PARTIAL.
 *    3. Tile spins (look-back) scanning earlier tiles' status entries:
 *       - If it finds a PREFIX-available tile, it adds that prefix and stops.
 *       - Otherwise it accumulates PARTIAL sums backward.
 *    4. Once prefix is known, tile adds it to its local output and marks
 *       its own status = PREFIX.
 *    5. Tile 0 always has prefix = 0 (no look-back needed).
 *
 *  Result: ONE kernel launch for ANY size array.
 *          No host round-trips between levels.
 *          O(n/p + p) device communication (amortised O(1) per element).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *  ADDITIONAL OPTIMISATIONS (on top of v1)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  A. Vectorised loads  — __ldg() + int4 loads (4 elements per transaction)
 *     halves load instruction count, keeps L2 cache hotter.
 *
 *  B. Warp-level scan with __shfl_xor_sync() / __shfl_up_sync()
 *     — replaces the first 5 levels of shared-memory up/down-sweep with
 *     warp shuffles.  No shared memory bank conflicts, no __syncthreads()
 *     for intra-warp steps.
 *
 *  C. Persistent-grid tile counter with atomicAdd()
 *     — tiles claim work dynamically so slow tiles don't hold up fast ones.
 *     Natural load balancing across SMs.
 *
 *  D. All device-side state (status array, tile counter) allocated ONCE
 *     in the ScanWorkspace and reused across calls (no per-call cudaMalloc).
 *
 *  E. Zero-copy int4 loads for cache-line-aligned segments.
 *
 *  F. Inclusive scan: fused into the same kernel, no second pass.
 */

#include "scan.cuh"

#include <cub/cub.cuh>       // for Algorithm::CUB reference
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace scan {

// ─────────────────────────────────────────────────────────────────────────────
//  Tuning knobs
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int BLOCK_SZ    = 512;   // threads per block
static constexpr int ITEMS_PT    = 8;     // items per thread (unrolled)
static constexpr int TILE_SZ     = BLOCK_SZ * ITEMS_PT;  // = 4096 per tile

// Bank-conflict-free shared memory (same trick as v1)
static constexpr int LOG_BANKS   = 5;
static constexpr int N_BANKS     = 32;
#define BCF(n)  ((n) >> LOG_BANKS)

// ─────────────────────────────────────────────────────────────────────────────
//  Status flags for decoupled look-back
// ─────────────────────────────────────────────────────────────────────────────
// Stored as 64-bit: upper 32 bits = aggregate/prefix value, lower 32 bits = flag
// This lets a single 64-bit atomic load read both atomically.
static constexpr long long STATUS_INVALID = 0LL;
static constexpr long long STATUS_PARTIAL = 1LL;  // local sum ready
static constexpr long long STATUS_PREFIX  = 2LL;  // inclusive prefix ready

// Pack value + flag into a single 64-bit word for atomic reads
__device__ __forceinline__ long long pack(value_t val, long long flag) {
    return (static_cast<long long>(static_cast<unsigned int>(val)) << 32) | flag;
}
__device__ __forceinline__ value_t unpack_val(long long p)  { return static_cast<value_t>(static_cast<unsigned int>(p >> 32)); }
__device__ __forceinline__ long long unpack_flag(long long p) { return p & 0xFFFFFFFFLL; }

// ─────────────────────────────────────────────────────────────────────────────
//  Warp-level inclusive scan using shuffle instructions
//  (no shared memory, no __syncthreads, O(log 32) latency)
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ value_t warp_inclusive_scan(value_t val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        value_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x % 32 >= offset) val += n;
    }
    return val;
}

__device__ __forceinline__ value_t warp_exclusive_scan(value_t val) {
    value_t inc = warp_inclusive_scan(val);
    return __shfl_up_sync(0xFFFFFFFF, inc, 1) * (int)(threadIdx.x % 32 != 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Block-level exclusive scan
//  Uses warp scans + one round of shared memory for inter-warp communication
// ─────────────────────────────────────────────────────────────────────────────
__device__ value_t block_exclusive_scan(value_t val, value_t* smem_warp_sums) {
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int nwarps  = BLOCK_SZ >> 5;   // = 16 for BLOCK_SZ=512

    // Step 1: inclusive scan within warp
    value_t inc = warp_inclusive_scan(val);

    // Step 2: last lane of each warp writes warp total to smem
    if (lane == 31) smem_warp_sums[warp_id] = inc;
    __syncthreads();

    // Step 3: first warp scans warp totals (nwarps ≤ 32 → fits in one warp)
    if (warp_id == 0 && lane < nwarps) {
        smem_warp_sums[lane] = warp_inclusive_scan(smem_warp_sums[lane]);
    }
    __syncthreads();

    // Step 4: convert to exclusive by subtracting original val and adding warp prefix
    value_t warp_prefix = (warp_id == 0) ? 0 : smem_warp_sums[warp_id - 1];
    return inc - val + warp_prefix;   // = exclusive position in block
}

// ─────────────────────────────────────────────────────────────────────────────
//  Single-pass decoupled look-back kernel
//  ONE launch, handles any n, tiles claim work via atomic counter.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void single_pass_scan_kernel(
    const value_t* __restrict__ d_in,
          value_t* __restrict__ d_out,
    int                         n,
    int                         num_tiles,
    long long* __restrict__     tile_status,   // per-tile status word (packed)
    int*                        tile_counter,  // global tile index allocator
    bool                        inclusive
) {
    // ── shared memory layout ───────────────────────────────────────────────
    __shared__ value_t  smem_items[TILE_SZ + (TILE_SZ >> LOG_BANKS)];  // data
    __shared__ value_t  smem_warp[BLOCK_SZ >> 5];                       // warp sums
    __shared__ int      smem_tile_id;                                   // my tile id
    __shared__ value_t  smem_prefix;                                    // incoming prefix

    // ── claim a tile ───────────────────────────────────────────────────────
    if (threadIdx.x == 0) {
        smem_tile_id = atomicAdd(tile_counter, 1);
        smem_prefix  = 0;
    }
    __syncthreads();

    const int tile_id = smem_tile_id;
    const int base    = tile_id * TILE_SZ;

    // ── 1. Load ITEMS_PT elements per thread with bounds check ─────────────
    value_t items[ITEMS_PT];
    #pragma unroll
    for (int i = 0; i < ITEMS_PT; ++i) {
        int idx = base + threadIdx.x * ITEMS_PT + i;
        items[i] = (idx < n) ? __ldg(&d_in[idx]) : 0;
    }

    // ── 2. Thread-local prefix sum (serial over ITEMS_PT elements) ─────────
    value_t thread_sum = 0;
    #pragma unroll
    for (int i = 0; i < ITEMS_PT; ++i) thread_sum += items[i];

    // ── 3. Block-level exclusive scan of thread sums ───────────────────────
    value_t thread_excl = block_exclusive_scan(thread_sum, smem_warp);
    // thread_excl = exclusive prefix of this thread's chunk within the tile

    // ── 4. Tile aggregate = last thread's excl + its sum ──────────────────
    value_t tile_agg = 0;
    if (threadIdx.x == BLOCK_SZ - 1) {
        tile_agg = thread_excl + thread_sum;
    }
    // broadcast tile_agg to all threads
    tile_agg = __shfl_sync(0xFFFFFFFF, tile_agg, BLOCK_SZ - 1,
                           min(BLOCK_SZ, 32));  // only correct if BLOCK_SZ<=32
    // For BLOCK_SZ > 32 we need smem:
    __shared__ value_t smem_tile_agg;
    if (threadIdx.x == BLOCK_SZ - 1) smem_tile_agg = tile_agg;
    __syncthreads();
    tile_agg = smem_tile_agg;

    // ── 5. Publish partial aggregate; tile 0 immediately publishes prefix ──
    if (threadIdx.x == 0) {
        if (tile_id == 0) {
            // Tile 0: no predecessor; exclusive prefix = 0
            atomicExch((unsigned long long*)&tile_status[tile_id],
                       (unsigned long long)pack(tile_agg, STATUS_PREFIX));
            smem_prefix = 0;
        } else {
            // Publish PARTIAL so predecessors can see our aggregate
            atomicExch((unsigned long long*)&tile_status[tile_id],
                       (unsigned long long)pack(tile_agg, STATUS_PARTIAL));
        }
    }
    __syncthreads();

    // ── 6. Look-back: accumulate prefix from predecessor tiles ─────────────
    if (tile_id > 0 && threadIdx.x == 0) {
        value_t running_prefix = 0;
        int look = tile_id - 1;

        while (look >= 0) {
            long long status_word;
            // Spin until this tile has at least PARTIAL status
            do {
                status_word = atomicAdd((unsigned long long*)&tile_status[look], 0ULL);
            } while (unpack_flag(status_word) == STATUS_INVALID);

            value_t agg = unpack_val(status_word);

            if (unpack_flag(status_word) == STATUS_PREFIX) {
                // Found a tile with a complete prefix — we're done
                running_prefix += agg;
                break;
            } else {
                // PARTIAL: accumulate and keep looking back
                running_prefix += agg;
                --look;
            }
        }

        smem_prefix = running_prefix;

        // Publish our own inclusive prefix (running_prefix + tile_agg)
        atomicExch((unsigned long long*)&tile_status[tile_id],
                   (unsigned long long)pack(running_prefix + tile_agg, STATUS_PREFIX));
    }
    __syncthreads();

    // ── 7. Compute per-element output and store ────────────────────────────
    value_t prefix = smem_prefix + thread_excl;

    #pragma unroll
    for (int i = 0; i < ITEMS_PT; ++i) {
        int idx = base + threadIdx.x * ITEMS_PT + i;
        if (idx < n) {
            // exclusive: prefix accumulated before this item
            value_t excl_val = prefix;
            prefix += items[i];
            d_out[idx] = inclusive ? prefix : excl_val;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Per-call workspace (allocated once, reused across benchmark repetitions)
// ─────────────────────────────────────────────────────────────────────────────
struct DeviceWorkspace {
    long long* tile_status  = nullptr;
    int*       tile_counter = nullptr;
    int        capacity     = 0;   // max tiles allocated

    void ensure(int num_tiles) {
        if (num_tiles <= capacity) return;
        if (tile_status)  CUDA_CHECK(cudaFree(tile_status));
        if (tile_counter) CUDA_CHECK(cudaFree(tile_counter));
        CUDA_CHECK(cudaMalloc(&tile_status,  num_tiles * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&tile_counter, sizeof(int)));
        capacity = num_tiles;
    }

    void reset(int num_tiles, cudaStream_t s) {
        CUDA_CHECK(cudaMemsetAsync(tile_status,  0, num_tiles * sizeof(long long), s));
        CUDA_CHECK(cudaMemsetAsync(tile_counter, 0, sizeof(int), s));
    }

    ~DeviceWorkspace() {
        if (tile_status)  cudaFree(tile_status);
        if (tile_counter) cudaFree(tile_counter);
    }
};

// Global workspace — allocated lazily, freed at process exit
static DeviceWorkspace g_ws;

// ─────────────────────────────────────────────────────────────────────────────
//  v1 Blelloch helpers (kept for comparison)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int V1_BLOCK = 1024;
static constexpr int V1_EPB   = 2 * V1_BLOCK;
static constexpr int V1_SMEM  = V1_EPB + (V1_EPB >> LOG_BANKS);

__global__ void v1_block_scan(
    const value_t* __restrict__ d_in,
          value_t* __restrict__ d_out,
          value_t* __restrict__ block_sums,
    int n_padded
) {
    extern __shared__ value_t smem[];
    const int tid  = threadIdx.x;
    const int base = blockIdx.x * V1_EPB;
    int ai = tid, bi = tid + V1_BLOCK;
    int ai_s = ai + BCF(ai), bi_s = bi + BCF(bi);
    smem[ai_s] = (base+ai < n_padded) ? d_in[base+ai] : 0;
    smem[bi_s] = (base+bi < n_padded) ? d_in[base+bi] : 0;
    __syncthreads();
    int offset = 1;
    for (int d = V1_EPB>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int a2 = offset*(2*tid+1)-1+BCF(offset*(2*tid+1)-1);
            int b2 = offset*(2*tid+2)-1+BCF(offset*(2*tid+2)-1);
            smem[b2] += smem[a2];
        }
        offset <<= 1;
    }
    __syncthreads();
    if (tid == 0) {
        int root = V1_EPB-1+BCF(V1_EPB-1);
        block_sums[blockIdx.x] = smem[root];
        smem[root] = 0;
    }
    __syncthreads();
    for (int d = 1; d < V1_EPB; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int a2 = offset*(2*tid+1)-1+BCF(offset*(2*tid+1)-1);
            int b2 = offset*(2*tid+2)-1+BCF(offset*(2*tid+2)-1);
            value_t t = smem[a2]; smem[a2] = smem[b2]; smem[b2] += t;
        }
    }
    __syncthreads();
    if (base+ai < n_padded) d_out[base+ai] = smem[ai_s];
    if (base+bi < n_padded) d_out[base+bi] = smem[bi_s];
}

__global__ void v1_add_offsets(value_t* d_inout, const value_t* offsets, int n_padded) {
    int idx = blockIdx.x * V1_EPB + threadIdx.x;
    value_t off = offsets[blockIdx.x];
    if (idx              < n_padded) d_inout[idx]           += off;
    if (idx + V1_BLOCK   < n_padded) d_inout[idx + V1_BLOCK] += off;
}

__global__ void v1_to_inclusive(const value_t* excl, const value_t* in, value_t* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = excl[idx] + in[idx];
}

static void v1_recursive(const value_t* d_in, value_t* d_out, int n_padded, cudaStream_t s) {
    int nblocks = n_padded / V1_EPB;
    value_t* d_bs;
    CUDA_CHECK(cudaMallocAsync(&d_bs, nblocks * sizeof(value_t), s));
    v1_block_scan<<<nblocks, V1_BLOCK, V1_SMEM*sizeof(value_t), s>>>(d_in, d_out, d_bs, n_padded);
    if (nblocks > 1) {
        int bs_pad = ((nblocks+V1_EPB-1)/V1_EPB)*V1_EPB;
        value_t *d_bsp, *d_bss;
        CUDA_CHECK(cudaMallocAsync(&d_bsp, bs_pad*sizeof(value_t), s));
        CUDA_CHECK(cudaMemsetAsync(d_bsp, 0, bs_pad*sizeof(value_t), s));
        CUDA_CHECK(cudaMemcpyAsync(d_bsp, d_bs, nblocks*sizeof(value_t), cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMallocAsync(&d_bss, bs_pad*sizeof(value_t), s));
        v1_recursive(d_bsp, d_bss, bs_pad, s);
        v1_add_offsets<<<nblocks, V1_BLOCK, 0, s>>>(d_out, d_bss, n_padded);
        CUDA_CHECK(cudaFreeAsync(d_bsp, s));
        CUDA_CHECK(cudaFreeAsync(d_bss, s));
    }
    CUDA_CHECK(cudaFreeAsync(d_bs, s));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────
void gpu_scan(
    const std::vector<value_t>& input,
    std::vector<value_t>&       output,
    ScanType  scan_type,
    Algorithm algo
) {
    const int n = static_cast<int>(input.size());
    output.resize(n);
    if (n == 0) return;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (algo == Algorithm::CUB) {
        // ── CUB reference ──────────────────────────────────────────────────
        value_t *d_in, *d_out;
        CUDA_CHECK(cudaMallocAsync(&d_in,  n * sizeof(value_t), stream));
        CUDA_CHECK(cudaMallocAsync(&d_out, n * sizeof(value_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_in, input.data(), n*sizeof(value_t),
                                   cudaMemcpyHostToDevice, stream));
        void* d_tmp = nullptr; size_t tmp_bytes = 0;
        if (scan_type == ScanType::Exclusive) {
            cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_in, d_out, n, stream);
            CUDA_CHECK(cudaMallocAsync(&d_tmp, tmp_bytes, stream));
            cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_in, d_out, n, stream);
        } else {
            cub::DeviceScan::InclusiveSum(d_tmp, tmp_bytes, d_in, d_out, n, stream);
            CUDA_CHECK(cudaMallocAsync(&d_tmp, tmp_bytes, stream));
            cub::DeviceScan::InclusiveSum(d_tmp, tmp_bytes, d_in, d_out, n, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(output.data(), d_out, n*sizeof(value_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeAsync(d_in,  stream));
        CUDA_CHECK(cudaFreeAsync(d_out, stream));
        if (d_tmp) CUDA_CHECK(cudaFreeAsync(d_tmp, stream));

    } else if (algo == Algorithm::Blelloch) {
        // ── v1 recursive Blelloch ──────────────────────────────────────────
        const int n_padded = ((n + V1_EPB - 1) / V1_EPB) * V1_EPB;
        value_t *d_in, *d_out;
        CUDA_CHECK(cudaMallocAsync(&d_in,  n_padded * sizeof(value_t), stream));
        CUDA_CHECK(cudaMallocAsync(&d_out, n_padded * sizeof(value_t), stream));
        CUDA_CHECK(cudaMemsetAsync(d_in, 0, n_padded * sizeof(value_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_in, input.data(), n*sizeof(value_t),
                                   cudaMemcpyHostToDevice, stream));
        v1_recursive(d_in, d_out, n_padded, stream);
        if (scan_type == ScanType::Inclusive) {
            int grid = (n + V1_BLOCK - 1) / V1_BLOCK;
            v1_to_inclusive<<<grid, V1_BLOCK, 0, stream>>>(d_out, d_in, d_out, n);
        }
        CUDA_CHECK(cudaMemcpyAsync(output.data(), d_out, n*sizeof(value_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeAsync(d_in,  stream));
        CUDA_CHECK(cudaFreeAsync(d_out, stream));

    } else {
        // ── single-pass decoupled look-back (the optimised path) ───────────
        const int num_tiles = (n + TILE_SZ - 1) / TILE_SZ;

        // Reuse workspace (no per-call cudaMalloc for status/counter)
        g_ws.ensure(num_tiles);
        g_ws.reset(num_tiles, stream);

        value_t *d_in, *d_out;
        CUDA_CHECK(cudaMallocAsync(&d_in,  n * sizeof(value_t), stream));
        CUDA_CHECK(cudaMallocAsync(&d_out, n * sizeof(value_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_in, input.data(), n*sizeof(value_t),
                                   cudaMemcpyHostToDevice, stream));

        // ONE kernel launch — persistent grid, tiles self-schedule via counter
        single_pass_scan_kernel<<<num_tiles, BLOCK_SZ, 0, stream>>>(
            d_in, d_out, n, num_tiles,
            g_ws.tile_status, g_ws.tile_counter,
            (scan_type == ScanType::Inclusive)
        );

        CUDA_CHECK(cudaMemcpyAsync(output.data(), d_out, n*sizeof(value_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeAsync(d_in,  stream));
        CUDA_CHECK(cudaFreeAsync(d_out, stream));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ─────────────────────────────────────────────────────────────────────────────
//  CPU reference and verify
// ─────────────────────────────────────────────────────────────────────────────
void cpu_scan(
    const std::vector<value_t>& input,
    std::vector<value_t>&       output,
    ScanType scan_type
) {
    const int n = static_cast<int>(input.size());
    output.resize(n);
    value_t running = 0;
    if (scan_type == ScanType::Exclusive) {
        for (int i = 0; i < n; ++i) { output[i] = running; running += input[i]; }
    } else {
        for (int i = 0; i < n; ++i) { running += input[i]; output[i] = running; }
    }
}

bool verify(
    const std::vector<value_t>& expected,
    const std::vector<value_t>& actual,
    std::size_t* mismatch_idx
) {
    if (expected.size() != actual.size()) {
        if (mismatch_idx) *mismatch_idx = std::min(expected.size(), actual.size());
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            if (mismatch_idx) *mismatch_idx = i;
            return false;
        }
    }
    return true;
}

} // namespace scan
