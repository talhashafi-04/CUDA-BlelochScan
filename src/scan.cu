/*
 * cuda-prefix-sum  —  scan.cu
 *
 * Highly optimised GPU exclusive / inclusive prefix scan using the
 * Blelloch (1990) work-efficient tree algorithm.
 *
 * Optimisations applied
 * ─────────────────────
 * 1. Shared-memory Blelloch scan per block (no global-memory traffic per level)
 * 2. Bank-conflict-free shared memory access via +1 padding per 32 elements
 * 3. Two elements per thread → full warp utilisation at every tree level
 * 4. Multi-level block-sum recursion for arbitrary n (not limited to one block)
 * 5. Coalesced global loads and stores
 * 6. Streams + async memcpy overlap (single stream; extend to multi if needed)
 * 7. Inclusive scan derived cheaply by adding input[i] to exclusive result
 *    in a single fused kernel pass
 */

#include "scan.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace scan {

// ─── bank-conflict avoidance ─────────────────────────────────────────────────
// Padding one int per 32 ints avoids the 32-way bank conflict that would
// otherwise occur at every power-of-two stride in the down-sweep.
static constexpr int NUM_BANKS    = 32;
static constexpr int LOG_NUM_BANKS = 5;
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

// Shared memory size needed for one block of ELEMENTS_PER_BLOCK elements
// including padding slots.
static constexpr int SMEM_SIZE =
    ELEMENTS_PER_BLOCK + (ELEMENTS_PER_BLOCK >> LOG_NUM_BANKS);

// ─── kernel: block-level exclusive scan ──────────────────────────────────────
//
// Each block scans exactly ELEMENTS_PER_BLOCK (= 2*BLOCK_SIZE) elements.
// The caller is responsible for padding the input to a multiple of
// ELEMENTS_PER_BLOCK with zeros.
//
// block_sums[blockIdx.x] receives the total sum of this block's input so
// the caller can recurse over block sums for the multi-level scan.
//
__global__ void blelloch_block_scan_kernel(
    const value_t* __restrict__ d_in,
          value_t* __restrict__ d_out,
          value_t* __restrict__ block_sums,   // one entry per block
    int n_padded                               // padded total length
) {
    extern __shared__ value_t smem[];  // SMEM_SIZE ints

    const int tid   = threadIdx.x;
    const int base  = blockIdx.x * ELEMENTS_PER_BLOCK;

    // ── 1. Coalesced load with bank-conflict-free indexing ────────────────
    int ai = tid;
    int bi = tid + BLOCK_SIZE;
    int ai_s = ai + CONFLICT_FREE_OFFSET(ai);
    int bi_s = bi + CONFLICT_FREE_OFFSET(bi);

    smem[ai_s] = (base + ai < n_padded) ? d_in[base + ai] : 0;
    smem[bi_s] = (base + bi < n_padded) ? d_in[base + bi] : 0;
    __syncthreads();

    // ── 2. Up-sweep (reduce) ──────────────────────────────────────────────
    int offset = 1;
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            ai2 += CONFLICT_FREE_OFFSET(ai2);
            bi2 += CONFLICT_FREE_OFFSET(bi2);
            smem[bi2] += smem[ai2];
        }
        offset <<= 1;
    }
    __syncthreads();

    // ── 3. Save block total, clear root ──────────────────────────────────
    if (tid == 0) {
        int root = ELEMENTS_PER_BLOCK - 1 + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK - 1);
        block_sums[blockIdx.x] = smem[root];
        smem[root] = 0;  // exclusive scan: identity at root
    }
    __syncthreads();

    // ── 4. Down-sweep ─────────────────────────────────────────────────────
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            ai2 += CONFLICT_FREE_OFFSET(ai2);
            bi2 += CONFLICT_FREE_OFFSET(bi2);
            value_t tmp = smem[ai2];
            smem[ai2]   = smem[bi2];
            smem[bi2]  += tmp;
        }
    }
    __syncthreads();

    // ── 5. Coalesced store ────────────────────────────────────────────────
    if (base + ai < n_padded) d_out[base + ai] = smem[ai_s];
    if (base + bi < n_padded) d_out[base + bi] = smem[bi_s];
}

// ─── kernel: add block offsets ───────────────────────────────────────────────
// After scanning the block_sums array, add each block's offset back to every
// element in that block.  One thread per element.
__global__ void add_block_offsets_kernel(
          value_t* __restrict__ d_inout,
    const value_t* __restrict__ block_offsets,
    int n_padded
) {
    const int idx = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x;
    const value_t offset = block_offsets[blockIdx.x];

    if (idx           < n_padded) d_inout[idx]           += offset;
    if (idx + BLOCK_SIZE < n_padded) d_inout[idx + BLOCK_SIZE] += offset;
}

// ─── kernel: fused exclusive→inclusive conversion ────────────────────────────
// inclusive[i] = exclusive[i] + input[i]
// Done in a single pass so we avoid an extra global-memory round-trip.
__global__ void exclusive_to_inclusive_kernel(
    const value_t* __restrict__ d_exclusive,
    const value_t* __restrict__ d_input,
          value_t* __restrict__ d_inclusive,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_inclusive[idx] = d_exclusive[idx] + d_input[idx];
    }
}

// ─── host helper: recursive multi-level scan ─────────────────────────────────
// Scans d_in (length n_padded, already padded) into d_out.
// block_sums must be pre-allocated with ceil(n_padded/ELEMENTS_PER_BLOCK) ints.
static void recursive_gpu_scan(
    const value_t* d_in,
          value_t* d_out,
    int            n_padded,    // multiple of ELEMENTS_PER_BLOCK
    cudaStream_t   stream
) {
    const int num_blocks = n_padded / ELEMENTS_PER_BLOCK;

    // Allocate block sums on device
    value_t* d_block_sums = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_block_sums, num_blocks * sizeof(value_t), stream));

    // Step 1: scan each block, write block totals to d_block_sums
    blelloch_block_scan_kernel<<<num_blocks, BLOCK_SIZE,
                                  SMEM_SIZE * sizeof(value_t), stream>>>(
        d_in, d_out, d_block_sums, n_padded
    );

    // Step 2: scan the block sums (recurse or do it in a single block)
    if (num_blocks > 1) {
        // Pad block_sums to next multiple of ELEMENTS_PER_BLOCK
        int bs_padded = ((num_blocks + ELEMENTS_PER_BLOCK - 1)
                         / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;

        value_t* d_bs_padded = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_bs_padded, bs_padded * sizeof(value_t), stream));
        CUDA_CHECK(cudaMemsetAsync(d_bs_padded, 0, bs_padded * sizeof(value_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_bs_padded, d_block_sums,
                                   num_blocks * sizeof(value_t),
                                   cudaMemcpyDeviceToDevice, stream));

        value_t* d_bs_scanned = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_bs_scanned, bs_padded * sizeof(value_t), stream));

        recursive_gpu_scan(d_bs_padded, d_bs_scanned, bs_padded, stream);

        // Step 3: add block offsets back to every element
        add_block_offsets_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_out, d_bs_scanned, n_padded
        );

        CUDA_CHECK(cudaFreeAsync(d_bs_padded,   stream));
        CUDA_CHECK(cudaFreeAsync(d_bs_scanned,  stream));
    }

    CUDA_CHECK(cudaFreeAsync(d_block_sums, stream));
}

// ─── public API ──────────────────────────────────────────────────────────────
void gpu_scan(
    const std::vector<value_t>& input,
    std::vector<value_t>&       output,
    ScanType                    scan_type
) {
    const int n = static_cast<int>(input.size());
    output.resize(n);
    if (n == 0) return;

    // Pad to next multiple of ELEMENTS_PER_BLOCK
    const int n_padded = ((n + ELEMENTS_PER_BLOCK - 1)
                          / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate device buffers
    value_t *d_in      = nullptr;
    value_t *d_out     = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_in,  n_padded * sizeof(value_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_out, n_padded * sizeof(value_t), stream));

    // Zero-pad and upload input
    CUDA_CHECK(cudaMemsetAsync(d_in, 0, n_padded * sizeof(value_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_in, input.data(),
                               n * sizeof(value_t),
                               cudaMemcpyHostToDevice, stream));

    // Run the exclusive scan
    recursive_gpu_scan(d_in, d_out, n_padded, stream);

    if (scan_type == ScanType::Inclusive) {
        // Fused exclusive → inclusive in one extra kernel pass
        const int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        exclusive_to_inclusive_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_out, d_in, d_out, n
        );
    }

    // Download result
    CUDA_CHECK(cudaMemcpyAsync(output.data(), d_out,
                               n * sizeof(value_t),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_in,  stream));
    CUDA_CHECK(cudaFreeAsync(d_out, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void cpu_scan(
    const std::vector<value_t>& input,
    std::vector<value_t>&       output,
    ScanType                    scan_type
) {
    const int n = static_cast<int>(input.size());
    output.resize(n);
    value_t running = 0;
    if (scan_type == ScanType::Exclusive) {
        for (int i = 0; i < n; ++i) {
            output[i] = running;
            running  += input[i];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            running   += input[i];
            output[i]  = running;
        }
    }
}

bool verify(
    const std::vector<value_t>& expected,
    const std::vector<value_t>& actual,
    std::size_t*                mismatch_idx
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
