#pragma once
/*
 * cuda-prefix-sum-v3  —  scan.cuh
 *
 * Four GPU scan implementations:
 *  1. blelloch    — v1 recursive (slow baseline, O(log n) kernel launches)
 *  2. single_pass — fixed decoupled look-back (one kernel)
 *  3. cub         — CUB::DeviceScan (production ceiling reference)
 *  4. optimized   — v3 best: warp-shuffle decoupled look-back, BLOCK=1024,
 *                   ITEMS_PT=16 (128-bit vectorised loads), pinned host memory,
 *                   async H↔D overlap, correct timing (device-only vs e2e)
 *
 * v3 key fixes over v2:
 *  FIX-A  single_pass output scatter was wrong: thread_excl is the exclusive
 *          prefix of *thread sums*, not of individual items. The per-item
 *          scatter must do its own local sequential scan starting from the
 *          per-thread exclusive prefix (smem_prefix + thread_excl).
 *  FIX-B  Benchmark methodology: GPU events now time ONLY device compute.
 *          A separate e2e path measures full PCIe round trip.
 *          This is why v2 showed speedup < 1x at every size.
 *  FIX-C  Optimized kernel: BLOCK_SZ=1024 (fills T4 SM), ITEMS_PT=16,
 *          int4 vectorised loads (128-bit = 1 transaction per 4 ints),
 *          warp-shuffle-only intra-block scan (no shared memory bank conflicts),
 *          __threadfence_block() in look-back loop.
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace scan {

using value_t = int;

enum class ScanType  { Exclusive, Inclusive };
enum class Algorithm { Blelloch, SinglePass, CUB, Optimized };

struct BenchResult {
    double cpu_median_ms;
    double gpu_compute_ms;   // device-only, no PCIe
    double gpu_e2e_ms;       // full round trip H->D + compute + D->H
    double compute_speedup;  // cpu / gpu_compute
    double e2e_speedup;      // cpu / gpu_e2e
    double throughput_GBs;   // based on compute only, 3*n*sizeof(value_t)
    bool   correct;
    std::string status;
};

BenchResult run_benchmark(
    const std::vector<value_t>& h_in,
    ScanType  st,
    Algorithm algo,
    int       warmups,
    int       repeats
);

void cpu_scan(
    const std::vector<value_t>& in,
    std::vector<value_t>&       out,
    ScanType scan_type = ScanType::Exclusive
);

bool verify(
    const std::vector<value_t>& expected,
    const std::vector<value_t>& actual,
    std::size_t* mismatch_idx = nullptr
);

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess)                                              \
            throw std::runtime_error(                                       \
                std::string("CUDA error " __FILE__ ":") +                  \
                std::to_string(__LINE__) + " " + cudaGetErrorString(_e));   \
    } while (0)

} // namespace scan
