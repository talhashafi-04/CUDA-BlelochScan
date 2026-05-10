#pragma once
/*
 * cuda-prefix-sum-v2  —  scan.cuh
 *
 * Three GPU scan implementations, from naive to production-grade:
 *
 *  1. blelloch_scan      — Multi-level recursive Blelloch (v1, the slow one).
 *                          Kept for comparison. Suffers from O(log n) kernel
 *                          launches, each carrying ~1 ms host overhead.
 *
 *  2. single_pass_scan   — Decoupled look-back single-pass scan.
 *                          Merrill & Garland (2016) "Single-pass Parallel
 *                          Prefix Scan with Decoupled Look-back".
 *                          ONE kernel launch regardless of n.  No host-side
 *                          recursion.  This is what CUB/Thrust actually use.
 *
 *  3. cub_scan           — Thin wrapper around cub::DeviceScan (reference).
 *                          Use this to see the absolute ceiling.
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace scan {

// ─── value type ──────────────────────────────────────────────────────────────
using value_t = int;

// ─── scan type ────────────────────────────────────────────────────────────────
enum class ScanType { Exclusive, Inclusive };

// ─── algorithm selection ──────────────────────────────────────────────────────
enum class Algorithm {
    Blelloch,    // v1 recursive (baseline)
    SinglePass,  // v2 decoupled look-back (optimised)
    CUB,         // CUB::DeviceScan reference
};

// ─── host API ─────────────────────────────────────────────────────────────────
void gpu_scan(
    const std::vector<value_t>& input,
    std::vector<value_t>&       output,
    ScanType  scan_type = ScanType::Exclusive,
    Algorithm algo      = Algorithm::SinglePass
);

void cpu_scan(
    const std::vector<value_t>& input,
    std::vector<value_t>&       output,
    ScanType scan_type = ScanType::Exclusive
);

bool verify(
    const std::vector<value_t>& expected,
    const std::vector<value_t>& actual,
    std::size_t* mismatch_idx = nullptr
);

// ─── CUDA error helper ────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            throw std::runtime_error(                                            \
                std::string("CUDA error at " __FILE__ ":") +                    \
                std::to_string(__LINE__) + "  " +                               \
                cudaGetErrorString(_e));                                         \
        }                                                                        \
    } while (0)

} // namespace scan
