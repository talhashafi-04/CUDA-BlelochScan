#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

namespace scan {

// ─── configuration ────────────────────────────────────────────────────────────
using value_t = int;

// Block size must be a power of two; 1024 saturates shared memory on sm_75+
static constexpr int BLOCK_SIZE = 1024;

// Each block handles 2×BLOCK_SIZE elements (two elements per thread)
static constexpr int ELEMENTS_PER_BLOCK = 2 * BLOCK_SIZE;

// ─── enums ────────────────────────────────────────────────────────────────────
enum class ScanType { Exclusive, Inclusive };

// ─── host-side API ────────────────────────────────────────────────────────────

// Full GPU prefix scan (exclusive or inclusive).
// Handles arbitrary n via multi-level block-sum recursion.
void gpu_scan(
    const std::vector<value_t>& input,
    std::vector<value_t>&       output,
    ScanType                    scan_type = ScanType::Exclusive
);

// Sequential CPU reference (correctness baseline).
void cpu_scan(
    const std::vector<value_t>& input,
    std::vector<value_t>&       output,
    ScanType                    scan_type = ScanType::Exclusive
);

// Returns true if vectors match element-wise.
bool verify(
    const std::vector<value_t>& expected,
    const std::vector<value_t>& actual,
    std::size_t*                mismatch_idx = nullptr
);

// ─── CUDA error helper ────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(                                           \
                std::string("CUDA error at " __FILE__ ":") +                   \
                std::to_string(__LINE__) + " — " +                             \
                cudaGetErrorString(_e));                                        \
        }                                                                       \
    } while (0)

} // namespace scan
