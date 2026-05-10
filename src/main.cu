/*
 * cuda-prefix-sum  —  main.cu
 *
 * CLI benchmark harness.
 * Usage:
 *   ./prefix_scan [--n N] [--repeats R] [--scan-type exclusive|inclusive]
 *                 [--test] [--csv]
 */

#include "scan.cuh"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace {

// ─── options ──────────────────────────────────────────────────────────────────
struct Options {
    std::size_t      n         = 1u << 24;   // 16 M elements default
    int              repeats   = 10;
    int              warmups   = 3;
    scan::ScanType   scan_type = scan::ScanType::Exclusive;
    bool             csv       = false;
    bool             run_tests = false;
};

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n\n"
        << "  --n <N>             Input size (default: 2^24 = 16 777 216)\n"
        << "  --repeats <R>       Timed repetitions (default: 10)\n"
        << "  --warmups <W>       Warm-up runs (default: 3)\n"
        << "  --scan-type <S>     exclusive | inclusive (default: exclusive)\n"
        << "  --csv               Print a single CSV row\n"
        << "  --test              Run correctness suite and exit\n"
        << "  --help\n";
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--help")   { print_usage(argv[0]); std::exit(0); }
        else if (a == "--csv")    { o.csv = true; }
        else if (a == "--test")   { o.run_tests = true; }
        else if (a == "--n"           && i+1 < argc) o.n         = std::stoull(argv[++i]);
        else if (a == "--repeats"     && i+1 < argc) o.repeats   = std::stoi(argv[++i]);
        else if (a == "--warmups"     && i+1 < argc) o.warmups   = std::stoi(argv[++i]);
        else if (a == "--scan-type"   && i+1 < argc) {
            std::string s = argv[++i];
            if (s == "inclusive") o.scan_type = scan::ScanType::Inclusive;
            else if (s != "exclusive")
                throw std::invalid_argument("scan-type must be exclusive or inclusive");
        }
        else { std::cerr << "Unknown option: " << a << "\n"; print_usage(argv[0]); std::exit(1); }
    }
    return o;
}

// ─── input generation ─────────────────────────────────────────────────────────
std::vector<scan::value_t> make_input(std::size_t n) {
    std::vector<scan::value_t> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<scan::value_t>(((i * 13) + 7) % 29);
    return v;
}

// ─── timing ──────────────────────────────────────────────────────────────────
// Uses CUDA events for GPU timing (more accurate than host chrono for GPU work)
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { cudaEventRecord(start_, 0); }
    float stop_ms() {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    std::size_t m = v.size() / 2;
    return (v.size() % 2 == 1) ? v[m] : (v[m-1] + v[m]) / 2.0;
}

// ─── correctness tests ────────────────────────────────────────────────────────
bool run_tests() {
    struct Case { std::string name; std::vector<scan::value_t> input; };
    std::vector<Case> cases = {
        {"empty",            {}},
        {"single",           {42}},
        {"small_pow2",       {3,1,7,0,4,1,6,3}},
        {"small_non_pow2",   {5,0,2,9,1,4,8}},
        {"one_block",        make_input(2048)},
        {"multi_block",      make_input(100003)},
        {"large",            make_input(1 << 20)},
        {"very_large",       make_input(1 << 24)},
    };

    std::size_t checks = 0, failures = 0;
    for (auto scan_type : {scan::ScanType::Exclusive, scan::ScanType::Inclusive}) {
        for (auto& c : cases) {
            std::vector<scan::value_t> expected, actual;
            scan::cpu_scan(c.input, expected, scan_type);
            scan::gpu_scan(c.input, actual,   scan_type);

            ++checks;
            std::size_t idx = 0;
            if (!scan::verify(expected, actual, &idx)) {
                ++failures;
                std::cout << "[FAIL] "
                    << (scan_type == scan::ScanType::Exclusive ? "excl" : "incl")
                    << "  case=" << c.name
                    << "  n=" << c.input.size()
                    << "  idx=" << idx
                    << "  expected=" << expected[idx]
                    << "  actual="   << actual[idx]
                    << "\n";
            }
        }
    }
    if (failures == 0)
        std::cout << "Correctness: PASS (" << checks << " checks)\n";
    else
        std::cout << "Correctness: FAIL (" << failures << "/" << checks << " failed)\n";
    return failures == 0;
}

} // namespace

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        // Print GPU info
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (!opt.csv) {
            std::cout << "GPU: " << prop.name
                      << "  SMs=" << prop.multiProcessorCount
                      << "  VRAM=" << (prop.totalGlobalMem >> 20) << " MB"
                      << "  peak_bw="
                      << std::fixed << std::setprecision(1)
                      << (2.0 * prop.memoryClockRate * 1e3 *
                          (prop.memoryBusWidth / 8) / 1e9)
                      << " GB/s\n\n";
        }

        if (opt.run_tests) return run_tests() ? 0 : 1;

        const auto input  = make_input(opt.n);
        const std::string type_str =
            opt.scan_type == scan::ScanType::Exclusive ? "exclusive" : "inclusive";

        // ── CPU reference ────────────────────────────────────────────────────
        std::vector<scan::value_t> cpu_out;
        std::vector<double> cpu_times;
        for (int r = 0; r < opt.warmups + opt.repeats; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            scan::cpu_scan(input, cpu_out, opt.scan_type);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (r >= opt.warmups)
                cpu_times.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double cpu_median = median_of(cpu_times);

        // ── GPU benchmark ─────────────────────────────────────────────────
        std::vector<scan::value_t> gpu_out;
        std::vector<double> gpu_times;
        GpuTimer timer;

        for (int r = 0; r < opt.warmups + opt.repeats; ++r) {
            timer.start();
            scan::gpu_scan(input, gpu_out, opt.scan_type);
            float ms = timer.stop_ms();
            if (r >= opt.warmups)
                gpu_times.push_back(static_cast<double>(ms));
        }
        double gpu_median = median_of(gpu_times);

        // ── verify ───────────────────────────────────────────────────────
        std::size_t mismatch = 0;
        bool ok = scan::verify(cpu_out, gpu_out, &mismatch);

        double speedup    = (gpu_median > 0) ? cpu_median / gpu_median : 0.0;
        double bytes      = static_cast<double>(opt.n) * sizeof(scan::value_t);
        // exclusive scan: 2 reads + 1 write per element (approx)
        double throughput = (3.0 * bytes / 1e9) / (gpu_median / 1e3); // GB/s

        if (opt.csv) {
            // header + row
            std::cout
                << "n,scan_type,cpu_median_ms,gpu_median_ms,speedup,"
                << "throughput_GBs,status\n"
                << opt.n << ","
                << type_str << ","
                << std::fixed << std::setprecision(4)
                << cpu_median << ","
                << gpu_median << ","
                << speedup    << ","
                << throughput << ","
                << (ok ? "OK" : "FAIL") << "\n";
        } else {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Scan type       : " << type_str            << "\n";
            std::cout << "Input size      : " << opt.n               << " elements\n";
            std::cout << "Repeats/warmups : " << opt.repeats << " / " << opt.warmups << "\n\n";
            std::cout << "CPU median (ms) : " << cpu_median          << "\n";
            std::cout << "GPU median (ms) : " << gpu_median          << "\n";
            std::cout << "Speedup         : " << speedup             << "x\n";
            std::cout << "Throughput      : " << throughput          << " GB/s\n";
            std::cout << "Correctness     : " << (ok ? "PASS" : "FAIL") << "\n";
            if (!ok) {
                std::cout << "  mismatch idx  : " << mismatch        << "\n";
                std::cout << "  expected      : " << cpu_out[mismatch] << "\n";
                std::cout << "  actual        : " << gpu_out[mismatch] << "\n";
            }
        }
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
