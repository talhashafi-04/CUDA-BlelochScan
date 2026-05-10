/*
 * cuda-prefix-sum-v3  —  main.cu
 *
 * Benchmarks four scan implementations.  v3 key change: GPU timing is split
 * into TWO columns so the source of speedup is unambiguous:
 *
 *   gpu_compute_ms  — CUDA event time, data already on device (no PCIe)
 *   gpu_e2e_ms      — wall-clock time including H↔D transfers (pinned)
 *
 * Usage:
 *   ./prefix_scan_v3 [--n N] [--repeats R] [--warmups W]
 *                    [--scan-type exclusive|inclusive]
 *                    [--algo blelloch|single|cub|optimized|all]
 *                    [--test] [--csv]
 */

#include "scan.cuh"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace {

struct Options {
    std::size_t    n         = 1u << 24;
    int            repeats   = 20;
    int            warmups   = 5;
    scan::ScanType scan_type = scan::ScanType::Exclusive;
    scan::Algorithm algo     = scan::Algorithm::Optimized;
    bool           all_algos = false;
    bool           csv       = false;
    bool           run_tests = false;
};

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n\n"
        << "  --n <N>               Input size (default 2^24 = 16M)\n"
        << "  --repeats <R>         Timed repetitions (default 20)\n"
        << "  --warmups <W>         Warm-up runs (default 5)\n"
        << "  --scan-type <S>       exclusive | inclusive\n"
        << "  --algo <A>            blelloch | single | cub | optimized | all\n"
        << "  --csv                 Output CSV\n"
        << "  --test                Correctness suite then exit\n"
        << "  --help\n";
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--help")  { print_usage(argv[0]); std::exit(0); }
        else if (a == "--csv")   { o.csv = true; }
        else if (a == "--test")  { o.run_tests = true; }
        else if (a == "--n"         && i+1<argc) o.n       = std::stoull(argv[++i]);
        else if (a == "--repeats"   && i+1<argc) o.repeats = std::stoi(argv[++i]);
        else if (a == "--warmups"   && i+1<argc) o.warmups = std::stoi(argv[++i]);
        else if (a == "--scan-type" && i+1<argc) {
            if (std::string(argv[++i]) == "inclusive")
                o.scan_type = scan::ScanType::Inclusive;
        }
        else if (a == "--algo" && i+1<argc) {
            std::string s = argv[++i];
            if      (s == "blelloch")  o.algo = scan::Algorithm::Blelloch;
            else if (s == "single")    o.algo = scan::Algorithm::SinglePass;
            else if (s == "cub")       o.algo = scan::Algorithm::CUB;
            else if (s == "optimized") o.algo = scan::Algorithm::Optimized;
            else if (s == "all")       o.all_algos = true;
        }
        else { std::cerr << "Unknown: " << a << "\n"; print_usage(argv[0]); std::exit(1); }
    }
    return o;
}

std::vector<scan::value_t> make_input(std::size_t n) {
    std::vector<scan::value_t> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<scan::value_t>(((i * 13) + 7) % 29);
    return v;
}

// ─── correctness suite ────────────────────────────────────────────────────────
static constexpr int TILE_OPT = 1024 * 16;
static constexpr int TILE_SP  = 256  * 8;

bool run_tests() {
    auto make = [](int n){ return make_input(n); };

    struct Case { std::string name; std::vector<scan::value_t> input; };
    std::vector<Case> cases = {
        {"empty",          {}},
        {"single",         {42}},
        {"small_pow2",     {3,1,7,0,4,1,6,3}},
        {"small_non_pow2", {5,0,2,9,1}},
        {"sp_one_tile",    make(TILE_SP)},
        {"sp_two_tiles",   make(TILE_SP + 1)},
        {"opt_one_tile",   make(TILE_OPT)},
        {"opt_two_tiles",  make(TILE_OPT + 1)},
        {"multi_tile",     make(TILE_OPT * 3 + 7777)},
        {"medium",         make(1 << 20)},
    };

    const std::vector<scan::Algorithm> algos = {
        scan::Algorithm::Blelloch,
        scan::Algorithm::SinglePass,
        scan::Algorithm::CUB,
        scan::Algorithm::Optimized,
    };
    const std::string algo_names[] = {"blelloch","single","cub","optimized"};

    int checks = 0, failures = 0;
    for (auto stype : {scan::ScanType::Exclusive, scan::ScanType::Inclusive}) {
        for (std::size_t ai = 0; ai < algos.size(); ++ai) {
            for (auto& c : cases) {
                if (c.input.empty()) {
                    ++checks;
                    continue;  // empty scan trivially correct
                }
                std::vector<scan::value_t> expected;
                scan::cpu_scan(c.input, expected, stype);

                auto res = scan::run_benchmark(c.input, stype, algos[ai], 0, 1);
                ++checks;
                if (!res.correct) {
                    ++failures;
                    std::cout << "[FAIL] algo=" << algo_names[ai]
                              << " type=" << (stype==scan::ScanType::Exclusive?"excl":"incl")
                              << " case=" << c.name << "\n";
                }
            }
        }
    }
    std::cout << "Correctness: " << (failures==0?"PASS":"FAIL")
              << " (" << checks << " checks, " << failures << " failures)\n";
    return failures == 0;
}

} // namespace

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (!opt.csv) {
            double peak_bw = 2.0 * prop.memoryClockRate * 1e3
                           * (prop.memoryBusWidth / 8) / 1e9;
            std::cout << "GPU: " << prop.name
                      << "  SMs=" << prop.multiProcessorCount
                      << "  VRAM=" << (prop.totalGlobalMem >> 20) << " MB"
                      << "  peak_bw=" << std::fixed << std::setprecision(1)
                      << peak_bw << " GB/s\n\n";
        }

        if (opt.run_tests) return run_tests() ? 0 : 1;

        const auto input = make_input(opt.n);
        const std::string type_str =
            (opt.scan_type == scan::ScanType::Exclusive) ? "exclusive" : "inclusive";

        std::vector<std::pair<scan::Algorithm,std::string>> algos_to_run;
        if (opt.all_algos) {
            algos_to_run = {
                {scan::Algorithm::Blelloch,   "blelloch"},
                {scan::Algorithm::SinglePass, "single_pass"},
                {scan::Algorithm::CUB,        "cub"},
                {scan::Algorithm::Optimized,  "optimized"},
            };
        } else {
            std::string name =
                (opt.algo == scan::Algorithm::Blelloch)   ? "blelloch"   :
                (opt.algo == scan::Algorithm::SinglePass) ? "single_pass":
                (opt.algo == scan::Algorithm::CUB)        ? "cub"        : "optimized";
            algos_to_run.push_back({opt.algo, name});
        }

        if (opt.csv) {
            std::cout << "n,scan_type,algo,"
                         "cpu_median_ms,gpu_compute_ms,gpu_e2e_ms,"
                         "compute_speedup,e2e_speedup,"
                         "throughput_GBs,status\n";
        }

        for (auto& [algo, name] : algos_to_run) {
            auto r = scan::run_benchmark(input, opt.scan_type, algo,
                                         opt.warmups, opt.repeats);
            if (opt.csv) {
                std::cout << std::fixed << std::setprecision(4)
                    << opt.n << "," << type_str << "," << name << ","
                    << r.cpu_median_ms   << ","
                    << r.gpu_compute_ms  << ","
                    << r.gpu_e2e_ms      << ","
                    << r.compute_speedup << ","
                    << r.e2e_speedup     << ","
                    << r.throughput_GBs  << ","
                    << r.status          << "\n";
            } else {
                std::cout << "─────────────────────────────────────────\n";
                std::cout << "Algorithm          : " << name       << "\n";
                std::cout << "Scan type          : " << type_str   << "\n";
                std::cout << "Input size         : " << opt.n      << "\n";
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "CPU median (ms)    : " << r.cpu_median_ms   << "\n";
                std::cout << "GPU compute (ms)   : " << r.gpu_compute_ms  << "  ← no PCIe\n";
                std::cout << "GPU e2e (ms)       : " << r.gpu_e2e_ms      << "  ← incl. H↔D\n";
                std::cout << "Compute speedup    : " << r.compute_speedup << "x\n";
                std::cout << "E2E speedup        : " << r.e2e_speedup     << "x\n";
                std::cout << "Throughput         : " << r.throughput_GBs  << " GB/s\n";
                std::cout << "Correctness        : " << r.status           << "\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
