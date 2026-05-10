/*
 * cuda-prefix-sum-v2  —  main.cu
 *
 * Benchmarks and compares three scan implementations:
 *   blelloch   — v1 recursive (slow baseline)
 *   single     — v2 single-pass decoupled look-back (optimised)
 *   cub        — CUB::DeviceScan (production ceiling reference)
 *
 * Usage:
 *   ./prefix_scan [--n N] [--repeats R] [--warmups W]
 *                 [--scan-type exclusive|inclusive]
 *                 [--algo blelloch|single|cub|all]
 *                 [--test] [--csv]
 */

#include "scan.cuh"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace {

struct Options {
    std::size_t    n         = 1u << 24;    // 16 M default
    int            repeats   = 10;
    int            warmups   = 3;
    scan::ScanType scan_type = scan::ScanType::Exclusive;
    scan::Algorithm algo     = scan::Algorithm::SinglePass;
    bool           all_algos = false;
    bool           csv       = false;
    bool           run_tests = false;
};

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n\n"
        << "  --n <N>               Input size (default 2^24)\n"
        << "  --repeats <R>         Timed repetitions (default 10)\n"
        << "  --warmups <W>         Warm-up runs (default 3)\n"
        << "  --scan-type <S>       exclusive | inclusive\n"
        << "  --algo <A>            blelloch | single | cub | all\n"
        << "  --csv                 Output CSV row(s)\n"
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
        else if (a == "--n"         && i+1 < argc) o.n       = std::stoull(argv[++i]);
        else if (a == "--repeats"   && i+1 < argc) o.repeats = std::stoi(argv[++i]);
        else if (a == "--warmups"   && i+1 < argc) o.warmups = std::stoi(argv[++i]);
        else if (a == "--scan-type" && i+1 < argc) {
            std::string s = argv[++i];
            if (s == "inclusive") o.scan_type = scan::ScanType::Inclusive;
        }
        else if (a == "--algo" && i+1 < argc) {
            std::string s = argv[++i];
            if      (s == "blelloch") o.algo = scan::Algorithm::Blelloch;
            else if (s == "single")   o.algo = scan::Algorithm::SinglePass;
            else if (s == "cub")      o.algo = scan::Algorithm::CUB;
            else if (s == "all")      o.all_algos = true;
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

// CUDA event timer
struct EvtTimer {
    cudaEvent_t s, e;
    EvtTimer()  { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~EvtTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void   start()   { cudaEventRecord(s, 0); }
    float  stop_ms() {
        cudaEventRecord(e, 0); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms;
    }
};

double median_of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    std::size_t m = v.size()/2;
    return (v.size()%2) ? v[m] : (v[m-1]+v[m])/2.0;
}

// ─── correctness suite ────────────────────────────────────────────────────────
bool run_tests() {
    auto make = [](int n){ return make_input(n); };

    struct Case { std::string name; std::vector<scan::value_t> input; };
    std::vector<Case> cases = {
        {"empty",          {}},
        {"single",         {42}},
        {"small_pow2",     {3,1,7,0,4,1,6,3}},
        {"small_non_pow2", {5,0,2,9,1}},
        {"one_tile",       make(4096)},
        {"two_tiles",      make(4097)},
        {"multi_tile",     make(100003)},
        {"large",          make(1<<20)},
        {"very_large",     make(1<<24)},
    };

    const std::vector<scan::Algorithm> algos = {
        scan::Algorithm::Blelloch,
        scan::Algorithm::SinglePass,
        scan::Algorithm::CUB,
    };
    const std::string algo_names[] = {"blelloch","single","cub"};

    int checks = 0, failures = 0;
    for (auto st : {scan::ScanType::Exclusive, scan::ScanType::Inclusive}) {
        for (std::size_t ai = 0; ai < algos.size(); ++ai) {
            for (auto& c : cases) {
                std::vector<scan::value_t> expected, actual;
                scan::cpu_scan(c.input, expected, st);
                scan::gpu_scan(c.input, actual,   st, algos[ai]);
                ++checks;
                std::size_t idx = 0;
                if (!scan::verify(expected, actual, &idx)) {
                    ++failures;
                    std::cout << "[FAIL] algo=" << algo_names[ai]
                              << " type=" << (st==scan::ScanType::Exclusive?"excl":"incl")
                              << " case=" << c.name
                              << " idx=" << idx
                              << " exp=" << expected[idx]
                              << " got=" << actual[idx] << "\n";
                }
            }
        }
    }
    std::cout << "Correctness: " << (failures==0?"PASS":"FAIL")
              << " (" << checks << " checks, " << failures << " failures)\n";
    return failures == 0;
}

// ─── single benchmark run ─────────────────────────────────────────────────────
struct Result {
    std::string algo_name;
    double cpu_ms, gpu_ms, speedup, throughput_gbs;
    bool ok;
};

Result bench(
    const std::vector<scan::value_t>& input,
    scan::ScanType st,
    scan::Algorithm algo,
    const std::string& algo_name,
    int warmups, int repeats
) {
    std::vector<scan::value_t> cpu_out, gpu_out;
    std::vector<double> cpu_times, gpu_times;
    EvtTimer timer;

    // CPU
    for (int r = 0; r < warmups + repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        scan::cpu_scan(input, cpu_out, st);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (r >= warmups)
            cpu_times.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
    }

    // GPU
    for (int r = 0; r < warmups + repeats; ++r) {
        timer.start();
        scan::gpu_scan(input, gpu_out, st, algo);
        float ms = timer.stop_ms();
        if (r >= warmups) gpu_times.push_back(ms);
    }

    double cpu_ms  = median_of(cpu_times);
    double gpu_ms  = median_of(gpu_times);
    double speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0;
    double bytes   = static_cast<double>(input.size()) * sizeof(scan::value_t);
    double tput    = (3.0 * bytes / 1e9) / (gpu_ms / 1e3);

    std::size_t idx = 0;
    bool ok = scan::verify(cpu_out, gpu_out, &idx);

    return {algo_name, cpu_ms, gpu_ms, speedup, tput, ok};
}

} // namespace

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (!opt.csv) {
            double peak_bw = 2.0 * prop.memoryClockRate * 1e3 *
                             (prop.memoryBusWidth / 8) / 1e9;
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
            };
        } else {
            std::string name = (opt.algo == scan::Algorithm::Blelloch)   ? "blelloch"    :
                               (opt.algo == scan::Algorithm::SinglePass) ? "single_pass" : "cub";
            algos_to_run.push_back({opt.algo, name});
        }

        if (opt.csv) {
            std::cout << "n,scan_type,algo,cpu_median_ms,gpu_median_ms,"
                         "speedup,throughput_GBs,status\n";
        }

        for (auto& [algo, name] : algos_to_run) {
            auto r = bench(input, opt.scan_type, algo, name,
                           opt.warmups, opt.repeats);
            if (opt.csv) {
                std::cout << std::fixed << std::setprecision(4)
                    << opt.n << "," << type_str << "," << name << ","
                    << r.cpu_ms << "," << r.gpu_ms << ","
                    << r.speedup << "," << r.throughput_gbs << ","
                    << (r.ok ? "OK" : "FAIL") << "\n";
            } else {
                std::cout << "─────────────────────────────────\n";
                std::cout << "Algorithm       : " << name       << "\n";
                std::cout << "Scan type       : " << type_str   << "\n";
                std::cout << "Input size      : " << opt.n      << "\n";
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "CPU median (ms) : " << r.cpu_ms   << "\n";
                std::cout << "GPU median (ms) : " << r.gpu_ms   << "\n";
                std::cout << "Speedup         : " << r.speedup  << "x\n";
                std::cout << "Throughput      : " << r.throughput_gbs << " GB/s\n";
                std::cout << "Correctness     : " << (r.ok ? "PASS" : "FAIL") << "\n";
            }
        }
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
