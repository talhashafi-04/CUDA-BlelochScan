# ─── cuda-prefix-sum-v2 Makefile ──────────────────────────────────────────────
#
# GPU arch targets:
#   sm_75  = Turing  (T4, RTX 2080)   ← Colab free tier
#   sm_80  = Ampere  (A100 40GB)      ← Colab Pro
#   sm_86  = Ampere  (RTX 3090)
#   sm_89  = Ada     (RTX 4090)
#   sm_70  = Volta   (V100)
#
# Usage:
#   make            # build for sm_75 (T4 default)
#   make ARCH=sm_80 # build for A100
#   make test       # correctness suite
#   make benchmark  # full sweep → results_v2.csv
#   make compare    # run all three algos at 16M

NVCC       ?= nvcc
ARCH       ?= sm_75
NVCCFLAGS   = -O3 -arch=$(ARCH) -std=c++17             \
              --expt-relaxed-constexpr                  \
              -Xcompiler -Wall,-Wextra                  \
              --generate-line-info                      \
              -DNDEBUG

TARGET     = prefix_scan_v2
SRC        = src/main.cu src/scan.cu
HDR        = src/scan.cuh

.PHONY: all clean test run benchmark compare fat

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@

test: $(TARGET)
	./$(TARGET) --test

run: $(TARGET)
	./$(TARGET) --n 16777216 --repeats 10 --warmups 3

compare: $(TARGET)
	@echo "=== Comparing all algorithms at n=16M, exclusive ==="
	./$(TARGET) --n 16777216 --algo all --scan-type exclusive --repeats 10

benchmark: $(TARGET)
	chmod +x scripts/run_benchmarks.sh
	./scripts/run_benchmarks.sh

clean:
	rm -f $(TARGET) results_v2.csv

# Build fat binary for all common arches (useful for sharing)
fat: $(SRC) $(HDR)
	$(NVCC) -O3 -std=c++17 --expt-relaxed-constexpr  \
	  -gencode arch=compute_70,code=sm_70             \
	  -gencode arch=compute_75,code=sm_75             \
	  -gencode arch=compute_80,code=sm_80             \
	  -gencode arch=compute_86,code=sm_86             \
	  -gencode arch=compute_75,code=compute_75        \
	  $(SRC) -o $(TARGET)_fat
