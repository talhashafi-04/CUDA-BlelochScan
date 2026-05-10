# ─── cuda-prefix-sum-v3 Makefile ──────────────────────────────────────────────
#
# GPU arch targets:
#   sm_75  = Turing  (T4, RTX 2080)   ← Colab free tier  ← DEFAULT
#   sm_80  = Ampere  (A100 40GB)      ← Colab Pro
#   sm_86  = Ampere  (RTX 3090)
#   sm_89  = Ada     (RTX 4090)
#   sm_70  = Volta   (V100)
#
# Usage:
#   make                  # build for sm_75 (T4)
#   make ARCH=sm_80       # build for A100
#   make test             # correctness suite (all 4 algos)
#   make benchmark        # full sweep → results_v3.csv
#   make compare          # all 4 algos at 16M, exclusive
#   make compare_incl     # all 4 algos at 16M, inclusive
#   make sizes            # sweep sizes with optimized only

NVCC       ?= nvcc
ARCH       ?= sm_75
NVCCFLAGS   = -O3 -arch=$(ARCH) -std=c++17            \
              --expt-relaxed-constexpr                 \
              -Xcompiler -Wall,-Wextra                 \
              --generate-line-info                     \
              -DNDEBUG

TARGET     = prefix_scan_v3
SRC        = src/main.cu src/scan.cu
HDR        = src/scan.cuh

.PHONY: all clean test run benchmark compare compare_incl sizes fat

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@

test: $(TARGET)
	./$(TARGET) --test

run: $(TARGET)
	./$(TARGET) --n 16777216 --algo optimized --repeats 20 --warmups 5

compare: $(TARGET)
	@echo "=== All algorithms, n=16M, exclusive ==="
	./$(TARGET) --n 16777216 --algo all --scan-type exclusive --repeats 20 --warmups 5

compare_incl: $(TARGET)
	@echo "=== All algorithms, n=16M, inclusive ==="
	./$(TARGET) --n 16777216 --algo all --scan-type inclusive --repeats 20 --warmups 5

sizes: $(TARGET)
	@echo "=== Optimized: size sweep ==="
	@for n in 10000 100000 1000000 10000000 100000000; do \
	    echo "--- n=$$n ---"; \
	    ./$(TARGET) --n $$n --algo optimized --repeats 20 --warmups 5; \
	done

benchmark: $(TARGET)
	chmod +x scripts/run_benchmarks.sh
	./scripts/run_benchmarks.sh

clean:
	rm -f $(TARGET) results_v3.csv

fat: $(SRC) $(HDR)
	$(NVCC) -O3 -std=c++17 --expt-relaxed-constexpr  \
	  -gencode arch=compute_70,code=sm_70             \
	  -gencode arch=compute_75,code=sm_75             \
	  -gencode arch=compute_80,code=sm_80             \
	  -gencode arch=compute_86,code=sm_86             \
	  -gencode arch=compute_75,code=compute_75        \
	  $(SRC) -o $(TARGET)_fat
