# ─── cuda-prefix-sum Makefile ─────────────────────────────────────────────────
NVCC      ?= nvcc
# sm_75  = Turing  (T4, RTX 2080)
# sm_86  = Ampere  (A100, RTX 3090)
# sm_89  = Ada     (RTX 4090)
# sm_70  = Volta   (V100)
# Colab T4 = sm_75
ARCH      ?= sm_75
NVCCFLAGS  = -O3 -arch=$(ARCH) -std=c++17 \
             --expt-relaxed-constexpr \
             -Xcompiler -Wall,-Wextra

TARGET    = prefix_scan
SRC       = src/main.cu src/scan.cu
HDR       = src/scan.cuh

.PHONY: all clean test run benchmark

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@

# correctness suite
test: $(TARGET)
	./$(TARGET) --test

# quick smoke run
run: $(TARGET)
	./$(TARGET) --n 16777216 --repeats 10 --warmups 3 --scan-type exclusive

# full benchmark sweep — writes results.csv
benchmark: $(TARGET)
	./scripts/run_benchmarks.sh

clean:
	rm -f $(TARGET) results.csv

# ── arch helpers ───────────────────────────────────────────────────────────────
# Detect GPU arch automatically (requires nvcc + a GPU present)
detect-arch:
	@$(NVCC) -arch=native --run src/detect_arch.cu -o /dev/null 2>/dev/null || \
	 echo "Could not auto-detect arch; set ARCH=sm_75 for T4"

# Build for multiple arches (PTX fallback for unknown GPUs)
fat: $(SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) \
	  -gencode arch=compute_70,code=sm_70 \
	  -gencode arch=compute_75,code=sm_75 \
	  -gencode arch=compute_80,code=sm_80 \
	  -gencode arch=compute_86,code=sm_86 \
	  -gencode arch=compute_75,code=compute_75 \
	  $(SRC) -o $(TARGET)_fat
