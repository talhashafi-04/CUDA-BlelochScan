#!/usr/bin/env bash
# cuda-prefix-sum-v2  —  run_benchmarks.sh
#
# Runs all three algorithms over a range of array sizes and scan types.
# Output: results_v2.csv
#
# Usage:
#   chmod +x scripts/run_benchmarks.sh
#   ./scripts/run_benchmarks.sh

set -e
BINARY="./prefix_scan_v2"
OUT="results_v2.csv"
REPEATS=10
WARMUPS=3

if [ ! -f "$BINARY" ]; then
    echo "Binary $BINARY not found. Run 'make' first."
    exit 1
fi

# Write CSV header
echo "n,scan_type,algo,cpu_median_ms,gpu_median_ms,speedup,throughput_GBs,status" > "$OUT"

SIZES=(10000 100000 1000000 10000000 100000000)
TYPES=(exclusive inclusive)
ALGOS=(blelloch single cub)

total=$(( ${#SIZES[@]} * ${#TYPES[@]} * ${#ALGOS[@]} ))
done=0

for size in "${SIZES[@]}"; do
    for type in "${TYPES[@]}"; do
        for algo in "${ALGOS[@]}"; do
            done=$((done + 1))
            echo -ne "\r[$done/$total] n=$size type=$type algo=$algo     "
            $BINARY --n "$size" --scan-type "$type" --algo "$algo" \
                    --repeats "$REPEATS" --warmups "$WARMUPS" --csv \
                | tail -n 1 >> "$OUT"
        done
    done
done

echo ""
echo "Done. Results written to $OUT"
