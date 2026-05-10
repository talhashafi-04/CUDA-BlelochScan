#!/usr/bin/env bash
# cuda-prefix-sum-v3  —  run_benchmarks.sh
#
# Full sweep: 4 algos x 5 sizes x 2 scan types -> results_v3.csv
# v3 CSV: gpu_compute_ms, gpu_e2e_ms, compute_speedup, e2e_speedup

set -e
BINARY="./prefix_scan_v3"
OUT="results_v3.csv"
REPEATS=20
WARMUPS=5

if [ ! -f "$BINARY" ]; then
    echo "Binary $BINARY not found. Run 'make' first."
    exit 1
fi

echo "n,scan_type,algo,cpu_median_ms,gpu_compute_ms,gpu_e2e_ms,compute_speedup,e2e_speedup,throughput_GBs,status" > "$OUT"

SIZES=(10000 100000 1000000 10000000 100000000)
TYPES=(exclusive inclusive)
ALGOS=(blelloch single cub optimized)

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
