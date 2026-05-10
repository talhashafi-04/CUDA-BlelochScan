#!/usr/bin/env bash
# run_benchmarks.sh — sweep sizes × scan types, emit results.csv

set -euo pipefail

BINARY="${BINARY:-./prefix_scan}"
SIZES="${SIZES:-10000 100000 1000000 10000000 100000000}"
REPEATS="${REPEATS:-10}"
WARMUPS="${WARMUPS:-3}"
OUT="${OUT:-results.csv}"

if [[ ! -x "$BINARY" ]]; then
    echo "Binary $BINARY not found. Run 'make' first." >&2
    exit 1
fi

echo "n,scan_type,cpu_median_ms,gpu_median_ms,speedup,throughput_GBs,status" > "$OUT"

for scan_type in exclusive inclusive; do
    for n in $SIZES; do
        echo -n "  n=$n  scan_type=$scan_type  ... "
        # --csv prints header + row; we strip the header with tail -n +2
        "$BINARY" \
            --n "$n" \
            --repeats "$REPEATS" \
            --warmups "$WARMUPS" \
            --scan-type "$scan_type" \
            --csv \
        | tail -n +2 >> "$OUT"
        echo "done"
    done
done

echo ""
echo "Results written to $OUT"
echo ""
column -t -s, "$OUT" 2>/dev/null || cat "$OUT"
