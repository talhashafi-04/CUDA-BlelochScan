#!/usr/bin/env python3
"""
cuda-prefix-sum-v2 — plot_results.py

Reads results_v2.csv and produces comparison charts showing:
  - Speedup (GPU vs CPU) for each algorithm across array sizes
  - Throughput (GB/s) for each algorithm
  - Speedup ratio: single_pass / blelloch  (how much better the fix is)

Usage:
    python3 scripts/plot_results.py [results_v2.csv]
"""

import sys
import csv
import os
import math

# Try matplotlib; fall back to a text summary if not available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

CSV_FILE = sys.argv[1] if len(sys.argv) > 1 else "results_v2.csv"
OUT_DIR  = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── load data ────────────────────────────────────────────────────────────────
rows = []
with open(CSV_FILE) as f:
    for row in csv.DictReader(f):
        rows.append({
            "n":             int(row["n"]),
            "scan_type":     row["scan_type"],
            "algo":          row["algo"],
            "cpu_ms":        float(row["cpu_median_ms"]),
            "gpu_ms":        float(row["gpu_median_ms"]),
            "speedup":       float(row["speedup"]),
            "throughput":    float(row["throughput_GBs"]),
            "ok":            row["status"] == "OK",
        })

algos      = ["blelloch", "single_pass", "cub"]
algo_label = {"blelloch": "Blelloch (v1 recursive)",
              "single_pass": "Single-Pass Look-back (v2)",
              "cub": "CUB::DeviceScan (reference)"}
colors     = {"blelloch": "#e74c3c", "single_pass": "#2ecc71", "cub": "#3498db"}
scan_types = ["exclusive", "inclusive"]

def get(scan_type, algo, field):
    pts = [(r["n"], r[field]) for r in rows
           if r["scan_type"] == scan_type and r["algo"] == algo]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]

if not HAS_MPL:
    # ─── text summary ─────────────────────────────────────────────────────────
    print(f"\n{'n':>12}  {'algo':>15}  {'type':>10}  {'speedup':>8}  {'GB/s':>8}")
    print("-" * 60)
    for r in sorted(rows, key=lambda x: (x["scan_type"], x["algo"], x["n"])):
        print(f"{r['n']:>12}  {r['algo']:>15}  {r['scan_type']:>10}  "
              f"{r['speedup']:>8.3f}  {r['throughput']:>8.2f}")
    sys.exit(0)

# ─── plotting ─────────────────────────────────────────────────────────────────
STYLE = {
    "font.family":     "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "figure.dpi":      150,
}
plt.rcParams.update(STYLE)

for st in scan_types:
    # ── 1. Speedup comparison ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in algos:
        xs, ys = get(st, algo, "speedup")
        if xs:
            ax.plot(xs, ys, "o-", color=colors[algo],
                    label=algo_label[algo], linewidth=2, markersize=6)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="CPU baseline (1×)")
    ax.set_xscale("log")
    ax.set_xlabel("Array size (n)", fontsize=12)
    ax.set_ylabel("Speedup  (CPU time / GPU time)", fontsize=12)
    ax.set_title(f"GPU Speedup vs CPU — {st} scan\nBlelloch vs Single-Pass vs CUB", fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}" if v < 1e6 else f"{v/1e6:.0f}M"))
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"speedup_{st}.svg")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    # ── 2. Throughput comparison ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in algos:
        xs, ys = get(st, algo, "throughput")
        if xs:
            ax.plot(xs, ys, "s-", color=colors[algo],
                    label=algo_label[algo], linewidth=2, markersize=6)

    ax.set_xscale("log")
    ax.set_xlabel("Array size (n)", fontsize=12)
    ax.set_ylabel("Throughput (GB/s)", fontsize=12)
    ax.set_title(f"GPU Throughput — {st} scan\n(T4 peak memory bandwidth ≈ 320 GB/s)", fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}" if v < 1e6 else f"{v/1e6:.0f}M"))
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"throughput_{st}.svg")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    # ── 3. Improvement ratio: single_pass / blelloch ──────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    xb, yb = get(st, "blelloch",   "gpu_ms")
    xs, ys = get(st, "single_pass","gpu_ms")
    xc, yc = get(st, "cub",        "gpu_ms")

    # align on common n
    nb = dict(zip(xb, yb))
    ns = dict(zip(xs, ys))
    nc = dict(zip(xc, yc))
    ns_keys = sorted(set(nb) & set(ns) & set(nc))

    ratios_s = [nb[n] / ns[n] for n in ns_keys]
    ratios_c = [nb[n] / nc[n] for n in ns_keys]

    ax.plot(ns_keys, ratios_s, "o-", color=colors["single_pass"],
            label="Single-Pass vs Blelloch", linewidth=2, markersize=7)
    ax.plot(ns_keys, ratios_c, "^-", color=colors["cub"],
            label="CUB vs Blelloch", linewidth=2, markersize=7)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Array size (n)", fontsize=12)
    ax.set_ylabel("Speedup ratio vs Blelloch (v1)", fontsize=12)
    ax.set_title(f"How much faster is v2 than v1? — {st} scan", fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}" if v < 1e6 else f"{v/1e6:.0f}M"))
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"improvement_{st}.svg")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

print("\nAll charts written to", OUT_DIR)
