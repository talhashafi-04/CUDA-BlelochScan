#!/usr/bin/env python3
"""plot_results.py — generate speedup and throughput SVG charts from results.csv"""

import csv, sys, os, math
from collections import defaultdict

def read_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                'n':            int(row['n']),
                'scan_type':    row['scan_type'],
                'cpu_ms':       float(row['cpu_median_ms']),
                'gpu_ms':       float(row['gpu_median_ms']),
                'speedup':      float(row['speedup']),
                'throughput':   float(row['throughput_GBs']),
                'status':       row['status'],
            })
    return rows

def svg_chart(title, xlabel, ylabel, series, outpath):
    W, H = 720, 420
    PAD = dict(top=40, right=30, bottom=60, left=70)
    cw = W - PAD['left'] - PAD['right']
    ch = H - PAD['top']  - PAD['bottom']

    all_x = sorted({x for s in series for x,_ in s['points']})
    all_y = [y for s in series for _,y in s['points']]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = 0, max(all_y) * 1.15

    colors = ['#1D9E75','#3B8BD4','#D85A30','#BA7517']

    def px(x): return PAD['left'] + (math.log10(x) - math.log10(x_min)) / \
                       (math.log10(x_max) - math.log10(x_min)) * cw
    def py(y): return PAD['top'] + ch - (y / y_max) * ch

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}">']
    lines.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    lines.append(f'<text x="{W//2}" y="24" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="15" font-weight="500">{title}</text>')

    # axes
    lines.append(f'<line x1="{PAD["left"]}" y1="{PAD["top"]}" '
                 f'x2="{PAD["left"]}" y2="{PAD["top"]+ch}" stroke="#888" stroke-width="1"/>')
    lines.append(f'<line x1="{PAD["left"]}" y1="{PAD["top"]+ch}" '
                 f'x2="{PAD["left"]+cw}" y2="{PAD["top"]+ch}" stroke="#888" stroke-width="1"/>')

    # x ticks (log scale)
    for x in all_x:
        xi = px(x)
        lbl = f'{x:,}' if x < 1_000_000 else f'{x//1_000_000}M' if x >= 1_000_000 else f'{x//1000}K'
        lines.append(f'<line x1="{xi:.1f}" y1="{PAD["top"]+ch}" '
                     f'x2="{xi:.1f}" y2="{PAD["top"]+ch+5}" stroke="#888" stroke-width="1"/>')
        lines.append(f'<text x="{xi:.1f}" y="{PAD["top"]+ch+18}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="11" fill="#555">{lbl}</text>')

    # y ticks
    n_yticks = 5
    for i in range(n_yticks + 1):
        yv = y_max * i / n_yticks
        yi = py(yv)
        lines.append(f'<line x1="{PAD["left"]-4}" y1="{yi:.1f}" '
                     f'x2="{PAD["left"]+cw}" y2="{yi:.1f}" stroke="#e0e0e0" stroke-width="0.5"/>')
        lines.append(f'<text x="{PAD["left"]-8}" y="{yi+4:.1f}" text-anchor="end" '
                     f'font-family="sans-serif" font-size="11" fill="#555">{yv:.1f}</text>')

    # axis labels
    lines.append(f'<text x="{PAD["left"]+cw//2}" y="{H-8}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="12" fill="#333">{xlabel}</text>')
    lines.append(f'<text x="14" y="{PAD["top"]+ch//2}" text-anchor="middle" '
                 f'transform="rotate(-90,14,{PAD["top"]+ch//2})" '
                 f'font-family="sans-serif" font-size="12" fill="#333">{ylabel}</text>')

    # series
    for si, s in enumerate(series):
        col = colors[si % len(colors)]
        pts = sorted(s['points'])
        d = ' '.join(f'{"M" if i==0 else "L"}{px(x):.1f},{py(y):.1f}'
                     for i,(x,y) in enumerate(pts))
        lines.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.5" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x,y in pts:
            lines.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" '
                         f'fill="{col}" stroke="#fff" stroke-width="1.5"/>')

        # legend
        lx = PAD['left'] + 12 + si * 140
        ly = PAD['top'] + 10
        lines.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{col}" rx="2"/>')
        lines.append(f'<text x="{lx+16}" y="{ly+10}" font-family="sans-serif" '
                     f'font-size="11" fill="#333">{s["label"]}</text>')

    lines.append('</svg>')
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  wrote {outpath}')

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'results.csv'
    rows = read_csv(path)

    os.makedirs('charts', exist_ok=True)

    for scan_type in ['exclusive', 'inclusive']:
        sub = [r for r in rows if r['scan_type'] == scan_type and r['status'] == 'OK']

        svg_chart(
            title   = f'GPU speedup over CPU — {scan_type} scan',
            xlabel  = 'Array size (n)',
            ylabel  = 'Speedup (×)',
            series  = [{'label': 'GPU / CPU', 'points': [(r['n'], r['speedup']) for r in sub]}],
            outpath = f'charts/speedup_{scan_type}.svg',
        )
        svg_chart(
            title   = f'GPU memory throughput — {scan_type} scan',
            xlabel  = 'Array size (n)',
            ylabel  = 'Throughput (GB/s)',
            series  = [{'label': 'GPU throughput', 'points': [(r['n'], r['throughput']) for r in sub]}],
            outpath = f'charts/throughput_{scan_type}.svg',
        )

if __name__ == '__main__':
    main()
