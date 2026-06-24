"""Plot training-progression graphs from a metrics CSV written by exit_training.py.

Each ExIt run writes metrics/<run_tag>.csv (one row per iteration). This renders the key
curves to PNGs. Parsing uses only the stdlib, so the CSV is trivial to graph with any tool
(pandas, Excel, gnuplot...) if you'd rather not use matplotlib.

Run:  python plot_metrics.py                 # newest metrics/*.csv
      python plot_metrics.py metrics/X.csv   # a specific run
"""

import csv
import glob
import os
import sys


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {k: [] for k in rows[0]} if rows else {}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except (ValueError, TypeError):
                cols[k].append(None)
    return cols


def _series(cols, key):
    """Return (iters, values) dropping rows where the metric is blank."""
    xs, ys = [], []
    for it, v in zip(cols["iter"], cols.get(key, [])):
        if v is not None:
            xs.append(it)
            ys.append(v)
    return xs, ys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        files = sorted(glob.glob("metrics/*.csv"), key=os.path.getmtime)
        if not files:
            print("no metrics/*.csv found; run exit_training.py first")
            return
        path = files[-1]
    print(f"plotting {path}")
    cols = load(path)
    if not cols.get("iter"):
        print("empty metrics file")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (pip install matplotlib). CSV is still at", path)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: strength curves (win-rates).
    for key, label in [("vs_random_wr", "vs random"), ("vs_bots_wr", "vs bots (mean)"),
                       ("vs_dragapult", "vs dragapult"), ("vs_abomasnow", "vs abomasnow"),
                       ("vs_iono", "vs iono"), ("selfplay_wr", "self-play")]:
        xs, ys = _series(cols, key)
        if xs:
            ax1.plot(xs, [y * 100 for y in ys], marker=".", label=label)
    ax1.set_xlabel("iteration"); ax1.set_ylabel("win rate (%)")
    ax1.set_title("Strength"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # Right: training losses.
    for key, label in [("policy_ce", "policy CE"), ("value_mse", "value MSE")]:
        xs, ys = _series(cols, key)
        if xs:
            ax2.plot(xs, ys, marker=".", label=label)
    ax2.set_xlabel("iteration"); ax2.set_ylabel("loss")
    ax2.set_title("Distillation losses"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    out = os.path.splitext(path)[0] + ".png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
