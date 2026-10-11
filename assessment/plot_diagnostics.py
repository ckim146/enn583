"""Plot every numeric column of results_vo_diagnostics.csv against frame index.

Usage (from the folder containing the CSV):
    python plot_diagnostics.py                     # stacked panels, one per unit
    python plot_diagnostics.py --per-frame         # also add per-frame error per metre
    python plot_diagnostics.py --overlay           # everything on one axis, scaled 0-1

The error panels read results_vo_errors.csv, which the checker writes
(it needs ground truth, so the diagnostics CSV alone can't provide it).
    python plot_diagnostics.py --csv path/to/results_vo_diagnostics.csv --show
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SKIP = {"frame_i", "frame_j", "accepted", "reasons", "inlier_ratio"}


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    cols = {}
    for name in rows[0]:
        if name in SKIP:
            continue
        try:
            cols[name] = np.array([float(r[name]) for r in rows])
        except ValueError:
            pass  # non-numeric column
    frames = np.array([int(r["frame_i"]) for r in rows])
    rejected = frames[np.array([r["accepted"] == "0" for r in rows])]
    return frames, cols, rejected


def load_errors(path):
    """Per-frame ground-truth errors written by the checker (None if missing)."""
    if not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    to_f = lambda v: float(v) if v not in ("", None) else np.nan
    return {name: np.array([to_f(r[name]) for r in rows]) for name in rows[0]}


def mark_rejected(ax, rejected):
    for k, f in enumerate(rejected):
        ax.axvline(f, color="red", alpha=0.25, lw=1, label="Rejected" if k == 0 else None)


def plot_stacked(frames, cols, rejected, errors=None, per_frame=False):
    # Each panel: (y label, [(x, y, legend label), ...])
    def series(names, x=frames, src=cols):
        return [(x, src[n], n) for n in names if n in src]

    panels = [
        ("Count", series(("n_matches", "n_points_3d", "n_inliers"))),
        ("Rotation [deg]", series(("rot_deg",))),
        ("Translation [m]", series(("trans_m",))),
    ]
    used = {"n_matches", "n_points_3d", "n_inliers", "rot_deg", "trans_m"}
    panels += [(c, series((c,))) for c in cols if c not in used]  # extra columns added later

    if errors is not None:
        # position error is per frame; pair i -> i+1 lines up with frame i+1
        pair_x = errors["frame"] - 1
        if per_frame:
            panels.append(("Step error [% of step]",
                           [(pair_x, errors["frame_trans_err_pct"],
                             "frame-to-frame translation error / true step length")]))
        panels.append(("Position error [m]",
                       [(pair_x, errors["position_error_m"],
                         "total trajectory error ||p_est - p_gt||")]))
    else:
        print("results_vo_errors.csv not found: run the checker first for the error panel.")
    panels = [p for p in panels if p[1]]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.2 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (ylabel, lines) in zip(axes, panels):
        for x, y, label in lines:
            ax.plot(x, y, lw=1.1, label=label)
        mark_rejected(ax, rejected)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Frame pair index (i \u2192 i+1)")
    axes[0].set_title("Visual odometry diagnostics per frame pair")
    fig.tight_layout()
    return fig


def plot_overlay(frames, cols, rejected):
    """Excel-style single chart. Each column is min-max scaled to 0-1 so
    counts (hundreds) and ratios (0-1) fit on one axis; compare shapes, not values."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, v in cols.items():
        lo, hi = np.nanmin(v), np.nanmax(v)
        ax.plot(frames, (v - lo) / (hi - lo + 1e-12), lw=1.1,
                label=f"{name}  [{lo:.3g} \u2013 {hi:.3g}]")
    mark_rejected(ax, rejected)
    ax.set_xlabel("Frame pair index (i \u2192 i+1)")
    ax.set_ylabel("Scaled value (0 = column min, 1 = column max)")
    ax.set_title("Visual odometry diagnostics per frame pair (each column scaled 0\u20131)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default="results_vo_diagnostics.csv")
    ap.add_argument("--errors", default="results_vo_errors.csv")
    ap.add_argument("--per-frame", action="store_true",
                    help="add a panel of per-frame translation error as %% of step length")
    ap.add_argument("--overlay", action="store_true", help="all columns on one axis, scaled 0-1")
    ap.add_argument("--out", default=None, help="output image path")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    args = ap.parse_args()

    frames, cols, rejected = load(args.csv)
    errors = load_errors(args.errors)
    if args.overlay:
        fig = plot_overlay(frames, cols, rejected)
    else:
        fig = plot_stacked(frames, cols, rejected, errors, args.per_frame)
    out = args.out or ("results_vo_diagnostics_overlay.png" if args.overlay
                       else "results_vo_diagnostics.png")
    fig.savefig(out, dpi=200)
    print(f"Wrote {out} ({len(frames)} frame pairs, {len(rejected)} rejected)")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()