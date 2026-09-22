"""Failure-analysis helpers for check_student_solution_custom.py.

Put this file in support/ (already on the checker's sys.path).
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # the checker saves figures rather than showing them
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

DIAGNOSTICS_CSV = "results_vo_diagnostics.csv"
FIRST_FAILURE_NPZ = "results_first_failure.npz"


def load_diagnostics(path=DIAGNOSTICS_CSV):
    """Rows written by student visual_odometry(): one per consecutive frame pair."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_trajectory_colored(student_poses, gt_poses, frame_err, rejected_frames=(),
                            err_label="Frame-to-frame rotation error [deg]",
                            save_path="results_vo_trajectory_error.png", title=None,
                            cmap="viridis", vmin=None, vmax=None, line_label=None):
    """Bird's-eye (x-z, camera frame) trajectory. Segment k (pose k -> k+1) is
    colored by frame_err[k]. Rejected frames are marked with red crosses."""
    est = np.array([np.asarray(p.t).ravel() for p in student_poses])
    gt = np.array([np.asarray(p.t).ravel() for p in gt_poses])
    frame_err = np.asarray(frame_err)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(gt[:, 0], gt[:, 2], "k--", lw=1.5, label="Ground truth")

    pts = est[:, [0, 2]]
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = max(np.percentile(frame_err, 99), vmin + 1e-9)  # one spike shouldn't wash out the scale
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin, vmax), lw=2.5)
    lc.set_array(frame_err)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label=err_label, extend="both")

    if len(rejected_frames):
        r = np.asarray(rejected_frames)
        ax.scatter(est[r, 0], est[r, 2], marker="x", c="red", s=45, zorder=3,
                   label=f"Rejected estimates ({len(r)})")
    ax.scatter(gt[0, 0], gt[0, 2], marker="o", c="black", s=50, zorder=4, label="Start")
    ax.plot([], [], color=plt.get_cmap(cmap)(0.7), lw=2.5,
            label=line_label or "Estimate (colored by error)")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z (forward) [m]")
    ax.set_title(title or "Estimated vs ground-truth trajectory")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def show_first_failure(dataset, npz_path=FIRST_FAILURE_NPZ,
                       save_path="results_vo_first_failure.png"):
    """Frame i on top, frame j below, match lines between them
    (green = PnP inliers, red = outliers). Rejection reasons printed underneath."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=False)
    i, j = int(d["frame_i"]), int(d["frame_j"])
    img1 = np.asarray(dataset.stereo(i)[0])
    img2 = np.asarray(dataset.stereo(j)[0])
    h = img1.shape[0]

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.imshow(np.vstack([img1, img2]), cmap="gray")

    p1 = d["pts_i"].reshape(-1, 2)
    p2 = d["pts_j"].reshape(-1, 2) + [0, h]  # shift into the lower image
    if len(p1):
        inl = d["inlier_mask"].astype(bool).ravel()
        for mask, color in [(~inl, "red"), (inl, "lime")]:
            segs = np.stack([p1[mask], p2[mask]], axis=1)
            ax.add_collection(LineCollection(segs, colors=color, lw=0.6, alpha=0.7))
        ax.legend(handles=[Line2D([], [], color="lime", label=f"PnP inliers ({inl.sum()})"),
                           Line2D([], [], color="red", label=f"Outliers ({(~inl).sum()})")],
                  loc="upper right")

    reasons = [str(r) for r in d["reasons"]]
    ax.set_title(f"First rejected frame pair: {i} \u2192 {j}  (top: frame {i}, bottom: frame {j})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Rejected because:\n" + "\n".join(reasons), fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def worst_region(position_errors, window=10):
    """Frame window where absolute position error grows fastest."""
    e = np.asarray(position_errors)
    if len(e) <= window:
        return 0, len(e) - 1, float(e[-1] - e[0])
    growth = e[window:] - e[:-window]
    k = int(np.argmax(growth))
    return k, k + window, float(growth[k])


def spike_frames(frame_err, k=5.0):
    """Frame pairs whose error is an outlier: > median + k * MAD."""
    err = np.asarray(frame_err)
    med = np.median(err)
    mad = np.median(np.abs(err - med)) + 1e-12
    return np.flatnonzero(err > med + k * mad)


def blur_score(img):
    """Variance of the Laplacian: lower = blurrier. Compare against other frames
    in the same sequence rather than an absolute threshold."""
    import cv2
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def show_frame_pair(images_dataset, frame_i, frame_j, info, max_lines=150,
                    seed=0, title_note="", save_path=None):
    """Frame i on top, frame j below, a random subset of matches drawn between them
    (green = PnP inliers, red = outliers). `info` is the dict returned by
    estimate_relative_pose(..., return_info=True)."""
    img1 = np.asarray(images_dataset.stereo(frame_i)[0])
    img2 = np.asarray(images_dataset.stereo(frame_j)[0])
    h = img1.shape[0]

    p1 = np.asarray(info["pts_i"]).reshape(-1, 2)
    p2 = np.asarray(info["pts_j"]).reshape(-1, 2) + [0, h]
    inl = np.asarray(info["inlier_mask"], bool).ravel()

    # Subsample so the lines stay readable, keeping the inlier/outlier proportion
    rng = np.random.default_rng(seed)
    keep = np.arange(len(p1))
    if len(keep) > max_lines:
        keep = rng.choice(keep, max_lines, replace=False)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.imshow(np.vstack([img1, img2]), cmap="gray")
    for mask, color in [(~inl[keep], "red"), (inl[keep], "lime")]:
        k = keep[mask]
        segs = np.stack([p1[k], p2[k]], axis=1)
        ax.add_collection(LineCollection(segs, colors=color, lw=0.8, alpha=0.8))
        ax.scatter(p1[k, 0], p1[k, 1], s=6, c=color)
        ax.scatter(p2[k, 0], p2[k, 1], s=6, c=color)

    n_pts, n_inl = len(p1), int(inl.sum())
    ax.legend(handles=[Line2D([], [], color="lime", label=f"PnP inliers ({n_inl} total)"),
                       Line2D([], [], color="red", label=f"Outliers ({n_pts - n_inl} total)")],
              loc="upper right")
    ax.set_title(f"Frames {frame_i} \u2192 {frame_j}  (top: {frame_i}, bottom: {frame_j})"
                 + (f"  |  {title_note}" if title_note else ""))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        f"Matches: {info.get('n_matches', '?')}   |   with valid depth: {n_pts}   |   "
        f"inliers: {n_inl} ({100 * n_inl / max(n_pts, 1):.0f}%)   |   "
        f"showing {len(keep)} random matches\n"
        f"Blur score (Laplacian variance, higher = sharper): "
        f"frame {frame_i}: {blur_score(img1):.0f}, frame {frame_j}: {blur_score(img2):.0f}",
        fontsize=10)
    fig.tight_layout()
    save_path = save_path or f"results_vo_pair_{frame_i}_{frame_j}.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def blur_profile(images_dataset, n_frames, step=1):
    """Blur score for every `step`-th frame, to see whether a frame is unusually blurry."""
    frames = np.arange(0, n_frames, step)
    return frames, np.array([blur_score(np.asarray(images_dataset.stereo(int(f))[0]))
                             for f in frames])


def plot_inliers_and_error_vs_frame(n_inliers, heading_err_deg,
                                    save_path="results_vo_inliers_vs_error.png"):
    """Two stacked panels sharing the frame axis: PnP inliers (top) and
    |heading error| (bottom). Shows whether error spikes line up with inlier dips."""
    frames = np.arange(len(n_inliers))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ax1.plot(frames, n_inliers, lw=1.2)
    ax1.set_ylabel("PnP inliers")
    ax1.grid(alpha=0.3)
    ax2.plot(frames, np.abs(heading_err_deg), lw=1.2, color="tab:red")
    ax2.set_ylabel("|Heading error| [deg]")
    ax2.set_xlabel("Frame pair index (k \u2192 k+1)")
    ax2.grid(alpha=0.3)
    r = np.corrcoef(n_inliers, np.abs(heading_err_deg))[0, 1]
    ax1.set_title(f"Inliers vs. heading error per frame pair (Pearson r = {r:.2f})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path, r