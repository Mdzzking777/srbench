"""
Plot DMT-KV F_ts time-domain signal from trajectory.csv.

This script intentionally uses a hard contact rule:
    contact iff s <= 0

No softplus smoothing or boundary softening is applied.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Fixed parameters (must match generate_trajectory_DMT_KV.py)
# ============================================================================

Estar = 15e6
R = 10e-9
Fad = 2.0e-9

WINDOW_OVERRIDES_US = {
    "first_contact": ("start", 1034.0),
    "max_x1_pp_change": ("stop", 1057.0),
    "tail_stable": ("stop", 1998.0),
}


def load_trajectory(csv_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No data loaded from {csv_path}")
    return data


def compute_f_ts_from_separation(s_sep_m: np.ndarray) -> np.ndarray:
    """
    Hard DMT contact force with no boundary smoothing.

    F_ts = F_Hertz - Fad    if s <= 0
         = 0                if s > 0
    """
    contact = s_sep_m <= 0.0
    delta = np.where(contact, -s_sep_m, 0.0)
    f_hertz = (4.0 / 3.0) * Estar * np.sqrt(R) * np.power(delta, 1.5)
    f_ts = np.where(contact, f_hertz - Fad, 0.0)
    return f_ts


def contact_onsets(contact_mask: np.ndarray) -> np.ndarray:
    contact_mask = np.asarray(contact_mask, dtype=bool)
    if contact_mask.size == 0:
        return np.array([], dtype=int)
    starts = contact_mask & np.concatenate(([True], ~contact_mask[:-1]))
    return np.flatnonzero(starts)


def peak_to_peak(values: np.ndarray) -> float:
    return float(np.nanmax(values) - np.nanmin(values))


def relocate_window_with_anchor(
    t_post_us: np.ndarray, start_idx: int, stop_idx: int, anchor: str, target_us: float
):
    """
    Relocate a window in post-contact coordinates while preserving its length.

    Indices are 0-based and inclusive.
    Returns (new_start_idx, new_stop_idx, applied_override).
    If the target is outside the feasible range for the current trajectory,
    keep the original window unchanged.
    """
    n = len(t_post_us)
    length = stop_idx - start_idx + 1
    if length <= 0 or n < length:
        return start_idx, stop_idx, False

    if anchor == "start":
        lo = t_post_us[0]
        hi = t_post_us[n - length]
        if not (lo <= target_us <= hi):
            return start_idx, stop_idx, False
        new_start = int(np.searchsorted(t_post_us, target_us, side="left"))
        new_start = min(max(new_start, 0), n - length)
        new_stop = new_start + length - 1
    elif anchor == "stop":
        lo = t_post_us[length - 1]
        hi = t_post_us[-1]
        if not (lo <= target_us <= hi):
            return start_idx, stop_idx, False
        new_stop = int(np.argmin(np.abs(t_post_us - target_us)))
        new_stop = min(max(new_stop, length - 1), n - 1)
        new_start = new_stop - length + 1
    else:
        raise ValueError(f"Unsupported anchor: {anchor}")

    return new_start, new_stop, True


def extend_stop_to_contact_end(contact_mask: np.ndarray, stop_idx: int) -> int:
    """
    If the window currently ends inside a contact segment, extend the stop index
    until that segment fully ends.
    """
    n = len(contact_mask)
    stop_idx = min(max(int(stop_idx), 0), n - 1)
    if not bool(contact_mask[stop_idx]):
        return stop_idx

    new_stop = stop_idx
    while new_stop + 1 < n and bool(contact_mask[new_stop + 1]):
        new_stop += 1
    return new_stop


def build_training_style_windows(t_s: np.ndarray, contact_mask: np.ndarray, x1_signal: np.ndarray):
    first_contact_idx = np.flatnonzero(contact_mask)
    if first_contact_idx.size == 0:
        raise RuntimeError("No contact point found in trajectory.csv")
    first_contact_idx = int(first_contact_idx[0])

    t_post = t_s[first_contact_idx:]
    t_post_us = t_post * 1e6
    contact_post = contact_mask[first_contact_idx:]
    x1_post = x1_signal[first_contact_idx:]

    onset_post = contact_onsets(contact_post)
    if onset_post.size < 3:
        raise RuntimeError("Need at least 3 contact onsets after first contact to build W1/W2/W3")

    cycle_pp = []
    for k in range(len(onset_post) - 1):
        lo = onset_post[k]
        hi = onset_post[k + 1] - 1
        if hi < lo:
            raise RuntimeError("Invalid cycle bounds while building training-style windows")
        cycle_pp.append(peak_to_peak(x1_post[lo : hi + 1]))

    candidates = []
    for k in range(len(onset_post) - 2):
        start_idx = int(onset_post[k])
        stop_idx = int(onset_post[k + 2] - 1)
        candidates.append(
            {
                "candidate_index": k + 1,
                "start_idx": start_idx,
                "stop_idx": stop_idx,
                "len": stop_idx - start_idx + 1,
                "t_start_us": float(t_post_us[start_idx]),
                "t_stop_us": float(t_post_us[stop_idx]),
                "cycle1_index": k + 1,
                "cycle2_index": k + 2,
                "x1_pp_cycle1": float(cycle_pp[k]),
                "x1_pp_cycle2": float(cycle_pp[k + 1]),
                "x1_pp_delta": float(abs(cycle_pp[k + 1] - cycle_pp[k])),
            }
        )

    if not candidates:
        raise RuntimeError("No valid two-cycle windows were built from the post-contact trajectory")

    first_idx = 0
    change_order = list(np.argsort([cand["x1_pp_delta"] for cand in candidates])[::-1])
    tail_order = list(range(len(candidates) - 1, -1, -1))

    selected = []
    seen = set()

    def add_window(role: str, candidate_idx: int) -> bool:
        cand = candidates[candidate_idx].copy()
        key = (cand["start_idx"], cand["stop_idx"])
        if key in seen:
            return False
        seen.add(key)
        cand["role"] = role
        cand["label"] = f"{role}_window"
        selected.append(cand)
        return True

    def add_first_unique(role: str, order):
        for candidate_idx in order:
            if add_window(role, int(candidate_idx)):
                return

    add_window("first_contact", first_idx)
    add_first_unique("max_x1_pp_change", change_order)
    add_first_unique("tail_stable", tail_order)

    if len(selected) < 3:
        raise RuntimeError(f"Could not find 3 unique windows (found {len(selected)})")

    role_to_rank = {
        "first_contact": "W1",
        "max_x1_pp_change": "W2",
        "tail_stable": "W3",
    }

    adjusted = []
    for win in selected:
        anchor, target_us = WINDOW_OVERRIDES_US[win["role"]]
        new_start, new_stop, applied = relocate_window_with_anchor(
            t_post_us, win["start_idx"], win["stop_idx"], anchor, target_us
        )
        if win["role"] == "first_contact":
            new_stop = extend_stop_to_contact_end(contact_post, new_stop)
        win["start_idx"] = new_start
        win["stop_idx"] = new_stop
        win["len"] = new_stop - new_start + 1
        win["t_start_us"] = float(t_post_us[new_start])
        win["t_stop_us"] = float(t_post_us[new_stop])
        win["override_anchor"] = anchor
        win["override_target_us"] = float(target_us)
        win["override_applied"] = bool(applied)
        win["rank"] = role_to_rank[win["role"]]
        win["full_start_idx"] = first_contact_idx + new_start
        win["full_stop_idx"] = first_contact_idx + new_stop
        adjusted.append(win)

    return adjusted


def save_plot(x, y, contact_mask, out_png: Path, out_pdf: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, y, color="tab:purple", linewidth=0.9, label="F_ts")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.7)

    if np.any(contact_mask):
        ax.fill_between(
            x,
            y,
            0.0,
            where=contact_mask,
            color="tab:red",
            alpha=0.20,
            label="contact (s <= 0)",
        )

    ax.set_xlabel("Time [μs]")
    ax.set_ylabel("F_ts [nN]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)


def save_training_window_plot(
    t_us: np.ndarray,
    f_ts_nN: np.ndarray,
    contact_mask: np.ndarray,
    windows,
    out_png: Path,
    out_pdf: Path,
):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharey=True)
    axes = np.atleast_1d(axes)

    y_vals = []
    for win in windows:
        sl = slice(win["full_start_idx"], win["full_stop_idx"] + 1)
        y_vals.append(f_ts_nN[sl])
    y_all = np.concatenate(y_vals)
    y_pad = 0.08 * max(1e-6, float(y_all.max() - y_all.min()))
    y_lo = float(y_all.min() - y_pad)
    y_hi = float(y_all.max() + y_pad)

    role_titles = {
        "first_contact": "first_contact",
        "max_x1_pp_change": "max_x1_pp_change",
        "tail_stable": "tail_stable",
    }

    for ax, win in zip(axes, windows):
        sl = slice(win["full_start_idx"], win["full_stop_idx"] + 1)
        x = t_us[sl]
        y = f_ts_nN[sl]
        c = contact_mask[sl]
        ax.plot(x, y, color="tab:purple", linewidth=1.0)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.7)
        if np.any(c):
            ax.fill_between(x, y, 0.0, where=c, color="tab:red", alpha=0.20)
        override_tag = "override" if win["override_applied"] else "auto"
        ax.set_title(
            f"{win['rank']} [{role_titles[win['role']]}]  "
            f"{win['t_start_us']:.3f}-{win['t_stop_us']:.3f} μs  "
            f"({override_tag})"
        )
        ax.set_ylabel("F_ts [nN]")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_lo, y_hi)

    axes[-1].set_xlabel("Time [μs]")
    fig.suptitle("AFM DMT-KV F_ts Time-Domain Signal (training-style W1 / W2 / W3 windows)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    csv_path = root / "trajectory.csv"
    plots_dir = root / "plots"
    plots_dir.mkdir(exist_ok=True)

    data = load_trajectory(csv_path)

    t_s = np.asarray(data["time_s"], dtype=float)
    x_tip_m = np.asarray(data["x_tip_m"], dtype=float)
    s_sep_m = np.asarray(data["s_separation_m"], dtype=float)

    f_ts_n = compute_f_ts_from_separation(s_sep_m)
    contact = s_sep_m <= 0.0

    t_us = t_s * 1e6
    f_ts_nN = f_ts_n * 1e9

    save_plot(
        t_us,
        f_ts_nN,
        contact,
        plots_dir / "F_ts_time_domain_full.png",
        plots_dir / "F_ts_time_domain_full.pdf",
        "AFM DMT-KV F_ts Time-Domain Signal (hard contact, no smoothing)",
    )

    training_windows = build_training_style_windows(t_s, contact, x_tip_m)
    save_training_window_plot(
        t_us,
        f_ts_nN,
        contact,
        training_windows,
        plots_dir / "F_ts_time_domain_zoomed.png",
        plots_dir / "F_ts_time_domain_zoomed.pdf",
    )

    print("Saved:")
    print(f"  {plots_dir / 'F_ts_time_domain_full.png'}")
    print(f"  {plots_dir / 'F_ts_time_domain_full.pdf'}")
    print(f"  {plots_dir / 'F_ts_time_domain_zoomed.png'}")
    print(f"  {plots_dir / 'F_ts_time_domain_zoomed.pdf'}")
    for win in training_windows:
        print(
            f"{win['rank']} [{win['role']}] -> "
            f"{win['t_start_us']:.3f}-{win['t_stop_us']:.3f} us "
            f"(override={'yes' if win['override_applied'] else 'no'})"
        )
    print(f"F_ts min/max [nN]: {f_ts_nN.min():.6f}, {f_ts_nN.max():.6f}")
    print(f"Contact fraction [%]: {100.0 * np.mean(contact):.2f}")


if __name__ == "__main__":
    main()
