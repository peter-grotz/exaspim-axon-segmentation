"""Per-patch 1x vs 5x image statistics from an snr-pair run: scatter and paired side-by-side plots.

Each scope is scored on its own patch at its own voxel size, on its own TRAILMAP detections.

Usage: uv run python scripts/plot_snr_pair_stats.py <snr_pair dir> <out.png> [--matched]

--matched compares the 1x with the 5x averaged to the 1x voxel size and segmented again
(``b.down_to_a``): same voxel size, same detector, same scoring.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.loads((Path(sys.argv[1]) / "snr_pair.json").read_text())
MATCHED = "--matched" in sys.argv


def side_b(b):
    return b.get("b", {}).get("down_to_a", {}) if MATCHED else b.get("b", {})


B = [b for b in d["blocks"] if b.get("a", {}).get("snr") is not None
     and side_b(b).get("snr") is not None]  # fmt: skip


def stat(side, f):
    return np.array([f(b["a"] if side == "a" else side_b(b)) for b in B], float)


STATS = [  # name, function of a scored side, log axis
    ("SNR = (signal − background) / noise", lambda s: s["snr"], False),
    ("contrast = (signal − background) / background", lambda s: (s["signal"] - s["background"])
     / s["background"], False),  # fmt: skip
    ("noise / background", lambda s: s["noise_sd"] / s["background"], False),
    ("detected axon (% of voxels)", lambda s: 100 * s["axon_fraction"], True),
    ("signal (raw counts)", lambda s: s["signal"], True),
    ("background (raw counts)", lambda s: s["background"], True),
    ("noise SD (raw counts)", lambda s: s["noise_sd"], True),
]
C1, C5 = "#b84a2e", "#2e6db8"
order = np.argsort(stat("a", STATS[0][1]))
fig, axs = plt.subplots(len(STATS), 2, figsize=(16, 4.2 * len(STATS)),
                        gridspec_kw={"width_ratios": [1, 2.4]})  # fmt: skip
summary = {}
for r, (name, f, log) in enumerate(STATS):
    x1, x5 = stat("a", f), stat("b", f)
    summary[name] = {"1x median": float(np.median(x1)), "5x median": float(np.median(x5)),
                     "5x/1x median ratio": float(np.median(x5 / x1)),
                     "patches 5x > 1x": int((x5 > x1).sum())}  # fmt: skip
    ax = axs[r, 0]
    ax.scatter(x1, x5, c="#444", s=22)
    for b, u, v in zip(B, x1, x5, strict=True):
        ax.annotate(str(b["block"]), (u, v), fontsize=7, alpha=0.6, xytext=(3, 2),
                    textcoords="offset points")  # fmt: skip
    lo = min(x1.min(), x5.min()) * (0.8 if log else 0)
    hi = max(x1.max(), x5.max()) * 1.15
    ax.plot([lo or 1e-9, hi], [lo or 1e-9, hi], "k--", lw=0.8)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
    ax.set(xlabel="1x patch", ylabel="5x patch", title=name)
    ax.grid(alpha=0.3)
    ax = axs[r, 1]
    pos = np.arange(len(B))
    for i, k in enumerate(order):
        ax.plot([i - 0.17, i + 0.17], [x1[k], x5[k]], color="#bbb", lw=0.8, zorder=1)
    ax.scatter(pos - 0.17, x1[order], c=C1, s=26, label=f"1x (median {np.median(x1):.3g})", zorder=2)
    ax.scatter(pos + 0.17, x5[order], c=C5, s=26, label=f"5x (median {np.median(x5):.3g})", zorder=2)
    ax.set_xticks(pos, [str(B[k]["block"]) for k in order], fontsize=7)
    if log:
        ax.set_yscale("log")
    ax.set(xlabel="patch (sorted by 1x SNR)", title=f"{name}: each patch, 1x and 5x side by side")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
what5 = "5x raw averaged to 15×15×20 µm, segmented again" if MATCHED else "5x raw (3×3×4 µm)"
fig.suptitle(f"791116 · same-day 1x (15×15×20 µm) vs {what5} · {len(B)} patches, "
             "each scope on its own patch and its own TRAILMAP detections", y=0.998)  # fmt: skip
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=70)
Path(sys.argv[2].replace(".png", ".json")).write_text(json.dumps(summary, indent=1))
for k, v in summary.items():
    print(f"{k:48s} 1x {v['1x median']:.3g} · 5x {v['5x median']:.3g} · ratio {v['5x/1x median ratio']:.2f} · 5x higher in {v['patches 5x > 1x']}/{len(B)}")
