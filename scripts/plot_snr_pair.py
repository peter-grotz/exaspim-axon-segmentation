"""Figures and summary of an snr-pair run: per-block MIPs and 1x-vs-5x metric scatter.

Usage: uv run python scripts/plot_snr_pair.py <snr_pair dir> <out prefix>
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d_dir, out = Path(sys.argv[1]), sys.argv[2]
d = json.loads((d_dir / "snr_pair.json").read_text())
B = [b for b in d["blocks"] if "skipped" not in b and "snr" in b.get("a", {})]
good = [b for b in B if b["match_ncc"] >= 0.3 and (b.get("raw_vs_fused_ncc") or 0) >= 0.6
        and b.get("b", {}).get("snr") is not None]  # fmt: skip
def get(bs, *k):
    out = []
    for b in bs:
        v = b
        for key in k:
            v = v.get(key) if isinstance(v, dict) else None
        out.append(np.nan if v is None else v)
    return np.array(out, float)



summary = {"blocks": len(B), "good_match": len(good)}
for name, k in {"1x SNR (own detection)": ("a", "snr"),
                     "5x raw SNR (own detection)": ("b", "snr"),
                     "1x SNR on 5x axons": ("cross", "a_snr_on_b_mask"),
                     "1x recall of 5x axons": ("cross", "recall_a"),
                     "1x precision vs 5x axons": ("cross", "precision_a"),
                     "1x noise SD (counts)": ("a", "noise_sd"),
                     "5x raw noise SD (counts)": ("b", "noise_sd"),
                     "1x detected %": ("a", "axon_fraction"),
                     "5x detected %": ("b", "axon_fraction"),
                     "5x axons on 1x grid %": ("cross", "b_axon_fraction_on_a"),
                     "matched voxel: 1x SNR": ("cross", "a_matched", "snr"),
                     "matched voxel: 5x SNR": ("cross", "b_matched", "snr"),
                     "matched voxel: 1x noise/background": ("cross", "a_matched", "nb"),
                     "matched voxel: 5x noise/background": ("cross", "b_matched", "nb")}.items():  # fmt: skip
    if k[-1] == "nb":
        v = get(good, *k[:-1], "noise_sd") / get(good, *k[:-1], "background")
    else:
        v = get(good, *k) * (100 if "%" in name else 1)
    summary[name] = {"median": float(np.nanmedian(v)), "p25": float(np.nanpercentile(v, 25)),
                     "p75": float(np.nanpercentile(v, 75))}  # fmt: skip
ratio = get(good, "cross", "b_matched", "snr") / get(good, "cross", "a_matched", "snr")
summary["matched voxel: 5x/1x SNR ratio"] = {"median": float(np.nanmedian(ratio)),
    "p25": float(np.nanpercentile(ratio, 25)), "p75": float(np.nanpercentile(ratio, 75))}  # fmt: skip
Path(f"{out}_summary.json").write_text(json.dumps(summary, indent=1))
for k, v in summary.items():
    print(k, v if not isinstance(v, dict) else "median %.3g [%.3g, %.3g]" % tuple(v.values()))

# scatter
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
a, b5, ab = get(good, "a", "snr"), get(good, "b", "snr"), get(good, "cross", "a_snr_on_b_mask")
lim = max(np.nanmax(a), np.nanmax(b5)) * 1.1
ax[0].scatter(b5, a, c="#b84a2e", label="1x on its own detections")
ax[0].scatter(b5, ab, c="#2e6db8", label="1x on the 5x axons")
ax[0].plot([0, lim], [0, lim], "k--", lw=0.8)
ax[0].set(xlabel="5x raw SNR (own detections)", ylabel="1x SNR", xlim=(0, lim), ylim=(0, lim),
          title="SNR per matched block")  # fmt: skip
ax[0].legend()
ax[1].scatter(get(good, "cross", "recall_a"), get(good, "cross", "precision_a"), c="#2e6db8")
ax[1].set(xlabel="1x recall of 5x axons", ylabel="1x precision vs 5x axons", xlim=(0, 1),
          ylim=(0, 1), title="1x detection against the 5x (within 1 1x voxel)")  # fmt: skip
ma, mb = get(good, "cross", "a_matched", "snr"), get(good, "cross", "b_matched", "snr")
lim2 = np.nanmax([ma, mb]) * 1.1
ax[2].scatter(mb, ma, c="#2e6db8")
for b, x, y in zip(good, mb, ma, strict=True):
    ax[2].annotate(str(b["block"]), (x, y), fontsize=7, alpha=0.6)
ax[2].plot([0, lim2], [0, lim2], "k--", lw=0.8)
ax[2].set(xlabel="5x SNR at the 1x voxel size", ylabel="1x SNR", xlim=(0, lim2), ylim=(0, lim2),
          title="same voxels (15x15x20 um), same mask (5x axons)")  # fmt: skip
for x in ax:
    x.grid(alpha=0.3)
fig.suptitle(f"791116 · same-day 1x (15x15x20 um) vs 5x raw (3x3x4 um) · {len(good)} of "
             f"{len(B)} blocks matched (match NCC >= 0.3, raw~fused >= 0.6)")  # fmt: skip
fig.tight_layout()
fig.savefig(f"{out}_scatter.png", dpi=90)

# per-block panels
show = sorted(good, key=lambda b: -b["match_ncc"])[:8]
fig, axs = plt.subplots(len(show), 5, figsize=(22, 4.4 * len(show)))
def nm(m):
    m = np.log1p(np.clip(m - np.percentile(m, 50), 0, None))
    lo, hi = np.percentile(m, [1, 99.8])
    return np.clip((m - lo) / (hi - lo + 1e-6), 0, 1)
for r, b in enumerate(show):
    k = b["block"]
    za, zb = np.load(d_dir / f"block{k:03d}_a.npz"), np.load(d_dir / f"block{k:03d}_b.npz")
    on = np.load(d_dir / f"block{k:03d}_b_on_a.npz")["mip"]
    a_, b_, c_ = b["a"], b["b"], b["cross"]
    panels = [(nm(za["raw_mip"]), "gray", f"block {k} · 1x image (15 um px)"),
              (za["prob_mip"], "magma", f"1x prob · {100 * a_['axon_fraction']:.2f}% · SNR {a_['snr']:.1f}"),
              (nm(zb["raw_mip"]), "gray", f"5x raw image (3 um px) · {b['b_tile']}"),
              (zb["prob_mip"], "magma", f"5x prob · {100 * b_['axon_fraction']:.3f}% · SNR {b_['snr']:.1f}"),
              (on, "gray", f"5x axons on 1x grid · 1x SNR there {c_['a_snr_on_b_mask']:.1f} · "
                           f"R {c_['recall_a']:.2f} P {c_['precision_a']:.2f}")]  # fmt: skip
    for c, (img, cmap, title) in enumerate(panels):
        axs[r, c].imshow(img, cmap=cmap)
        axs[r, c].set_title(title, fontsize=9)
        axs[r, c].axis("off")
fig.suptitle("matched blocks (z-MIP over 1.28 mm), best matches first", y=0.999)
fig.tight_layout()
fig.savefig(f"{out}_blocks.png", dpi=70)
