"""Fit the 1x degradation of averaged 5x data, v2: local shift refinement, non-negative haze,
background-floor noise. See scripts/fit_degradation.py for the model.

Usage: uv run python scripts/fit_degradation2.py <pairs dir> <out.json> <figure.png>
"""

import itertools
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift
from scipy.optimize import nnls

from exaspim_axon_segmentation.config import NormalizationConfig
from exaspim_axon_segmentation.inference.normalize import estimate_scaling_array

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

norm = NormalizationConfig()


def normalized(v):
    s = estimate_scaling_array(v, v > np.percentile(v, 10), norm)
    return (v - s.offset) / s.scale


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


C = (slice(4, -4),) * 3  # core, away from the edges and the shift border
data = []
for f in sorted(Path(sys.argv[1]).glob("pair*.npz")):
    z = np.load(f)
    try:
        n1, n5 = normalized(z["one"]), normalized(z["five"])
    except ValueError:
        continue
    ref = gaussian_filter(n5, (1.0, 1.0, 1.0))
    best = (-2.0, (0, 0, 0))
    for d in itertools.product(range(-3, 4), repeat=3):  # 1x voxels: +-60 um z, +-45 um y/x
        c = ncc(n1[C], np.roll(ref, d, (0, 1, 2))[C])
        if c > best[0]:
            best = (c, d)
    n5s = np.roll(n5, best[1], (0, 1, 2))
    data.append({"name": f.stem, "n1": n1, "n5": n5s, "ncc": best[0], "shift": best[1]})
    print(f.stem, "shift", best[1], "NCC %.2f" % best[0], flush=True)
good = [d for d in data if d["ncc"] >= 0.5]
print(len(good), "of", len(data), "pairs with NCC >= 0.5")

grid = []
for sz, sxy, h in itertools.product((0.0, 0.5, 1.0, 1.5, 2.0), (0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
                                    (4.0, 8.0, 16.0)):  # fmt: skip
    ss_res = ss_tot = 0.0
    coefs = []
    for d in good:
        f1 = gaussian_filter(d["n5"], (sz, sxy, sxy))[C].ravel()
        f2 = gaussian_filter(d["n5"], (h / 2, h, h))[C].ravel()
        y = d["n1"][C].ravel()
        X = np.stack([f1, f2, np.ones_like(f1), -np.ones_like(f1)], 1)  # offset of either sign
        c, _ = nnls(X, y)
        r = y - X @ c
        ss_res += float((r**2).sum())
        ss_tot += float(((y - y.mean()) ** 2).sum())
        coefs.append([c[0], c[1], c[2] - c[3]])
    grid.append({"sigma_z": sz, "sigma_xy": sxy, "haze_xy": h, "r2": 1 - ss_res / ss_tot,
                 "coef": np.median(coefs, 0).tolist()})  # fmt: skip
best = max(grid, key=lambda r: r["r2"])
r2_none = max(r["r2"] for r in grid if r["sigma_z"] == 0 and r["sigma_xy"] == 0)


def floor_sd(v):
    """Noise SD in the dimmest 40% of voxels: MAD of the difference of neighbouring x voxels / sqrt 2."""
    dv = np.diff(v, axis=2)[C]
    lo = v[C][:, :, :-1] if dv.shape[2] < v[C].shape[2] else v[C]
    sel = dv[lo[..., : dv.shape[2]] <= np.percentile(lo, 40)]
    return 1.4826 * float(np.median(np.abs(sel - np.median(sel)))) / np.sqrt(2)


sd1 = float(np.median([floor_sd(d["n1"]) for d in good]))
sd5 = float(np.median([floor_sd(d["n5"]) for d in good]))
extra_sd = float(np.sqrt(max(sd1**2 - sd5**2, 0.0)))
out = {
    "grid_um": [20.0, 15.04, 15.04],
    "pairs_used": [d["name"] for d in good],
    "shifts_vox": {d["name"]: list(d["shift"]) for d in data},
    "pair_ncc": {d["name"]: d["ncc"] for d in data},
    "blur_sigma_vox": {"z": best["sigma_z"], "xy": best["sigma_xy"]},
    "haze_sigma_vox": {"z": best["haze_xy"] / 2, "xy": best["haze_xy"]},
    "gain_blur": best["coef"][0], "gain_haze": best["coef"][1], "offset": best["coef"][2],
    "r2": best["r2"], "r2_no_blur": r2_none,
    "noise_floor_sd": {"1x": sd1, "5x_averaged": sd5, "extra": extra_sd,
                       "note": "normalized units (background 0, median foreground 0.25)"},
    "grid": grid,
}  # fmt: skip
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("grid", "shifts_vox", "pair_ncc")},
                 indent=1))  # fmt: skip

show = sorted(good, key=lambda d: -d["ncc"])[:4]
fig, axs = plt.subplots(len(show), 3, figsize=(15, 5 * len(show)))
rng = np.random.default_rng(0)
for r, d in enumerate(show):
    deg = (out["gain_blur"] * gaussian_filter(d["n5"], (best["sigma_z"], best["sigma_xy"],
           best["sigma_xy"])) + out["gain_haze"] * gaussian_filter(d["n5"], (best["haze_xy"] / 2,
           best["haze_xy"], best["haze_xy"])) + out["offset"])  # fmt: skip
    deg = deg + rng.normal(size=deg.shape) * extra_sd
    zc = d["n1"].shape[0] // 2
    for c, (v, t) in enumerate(((d["n1"], "1x"), (d["n5"], "5x averaged to 15 µm"),
                                (deg, "5x averaged + fitted blur, haze, noise"))):  # fmt: skip
        axs[r, c].imshow(np.clip(v[zc - 1 : zc + 1].max(0), 0, 0.6), cmap="gray")
        axs[r, c].set_title(f"{d['name']} (match {d['ncc']:.2f}) · {t}", fontsize=10)
        axs[r, c].axis("off")
fig.suptitle(f"791116 same-day pairs, 40 µm slabs · fit: blur σ z {best['sigma_z']} / xy "
             f"{best['sigma_xy']} voxels, haze σ {best['haze_xy']} voxels, R² {best['r2']:.2f} "
             f"(no blur {r2_none:.2f}) · noise floor 1x {sd1:.3f}, 5x {sd5:.3f}")  # fmt: skip
fig.tight_layout()
fig.savefig(sys.argv[3], dpi=70)
