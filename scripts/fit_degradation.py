"""Fit the 1x degradation of averaged 5x data from aligned pairs (scripts/degrade_pairs.py).

Both blocks are normalized as the network sees them (per-block foreground scaling: background -> 0,
median foreground -> 0.25). Model, on the 1x grid (15 x 15 x 20 um voxels):

    one = a * blur_s(five) + b * blur_H(five) + c + noise,    var(noise) = n0 + n1 * max(model, 0)

blur_s is a Gaussian (sigma_z, sigma_xy in voxels) for the 1x optics, blur_H a wide Gaussian for
haze (scattered light). a, b, c by least squares for each (sigma, H) on a grid; the best R^2 wins.
Noise from each image alone: local residual against a 3x3x3 median, binned by local mean, for 1x and
averaged 5x; the augmentation adds the difference.

Usage: uv run python scripts/fit_degradation.py <pairs dir> <out.json> [<figure.png>]
"""

import itertools
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter

from exaspim_axon_segmentation.config import NormalizationConfig
from exaspim_axon_segmentation.inference.normalize import estimate_scaling_array

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pairs = sorted(Path(sys.argv[1]).glob("pair*.npz"))
norm = NormalizationConfig()


def normalized(v):
    s = estimate_scaling_array(v, v > np.percentile(v, 10), norm)
    return (v - s.offset) / s.scale  # unclipped


SIG_Z = (0.0, 0.5, 1.0, 1.5, 2.0)
SIG_XY = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
HAZE = (4.0, 8.0, 16.0)
data = []
for f in pairs:
    z = np.load(f)
    one, five, inside = z["one"], z["five"], z["inside"]
    try:
        n1, n5 = normalized(one), normalized(five)
    except ValueError:
        continue
    keep = inside.copy()
    keep[:3], keep[-3:], keep[:, :3], keep[:, -3:], keep[:, :, :3], keep[:, :, -3:] = (False,) * 6
    data.append((f.stem, n1, n5, keep))
print(len(data), "pairs")

results = []
for sz, sxy, h in itertools.product(SIG_Z, SIG_XY, HAZE):
    ss_res = ss_tot = 0.0
    coefs = []
    for _, n1, n5, keep in data:
        f1 = gaussian_filter(n5, (sz, sxy, sxy))[keep]
        f2 = gaussian_filter(n5, (h / 2, h, h))[keep]  # z voxels are 1.33x longer
        y = n1[keep]
        X = np.stack([f1, f2, np.ones_like(f1)], 1)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ c
        ss_res += float((r**2).sum())
        ss_tot += float(((y - y.mean()) ** 2).sum())
        coefs.append(c)
    results.append({"sigma_z": sz, "sigma_xy": sxy, "haze_xy": h, "r2": 1 - ss_res / ss_tot,
                    "coef": np.median(coefs, 0).tolist()})  # fmt: skip
best = max(results, key=lambda r: r["r2"])
none = next(r for r in results if r["sigma_z"] == 0 and r["sigma_xy"] == 0 and r["haze_xy"] == 4.0)
print("best", best, "\nno blur", none["r2"])


def noise_curve(vols):
    """Noise SD vs local mean (normalized units), from residuals against a 3x3x3 median."""
    bins = np.array([-0.1, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 10])
    mean_all, sd_all = [], []
    for v, keep in vols:
        loc = uniform_filter(v, 3)
        res = v - median_filter(v, 3)
        idx = np.digitize(loc[keep], bins) - 1
        r = res[keep]
        m, s = [], []
        for i in range(len(bins) - 1):
            sel = r[idx == i]
            m.append(float(np.median(loc[keep][idx == i])) if sel.size > 200 else np.nan)
            s.append(1.4826 * float(np.median(np.abs(sel - np.median(sel)))) if sel.size > 200
                     else np.nan)  # fmt: skip
        mean_all.append(m)
        sd_all.append(s)
    return np.nanmedian(mean_all, 0), np.nanmedian(sd_all, 0)


m1, s1 = noise_curve([(n1, k) for _, n1, _, k in data])
m5, s5 = noise_curve([(n5, k) for _, _, n5, k in data])
ok = ~np.isnan(m1) & ~np.isnan(s1)
fit1 = np.polyfit(np.clip(m1[ok], 0, None), s1[ok] ** 2, 1)  # var = n1 * I + n0
ok5 = ~np.isnan(m5) & ~np.isnan(s5)
fit5 = np.polyfit(np.clip(m5[ok5], 0, None), s5[ok5] ** 2, 1)
extra = [max(fit1[1] - fit5[1], 0.0), max(fit1[0] - fit5[0], 0.0)]
out = {
    "grid_um": [20.0, 15.04, 15.04],
    "pairs": [n for n, *_ in data],
    "blur_sigma_vox": {"z": best["sigma_z"], "xy": best["sigma_xy"]},
    "haze_sigma_vox": {"z": best["haze_xy"] / 2, "xy": best["haze_xy"]},
    "gain_blur": best["coef"][0], "gain_haze": best["coef"][1], "offset": best["coef"][2],
    "r2": best["r2"], "r2_no_blur": none["r2"],
    "noise_1x": {"local_mean": m1.tolist(), "sd": s1.tolist(), "var_fit": fit1[::-1].tolist()},
    "noise_5x_averaged": {"local_mean": m5.tolist(), "sd": s5.tolist(),
                          "var_fit": fit5[::-1].tolist()},
    "noise_extra_var": {"n0": extra[0], "n1": extra[1],
                        "note": "variance added: n0 + n1 * max(I, 0), normalized units"},
    "grid": results,
}  # fmt: skip
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("grid", "pairs")}, indent=1))

if len(sys.argv) > 3:  # figure: 1x, averaged 5x, degraded 5x for three pairs; noise curves
    show = data[: min(3, len(data))]
    fig, axs = plt.subplots(len(show) + 1, 3, figsize=(15, 5 * (len(show) + 1)))
    rng = np.random.default_rng(0)
    for r, (name, n1, n5, _) in enumerate(show):
        deg = (out["gain_blur"] * gaussian_filter(n5, (best["sigma_z"], best["sigma_xy"],
               best["sigma_xy"])) + out["gain_haze"] * gaussian_filter(n5, (best["haze_xy"] / 2,
               best["haze_xy"], best["haze_xy"])) + out["offset"])  # fmt: skip
        deg = deg + rng.normal(size=deg.shape) * np.sqrt(extra[0] + extra[1] * np.clip(deg, 0, None))
        zc = n1.shape[0] // 2
        for c, (v, t) in enumerate(((n1, "1x"), (n5, "5x averaged to 15 µm"),
                                    (deg, "5x averaged + fitted blur, haze, noise"))):  # fmt: skip
            axs[r, c].imshow(np.clip(v[zc - 1 : zc + 1].max(0), 0, 1), cmap="gray")
            axs[r, c].set_title(f"{name} · {t}")
            axs[r, c].axis("off")
    axs[-1, 0].plot(m1, s1, "o-", c="#b84a2e", label="1x")
    axs[-1, 0].plot(m5, s5, "o-", c="#2e6db8", label="5x averaged")
    axs[-1, 0].set(xlabel="local mean (normalized)", ylabel="noise SD (normalized)",
                   xscale="symlog", title="noise vs brightness")  # fmt: skip
    axs[-1, 0].legend()
    g = np.array([[r["sigma_xy"], r["r2"]] for r in results if r["sigma_z"] == best["sigma_z"]
                  and r["haze_xy"] == best["haze_xy"]])  # fmt: skip
    axs[-1, 1].plot(g[:, 0], g[:, 1], "o-")
    axs[-1, 1].set(xlabel="in-plane blur sigma (1x voxels)", ylabel="R²", title="fit vs blur")
    axs[-1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(sys.argv[3], dpi=70)
