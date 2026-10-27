"""5x SNR, 1x SNR and the 5x/1x ratio per brain, grouped by microscope, genotype/virus and date.

Per brain: median block SNR at one threshold with the 25-75% range (5x: rescore-tiles results; 1x:
snr_1x_blocks.py results) and the ratio of medians with a 95% bootstrap interval. Three figures,
one per grouping, each with three panels (5x, 1x, ratio); brains without a same-day 1x scan appear
in the 5x panel only.

Usage: uv run python scripts/plot_snr_by_group.py <out prefix> <threshold>
"""

import json
import re
import sys
from pathlib import Path

import fsspec
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

out, t = sys.argv[1], float(sys.argv[2])
key = f"p{t:g}"
EXCLUDED = {"851474"}
BRIGHT = {"665465", "719653", "787423", "852470", "853533"}
ONE_X = {s: f"results/snr_1x/{s}_1x_0.json" for s in
         ("853533", "787423", "841260", "819682", "652441", "663412", "787425", "823507", "823508")}  # fmt: skip
ONE_X["791116"] = "results/snr_1x/791116_1x_1.json"
subs = json.loads(Path("results/run_snrtiles_all.json").read_text())
sheet = {
    str(d.get("Sample")).split("_")[0]: d
    for d in json.loads(Path("results/exm_dataset_summary.json").read_text())
}
fs = fsspec.filesystem("s3", anon=True)


def snrs(blocks):
    return np.array([b[key]["snr"] for b in blocks if b[key].get("snr") is not None
                     and b[key].get("noise_sd", 1) > 0], float)  # fmt: skip


rng = np.random.default_rng(0)
rows = []
for sid in sorted(subs):
    if sid in EXCLUDED:
        continue
    v5 = snrs([b for f in sorted(Path(f"results/rescore/{sid}/rescore").glob("*/snr_rescore.json"))
               for b in json.loads(f.read_text())["blocks"]])  # fmt: skip
    v1 = snrs(json.loads(Path(ONE_X[sid]).read_text())["blocks"]) if sid in ONE_X else None
    base = subs[sid]["root"].replace("s3://", "").rstrip("/").rsplit("/SPIM", 1)[0]
    acq = fs.cat(base + "/acquisition.json").decode()
    ins = (re.findall(r'"instrument_id": "([^"]*)"', acq) or ["missing"])[0]
    ins = "missing" if ins in ("None", "") else ins
    row = {"sid": sid, "v5": v5, "v1": v1, "instrument": ins,
           "microscope_sheet": sheet.get(sid, {}).get("Microscope"),
           "genotype": sheet.get(sid, {}).get("Genotype") or "unknown",
           "date": re.search(r"(\d{4}-\d{2}-\d{2})", subs[sid]["root"]).group(1),
           "group": "bright" if sid in BRIGHT else "dim"}  # fmt: skip
    if v1 is not None:
        boot = [
            np.median(rng.choice(v5, len(v5))) / np.median(rng.choice(v1, len(v1)))
            for _ in range(2000)
        ]
        row["ratio"] = (np.median(v5) / np.median(v1), *np.percentile(boot, [2.5, 97.5]))
    rows.append(row)
Path(f"{out}_table.json").write_text(json.dumps(
    [{k: (v if k not in ("v5", "v1") else (None if v is None else float(np.median(v)))) for k, v in r.items()}
     for r in rows], indent=1, default=float))  # fmt: skip
col = {"bright": A.BRAND["blue"], "dim": A.BRAND["orange"]}


def short(g):
    return g.replace("C57BL/6J + ", "").replace(" / ", "/").strip()


def grouped(field, title, order=None, rotate=False, by_date=False):
    groups = sorted({r[field] for r in rows}) if order is None else order
    fig, axs = plt.subplots(1, 3, figsize=(22, 7.5), sharex=not by_date)
    for ax, (name, getter) in zip(axs, (("5x median SNR", "v5"), ("1x median SNR", "v1"),
                                        ("5x / 1x SNR ratio", "ratio")), strict=True):  # fmt: skip
        xt = []
        for gi, g in enumerate(groups):
            members = sorted([r for r in rows if r[field] == g], key=lambda r: r["sid"])
            for k, r in enumerate(members):
                x = (np.datetime64(r["date"]).astype(float) if by_date
                     else gi + (k - (len(members) - 1) / 2) * 0.12)  # fmt: skip
                c = col[r["group"]]
                if getter == "ratio":
                    if "ratio" not in r:
                        continue
                    m, lo, hi = r["ratio"]
                else:
                    v = r[getter]
                    if v is None:
                        continue
                    lo, m, hi = np.percentile(v, [25, 50, 75])
                ax.plot([x, x], [lo, hi], color=c, lw=2, alpha=0.45)
                ax.scatter([x], [m], s=55, color=c, edgecolor="white", zorder=3)
                ax.annotate(r["sid"], (x, m), xytext=(4, 3), textcoords="offset points", fontsize=7)
            xt.append(g)
        if by_date:
            ticks = sorted({np.datetime64(r["date"]).astype(float) for r in rows})
            ax.set_xticks(ticks[::2])
            ax.set_xticklabels(
                [str(np.datetime64(int(x), "D"))[5:] for x in ticks[::2]], rotation=60, fontsize=8
            )
            ax.set_xlabel("5x acquisition date (2026, month-day)")
        else:
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([short(g) for g in xt], rotation=35 if rotate else 0, ha="right" if rotate else "center",
                               fontsize=9)  # fmt: skip
        ax.set_ylim(0, None)
        A.style_ax(ax, title=f"{name} · p >= {t:g}", title_size=13)
    axs[0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=col[g], label=g) for g in col],
                  frameon=False, loc="upper left")  # fmt: skip
    fig.suptitle(
        A.safe(f"SNR by {title}/"),
        x=0.01,
        ha="left",
        fontfamily=A.HEAD,
        fontweight="bold",
        fontsize=20,
    )
    fig.text(0.01, 0.01, A.safe(
        "dot: median block SNR, line: 25-75% of blocks (ratio: 95% bootstrap interval) · 5x: 19 brains (851474 "
        "excluded); 1x and ratio: the 10 brains with a same-day 1x scan · colour: bright/dim group"),
        fontsize=9, color=A.C["muted"])  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


ins_order = [
    g
    for g in ("exaSPIM-beta-01", "exaspim-5x-02", "missing")
    if any(r["instrument"] == g for r in rows)
]
A.apply_style("white")
f1 = grouped("instrument", "microscope (acquisition.json instrument_id)", order=ins_order)
A.savefig(f1, f"{out}_microscope.png", dpi=100)
f2 = grouped("genotype", "genotype / virus (ExM Dataset Summary)", rotate=True)
A.savefig(f2, f"{out}_genotype.png", dpi=100)
f3 = grouped("date", "5x acquisition date", order=sorted({r["date"] for r in rows}), by_date=True)
A.savefig(f3, f"{out}_date.png", dpi=100)
print("saved", out)
