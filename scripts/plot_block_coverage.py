"""5x blocks per brain: blocks sampled, blocks scored at a threshold, and the share of the brain's tissue blocks scored.

Tissue blocks per tile (non-overlapping block positions in tissue): tiles with a brain mask use
the selection's n_qualifying_blocks (positions with enough mask tissue); tiles without one are
estimated as all block positions in the tile x the share of candidates that passed the tissue
check (p99.9 - median >= 3 at level 4). Neighbouring tiles overlap, so the per-tile counts include
the overlap twice; sampled blocks are drawn the same way, so the share is still comparable.

Usage: uv run python scripts/plot_block_coverage.py <out.png> <threshold> <subject>:<5x rescore dir> ...
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

from exaspim_axon_segmentation.snr import open_ome

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

out, t, specs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
key = f"p{t:g}"
rows = []
for spec in specs:
    sid, five = spec.split(":")
    n_blk = n_ok = 0
    tissue = est = 0.0
    for f in sorted(Path(five).glob("*/snr_rescore.json")):
        d = json.loads(f.read_text())
        bl = d["blocks"]
        n_blk += len(bl)
        n_ok += sum(b[key].get("snr") is not None and b[key].get("noise_sd", 1) > 0 for b in bl)
        if d["selection"] == "brain mask":
            tissue += d["n_qualifying_blocks"]
        else:
            tc = d["tissue_check"]
            arr, lead, sc0, _ = open_ome(d["url"], 0)
            blk0 = np.array(bl[0]["shape"] if bl else [321, 483, 483]) * 4  # level-2 block -> level 0
            total = int(np.prod(np.array(arr.shape[lead:]) // blk0))
            e = total * (tc["checked"] - tc["rejected"]) / max(tc["checked"], 1)
            tissue += e
            est += e
    rows.append({"sid": sid, "blocks": n_blk, "scored": n_ok, "tissue": tissue, "est_share": est / tissue,
                 "pct": 100 * n_ok / tissue})  # fmt: skip
    print(sid, rows[-1], flush=True)

A.apply_style("white")
fig, axs = plt.subplots(1, 3, figsize=(20, 6.5))
x = np.arange(len(rows))
for ax, (k, title, fmt) in zip(axs, (("blocks", "5x blocks sampled", "{:.0f}"), ("scored", f"5x blocks scored · p >= {t:g}", "{:.0f}"),
                                     ("pct", "% of tissue blocks scored", "{:.0f}%")), strict=True):  # fmt: skip
    v = [r[k] for r in rows]
    ax.bar(x, v, color=A.BRAND["blue"], width=0.7)
    for xi, vi in zip(x, v, strict=True):
        ax.text(xi, vi, fmt.format(vi), ha="center", va="bottom", fontsize=11, fontfamily=A.HEAD, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r["sid"] for r in rows], rotation=45, ha="right")
    ax.set_ylim(0, max(v) * 1.12)
    A.style_ax(ax, title=title, title_size=14)
fig.suptitle(A.safe(f"5x block sampling per brain / {len(rows)} brains with a same-day 1x scan"), x=0.01,
             ha="left", fontfamily=A.HEAD, fontweight="bold", fontsize=20)  # fmt: skip
fig.text(0.01, 0.01, A.safe(
    "sampled: up to 40 blocks per tile (1.28 x 1.44 x 1.44 mm) that pass the tissue check · scored: blocks with an SNR at "
    f"p >= {t:g} · tissue blocks: non-overlapping block positions in tissue, from the brain mask, or for tiles "
    "without one estimated from the tissue-check pass rate (787423: no masks)"),
    fontsize=9, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
A.savefig(fig, out, dpi=100)
print("saved", out)
