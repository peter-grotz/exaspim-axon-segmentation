"""Train the dendrite-vs-axon skeleton classifier on simulated skeletons of traced neurons.

Usage::

    uv run python scripts/train_skeleton_classifier.py <traces root> <output dir> [--holdout 841260]

Every brain except the held-out one is simulated twice (fresh gaps, spurs, distractors); the
held-out brain once, with distractors drawn only from the training brains' neurons. Stops if the
process grows beyond --max-gb of memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import resource
import sys
from pathlib import Path

from exaspim_axon_segmentation.classify import train
from exaspim_axon_segmentation.classify.traces import load_traced_neurons

log = logging.getLogger("train_skeleton_classifier")


def rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1e9 if sys.platform == "darwin" else 1e6)  # bytes on macOS, kB on Linux


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--holdout", default="841260")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-gb", type=float, default=8.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    neurons = load_traced_neurons(args.traces)
    brains = sorted({n.brain for n in neurons} - {args.holdout})
    log.info("%d neurons; training brains %s; held out %s", len(neurons), brains, args.holdout)
    tables = []
    for brain in brains:
        tables += train.simulate_tables(
            neurons,
            [brain],
            repeats=args.repeats,
            exclude_from_pool=[args.holdout],
            seed=len(tables),
        )
        log.info("peak memory %.1f GB", rss_gb())
        if rss_gb() > args.max_gb:
            raise SystemExit(f"stopping: memory above {args.max_gb} GB")
    held = train.simulate_tables(
        neurons, [args.holdout], repeats=1, exclude_from_pool=[args.holdout], seed=99
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables.pkl").write_bytes(pickle.dumps({"train": tables, "holdout": held}))

    x, y, w = train.stack(tables)
    xt, yt, wt = train.stack(held)
    log.info("training on %d branches (%.0f mm)", len(y), w.sum() / 1e3)
    model = train.fit(x, y, w)
    scores = {"train": train.score(model, x, y, w), "holdout": train.score(model, xt, yt, wt)}
    for part, s in scores.items():
        log.info("%s: %s", part, s.text())
    importance = train.feature_importance(model, xt, yt, wt)
    for name, value in importance[:10]:
        log.info("  %-16s %.3f", name, value)
    train.save(model, args.out / "compartment_classifier.pkl")
    (args.out / "scores.json").write_text(
        json.dumps({k: vars(v) for k, v in scores.items()} | {"importance": importance}, indent=2)
    )


if __name__ == "__main__":
    main()
