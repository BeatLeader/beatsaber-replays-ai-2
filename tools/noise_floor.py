"""
How good can the model get? Estimates the noise in the training targets themselves.

Each map's players are split at random into two halves, and both halves' targets are computed exactly like
training does. If the two halves disagree, that disagreement is noise in the data that no model can predict.
Scaled to the full player group, it gives the error a perfect model would still have against the real data
("noise floor"), to compare with the model's test error.

    python tools/noise_floor.py                                         # all maps in replays/
    python tools/noise_floor.py --maps-from runs/<timestamp>/predictions_test.csv   # same maps as a run's test set
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # data.py, evaluation.py
import data
import evaluation


def _map_value(m: data.MapData) -> float:
    mask = m.weights > 0
    return float(data.acc_to_percent(m.targets[mask].mean()))


def split_map(args):
    """Noise estimate for one map, averaged over several random splits."""
    folder, splits, seed = args
    cfg = data.PrepConfig()
    full, reason = data.load_map(folder, cfg)
    if full is None:
        return None
    files = data.list_replay_files(folder)
    n = len(files)
    a = n // 2
    b = n - a
    if a < 3:
        return None
    # Halves have fewer replays than the full-map minimum, so don't skip them for that.
    half_cfg = data.PrepConfig(min_replays=1, full_weight_replays=0)
    # Each half is an average of a or b players; the full target averages all n. Scaling the halves'
    # difference by sqrt(a*b)/n gives the noise of the full-group target.
    scale = math.sqrt(a * b) / n

    rng = random.Random(f"{seed}:{os.path.basename(folder)}")
    map_diffs, note_diffs = [], []
    for _ in range(splits):
        shuffled = files[:]
        rng.shuffle(shuffled)
        h1, _ = data.load_map(folder, half_cfg, shuffled[:a])
        h2, _ = data.load_map(folder, half_cfg, shuffled[a:])
        if h1 is None or h2 is None:
            continue
        map_diffs.append(abs(_map_value(h1) - _map_value(h2)) * scale)
        # Per-note: notes trusted in both halves
        t1 = {(int(i), round(float(t), 4)): v for i, t, v, w in zip(h1.note_ids, h1.times, h1.targets, h1.weights) if w > 0}
        diffs = [abs(t1[k] - v) for k, v in
                 (((int(i), round(float(t), 4)), v) for i, t, v, w in zip(h2.note_ids, h2.times, h2.targets, h2.weights) if w > 0)
                 if k in t1]
        if diffs:
            note_diffs.append(float(np.mean(diffs)) * scale)
    if not map_diffs:
        return None

    duration = float(full.times[-1] - full.times[0]) if len(full.times) > 1 else 0.0
    noise = float(np.mean(map_diffs))
    return {
        "LeaderboardId": full.leaderboard_id,
        "Replays": n,
        "Expected": round(_map_value(full), 4),
        "Difference": round(noise, 4),       # used by evaluation.breakdown: its MAE column is the noise floor
        "NoteNoise": round(float(np.mean(note_diffs)), 5) if note_diffs else float("nan"),
        "NJS": round(float(full.njs), 2),
        "NPS": round(len(full.times) / duration, 2) if duration > 0 else 0.0,
    }


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (this script is in tools/)
    p = argparse.ArgumentParser()
    p.add_argument("--replays", default=os.path.join(here, "replays"))
    p.add_argument("--maps-from", help="only the leaderboards in a predictions CSV (e.g. a run's predictions_test.csv)")
    p.add_argument("--splits", type=int, default=3, help="random splits per map (more = steadier estimate)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", default="6969")
    args = p.parse_args()

    ids = data.list_leaderboards(args.replays)
    if args.maps_from:
        with open(args.maps_from, newline="", encoding="utf-8") as f:
            wanted = {row["LeaderboardId"] for row in csv.DictReader(f)}
        ids = [lb for lb in ids if lb in wanted]
    jobs = [(os.path.join(args.replays, lb), args.splits, args.seed) for lb in ids]
    print(f"Splitting the players of {len(jobs)} maps in two halves ({args.splits} random splits each)...")

    if args.workers == 1:
        results = [split_map(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(split_map, jobs, chunksize=8))
    rows = [r for r in results if r is not None]
    if not rows:
        raise SystemExit("No maps with at least 6 replays.")

    map_noise = float(np.mean([r["Difference"] for r in rows]))
    note_noise = float(np.nanmean([r["NoteNoise"] for r in rows]))
    print(f"\n{len(rows)} maps")
    print(f"Noise floor, map level:  {map_noise:.3f} points  (compare with the run's 'map MAE')")
    print(f"Noise floor, note level: {note_noise:.4f}         (compare with the run's 'note MAE')")
    print("A perfect model would still be about this far from the real data, because the targets themselves are noisy.")
    evaluation.breakdown(rows, "Noise floor")
    print("  (for the noise floor only the first number of each pair matters; the second is not a bias)")

    out = os.path.join(here, "noise_floor.csv")
    evaluation.write_csv(rows, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
