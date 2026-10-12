"""Map-level evaluation helpers (no TensorFlow import, so they also work with an ONNX model)."""

from __future__ import annotations

import csv

import numpy as np

import data


def predict_map(predict, m: data.MapData) -> np.ndarray:
    """Per-note predicted accuracy component for one map, shape (n_notes,)."""
    X = data.make_windows(m.features)
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    pred = np.asarray(predict(X.astype(np.float32)))
    return pred.reshape(-1)[: len(m.features)]


def evaluate_maps(predict, maps: dict[str, data.MapData], leaderboard_ids) -> list[dict]:
    """Map-level comparison: predicted vs real average accuracy over the trusted notes."""
    rows = []
    for lb in leaderboard_ids:
        m = maps[lb]
        pred = predict_map(predict, m)
        mask = m.weights > 0
        if mask.sum() == 0 or len(pred) == 0:
            continue
        predicted = float(data.acc_to_percent(pred[mask].mean()))
        expected = float(data.acc_to_percent(m.targets[mask].mean()))
        duration = float(m.times[-1] - m.times[0]) if len(m.times) > 1 else 0.0
        rows.append({
            "LeaderboardId": lb,
            "Url": f"https://beatleader.xyz/leaderboard/global/{lb}",
            "Prediction": round(predicted, 4),
            "Expected": round(expected, 4),
            "Difference": round(expected - predicted, 4),
            "NoteMAE": round(float(np.abs(pred[mask] - m.targets[mask]).mean()), 5),
            "LeaderboardAccuracy": round(m.leaderboard_accuracy * 100, 4),
            "Notes": int(mask.sum()),
            "Replays": m.n_replays,
            "NJS": round(float(m.njs), 2),
            "NPS": round(len(m.times) / duration, 2) if duration > 0 else 0.0,
        })
    return rows


def summarize(rows: list[dict], name: str) -> dict:
    if not rows:
        print(f"{name}: no maps")
        return {}
    diff = np.array([r["Difference"] for r in rows])
    s = {
        "maps": len(rows),
        "map_mae_pct": float(np.abs(diff).mean()),
        "map_bias_pct": float(diff.mean()),          # > 0 means the model predicts too low
        "map_max_err_pct": float(np.abs(diff).max()),
        "note_mae": float(np.mean([r["NoteMAE"] for r in rows])),
    }
    print(f"{name}: {s['maps']} maps | map MAE {s['map_mae_pct']:.3f} pts | bias {s['map_bias_pct']:+.3f} pts"
          f" | worst {s['map_max_err_pct']:.3f} pts | note MAE {s['note_mae']:.4f}")
    return s


# Map groups for the breakdown: the weak spots of the previous model were low-density (acc) maps,
# extreme speed and uncommon NJS. Each entry: (name, field, [(label, low, high), ...]).
GROUPS = [
    ("Notes per second", "NPS", [("under 2", 0, 2), ("2 to 4", 2, 4), ("4 to 6", 4, 6), ("6 to 8", 6, 8),
                                 ("8 to 10", 8, 10), ("10 and up", 10, float("inf"))]),
    ("NJS", "NJS", [("under 12", 0, 12), ("12 to 16", 12, 16), ("16 to 20", 16, 20), ("20 to 24", 20, 24),
                    ("24 and up", 24, float("inf"))]),
    ("Real accuracy", "Expected", [("under 96%", 0, 96), ("96 to 97%", 96, 97), ("97 to 98%", 97, 98),
                                   ("98 to 99%", 98, 99), ("99% and up", 99, float("inf"))]),
]


def breakdown(rows: list[dict], name: str = "") -> dict:
    """Map error per group (notes per second, NJS, real accuracy). Returns {group: {bucket: stats}}."""
    result = {}
    if not rows:
        return result
    if name:
        print(f"  {name} by map type (MAE / bias in points, positive bias = predicted too low):")
    for group, field, buckets in GROUPS:
        result[group] = {}
        parts = []
        for label, low, high in buckets:
            diffs = np.array([r["Difference"] for r in rows if low <= r[field] < high])
            if len(diffs) == 0:
                continue
            stats = {"maps": int(len(diffs)), "map_mae_pct": float(np.abs(diffs).mean()),
                     "map_bias_pct": float(diffs.mean())}
            result[group][label] = stats
            parts.append(f"{label}: {stats['map_mae_pct']:.2f}/{stats['map_bias_pct']:+.2f} ({stats['maps']})")
        if name:
            print(f"    {group:<17} " + " | ".join(parts))
    return result


def write_csv(rows: list[dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
