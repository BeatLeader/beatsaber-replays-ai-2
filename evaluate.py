"""
Evaluate a trained model (.keras or .onnx) on a folder of leaderboards.

    python evaluate.py runs/<timestamp>/best.keras
    python evaluate.py runs/<timestamp>/model_sleep_bl.onnx --replays replays_validation --plot
    python evaluate.py model.onnx --maps 338afxx91 3aa06xxxxxxxxxxx91 --plot

Compare two models on the same maps, e.g. the current BeatLeader model on a run's test maps:
    python evaluate.py runs/<timestamp>/model_sleep_bl.onnx --maps-from runs/<timestamp>/predictions_test.csv
    python evaluate.py old_model.onnx --maps-from runs/<timestamp>/predictions_test.csv
(with --maps-from the maps are read from replays/, or from replays_validation/ for a predictions_handpicked CSV)
Old 49-value models are detected from the ONNX input size and get the old note layout automatically.

Writes a CSV next to the model and prints map-level errors. --plot shows the smoothed
per-note real vs predicted accuracy for each map (like the old notebook's cell 7).
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import data
import evaluation


def load_predictor(model_path: str):
    if model_path.endswith(".onnx"):
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        use_layout(session.get_inputs()[0].shape[-1])

        def predict(X):
            outs = []
            for i in range(0, len(X), 1024):
                outs.append(session.run(None, {input_name: X[i:i + 1024]})[0])
            return np.concatenate(outs)
        return predict

    import keras
    import model as _  # registers the custom loss / metric
    from model import keras_predictor
    loaded = keras.models.load_model(model_path, compile=False)
    use_layout(loaded.input_shape[-1])
    return keras_predictor(loaded)


def use_layout(model_note_size: int):
    """Build note features in the layout the model was trained with (53 = with note types, 49 = old model)."""
    if model_note_size == data.BASE_NOTE_SIZE + len(data.FEATURE_SCORING_TYPES):
        return
    if model_note_size != data.BASE_NOTE_SIZE:
        raise SystemExit(f"Model expects {model_note_size} values per note; supported: "
                         f"{data.BASE_NOTE_SIZE} or {data.BASE_NOTE_SIZE + len(data.FEATURE_SCORING_TYPES)}.")
    print(f"Old {data.BASE_NOTE_SIZE}-value model: using the layout without note types.")
    os.environ["ACCAI_NOTE_TYPES"] = "0"  # also seen by the preprocessing worker processes
    data.USE_SCORING_TYPE_FEATURES = False
    data.NOTE_SIZE = data.BASE_NOTE_SIZE


def read_map_ids(csv_path: str) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [row["LeaderboardId"] for row in csv.DictReader(f)]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--replays", default=None,
                   help="folder of leaderboards (default: replays/ with --maps-from, except for a predictions_handpicked "
                        "CSV; replays_validation/ otherwise)")
    p.add_argument("--maps", nargs="*", help="only these leaderboard ids")
    p.add_argument("--maps-from", help="only the leaderboards listed in a predictions CSV (e.g. a run's predictions_test.csv)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    if args.replays is None:
        # A run's val/test maps live in replays/; its hand-picked maps (and everything else) in replays_validation/.
        from_run = args.maps_from and "handpicked" not in os.path.basename(args.maps_from)
        args.replays = os.path.join(here, "replays" if from_run else "replays_validation")
        print(f"Using {args.replays}")

    predict = load_predictor(args.model)
    maps = data.load_dataset(args.replays, data.PrepConfig(), workers=args.workers, rebuild=args.rebuild)
    wanted = args.maps or (read_map_ids(args.maps_from) if args.maps_from else sorted(maps))
    ids = [lb for lb in wanted if lb in maps]
    if len(ids) < len(wanted):
        print(f"{len(wanted) - len(ids)} of {len(wanted)} requested leaderboards are not in {args.replays}")

    rows = evaluation.evaluate_maps(predict, maps, ids)
    for r in rows:
        print(f"{r['LeaderboardId']:>22}  predicted {r['Prediction']:.3f}  real {r['Expected']:.3f}"
              f"  diff {r['Difference']:+.3f}  note MAE {r['NoteMAE']:.4f}")
    name = os.path.basename(os.path.normpath(args.replays))
    evaluation.summarize(rows, name)
    evaluation.breakdown(rows, name)

    suffix = os.path.splitext(os.path.basename(args.maps_from))[0] if args.maps_from else name
    out = os.path.splitext(args.model)[0] + f"_{suffix}.csv"
    evaluation.write_csv(rows, out)
    print(f"Saved {out}")

    if args.plot:
        import matplotlib.pyplot as plt
        k = 15
        for lb in ids:
            m = maps[lb]
            pred = evaluation.predict_map(predict, m)
            mask = m.weights > 0
            real = np.convolve(m.targets[mask], np.ones(k) / k, mode="valid")
            predicted = np.convolve(pred[mask], np.ones(k) / k, mode="valid")
            plt.figure(figsize=(10, 4))
            plt.plot(real, label="real")
            plt.plot(predicted, label="predicted")
            plt.title(lb)
            plt.xlabel("note")
            plt.ylabel(f"accuracy component ({k}-note average)")
            plt.legend()
            plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
