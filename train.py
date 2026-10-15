"""
Train the accuracy model.

    python train.py                          # train on replays/, evaluate, export ONNX
    python train.py --rebuild                # ignore the preprocessing cache
    python train.py --epochs 200 --batch-size 512
    python train.py --no-player-skill        # raw replay accuracy as targets (no player adjustment)
    python train.py --backend torch          # train on the GPU through PyTorch (native Windows TensorFlow is CPU only);
                                             # the ONNX export still goes through TensorFlow, in a separate process
    python train.py --init runs/<run>/best.keras --lr 5e-4
                                             # start from an earlier model (same --lstm-units / --dense-units) instead of
                                             # random weights: when only the targets changed a little, it converges in a
                                             # fraction of the epochs. Train the final model from scratch.

Outputs go to runs/<timestamp>/:
    best.keras            best weights by validation loss
    model_sleep_bl.onnx   ONNX export of the best model
    history.csv           per-epoch losses
    predictions_*.csv     map-level predictions for the val / test / hand-picked sets
    summary.json          the headline numbers
TensorBoard logs: logs/fit/<timestamp>  (tensorboard --logdir logs)

Validation is automatic: replays/ is split by song into train (70%), val (15%, picks the best epoch) and
test (15%, never used for training). Maps in replays_validation/ are excluded from all three and scored too.
Every set is reported overall and by map type (notes per second, NJS, real accuracy).

    python train.py --final                  # same settings, but also train on the test maps and replays_validation
                                             # (val still picks the best epoch); use the test score of a normal run
                                             # as this model's expected error

    python train.py --final --all-data runs/<normal run>/history.csv --epochs 30
                                             # train on every map, val included, for exactly 30 epochs (the normal
                                             # run's best epoch) with that run's learning-rate drops; keeps the last
                                             # epoch. Nothing is scored; the normal run's test score is the expected error
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--replays", default=os.path.join(here, "replays"))
    p.add_argument("--validation", default=os.path.join(here, "replays_validation"),
                   help="hand-picked maps, never trained on (pass '' to skip)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lstm-units", default="12,10",
                   help="units of each bidirectional LSTM layer, e.g. 32,24 for a bigger model")
    p.add_argument("--dense-units", type=int, default=16)
    p.add_argument("--patience", type=int, default=8,
                   help="epochs without a real improvement (see --min-delta) before stopping")
    p.add_argument("--min-delta", type=float, default=1e-5,
                   help="smallest val_loss decrease that counts as an improvement for early stopping")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=None, help="processes for preprocessing (default: all cores)")
    p.add_argument("--rebuild", action="store_true", help="rebuild the preprocessing cache")
    p.add_argument("--seed", type=int, default=6969)
    p.add_argument("--backend", choices=["tensorflow", "torch"], default=os.environ.get("KERAS_BACKEND", "tensorflow"),
                   help="Keras backend; torch trains on an NVIDIA GPU on native Windows (pip install torch, CUDA build)")
    p.add_argument("--init", metavar="KERAS_FILE",
                   help="start from the weights of an earlier run's best.keras (same model size) instead of random weights")
    p.add_argument("--no-player-skill", action="store_true",
                   help="train on the raw replay accuracy instead of the player-adjusted targets (data.PrepConfig)")
    p.add_argument("--no-onnx", action="store_true")
    p.add_argument("--final", action="store_true",
                   help="also train on the test maps and replays_validation; only val is scored")
    p.add_argument("--all-data", metavar="HISTORY_CSV",
                   help="with --final: train on every map (val too) for exactly --epochs epochs, replaying the "
                        "learning rates of a normal run's history.csv; set --epochs to that run's best epoch")
    p.add_argument("--lr-drops", default="",
                   help="with --all-data, if the history.csv has no learning_rate column: epochs (as printed, "
                        "1-based) after which the learning rate was halved, e.g. 18,24")
    return p.parse_args()


def lr_schedule(history_csv: str, base_lr: float, epochs: int, drops: str) -> list[float]:
    """Learning rate of each epoch, taken from a previous run so a fixed-length run follows the same path."""
    import csv
    with open(history_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "learning_rate" in rows[0] and not drops:
        lrs = [float(r["learning_rate"]) for r in rows]
    else:
        if not drops:
            raise SystemExit(f"{history_csv} has no learning_rate column. Look for 'ReduceLROnPlateau reducing "
                             f"learning rate' in that run's log and pass the epochs with --lr-drops, e.g. 18,24.")
        after = sorted(int(e) for e in drops.split(","))
        lrs = [max(base_lr * 0.5 ** sum(1 for d in after if d < e + 1), 1e-5) for e in range(epochs)]
    if len(lrs) < epochs:
        print(f"History has {len(lrs)} epochs; keeping the last learning rate for the rest.")
        lrs += [lrs[-1]] * (epochs - len(lrs))
    return lrs[:epochs]


def export_onnx(model, path, keras_path=None):
    import keras
    if keras.backend.backend() != "tensorflow":
        # tf2onnx needs a TensorFlow model: reload the saved .keras (backend independent) with TensorFlow in a new process
        env = dict(os.environ, KERAS_BACKEND="tensorflow")
        result = subprocess.run([sys.executable, os.path.abspath(__file__), "--export-onnx", keras_path, path], env=env)
        if result.returncode != 0:
            print(f"ONNX export failed; the model is in {keras_path}. Export it later with: "
                  f"python train.py --export-onnx {keras_path} {path}")
        return
    import tensorflow as tf
    import data
    spec = (tf.TensorSpec((None, data.SEGMENT_SIZE, data.NOTE_SIZE), tf.float32, name="input_1"),)
    try:
        import tf2onnx
        tf2onnx.convert.from_keras(model, input_signature=spec, opset=None, output_path=path)
    except Exception as e:
        print(f"tf2onnx export failed ({e}), trying keras export")
        model.export(path, format="onnx")
    print(f"ONNX model saved to {path}")


def export_onnx_from_file(keras_path, onnx_path):
    import keras
    import model as _model  # registers the custom loss / metric
    export_onnx(keras.models.load_model(keras_path, compile=False), onnx_path)


def main():
    args = parse_args()
    # The backend has to be chosen before Keras is imported.
    os.environ["KERAS_BACKEND"] = args.backend

    # Imports after argparse so `python train.py -h` is instant.
    import keras

    import data
    import evaluation
    from model import build_model, compile_model, keras_predictor

    keras.utils.set_random_seed(args.seed)
    if args.backend == "torch":
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)"
        print(f"PyTorch {torch.__version__}, Keras {keras.__version__}, GPU: {gpu}")
    else:
        import tensorflow as tf
        print(f"TensorFlow {tf.__version__}, Keras {keras.__version__}, GPUs: {tf.config.list_physical_devices('GPU')}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", stamp)
    os.makedirs(run_dir, exist_ok=True)

    # ---- data -------------------------------------------------------------
    cfg = data.PrepConfig(player_skill_file="") if args.no_player_skill else data.PrepConfig()
    maps = data.load_dataset(args.replays, cfg, workers=args.workers, rebuild=args.rebuild)

    handpicked = {}
    if args.validation and os.path.isdir(args.validation):
        handpicked = data.load_dataset(args.validation, cfg, workers=args.workers, rebuild=args.rebuild)
    excluded_songs = {data.song_group(lb) for lb in handpicked}

    train_ids, val_ids, test_ids = data.split_by_song(
        maps.keys(), args.val_fraction, args.test_fraction, exclude_songs=excluded_songs, seed=str(args.seed))
    n_excluded = len(maps) - len(train_ids) - len(val_ids) - len(test_ids)
    print(f"Maps: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test"
          f" ({n_excluded} excluded because the song is in the hand-picked set)")

    if args.final:
        # Final model: everything except val goes into training. Val still picks the best epoch.
        train_ids = train_ids + test_ids
        test_ids = []
        for lb, m in handpicked.items():
            maps.setdefault(lb, m)
            train_ids.append(lb)
        handpicked = {}
        if args.all_data:
            train_ids = train_ids + val_ids
            val_ids = []
        print(f"Final training: {len(train_ids)} maps in training, {len(val_ids)} in val. "
              f"Use the test score of a normal run as this model's expected error.")
    elif args.all_data:
        raise SystemExit("--all-data only works together with --final.")

    train_x, train_y = data.build_arrays(maps, train_ids)
    val_x, val_y = data.build_arrays(maps, val_ids) if val_ids else (None, None)
    print(f"Windows: {len(train_x)} train, {0 if val_x is None else len(val_x)} val, input {train_x.shape[1:]}")
    if len(train_x) == 0 or (val_x is not None and len(val_x) == 0) or (val_x is None and not args.all_data):
        raise SystemExit("Not enough data to train (need maps in both train and val).")

    # ---- model ------------------------------------------------------------
    lstm_units = tuple(int(u) for u in args.lstm_units.split(","))
    model = compile_model(build_model(data.NOTE_SIZE, lstm_units, args.dense_units), args.lr)
    if args.init:
        start = keras.models.load_model(args.init, compile=False)
        try:
            model.set_weights(start.get_weights())
        except ValueError as e:
            raise SystemExit(f"{args.init} doesn't match this model (check --lstm-units / --dense-units): {e}")
        print(f"Starting from the weights of {args.init}")
    model.summary()

    best_path = os.path.join(run_dir, "best.keras")

    class LogLearningRate(keras.callbacks.Callback):
        # Writes the learning rate used during each epoch to history.csv, so --all-data can replay it later.
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                logs["learning_rate"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))

    callbacks = [LogLearningRate()]
    if args.all_data:
        # No val maps: run exactly --epochs epochs with the learning rates of the earlier run, keep the last epoch.
        lrs = lr_schedule(args.all_data, args.lr, args.epochs, args.lr_drops)
        print("Learning rate per epoch: " + ", ".join(f"{lr:g}" for lr in lrs))
        callbacks += [
            keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs[min(epoch, len(lrs) - 1)]),
            keras.callbacks.ModelCheckpoint(best_path, save_best_only=False, verbose=0),
        ]
    else:
        callbacks += [
            keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
            # Stops once val_loss hasn't dropped by at least min_delta for `patience` epochs. The checkpoint above
            # still keeps the very best epoch, and that's the model reloaded below.
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, min_delta=args.min_delta,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, args.patience // 3),
                                              min_delta=args.min_delta, min_lr=1e-5, verbose=1),
        ]
    callbacks += [
        keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]
    if args.backend == "tensorflow":   # the TensorBoard callback writes through TensorFlow
        callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join("logs", "fit", stamp)))

    model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y) if val_x is not None else None,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    # EarlyStopping restores the best weights only when it triggers; reload to be sure.
    model = keras.models.load_model(best_path, compile=False)

    # ---- evaluation ---------------------------------------------------------
    predict = keras_predictor(model)
    summary = {"run": stamp, "final": args.final, "all_data_from": args.all_data, "init": args.init, "epochs": args.epochs,"lstm_units": args.lstm_units, "dense_units": args.dense_units, "train_maps": len(train_ids), "note_size": data.NOTE_SIZE, "prep": cfg.key()}
    for name, source, ids in [("val", maps, val_ids), ("test", maps, test_ids),
                              ("handpicked", handpicked, sorted(handpicked))]:
        if not ids:
            continue
        rows = evaluation.evaluate_maps(predict, source, ids)
        summary[name] = evaluation.summarize(rows, name)
        summary[name + "_by_type"] = evaluation.breakdown(rows, name)
        evaluation.write_csv(rows, os.path.join(run_dir, f"predictions_{name}.csv"))
        if name == "handpicked":
            for r in rows:
                print(f"  {r['LeaderboardId']:>22}  predicted {r['Prediction']:.3f}  real {r['Expected']:.3f}"
                      f"  diff {r['Difference']:+.3f}")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if not args.no_onnx:
        export_onnx(model, os.path.join(run_dir, "model_sleep_bl.onnx"), best_path)
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":  # required on Windows because preprocessing uses multiple processes
    if len(sys.argv) == 4 and sys.argv[1] == "--export-onnx":   # used by export_onnx with the torch backend
        export_onnx_from_file(sys.argv[2], sys.argv[3])
    else:
        main()
