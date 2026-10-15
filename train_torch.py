"""
Train the accuracy model with PyTorch on an NVIDIA GPU (same data, model, loss, schedule and outputs as train.py).

    python train_torch.py --replays replays_wide --lstm-units 32,24
    python train_torch.py ... --init runs/<run>/best.pt --lr 5e-4      # start from an earlier train_torch.py run
    python train_torch.py ... --final                                   # also train on the test maps (see train.py)

Why a separate trainer: Keras with the PyTorch backend runs the LSTMs one time step at a time, so even on an RTX 5090 it
is slower than TensorFlow on the CPU. torch.nn.LSTM uses cuDNN. The model is the same as model.build_model:

    2 bidirectional LSTMs over the 32-note window -> concat with the raw input -> keep the 8 predicted notes
    -> Dense(16, relu) -> Dense(16, relu) -> Dense(1)

with Keras' initialisation (glorot uniform, orthogonal recurrent weights, forget-gate bias 1), Adam with each gradient
clipped to norm 1 (Keras clipnorm), the masked MSE of model.py, ReduceLROnPlateau (x0.5) and early stopping on val_loss.

The ONNX export keeps the names RatingAPI uses: input "input_1" (batch, 32, NOTE_SIZE), output "time_distributed_2"
(batch, 8, 1). The export is checked against onnxruntime before the run ends.

Outputs in runs/<timestamp>/: best.pt, model_sleep_bl.onnx, history.csv, predictions_*.csv, summary.json
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import time

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
    p.add_argument("--lstm-units", default="12,10")
    p.add_argument("--dense-units", type=int, default=16)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--seed", type=int, default=6969)
    p.add_argument("--init", metavar="PT_FILE", help="start from an earlier train_torch.py run's best.pt (same model size)")
    p.add_argument("--no-player-skill", action="store_true")
    p.add_argument("--no-centering", action="store_true",
                   help="keep the whole player adjustment (no centering on the kind of map, data.PrepConfig.center_skill)")
    p.add_argument("--no-onnx", action="store_true")
    p.add_argument("--final", action="store_true",
                   help="also train on the test maps and replays_validation; only val is scored")
    p.add_argument("--all-data", metavar="HISTORY_CSV",
                   help="with --final: train on every map for exactly --epochs epochs with that run's learning rates")
    p.add_argument("--lr-drops", default="")
    return p.parse_args()


def build(note_size, lstm_units, dense_units, pre, pred_size):
    import torch
    from torch import nn

    class AccModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstms = nn.ModuleList()
            size = note_size
            for units in lstm_units:
                self.lstms.append(nn.LSTM(size, units, batch_first=True, bidirectional=True))
                size = units * 2
            self.dense1 = nn.Linear(size + note_size, dense_units)
            self.dense2 = nn.Linear(dense_units, dense_units)
            self.out = nn.Linear(dense_units, 1)
            self.pre, self.pred_size = pre, pred_size
            self._keras_init()

        def _keras_init(self):
            for lstm in self.lstms:
                for name, w in lstm.named_parameters():
                    if name.startswith("weight_ih"):
                        # Keras: one glorot-uniform kernel for the 4 gates together (input -> 4 * units)
                        nn.init.xavier_uniform_(w)
                    elif name.startswith("weight_hh"):
                        # Keras: orthogonal recurrent kernel (units -> 4 * units)
                        nn.init.orthogonal_(w)
                    elif name.startswith("bias_ih"):
                        nn.init.zeros_(w)
                        h = w.shape[0] // 4
                        with torch.no_grad():
                            w[h:2 * h] = 1.0       # forget gate (torch gate order: input, forget, cell, output)
                    elif name.startswith("bias_hh"):
                        nn.init.zeros_(w)          # Keras has one bias: keep the second one at 0
            for layer in (self.dense1, self.dense2, self.out):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        def forward(self, x):
            h = x
            for lstm in self.lstms:
                h, _ = lstm(h)
            h = torch.cat([h, x], dim=-1)[:, self.pre:self.pre + self.pred_size]
            h = torch.relu(self.dense1(h))
            h = torch.relu(self.dense2(h))
            return self.out(h)

    return AccModel()


def masked_losses(y, pred):
    """Same as model.masked_mse / masked_mae. y: (b, 8, 2) target + weight, pred: (b, 8, 1)."""
    import torch
    target, weight, p = y[..., 0], y[..., 1], pred[..., 0]
    used = (weight > 0).float()
    n = torch.clamp(used.sum(-1), min=1.0)
    mse = (weight * (target - p) ** 2).sum(-1) / n
    mae = (used * (target - p).abs()).sum(-1) / n
    return mse.mean(), mae.mean()


def main():
    args = parse_args()
    import torch

    import data
    import evaluation
    from train import lr_schedule

    if not torch.cuda.is_available():
        print("Warning: no CUDA GPU found, training on the CPU (use train.py for CPU training)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print(f"PyTorch {torch.__version__}, device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", stamp)
    os.makedirs(run_dir, exist_ok=True)

    # ---- data (same as train.py) ----------------------------------------------
    cfg = data.PrepConfig(player_skill_file="") if args.no_player_skill else data.PrepConfig(center_skill=not args.no_centering)
    maps = data.load_dataset(args.replays, cfg, workers=args.workers, rebuild=args.rebuild)
    handpicked = {}
    if args.validation and os.path.isdir(args.validation):
        handpicked = data.load_dataset(args.validation, cfg, workers=args.workers, rebuild=args.rebuild)
    excluded_songs = {data.song_group(lb) for lb in handpicked}
    train_ids, val_ids, test_ids = data.split_by_song(
        maps.keys(), args.val_fraction, args.test_fraction, exclude_songs=excluded_songs, seed=str(args.seed))
    print(f"Maps: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    if args.final:
        train_ids = train_ids + test_ids
        test_ids = []
        for lb, m in handpicked.items():
            maps.setdefault(lb, m)
            train_ids.append(lb)
        handpicked = {}
        if args.all_data:
            train_ids = train_ids + val_ids
            val_ids = []
        print(f"Final training: {len(train_ids)} maps in training, {len(val_ids)} in val.")
    elif args.all_data:
        raise SystemExit("--all-data only works together with --final.")

    train_x, train_y = data.build_arrays(maps, train_ids)
    val_x, val_y = data.build_arrays(maps, val_ids) if val_ids else (None, None)
    print(f"Windows: {len(train_x)} train, {0 if val_x is None else len(val_x)} val, input {train_x.shape[1:]}")
    # The whole set fits in GPU memory (a few GB): no per-batch copies
    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(train_y).to(device)
    vx = torch.from_numpy(val_x).to(device) if val_x is not None else None
    vy = torch.from_numpy(val_y).to(device) if val_y is not None else None
    del train_x, train_y, val_x, val_y

    # ---- model ----------------------------------------------------------------
    lstm_units = tuple(int(u) for u in args.lstm_units.split(","))
    model = build(data.NOTE_SIZE, lstm_units, args.dense_units, data.PRE_SEGMENT_SIZE, data.PREDICTION_SIZE).to(device)
    config = {"note_size": data.NOTE_SIZE, "lstm_units": lstm_units, "dense_units": args.dense_units}
    config = {**config, "lstm_units": list(lstm_units)}
    if args.init:
        saved = torch.load(args.init, map_location=device)
        if saved.get("config") != config:
            raise SystemExit(f"{args.init} was trained with {saved.get('config')}, this run uses {config}")
        model.load_state_dict(saved["state_dict"])
        print(f"Starting from the weights of {args.init}")
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_path = os.path.join(run_dir, "best.pt")
    lrs = lr_schedule(args.all_data, args.lr, args.epochs, args.lr_drops) if args.all_data else None

    def evaluate_loss(x, y, batch=8192):
        model.eval()
        total_mse, total_mae, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(x), batch):
                mse, mae = masked_losses(y[i:i + batch], model(x[i:i + batch]))
                k = len(x[i:i + batch])
                total_mse += mse.item() * k
                total_mae += mae.item() * k
                n += k
        return total_mse / n, total_mae / n

    def save():
        torch.save({"state_dict": model.state_dict(), "config": config}, best_path)

    history = []
    # Keras semantics: the checkpoint keeps any lower val_loss; early stopping and the learning-rate drops only count
    # improvements of at least min_delta
    best, best_epoch = float("inf"), -1
    stop_best, wait = float("inf"), 0
    plateau_best, plateau_wait = float("inf"), 0
    lr_patience = max(2, args.patience // 3)
    for epoch in range(args.epochs):
        lr = lrs[epoch] if lrs else opt.param_groups[0]["lr"]
        for g in opt.param_groups:
            g["lr"] = lr
        started = time.time()
        model.train()
        perm = torch.randperm(len(tx), device=device)
        total_mse = torch.zeros((), device=device)
        total_mae = torch.zeros((), device=device)
        n = 0
        for i in range(0, len(tx), args.batch_size):
            idx = perm[i:i + args.batch_size]
            mse, mae = masked_losses(ty[idx], model(tx[idx]))
            opt.zero_grad(set_to_none=True)
            mse.backward()
            for p in model.parameters():   # Keras clipnorm=1.0: each gradient clipped on its own
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_([p], 1.0)
            opt.step()
            total_mse += mse.detach() * len(idx)     # stays on the GPU: no sync every batch
            total_mae += mae.detach() * len(idx)
            n += len(idx)
        row = {"epoch": epoch, "learning_rate": lr, "loss": total_mse.item() / n, "masked_mae": total_mae.item() / n}
        msg = f"Epoch {epoch + 1}/{args.epochs} - {time.time() - started:.0f}s - loss {row['loss']:.6f} - mae {row['masked_mae']:.5f}"

        if vx is not None:
            row["val_loss"], row["val_masked_mae"] = evaluate_loss(vx, vy)
            msg += f" - val_loss {row['val_loss']:.6f} - val_mae {row['val_masked_mae']:.5f} - lr {lr:g}"
            if row["val_loss"] < best:
                best, best_epoch = row["val_loss"], epoch
                save()
                msg += " (saved)"
            if row["val_loss"] < stop_best - args.min_delta:
                stop_best, wait = row["val_loss"], 0
            else:
                wait += 1
            if not lrs:
                # ReduceLROnPlateau (Keras semantics): halve after lr_patience epochs without a min_delta improvement
                if row["val_loss"] < plateau_best - args.min_delta:
                    plateau_best, plateau_wait = row["val_loss"], 0
                else:
                    plateau_wait += 1
                    if plateau_wait >= lr_patience and lr > 1e-5:
                        for g in opt.param_groups:
                            g["lr"] = max(lr * 0.5, 1e-5)
                        plateau_wait = 0
                        msg += f" -> lr {max(lr * 0.5, 1e-5):g}"
        else:
            save()
        print(msg, flush=True)
        history.append(row)
        with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
        if not row["loss"] == row["loss"]:   # NaN
            raise SystemExit("Loss is NaN, stopping")
        if vx is not None and wait >= args.patience:
            print(f"Early stopping: no improvement of {args.min_delta} in {args.patience} epochs (best epoch {best_epoch + 1})")
            break

    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    model.eval()

    def predict(X: np.ndarray, batch=8192) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                out.append(model(torch.from_numpy(X[i:i + batch]).to(device)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, data.PREDICTION_SIZE, 1), np.float32)

    # ---- evaluation (same as train.py) ------------------------------------------
    summary = {"run": stamp, "trainer": "torch", "final": args.final, "all_data_from": args.all_data, "init": args.init,
               "epochs": args.epochs, "best_epoch": best_epoch + 1, "lstm_units": args.lstm_units,
               "dense_units": args.dense_units, "batch_size": args.batch_size, "lr": args.lr,
               "train_maps": len(train_ids), "note_size": data.NOTE_SIZE, "prep": cfg.key(),
               "centering": bool(cfg.player_skill_file) and cfg.center_skill}
    for name, source, ids in [("val", maps, val_ids), ("test", maps, test_ids), ("handpicked", handpicked, sorted(handpicked))]:
        if not ids:
            continue
        rows = evaluation.evaluate_maps(predict, source, ids)
        summary[name] = evaluation.summarize(rows, name)
        summary[name + "_by_type"] = evaluation.breakdown(rows, name)
        evaluation.write_csv(rows, os.path.join(run_dir, f"predictions_{name}.csv"))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if not args.no_onnx:
        export_onnx(model, os.path.join(run_dir, "model_sleep_bl.onnx"), data, predict, maps, val_ids or train_ids)
    print(f"Done. Results in {run_dir}")


def export_onnx(model, path, data, predict, maps, ids):
    """Exports with RatingAPI's names and checks the ONNX model against PyTorch on real windows."""
    import torch
    cpu_model = model.to("cpu").eval()
    dummy = torch.zeros(2, data.SEGMENT_SIZE, data.NOTE_SIZE)
    kwargs = dict(input_names=["input_1"], output_names=["time_distributed_2"],
                  dynamic_axes={"input_1": {0: "batch"}, "time_distributed_2": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(cpu_model, (dummy,), path, dynamo=False, **kwargs)
    except TypeError:          # older PyTorch without the dynamo argument
        torch.onnx.export(cpu_model, (dummy,), path, **kwargs)
    except Exception as e:     # newer PyTorch without the TorchScript exporter: the dynamo one needs onnxscript
        print(f"TorchScript ONNX export failed ({e}), trying the dynamo exporter (pip install onnxscript)")
        try:
            torch.onnx.export(cpu_model, (dummy,), path, dynamo=True, **kwargs)
        except Exception as e2:
            print(f"ONNX export failed: {e2}. The model is in best.pt; send me the error.")
            return
    print(f"ONNX model saved to {path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    X = data.make_windows(maps[ids[0]].features).astype(np.float32)[:256]
    with torch.no_grad():
        expected = cpu_model(torch.from_numpy(X)).numpy()
    got = sess.run(None, {"input_1": X})[0]
    diff = float(np.abs(expected - got).max())
    print(f"ONNX check: input {sess.get_inputs()[0].name} {sess.get_inputs()[0].shape}, "
          f"output {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}, max difference vs PyTorch {diff:.2e}")
    if diff > 1e-4:
        print("Warning: the ONNX model doesn't match PyTorch, don't use it")


if __name__ == "__main__":  # required on Windows because preprocessing uses multiple processes
    main()
