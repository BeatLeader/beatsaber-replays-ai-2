# BeatLeader AccAI

Predicts per-note accuracy (and so the expected map accuracy) of top players on a Beat Saber map,
trained on BeatLeader replays. The model is used by RatingAPI (in portaBLe) for each map's predicted acc and acc rating.

## Setup

Python 3.12 in the repo's `.venv` (the old conda `tf-gpu` environment is not used anymore):

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Always run the scripts with `.venv\Scripts\python.exe` (plain `python` may be the Microsoft Store one).
TensorFlow runs on the CPU on Windows, which is fine for this model size. `requirements.txt` pins protobuf, ml_dtypes,
numpy and onnx to versions TensorFlow 2.18 works with; don't upgrade them on their own.

## From replays to a model in RatingAPI

1. **Download** with `BLReplaysPrepare` (C#):
   ```
   BLReplaysPrepare all --min-scores 8
   ```
   It writes `replays/<leaderboardId>/<playerId>-<accuracy>-<baseScore>-<njs>.npy`, one row per note:
   `noteID, accuracy, spawnTime, eventType` (older files have only the first 3 columns; they still load).
   Finished leaderboards are listed in `replays/completed.txt`, so a run can be stopped and resumed.
2. **Clean up** maps downloaded before a filter existed (both only list until you add `--move`, which moves
   folders to `replays_excluded/`; nothing is deleted):
   ```
   .venv\Scripts\python.exe remove_modded_maps.py --requirements 396 --move
   .venv\Scripts\python.exe remove_short_maps.py --move
   ```
3. **Train and check** (a normal run: train / val / test split):
   ```
   .venv\Scripts\python.exe train.py --lstm-units 32,24
   ```
   Note the test score and the best epoch (`Restoring model weights from the end of the best epoch: N`).
4. **Final model** with the same settings, trained on every map for exactly that many epochs:
   ```
   .venv\Scripts\python.exe train.py --lstm-units 32,24 --final --all-data runs\<normal run>\history.csv --epochs N
   ```
5. **Deploy**: copy `runs\<final run>\model_sleep_bl.onnx` to `..\portaBLe\RatingAPI\model_sleep_bl.onnx`
   and **Rebuild** RatingAPI (the model is embedded in the DLL), then portaBLe. RatingAPI reads the input size
   from the model, so 49- and 53-value models both work.

`data.py` is not run on its own: `train.py`, `evaluate.py` and `noise_floor.py` import it. Preprocessed data is
cached in `cache/` and rebuilt automatically when the replay folders change (folders added or removed, files added).
Add `--rebuild` only after changing preprocessing code or settings without touching the replays.

## BLReplaysPrepare

A leaderboard is kept when it has 200+ plays (nomods), 20+ notes, no excluded map requirement, the #1 score is
93%+, and at least `--min-scores` scores are within 0.5% of the #1. The replays of those top scores are downloaded.

| Option | Default | |
|---|---|---|
| `all` | | scan every leaderboard, most played first |
| `test <leaderboardId>` | | one leaderboard, into `replays_test/` |
| `ratetest [leaderboardId]` | | measure the API rate limit |
| `--min-scores` | 10 | scores within 0.5% of #1 needed (8 is used for the current data) |
| `--min-plays` | 200 | the scan stops at the first page entirely below this |
| `--min-notes` | 20 | BeatLeader's note count |
| `--exclude-requirements` | 396 | BeatLeader map requirement flags left out: Noodle 4 + Mapping Extensions 8 + VNJS 128 + Vivify 256. 0 keeps all |
| `--recheck-skipped` | | check again the leaderboards in `completed.txt` that have no folder (after lowering a filter) |
| `--top N` / `--out folder` | | keep the top N scores whatever their spread, into another folder |
| `--parallel-leaderboards` / `--parallel-replays` | 4 / 16 | |
| `--api-delay-ms` | 250 | BeatLeader allows 50 requests per 10 s; 250 ms stays under it |

## Training

- Train / val / test are split **by song** (70 / 15 / 15), so the difficulties of one song never end up on both sides.
- Songs in `replays_validation/` are excluded from training entirely, so that folder is a true hold-out set.
- Each run writes `runs/<timestamp>/`: `best.keras`, `model_sleep_bl.onnx`, `history.csv` (with the learning rate),
  `predictions_val.csv`, `predictions_test.csv`, `summary.json`. TensorBoard logs go to `logs/fit/<timestamp>`.
- The headline metric is **map MAE in percentage points**: predicted vs real average accuracy per map, for val, test
  and the hand-picked maps. Each set is also broken down by notes per second, NJS and real accuracy.
- Early stopping: stops when `val_loss` hasn't improved by `--min-delta` (1e-5) for `--patience` (8) epochs;
  the learning rate is halved on plateaus. The best epoch is kept.
- `--lstm-units 32,24` is the model in use (the default `12,10` is the original, smaller one).
- `--final` trains on train + test + `replays_validation/` (val still picks the best epoch).
  `--final --all-data <history.csv> --epochs N` also trains on val, for exactly N epochs with the same learning-rate
  drops, and keeps the last epoch. Neither is scored: use the test score of the normal run as the model's expected error.
  (Runs from before `history.csv` had a `learning_rate` column need `--lr-drops 18,24` with the drop epochs from their log.)

## Checking a model

```
.venv\Scripts\python.exe evaluate.py runs\<run>\model_sleep_bl.onnx --maps-from runs\<run>\predictions_test.csv
.venv\Scripts\python.exe evaluate.py ..\portaBLe\RatingAPI\model_sleep_bl.onnx --maps-from runs\<run>\predictions_test.csv
.venv\Scripts\python.exe noise_floor.py --maps-from runs\<run>\predictions_test.csv
```

- `evaluate.py` scores any `.keras` or `.onnx` model; old 49-value models get the old input layout automatically.
  With `--maps-from` it uses the maps of a run's predictions CSV from `replays/` (or `replays_validation/` for a
  `predictions_handpicked.csv`); without it, the maps in `replays_validation/`. `--replays` picks another folder,
  `--plot` shows real vs predicted per note.
- A model trained with `--final` has seen those maps, so evaluating it only checks that the export works.
- `noise_floor.py` splits each map's players in two random halves and compares their targets: the error a perfect
  model would still have, to compare with the run's map MAE and note MAE.

## Settings worth knowing

- `data.PrepConfig`: at least 8 replays per map (`min_replays`), maps with fewer than 10 count less in training
  (`full_weight_replays`, 8 replays = 0.8), how many players must have cut a note well for its target to be trusted,
  mean vs median, and whether to skip maps with a missing NJS.
- `data.USE_SCORING_TYPE_FEATURES` (default `True`) adds 4 note-type features (normal, arc head,
  arc tail, chain head), so the model input is `(None, 32, 53)` instead of `(None, 32, 49)`.
  RatingAPI (`portaBLe/RatingAPI/Controllers/DataProcessing.cs`) builds the same values and reads the
  size from the ONNX model, so it works with either layout.
- Chain links are left out of the note sequence, as in RatingAPI, which counts them as fixed points.
- Notes are ordered by time, then x, y, direction and color, the same order RatingAPI uses.
- Jump distance and reaction time are deliberately not features: players change them freely.

## Using a notebook

Notebooks are fine for exploring; keep the pipeline in the `.py` files and import it:

```python
import data, evaluation
maps = data.load_dataset("replays")
```
