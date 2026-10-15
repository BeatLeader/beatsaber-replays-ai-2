# BeatLeader AccAI

Predicts per-note accuracy (and so the expected map accuracy) of top players on a Beat Saber map,
trained on BeatLeader replays. The model is used by RatingAPI (in portaBLe) for each map's predicted acc and acc rating.

## Layout

| | |
|---|---|
| `data.py`, `model.py`, `evaluation.py` | preprocessing, model, map-level scoring (imported by the scripts) |
| `train_torch.py` | training on an NVIDIA GPU (PyTorch, cuDNN): the fast way to train |
| `train.py` | the same training with TensorFlow / Keras (CPU on Windows) |
| `evaluate.py` | scores a `.keras` / `.onnx` model on a folder of leaderboards |
| `tools/` | `player_skill.py` (player-adjusted targets), `noise_floor.py`, `remove_modded_maps.py`, `remove_short_maps.py` |
| `BLReplaysPrepare/` | C# downloader (`Research/`: parity and head-tracking exports used to validate the Analyzer) |
| `playlists/` | ranked AccSaber / ScoreSaber / BeatLeader playlists for `BLReplaysPrepare all --playlists` (not in git) |
| `legacy/` | the original notebooks, processors and models, kept for reference |

Data folders (`replays*`, `cache`, `runs`, `logs`, `parity*`, `heads`) and generated files (`player-skill.json`,
`skill-centering.json`) are not in git.

## Setup

Python 3.12 in the repo's `.venv` (the old conda `tf-gpu` environment is not used anymore):

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Always run the scripts with `.venv\Scripts\python.exe` (plain `python` may be the Microsoft Store one).
TensorFlow runs on the CPU on Windows. For `train_torch.py`, also install PyTorch with CUDA (RTX 50 series needs the
CUDA 12.8+ build): `.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128`. `requirements.txt` pins protobuf, ml_dtypes,
numpy and onnx to versions TensorFlow 2.18 works with; don't upgrade them on their own.

## From replays to a model in RatingAPI

1. **Download** with `BLReplaysPrepare` (C#). The model in use is trained on the top 50 scores of the ranked maps
   (wider than the tight top group, see Player-adjusted targets):
   ```
   BLReplaysPrepare all --playlists --top 50 --out replays_wide
   ```
   or the tight top groups of every leaderboard (the original data): `BLReplaysPrepare all --min-scores 8`.
   It writes `replays/<leaderboardId>/<playerId>-<accuracy>-<baseScore>-<njs>.npy`, one row per note:
   `noteID, accuracy, spawnTime, eventType` (older files have only the first 3 columns; they still load).
   Finished leaderboards are listed in `replays/completed.txt`, so a run can be stopped and resumed.
2. **Clean up** maps downloaded before a filter existed (both only list until you add `--move`, which moves
   folders to `replays_excluded/`; nothing is deleted):
   ```
   .venv\Scripts\python.exe tools\remove_modded_maps.py --requirements 396 --move
   .venv\Scripts\python.exe tools\remove_short_maps.py --move
   ```
3. **Player skills** (once per new scores export, see Player-adjusted targets):
   ```
   .venv\Scripts\python.exe tools\player_skill.py ..\portaBLeRebalancer\wwwroot
   ```
4. **Train and check** (a normal run: train / val / test split):
   ```
   .venv\Scripts\python.exe train_torch.py --replays replays_wide --lstm-units 32,24 --batch-size 1024 --lr 2e-3 --no-centering
   ```
   Note the test score and the best epoch (`best_epoch` in `summary.json`). `train.py` takes the same options
   (without `--no-centering`) and runs on the CPU.
5. **Final model** with the same settings, also trained on the test maps (val still picks the best epoch):
   ```
   .venv\Scripts\python.exe train_torch.py --replays replays_wide --lstm-units 32,24 --batch-size 1024 --lr 2e-3 --no-centering --final
   ```
6. **Deploy**: copy `runs\<final run>\model_sleep_bl.onnx` to `..\portaBLeRebalancer\RatingAPI\model_sleep_bl.onnx`
   and **Rebuild** RatingAPI (the model is embedded in the DLL), then portaBLe. RatingAPI reads the input size
   from the model, so 49- and 53-value models both work.

`data.py` is not run on its own: the training scripts, `evaluate.py` and `tools/noise_floor.py` import it. Preprocessed data is
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
| `--playlists [folder]` | `playlists\` | with `all`: only the leaderboards of those playlists. Lookups are saved in `<out>\playlist-leaderboards.txt` |
| `<out>\excluded.txt` | | leaderboard ids never downloaded (one per line, `#` comments), e.g. VNJS maps without the requirement flag |
| `--parallel-leaderboards` / `--parallel-replays` | 4 / 16 | |
| `--api-delay-ms` | 250 | BeatLeader allows 50 requests per 10 s; 250 ms stays under it |

## Training

- Train / val / test are split **by song** (70 / 15 / 15), so the difficulties of one song never end up on both sides.
- Songs in `replays_validation/` are excluded from training entirely, so that folder is a true hold-out set.
- Each run writes `runs/<timestamp>/`: `best.keras` (`best.pt` with `train_torch.py`), `model_sleep_bl.onnx`, `history.csv` (with the learning rate),
  `predictions_val.csv`, `predictions_test.csv`, `summary.json`. TensorBoard logs go to `logs/fit/<timestamp>` (`train.py` only).
- The headline metric is **map MAE in percentage points**: predicted vs real average accuracy per map, for val, test
  and the hand-picked maps. Each set is also broken down by notes per second, NJS and real accuracy.
- Early stopping: stops when `val_loss` hasn't improved by `--min-delta` (1e-5) for `--patience` (8) epochs;
  the learning rate is halved on plateaus. The best epoch is kept.
- `--lstm-units 32,24` is the model in use (the default `12,10` is the original, smaller one).
- `--init <run>/best.pt` (or `best.keras` with `train.py`) starts from an earlier model of the same size: much faster
  for experiments where only the targets changed. Train final models from scratch.
- `train.py --backend torch` runs Keras on PyTorch, but its LSTMs are slow on the GPU; use `train_torch.py` instead.
- `--final` trains on train + test + `replays_validation/` (val still picks the best epoch).
  `--final --all-data <history.csv> --epochs N` also trains on val, for exactly N epochs with the same learning-rate
  drops, and keeps the last epoch. Neither is scored: use the test score of the normal run as the model's expected error.
  (Runs from before `history.csv` had a `learning_rate` column need `--lr-drops 18,24` with the drop epochs from their log.)

## Checking a model

```
.venv\Scripts\python.exe evaluate.py runs\<run>\model_sleep_bl.onnx --maps-from runs\<run>\predictions_test.csv
.venv\Scripts\python.exe evaluate.py ..\portaBLeRebalancer\RatingAPI\model_sleep_bl.onnx --maps-from runs\<run>\predictions_test.csv
.venv\Scripts\python.exe tools\noise_floor.py --maps-from runs\<run>\predictions_test.csv
```

- `evaluate.py` scores any `.keras` or `.onnx` model; old 49-value models get the old input layout automatically.
  With `--maps-from` it uses the maps of a run's predictions CSV from `replays/` (or `replays_validation/` for a
  `predictions_handpicked.csv`); without it, the maps in `replays_validation/`. `--replays` picks another folder,
  `--plot` shows real vs predicted per note.
- A model trained with `--final` has seen those maps, so evaluating it only checks that the export works.
- `tools/noise_floor.py` splits each map's players in two random halves and compares their targets: the error a perfect
  model would still have, to compare with the run's map MAE and note MAE.

## Player-adjusted targets

The per-note targets are the average of the downloaded replays, and who plays a map varies a lot (grind maps get
the very top players). So before averaging, each replay's misses are scaled to a reference player:
`acc' = 1 - (1 - acc) * exp(reference - player skill)`. The skills come from `player-skill.json`, made by
`tools/player_skill.py <portaBLe wwwroot>` from portaBLe's `RatingFitExport.RunScores` export (all ranked scores).
Scores are only used to build the training data; the model still rates a map from the map alone.
Regenerate the file when the scores change (the preprocessing cache rebuilds by itself).

- `--no-player-skill`: raw replay accuracy as targets, no adjustment.
- Centering (`PrepConfig.center_skill`, on by default in `train.py`; `--no-centering` in `train_torch.py` turns it
  off): takes out the part of each map's adjustment that its difficulty, speed and mode predict (fitted over all maps,
  saved to `skill-centering.json`). It removes the one-saber / acc-map tilt of the adjustment but in practice flips the
  pass / tech tilts; the model in use is trained without it, and portaBLe's balance constants correct the tilts.
- Checked in portaBLe with megametric (outliers) and the Balance Tuning page: wide replays + adjustment, not centered,
  32,24, gave the smallest outlier spread (0.534 vs 0.584 for the original model) once the constants were retuned.

## Settings worth knowing

- `data.PrepConfig`: at least 8 replays per map (`min_replays`), maps with fewer than 10 count less in training
  (`full_weight_replays`, 8 replays = 0.8), how many players must have cut a note well for its target to be trusted,
  mean vs median, and whether to skip maps with a missing NJS.
- `data.USE_SCORING_TYPE_FEATURES` (default `True`) adds 4 note-type features (normal, arc head,
  arc tail, chain head), so the model input is `(None, 32, 53)` instead of `(None, 32, 49)`.
  RatingAPI (`portaBLeRebalancer/RatingAPI/Controllers/DataProcessing.cs`) builds the same values and reads the
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
