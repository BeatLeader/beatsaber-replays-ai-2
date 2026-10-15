# Changelog: beatsaber-replays-ai-2 (BeatLeader AccAI)

The Acc AI predicts top players' per-note accuracy on a map. RatingAPI uses it for each map's acc rating. Work is on the
`loloppe` branch; the ratings and balance are in portaBLeRebalancer.

- **Player-adjusted targets.**
  - A map's training target depended on who played it: grind maps get the very top players. That made some maps
    overweighted.
  - `tools/player_skill.py` fits a skill for each player from all ranked scores, and each replay's lost accuracy is scaled
    to a reference player.
  - Scores are only used to build the training data. The model still rates maps from the map alone.
- **Wider training data.** `BLReplaysPrepare --playlists` downloads the top 50 replays of the ranked AccSaber, ScoreSaber
  and BeatLeader maps (`replays_wide`).
  - `excluded.txt` lists maps that are never downloaded, such as VNJS maps the requirements filter misses.
- **GPU training.** `train_torch.py` trains the same model with PyTorch and cuDNN, much faster than Keras on an RTX 5090.
- **Production model**, trained with player-adjusted targets and without centering:

  ```
  train_torch.py --replays replays_wide --lstm-units 32,24 --batch-size 1024 --lr 2e-3 --no-centering --final
  ```

  - Outlier spread in portaBLe is 0.531, down from 0.584 with the previous model.
- **Repo cleanup.**
  - The helper scripts moved to `tools/`.
  - The old notebooks and models moved to `legacy/`.
  - The Analyzer research exports moved to `BLReplaysPrepare/Research/`.
  - New README and `.gitignore`.

- **Notebooks replaced by scripts:** `data.py`, `model.py`, `train.py`, `evaluate.py`.
  - The train / val / test split is by song.
  - The main metric is map-level MAE.
- **4 note-type input features** (53 values per note instead of 49). The model was deployed to RatingAPI with LSTMs of
  32,24.
- **BLReplaysPrepare rewritten.** It filters maps (plays, notes, top-score spread, modded requirements), can resume a
  stopped run, and has a `--top` mode.

- **`player-skill.json` is not in git.** Rebuild it from a fresh portaBLe score export when the ranked scores change a lot.
