"""
Data loading and preprocessing for the BeatLeader accuracy model.

Pipeline:
    replays/<leaderboard_id>/<playerId>-<accuracy>-<baseScore>-<njs>.npy
        -> per-map note list (one entry per note, accuracy averaged over players,
           each player's misses first scaled to a reference player, see PrepConfig.player_skill_file)
        -> per-note feature vectors
        -> fixed-size windows (pre context | predicted notes | post context)

Replay file columns (written by BLReplaysPrepare/Program.cs):
    0: noteID      (BeatLeader encoding, see decode_note_id)
    1: accuracy    1 - clamp(cutDistanceToCenter / 0.3, 0, 1) for good cuts, 0 otherwise
    2: spawnTime   seconds
    3: eventType   0 good, 1 bad, 2 miss, 3 bomb   (only in files written by the updated Program.cs)

Legacy 3-column files have no eventType, so a stored accuracy of exactly 0 is treated as
"not a good cut" (miss / bad cut / bomb).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRE_SEGMENT_SIZE = 12
POST_SEGMENT_SIZE = 12
PREDICTION_SIZE = 8
SEGMENT_SIZE = PRE_SEGMENT_SIZE + PREDICTION_SIZE + POST_SEGMENT_SIZE

# Scoring types, from ReplayDecoder/NoteParams.cs (noteID // 10000).
SCORING_DEFAULT = 0          # old replays without a scoring type: treated as a normal note
SCORING_IGNORE = 1
SCORING_NOSCORE = 2
SCORING_NORMAL = 3
SCORING_ARC_HEAD = 4         # SliderHead
SCORING_ARC_TAIL = 5         # SliderTail
SCORING_CHAIN_HEAD = 6       # BurstSliderHead
SCORING_CHAIN_LINK = 7       # BurstSliderElement: fixed 20 points, no accuracy component
SCORING_ARC_HEAD_ARC_TAIL = 8
SCORING_CHAIN_HEAD_ARC_TAIL = 9

# Scoring types that get a one-hot slot in the features. Combined types (a note that is both an arc
# head and an arc tail, ...) get no slot (all zeros). RatingAPI's DataProcessing.cs does the same.
FEATURE_SCORING_TYPES = [SCORING_NORMAL, SCORING_ARC_HEAD, SCORING_ARC_TAIL, SCORING_CHAIN_HEAD]
# Scoring types whose accuracy is a meaningful training target.
TARGET_SCORING_TYPES = {SCORING_NORMAL, SCORING_ARC_HEAD, SCORING_ARC_TAIL, SCORING_CHAIN_HEAD,
                        SCORING_ARC_HEAD_ARC_TAIL, SCORING_CHAIN_HEAD_ARC_TAIL}
# Chain links are left out of the note sequence entirely: RatingAPI only feeds the map's notes
# (chain heads included) to the model and counts chain links separately as fixed points.
EXCLUDED_SCORING_TYPES = {SCORING_IGNORE, SCORING_NOSCORE, SCORING_CHAIN_LINK}

# Set to False to keep the original 49-value note vector (the interface the old ONNX model used).
# The environment variable lets evaluate.py switch to the old 49-value layout, including in its worker processes.
USE_SCORING_TYPE_FEATURES = os.environ.get("ACCAI_NOTE_TYPES", "1") != "0"

BASE_NOTE_SIZE = 49
NOTE_SIZE = BASE_NOTE_SIZE + (len(FEATURE_SCORING_TYPES) if USE_SCORING_TYPE_FEATURES else 0)


@dataclass
class PrepConfig:
    min_replays: int = 8             # maps with fewer replay files are skipped
    full_weight_replays: int = 10    # maps with fewer replays count less in training (8 replays -> 0.8)
    min_note_presence: float = 0.8   # a note must appear in this fraction of replays (drops per-player bomb hits etc.)
    min_good_fraction: float = 0.5   # a note's target is only trusted if this fraction of players cut it well
    aggregate: str = "mean"          # "mean" or "median" over the players' good cuts
    skip_missing_njs: bool = True    # skip maps whose NJS is 0 / missing in the filename
    # Player-adjusted targets: each replay's misses are scaled to what the reference player would miss, using the
    # player skills fitted on all ranked scores (tools/player_skill.py). Without it the targets depend on who played the
    # map (grind maps get the very top players), which is what made some maps overweighted. "" turns it off.
    player_skill_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player-skill.json")
    # cap on the scaling of one replay's lost accuracy; wide replay sets (top 50) have players down to ~x0.3
    skill_factor_limits: tuple = (0.25, 4.0)
    # Only correct for player pools that are unusual for the kind of map. A map's pool is partly explained by its type:
    # easy / acc maps and one saber maps are played by more players outside the top, often specialists who do better
    # there than their overall skill says. Scores can't tell "these players are better here" from "this map is easier",
    # and correcting the whole type made acc and one saber maps clearly underweighted (portaBLe balance tuner). So the
    # part of each map's adjustment that its difficulty, speed and mode predict is taken out again (fitted over all maps).
    center_skill: bool = True

    def key(self) -> str:
        skill = "-s0"
        if self.player_skill_file:
            with open(self.player_skill_file, "rb") as f:   # a new skill file rebuilds the cache
                skill = f"-s{hashlib.md5(f.read()).hexdigest()[:8]}-{self.skill_factor_limits[0]}-{self.skill_factor_limits[1]}"
            skill += "-v2"   # cache with the fields used by the centering (centering itself is applied after loading)
        return (f"r{self.min_replays}-w{self.full_weight_replays}-p{self.min_note_presence}-g{self.min_good_fraction}"
                f"-{self.aggregate}-n{int(self.skip_missing_njs)}-f{NOTE_SIZE}{skill}")


@functools.lru_cache(maxsize=4)
def load_player_skills(path: str) -> tuple[float, dict]:
    """(reference skill, {playerId: skill}) from tools/player_skill.py. Skill = the player's offset in log(1 - accuracy)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run tools/player_skill.py (or set PrepConfig.player_skill_file = \"\" "
                                f"to train on the raw accuracy)")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return float(d["Reference"]), d["Skills"]


def skill_factor(player_id: str, cfg: PrepConfig) -> float:
    """How much this player's misses are scaled to match the reference player (1 for players without a skill)."""
    if not cfg.player_skill_file:
        return 1.0
    reference, skills = load_player_skills(cfg.player_skill_file)
    skill = skills.get(player_id)
    if skill is None:
        return 1.0
    lo, hi = cfg.skill_factor_limits
    return float(np.clip(np.exp(reference - skill), lo, hi))


# ---------------------------------------------------------------------------
# Filenames and note IDs
# ---------------------------------------------------------------------------

@dataclass
class ReplayFileInfo:
    path: str
    player_id: str
    accuracy: float
    base_score: int
    njs: float


def parse_replay_filename(path: str) -> ReplayFileInfo:
    """Parse '<playerId>-<accuracy>-<baseScore>-<njs>.npy' from the file name only.

    Only the basename is split, so hyphens in folder names can no longer shift the fields
    (that was the NJS bug in the old notebook). The NJS may itself contain a '.', e.g. 20.5.
    """
    name = os.path.basename(path)
    if name.endswith(".npy"):
        name = name[:-4]
    parts = name.split("-")
    if len(parts) != 4:
        raise ValueError(f"Unexpected replay filename: {os.path.basename(path)}")
    player_id, acc, score, njs = parts
    return ReplayFileInfo(path, player_id, float(acc.replace(",", ".")), int(score),
                          float(njs.replace(",", ".")))


def decode_note_id(note_id: int):
    """Decode a BeatLeader noteID into (scoring_type, line_index, line_layer, color, cut_direction).

    Mirrors ReplayDecoder/NoteParams.cs, including the long layout used for IDs >= 100000.
    """
    i = int(note_id)
    if i < 0:
        return None
    if i < 100000:
        scoring = i // 10000
        i -= scoring * 10000
        line = i // 1000
        i -= line * 1000
        layer = i // 100
        i -= layer * 100
    else:
        scoring = i // 10000000
        i -= scoring * 10000000
        line = i // 1000000
        i -= line * 1000000
        layer = i // 100000
        i -= layer * 100000
    color = i // 10
    direction = i - color * 10
    if scoring == SCORING_DEFAULT:
        scoring = SCORING_NORMAL
    return scoring, line, layer, color, direction


def song_group(leaderboard_id: str) -> str:
    """Song part of a BeatLeader leaderboard id (drops the difficulty + mode digits).

    '316bfxxxxxxxxxxx91' and '316bfxxxxxxxxxxx51' are the same song, so they must end up on the
    same side of the train/validation split.
    """
    return leaderboard_id[:-2] if len(leaderboard_id) > 2 else leaderboard_id


# ---------------------------------------------------------------------------
# Per-map aggregation
# ---------------------------------------------------------------------------

@dataclass
class MapData:
    leaderboard_id: str
    njs: float
    n_replays: int
    leaderboard_accuracy: float          # mean accuracy of the replays, from the filenames
    times: np.ndarray                    # (n_notes,)
    note_ids: np.ndarray                 # (n_notes,)
    targets: np.ndarray                  # (n_notes,) averaged accuracy (0..1)
    weights: np.ndarray                  # (n_notes,) 1 if the target is trusted, else 0
    good_fraction: np.ndarray            # (n_notes,) fraction of players who cut the note well
    features: np.ndarray = field(default=None)   # (n_notes, NOTE_SIZE)
    skill_factor: float = 1.0            # mean scaling of the replays' lost accuracy (player-adjusted targets)
    skill_coverage: float = 1.0          # fraction of the replays whose player has a fitted skill
    skill_log_shift: float = 0.0         # mean log scaling of the replays' lost accuracy (before centering)
    skill_centering: float = 0.0         # log scaling taken out again by the centering (see PrepConfig.center_skill)
    replay_log_loss: float = 0.0         # mean log(1 - accuracy) of the replays, from the filenames (map difficulty)
    notes_per_second: float = 0.0


def list_leaderboards(replays_dir: str) -> list[str]:
    """Leaderboard folders only (ignores completed.txt and other stray files)."""
    return sorted(e.name for e in os.scandir(replays_dir) if e.is_dir())


def list_replay_files(folder: str) -> list[str]:
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy"))


def load_map(folder: str, cfg: PrepConfig, files: list[str] | None = None) -> tuple[MapData | None, str]:
    """Aggregate the replays of one leaderboard folder (or only `files`). Returns (map, reason_if_skipped)."""
    leaderboard_id = os.path.basename(os.path.normpath(folder))
    if files is None:
        files = list_replay_files(folder)
    if len(files) < cfg.min_replays:
        return None, f"only {len(files)} replays"

    infos = []
    for f in files:
        try:
            infos.append(parse_replay_filename(f))
        except ValueError as e:
            return None, str(e)

    njs = infos[0].njs
    if njs <= 0 and cfg.skip_missing_njs:
        return None, "missing NJS"

    # key (noteID, spawnTime) -> list of (accuracy, is_good)
    notes: dict[tuple[int, float], list[tuple[float, bool]]] = {}
    n_loaded = 0
    factors = []
    for info in infos:
        try:
            arr = np.load(info.path)
        except Exception as e:  # corrupted / partially written file
            print(f"  could not read {info.path}: {e}")
            continue
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
            continue
        n_loaded += 1
        factor = skill_factor(info.player_id, cfg)
        known = not cfg.player_skill_file or info.player_id in load_player_skills(cfg.player_skill_file)[1]
        factors.append((factor, known))
        has_event = arr.shape[1] >= 4
        for row in arr:
            note_id = int(row[0])
            acc = float(row[1])
            if has_event:
                event = int(row[3])
                if event == 3:          # bomb
                    continue
                good = event == 0
            else:
                good = acc > 0.0
            if good and factor != 1.0:
                # lost accuracy scaled to the reference player: a better player's cut counts as a slightly worse one
                acc = min(1.0, max(0.0, 1.0 - (1.0 - acc) * factor))
            key = (note_id, round(float(row[2]), 4))
            notes.setdefault(key, []).append((acc, good))

    if n_loaded < cfg.min_replays:
        return None, f"only {n_loaded} readable replays"

    min_count = max(1, int(np.ceil(cfg.min_note_presence * n_loaded)))
    # Targets averaged over fewer players are noisier, so those maps count a bit less.
    map_weight = min(1.0, n_loaded / cfg.full_weight_replays) if cfg.full_weight_replays > 0 else 1.0
    rows = []
    for (note_id, t), entries in notes.items():
        if len(entries) < min_count:
            continue  # e.g. a bomb one player hit, or a note missing from a broken replay
        decoded = decode_note_id(note_id)
        if decoded is None:
            continue
        scoring, line, layer, color, direction = decoded
        if scoring in EXCLUDED_SCORING_TYPES or scoring > SCORING_CHAIN_HEAD_ARC_TAIL or color not in (0, 1):
            continue
        if not (0 <= line < 4 and 0 <= layer < 3 and 0 <= direction < 10):
            continue
        good_accs = [a for a, g in entries if g]
        good_fraction = len(good_accs) / len(entries)
        if good_accs:
            target = float(np.median(good_accs) if cfg.aggregate == "median" else np.mean(good_accs))
        else:
            target = 0.0
        trusted = (scoring in TARGET_SCORING_TYPES) and good_fraction >= cfg.min_good_fraction
        rows.append((t, note_id, target, map_weight if trusted else 0.0, good_fraction, (line, layer, direction, color)))

    if len(rows) < PREDICTION_SIZE:
        return None, "too few notes"

    # Same order as RatingAPI: time, then x, y, direction, color.
    rows.sort(key=lambda r: (r[0], r[5]))
    rows = [r[:5] for r in rows]
    rows = np.array(rows, dtype=np.float64)
    m = MapData(
        leaderboard_id=leaderboard_id,
        njs=njs,
        n_replays=n_loaded,
        leaderboard_accuracy=float(np.mean([i.accuracy for i in infos])),
        times=rows[:, 0].astype(np.float32),
        note_ids=rows[:, 1].astype(np.int64),
        targets=rows[:, 2].astype(np.float32),
        weights=rows[:, 3].astype(np.float32),
        good_fraction=rows[:, 4].astype(np.float32),
        skill_factor=float(np.mean([f for f, _ in factors])),
        skill_coverage=float(np.mean([k for _, k in factors])),
        skill_log_shift=float(np.mean([np.log(f) for f, _ in factors])),
        replay_log_loss=float(np.mean([np.log(1 - min(i.accuracy, 0.9999)) for i in infos])),
    )
    span = float(m.times[-1] - m.times[0]) if len(m.times) > 1 else 0.0
    m.notes_per_second = len(m.times) / max(span, 1.0)
    if m.weights.sum() == 0:
        return None, "no trusted notes"
    m.features = build_features(m.note_ids, m.times, m.njs)
    return m, ""


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _timing(delta: float):
    # Timing difficulty is not linear and caps out around 2 seconds.
    return max(0.0, 0.5 - delta) * 2, max(0.0, 2 - delta) / 2


def build_features(note_ids: np.ndarray, times: np.ndarray, njs: float) -> np.ndarray:
    """One feature vector per note, same layout as the original notebook:

        [red pos(12), red dir(10), blue pos(12), blue dir(10),
         red short, red long, blue short, blue long, njs/30, (scoring type one-hot)]

    The note's own block is filled and the other color's block is zero; the timing values are
    "time since the previous note of this color" and "time since the previous note of the other color".
    """
    out = np.zeros((len(note_ids), NOTE_SIZE), dtype=np.float32)
    prev_time = [0.0, 0.0]
    for i, (nid, t) in enumerate(zip(note_ids, times)):
        scoring, line, layer, color, direction = decode_note_id(int(nid))
        own_delta = t - prev_time[color]
        other_delta = t - prev_time[1 - color]
        prev_time[color] = t

        base = 0 if color == 0 else 22
        out[i, base + line * 3 + layer] = 1
        out[i, base + 12 + direction] = 1

        own_short, own_long = _timing(own_delta)
        other_short, other_long = _timing(other_delta)
        if color == 0:
            out[i, 44:48] = (own_short, own_long, other_short, other_long)
        else:
            out[i, 44:48] = (other_short, other_long, own_short, own_long)
        out[i, 48] = njs / 30.0

        if USE_SCORING_TYPE_FEATURES and scoring in FEATURE_SCORING_TYPES:
            out[i, BASE_NOTE_SIZE + FEATURE_SCORING_TYPES.index(scoring)] = 1
    return out


def make_windows(features: np.ndarray, targets: np.ndarray | None = None,
                 weights: np.ndarray | None = None):
    """Cut a map into windows of PRE | PREDICTION | POST notes.

    Windows step by PREDICTION_SIZE so every note is predicted exactly once. The last window is
    padded (with weight 0) instead of dropping the map's final notes.

    Returns X (n, SEGMENT_SIZE, NOTE_SIZE) and, if targets are given,
    Y (n, PREDICTION_SIZE, 2) where Y[..., 0] is the target and Y[..., 1] its weight.
    """
    n = len(features)
    n_windows = int(np.ceil(n / PREDICTION_SIZE))
    padded = np.zeros((PRE_SEGMENT_SIZE + n_windows * PREDICTION_SIZE + POST_SEGMENT_SIZE,
                       features.shape[1]), dtype=np.float32)
    padded[PRE_SEGMENT_SIZE:PRE_SEGMENT_SIZE + n] = features

    idx = np.arange(n_windows)[:, None] * PREDICTION_SIZE + np.arange(SEGMENT_SIZE)[None, :]
    X = padded[idx]
    if targets is None:
        return X

    t = np.zeros(n_windows * PREDICTION_SIZE, dtype=np.float32)
    w = np.zeros(n_windows * PREDICTION_SIZE, dtype=np.float32)
    t[:n] = targets
    w[:n] = weights if weights is not None else 1.0
    Y = np.stack([t, w], axis=-1).reshape(n_windows, PREDICTION_SIZE, 2)
    return X, Y


# ---------------------------------------------------------------------------
# Dataset building, caching and splitting
# ---------------------------------------------------------------------------

def _load_map_worker(args):
    folder, cfg = args
    try:
        return load_map(folder, cfg)
    except Exception as e:
        return None, f"error: {e}"


def _folder_signature(replays_dir: str, leaderboards: list[str]) -> str:
    h = hashlib.md5()
    for lb in leaderboards:
        folder = os.path.join(replays_dir, lb)
        files = sorted(os.listdir(folder))
        mtime = max((os.path.getmtime(os.path.join(folder, f)) for f in files), default=0)
        h.update(f"{lb}:{len(files)}:{mtime:.0f};".encode())
    return h.hexdigest()


def load_dataset(replays_dir: str, cfg: PrepConfig | None = None, cache_dir: str = "cache",
                 workers: int | None = None, rebuild: bool = False, verbose: bool = True) -> dict[str, MapData]:
    """Load every leaderboard folder in replays_dir. Results are cached in cache_dir and only
    rebuilt when the folder contents or the preprocessing settings change."""
    cfg = cfg or PrepConfig()
    leaderboards = list_leaderboards(replays_dir)
    signature = _folder_signature(replays_dir, leaderboards)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{os.path.basename(os.path.normpath(replays_dir))}_{cfg.key()}.pkl")

    if not rebuild and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        if cached.get("signature") == signature:
            if verbose:
                print(f"Loaded {len(cached['maps'])} maps from cache {cache_file}")
            # the cache holds the targets before centering, which is redone on every load (cheap)
            center_player_skill(cached["maps"], cfg, verbose)
            return cached["maps"]

    maps: dict[str, MapData] = {}
    skipped: dict[str, int] = {}
    jobs = [(os.path.join(replays_dir, lb), cfg) for lb in leaderboards]
    if workers == 1:
        results = map(_load_map_worker, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_load_map_worker, jobs, chunksize=16)

    for i, (m, reason) in enumerate(results):
        if m is None:
            key = re.sub(r"\d+", "N", reason.split(":")[0])
            skipped[key] = skipped.get(key, 0) + 1
        else:
            maps[m.leaderboard_id] = m
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(jobs)} leaderboards processed")
    if workers != 1:
        pool.shutdown()

    if verbose:
        print(f"Loaded {len(maps)} maps from {replays_dir} ({len(leaderboards) - len(maps)} skipped)")
        for reason, count in sorted(skipped.items(), key=lambda x: -x[1]):
            print(f"  skipped {count}: {reason}")

    with open(cache_file, "wb") as f:
        pickle.dump({"signature": signature, "maps": maps}, f)
    center_player_skill(maps, cfg, verbose)
    return maps


SKILL_CENTERING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill-centering.json")
MIN_MAPS_TO_FIT_CENTERING = 500


def _centering_features(m: MapData) -> np.ndarray:
    x = m.replay_log_loss
    s = np.log(max(m.notes_per_second, 0.1))
    one_saber = 1.0 if m.leaderboard_id.endswith("2") else 0.0     # BeatLeader ids end with the mode: 2 = OneSaber
    return np.array([1.0, x, x * x, one_saber, s, s * s, s * x])


def center_player_skill(maps: dict[str, MapData], cfg: PrepConfig, verbose: bool = True):
    """Takes out the part of each map's player adjustment that its kind of map predicts (PrepConfig.center_skill).

    Fits mean log scaling ~ difficulty (replay accuracy), speed (notes per second) and one saber over the maps, then
    rescales each map's targets by exp(-(predicted - average)): only pools that are unusual for that kind of map stay
    corrected, and the average adjustment (the reference player's level) is kept.
    Big datasets fit the coefficients and save them to skill-centering.json; small ones (replays_validation) reuse them.
    """
    if not cfg.player_skill_file or not cfg.center_skill or not maps:
        return
    ms = list(maps.values())
    X = np.stack([_centering_features(m) for m in ms])
    yv = np.array([m.skill_log_shift for m in ms])
    with open(cfg.player_skill_file, "rb") as f:
        skill_hash = hashlib.md5(f.read()).hexdigest()[:8]
    if len(ms) >= MIN_MAPS_TO_FIT_CENTERING:
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        level = float(yv.mean())
        with open(SKILL_CENTERING_FILE, "w", encoding="utf-8") as f:
            json.dump({"SkillFile": skill_hash, "Maps": len(ms), "Coefficients": beta.tolist(), "Level": level}, f)
    else:
        if not os.path.exists(SKILL_CENTERING_FILE):
            raise FileNotFoundError(f"{SKILL_CENTERING_FILE} not found: load the training replays first, it is fitted there")
        with open(SKILL_CENTERING_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("SkillFile") != skill_hash:
            print(f"  warning: {SKILL_CENTERING_FILE} was fitted with another player-skill.json")
        beta = np.array(saved["Coefficients"])
        if "Level" not in saved:
            raise ValueError(f"{SKILL_CENTERING_FILE} is from an older version: load the training replays again to refit it")
        level = float(saved["Level"])
    predicted = X @ beta - level     # only the part that differs from the average adjustment
    for m, p in zip(ms, predicted):
        m.skill_centering = float(p)
        m.targets = np.clip(1.0 - (1.0 - m.targets) * np.exp(-p), 0.0, 1.0).astype(np.float32)
    if verbose:
        left = yv - predicted
        print(f"  player adjustment: mean log scaling {yv.mean():+.3f} (sd {yv.std():.3f}); "
              f"after centering on the kind of map {left.mean():+.3f} (sd {left.std():.3f}), level kept at {level:+.3f}")


def split_by_song(leaderboard_ids, val_fraction=0.15, test_fraction=0.15, exclude_songs=(), seed="6969"):
    """Deterministic train/val/test split by song (all difficulties of a song stay together).

    Songs in exclude_songs (e.g. the hand-picked replays_validation maps) are kept out of all three
    sets so the hand-picked evaluation is never trained on.
    """
    exclude = set(exclude_songs)
    train, val, test = [], [], []
    for lb in sorted(leaderboard_ids):
        song = song_group(lb)
        if song in exclude:
            continue
        bucket = int(hashlib.md5(f"{seed}:{song}".encode()).hexdigest(), 16) % 10000 / 10000
        if bucket < test_fraction:
            test.append(lb)
        elif bucket < test_fraction + val_fraction:
            val.append(lb)
        else:
            train.append(lb)
    return train, val, test


def build_arrays(maps: dict[str, MapData], leaderboard_ids):
    xs, ys = [], []
    for lb in leaderboard_ids:
        m = maps[lb]
        X, Y = make_windows(m.features, m.targets, m.weights)
        xs.append(X)
        ys.append(Y)
    if not xs:
        return (np.zeros((0, SEGMENT_SIZE, NOTE_SIZE), np.float32),
                np.zeros((0, PREDICTION_SIZE, 2), np.float32))
    return np.concatenate(xs), np.concatenate(ys)


def acc_to_percent(acc_component):
    """Per-note accuracy component (0..1) -> score percentage, assuming full swing points.

    Same formula as the old notebook: (100 + 15 * acc) / 115.
    """
    return (100.0 + 15.0 * np.asarray(acc_component)) / 115.0 * 100.0
