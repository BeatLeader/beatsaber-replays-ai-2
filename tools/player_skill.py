"""Player skill from all ranked scores, for the training targets (data.py PrepConfig.player_skill_file).

    python tools/player_skill.py <portaBLe wwwroot folder> [output, default player-skill.json in the repo root]

Reads rating-fit-scores.bin/.json (portaBLe RatingFitExport.RunScores) and fits, on every vanilla score,
    log(1 - accuracy) = player skill + map difficulty
(alternating least squares, a bit of shrinkage on the players). Lower skill = the player misses less.

Why: the targets are the average accuracy of the replays that were downloaded, and who plays a map changes a lot
(grind maps get the very top players, others don't). Without this the model learns "this map gives high acc" where it
should learn "this map is easy". Checked in portaBLe: the maps players consider overweighted (high megametric) are the
maps where players do better than the Acc AI predicts once who played them is accounted for (correlation -0.70);
measured on the raw accuracy the link is only 0.14.

Scores are only used here, to build the training data: the model still rates a map from the map alone.
"""
import json, sys
import numpy as np

OK_MODS = {'PM', 'BE', 'IF'}        # modifiers that don't change what has to be hit or how


def fit(pl, lb, y, n_players, n_maps, min_pl=10, min_lb=20, iters=30, shrink=5.0):
    m = np.ones(len(y), bool)
    for _ in range(5):          # drop players / maps with too few scores (repeatedly, they depend on each other)
        cp = np.bincount(pl[m], minlength=n_players); cl = np.bincount(lb[m], minlength=n_maps)
        m = m & (cp[pl] >= min_pl) & (cl[lb] >= min_lb)
    p, l, yy = pl[m], lb[m], y[m]
    a = np.zeros(n_players); b = np.zeros(n_maps)
    na = np.bincount(p, minlength=n_players); nb = np.bincount(l, minlength=n_maps)
    for _ in range(iters):
        a = np.bincount(p, yy - b[l], n_players) / (na + shrink)
        b = np.bincount(l, yy - a[p], n_maps) / np.maximum(nb, 1)
    return a, b, m, na


def main(wwwroot, out):
    meta = json.load(open(f'{wwwroot}/rating-fit-scores.json', encoding='utf-8-sig'))
    raw = np.fromfile(f'{wwwroot}/rating-fit-scores.bin', dtype='<i4').reshape(-1, 4)
    pl = raw[:, 0]; lb = raw[:, 1]; acc = raw[:, 2].view('<f4').astype(float); mod = raw[:, 3] & 0xFFFF
    modok = np.array([all(x in OK_MODS for x in m.split(',') if x) for m in meta['Modifiers']])
    keep = modok[mod] & (acc > 0.3) & (acc < 0.9999)
    pl, lb, acc = pl[keep], lb[keep], acc[keep]
    players, maps = meta['Players'], meta['Leaderboards']
    a, b, used, na = fit(pl, lb, np.log(1 - acc), len(players), len(maps))
    fitted = na >= 10
    # Reference: the typical player in the downloaded replays (the top 10 of each leaderboard), so on average the
    # targets keep about the same level and only who-played-which-map is taken out.
    order = np.lexsort((-acc, lb)); lbs = lb[order]
    rank = np.arange(len(lbs)) - np.searchsorted(lbs, lbs)
    top = order[rank < 10]
    top = top[fitted[pl[top]]]
    reference = float(np.median(a[pl[top]]))
    skills = {players[i]: round(float(a[i]), 4) for i in np.where(fitted)[0]}
    json.dump({'Reference': round(reference, 4), 'Scores': int(used.sum()), 'Skills': skills}, open(out, 'w'))
    sd = np.std(a[pl[top]])
    print(f'{len(skills)} players fitted from {used.sum()} scores; reference {reference:.3f}, '
          f'spread among top-10 players {sd:.3f} (x{np.exp(sd):.2f} on the misses) -> {out}')


if __name__ == '__main__':
    import os
    default_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'player-skill.json')
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else default_out)
