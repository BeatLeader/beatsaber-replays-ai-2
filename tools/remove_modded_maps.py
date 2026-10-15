"""
Find leaderboards in replays/ that use Noodle Extensions or Mapping Extensions (or other map requirements)
and move them out of the training data.

    python tools/remove_modded_maps.py              # list them only
    python tools/remove_modded_maps.py --move       # move them to replays_excluded/
    python tools/remove_modded_maps.py --requirements 396 --move   # also VNJS (128) and Vivify (256)

Asks BeatLeader for every leaderboard with those requirements (mapRequirements=<flags>&allRequirements=0, "any"),
most played first, down to --min-plays. Only folder names are compared; nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request

API = "https://api.beatleader.xyz/leaderboards"


def fetch_ids(requirements: int, min_plays: int) -> set[str]:
    ids: set[str] = set()
    page = 1
    while True:
        url = (f"{API}?leaderboardContext=nomods&page={page}&count=1000&type=all&sortBy=playcount&order=desc"
               f"&allTypes=0&mapRequirements={requirements}&allRequirements=0")
        req = urllib.request.Request(url, headers={"User-Agent": "beatsaber-replays-ai"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = json.loads(r.read().decode("utf-8"))
                break
            except Exception as e:  # rate limit or network hiccup
                if attempt == 4:
                    raise
                print(f"  retrying page {page} ({e})")
                time.sleep(5 * (attempt + 1))
        rows = body.get("data") or []
        for lb in rows:
            if (lb.get("plays") or 0) >= min_plays:
                ids.add(lb["id"])
        print(f"  page {page}: {len(rows)} leaderboards")
        if len(rows) < 1000 or all((lb.get("plays") or 0) < min_plays for lb in rows):
            break
        page += 1
        time.sleep(0.5)
    return ids


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (this script is in tools/)
    p = argparse.ArgumentParser()
    p.add_argument("--replays", default=os.path.join(here, "replays"))
    p.add_argument("--requirements", type=int, default=12, help="BeatLeader requirement flags (12 = Noodle + Mapping Extensions)")
    p.add_argument("--min-plays", type=int, default=200)
    p.add_argument("--move", action="store_true", help="move the matching folders to replays_excluded/")
    args = p.parse_args()

    print(f"Asking BeatLeader for leaderboards with requirements {args.requirements}...")
    flagged = fetch_ids(args.requirements, args.min_plays)
    print(f"{len(flagged)} such leaderboards with {args.min_plays}+ plays")

    folders = {e.name for e in os.scandir(args.replays) if e.is_dir()}
    matches = sorted(folders & flagged)
    print(f"{len(matches)} of {len(folders)} folders in {args.replays} use them:")
    for lb in matches:
        print(f"  {lb}")

    if args.move and matches:
        target = os.path.join(os.path.dirname(os.path.abspath(args.replays)), "replays_excluded")
        os.makedirs(target, exist_ok=True)
        for lb in matches:
            shutil.move(os.path.join(args.replays, lb), os.path.join(target, lb))
        print(f"Moved {len(matches)} folders to {target}")
    elif matches:
        print("Run again with --move to move them out of the training data.")


if __name__ == "__main__":
    main()
