"""
Find leaderboards in replays/ with fewer than 20 notes (BeatLeader's note count) and move them out of the training data.

    python remove_short_maps.py               # list them only
    python remove_short_maps.py --move        # move them to replays_excluded/
    python remove_short_maps.py --min-notes 30 --replays replays_validation

Reads the note count of every leaderboard from BeatLeader's leaderboard list (1000 per request, most played first,
down to --min-plays); folders not found that way are looked up one by one. Nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.parse
import urllib.request

API = "https://api.beatleader.xyz"


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "beatsaber-replays-ai"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # rate limit or network hiccup
            if attempt == 4:
                raise
            print(f"  retrying ({e})")
            time.sleep(5 * (attempt + 1))


def notes_from_list(min_plays: int) -> dict[str, int]:
    notes: dict[str, int] = {}
    page = 1
    while True:
        body = get_json(f"{API}/leaderboards?leaderboardContext=nomods&page={page}&count=1000&type=all"
                        f"&sortBy=playcount&order=desc&allTypes=0")
        rows = body.get("data") or []
        for lb in rows:
            diff = lb.get("difficulty") or {}
            if diff.get("notes") is not None:
                notes[lb["id"]] = int(diff["notes"])
        print(f"  page {page}: {len(rows)} leaderboards")
        if len(rows) < 1000 or all((lb.get("plays") or 0) < min_plays for lb in rows):
            break
        page += 1
        time.sleep(0.5)
    return notes


def notes_of(leaderboard_id: str) -> int | None:
    body = get_json(f"{API}/leaderboard/scores/{urllib.parse.quote(leaderboard_id)}"
                    f"?leaderboardContext=nomods&page=1&count=1")
    diff = (body or {}).get("difficulty") or {}
    return int(diff["notes"]) if diff.get("notes") is not None else None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--replays", default=os.path.join(here, "replays"))
    p.add_argument("--min-notes", type=int, default=20, help="leaderboards with fewer notes are removed")
    p.add_argument("--min-plays", type=int, default=200, help="how far down the most-played list to read")
    p.add_argument("--move", action="store_true", help="move the matching folders to replays_excluded/")
    args = p.parse_args()

    folders = sorted(e.name for e in os.scandir(args.replays) if e.is_dir())
    print(f"Reading note counts from BeatLeader for {len(folders)} folders in {args.replays}...")
    notes = notes_from_list(args.min_plays)

    missing = [lb for lb in folders if lb not in notes]
    if missing:
        print(f"Looking up {len(missing)} leaderboards one by one...")
        for i, lb in enumerate(missing, 1):
            try:
                n = notes_of(lb)
            except Exception as e:
                print(f"  {lb}: failed ({e}), kept")
                continue
            if n is not None:
                notes[lb] = n
            time.sleep(0.25)
            if i % 100 == 0:
                print(f"  {i}/{len(missing)}")

    unknown = [lb for lb in folders if lb not in notes]
    matches = [lb for lb in folders if lb in notes and notes[lb] < args.min_notes]
    print(f"{len(matches)} of {len(folders)} folders have fewer than {args.min_notes} notes:")
    for lb in matches:
        print(f"  {lb}  ({notes[lb]} notes)")
    if unknown:
        print(f"{len(unknown)} folders without a note count were kept: {', '.join(unknown[:20])}"
              f"{' ...' if len(unknown) > 20 else ''}")

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
