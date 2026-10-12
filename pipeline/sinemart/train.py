"""Train the hills model on the team's picks and refill the app's pair queue.

    python -m sinemart.train out/siglip2.npz picks.csv out/

picks.csv comes from D1 (see README): columns winner_id, loser_id. Writes:

    out/report.json    held-out + per-look agreement for the hills model and the
                       A/B/C baselines, faces per hill, and the 10-band check
    out/scores.csv     applicant_id, score, score_std, hill (hills model)
    out/update.sql     scores + a fresh pair batch, for `wrangler d1 execute --file`
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .bt import HillsHead, LinearHead, Looks, band_check, compare_heads, hills_with_std, next_pairs, rater_ceiling


def load_picks(path: Path, id_to_idx: dict[int, int]):
    w, l, dropped = [], [], 0
    with open(path) as f:
        for row in csv.DictReader(f):
            a, b = int(row["winner_id"]), int(row["loser_id"])
            if a in id_to_idx and b in id_to_idx:
                w.append(id_to_idx[a])
                l.append(id_to_idx[b])
            else:
                dropped += 1
    return np.asarray(w, np.int64), np.asarray(l, np.int64), dropped


def run(npz: Path, picks: Path, out: Path, n_pairs: int = 2000, looks_k: int = 12, seed: int = 0):
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(npz)
    ids, emb = data["ids"], data["emb"].astype(np.float64)
    model = str(data["model"])
    idx = {int(a): i for i, a in enumerate(ids)}
    winners, losers, dropped = load_picks(picks, idx)
    looks = Looks.fit(emb, k=looks_k, seed=seed)

    report = {"model": model, "n_picks": len(winners), "dropped_picks": dropped}
    report["rater_ceiling"], report["repeat_picks"] = rater_ceiling(winners, losers)
    report["ranker"] = HillsHead.name
    if len(winners) >= 50:
        report["heads"] = compare_heads(emb, winners, losers, looks, seed=seed)

    hill = looks.labels
    if len(winners):
        hills, std = hills_with_std(emb, winners, losers, looks, seed=seed)
        scores, hill = hills.score(emb), hills.hill(emb)
        arrow = LinearHead(1.0).fit(emb, winners, losers).score(emb)
        report["bands"], report["band_peak"] = band_check(arrow, winners, losers)
        report["faces_per_hill"] = np.bincount(hill, minlength=looks_k).tolist()
    else:
        scores, std = np.zeros(len(ids)), np.ones(len(ids))  # cold start: random pairs

    (out / "report.json").write_text(json.dumps(report, indent=2))
    with open(out / "scores.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["applicant_id", "score", "score_std", "hill"])
        for i in np.argsort(-scores):
            wr.writerow([int(ids[i]), f"{scores[i]:.5f}", f"{std[i]:.5f}", int(hill[i])])

    pairs, kinds = next_pairs(scores, std, n_pairs, seed=seed)
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    lines = ["DELETE FROM pair_queue WHERE served_at IS NULL;"]
    for i in range(len(ids)):
        lines.append(
            f"UPDATE applicants SET score={scores[i]:.6f}, score_std={std[i]:.6f}, look={int(hill[i])}, "
            f"model='{HillsHead.name}', updated_at='{batch}' WHERE id={int(ids[i])};"
        )
    for (a, b), k in zip(pairs, kinds):
        lines.append(
            f"INSERT INTO pair_queue (left_id, right_id, kind, batch) VALUES ({int(ids[a])}, {int(ids[b])}, '{k}', '{batch}');"
        )
    (out / "update.sql").write_text("\n".join(lines) + "\n")
    return report


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        sys.exit("usage: python -m sinemart.train EMB.npz picks.csv OUT_DIR")
    report = run(Path(argv[1]), Path(argv[2]), Path(argv[3]))
    print(json.dumps({k: v for k, v in report.items() if k not in ("heads", "bands")}, indent=2))
    for name, r in report.get("heads", {}).items():
        print(f"{name:16s} held-out {r['heldout']:.3f}  train {r['train']:.3f}  ({r['n_test']} test pairs)")
    for b in report.get("bands", []):
        print(f"band {b['band']:2d}  win rate {b['win_rate']:.2f}  ({b['games']} games)")


if __name__ == "__main__":
    main(sys.argv)
