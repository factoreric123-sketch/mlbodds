"""Grade logged WNBA consensus-vs-PP edges against ESPN box scores.

Reads pp_logs/wnba_edges.csv (written by pp_vs_consensus.py --log wnba_edges.csv),
pulls final box scores for the given date from ESPN, grades each MORE/LESS leg,
and prints that date's record plus the cumulative record across all graded dates.

Usage: python wnba_grade.py --date 2026-07-02
"""
from __future__ import annotations
import argparse
import os
import unicodedata
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "pp_logs", "wnba_edges.csv")
STAT_COL = {"points": "PTS", "rebounds": "REB", "assists": "AST", "pra": "PRA", "threes": "3PTM"}


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower().strip()


def box_scores(date: str) -> dict:
    """date 'YYYY-MM-DD' -> {norm_name: {PTS,REB,AST,PRA}} for FINAL games only."""
    ymd = date.replace("-", "")
    r = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard",
                     params={"dates": ymd})
    r.raise_for_status()
    gids = [ev["id"] for ev in r.json().get("events", [])
            if ev["competitions"][0]["status"]["type"]["name"] == "STATUS_FINAL"]
    out = {}
    for gid in gids:
        d = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary",
                         params={"event": gid}).json()
        for team in d.get("boxscore", {}).get("players", []):
            for grp in team.get("statistics", []):
                labels = grp.get("labels", [])
                for a in grp.get("athletes", []):
                    vals = a.get("stats", [])
                    if not vals:
                        continue
                    row = dict(zip(labels, vals))

                    def gi(x):
                        v = row.get(x)
                        return int(v) if v not in (None, "--", "") else None

                    pts, reb, ast = gi("PTS"), gi("REB"), gi("AST")
                    pra = None if None in (pts, reb, ast) else pts + reb + ast
                    tpm = None
                    v3 = row.get("3PT")  # ESPN reports made-attempted, e.g. "2-5"
                    if v3 not in (None, "--", "") and "-" in str(v3):
                        tpm = int(str(v3).split("-")[0])
                    out[norm(a["athlete"]["displayName"])] = {"PTS": pts, "REB": reb, "AST": ast,
                                                              "PRA": pra, "3PTM": tpm}
    return out


def grade_leg(actual, line, side):
    if actual is None:
        return None
    over = actual > line  # PP lines are .5 so no pushes on these
    win = over if side.upper() == "MORE" else (not over)
    return "win" if win else "loss"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()

    if not os.path.exists(LOG):
        print(f"no edges log at {LOG}")
        return
    edges = pd.read_csv(LOG)
    day = edges[edges["date"] == args.date].copy()
    if not len(day):
        print(f"no logged edges for {args.date}")
        return

    box = box_scores(args.date)
    results = []
    for _, r in day.iterrows():
        col = STAT_COL.get(r["stat"])
        actual = box.get(norm(r["pitcher"]), {}).get(col)
        res = grade_leg(actual, r["pp_line"], r["side"])
        results.append(dict(name=r["pitcher"], stat=r["stat"], line=r["pp_line"],
                            side=r["side"], edge=r["edge"], tag=r.get("tag", "core"),
                            actual=actual, result=res))
    rd = pd.DataFrame(results)

    def record(sub):
        g = sub[sub["result"].notna()]
        w = int((g["result"] == "win").sum())
        l = int((g["result"] == "loss").sum())
        rate = f" ({w/(w+l):.1%})" if (w + l) else ""
        return f"{w}W-{l}L{rate}"

    print(f"### WNBA edge grades {args.date} ###\n")
    print(rd.to_string(index=False))
    nd = rd["result"].isna().sum()
    print(f"\n{args.date}:  \u2b50favorite {record(rd[rd.tag=='favorite'])}   |   "
          f"core {record(rd[rd.tag=='core'])}   |   "
          f"lowconf {record(rd[rd.tag=='lowconf'])}   ({nd} not-yet-final/no-data)")

    # persist grades so cumulative record survives across runs
    gpath = os.path.join(ROOT, "pp_logs", "wnba_grades.csv")
    rd.insert(0, "date", args.date)
    if os.path.exists(gpath):
        prior = pd.read_csv(gpath)
        prior = prior[prior["date"] != args.date]  # overwrite this date if regraded
        allg = pd.concat([prior, rd], ignore_index=True)
    else:
        allg = rd
    allg.to_csv(gpath, index=False)

    print(f"CUMULATIVE:  \u2b50favorite {record(allg[allg.tag=='favorite'])}   |   "
          f"core {record(allg[allg.tag=='core'])}   |   "
          f"lowconf {record(allg[allg.tag=='lowconf'])}   "
          f"over {allg['date'].nunique()} date(s)  -> {os.path.relpath(gpath, ROOT)}")


if __name__ == "__main__":
    main()
