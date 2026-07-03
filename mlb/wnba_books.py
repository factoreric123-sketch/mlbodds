"""Capture WNBA player-prop sharp lines and de-vig them to a fair consensus.

Mirrors the MLB book-capture: for each player+stat, pull every book's over/under,
de-vig each book's two-way price, take the modal line and the MEDIAN de-vigged
P(over) across books at that line. Writes wnba_books_<date>.csv in the same schema
that pp_vs_consensus.py already consumes: name, stat, book_line, book_p_over, n_books.

Usage: python wnba_books.py --date 2026-07-02
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import requests
import unicodedata

import predict_day

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOKS = os.path.join(ROOT, "pp_logs", "books")
KEY = os.environ.get("ODDS_API_KEY", "fe9d1397428c77c97755aa7558a6bb19")
SPORT = "basketball_wnba"

MARKETS = {"points": "player_points", "rebounds": "player_rebounds", "assists": "player_assists",
           "pra": "player_points_rebounds_assists", "threes": "player_threes"}
REV = {v: k for k, v in MARKETS.items()}


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower().strip()


def amer_to_dec(a):
    a = float(a)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def list_events():
    r = requests.get(f"https://api.the-odds-api.com/v4/sports/{SPORT}/events", params={"apiKey": KEY})
    r.raise_for_status()
    return r.json()


def event_props(ev_id):
    r = requests.get(f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{ev_id}/odds",
                     params={"apiKey": KEY, "regions": "us",
                             "markets": ",".join(MARKETS.values()), "oddsFormat": "american"})
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()

    events = list_events()
    agg = {}  # (stat, name, disp) -> list of (line, dec_over, dec_under)
    remaining = None
    for e in events:
        try:
            data, remaining = event_props(e["id"])
        except Exception:
            continue
        for bm in data.get("bookmakers", []):
            for m in bm.get("markets", []):
                stat = REV.get(m["key"])
                if not stat:
                    continue
                for player in {o.get("description") for o in m["outcomes"]}:
                    q = predict_day.two_way(m, player)
                    if q is None:
                        continue
                    line, over, under = q
                    agg.setdefault((stat, norm(player), player), []).append(
                        (line, amer_to_dec(over), amer_to_dec(under)))

    rows = []
    for (stat, name, disp), quotes in agg.items():
        lines = [q[0] for q in quotes]
        modal = pd.Series(lines).mode().iloc[0]
        at = [q for q in quotes if q[0] == modal]
        if len(at) < 2:  # need >=2 books at the modal line
            continue
        fair_overs = [predict_day.devig(o, u)[0] for _, o, u in at]
        rows.append(dict(date=args.date, stat=stat, pitcher=disp, name=name,
                         book_line=modal, book_p_over=round(float(np.median(fair_overs)), 4),
                         n_books=len(at)))

    df = pd.DataFrame(rows)
    os.makedirs(BOOKS, exist_ok=True)
    path = os.path.join(BOOKS, f"wnba_books_{args.date}.csv")
    df.to_csv(path, index=False)
    by = df.groupby("stat").size().to_dict() if len(df) else {}
    print(f"[wnba] {args.date}: {len(df)} lines {by} -> {os.path.relpath(path, ROOT)}  (remaining: {remaining})")


if __name__ == "__main__":
    main()
