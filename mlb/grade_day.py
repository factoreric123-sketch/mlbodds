"""
grade_day.py
------------
Forward-test step 2: grade logged bets after games finish.

Reads mlb/logs/bets_<DATE>.csv, looks up each pitcher's actual strikeouts for
that date via StatsAPI, marks win/loss/push, computes per-bet profit at the
logged decimal price (flat 1u stake), and prints running ROI across all graded
days. Updates the CSVs in place (graded/actual_k/result columns).

Only bets that cleared MIN_EDGE in predict_day.py are counted toward ROI; the
rest are logged for calibration analysis but not "bet".

Usage:
    python3 grade_day.py                 # grade all ungraded logs
    python3 grade_day.py --date 2026-06-30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import unicodedata
import urllib.request

import pandas as pd

import numpy as np

import predict_day  # reuse odds helpers (list_events, event_props, two_way, devig)

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
MIN_EDGE = 0.04


def norm_name(name: str) -> str:
    """Accent-fold + lowercase + collapse whitespace for cross-source name matching."""
    n = unicodedata.normalize("NFKD", str(name))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(n.lower().split())


def http(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def actual_ks(date: str) -> dict:
    """{pitcher_name: strikeouts} for every starter on `date` (final games only)."""
    sched = http(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}")
    game_pks = []
    for d in sched.get("dates", []):
        for g in d["games"]:
            if g.get("status", {}).get("abstractGameState") == "Final":
                game_pks.append(g["gamePk"])
    ks = {}
    for pk in game_pks:
        box = http(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore")
        for side in ("home", "away"):
            players = box["teams"][side]["players"]
            for pdata in players.values():
                st = pdata.get("stats", {}).get("pitching", {})
                if st.get("gamesStarted", 0) == 1:
                    name = pdata["person"]["fullName"]
                    ks[norm_name(name)] = st.get("strikeOuts", 0)
    return ks


def closing_quotes() -> dict:
    """{norm_name: {line -> [(over, under), ...]}} from current live odds.
    Run near first pitch so the snapshot approximates the closing line (the free
    odds tier has no historical/closing endpoint)."""
    out = {}
    for ev in predict_day.list_events():
        try:
            data, _ = predict_day.event_props(ev["id"])
        except Exception:
            continue
        for b in data.get("bookmakers", []):
            for mk in b["markets"]:
                for pit in {o.get("description") for o in mk["outcomes"]}:
                    tw = predict_day.two_way(mk, pit)
                    if not tw:
                        continue
                    line, po, pu = tw
                    out.setdefault(norm_name(pit), {}).setdefault(line, []).append((po, pu))
    return out


def snapshot_closing(path: str) -> None:
    """Record the current (≈closing) line for each logged bet's side -> CLV columns."""
    df = pd.read_csv(path)
    quotes = closing_quotes()

    def cap(row):
        q = quotes.get(norm_name(row["pitcher"]))
        if not q:
            return pd.Series([np.nan, np.nan, np.nan])
        # match the line we bet if still offered, else the most-booked line
        line = row["line"] if row["line"] in q else max(q, key=lambda L: len(q[L]))
        books = q[line]
        med_o, med_u = float(np.median([o for o, _ in books])), float(np.median([u for _, u in books]))
        fo, fu = predict_day.devig(med_o, med_u)
        if row["side"] == "Over":
            close_price, close_fair = max(o for o, _ in books), fo
        else:
            close_price, close_fair = max(u for _, u in books), fu
        return pd.Series([line, round(close_price, 3), round(close_fair, 3)])

    df[["close_line", "close_price", "close_fair"]] = df.apply(cap, axis=1)
    # CLV in no-vig prob terms: our side's closing fair prob minus the fair prob we logged.
    # Positive => the market moved toward our side after we bet (we beat the close).
    df["clv"] = (df["close_fair"] - df["market_p"]).round(3)
    df.to_csv(path, index=False)
    got = df["close_fair"].notna().sum()
    print(f"{os.path.basename(path)}: captured closing line for {got}/{len(df)} bets")


def grade_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date = df["target_date"].iloc[0]
    ks = actual_ks(date)
    if not ks:
        print(f"{os.path.basename(path)}: no final games yet, skipping")
        return None

    def settle(row):
        key = norm_name(row["pitcher"])
        if key not in ks:
            return pd.Series(["", "", ""])
        k = ks[key]
        line = row["line"]
        if k == line:
            res = "push"
        elif (row["side"] == "Over") == (k > line):
            res = "win"
        else:
            res = "loss"
        return pd.Series(["Y", k, res])

    df[["graded", "actual_k", "result"]] = df.apply(settle, axis=1)
    df.to_csv(path, index=False)
    return df


def profit(row) -> float:
    """Flat 1u stake at decimal price. push=0, loss=-1, win=price-1."""
    if row["result"] == "win":
        return row["price"] - 1
    if row["result"] == "loss":
        return -1.0
    return 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--snapshot", action="store_true",
                    help="capture current (~closing) lines into the log for CLV, then exit")
    args = ap.parse_args(argv)

    files = ([os.path.join(LOGS, f"bets_{args.date}.csv")] if args.date
             else sorted(glob.glob(os.path.join(LOGS, "bets_*.csv"))))

    if args.snapshot:
        for f in files:
            if not os.path.exists(f):
                print("missing", f); continue
            snapshot_closing(f)
        return

    graded = []
    for f in files:
        if not os.path.exists(f):
            print("missing", f); continue
        out = grade_file(f)
        if out is not None:
            graded.append(out)

    if not graded:
        print("nothing graded.")
        return

    allg = pd.concat(graded, ignore_index=True)
    # only real bets: cleared edge AND were settled (pitcher actually started)
    bets = allg[(allg.edge >= MIN_EDGE) & (allg.result.isin(["win", "loss", "push"]))].copy()
    if bets.empty:
        print("no settled bets above edge threshold yet.")
        return
    bets["profit"] = bets.apply(profit, axis=1)

    n = len(bets)
    wins = (bets.result == "win").sum()
    losses = (bets.result == "loss").sum()
    pushes = (bets.result == "push").sum()
    pnl = bets.profit.sum()
    roi = pnl / (n - pushes) if (n - pushes) else 0.0

    print(f"\n=== forward-test running totals ===")
    print(f"bets graded: {n}  (W {wins} / L {losses} / push {pushes})")
    print(f"P&L: {pnl:+.2f}u  | ROI: {roi:+.1%}  (per non-push unit staked)")
    if "clv" in bets and bets["clv"].notna().any():
        clv = bets["clv"].dropna()
        beat = (clv > 0).mean()
        print(f"CLV: mean {clv.mean():+.3f} fair-prob  | beat close {beat:.0%} "
              f"({clv.notna().sum()} bets w/ closing line)")
    print(f"\nby day:")
    by = bets.groupby("target_date").agg(
        bets=("profit", "size"), pnl=("profit", "sum")).round(2)
    print(by.to_string())
    print(f"\nrecent settled bets:")
    cols = ["target_date", "pitcher", "side", "line", "price",
            "actual_k", "result", "edge"]
    print(bets[cols].tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
