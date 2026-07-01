"""
predict_day.py
--------------
Daily forward-test step 1. Predicts strikeouts for UPCOMING (not-yet-played)
starts only, compares to the de-vigged market line, logs candidates.

Flow (leak-free):
  1. StatsAPI schedule -> probable pitchers for a future date (default tomorrow),
     each mapped to their team and the OPPONENT team.
  2. snapshot(df, before=date) -> each pitcher's season-to-date features using
     only starts BEFORE the target date (excludes the game we're predicting).
  3. the-odds-api -> live pitcher_strikeouts lines for that pitcher.
  4. Poisson model -> P(over). Edge vs de-vigged market. Log to bets_<DATE>.csv.

Why "future date": the prop must be for a game that hasn't happened, otherwise
the pitcher's own result has leaked into the season-to-date features (this bit us
in testing: a settled line for an already-final game produced a phantom +38% edge).

Usage:
    python3 predict_day.py                 # predicts tomorrow's slate
    python3 predict_day.py --date 2026-06-30
    python3 predict_day.py --refresh       # rebuild dataset from StatsAPI first
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import poisson

import kprops

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
os.makedirs(LOGS, exist_ok=True)

ODDS_KEY = os.environ.get("ODDS_API_KEY", "fe9d1397428c77c97755aa7558a6bb19")
SPORT = "baseball_mlb"
MIN_EDGE = 0.04


def norm_name(name: str) -> str:
    """Accent-fold + lowercase + collapse whitespace, so 'José Soriano' (StatsAPI)
    matches 'Jose Soriano' (odds feed). Used only for cross-source matching, never
    for display/logging."""
    n = unicodedata.normalize("NFKD", str(name))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(n.lower().split())


def http(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r), r.headers


def probables(date: str) -> dict:
    """{norm_name: (display_name, opponent_team_name, is_home)} for probable SPs.
    Keyed on the accent-folded name so the odds feed can match it."""
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
           f"&hydrate=probablePitcher")
    data, _ = http(url)
    out = {}
    for d in data.get("dates", []):
        for g in d["games"]:
            h, a = g["teams"]["home"], g["teams"]["away"]
            hp = h.get("probablePitcher", {}).get("fullName")
            ap = a.get("probablePitcher", {}).get("fullName")
            if hp:
                out[norm_name(hp)] = (hp, a["team"]["name"], True)   # home faces away
            if ap:
                out[norm_name(ap)] = (ap, h["team"]["name"], False)  # away faces home
    return out


def snapshot_before(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Per-pitcher state for the next start, using only starts strictly before `date`.
    cur_krate uses the same recency blend as the trained model (kprops.next_krate)."""
    d = df[df.date < date].sort_values("date")
    rows = []
    for pid, g in d.groupby("pid", sort=False):
        rows.append(dict(
            pid=pid, name=g["name"].iloc[0],
            cur_krate=kprops.next_krate(g["K"].to_numpy(float), g["bf"].to_numpy(float)),
            cur_bf_exp=g["bf"].mean(),
            cur_n=len(g),
        ))
    out = pd.DataFrame(rows)
    out["key"] = out["name"].map(norm_name)
    return out.set_index("key")


def team_krate_before(df: pd.DataFrame, date: str):
    """(per-opponent season-to-date K-rate dict, prior-to-date league K-rate).
    The league rate is the cold-start fallback, computed from data rather than a
    hardcoded constant so it stays correct early in a season (matches training)."""
    d = df[df.date < date]
    g = d.groupby("opp")
    okr = (g["K"].sum() / g["bf"].sum()).to_dict()
    lg = d["K"].sum() / d["bf"].sum()
    return okr, lg


def list_events():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events?apiKey={ODDS_KEY}"
    return http(url)[0]


def event_props(event_id: str):
    url = (f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds"
           f"?apiKey={ODDS_KEY}&regions=us&markets=pitcher_strikeouts&oddsFormat=decimal")
    data, hdr = http(url)
    return data, hdr.get("x-requests-remaining")


def devig(po: float, pu: float):
    io, iu = 1 / po, 1 / pu
    s = io + iu
    return io / s, iu / s


def p_over(mu: float, line: float) -> float:
    return float(1 - poisson.cdf(np.floor(line), mu))


def two_way(market: dict, pitcher: str):
    over = under = line = None
    for o in market["outcomes"]:
        if o.get("description") != pitcher:
            continue
        line = o.get("point")
        if o["name"] == "Over":
            over = o["price"]
        elif o["name"] == "Under":
            under = o["price"]
    return (line, over, under) if (over and under and line is not None) else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=(dt.date.today() + dt.timedelta(days=1)).isoformat())
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--max-events", type=int, default=20)
    args = ap.parse_args(argv)
    target = args.date

    df = kprops.build_dataset() if args.refresh else pd.read_parquet(kprops.DATASET)
    model = kprops.fit_model(df)
    snap = snapshot_before(df, target)
    team_kr, lg_krate = team_krate_before(df, target)
    probs = probables(target)
    print(f"target {target} | {len(probs)} probable SPs | "
          f"{len(snap)} pitchers known | model ready")

    events = list_events()
    remaining = None
    # Collect every book's two-way quote per pitcher, then settle ONE bet per pitcher
    # at the consensus fair line + best available price (no duplicate-book inflation).
    quotes = {}   # key -> {name, opp, is_home, line -> [(book, over, under), ...]}
    for ev in events[: args.max_events]:
        try:
            data, remaining = event_props(ev["id"])
        except Exception as e:
            print("  odds err", e); continue
        if not data.get("bookmakers"):
            continue
        for b in data["bookmakers"]:
            for mk in b["markets"]:
                for pit in {o.get("description") for o in mk["outcomes"]}:
                    key = norm_name(pit)
                    if key not in snap.index or key not in probs:
                        continue  # must be tracked AND a confirmed probable SP today
                    tw = two_way(mk, pit)
                    if not tw:
                        continue
                    line, po, pu = tw
                    name, opp_team, is_home = probs[key]
                    q = quotes.setdefault(key, dict(
                        name=name, opp=opp_team, is_home=is_home, lines={}))
                    q["lines"].setdefault(line, []).append((b["key"], po, pu))
        time.sleep(0.15)

    rows = []
    for key, q in quotes.items():
        # consensus line = the one the most books posted (ties -> lowest line)
        line = max(q["lines"], key=lambda L: (len(q["lines"][L]), -L))
        books = q["lines"][line]
        overs = [o for _, o, _ in books]
        unders = [u for _, _, u in books]
        # consensus fair from de-vigged MEDIAN prices (robust to one stale book)
        med_o, med_u = float(np.median(overs)), float(np.median(unders))
        fo, fu = devig(med_o, med_u)
        okr = team_kr.get(q["opp"], lg_krate)
        s = snap.loc[key]
        x = pd.DataFrame([[s.cur_krate, okr, s.cur_bf_exp, int(q["is_home"])]],
                         columns=kprops.FEATS)
        mu = float(model.predict(x)[0])
        mp_o = p_over(mu, line); mp_u = 1 - mp_o
        eo, eu = mp_o - fo, mp_u - fu
        if eo >= eu:
            side, edge, mp, fair = "Over", eo, mp_o, fo
            best_book, best_price = max(((bk, o) for bk, o, _ in books), key=lambda t: t[1])
        else:
            side, edge, mp, fair = "Under", eu, mp_u, fu
            best_book, best_price = max(((bk, u) for bk, _, u in books), key=lambda t: t[1])
        rows.append(dict(
            target_date=target, pitcher=q["name"], book=best_book,
            opp=q["opp"], line=line, side=side, price=round(best_price, 3),
            mu=round(mu, 2), model_p=round(mp, 3), market_p=round(fair, 3),
            edge=round(edge, 3), opp_krate=round(okr, 3), cur_n=int(s.cur_n),
            n_books=len(books), graded="", actual_k="", result="",
        ))

    if not rows:
        print(f"no tracked probable-SP K-prop lines yet for {target}.")
        print(f"(odds requests remaining: {remaining})")
        return

    rdf = pd.DataFrame(rows)
    out_path = os.path.join(LOGS, f"bets_{target}.csv")
    rdf.to_csv(out_path, index=False)
    bets = rdf[rdf.edge >= MIN_EDGE].sort_values("edge", ascending=False)
    print(f"\nlogged {len(rdf)} lines -> {out_path}  | requests remaining: {remaining}")
    print(f"\n{len(bets)} bets clear edge >= {MIN_EDGE}:")
    if len(bets):
        cols = ["pitcher", "book", "opp", "line", "side", "price",
                "mu", "model_p", "market_p", "edge"]
        print(bets[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
