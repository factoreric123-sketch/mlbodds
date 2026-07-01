"""
kprops.py
---------
Shared logic for the MLB pitcher-strikeout prop forward-test.

Pipeline:
  build_dataset()  -> pull every tracked starter's 2026 game log from MLB StatsAPI,
                      compute leak-free features (rolling pit_krate, opp_krate),
                      save mlb/data/dataset.parquet.
  fit_model(df)    -> Poisson GLM on expected strikeouts.

Leak discipline: every feature uses only starts strictly BEFORE the row's date
(shift(1) + expanding). K counts are ~Poisson (var/mean ~ 1.07), so a Poisson
model is the correct distribution for over/under probabilities.

No paid data, no installs beyond sklearn/scipy/pandas. StatsAPI is free + keyless.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
DATASET = os.path.join(DATA, "dataset.parquet")
PITCHERS = os.path.join(DATA, "pitchers.json")

SEASON = 2026
FEATS = ["pit_krate", "opp_krate", "pit_bf_exp", "home"]
DECAY = 0.5 ** (1 / 3)   # EWMA decay, half-life = 3 starts


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def ip_to_outs(ip) -> int:
    """'6.2' innings -> 6*3 + 2 outs."""
    whole, _, frac = str(ip).partition(".")
    return int(whole) * 3 + (int(frac) if frac else 0)


def fetch_starters(limit: int = 150) -> list:
    """Qualified 2026 starters sorted by strikeouts. Returns [[pid,name,gs,k,bf],...]."""
    url = (f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=pitching"
           f"&season={SEASON}&sportId=1&playerPool=qualified&limit={limit}"
           f"&sortStat=strikeOuts")
    data = _get(url)
    out = []
    for split in data["stats"][0]["splits"]:
        p = split["player"]
        s = split["stat"]
        out.append([p["id"], p["fullName"], s.get("gamesStarted", 0),
                    s.get("strikeOuts", 0), s.get("battersFaced", 0)])
    return out


def fetch_gamelog(pid: int) -> list:
    """Per-start rows for one pitcher's 2026 season."""
    url = (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=gameLog"
           f"&group=pitching&season={SEASON}")
    data = _get(url)
    rows = []
    splits = data["stats"][0]["splits"] if data.get("stats") else []
    for sp in splits:
        st = sp["stat"]
        if st.get("gamesStarted", 0) != 1:   # starts only
            continue
        opp = sp.get("opponent", {})
        rows.append(dict(
            date=sp["date"],
            game_pk=sp.get("game", {}).get("gamePk"),
            opp_id=opp.get("id"),
            opp=opp.get("name"),
            home=sp.get("isHome", False),
            K=st.get("strikeOuts", 0),
            bf=st.get("battersFaced", 0),
            outs=ip_to_outs(st.get("inningsPitched", "0.0")),
        ))
    return rows


def _blend_at(pk: np.ndarray, pbf: np.ndarray) -> float:
    """
    Recency-weighted K-rate from a pitcher's prior starts (pk, pbf chronological,
    oldest first). Blend = 0.8*EWMA + 0.2*flat, EWMA half-life = 3 starts. Chosen
    by held-out over/under log-loss (see exp_blend.py); beats pure flat and pure EWMA.
    """
    flat = pk.sum() / pbf.sum()
    w = DECAY ** np.arange(len(pk) - 1, -1, -1)      # newest weight 1
    ewma = (w * pk).sum() / (w * pbf).sum()
    return 0.8 * ewma + 0.2 * flat


def pit_krate_blend(k: np.ndarray, bf: np.ndarray) -> np.ndarray:
    """Per-start blended K-rate using PRIOR starts only. First start is NaN."""
    n = len(k)
    out = np.full(n, np.nan)
    for i in range(1, n):
        out[i] = _blend_at(k[:i], bf[:i])
    return out


def next_krate(k: np.ndarray, bf: np.ndarray) -> float:
    """Blended K-rate for a pitcher's NEXT (upcoming) start, given all prior starts."""
    return _blend_at(np.asarray(k, float), np.asarray(bf, float))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-free rolling features. Every value uses only prior starts."""
    df = df.sort_values(["pid", "date", "game_pk"]).reset_index(drop=True)
    # blended pit_krate, assigned by position per pitcher (no index realignment)
    df["pit_krate"] = np.nan
    for _, idx in df.groupby("pid", sort=False).groups.items():
        sub = df.loc[idx]
        vals = pit_krate_blend(sub["K"].to_numpy(float), sub["bf"].to_numpy(float))
        df.loc[idx, "pit_krate"] = vals
    g = df.groupby("pid", group_keys=False)
    df["pit_bf_exp"] = g["bf"].apply(lambda s: s.shift(1).expanding().mean())
    df["n_prior"] = g.cumcount()

    df = df.sort_values(["opp_id", "date", "game_pk"])
    go = df.groupby("opp_id", group_keys=False)
    df["opp_cum_k"] = go["K"].apply(lambda s: s.shift(1).cumsum())
    df["opp_cum_bf"] = go["bf"].apply(lambda s: s.shift(1).cumsum())
    df["opp_krate"] = df["opp_cum_k"] / df["opp_cum_bf"]
    df["opp_n_prior"] = go.cumcount()

    # Cold-start fill for opp_krate: use the league K-rate from games STRICTLY BEFORE
    # each row's date (not the whole-season mean, which would leak future data backward).
    by_date = df.groupby("date")[["K", "bf"]].sum().sort_index()
    prior_k = by_date["K"].cumsum().shift(1)
    prior_bf = by_date["bf"].cumsum().shift(1)
    lg_by_date = (prior_k / prior_bf)                       # league rate prior to each date
    lg_global = df["K"].sum() / df["bf"].sum()              # only for the earliest date(s)
    lg_fill = df["date"].map(lg_by_date).fillna(lg_global)
    df["opp_krate"] = df["opp_krate"].fillna(lg_fill)
    return df.sort_values(["date", "pid"]).reset_index(drop=True)


def build_dataset(save: bool = True) -> pd.DataFrame:
    """Pull all starters' logs, build the full leak-free start table."""
    starters = fetch_starters()
    with open(PITCHERS, "w") as fh:
        json.dump(starters, fh)
    rows = []
    for pid, name, *_ in starters:
        for r in fetch_gamelog(pid):
            r["pid"], r["name"] = pid, name
            rows.append(r)
        time.sleep(0.05)
    df = pd.DataFrame(rows)
    df = add_features(df)
    if save:
        df.to_parquet(DATASET)
    return df


def fit_model(df: pd.DataFrame) -> PoissonRegressor:
    """Poisson GLM on starts where features are warm (>=3 prior starts)."""
    d = df[(df.n_prior >= 3) & df.pit_krate.notna() & df.pit_bf_exp.notna()]
    m = PoissonRegressor(alpha=1e-4, max_iter=500)
    m.fit(d[FEATS], d["K"])
    return m


def team_krate(df: pd.DataFrame) -> dict:
    """Season-to-date strikeout rate by opponent team name -> rate."""
    g = df.groupby("opp")
    rates = (g["K"].sum() / g["bf"].sum()).to_dict()
    return rates


if __name__ == "__main__":
    df = build_dataset()
    print(f"dataset: {len(df)} starts, {df.pid.nunique()} pitchers, "
          f"{df.date.min()}..{df.date.max()}")
