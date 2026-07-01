"""
hits_model.py
-------------
Hits-Allowed model for pitcher props. Mirrors the strikeout model exactly:
hits ~ Poisson (verified var/mean = 0.99), so a Poisson GLM is the correct
distribution for over/under probabilities.

Features (all leak-free, prior starts only):
    pit_hrate  : recency-blended hits/BF allowed (0.8 EWMA hl=3 + 0.2 flat)
    opp_hrate  : opponent team's hits/BF drawn, prior-to-date (cold-start = league)
    pit_bf_exp : expanding mean of prior BF (reused from K dataset)
    home       : home/away

Run:  python3 hits_model.py            # held-out o/u log-loss vs coin
      python3 hits_model.py --slate    # score today's board (reads SLATE)
"""
from __future__ import annotations

import sys
import unicodedata
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

import kprops

FEATS = ["pit_hrate", "opp_hrate", "pit_bf_exp", "home"]
DECAY = kprops.DECAY


def norm(n):
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(n.lower().split())


def _blend_at(h, bf):
    flat = h.sum() / bf.sum()
    w = DECAY ** np.arange(len(h) - 1, -1, -1)
    ewma = (w * h).sum() / (w * bf).sum()
    return 0.8 * ewma + 0.2 * flat


def pit_hrate_blend(h, bf):
    out = np.full(len(h), np.nan)
    for i in range(1, len(h)):
        out[i] = _blend_at(h[:i], bf[:i])
    return out


def next_hrate(h, bf):
    return _blend_at(np.asarray(h, float), np.asarray(bf, float))


def add_hits_features(df):
    """Leak-free rolling hit-rate features. Mirrors kprops.add_features."""
    df = df.sort_values(["pid", "date", "game_pk"]).reset_index(drop=True)
    df["pit_hrate"] = np.nan
    for _, idx in df.groupby("pid", sort=False).groups.items():
        sub = df.loc[idx]
        df.loc[idx, "pit_hrate"] = pit_hrate_blend(
            sub["hits"].to_numpy(float), sub["bf"].to_numpy(float))

    df = df.sort_values(["opp_id", "date", "game_pk"])
    go = df.groupby("opp_id", group_keys=False)
    opp_cum_h = go["hits"].apply(lambda s: s.shift(1).cumsum())
    opp_cum_bf = go["bf"].apply(lambda s: s.shift(1).cumsum())
    df["opp_hrate"] = opp_cum_h / opp_cum_bf

    # cold-start fill: league hit-rate from games strictly BEFORE each date
    by_date = df.groupby("date")[["hits", "bf"]].sum().sort_index()
    prior_h = by_date["hits"].cumsum().shift(1)
    prior_bf = by_date["bf"].cumsum().shift(1)
    lg_by_date = prior_h / prior_bf
    lg_global = df["hits"].sum() / df["bf"].sum()
    lg_fill = df["date"].map(lg_by_date).fillna(lg_global)
    df["opp_hrate"] = df["opp_hrate"].fillna(lg_fill)
    return df.sort_values(["date", "pid"]).reset_index(drop=True)


def fit_hits(df):
    d = df[(df.n_prior >= 3) & df.pit_hrate.notna() & df.pit_bf_exp.notna()]
    m = PoissonRegressor(alpha=1e-4, max_iter=500)
    m.fit(d[FEATS], d["hits"])
    return m


def p_over(mu, line):
    return float(1 - poisson.cdf(np.floor(line), mu))


def holdout_report(df):
    df = add_hits_features(df)
    dates = sorted(df.date.unique())
    rows = []
    for d in dates:
        train = df[df.date < d]
        test = df[df.date == d]
        if len(train) < 200 or test.empty:
            continue
        warm = test[(test.n_prior >= 3) & test.pit_hrate.notna() & test.pit_bf_exp.notna()]
        if warm.empty:
            continue
        m = fit_hits(train)
        mu = m.predict(warm[FEATS])
        line = np.floor(warm["pit_hrate"].to_numpy() * warm["pit_bf_exp"].to_numpy()) + 0.5
        po = 1 - poisson.cdf(np.floor(line), mu)
        y = (warm["hits"].to_numpy() > line).astype(int)
        rows.append(pd.DataFrame({"po": po, "y": y}))
    R = pd.concat(rows, ignore_index=True)
    p = np.clip(R["po"].to_numpy(), 1e-6, 1 - 1e-6)
    y = R["y"].to_numpy()
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    disp = df["hits"].var() / df["hits"].mean()
    print(f"HITS held-out o/u log-loss: {ll:.4f}  (coin 0.6931)")
    print(f"  n graded: {len(y)} | over-rate: {y.mean():.3f} | hits var/mean: {disp:.2f}")
    print(f"  -> {'BEATS coin (keep)' if ll < 0.6931 else 'WORSE (reject)'}")
    return ll


# (name, opp_team, line, is_home) -- today's PrizePicks Hits Allowed board
SLATE = []


def score_slate(df):
    df = add_hits_features(df)
    model = fit_hits(df)
    g = df.groupby("opp")
    opp_hr = (g["hits"].sum() / g["bf"].sum()).to_dict()
    lg = df["hits"].sum() / df["bf"].sum()
    gp = df.sort_values(["pid", "date", "game_pk"]).groupby("name")
    last = {}
    for name, sub in gp:
        last[norm(name)] = dict(
            hrate=next_hrate(sub["hits"].to_numpy(float), sub["bf"].to_numpy(float)),
            bf_exp=sub["bf"].mean(), n=len(sub))

    print(f"{'Pitcher':21}{'Line':>5}{'muH':>6}{'P(Ov)':>7}{'n':>4}  pick(|edge|)")
    print("-" * 60)
    for name, opp, line, home in SLATE:
        k = norm(name)
        if k not in last:
            print(f"{name:21}{line:5.1f}{'--':>6}{'--':>7}{'--':>4}  not in universe"); continue
        info = last[k]
        okr = opp_hr.get(opp, lg)
        x = pd.DataFrame([[info["hrate"], okr, info["bf_exp"], int(home)]], columns=FEATS)
        mu = float(model.predict(x)[0]); po = p_over(mu, line)
        side = "MORE" if po >= 0.5 else "LESS"
        print(f"{name:21}{line:5.1f}{mu:6.2f}{po:7.3f}{info['n']:4d}  {side} ({abs(po-0.5):.3f})")


if __name__ == "__main__":
    df = pd.read_parquet(kprops.DATASET)
    if "--slate" in sys.argv:
        score_slate(df)
    else:
        holdout_report(df)
