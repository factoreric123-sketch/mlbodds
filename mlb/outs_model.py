"""
outs_model.py
-------------
Prototype Pitching-Outs (PO) model, sibling to the strikeout model.

Target = outs recorded in a start (IP*3). Mirrors kprops feature discipline:
every feature uses only PRIOR starts (recency-blended outs/start + expanding mean).
Validates held-out over/under log-loss vs a coin before being trusted.

Run:  python3 outs_model.py            # held-out validation report
      python3 outs_model.py --slate    # score today's PrizePicks slate (reads SLATE below)
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

import kprops

FEATS = ["pit_outs_exp", "pit_bf_exp", "home"]
DECAY = kprops.DECAY  # same half-life=3-start EWMA


def _blend_at(vals: np.ndarray) -> float:
    """Recency blend (0.8 EWMA + 0.2 flat) of a per-start quantity, prior starts only."""
    flat = vals.mean()
    w = DECAY ** np.arange(len(vals) - 1, -1, -1)
    ewma = (w * vals).sum() / w.sum()
    return 0.8 * ewma + 0.2 * flat


def pit_outs_blend(outs: np.ndarray) -> np.ndarray:
    """Per-start blended outs/start using PRIOR starts only. First start NaN."""
    n = len(outs)
    out = np.full(n, np.nan)
    for i in range(1, n):
        out[i] = _blend_at(outs[:i])
    return out


def next_outs(outs) -> float:
    return _blend_at(np.asarray(outs, float))


def add_outs_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["pid", "date", "game_pk"]).reset_index(drop=True)
    df["pit_outs_exp"] = np.nan
    for _, idx in df.groupby("pid", sort=False).groups.items():
        sub = df.loc[idx]
        df.loc[idx, "pit_outs_exp"] = pit_outs_blend(sub["outs"].to_numpy(float))
    return df


def fit_outs(df: pd.DataFrame) -> PoissonRegressor:
    d = df[(df.n_prior >= 3) & df.pit_outs_exp.notna() & df.pit_bf_exp.notna()]
    m = PoissonRegressor(alpha=1e-4, max_iter=500)
    m.fit(d[FEATS], d["outs"])
    return m


def p_over(mu: float, line: float) -> float:
    """P(outs > line). Lines are .5 so floor(line) is the highest losing integer."""
    return float(1 - poisson.cdf(np.floor(line), mu))


def holdout_report(df: pd.DataFrame):
    """Walk-forward by date: train on past, predict next day, score o/u log-loss.
    Uses each row's own line as the median? No line in data -> use the model's own
    mean as a self-consistency check is meaningless. Instead grade vs a synthetic
    line = round(prior blended outs) to test calibration of the over/under split."""
    df = add_outs_features(df)
    dates = sorted(df.date.unique())
    rows = []
    for d in dates:
        train = df[df.date < d]
        test = df[df.date == d]
        if len(train) < 200 or test.empty:
            continue
        warm = test[(test.n_prior >= 3) & test.pit_outs_exp.notna() & test.pit_bf_exp.notna()]
        if warm.empty:
            continue
        m = fit_outs(train)
        mu = m.predict(warm[FEATS])
        # synthetic line at .5 below each pitcher's blended expectation -> tests
        # whether P(over) is calibrated against actual over/under outcomes
        line = np.floor(warm["pit_outs_exp"].to_numpy()) + 0.5
        po = 1 - poisson.cdf(np.floor(line), mu)
        actual_over = (warm["outs"].to_numpy() > line).astype(int)
        rows.append(pd.DataFrame({"po": po, "y": actual_over}))
    R = pd.concat(rows, ignore_index=True)
    p = np.clip(R["po"].to_numpy(), 1e-6, 1 - 1e-6)
    y = R["y"].to_numpy()
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    base = y.mean()
    coin = 0.6931
    # dispersion check
    disp = df["outs"].var() / df["outs"].mean()
    print(f"PO held-out o/u log-loss: {ll:.4f}  (coin {coin:.4f})")
    print(f"  n graded: {len(y)} | over-rate: {base:.3f} | outs var/mean: {disp:.2f}")
    print(f"  -> {'BEATS coin (keep)' if ll < coin else 'WORSE than coin (reject)'}")
    return ll


# --- today's PrizePicks slate (name, opp_team_name, line, is_home, goblin) ---
SLATE = [
    ("Trey Gibson",        "Chicago White Sox",  14.5, True,  False),
    ("Erick Fedde",        "Baltimore Orioles",  14.5, False, False),
    ("Tanner Bibee",       "Texas Rangers",      17.5, True,  True),
    ("Jacob deGrom",       "Cleveland Guardians",17.5, False, True),
    ("Cristopher Sánchez", "Pittsburgh Pirates", 19.5, True,  False),
    ("Bubba Chandler",     "Philadelphia Phillies",14.5,False, True),
    ("Cam Schlittler",     "Detroit Tigers",     17.5, True,  True),
    ("Tarik Skubal",       "New York Yankees",   17.5, False, True),
    ("Kevin Gausman",      "New York Mets",      17.5, True,  True),
    ("Nolan McLean",       "Toronto Blue Jays",  17.5, False, True),
    ("Cade Cavalli",       "Boston Red Sox",     15.5, False, False),
    ("Connelly Early",     "Washington Nationals",17.0,True,  False),
    ("Matthew Liberatore", "Atlanta Braves",     15.5, False, False),
    ("Martín Pérez",       "St. Louis Cardinals",14.5, True,  True),
    ("Rhett Lowder",       "Milwaukee Brewers",  14.5, False, True),
    ("Brandon Sproat",     "Cincinnati Reds",    14.5, True,  True),
    ("Noah Cameron",       "Tampa Bay Rays",     17.0, True,  False),
    ("Griffin Jax",        "Kansas City Royals", 14.5, False, True),
    ("Matthew Boyd",       "San Diego Padres",   14.5, True,  True),
    ("JP Sears",           "Chicago Cubs",       14.5, False, True),
    ("Joe Ryan",           "Houston Astros",     17.5, False, True),
    ("Mike Burrows",       "Minnesota Twins",    16.5, True,  False),
    ("Eury Pérez",         "Colorado Rockies",   14.5, False, True),
    ("Tanner Gordon",      "Miami Marlins",      10.5, True,  False),
    ("Bryan Woo",          "Los Angeles Angels", 17.5, True,  True),
    ("José Soriano",       "Seattle Mariners",   14.5, False, True),
    ("Landen Roupp",       "Arizona Diamondbacks",14.5,False, True),
    ("Justin Wrobleski",   "Athletics",          17.5, False, True),
    ("Jeffrey Springs",    "Los Angeles Dodgers",14.5, True,  True),
]


def score_slate(df: pd.DataFrame):
    import unicodedata
    def norm(n):
        n = unicodedata.normalize("NFKD", str(n))
        n = "".join(c for c in n if not unicodedata.combining(c))
        return " ".join(n.lower().split())

    df = add_outs_features(df)
    model = fit_outs(df)
    # latest blended outs/bf per pitcher (all prior starts)
    g = df.sort_values(["pid", "date", "game_pk"]).groupby("name")
    last = {}
    for name, sub in g:
        last[norm(name)] = dict(
            outs_exp=next_outs(sub["outs"].to_numpy(float)),
            bf_exp=sub["bf"].mean(),
            n=len(sub),
        )

    out = []
    for name, opp, line, home, goblin in SLATE:
        k = norm(name)
        if k not in last:
            out.append((name, line, goblin, None, None, None, last.get(k), "NOT in universe"))
            continue
        info = last[k]
        x = pd.DataFrame([[info["outs_exp"], info["bf_exp"], int(home)]], columns=FEATS)
        mu = float(model.predict(x)[0])
        po = p_over(mu, line)
        out.append((name, line, goblin, mu, po, info["n"], None, ""))

    print(f"{'Pitcher':22} {'Line':>5} {'Gob':>4} {'muOuts':>7} {'P(Over)':>8} {'n':>3}  pick")
    print("-" * 70)
    picks = []
    for name, line, goblin, mu, po, n, _, note in out:
        if mu is None:
            print(f"{name:22} {line:5.1f} {'G' if goblin else '-':>4} {'--':>7} {'--':>8} {'--':>3}  {note}")
            continue
        side = "MORE" if po >= 0.5 else "LESS"
        conf = abs(po - 0.5)
        print(f"{name:22} {line:5.1f} {'G' if goblin else '-':>4} {mu:7.2f} {po:8.3f} {n:3d}  {side} ({conf:.3f})")
        picks.append((name, line, goblin, side, po, conf))
    print("-" * 70)
    # PrizePicks standard (non-goblin) needs ~ >57-58% per leg to clear 2-pick break-even.
    picks.sort(key=lambda t: -t[5])
    print("\nStrongest leans (|P-0.5| desc):")
    for name, line, goblin, side, po, conf in picks[:8]:
        tag = "GOBLIN (shaded payout!)" if goblin else "standard"
        print(f"  {side:4} {name:20} {line:5.1f}  P(over)={po:.3f}  [{tag}]")


if __name__ == "__main__":
    df = pd.read_parquet(kprops.DATASET)
    if "--slate" in sys.argv:
        score_slate(df)
    else:
        holdout_report(df)
