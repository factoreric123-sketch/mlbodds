"""
exp_blend.py
------------
Follow-up to exp_recency.py: can we get EWMA's better count-accuracy AND flat's
better calibration by combining them?

Two combination strategies, both leak-free, judged on held-out starts:
  A) input blend:  kr = a*ewma + (1-a)*flat, sweep a in [0..1]
  B) both features: give the Poisson model ewma AND flat side-by-side and let it
     weight them itself.

Metrics (held-out, last 30% of season):
  MAE     - expected-K accuracy
  o/u LL  - log-loss of P(over) on the realized over/under at line floor(mu)+0.5.
            This is the calibration metric that actually maps to betting ROI;
            lower is better and it punishes noisy probabilities, not just bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson

import kprops
from exp_recency import prior_rates, FEATS_BASE

EPS = 1e-9


def ou_logloss(mu, y):
    line = np.floor(mu) + 0.5
    p_over = np.clip(1 - poisson.cdf(np.floor(line), mu), EPS, 1 - EPS)
    over = (y > line).astype(float)
    return -(over * np.log(p_over) + (1 - over) * np.log(1 - p_over)).mean()


def fit_eval(tr, te, feats):
    m = PoissonRegressor(alpha=1e-4, max_iter=500).fit(tr[feats], tr["K"])
    mu = m.predict(te[feats])
    y = te["K"].values
    return np.abs(mu - y).mean(), ou_logloss(mu, y), m


def main():
    df = prior_rates(pd.read_parquet(kprops.DATASET))
    lg = df["K"].sum() / df["bf"].sum()
    df["opp_krate"] = df["opp_krate"].fillna(lg)
    d = df[(df.n_prior >= 3) & df.kr_flat.notna() & df.kr_ewma.notna()
           & df.pit_bf_exp.notna()].copy()
    dates = sorted(d["date"].unique())
    cut = dates[int(len(dates) * 0.70)]
    tr, te = d[d.date <= cut].copy(), d[d.date > cut].copy()
    print(f"usable {len(d)} | train {len(tr)} test {len(te)} | cut {cut}\n")

    # reference points
    for name, col in [("flat (current)", "kr_flat"), ("ewma", "kr_ewma")]:
        mae, ll, _ = fit_eval(tr, te, [col] + FEATS_BASE)
        print(f"{name:18s} MAE {mae:.3f}   o/u-LL {ll:.4f}")
    print("-" * 50)

    # A) input blend sweep
    print("A) input blend  kr = a*ewma + (1-a)*flat")
    best = None
    for a in np.linspace(0, 1, 11):
        tr["kr_mix"] = a * tr["kr_ewma"] + (1 - a) * tr["kr_flat"]
        te["kr_mix"] = a * te["kr_ewma"] + (1 - a) * te["kr_flat"]
        mae, ll, _ = fit_eval(tr, te, ["kr_mix"] + FEATS_BASE)
        tag = ""
        if best is None or ll < best[2]:
            best = (a, mae, ll); tag = ""
        print(f"  a={a:.1f}  MAE {mae:.3f}   o/u-LL {ll:.4f}")
    print(f"  -> best by o/u-LL: a={best[0]:.1f} (MAE {best[1]:.3f}, LL {best[2]:.4f})")

    # B) both as separate features
    print("\nB) both features (model weights them)")
    mae, ll, m = fit_eval(tr, te, ["kr_ewma", "kr_flat"] + FEATS_BASE)
    coefs = dict(zip(["kr_ewma", "kr_flat"] + FEATS_BASE, m.coef_.round(3)))
    print(f"  MAE {mae:.3f}   o/u-LL {ll:.4f}")
    print(f"  coefs: {coefs}")


if __name__ == "__main__":
    main()
