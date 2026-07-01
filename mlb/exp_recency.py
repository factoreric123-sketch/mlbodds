"""
exp_recency.py
--------------
Experiment (does NOT touch the production model): does a recency-weighted
pitcher K-rate predict held-out starts better than the flat season average?

For each start we build several PRIOR-ONLY (shift(1)) K-rate estimates:
  flat   : cumulative K / cumulative BF over all prior starts        (current model)
  lastN  : K-rate over the last N starts only (N=3,5)
  ewma   : exponentially-weighted K-rate (half-life in starts)
  blend  : 0.6*last5 + 0.4*flat (shrink noisy short window to season)

Then a temporal split (train early, test late) and compare each as the
pit_krate input to the same Poisson model. Metric: MAE of expected K, plus
over/under calibration. Winner is whatever lowers held-out error; recency that
adds noise will LOSE here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson

import kprops

FEATS_BASE = ["opp_krate", "pit_bf_exp", "home"]


def prior_rates(df: pd.DataFrame) -> pd.DataFrame:
    """All leak-free K-rate variants, per pitcher, using prior starts only."""
    df = df.sort_values(["pid", "date"]).copy()
    decay = 0.5 ** (1 / 3)   # exponential weights, half-life = 3 starts
    parts = []
    for pid, g in df.groupby("pid", sort=False):
        g = g.copy()
        k = g["K"].to_numpy(float)
        bf = g["bf"].to_numpy(float)
        n = len(g)
        flat = np.full(n, np.nan)
        last3 = np.full(n, np.nan)
        last5 = np.full(n, np.nan)
        ewma = np.full(n, np.nan)
        for i in range(1, n):
            pk, pbf = k[:i], bf[:i]
            flat[i] = pk.sum() / pbf.sum()
            last3[i] = pk[-3:].sum() / pbf[-3:].sum()
            last5[i] = pk[-5:].sum() / pbf[-5:].sum()
            w = decay ** np.arange(i - 1, -1, -1)   # newest weight 1
            ewma[i] = (w * pk).sum() / (w * pbf).sum()
        g["kr_flat"] = flat
        g["kr_last3"] = last3
        g["kr_last5"] = last5
        g["kr_ewma"] = ewma
        g["kr_blend"] = 0.6 * last5 + 0.4 * flat
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["date", "pid"]).reset_index(drop=True)


def evaluate(df: pd.DataFrame, ratecol: str, tr, te):
    feats = [ratecol] + FEATS_BASE
    m = PoissonRegressor(alpha=1e-4, max_iter=500).fit(tr[feats], tr["K"])
    mu = m.predict(te[feats])
    mae = np.abs(mu - te["K"].values).mean()
    # over/under calibration at line = floor(mu)+0.5
    line = np.floor(mu) + 0.5
    p_over = 1 - poisson.cdf(np.floor(line), mu)
    actual_over = (te["K"].values > line).astype(int)
    cal_err = abs(p_over.mean() - actual_over.mean())
    return mae, cal_err


def main():
    df = pd.read_parquet(kprops.DATASET)
    df = prior_rates(df)

    # league fallback for opp_krate already in dataset; ensure present
    if "opp_krate" not in df or df["opp_krate"].isna().any():
        lg = df["K"].sum() / df["bf"].sum()
        df["opp_krate"] = df.get("opp_krate", lg)
        df["opp_krate"] = df["opp_krate"].fillna(lg)

    # usable: need >=3 prior starts so every variant is defined
    d = df[(df.n_prior >= 3) & df.kr_flat.notna() & df.kr_last5.notna()
           & df.pit_bf_exp.notna()].copy()
    dates = sorted(d["date"].unique())
    cut = dates[int(len(dates) * 0.70)]
    tr, te = d[d.date <= cut], d[d.date > cut]
    print(f"usable {len(d)} | train {len(tr)} test {len(te)} | cut {cut}\n")

    # naive baseline: predict each pitcher's flat prior mean K directly (no model)
    base = np.abs((te["kr_flat"] * te["pit_bf_exp"]).values - te["K"].values).mean()
    print(f"{'naive flat-avg (no model)':28s} MAE {base:.3f}")
    print("-" * 52)

    variants = ["kr_flat", "kr_last3", "kr_last5", "kr_ewma", "kr_blend"]
    results = []
    for v in variants:
        mae, cal = evaluate(d, v, tr, te)
        results.append((v, mae, cal))
    for v, mae, cal in sorted(results, key=lambda r: r[1]):
        star = "  <-- current model" if v == "kr_flat" else ""
        print(f"{v:28s} MAE {mae:.3f}   calib_err {cal:.3f}{star}")

    best = min(results, key=lambda r: r[1])
    flat_mae = [r[1] for r in results if r[0] == "kr_flat"][0]
    print(f"\nbest: {best[0]} (MAE {best[1]:.3f})  vs flat {flat_mae:.3f}  "
          f"-> delta {best[1]-flat_mae:+.4f}")
    if best[0] != "kr_flat" and (flat_mae - best[1]) > 0.01:
        print("=> recency helps meaningfully; worth adopting.")
    else:
        print("=> recency does NOT beat flat average by a meaningful margin.")


if __name__ == "__main__":
    main()
