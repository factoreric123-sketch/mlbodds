"""Find sportsbook prop edges from the per-book detail capture.

Two independent signals, both at the 2%+ threshold:
  1) LINE-SHOP: best available price on a side vs the de-vigged fair prob of that side.
     EV% = best_decimal * fair_prob - 1. This is pure market inefficiency (one book
     posting a price better than the consensus fair value). No model skill required.
  2) MODEL vs CONSENSUS: our Poisson model's P(over) vs the de-vigged consensus P(over)
     at the modal line. Flags where the model disagrees with the market. UNPROVEN edge
     (model has not beaten the market forward yet) -> report-only, do not bet blindly.

Usage: python edge_finder.py --date 2026-07-02 [--min-ev 0.02]
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

import kprops
import hits_model
import outs_model
import predict_day

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOKS = os.path.join(ROOT, "pp_logs", "books")


def _fair_from_median(sub: pd.DataFrame):
    """De-vig the median over/under decimal prices at the modal line -> fair P(over/under)."""
    modal = sub["line"].mode().iloc[0]
    at = sub[sub["line"] == modal]
    med_o = float(np.median(at["over_dec"]))
    med_u = float(np.median(at["under_dec"]))
    fair_over, fair_under = predict_day.devig(med_o, med_u)
    return modal, fair_over, fair_under, at


def line_shop(detail: pd.DataFrame, min_ev: float):
    """For each (stat, pitcher): best over-price and best under-price vs fair value."""
    out = []
    for (stat, name, disp), sub in detail.groupby(["stat", "name", "pitcher"]):
        modal, fair_o, fair_u, at = _fair_from_median(sub)
        if len(at) < 2:  # need >=2 books at the modal line to trust a consensus
            continue
        # best (highest) decimal price available on each side, and which book
        bo = at.loc[at["over_dec"].idxmax()]
        bu = at.loc[at["under_dec"].idxmax()]
        ev_over = bo["over_dec"] * fair_o - 1
        ev_under = bu["under_dec"] * fair_u - 1
        for side, ev, price, book, fair in [
            ("OVER", ev_over, bo["over_dec"], bo["book"], fair_o),
            ("UNDER", ev_under, bu["under_dec"], bu["book"], fair_u)]:
            if ev >= min_ev:
                out.append(dict(stat=stat, pitcher=disp, side=side, line=modal,
                                best_book=book, best_dec=round(price, 3),
                                fair_p=round(fair, 3), ev=round(ev, 4),
                                n_books=len(at)))
    return pd.DataFrame(out).sort_values("ev", ascending=False) if out else pd.DataFrame()


def _model_p_over(stat: str):
    """Return dict name-> (mu builder). Reuse the tracker's scoring per stat."""
    df = pd.read_parquet(kprops.DATASET)
    if stat == "ks":
        model = kprops.fit_model(df)
        team_kr = kprops.team_krate(df)
        lg = df["K"].sum() / df["bf"].sum()
        feats, rk = kprops.FEATS, "next_krate"
        last = {}
        for name, sub in df.sort_values(["pid", "date", "game_pk"]).groupby("name"):
            last[_norm(name)] = dict(rate=kprops.next_krate(sub["K"].to_numpy(float), sub["bf"].to_numpy(float)),
                                     bf=sub["bf"].mean(), n=len(sub))
        def mu(info):
            x = pd.DataFrame([[info["rate"], lg, info["bf"], 0]], columns=feats)
            return float(model.predict(x)[0])
        return last, mu
    if stat == "hits":
        df = hits_model.add_hits_features(df)
        model = hits_model.fit_hits(df)
        lg = df["hits"].sum() / df["bf"].sum()
        last = {}
        for name, sub in df.sort_values(["pid", "date", "game_pk"]).groupby("name"):
            last[_norm(name)] = dict(rate=hits_model.next_hrate(sub["hits"].to_numpy(float), sub["bf"].to_numpy(float)),
                                     bf=sub["bf"].mean(), n=len(sub))
        def mu(info):
            x = pd.DataFrame([[info["rate"], lg, info["bf"], 0]], columns=hits_model.FEATS)
            return float(model.predict(x)[0])
        return last, mu
    # outs
    df = outs_model.add_outs_features(df)
    model = outs_model.fit_outs(df)
    last = {}
    for name, sub in df.sort_values(["pid", "date", "game_pk"]).groupby("name"):
        last[_norm(name)] = dict(oe=outs_model.next_outs(sub["outs"].to_numpy(float)),
                                 bf=sub["bf"].mean(), n=len(sub))
    def mu(info):
        x = pd.DataFrame([[info["oe"], info["bf"], 0]], columns=outs_model.FEATS)
        return float(model.predict(x)[0])
    return last, mu


import unicodedata
def _norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower().strip()


def model_vs_consensus(consensus: pd.DataFrame, min_ev: float):
    from scipy.stats import poisson
    out = []
    for stat in ["ks", "hits", "outs"]:
        sub = consensus[consensus["stat"] == stat]
        if sub.empty:
            continue
        last, mu_of = _model_p_over(stat)
        for _, r in sub.iterrows():
            info = last.get(r["name"])
            if not info or pd.isna(info.get("bf")):
                continue
            mu = mu_of(info)
            p_over = float(1 - poisson.cdf(np.floor(r["book_line"]), mu))
            edge = p_over - r["book_p_over"]
            if abs(edge) >= min_ev:
                out.append(dict(stat=stat, pitcher=r["pitcher"], line=r["book_line"],
                                model_p_over=round(p_over, 3),
                                book_p_over=round(r["book_p_over"], 3),
                                edge=round(edge, 3),
                                lean="OVER" if edge > 0 else "UNDER"))
    return pd.DataFrame(out).sort_values("edge", key=abs, ascending=False) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--min-ev", type=float, default=0.02)
    args = ap.parse_args()

    dpath = os.path.join(BOOKS, f"books_detail_{args.date}.csv")
    cpath = os.path.join(BOOKS, f"books_{args.date}.csv")
    if not os.path.exists(dpath):
        print(f"no per-book detail file for {args.date} ({dpath})"); return
    detail = pd.read_csv(dpath)
    consensus = pd.read_csv(cpath)

    print(f"### EDGE FINDER {args.date}  (min EV {args.min_ev:.0%}) ###\n")

    print("=== 1) LINE-SHOP EDGES (best book price vs de-vigged fair value) ===")
    ls = line_shop(detail, args.min_ev)
    if ls.empty:
        print("  none >= threshold (market efficient / thin books)")
    else:
        print(ls.to_string(index=False))

    print("\n=== 2) MODEL vs CONSENSUS (unproven; report-only) ===")
    mc = model_vs_consensus(consensus, args.min_ev)
    if mc.empty:
        print("  no model disagreements >= threshold")
    else:
        print(mc.to_string(index=False))


if __name__ == "__main__":
    main()
