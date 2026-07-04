"""Consensus-vs-PrizePicks edge finder.

THESIS: the de-vigged sharp-book consensus is TRUTH. PrizePicks is a softer market.
Where PrizePicks' line/side deviates from the sharp consensus, that gap is the edge --
independent of our own model (which loses to the consensus). We invert the sharp
consensus's (line, P_over) to a fair implied mean (mu), then evaluate the PrizePicks
line against that mu to get a model-free P(hit) for the PP side. If that beats the PP
break-even (payout-adjusted), it's +EV.

Usage:
  python pp_vs_consensus.py --date 2026-07-02 --board board.txt
where board.txt has one PP leg per line:  "Pitcher Name | stat | line | side"
  stat in {ks,hits,outs};  side in {MORE,LESS}
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import brentq
import unicodedata

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOKS = os.path.join(ROOT, "pp_logs", "books")

# PrizePicks break-even by product (P(hit) needed on a single pick-equivalent).
# Standard 2-way is ~0.50 fair; the real hurdle is the payout structure of the entry.
# We report raw P(hit) vs a reference 0.50 and vs common Goblin/Demon thresholds.
PRODUCT_BE = {"std": 0.50, "goblin": 0.60, "demon": 0.65}  # rough single-leg break-evens

# Confidence tiers for a REAL-market edge (not threes). We blend two signals:
#   line-gap = |sharp_line - pp_line|  -> model-FREE soft-line discount (strongest)
#   edge     = P(hit) - break-even     -> Poisson-derived (secondary, shakier)
# great: a big gap OR a fat edge; good: solid edge; okay: marginal (>= flag threshold).
GREAT_GAP = 2.0
GREAT_EDGE = 0.12
GOOD_EDGE = 0.07


def edge_tier(gap: float, edge: float) -> str:
    if gap >= GREAT_GAP or edge >= GREAT_EDGE:
        return "great"
    if edge >= GOOD_EDGE:
        return "good"
    return "okay"


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower().strip()


def implied_mu(line: float, p_over: float) -> float:
    """Invert Poisson: find mu such that P(X > line) == p_over."""
    p_over = min(max(p_over, 1e-4), 1 - 1e-4)
    k = np.floor(line)
    f = lambda mu: (1 - poisson.cdf(k, mu)) - p_over
    # mu bounds: strikeouts/outs stay within [0.1, 40]
    lo, hi = 0.1, 40.0
    if f(lo) > 0:  # even tiny mu already over -> clamp
        return lo
    if f(hi) < 0:
        return hi
    return brentq(f, lo, hi)


def p_hit_at(mu: float, pp_line: float, side: str) -> float:
    p_over = float(1 - poisson.cdf(np.floor(pp_line), mu))
    return p_over if side == "MORE" else 1 - p_over


def load_board(path: str):
    legs = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) < 4:
                continue
            name, stat, line, side = parts[0], parts[1].lower(), float(parts[2]), parts[3].upper()
            prod = parts[4].lower() if len(parts) > 4 else "std"
            legs.append(dict(pitcher=name, name=norm(name), stat=stat, pp_line=line, side=side, product=prod))
    return legs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--board", required=True, help="text file of PP legs: name|stat|line|side[|product]")
    ap.add_argument("--min-edge", type=float, default=0.03, help="min P(hit)-over-breakeven to flag")
    ap.add_argument("--books", default="books", help="consensus file prefix (e.g. wnba_books)")
    ap.add_argument("--log", default=None, help="append EDGE rows to this CSV under pp_logs/ (e.g. wnba_edges.csv)")
    ap.add_argument("--tag", default="core", help="pass 'lowconf' for threes; else tiers are auto-assigned great/good/okay")
    args = ap.parse_args()

    cons = pd.read_csv(os.path.join(BOOKS, f"{args.books}_{args.date}.csv"))
    cons["key"] = cons["name"] + "|" + cons["stat"]
    cons = cons.set_index("key")

    legs = load_board(args.board)
    out = []
    for lg in legs:
        key = lg["name"] + "|" + lg["stat"]
        if key not in cons.index:
            out.append({**lg, "verdict": "NO CONSENSUS (rookie/unlisted)"})
            continue
        row = cons.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        mu = implied_mu(row["book_line"], row["book_p_over"])
        ph = p_hit_at(mu, lg["pp_line"], lg["side"])
        be = PRODUCT_BE.get(lg["product"], 0.50)
        edge = ph - be
        out.append({**lg, "sharp_line": row["book_line"], "sharp_mu": round(mu, 2),
                    "pp_p_hit": round(ph, 3), "breakeven": be, "edge": round(edge, 3),
                    "n_books": int(row["n_books"]),
                    "verdict": "EDGE" if edge >= args.min_edge else "pass"})

    df = pd.DataFrame(out)
    print(f"### CONSENSUS vs PRIZEPICKS  {args.date}  (sharp consensus = truth) ###\n")
    cols = ["pitcher", "stat", "pp_line", "side", "product", "sharp_line", "sharp_mu",
            "pp_p_hit", "breakeven", "edge", "n_books", "verdict"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].sort_values("edge", ascending=False, na_position="last").to_string(index=False))

    edges = df[df.get("verdict") == "EDGE"]
    print(f"\n>>> {len(edges)} EDGE(s) where PrizePicks deviates from sharp consensus by >= {args.min_edge:.0%}")
    if len(edges):
        print("    (these are model-FREE: grounded in the sharp market, not our model)")

    if args.log and len(edges):
        edges = edges.copy()
        edges.insert(0, "date", args.date)
        # threes (--tag lowconf) stay in their own bucket -- Poisson is unreliable on
        # them, so they're excluded from the great/good/okay confidence tiers.
        if args.tag == "lowconf":
            edges["tier"] = "threes"
        else:
            gap = (edges["sharp_line"] - edges["pp_line"]).abs()
            edges["tier"] = [edge_tier(g, e) for g, e in zip(gap, edges["edge"])]
        path = os.path.join(BOOKS, "..", args.log)
        path = os.path.normpath(path)
        header = not os.path.exists(path)
        edges[["date", "pitcher", "stat", "pp_line", "side", "product",
               "sharp_line", "sharp_mu", "pp_p_hit", "breakeven", "edge",
               "n_books", "tier"]].to_csv(path, mode="a", header=header, index=False)
        by = edges["tier"].value_counts().to_dict()
        print(f"    logged {len(edges)} edge(s) {by} -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
