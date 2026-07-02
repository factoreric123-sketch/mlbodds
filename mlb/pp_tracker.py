"""
pp_tracker.py
-------------
PrizePicks data-collection tracker (strikeouts). We are NOT placing bets — we are
logging the model's pick for every in-universe leg each day, plus the Flex tickets
the model would build at 3/4/5/6 legs, then grading them against actual Ks.

Goal: measure whether the K model's leg-level P(side) and the resulting ticket EV
hold up out-of-sample. This is the PrizePicks analogue of the sportsbook CLV log.

Files (mlb/pp_logs/):
    legs_<DATE>.csv      one row per in-universe pitcher: line, side, model P(hit), result
    tickets_<DATE>.csv   the 3/4/5/6-leg Flex tickets built from the strongest legs

Flex payout tables (entry-multiplier on correct count), from PrizePicks:
    3: 3->3.0,  2->1.0
    4: 4->6.0,  3->1.5
    5: 5->10.0, 4->2.0,  3->0.4
    6: 6->28.0, 5->2.0,  4->0.4

Usage:
    python3 pp_tracker.py log    --date 2026-07-01   # build legs+tickets from SLATE
    python3 pp_tracker.py grade  [--date ...]         # grade vs actual Ks (all if no date)
    python3 pp_tracker.py report                      # cumulative leg + ticket performance
"""
from __future__ import annotations

import argparse
import glob
import os
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import poisson

import json
import urllib.request

import kprops
import grade_day  # reuse actual_ks()
import hits_model

ROOT = os.path.dirname(os.path.abspath(__file__))
PP_LOGS = os.path.join(ROOT, "pp_logs")
os.makedirs(PP_LOGS, exist_ok=True)


def actual_stat(date: str, stat: str) -> dict:
    """{norm_name: value} of a pitching boxscore stat for every starter on `date`.
    stat in {'strikeOuts','hits'} (StatsAPI keys). Final games only."""
    sched = grade_day.http(
        f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}")
    pks = [g["gamePk"] for d in sched.get("dates", []) for g in d["games"]
           if g.get("status", {}).get("abstractGameState") == "Final"]
    out = {}
    for pk in pks:
        box = grade_day.http(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore")
        for side in ("home", "away"):
            for pdata in box["teams"][side]["players"].values():
                st = pdata.get("stats", {}).get("pitching", {})
                if st.get("gamesStarted", 0) == 1:
                    out[norm(pdata["person"]["fullName"])] = st.get(stat, 0)
    return out

FLEX = {
    3: {3: 3.0, 2: 1.0},
    4: {4: 6.0, 3: 1.5},
    5: {5: 10.0, 4: 2.0, 3: 0.4},
    6: {6: 28.0, 5: 2.0, 4: 0.4},
}
STAKE = 20.0


def norm(n: str) -> str:
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(n.lower().split())


def p_over(mu: float, line: float) -> float:
    return float(1 - poisson.cdf(np.floor(line), mu))


# ---- edit this each day with the PrizePicks board (name, opp_team, line, is_home) ----
# 2026-07-02 board. Goblin/Demon lines excluded from standard scoring (different payout structure).
SLATE = [
    ("Walker Buehler", "Chicago Cubs", 5.5, False),
    ("Freddy Peralta", "Toronto Blue Jays", 4.5, False),
    ("Zack Wheeler", "Pittsburgh Pirates", 7.0, True),
    ("Reynaldo López", "St. Louis Cardinals", 3.0, True),
    ("Michael McGreevy", "Atlanta Braves", 3.0, False),
    ("Seth Lugo", "Tampa Bay Rays", 4.0, True),
    ("Shane McClanahan", "Kansas City Royals", 4.5, False),
    ("Tatsuya Imai", "Minnesota Twins", 6.0, True),
    ("Shane Drohan", "Cincinnati Reds", 5.0, True),
    ("Taj Bradley", "Houston Astros", 6.0, False),
    ("Kyle Freeland", "Miami Marlins", 4.0, True),
    ("Max Meyer", "Colorado Rockies", 6.0, False),
    ("Zac Gallen", "San Francisco Giants", 3.5, True),
    ("Trevor McDonald", "Arizona Diamondbacks", 3.5, False),
    ("J.T. Ginn", "Los Angeles Dodgers", 4.5, True),
]


# today's PrizePicks Hits-Allowed board (name, opp_team, line, is_home)
SLATE_HITS = [
    ("Trey Gibson", "Chicago White Sox", 4.5, True),
    ("Erick Fedde", "Baltimore Orioles", 4.5, False),
    ("Tanner Bibee", "Texas Rangers", 4.5, True),
    ("Jacob deGrom", "Cleveland Guardians", 3.5, False),
    ("Cristopher Sánchez", "Pittsburgh Pirates", 5.5, True),
    ("Bubba Chandler", "Philadelphia Phillies", 4.5, False),
    ("Cam Schlittler", "Detroit Tigers", 4.5, True),
    ("Tarik Skubal", "New York Yankees", 4.5, False),
    ("Kevin Gausman", "New York Mets", 4.5, True),
    ("Nolan McLean", "Toronto Blue Jays", 4.5, False),
    ("Cade Cavalli", "Boston Red Sox", 5.5, False),
    ("Connelly Early", "Washington Nationals", 4.5, True),
    ("Matthew Liberatore", "Atlanta Braves", 5.5, False),
    ("Martín Pérez", "St. Louis Cardinals", 4.5, True),
    ("Rhett Lowder", "Milwaukee Brewers", 4.5, False),
    ("Brandon Sproat", "Cincinnati Reds", 3.5, True),
    ("Noah Cameron", "Tampa Bay Rays", 5.5, True),
    ("Griffin Jax", "Kansas City Royals", 4.5, False),
    ("JP Sears", "Chicago Cubs", 5.5, False),
    ("Matthew Boyd", "San Diego Padres", 5.5, True),
    ("Mike Burrows", "Minnesota Twins", 4.5, True),
    ("Joe Ryan", "Houston Astros", 4.5, False),
    ("Tanner Gordon", "Miami Marlins", 4.5, True),
    ("Eury Pérez", "Colorado Rockies", 4.5, False),
    ("Bryan Woo", "Los Angeles Angels", 4.5, True),
    ("José Soriano", "Seattle Mariners", 4.5, False),
    ("Brandon Pfaadt", "San Francisco Giants", 4.5, True),
    ("Landen Roupp", "Arizona Diamondbacks", 4.5, False),
    ("Justin Wrobleski", "Athletics", 5.5, False),
    ("Jeffrey Springs", "Los Angeles Dodgers", 5.5, True),
]


def _score_ks(df):
    """Return name->P(over) callable bits for the strikeout model."""
    model = kprops.fit_model(df)
    team_kr = kprops.team_krate(df)
    lg = df["K"].sum() / df["bf"].sum()
    last = {}
    for name, sub in df.sort_values(["pid", "date", "game_pk"]).groupby("name"):
        last[norm(name)] = dict(
            rate=kprops.next_krate(sub["K"].to_numpy(float), sub["bf"].to_numpy(float)),
            bf_exp=sub["bf"].mean(), n=len(sub))

    def mu_of(info, opp, home):
        x = pd.DataFrame([[info["rate"], team_kr.get(opp, lg), info["bf_exp"], int(home)]],
                         columns=kprops.FEATS)
        return float(model.predict(x)[0])
    return last, mu_of


def _score_hits(df):
    """Return name->P(over) callable bits for the hits-allowed model."""
    df = hits_model.add_hits_features(df)
    model = hits_model.fit_hits(df)
    g = df.groupby("opp")
    opp_hr = (g["hits"].sum() / g["bf"].sum()).to_dict()
    lg = df["hits"].sum() / df["bf"].sum()
    last = {}
    for name, sub in df.sort_values(["pid", "date", "game_pk"]).groupby("name"):
        last[norm(name)] = dict(
            rate=hits_model.next_hrate(sub["hits"].to_numpy(float), sub["bf"].to_numpy(float)),
            bf_exp=sub["bf"].mean(), n=len(sub))

    def mu_of(info, opp, home):
        x = pd.DataFrame([[info["rate"], opp_hr.get(opp, lg), info["bf_exp"], int(home)]],
                         columns=hits_model.FEATS)
        return float(model.predict(x)[0])
    return last, mu_of


STAT_CONFIG = {
    "ks":   dict(slate=lambda: SLATE,       score=_score_ks,   api_key="strikeOuts"),
    "hits": dict(slate=lambda: SLATE_HITS,  score=_score_hits, api_key="hits"),
}


def _logdir(stat):
    d = os.path.join(PP_LOGS, stat)
    os.makedirs(d, exist_ok=True)
    return d


# --- sportsbook odds capture (must run BEFORE first pitch: free tier has no history) ---
import predict_day  # reuse list_events, devig, http, norm_name

ODDS_KEY = predict_day.ODDS_KEY
SPORT = predict_day.SPORT
BOOK_MARKETS = {"ks": "pitcher_strikeouts", "hits": "pitcher_hits_allowed",
                "outs": "pitcher_outs"}


def _amer_to_dec(price: float) -> float:
    p = float(price)
    return 1 + (p / 100 if p > 0 else 100 / -p)


def _event_props_multi(event_id: str):
    markets = ",".join(BOOK_MARKETS.values())
    url = (f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds"
           f"?apiKey={ODDS_KEY}&regions=us&markets={markets}&oddsFormat=american")
    data, hdr = predict_day.http(url)
    return data, hdr.get("x-requests-remaining")


def capture_books(date: str):
    """Snapshot two-sided book props for all 3 markets, de-vig to a book consensus
    P(over), and save one row per (stat, pitcher). Must run before games start; the
    free odds tier returns current lines only (no closing/historical endpoint)."""
    rev_market = {v: k for k, v in BOOK_MARKETS.items()}
    events = predict_day.list_events()
    rows = []
    detail = []  # one row per (stat, pitcher, book): raw prices for line-shopping / CLV
    remaining = None
    for e in events:
        try:
            data, remaining = _event_props_multi(e["id"])
        except Exception:
            continue
        # collect all (line, over, under) quotes per (stat, pitcher) across books
        agg = {}
        for bm in data.get("bookmakers", []):
            book = bm.get("key", "?")
            for m in bm.get("markets", []):
                stat = rev_market.get(m["key"])
                if not stat:
                    continue
                for pitcher in {o.get("description") for o in m["outcomes"]}:
                    q = predict_day.two_way(m, pitcher)
                    if q is None:
                        continue
                    line, over, under = q
                    agg.setdefault((stat, norm(pitcher), pitcher), []).append(
                        (line, _amer_to_dec(over), _amer_to_dec(under)))
                    detail.append(dict(date=date, stat=stat, pitcher=pitcher,
                                       name=norm(pitcher), book=book, line=line,
                                       over_dec=round(_amer_to_dec(over), 4),
                                       under_dec=round(_amer_to_dec(under), 4)))
        for (stat, key, disp), quotes in agg.items():
            # consensus at the modal line, median price, de-vigged
            from collections import Counter
            modal_line = Counter(l for l, _, _ in quotes).most_common(1)[0][0]
            at = [(o, u) for l, o, u in quotes if l == modal_line]
            med_o = float(np.median([o for o, _ in at]))
            med_u = float(np.median([u for _, u in at]))
            fair_over, _ = predict_day.devig(med_o, med_u)
            rows.append(dict(date=date, stat=stat, pitcher=disp, name=key,
                             book_line=modal_line, book_p_over=round(fair_over, 4),
                             n_books=len(at)))
    if not rows:
        print(f"[books] {date}: no live book props (games may have started / not posted)")
        return
    df = pd.DataFrame(rows).sort_values(["stat", "book_p_over"], ascending=[True, False])
    bdir = os.path.join(PP_LOGS, "books")
    os.makedirs(bdir, exist_ok=True)
    path = os.path.join(bdir, f"books_{date}.csv")
    df.to_csv(path, index=False)
    if detail:
        ddf = pd.DataFrame(detail).sort_values(["stat", "name", "book"])
        dpath = os.path.join(bdir, f"books_detail_{date}.csv")
        ddf.to_csv(dpath, index=False)
    by = df.groupby("stat").size().to_dict()
    print(f"[books] {date}: captured {len(df)} lines {by} -> {os.path.relpath(path, ROOT)}"
          f"  ({len(detail)} per-book quotes)  (odds-api remaining: {remaining})")


def log_day(date: str, stat: str = "ks"):
    cfg = STAT_CONFIG[stat]
    df = pd.read_parquet(kprops.DATASET)
    last, mu_of = cfg["score"](df)
    slate = cfg["slate"]()
    logdir = _logdir(stat)

    legs = []
    for name, opp, line, home in slate:
        k = norm(name)
        if k not in last:
            continue
        info = last[k]
        mu = mu_of(info, opp, home)
        po = p_over(mu, line)
        side = "MORE" if po >= 0.5 else "LESS"
        p_hit = po if side == "MORE" else 1 - po
        legs.append(dict(date=date, stat=stat, pitcher=name, opp=opp, line=line, home=int(home),
                         mu=round(mu, 2), side=side, p_hit=round(p_hit, 3),
                         conf=round(abs(po - 0.5), 3), n_prior=info["n"],
                         actual="", leg_result=""))
    if not legs:
        print(f"[{stat}] no in-universe legs on the slate (is SLATE_{stat.upper()} populated?)")
        return
    legs_df = pd.DataFrame(legs).sort_values("conf", ascending=False).reset_index(drop=True)
    legs_path = os.path.join(logdir, f"legs_{date}.csv")
    legs_df.to_csv(legs_path, index=False)

    # build n-leg Flex tickets from the strongest legs (highest p_hit)
    ranked = legs_df.sort_values("p_hit", ascending=False).reset_index(drop=True)
    tickets = []
    for n in (3, 4, 5, 6):
        if n > len(ranked):
            continue  # not enough in-universe legs for this ticket size
        pick = ranked.head(n)
        probs = pick["p_hit"].to_numpy()
        ev = flex_ev(probs, FLEX[n])
        tickets.append(dict(
            date=date, n_legs=n, type="Flex", stake=STAKE,
            legs="; ".join(f"{r.pitcher} {r.side} {r.line}" for r in pick.itertuples()),
            avg_p=round(probs.mean(), 3), ev_per_1=round(ev, 3),
            exp_return=round(ev * STAKE, 2),
            n_correct="", payout_mult="", payout="", ticket_result=""))
    tickets_df = pd.DataFrame(tickets)
    tickets_path = os.path.join(logdir, f"tickets_{date}.csv")
    tickets_df.to_csv(tickets_path, index=False)

    print(f"[{stat}] logged {len(legs_df)} in-universe legs -> {stat}/{os.path.basename(legs_path)}")
    print(legs_df[["pitcher", "line", "side", "mu", "p_hit", "conf"]].head(10).to_string(index=False))
    print(f"\n{len(tickets_df)} tickets -> {os.path.basename(tickets_path)}")
    print(tickets_df[["n_legs", "avg_p", "ev_per_1", "exp_return", "legs"]].to_string(index=False))


def flex_ev(probs, payout):
    """EV per $1 for a Flex ticket. probs = per-leg hit probs; payout = {k_correct: mult}."""
    dist = [1.0]
    for p in probs:
        new = [0.0] * (len(dist) + 1)
        for k, pr in enumerate(dist):
            new[k] += pr * (1 - p)
            new[k + 1] += pr * p
        dist = new
    return sum(dist[k] * m for k, m in payout.items())


def grade_day_pp(date: str, stat: str = "ks"):
    cfg = STAT_CONFIG[stat]
    logdir = _logdir(stat)
    legs_path = os.path.join(logdir, f"legs_{date}.csv")
    tic_path = os.path.join(logdir, f"tickets_{date}.csv")
    if not os.path.exists(legs_path):
        print("missing", legs_path); return
    ks = actual_stat(date, cfg["api_key"])
    if not ks:
        print(f"[{stat}] {date}: no final games yet, skipping"); return

    legs = pd.read_csv(legs_path)

    def settle(row):
        k = ks.get(norm(row["pitcher"]))
        if k is None:
            return pd.Series(["", ""])
        if k == row["line"]:
            res = "push"
        elif (row["side"] == "MORE") == (k > row["line"]):
            res = "win"
        else:
            res = "loss"
        return pd.Series([k, res])

    legs[["actual", "leg_result"]] = legs.apply(settle, axis=1)
    legs.to_csv(legs_path, index=False)

    # grade tickets: count wins among that ticket's legs (push counts as win on PP)
    res_by_pitcher = {norm(r.pitcher): (r.side, r.line) for r in legs.itertuples()}
    tickets = pd.read_csv(tic_path)

    def grade_ticket(row):
        names = [s.rsplit(" ", 2)[0] for s in row["legs"].split("; ")]
        correct = 0
        graded = 0
        for nm in names:
            k = ks.get(norm(nm))
            if k is None:
                continue
            side, line = res_by_pitcher[norm(nm)]
            graded += 1
            if k == line:
                correct += 1  # PrizePicks: push = win on that leg
            elif (side == "MORE") == (k > line):
                correct += 1
        if graded < row["n_legs"]:
            return pd.Series(["", "", "", "pending"])  # not all legs final
        mult = FLEX[row["n_legs"]].get(correct, 0.0)
        payout = mult * row["stake"]
        result = "win" if payout > row["stake"] else ("refund" if payout == row["stake"] else "loss")
        return pd.Series([correct, mult, round(payout, 2), result])

    tickets[["n_correct", "payout_mult", "payout", "ticket_result"]] = tickets.apply(grade_ticket, axis=1)
    tickets.to_csv(tic_path, index=False)

    done = (legs["leg_result"] != "").sum()
    print(f"[{stat}] {date}: graded {done}/{len(legs)} legs")
    print(legs[["pitcher", "line", "side", "p_hit", "actual", "leg_result"]].to_string(index=False))
    print()
    print(tickets[["n_legs", "n_correct", "payout_mult", "payout", "ticket_result"]].to_string(index=False))


def report(stat: str = "ks"):
    logdir = _logdir(stat)
    print(f"################  STAT: {stat.upper()}  ################")
    leg_files = sorted(glob.glob(os.path.join(logdir, "legs_*.csv")))
    tic_files = sorted(glob.glob(os.path.join(logdir, "tickets_*.csv")))
    if not leg_files:
        print("no logs yet."); return

    legs = pd.concat([pd.read_csv(f) for f in leg_files], ignore_index=True)
    g = legs[legs["leg_result"].isin(["win", "loss", "push"])].copy()
    print("=== LEG-LEVEL (model pick accuracy) ===")
    if g.empty:
        print("  no graded legs yet.")
    else:
        w = (g.leg_result == "win").sum(); l = (g.leg_result == "loss").sum()
        p = (g.leg_result == "push").sum()
        wr = w / (w + l) if (w + l) else 0
        print(f"  legs graded: {len(g)}  | win {w} / loss {l} / push {p}  | hit rate {wr:.1%}")
        # calibration: model said p_hit, did it hit?
        g2 = g[g.leg_result != "push"]
        print(f"  model mean p_hit on picks: {g2.p_hit.mean():.3f}  vs actual hit rate {wr:.3f}")
        for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]:
            bucket = g2[(g2.p_hit >= lo) & (g2.p_hit < hi)]
            if len(bucket):
                hr = (bucket.leg_result == "win").mean()
                print(f"    p_hit [{lo:.2f},{hi:.2f}): n={len(bucket):3d}  predicted~{bucket.p_hit.mean():.3f}  actual {hr:.3f}")

    print("\n=== TICKET-LEVEL (Flex P&L, $20 each) ===")
    if not tic_files:
        print("  none."); return
    tic = pd.concat([pd.read_csv(f) for f in tic_files], ignore_index=True)
    done = tic[tic.ticket_result.isin(["win", "loss", "refund"])].copy()
    if done.empty:
        print("  no graded tickets yet."); return
    for n in (3, 4, 5, 6):
        sub = done[done.n_legs == n]
        if sub.empty:
            continue
        staked = sub.stake.sum()
        ret = sub.payout.sum()
        roi = (ret - staked) / staked
        wins = (sub.ticket_result == "win").sum()
        print(f"  {n}-leg Flex: {len(sub):3d} tickets | staked ${staked:.0f} | returned ${ret:.0f} "
              f"| ROI {roi:+.1%} | cashed {wins}/{len(sub)}")
    staked = done.stake.sum(); ret = done.payout.sum()
    print(f"  TOTAL: staked ${staked:.0f} | returned ${ret:.0f} | ROI {(ret-staked)/staked:+.1%}")

    _book_vs_model(stat, legs)
    if stat == "ks":
        _h1_tracker(legs)


def _h1_tracker(legs: pd.DataFrame):
    """H1 (see NOTES.md): does the K model over-bet LESS and lose, and does its K line
    sit below the book's? Only meaningful for the K stat."""
    g = legs[legs.leg_result.isin(["win", "loss"])].copy()
    if g.empty:
        return
    print("\n=== H1: K-model LESS bias (see NOTES.md) ===")
    for date in sorted(g.date.unique()):
        d = g[g.date == date]
        less = d[d.side == "LESS"]
        if less.empty:
            row = f"  {date}: no LESS picks"
        else:
            wr = (less.leg_result == "win").mean()
            pred = less.p_hit.mean()
            row = (f"  {date}: LESS {len(less)}/{len(d)} legs | win {wr:.0%} "
                   f"| predicted {pred:.0%} | gap {wr-pred:+.0%}")
        print(row)
    # cumulative
    less_all = g[g.side == "LESS"]
    if len(less_all):
        wr = (less_all.leg_result == "win").mean()
        print(f"  CUMULATIVE LESS: n={len(less_all)} | win {wr:.1%} "
              f"| predicted {less_all.p_hit.mean():.1%} "
              f"| verdict: {'LESS BIAS LOSING (H1 support)' if wr < less_all.p_hit.mean()-0.05 else 'calibrated / inconclusive'}")

    # model-vs-book K line direction (does model project fewer Ks than the book?)
    bfiles = sorted(glob.glob(os.path.join(PP_LOGS, "books", "books_*.csv")))
    if not bfiles or "mu" not in legs.columns:
        return
    books = pd.concat([pd.read_csv(f) for f in bfiles], ignore_index=True)
    books = books[books.stat == "ks"]
    if books.empty:
        return
    m = legs.dropna(subset=["mu"]).copy()
    m["name"] = m.pitcher.map(norm)
    j = m.merge(books[["date", "name", "book_line"]], on=["date", "name"])
    if j.empty:
        print("  model-vs-book K line: no overlap yet.")
        return
    below = (j.mu < j.book_line).mean()
    print(f"  model K μ vs book line: n={len(j)} | model BELOW book {below:.0%} "
          f"| mean(μ - book_line) = {(j.mu - j.book_line).mean():+.2f} "
          f"(negative = model projects fewer Ks → H1 support)")


def _book_vs_model(stat: str, legs: pd.DataFrame):
    """Where we captured a book line for the same pitcher/date, compare the model's
    P(over) vs the de-vigged book P(over), and (for graded legs) who called it."""
    bfiles = sorted(glob.glob(os.path.join(PP_LOGS, "books", "books_*.csv")))
    if not bfiles:
        return
    books = pd.concat([pd.read_csv(f) for f in bfiles], ignore_index=True)
    books = books[books.stat == stat]
    if books.empty or "mu" not in legs.columns:
        return
    # model P(over) from stored mu at the BOOK's line (comparable to book_p_over)
    m = legs.dropna(subset=["mu"]).copy()
    m["name"] = m.pitcher.map(norm)
    j = m.merge(books[["date", "name", "book_line", "book_p_over"]],
                on=["date", "name"])
    if j.empty:
        print("\n=== MODEL vs BOOK ===\n  no overlapping pitchers captured yet.")
        return
    po = 1 - poisson.cdf(np.floor(j.book_line), j.mu)
    j["model_p_over"] = po
    j["edge"] = j.model_p_over - j.book_p_over
    print("\n=== MODEL vs BOOK (de-vigged) ===")
    print(f"  overlap: {len(j)} pitcher-days | mean |model-book| = {j.edge.abs().mean():.3f}")
    graded = j[j.leg_result.isin(["win", "loss"])]
    if len(graded):
        # on legs where model & book disagreed on side (>0.5 vs <0.5), who was right?
        dis = graded[(graded.model_p_over - 0.5) * (graded.book_p_over - 0.5) < 0]
        if len(dis):
            model_right = (dis.leg_result == "win").mean()
            print(f"  disagreements (opposite sides): n={len(dis)} | model correct {model_right:.1%}")
    show = j.reindex(j.edge.abs().sort_values(ascending=False).index)
    cols = ["date", "pitcher", "book_line", "model_p_over", "book_p_over", "edge"]
    print(show[cols].head(8).to_string(index=False,
          formatters={"model_p_over": "{:.3f}".format, "book_p_over": "{:.3f}".format,
                      "edge": "{:+.3f}".format}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["log", "grade", "report", "books"])
    ap.add_argument("--date")
    ap.add_argument("--stat", choices=["ks", "hits", "all"], default="ks")
    ap.add_argument("--no-books", action="store_true",
                    help="skip the sportsbook odds capture during 'log'")
    args = ap.parse_args()
    stats = ["ks", "hits"] if args.stat == "all" else [args.stat]

    if args.cmd == "books":
        import datetime as dt
        capture_books(args.date or dt.date.today().isoformat())
        return

    if args.cmd == "log":
        import datetime as dt
        d = args.date or dt.date.today().isoformat()
        for s in stats:
            log_day(d, s)
        if not args.no_books:
            capture_books(d)  # book lines vanish after first pitch — grab them now
    elif args.cmd == "grade":
        for s in stats:
            if args.date:
                grade_day_pp(args.date, s)
            else:
                for f in sorted(glob.glob(os.path.join(_logdir(s), "legs_*.csv"))):
                    grade_day_pp(os.path.basename(f)[5:-4], s)
    else:
        for s in stats:
            report(s)


if __name__ == "__main__":
    main()
