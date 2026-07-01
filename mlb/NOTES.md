# MLB Prop Model — Hypothesis Log

Running log of testable hypotheses about the models. Each gets tracked day-by-day
(right/wrong) until we have enough data to act on it. Data is collected via
`pp_tracker.py` (legs + tickets + book lines). Do NOT act on any hypothesis until
it holds over multiple days — one slate is noise.

---

## H1 — K model over-bets LESS and loses (possible volume/innings underestimate)

**Status:** OPEN — collecting. Need 3-4 more days to confirm or kill.

**Claim:** The strikeout model systematically leans LESS (under) and those LESS
picks lose more than the model's confidence implies. Suspected root cause: the model
under-estimates pitcher *volume* (innings / batters faced), so it projects too few
Ks → leans LESS → gets punished when starters go deep.

**Why this is plausible (not proven):** On 2026-06-30 the same pitchers who blew past
their K-UNDER also cleared their HITS-OVER (deGrom, Soriano, Gausman all threw a lot).
More innings = more Ks AND more hits. One volume error would explain both:
- K model leaned LESS and *lost* (predicted ~0.68, actual 0.46 — overconfident, too big to be pure variance at n=13).
- Hits model leaned MORE and *won* (predicted 0.59, actual 0.62 — got "saved" by the same deep outings).

**How we test it (daily):**
1. Each graded day, record: K legs that were LESS, and their win rate.
2. Compare model's predicted K hit-rate vs actual, specifically on LESS picks.
3. Cross-check: on days the K-LESS bias loses, did the same pitchers clear hits-OVER? (volume signature)
4. Once book capture overlaps, check: is the model's K line consistently BELOW the de-vigged book line? (would confirm model projects too few Ks)

**Kill / confirm criteria:**
- CONFIRM if over ~4-5 graded days: K-LESS picks hit clearly below their predicted rate
  AND the model's K line runs below the book's K line. → then model the volume/innings term.
- KILL if the LESS bias calibrates out (predicted ≈ actual) as sample grows. → 06-30 was variance.

### Daily tracking
| Date | K legs | K LESS n | K LESS win% | Pred p_hit | Actual | K vs book (model lower?) | Notes |
|------|--------|----------|-------------|-----------|--------|--------------------------|-------|
| 2026-06-30 | 13 | 11 | see below | 0.68 | 0.46 | n/a (no book capture) | 5 LESS torched by deep outings; same pitchers cleared hits-OVER |

*(2026-06-30 K-LESS detail: Woo/Ryan/Schlittler/Bibee/Roupp LESS won; Sánchez/Wrobleski/deGrom/Gausman/Soriano LESS lost. deGrom/Soriano/Gausman also cleared hits-OVER same day = volume signature.)*

### Evidence — deGrom case (2026-06-30)
Cleanest example of the LESS bias: bet **deGrom LESS 6.5**, but his last-5 K = [8,6,5,9,8]
(avg **7.2**) — he'd cleared 6.5 in 4 of his last 5. Model projected him *under his own
recent floor*. He threw 9. Not variance — a projection that under-weighted a genuine
high-K arm. Same shape on Sánchez (L5 6.4, bet LESS 7.5) and Gausman (L5 5.4, recent 8,5,7).

### Lookback-window sweep (backtest: trailing-N-start mean predicts next start)
Tested N = 4,5,6,7,8,9,10 on all pitcher-starts with >=N priors.
| target | lowest MAE | highest corr | sweet spot |
|--------|-----------|--------------|------------|
| K      | N=8 (1.854) | N=10 (0.352) | ~8-10 |
| hits   | N=9 (1.769) | N=10 (0.119) | ~9-10 |
| outs   | N=8 (2.636) | N=8 (0.145)  | ~8 |

**Key finding (refines H1):** longer windows beat short ones for ALL stats. The current
model's recency blend leans on ~last-3 (EWMA half-life=3), which sits at the NOISY end.
The fix is likely a **longer/steadier window (~8)**, NOT "trust recent more." deGrom's
last-8 would still read high (he's genuinely 7 K) but a short window overreacts to one dud.

**Caveats:** correlations are LOW (K 0.35, hits 0.12, outs 0.15) — trailing avg alone is
a weak predictor, esp. hits (opponent dominates). Bias is small & negative (-0.05..-0.25),
mildly consistent with "projects too few." Long-N shrinks sample (399 vs 777 obs) →
partly survivorship. DO NOT change the model off this alone — validate against held-out
log-loss + the daily tracker before touching the window.

**RESULT — WINDOW FIX KILLED (2026-06-30 backtest).** Tested the real blend
(w_ewma·EWMA(half_life) + (1-w)·flat) through the full GLM on held-out o/u log-loss:
| half-life, w_ewma | log-loss |
|-------------------|----------|
| **hl=3, w=0.8 (CURRENT)** | **0.6655** ← best |
| hl=4, w=0.8 | 0.6736 |
| hl=6, w=0.8 | 0.6799 |
| hl=8, w=0.8 | 0.6806 |
| hl=100 (≈flat) | 0.6814 |

**The current SHORT window is optimal — every longer window is monotonically WORSE.**
Why the sweep misled us: the sweep tested trailing-mean → raw-K MAE (a standalone naive
predictor). The model uses pit_krate as a *rate* inside a GLM alongside opponent + volume
terms; in that full system, shorter recency wins because the other features supply stability.
deGrom's 9-on-LESS-6.5 was variance: a model leaning harder on his recent form scores WORSE
overall. The K model (0.8·EWMA hl=3 + 0.2·flat, Poisson) is already well-tuned; leave it.

**H1 net:** the LESS-bias question (does K-LESS keep losing forward) is STILL open and needs
daily data — but the two *proposed fixes* (longer window, wider distribution) are both dead.
If the LESS bias is real going forward, the cause is NOT the recency window or the
distribution — look elsewhere (opponent K-rate staleness? volume/bf term? line vs book).

---

## H2 — Erratic pitchers: model is OVERCONFIDENT, not inaccurate (Poisson too narrow)

**Status:** OPEN — promising, testable. Refines the Wrobleski counter-example to H1.

**Origin:** Wrobleski bet LESS 5.0, threw 11. His K series [6,0,7,5,4,9,4,1,5,3] is wildly
erratic (avg 4.4 but huge spread). Idea: avoid / down-weight unpredictable pitchers.

**Backtest (trailing-6 K-std vs prediction error, N>=6 priors):**
| vol bucket | trailing std | MAE | RMSE |
|-----------|--------------|-----|------|
| Q1 stable  | 1.15 | 1.856 | 2.239 |
| Q2         | 1.61 | 1.891 | 2.507 |
| Q3         | 2.05 | 1.973 | 2.600 |
| Q4 erratic | 2.72 | 1.834 | 2.259 |

corr(volatility, |error|) = **0.008** (≈zero).

**Finding (corrects the naive intuition):** erratic pitchers are NOT harder to predict on
average — MAE is flat across buckets. BUT their outcome distribution is wider than Poisson
assumes (Poisson variance = mean ≈ 4.4, but Wrobleski's real variance ≈ 7). So the model's
**P(over) is falsely confident** for volatile arms — it reports 0.68 when the true spread
makes it closer to a coin flip. The error is in the *confidence*, not the *point estimate*.

**Two candidate fixes to test (validate on held-out log-loss first):**
1. **Confidence filter:** skip/down-weight bets where a pitcher's trailing K-std is high
   (don't change the μ, just don't bet the fake-confident ones).
2. **Wider distribution:** Negative Binomial instead of Poisson for K (and check outs, which
   is UNDER-dispersed so it needs the opposite) so P(over) isn't overstated on volatile arms.

**Confirm/kill:** does adding a volatility filter or NB distribution improve held-out
log-loss AND raise the tracker's realized hit-rate on kept bets? If not, kill.

**RESULT — KILLED (2026-06-30 backtest).** K var/mean = **1.05** → strikeouts are almost
exactly Poisson across the dataset; there is no over-dispersion to fix. Held-out log-loss:
Poisson baseline **0.6655**; NegBin all WORSE (r=4→0.6741, r=8→0.6688, r=117≈Poisson);
volatility filter WORSE (vol≤2.5 → 0.6744, and tosses 25% of bets). Wrobleski's 11 was a
genuine ~2% Poisson tail, not model overconfidence — we pattern-matched one outlier into a
bias that isn't there. **Do not add NB or a vol filter for Ks.** (Note: OUTS is under-
dispersed var/mean=0.64 — opposite problem — so this result does NOT transfer to outs.)

---

## Calibration baseline (as of 2026-06-30, n=1 day — noise)
- **Hits model:** predicted 0.590 / actual 0.615 → calibrated. Looking legit.
- **K model:** predicted ~0.68 / actual 0.46 → overconfident. See H1.
- Held-out log-loss (backtest, relative signal only): K 0.6800, Hits 0.6559, Outs 0.6468, coin 0.6931.

## H3 — New features (days-rest, fatigue, opponent recent form) — KILLED

**Status:** KILLED (2026-06-30 backtest). Tested three NEW features the model lacks, on
held-out K o/u log-loss (baseline 0.6655):
| feature added | log-loss |
|---------------|----------|
| BASELINE (4 feats) | **0.6655** |
| + rest_days | 0.6664 |
| + prev_bf (fatigue proxy) | 0.6663 |
| + rest + prev_bf | 0.6669 |
| opp_krate → opp_krate_recent(5) | 0.6658 |
| + opp_krate_recent (both) | 0.6679 |
| ALL new features | 0.6693 |

**All WORSE than baseline.** Why: (1) rest — MLB starters are on a rigid 5-day rotation,
almost no variation to exploit; (2) fatigue — already captured by pit_bf_exp (expected
volume), prev_bf is redundant; (3) opp recent form — season opp_krate already captures
lineup quality; last-5 is noisier per-opponent without being more predictive. (No pitch-
count column exists in the dataset; used prev bf as proxy.)

## META: four straight killed hypotheses (H1-window, H2-dist, H2-volfilter, H3-features)
Every "obvious" model improvement has FAILED the held-out test. Read: the model has already
extracted the signal available in this data; remaining error (~0.665 log-loss, over-rate
~0.50) is largely IRREDUCIBLE per-start variance. Stop tuning the model internals. The open
question that CAN still find value is external: **model vs de-vigged book line** (does the
model beat the market, and on which segments?) — needs the forward book-capture data, not
more backtests.

## H4 — Park factors for Hits — NOT A KEEPER (noise-level gain)
**Status:** effectively KILLED (2026-06-30 backtest). Held-out Hits log-loss:
baseline 0.6560 → +opp_park_pf **0.6557** (−0.0003, within noise at n=823).
Two red flags: (1) dataset has no venue column and no pitcher-team, so the factor could only
be built for AWAY games (venue = opp park); home starts defaulted to neutral 1.0 — half the
signal missing. (2) Park-factor spread came out 0.20–2.12 — physically implausible (real park
factors ~0.85–1.15), meaning it's capturing small-sample matchup NOISE, not venue. A real
park feature needs true venue per game + enough games/park to stabilize; this partial-season
~1000-row set can't support it. Revisit only with a full season + venue column.

## H5 — Home/away splits — SIGNAL REAL, feature already captures it (no change)
**Status:** KILLED as an actionable change (2026-06-30 backtest). The home/away *signal is real*:
| stat | HOME/bf | AWAY/bf | diff |
|------|---------|---------|------|
| K    | 0.2359  | 0.2287  | **+3.2%** (K more at home) |
| hits | 0.2091  | 0.2163  | **−3.3%** (fewer hits at home) |

Both directions match home-field advantage. But the model already has a `home` feature, so the
question is "does it earn its place?" Held-out o/u log-loss, WITH vs WITHOUT `home`:
- **K model:** WITH 0.6655 / WITHOUT 0.6660 → home HELPS. Keep it.
- **Hits model:** WITH 0.6560 / WITHOUT 0.6547 → looked like dropping home helps (−0.0013)...
  **but a paired bootstrap kills it:** 95% CI [−0.0024, +0.0051] straddles zero, only 77% of
  samples show any benefit (need ~95%), and the `home` coef has the CORRECT sign (−0.026 →
  home = fewer hits). It's noise at n=823, same as park factors. **Leave hits_model.py alone.**

Takeaway: the home/away effect is genuine but ALREADY absorbed by pit_hrate/opp_hrate + the
existing `home` term. No free value in touching it.

## META (updated): SIX straight killed/noise hypotheses
H1-window, H2-dist, H2-volfilter, H3-features, H4-park, H5-homeaway. Model internals are exhausted
on this data. STOP backtesting model changes. Value question is now purely EXTERNAL (model vs book,
forward). Only revisit internals with (a) a full-season dataset, or (b) a genuinely new data
source (handedness splits, batted-ball, true venue).

## Backlog (deferred, not started)
- **Pitcher-vs-specific-team history** (e.g. "deGrom owns the Marlins") — CAN'T TEST on this data.
  Partial season: 693 matchups seen once, 162 twice, 4 thrice, 0 four+. Only 170/1029 starts
  (16.5%) have even ONE prior meeting. Where measurable, corr(history-vs-team edge, actual edge)
  = **−0.013** (zero) — no durable effect beyond "this pitcher is good," already in pit_krate.
  Real p-vs-team signal needs YEARS of meetings. Revisit only with a multi-season dataset.
- Handedness splits feature — needs a new data pull; priors suggest low odds but untested.
- Model-vs-book segmentation once book overlap accumulates (~1 week). ← HIGHEST VALUE NOW.
- Recalibration layer if forward data shows persistent miscalibration.
- Re-test park factors ONLY after acquiring a full season + real venue column.
