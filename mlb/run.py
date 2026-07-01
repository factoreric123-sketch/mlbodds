"""
run.py
------
One-command daily driver for the K-prop forward-test.

  1. GRADE: score every logged-but-ungraded day against final boxscores,
     print running ROI.
  2. PREDICT: refresh the dataset from StatsAPI and log tomorrow's candidate
     bets (or --date for a specific slate).

Usage:
    python3 run.py                 # grade past logs, then log tomorrow
    python3 run.py --today         # morning run: grade past, then log TODAY's slate
    python3 run.py --date 2026-07-02
    python3 run.py --no-refresh    # skip the StatsAPI rebuild (faster, uses cached data)
    python3 run.py --grade-only
    python3 run.py --predict-only
    python3 run.py --snapshot      # near first pitch: capture today's closing lines for CLV

Daily cadence:
    morning   -> python3 run.py            (grade yesterday, log tomorrow)
    pre-game  -> python3 run.py --snapshot  (freeze closing lines on today's logged bets)
"""
from __future__ import annotations

import argparse
import datetime as dt

import grade_day
import predict_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="slate to predict (default: tomorrow, or today with --today)")
    ap.add_argument("--today", action="store_true",
                    help="predict today's slate instead of tomorrow's (the morning run)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip rebuilding the dataset from StatsAPI")
    ap.add_argument("--grade-only", action="store_true")
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--snapshot", action="store_true",
                    help="capture today's (~closing) lines for CLV, then exit")
    args = ap.parse_args()

    if args.snapshot:
        # snapshot freezes the closing line on bets already logged for a slate that is
        # about to start -> default to today, not tomorrow.
        snap_date = args.date or dt.date.today().isoformat()
        print("=" * 60)
        print(f"SNAPSHOT closing lines for {snap_date}")
        print("=" * 60)
        grade_day.main(["--snapshot", "--date", snap_date])
        return

    if args.date is None:
        offset = 0 if args.today else 1
        args.date = (dt.date.today() + dt.timedelta(days=offset)).isoformat()

    if not args.predict_only:
        print("=" * 60)
        print("GRADING past logs")
        print("=" * 60)
        grade_day.main([])           # grade all ungraded days

    if not args.grade_only:
        print("\n" + "=" * 60)
        print(f"PREDICTING {args.date}")
        print("=" * 60)
        pred_args = ["--date", args.date]
        if not args.no_refresh:
            pred_args.append("--refresh")
        predict_day.main(pred_args)


if __name__ == "__main__":
    main()
