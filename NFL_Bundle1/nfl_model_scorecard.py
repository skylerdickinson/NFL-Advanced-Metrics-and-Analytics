# ==============================================================================
# nfl_model_scorecard.py
# ==============================================================================
# ON-DEMAND TOOL -- not part of the weekly routine (nfl_data_pull.py then
# nfl_matchup_sheet.py still run every week as normal; this doesn't touch
# either of their input files, only reads/updates nfl_internal_model_track.csv).
#
# . Every time nfl_matchup_sheet.py runs, it quietly logs two real signals
# per game into nfl_internal_model_track.csv: (1) the win-probability model
# behind the >=80% turquoise TEAM NAME highlight, and (2) the quality-stat
# consensus BADGE count/percentage. Neither one is scored against real
# results automatically as part of that weekly run -- that's this script's
# job. Run it anytime (once a week's real games are final is the obvious
# time) and it will:
#
#   1. Pull real final scores from nflverse (load_schedules -- same source
#      as everything else in this pipeline) for every real season that
#      appears in the log, and auto-fill actual_away_score/actual_home_score
#      for any game that's now final but hasn't been scored yet. No more
#      typing scores in by hand.
#   2. Print a real accuracy readout for BOTH signals -- overall, and for
#      the specific "confident" subset of each (win-prob >=80% for the
#      model, count >=QUALITY_CONSENSUS_HOT_THRESHOLD for the badge -- i.e.
#      exactly the two thresholds that turn each signal turquoise on the
#      real sheet), so you can see whether turquoise is actually earning
#      its color, not just how the signal does on average.
#
# Real games only: nothing here is simulated. If a game hasn't been played
# yet, or nflverse hasn't posted a final score, it's correctly left
# unscored rather than guessed at -- the readout always says how many real
# decided games it's based on.
#
# Usage:
#   python nfl_model_scorecard.py
#       -> auto-fills real scores + prints the real accuracy readout
#   python nfl_model_scorecard.py --dry-run
#       -> prints the readout using whatever's already scored, without
#          pulling nflverse or writing anything back to the CSV
#   python nfl_model_scorecard.py --seasons 2026
#       -> only pull/score these real season(s), instead of every season
#          already present in the log (useful the first time you run this
#          against an old-format log with no 'season' column at all -- see
#          the LEGACY ROWS note below)
# ==============================================================================

import argparse
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent

# Mirrors QUALITY_CONSENSUS_HOT_THRESHOLD in nfl_matchup_sheet.py -- if you
# ever change that constant there (say, after re-running the backtests with
# more real seasons and the right bar moves), update this copy to match, or
# the "confident badge" subset below will be scoring the wrong bar.
QUALITY_CONSENSUS_HOT_THRESHOLD = 8
# Mirrors the >=80% cutoff used for the team-name turquoise highlight in
# nfl_matchup_sheet.py's build_matchup_elements.
WIN_PROB_HOT_THRESHOLD = 0.80

cli = argparse.ArgumentParser(
    description="Auto-fill real final scores into nfl_internal_model_track.csv and report real accuracy "
                "for the win-prob model and the quality-stat consensus badge."
)
cli.add_argument("--seasons", type=str, default=None,
                  help="Comma-separated real season(s) to pull/score, e.g. 2025,2026. "
                       "Default: every season already present in the log.")
cli.add_argument("--dry-run", action="store_true",
                  help="Only print the readout from what's already scored -- don't pull nflverse or write to the CSV.")
args = cli.parse_args()

internal_path = OUT_DIR / 'nfl_internal_model_track.csv'
if not internal_path.exists():
    raise SystemExit(f"No {internal_path.name} found in this folder yet -- run nfl_matchup_sheet.py at least "
                      f"once first (it logs to this file every time it runs).")

log = pd.read_csv(internal_path)
if log.empty:
    raise SystemExit(f"{internal_path.name} exists but has no rows yet.")

# ---- LEGACY ROWS: rows logged before this feature added 'season'/'week'
# columns have no real way to be matched to a schedule pull (week_tag alone
# isn't reliable -- see nfl_matchup_sheet.py's Step 4b comment). They're
# left in the file and counted here, but skipped for auto-scoring; if one
# of them already has a real actual_away_score/actual_home_score recorded
# by hand, it's still included in the accuracy readout below, just never
# auto-filled.
has_season_week = 'season' in log.columns and 'week' in log.columns
if has_season_week:
    legacy_mask = log['season'].isna() | log['week'].isna()
else:
    legacy_mask = pd.Series([True] * len(log), index=log.index)
n_legacy = int(legacy_mask.sum())

# ==============================================================================
# STEP 1: auto-fill real final scores from nflverse for anything not
# already scored (skipped entirely in --dry-run).
# ==============================================================================
n_filled = 0
if not args.dry_run and (~legacy_mask).any():
    import nflreadpy as nfl

    scorable = log[~legacy_mask]
    if args.seasons:
        seasons = sorted({int(y.strip()) for y in args.seasons.split(",") if y.strip()})
    else:
        seasons = sorted({int(s) for s in scorable['season'].dropna().unique()})

    real_scores = {}  # (season, week, away, home) -> (away_score, home_score)
    for season in seasons:
        try:
            sched = nfl.load_schedules(seasons=[season]).to_pandas()
        except Exception as e:
            print(f" -> Couldn't pull the real {season} schedule from nflverse ({e}); skipping that season.")
            continue
        sched = sched[sched['game_type'] == 'REG'].copy()
        sched = sched.dropna(subset=['away_score', 'home_score'])  # only real FINAL games
        for _, g in sched.iterrows():
            real_scores[(season, int(g['week']), g['away_team'], g['home_team'])] = (g['away_score'], g['home_score'])

    for i, row in log.iterrows():
        if legacy_mask.loc[i]:
            continue
        if pd.notna(row.get('actual_away_score')) and pd.notna(row.get('actual_home_score')):
            continue  # already scored, don't touch it
        key = (int(row['season']), int(row['week']), row['away'], row['home'])
        if key in real_scores:
            away_sc, home_sc = real_scores[key]
            log.at[i, 'actual_away_score'] = away_sc
            log.at[i, 'actual_home_score'] = home_sc
            n_filled += 1

    if n_filled:
        log.to_csv(internal_path, index=False)
    print(f"Step 1: pulled real final scores from nflverse for season(s) {seasons} -> "
          f"auto-filled {n_filled} newly-completed real game(s) into {internal_path.name}.")
elif args.dry_run:
    print("Step 1: --dry-run -- skipped pulling nflverse, scoring only what's already recorded.")
else:
    print("Step 1: no real, non-legacy games to score yet.")

# ==============================================================================
# STEP 2: score both signals against every real game that has a recorded
# actual_away_score/actual_home_score (whether just auto-filled above, or
# already there from before -- including legacy rows scored by hand).
# ==============================================================================
scored = log.dropna(subset=['actual_away_score', 'actual_home_score']).copy()
n_ties = int((scored['actual_away_score'] == scored['actual_home_score']).sum())
scored = scored[scored['actual_away_score'] != scored['actual_home_score']]  # real NFL ties excluded, not guessed
scored['real_winner_is_away'] = scored['actual_away_score'] > scored['actual_home_score']


def _score_signal(df, pick_is_away_col, confident_mask, label, confident_label):
    d = df.dropna(subset=[pick_is_away_col])
    n_push = int(df[pick_is_away_col].isna().sum())  # ties/pushes on this signal specifically
    correct = (d[pick_is_away_col] == d['real_winner_is_away']).sum()
    n = len(d)
    acc = (correct / n * 100) if n else 0.0
    print(f" -> {label}: {correct}/{n} = {acc:.1f}% straight-up"
          + (f"  ({n_push} push/tie excluded)" if n_push else ""))

    conf = d[confident_mask.loc[d.index]]
    if len(conf):
        conf_correct = (conf[pick_is_away_col] == conf['real_winner_is_away']).sum()
        conf_acc = conf_correct / len(conf) * 100
        print(f"      {confident_label}: {conf_correct}/{len(conf)} = {conf_acc:.1f}%")
    else:
        print(f"      {confident_label}: no real games have hit that bar yet.")


print(f"\nStep 2: scoring against {len(scored)} real decided game(s)"
      + (f" ({n_ties} real tie(s) excluded)" if n_ties else "")
      + (f", {n_legacy} legacy-format row(s) skipped (no season/week recorded)" if n_legacy else "")
      + ".")

if scored.empty:
    print("\nNo real decided games scored yet -- nothing to report. Run this again once some real games are final.")
    raise SystemExit(0)

print("\n== WIN-PROBABILITY MODEL (the >=80% turquoise TEAM NAME highlight) ==")
scored['model_pick_is_away'] = scored['model_win_prob_away'].apply(
    lambda p: True if p > 0.5 else (False if p < 0.5 else None))
model_confident = (scored['model_win_prob_away'] >= WIN_PROB_HOT_THRESHOLD) | \
                   (scored['model_win_prob_away'] <= (1 - WIN_PROB_HOT_THRESHOLD))
_score_signal(scored, 'model_pick_is_away', model_confident,
              "Overall", f"Confident only (>= {WIN_PROB_HOT_THRESHOLD*100:.0f}% either way -- the turquoise cases)")

print("\n== QUALITY STAT CONSENSUS BADGE (the >=8-of-12 turquoise BADGE) ==")
if 'quality_count_away' in scored.columns:
    scored['quality_pick_is_away'] = scored.apply(
        lambda r: True if r['quality_count_away'] > r['quality_count_home']
        else (False if r['quality_count_home'] > r['quality_count_away'] else None), axis=1)
    quality_confident = scored[['quality_count_away', 'quality_count_home']].max(axis=1) >= QUALITY_CONSENSUS_HOT_THRESHOLD
    _score_signal(scored, 'quality_pick_is_away', quality_confident,
                  "Overall", f"Confident only (>= {QUALITY_CONSENSUS_HOT_THRESHOLD}/12 either way -- the turquoise cases)")
else:
    print(" -> No quality-badge columns in this log yet -- these rows predate that feature. "
          "Run nfl_matchup_sheet.py again to start logging it, then re-run this script later.")

print("\n== PROJECTED TOTAL POINTS (the 'PROJ TOTAL' number under vs in the header) ==")
if 'model_total' in scored.columns:
    tp = scored.dropna(subset=['model_total']).copy()
    tp['real_total'] = tp['actual_away_score'] + tp['actual_home_score']
    tp['total_error'] = tp['model_total'] - tp['real_total']  # positive = model projected too HIGH
    n_tp = len(tp)
    if n_tp:
        mae = tp['total_error'].abs().mean()
        bias = tp['total_error'].mean()
        bias_word = "high" if bias > 0 else ("low" if bias < 0 else "even")
        print(f" -> Average miss: {mae:.1f} points (mean absolute error) on {n_tp} real scored game(s).")
        print(f"      Average direction: {bias:+.1f} points ({'projects too ' + bias_word if abs(bias) >= 0.5 else 'no real lean either way'})")
        for within in (3, 7, 10):
            pct_within = (tp['total_error'].abs() <= within).mean() * 100
            print(f"      Within {within} points of the real total: {pct_within:.0f}%")
    else:
        print(" -> No real scored games with a logged projection yet.")
else:
    print(" -> No model_total column in this log yet -- these rows predate this feature. "
          "Run nfl_matchup_sheet.py again, then re-run this script later.")

# ---- per-week breakdown, so you can watch it develop week to week instead
# of just one lifetime number ----
if 'week' in scored.columns and scored['week'].notna().any():
    print("\n== BY WEEK ==")
    for (season, week), g in scored.dropna(subset=['week']).groupby(['season', 'week']):
        model_g = g.dropna(subset=['model_pick_is_away'])
        model_correct = (model_g['model_pick_is_away'] == model_g['real_winner_is_away']).sum()
        line = f" -> {int(season)} week {int(week)}: win-prob model {model_correct}/{len(model_g)}"
        if 'quality_pick_is_away' in g.columns:
            q_g = g.dropna(subset=['quality_pick_is_away'])
            q_correct = (q_g['quality_pick_is_away'] == q_g['real_winner_is_away']).sum()
            line += f", quality badge {q_correct}/{len(q_g)}"
        if 'model_total' in g.columns:
            tp_g = g.dropna(subset=['model_total'])
            if len(tp_g):
                tp_g_err = (tp_g['model_total'] - (tp_g['actual_away_score'] + tp_g['actual_home_score'])).abs().mean()
                line += f", total pts avg miss {tp_g_err:.1f}"
        print(line)
