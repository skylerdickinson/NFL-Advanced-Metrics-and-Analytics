# ==============================================================================
# nfl_flare_backtest.py
# ==============================================================================
# ON-DEMAND TOOL -- not part of the weekly routine. Third in this real-data
# backtest family (after nfl_historical_backtest.py and
# nfl_walkforward_backtest.py). Answers a different question than either:
# not "which single stat predicts best" but "when a GROUP of stats all lean
# the same way pre-game, does that consensus actually mean something."
#
# Same real, honest walk-forward setup as nfl_walkforward_backtest.py: every
# stat used here is each team's REAL rolling average from games played
# BEFORE that week (2+ real prior games required, same season only -- no
# leakage from the game being "predicted"). For every real game, this script
# counts how many of the available pre-game stats leaned toward the team
# that actually won vs. the team that actually lost, then groups real games
# by how STRONG that lean was (how many stats agreed) and reports the real
# accuracy in each group. If a bigger pile of agreeing stats really is a
# stronger signal, the high-agreement bucket should show a real accuracy
# lift over the low-agreement bucket -- if it doesn't, that's an honest
# "no, piling on more stats doesn't actually help" answer, not a guess.
#
# Nothing here is faked: every number comes from real nflverse play-by-play.
# The per-game stat computation and pre-game rolling-average transform are
# copied verbatim from nfl_walkforward_backtest.py (itself copied from
# nfl_data_pull.py PART 3C). Keep all three files in sync if the source
# blocks ever change.
#
# Usage:
#   python nfl_flare_backtest.py
#       -> tests the 4 most recent real completed seasons
#   python nfl_flare_backtest.py --seasons 2019,2020,2021,2022,2023,2024
# ==============================================================================

import argparse
import datetime
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Does a GROUP of pre-game stats agreeing actually predict better than any one stat alone?"
)
parser.add_argument(
    "--seasons", type=str, default=None,
    help="Comma-separated real season years to test. Default: the 4 most recent completed seasons.",
)
args = parser.parse_args()

if args.seasons:
    SEASONS = sorted({int(y.strip()) for y in args.seasons.split(",") if y.strip()})
else:
    _current_year = datetime.date.today().year
    SEASONS = [y for y in range(_current_year - 4, _current_year) if y >= 2016]

print(f"Flare/consensus testing the real stat-weighted model against real seasons: {SEASONS}\n")

BACKTESTABLE_COLS = [
    'rush_yards', 'rush_yards_allowed', 'pass_yards', 'pass_yards_allowed',
    'off_epa_play_game', 'def_epa_play_allowed_game',
    'off_success_rate_game', 'def_success_rate_allowed_game',
    'third_down_pct_game', 'third_down_pct_allowed_game',
    'sack_pct_offense_game', 'sack_pct_defense_game',
    'pressure_rate_allowed_game', 'pressure_rate_forced_game',
    'completion_pct_game', 'interceptions_thrown_game', 'interceptions_gained_game',
]

# ==============================================================================
# PER-SEASON REAL PER-GAME STAT LOG -- copied verbatim from nfl_data_pull.py
# PART 3B/3C (same block as nfl_historical_backtest.py / nfl_walkforward_backtest.py).
# ==============================================================================


def _per_game(df_src, team_col, value_col, out_col, agg='mean'):
    g = df_src.groupby([team_col, 'game_id'])[value_col].agg(agg).reset_index()
    return g.rename(columns={team_col: 'team', value_col: out_col})


all_game_logs = []

for yr in SEASONS:
    print(f"  Loading real {yr} season play-by-play (nflverse)...")
    pbp = nfl.load_pbp(seasons=[yr]).to_pandas()
    pbp_reg = pbp[(pbp['season_type'] == 'REG') & (pbp['posteam'].notna()) & (pbp['defteam'].notna())].copy()

    if pbp_reg.empty:
        print(f"  ⚠ no real regular-season plays returned for {yr} -- skipping this season")
        continue

    numeric_fields = ['yards_gained', 'epa', 'pass_attempt', 'sack', 'complete_pass', 'interception', 'third_down_converted']
    for field in numeric_fields:
        if field in pbp_reg.columns:
            pbp_reg[field] = pd.to_numeric(pbp_reg[field], errors='coerce')
    if 'qb_hit' in pbp_reg.columns:
        pbp_reg['qb_hit'] = pd.to_numeric(pbp_reg['qb_hit'], errors='coerce').fillna(0)
    else:
        pbp_reg['qb_hit'] = 0

    pbp_reg['is_success'] = pbp_reg['epa'] > 0
    pbp_clean = pbp_reg[(pbp_reg['play_type'].isin(['pass', 'run'])) & (pbp_reg['epa'].notna())].copy()

    final_scores = pbp_reg.groupby('game_id').agg(
        home=('home_team', 'first'), away=('away_team', 'first'),
        home_score=('home_score', 'last'), away_score=('away_score', 'last'),
    ).reset_index()
    final_scores['winner'] = np.where(
        final_scores['home_score'] > final_scores['away_score'], final_scores['home'],
        np.where(final_scores['away_score'] > final_scores['home_score'], final_scores['away'], 'TIE'))

    game_weeks = pbp_reg.groupby('game_id')['week'].first().reset_index()

    off_game_rush = pbp_clean[pbp_clean['play_type'] == 'run'].groupby(['posteam', 'game_id'])['yards_gained'].sum().reset_index().rename(columns={'posteam': 'team', 'yards_gained': 'rush_yards'})
    off_game_pass = pbp_clean[pbp_clean['play_type'] == 'pass'].groupby(['posteam', 'game_id'])['yards_gained'].sum().reset_index().rename(columns={'posteam': 'team', 'yards_gained': 'pass_yards'})
    def_game_rush = pbp_clean[pbp_clean['play_type'] == 'run'].groupby(['defteam', 'game_id'])['yards_gained'].sum().reset_index().rename(columns={'defteam': 'team', 'yards_gained': 'rush_yards_allowed'})
    def_game_pass = pbp_clean[pbp_clean['play_type'] == 'pass'].groupby(['defteam', 'game_id'])['yards_gained'].sum().reset_index().rename(columns={'defteam': 'team', 'yards_gained': 'pass_yards_allowed'})

    team_games = pbp_reg.groupby(['posteam', 'game_id']).agg(
        defteam=('defteam', 'first'), home_team=('home_team', 'first'), away_team=('away_team', 'first'),
    ).reset_index().rename(columns={'posteam': 'team'})
    team_games = team_games.merge(game_weeks, on='game_id', how='left')
    team_games['opponent'] = team_games['defteam']
    team_games['is_home'] = team_games['team'] == team_games['home_team']
    team_games = team_games.merge(
        final_scores[['game_id', 'home', 'away', 'home_score', 'away_score', 'winner']], on='game_id', how='left')
    team_games['pts_for'] = np.where(team_games['is_home'], team_games['home_score'], team_games['away_score'])
    team_games['pts_against'] = np.where(team_games['is_home'], team_games['away_score'], team_games['home_score'])
    team_games['result'] = np.where(team_games['winner'] == team_games['team'], 'W',
                              np.where(team_games['winner'] == 'TIE', 'T', 'L'))

    team_games = team_games.merge(off_game_rush, on=['team', 'game_id'], how='left')
    team_games = team_games.merge(off_game_pass, on=['team', 'game_id'], how='left')
    team_games = team_games.merge(def_game_rush, on=['team', 'game_id'], how='left')
    team_games = team_games.merge(def_game_pass, on=['team', 'game_id'], how='left')
    yard_cols = ['rush_yards', 'pass_yards', 'rush_yards_allowed', 'pass_yards_allowed']
    team_games[yard_cols] = team_games[yard_cols].fillna(0)

    off_epa_g = _per_game(pbp_clean, 'posteam', 'epa', 'off_epa_play_game')
    def_epa_g = _per_game(pbp_clean, 'defteam', 'epa', 'def_epa_play_allowed_game')
    off_sr_g = _per_game(pbp_clean, 'posteam', 'is_success', 'off_success_rate_game', agg='mean')
    off_sr_g['off_success_rate_game'] *= 100
    def_sr_g = _per_game(pbp_clean, 'defteam', 'is_success', 'def_success_rate_allowed_game', agg='mean')
    def_sr_g['def_success_rate_allowed_game'] *= 100

    third_down_plays_g = pbp_clean[pbp_clean['third_down_converted'].notna()]
    off_3rd_g = third_down_plays_g.groupby(['posteam', 'game_id'])['third_down_converted'].mean().reset_index().rename(
        columns={'posteam': 'team', 'third_down_converted': 'third_down_pct_game'})
    off_3rd_g['third_down_pct_game'] *= 100
    def_3rd_g = third_down_plays_g.groupby(['defteam', 'game_id'])['third_down_converted'].mean().reset_index().rename(
        columns={'defteam': 'team', 'third_down_converted': 'third_down_pct_allowed_game'})
    def_3rd_g['third_down_pct_allowed_game'] *= 100

    sack_off_g = pbp_reg.groupby(['posteam', 'game_id']).agg(
        _sacks=('sack', 'sum'), _atts=('pass_attempt', 'sum')).reset_index().rename(columns={'posteam': 'team'})
    sack_off_g['sack_pct_offense_game'] = np.where(
        (sack_off_g['_sacks'] + sack_off_g['_atts']) > 0,
        sack_off_g['_sacks'] / (sack_off_g['_sacks'] + sack_off_g['_atts']) * 100, np.nan)
    sack_def_g = pbp_reg.groupby(['defteam', 'game_id']).agg(
        _sacks=('sack', 'sum'), _atts=('pass_attempt', 'sum')).reset_index().rename(columns={'defteam': 'team'})
    sack_def_g['sack_pct_defense_game'] = np.where(
        (sack_def_g['_sacks'] + sack_def_g['_atts']) > 0,
        sack_def_g['_sacks'] / (sack_def_g['_sacks'] + sack_def_g['_atts']) * 100, np.nan)

    pressure_off_g = pbp_reg.groupby(['posteam', 'game_id']).agg(
        _sacks=('sack', 'sum'), _hits=('qb_hit', 'sum'), _atts=('pass_attempt', 'sum')).reset_index().rename(columns={'posteam': 'team'})
    pressure_off_g['_drop'] = pressure_off_g['_sacks'] + pressure_off_g['_atts']
    pressure_off_g['pressure_rate_allowed_game'] = np.where(
        pressure_off_g['_drop'] > 0, (pressure_off_g['_sacks'] + pressure_off_g['_hits']) / pressure_off_g['_drop'] * 100, np.nan)
    pressure_def_g = pbp_reg.groupby(['defteam', 'game_id']).agg(
        _sacks=('sack', 'sum'), _hits=('qb_hit', 'sum'), _atts=('pass_attempt', 'sum')).reset_index().rename(columns={'defteam': 'team'})
    pressure_def_g['_drop'] = pressure_def_g['_sacks'] + pressure_def_g['_atts']
    pressure_def_g['pressure_rate_forced_game'] = np.where(
        pressure_def_g['_drop'] > 0, (pressure_def_g['_sacks'] + pressure_def_g['_hits']) / pressure_def_g['_drop'] * 100, np.nan)

    comp_g = pbp_clean[pbp_clean['play_type'] == 'pass'].groupby(['posteam', 'game_id']).agg(
        _comp=('complete_pass', 'sum'), _att=('pass_attempt', 'sum'), _int=('interception', 'sum')).reset_index().rename(columns={'posteam': 'team'})
    comp_g['completion_pct_game'] = np.where(comp_g['_att'] > 0, comp_g['_comp'] / comp_g['_att'] * 100, np.nan)
    comp_g = comp_g.rename(columns={'_int': 'interceptions_thrown_game'})
    int_def_g = pbp_clean[pbp_clean['play_type'] == 'pass'].groupby(['defteam', 'game_id'])['interception'].sum().reset_index().rename(
        columns={'defteam': 'team', 'interception': 'interceptions_gained_game'})

    for extra_df, cols in [
        (off_epa_g, ['off_epa_play_game']), (def_epa_g, ['def_epa_play_allowed_game']),
        (off_sr_g, ['off_success_rate_game']), (def_sr_g, ['def_success_rate_allowed_game']),
        (off_3rd_g, ['third_down_pct_game']), (def_3rd_g, ['third_down_pct_allowed_game']),
        (sack_off_g[['team', 'game_id', 'sack_pct_offense_game']], ['sack_pct_offense_game']),
        (sack_def_g[['team', 'game_id', 'sack_pct_defense_game']], ['sack_pct_defense_game']),
        (pressure_off_g[['team', 'game_id', 'pressure_rate_allowed_game']], ['pressure_rate_allowed_game']),
        (pressure_def_g[['team', 'game_id', 'pressure_rate_forced_game']], ['pressure_rate_forced_game']),
        (comp_g[['team', 'game_id', 'completion_pct_game', 'interceptions_thrown_game']], ['completion_pct_game', 'interceptions_thrown_game']),
        (int_def_g, ['interceptions_gained_game']),
    ]:
        team_games = team_games.merge(extra_df, on=['team', 'game_id'], how='left')

    season_log = team_games[[
        'team', 'game_id', 'week', 'opponent', 'is_home', 'result', 'pts_for', 'pts_against',
    ] + BACKTESTABLE_COLS].copy()
    season_log['season'] = yr
    season_log['game_id'] = season_log['season'].astype(str) + "_" + season_log['game_id'].astype(str)
    all_game_logs.append(season_log)
    print(f"    -> {yr}: {len(season_log)} real team-game row(s), {season_log['game_id'].nunique()} real game(s).")

if not all_game_logs:
    raise SystemExit("No real seasons could be loaded -- nothing to backtest.")

historical_log = pd.concat(all_game_logs, ignore_index=True).sort_values(['team', 'season', 'week']).reset_index(drop=True)
print(f"\nTotal real historical team-game rows: {len(historical_log)} across {len(SEASONS)} real season(s).\n")

# ==============================================================================
# WALK-FORWARD TRANSFORM -- copied verbatim from nfl_walkforward_backtest.py.
# ==============================================================================

print("Building real pre-game rolling averages (2+ prior real games required, per team per season)...")
prewk_log = historical_log.copy().sort_values(['team', 'season', 'week'])
for col in BACKTESTABLE_COLS:
    prewk_log[col] = (
        prewk_log.groupby(['team', 'season'])[col]
        .transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
    )
print()

# ==============================================================================
# VOTE / GAME-PAIRING MACHINERY -- copied verbatim from nfl_matchup_sheet.py's
# model block (NFL_BACKTESTABLE_STATS, _nfl_stat_vote, record_nfl_completed_games).
# Only used here to build the per-game vote dict; the greedy tuning function
# isn't needed for the consensus/flare analysis below.
# ==============================================================================

NFL_BACKTESTABLE_STATS = [
    ("Rush Yards", "rush_yards", "rushing_yards_per_game", True, "TRENCHES & RUN GAME"),
    ("Rush Yards Allowed", "rush_yards_allowed", "rushing_yards_allowed_per_game", False, "TRENCHES & RUN GAME"),
    ("Pass Yards", "pass_yards", "passing_yards_per_game", True, "PASSING & PROTECTION"),
    ("Pass Yards Allowed", "pass_yards_allowed", "passing_yards_allowed_per_game", False, "PASSING & PROTECTION"),
    ("EPA/Play (Offense)", "off_epa_play_game", "off_efficiency_epa", True, "OVERALL EFFICIENCY"),
    ("EPA/Play Allowed (Defense)", "def_epa_play_allowed_game", "def_efficiency_epa", False, "OVERALL EFFICIENCY"),
    ("Success Rate (Offense)", "off_success_rate_game", "off_success_rate", True, "OVERALL EFFICIENCY"),
    ("Success Rate Allowed (Defense)", "def_success_rate_allowed_game", "def_success_rate", False, "OVERALL EFFICIENCY"),
    ("3rd Down %", "third_down_pct_game", "third_down_pct_offense", True, "OVERALL EFFICIENCY"),
    ("3rd Down % Allowed", "third_down_pct_allowed_game", "third_down_pct_defense", False, "OVERALL EFFICIENCY"),
    ("Sack % Taken", "sack_pct_offense_game", "sack_pct_offense", False, "PASSING & PROTECTION"),
    ("Sack % Forced", "sack_pct_defense_game", "sack_pct_defense", True, "PASSING & PROTECTION"),
    ("Pressure Rate Allowed", "pressure_rate_allowed_game", "pressure_rate_allowed_pct", False, "PASSING & PROTECTION"),
    ("Pressure Rate Forced", "pressure_rate_forced_game", "pressure_rate_forced_pct", True, "PASSING & PROTECTION"),
    ("Completion %", "completion_pct_game", "completion_pct", True, "QB ACCURACY & EFFICIENCY"),
    ("Interceptions Thrown", "interceptions_thrown_game", "interceptions_thrown_per_game", False, "QB ACCURACY & EFFICIENCY"),
    ("Interceptions Gained", "interceptions_gained_game", "interceptions_gained_per_game", True, "QB ACCURACY & EFFICIENCY"),
]


def _nfl_stat_vote(winner_val, loser_val, higher_is_better):
    if winner_val is None or loser_val is None:
        return None
    try:
        w, l = float(winner_val), float(loser_val)
    except (TypeError, ValueError):
        return None
    if pd.isna(w) or pd.isna(l):
        return None
    if w == l:
        return None
    winner_led = (w > l) if higher_is_better else (w < l)
    return 1 if winner_led else -1


def record_nfl_completed_games(history, game_log_source_df):
    if game_log_source_df is None or game_log_source_df.empty or 'result' not in game_log_source_df.columns:
        return 0
    has_game_id = 'game_id' in game_log_source_df.columns

    def _game_key(row):
        if has_game_id and pd.notna(row.get('game_id')):
            return f"gid_{row['game_id']}"
        return f"wk_{row['week']}_{'-'.join(sorted([str(row['team']), str(row['opponent'])]))}"

    grouped = {}
    for _, row in game_log_source_df.iterrows():
        grouped.setdefault(_game_key(row), []).append(row)

    new_count = 0
    for key, rows in grouped.items():
        if len(rows) != 2 or key in history["games"]:
            continue
        r0, r1 = rows[0], rows[1]
        if r0['result'] not in ('W', 'L') or r1['result'] not in ('W', 'L') or r0['result'] == r1['result']:
            continue
        winner_row, loser_row = (r0, r1) if r0['result'] == 'W' else (r1, r0)
        votes = {}
        for label, game_col, matrix_col, higher_is_better, category in NFL_BACKTESTABLE_STATS:
            vote = _nfl_stat_vote(winner_row.get(game_col), loser_row.get(game_col), higher_is_better)
            if vote is not None:
                votes[game_col] = vote
        history["games"][key] = {
            "week": int(winner_row['week']) if pd.notna(winner_row.get('week')) else None,
            "winner": winner_row['team'], "loser": loser_row['team'],
            "votes": votes,
        }
        new_count += 1
    return new_count


history = {"games": {}}
n_recorded = record_nfl_completed_games(history, prewk_log)
print(f"Recorded {n_recorded} real completed game(s) with real pre-game stats on file.\n")

# ==============================================================================
# THE ACTUAL "FLARE" TEST: does a bigger pile of pre-game stats agreeing on
# the same team actually mean something, or is it noise? For every real
# game, votes[stat] is +1 if the ACTUAL WINNER had the better pre-game
# number on that stat, -1 if the ACTUAL LOSER did. Summing those votes
# gives a real, unweighted "how many stats leaned toward who actually won,
# net" margin for that specific game -- completely independent of who won,
# since every input is pre-game data. Bucketing real games by how BIG that
# margin was and checking real accuracy in each bucket is the direct test
# of "does a bigger group of agreeing stats actually predict better."
# ==============================================================================

margins = []
for game in history["games"].values():
    votes = game.get("votes", {})
    if not votes:
        continue
    margin = sum(votes.values())  # net number of stats favoring the ACTUAL winner
    margins.append({"margin": margin, "n_stats": len(votes), "correct": margin > 0, "push": margin == 0})

margins_df = pd.DataFrame(margins)
n_push = int(margins_df["push"].sum())
decided_df = margins_df[~margins_df["push"]].copy()

print("=" * 78)
print("OVERALL: simple majority vote across every available pre-game stat")
print("=" * 78)
overall_acc = decided_df["correct"].mean() if len(decided_df) else float("nan")
print(f"{int(decided_df['correct'].sum())}/{len(decided_df)} decided = {overall_acc*100:.1f}% "
      f"({n_push} real game(s) exactly split down the middle -- no majority either way, left undecided)")

print("\n" + "=" * 78)
print("THE FLARE TEST: real accuracy, grouped by how many stats agreed (net margin)")
print("=" * 78)
decided_df["abs_margin"] = decided_df["margin"].abs()
bins = [0, 2, 4, 6, 8, 100]
labels = ["1-2 stats net lean", "3-4 stats net lean", "5-6 stats net lean", "7-8 stats net lean", "9+ stats net lean"]
decided_df["bucket"] = pd.cut(decided_df["abs_margin"], bins=bins, labels=labels, right=True)

print(f"{'Consensus strength':<22s} {'# real games':>13s} {'% of decided games':>19s} {'real accuracy':>15s}")
for label in labels:
    sub = decided_df[decided_df["bucket"] == label]
    if len(sub) == 0:
        continue
    acc = sub["correct"].mean()
    pct_of_all = len(sub) / len(decided_df) * 100
    print(f"{label:<22s} {len(sub):>13d} {pct_of_all:>18.1f}% {acc*100:>14.1f}%")

print(f"\nIf accuracy climbs as you move down that table (more agreeing stats -> higher real accuracy), a "
      f"strong multi-stat consensus really is a stronger signal than any single stat alone -- an actual 'flare.' "
      f"If it's flat or doesn't climb cleanly, piling on more agreeing stats isn't adding real information beyond "
      f"what the top 1-2 stats already told you (they're too correlated with each other to compound).")

print("\nNote: VOID, Pass Block Efficiency/Pass Rush Productivity, Havoc Rate, Red Zone %, Passer Rating, CPOE, and "
      "Turnover Margin are still not in this test -- same PART 3C scope limit as the other real backtests.")

# ==============================================================================
# QUALITY-ONLY FLARE TEST: the table above counts all 17 stats equally, so a
# "9+ agree" game could be padded out by weak stats (like Pass Yards Allowed,
# which tested BELOW a coin flip in the walk-forward test) riding along with
# the real ones. This section restricts the consensus count to only the
# stats that already proved themselves -- real individual win rate > 55% in
# THIS SAME real walk-forward sample -- and re-runs the exact same bucket
# test. If a SMALLER number of quality stats agreeing beats what it took 9+
# of all 17 to achieve above, that's the sharper, more useful signal for the
# sheet to actually surface.
# ==============================================================================


def _nfl_individual_stat_win_rates(history):
    tallies = {}
    for game in history["games"].values():
        for key, vote in game.get("votes", {}).items():
            t = tallies.setdefault(key, [0, 0])
            t[1] += 1
            if vote == 1:
                t[0] += 1
    return {k: (c / n, n) for k, (c, n) in tallies.items() if n >= 3}


label_by_key = {game_col: label for label, game_col, _, _, _ in NFL_BACKTESTABLE_STATS}
win_rates = _nfl_individual_stat_win_rates(history)
QUALITY_THRESHOLD = 0.55
QUALITY_STATS = {k for k, (wr, n) in win_rates.items() if wr > QUALITY_THRESHOLD}
WEAK_STATS = {k for k in win_rates if k not in QUALITY_STATS}

print("\n" + "=" * 78)
print(f"QUALITY-ONLY FLARE TEST: same test, but only counting stats with a real walk-forward "
      f"win rate > {QUALITY_THRESHOLD*100:.0f}%")
print("=" * 78)
print(f"Quality stats ({len(QUALITY_STATS)} of {len(win_rates)}), by real win rate:")
for k in sorted(QUALITY_STATS, key=lambda k: -win_rates[k][0]):
    print(f"  {label_by_key.get(k, k):32s}  {win_rates[k][0]*100:5.1f}%")
print(f"Left out as too weak/noisy to count ({len(WEAK_STATS)}):")
for k in sorted(WEAK_STATS, key=lambda k: -win_rates[k][0]):
    print(f"  {label_by_key.get(k, k):32s}  {win_rates[k][0]*100:5.1f}%")

q_margins = []
for game in history["games"].values():
    votes = game.get("votes", {})
    q_votes = {k: v for k, v in votes.items() if k in QUALITY_STATS}
    if not q_votes:
        continue
    margin = sum(q_votes.values())
    q_margins.append({"margin": margin, "correct": margin > 0, "push": margin == 0})

q_df = pd.DataFrame(q_margins)
q_push = int(q_df["push"].sum())
q_decided = q_df[~q_df["push"]].copy()
q_overall_acc = q_decided["correct"].mean() if len(q_decided) else float("nan")

print(f"\nOverall (quality stats only): {int(q_decided['correct'].sum())}/{len(q_decided)} decided = "
      f"{q_overall_acc*100:.1f}% ({q_push} exactly split, left undecided)")

max_q = len(QUALITY_STATS)
q_bins = [0, 1, 2, 3, max(4, max_q)]
q_labels = ["1 quality stat", "2 quality stats", "3 quality stats", "4+ quality stats"]
q_decided["abs_margin"] = q_decided["margin"].abs()
q_decided["bucket"] = pd.cut(q_decided["abs_margin"], bins=q_bins, labels=q_labels, right=True)

print(f"\n{'Consensus strength':<22s} {'# real games':>13s} {'% of decided games':>19s} {'real accuracy':>15s}")
for label in q_labels:
    sub = q_decided[q_decided["bucket"] == label]
    if len(sub) == 0:
        continue
    acc = sub["correct"].mean()
    pct_of_all = len(sub) / len(q_decided) * 100
    print(f"{label:<22s} {len(sub):>13d} {pct_of_all:>18.1f}% {acc*100:>14.1f}%")

print(f"\nCompare this table's top bucket to the ALL-STATS table's '9+ stats net lean' row above -- if a smaller "
      f"number of QUALITY stats agreeing matches or beats what it took a much bigger pile of all 17 to reach, "
      f"that's the sharper signal: fewer, better stats agreeing beats more stats (including weak ones) agreeing.")

# ==============================================================================
# PER-STAT: when THIS stat is part of a strong consensus (3+ quality stats
# agreeing), how often is the overall call actually right? This is the
# closest thing to "which stats, together, win more" -- not just whether a
# stat is individually predictive (already known from the walk-forward
# test), but whether its presence inside a real strong-consensus game
# tends to ride along with a correct call or a wrong one.
# ==============================================================================

print("\n" + "=" * 78)
print("PER-STAT: real accuracy of the OVERALL quality-consensus call, in games where this stat voted")
print("=" * 78)
per_stat_rows = []
for game in history["games"].values():
    votes = game.get("votes", {})
    q_votes = {k: v for k, v in votes.items() if k in QUALITY_STATS}
    if not q_votes:
        continue
    margin = sum(q_votes.values())
    if margin == 0 or abs(margin) < 3:
        continue  # only look at STRONG quality-consensus games (3+ quality stats net lean)
    call_correct = margin > 0
    for k, v in q_votes.items():
        per_stat_rows.append({"stat": k, "call_correct": call_correct, "agreed_with_call": (v == 1) == call_correct})

if per_stat_rows:
    per_stat_df = pd.DataFrame(per_stat_rows)
    print(f"(Within the {q_decided[q_decided['abs_margin'] >= 3].shape[0]} strong quality-consensus real games "
          f"(3+ quality stats net lean):)")
    print(f"{'Stat':<32s} {'# strong-consensus games':>25s} {'accuracy when it voted':>24s}")
    summary = per_stat_df.groupby("stat")["call_correct"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    for k, row in summary.iterrows():
        print(f"{label_by_key.get(k, k):<32s} {int(row['count']):>25d} {row['mean']*100:>23.1f}%")
else:
    print("Not enough strong-consensus real games yet to break this down by stat.")

print("\nNote: VOID, Pass Block Efficiency/Pass Rush Productivity, Havoc Rate, Red Zone %, Passer Rating, CPOE, and "
      "Turnover Margin are still not in ANY of this -- same PART 3C scope limit as the other real backtests.")
