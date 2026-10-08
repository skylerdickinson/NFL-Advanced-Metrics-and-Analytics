# ==============================================================================
# nfl_historical_backtest.py
# ==============================================================================
# ON-DEMAND TOOL -- not part of the weekly routine (nfl_data_pull.py then
# nfl_matchup_sheet.py still runs every week as normal; this doesn't touch
# either of their output files). Run it whenever you want to know "does the
# stat-weighted win prediction model actually hold up," using real PAST
# completed seasons instead of waiting for the current season to slowly
# build up a big enough sample in nfl_stat_history.json.
#
# Right now (in-season) the weekly model only has 1-2 real games per team to
# learn from, which isn't enough to trust. This script pulls several real,
# fully-played past NFL seasons from nflverse, computes the exact same
# real per-game stats nfl_data_pull.py's PART 3C computes for the current
# season (same formulas, same columns -- copied verbatim, not reimplemented,
# so behavior matches exactly), and runs the exact same backtest/weight-
# tuning model nfl_matchup_sheet.py uses (also copied verbatim) against
# hundreds of real completed games at once.
#
# Nothing here is faked: every number comes from real nflverse play-by-play.
# If either of the two source blocks below (PART 3C's per-game stats, or the
# model block) ever changes in nfl_data_pull.py / nfl_matchup_sheet.py,
# update the matching copy here too -- see the comments marking each one.
#
# Usage:
#   python nfl_historical_backtest.py
#       -> tests the 4 most recent real completed seasons
#   python nfl_historical_backtest.py --seasons 2019,2020,2021,2022,2023,2024
#       -> tests exactly the real seasons you list
# ==============================================================================

import argparse
import datetime
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Backtest the real stat-weighted win prediction model against real past NFL seasons."
)
parser.add_argument(
    "--seasons", type=str, default=None,
    help="Comma-separated real season years to test, e.g. 2020,2021,2022,2023,2024. "
         "Default: the 4 most recent completed seasons.",
)
args = parser.parse_args()

if args.seasons:
    SEASONS = sorted({int(y.strip()) for y in args.seasons.split(",") if y.strip()})
else:
    # Same HISTORY_WINDOW_YEARS=4 convention nfl_data_pull.py already uses
    # for its own historical reference pools -- the CURRENT year is likely
    # mid-season (incomplete), so default to the 4 years before it.
    _current_year = datetime.date.today().year
    SEASONS = [y for y in range(_current_year - 4, _current_year) if y >= 2016]

print(f"Backtesting the real stat-weighted model against real seasons: {SEASONS}\n")

# ==============================================================================
# PER-SEASON REAL PER-GAME STAT LOG
# Copied verbatim from nfl_data_pull.py PART 3B (base per-game frame) and
# PART 3C (extended per-game stats: EPA/play, success rate, 3rd down %,
# sack %, pressure rate, completion %, interceptions) -- same _per_game
# helper, same column formulas, just looped across several real past
# seasons instead of the current in-progress one.
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
        'rush_yards', 'pass_yards', 'rush_yards_allowed', 'pass_yards_allowed',
        'off_epa_play_game', 'def_epa_play_allowed_game',
        'off_success_rate_game', 'def_success_rate_allowed_game',
        'third_down_pct_game', 'third_down_pct_allowed_game',
        'sack_pct_offense_game', 'sack_pct_defense_game',
        'pressure_rate_allowed_game', 'pressure_rate_forced_game',
        'completion_pct_game', 'interceptions_thrown_game', 'interceptions_gained_game',
    ]].copy()
    season_log['season'] = yr
    # game_id is only guaranteed unique WITHIN a season -- prefix with the
    # real season year so record_nfl_completed_games' game-pairing key never
    # collides two different real seasons' games that happen to share an id.
    season_log['game_id'] = season_log['season'].astype(str) + "_" + season_log['game_id'].astype(str)
    all_game_logs.append(season_log)
    print(f"    -> {yr}: {len(season_log)} real team-game row(s), {season_log['game_id'].nunique()} real game(s).")

if not all_game_logs:
    raise SystemExit("No real seasons could be loaded -- nothing to backtest.")

historical_log = pd.concat(all_game_logs, ignore_index=True).sort_values(['team', 'season', 'week']).reset_index(drop=True)
print(f"\nTotal real historical team-game rows: {len(historical_log)} across {len(SEASONS)} real season(s).\n")

# ==============================================================================
# BACKTEST / WEIGHT-TUNING MODEL
# Copied verbatim from nfl_matchup_sheet.py's "STAT-WEIGHTED WIN PREDICTION
# MODEL" block: NFL_BACKTESTABLE_STATS, _nfl_stat_vote, record_nfl_completed_
# games, _nfl_individual_stat_win_rates, _nfl_backtest_accuracy, and
# tune_nfl_stat_weights (same greedy forward-selection, no early stop at any
# fixed accuracy target -- it only stops when nothing left improves it).
# If that block ever changes in nfl_matchup_sheet.py, update this copy too.
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


def _nfl_individual_stat_win_rates(history):
    tallies = {}
    for game in history["games"].values():
        for key, vote in game.get("votes", {}).items():
            t = tallies.setdefault(key, [0, 0])
            t[1] += 1
            if vote == 1:
                t[0] += 1
    return {k: (c / n, n) for k, (c, n) in tallies.items() if n >= 3}


def _nfl_backtest_accuracy(history, included_keys, weights):
    correct, decided, total = 0, 0, 0
    for game in history["games"].values():
        total += 1
        votes = game.get("votes", {})
        score = sum(weights.get(k, 0) * votes[k] for k in included_keys if k in votes)
        if score == 0:
            continue
        decided += 1
        if score > 0:
            correct += 1
    return correct, decided, total


def tune_nfl_stat_weights(history):
    win_rates = _nfl_individual_stat_win_rates(history)
    total_games = len(history["games"])
    label_by_key = {game_col: label for label, game_col, _, _, _ in NFL_BACKTESTABLE_STATS}

    candidates = {k: wr for k, (wr, n) in win_rates.items() if wr > 0.5}
    weights = {k: (win_rates[k][0] - 0.5) for k in candidates}

    included_keys = []
    best_accuracy, best_correct, best_decided = 0.0, 0, 0
    remaining = set(candidates.keys())
    min_decided = max(1, int(round(total_games * 0.6))) if total_games else 0

    while remaining:
        best_add, best_add_result = None, None
        for k in remaining:
            trial_keys = included_keys + [k]
            correct, decided, _total = _nfl_backtest_accuracy(history, trial_keys, weights)
            if decided < min_decided:
                continue
            acc = correct / decided if decided else 0.0
            if best_add_result is None or acc > best_add_result[0]:
                best_add, best_add_result = k, (acc, correct, decided)
        if best_add is None:
            break
        acc, correct, decided = best_add_result
        if acc <= best_accuracy and included_keys:
            break
        included_keys.append(best_add)
        remaining.discard(best_add)
        best_accuracy, best_correct, best_decided = acc, correct, decided

    included = [
        (label_by_key.get(k, k), k, weights[k], win_rates[k][0], win_rates[k][1])
        for k in included_keys
    ]
    return included, best_accuracy, best_correct, best_decided, total_games


# ==============================================================================
# RUN THE BACKTEST AND PRINT RESULTS
# ==============================================================================

history = {"games": {}}
n_recorded = record_nfl_completed_games(history, historical_log)
print(f"Recorded {n_recorded} real completed game(s) into this backtest ({len(history['games'])} total).\n")

included, accuracy, correct, decided, total = tune_nfl_stat_weights(history)

print("=" * 78)
print(f"REAL HISTORICAL BACKTEST -- seasons {', '.join(str(s) for s in SEASONS)} "
      f"({total} real completed games)")
print("=" * 78)
if not included:
    print("No stat cleared the model's inclusion bar on this real sample -- try more seasons with --seasons.")
else:
    print(f"Combined tuned model accuracy: {correct}/{decided} decided games = {accuracy*100:.1f}%")
    undecided = total - decided
    if undecided:
        print(f"({undecided} of {total} real games had a net-zero stat lean across the included stats and were "
              f"left undecided -- not counted as right or wrong.)")
    print("\nStats the model kept, in the order it greedily added them (each one already improving on the last,"
          " stopped only when nothing left helped further):")
    for label, game_col, weight, win_rate, n in included:
        print(f"  {label:32s}  real win rate {win_rate*100:5.1f}%  (n={n:4d} real games)   weight={weight:+.3f}")

print("\nEvery real backtestable stat's OWN individual win rate (n=3+ real games) -- "
      "ranked highest first:")
all_rates = _nfl_individual_stat_win_rates(history)
label_by_key = {game_col: label for label, game_col, _, _, _ in NFL_BACKTESTABLE_STATS}
included_keys_set = {k for _, k, *_ in included}
for key, (wr, n) in sorted(all_rates.items(), key=lambda kv: -kv[1][0]):
    tag = "  <- IN THE TUNED MODEL" if key in included_keys_set else ""
    print(f"  {label_by_key.get(key, key):32s}  real win rate {wr*100:5.1f}%  (n={n:4d} real games){tag}")

print("\nNote: VOID, Pass Block Efficiency/Pass Rush Productivity, Havoc Rate, Red Zone %, Passer Rating, CPOE, and "
      "Turnover Margin are NOT in this backtest -- they're not in the real per-game log yet (same PART 3C scope "
      "limit documented in README.txt), so there's no real per-game history to test them against. This only tests "
      "the stats that already have real per-game numbers: rush/pass yards, EPA/play, success rate, 3rd down %, "
      "sack %, pressure rate, completion %, and interceptions.")
