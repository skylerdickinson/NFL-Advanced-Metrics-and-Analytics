# ==============================================================================
# nfl_sos_backtest.py
# ==============================================================================
# ON-DEMAND TOOL -- not part of the weekly routine. Direct follow-up to your
# SOS ask:
#
#   "can we create a sos that is based off the data we have already... like
#   label each team off and def with rank based off our data and create a
#   sos and factor that into our stats and predictions"
#   "but we can use the sos and factor that into games already played and
#   how that effects the teams currents stats correct, and it should alter
#   their stats to be a lil closer to actual"
#
# nfl_matchup_sheet.py now computes a REAL composite OFFENSE/DEFENSE power
# rank (see its "COMPOSITE OFFENSE/DEFENSE POWER RANK + STRENGTH OF
# SCHEDULE" section) from the same 15 real NFL_QUALITY_STATS the weekly
# badge already uses, plus a real opponent-adjustment (a team's composite +
# the average real strength of the opponents it's actually played) meant to
# correct exactly the kind of bias you flagged in the CAR/ATL example --
# ATL's defensive number looking good partly because Pittsburgh's offense
# had a quiet day, CAR's looking bad partly because Chicago's had a huge
# one. That live version is a REFERENCE LAYER ONLY right now -- real ranks
# on the PDF, real adjusted numbers computed, but NOT fed into
# compute_nfl_quality_consensus/this week's predictions.
#
# This script is the proof step before that changes. It builds the REAL
# WALK-FORWARD (causally valid -- only real games already played, same
# "only knowable information going into that Sunday" rule as
# nfl_walkforward_backtest.py) version of the same composite + SOS
# adjustment, across real historical completed seasons, and asks the
# honest question: does the SOS-ADJUSTED composite predict the real winner
# more often than the RAW (unadjusted) composite alone? If not, the
# adjustment doesn't belong in the live badge no matter how intuitive it
# sounds -- same "prove it before trusting it" standard every other stat in
# this project has been held to.
#
# Data + methodology: reuses nfl_quality_stat_matchup_analysis.py's real
# per-season pull + walk-forward pre-game-rolling-average transform
# VERBATIM (same 15 real per-game columns behind the live NFL_QUALITY_STATS
# badge -- see that file for the full per-stat real formulas). Keep this
# file's copied block in sync if that one changes.
#
# Usage:
#   python nfl_sos_backtest.py
#       -> tests the 4 most recent real completed seasons
#   python nfl_sos_backtest.py --seasons 2019,2020,2021,2022,2023,2024
# ==============================================================================

import argparse
import datetime
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Does a real SOS-adjusted composite offense/defense score predict real game winners better than the raw composite alone?"
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
    _current_year = datetime.date.today().year
    SEASONS = [y for y in range(_current_year - 4, _current_year) if y >= 2016]

print(f"Real SOS-adjustment backtest against real seasons: {SEASONS}\n")

BACKTESTABLE_COLS = [
    'rush_yards', 'rush_yards_allowed', 'pass_yards', 'pass_yards_allowed',
    'off_epa_play_game', 'def_epa_play_allowed_game',
    'off_success_rate_game', 'def_success_rate_allowed_game',
    'third_down_pct_game', 'third_down_pct_allowed_game',
    'sack_pct_offense_game', 'sack_pct_defense_game',
    'pressure_rate_allowed_game', 'pressure_rate_forced_game',
    'completion_pct_game', 'interceptions_thrown_game', 'interceptions_gained_game',
    'off_run_success_rate_game', 'pbe_score_game', 'def_rush_ypp_allowed_game',
]

# ==============================================================================
# PER-SEASON REAL PER-GAME STAT LOG -- copied verbatim from
# nfl_quality_stat_matchup_analysis.py (itself copied verbatim from
# nfl_walkforward_backtest.py / nfl_data_pull.py PART 3B/3C). See those
# files for the full explanatory comments on each step. Keep in sync if any
# of those change.
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

    run_plays_clean = pbp_clean[pbp_clean['play_type'] == 'run']
    off_run_sr_g = run_plays_clean.groupby(['posteam', 'game_id'])['is_success'].mean().reset_index().rename(
        columns={'posteam': 'team', 'is_success': 'off_run_success_rate_game'})
    off_run_sr_g['off_run_success_rate_game'] *= 100
    pressure_off_g['pbe_score_game'] = np.where(
        pressure_off_g['_drop'] > 0,
        100 - ((pressure_off_g['_sacks'] + 0.75 * pressure_off_g['_hits']) / pressure_off_g['_drop'] * 100), np.nan)
    def_rush_ypp_g = run_plays_clean.groupby(['defteam', 'game_id'])['yards_gained'].agg(['sum', 'count']).reset_index().rename(
        columns={'defteam': 'team', 'sum': '_rush_yds_allowed', 'count': '_rush_n_allowed'})
    def_rush_ypp_g['def_rush_ypp_allowed_game'] = np.where(
        def_rush_ypp_g['_rush_n_allowed'] > 0, def_rush_ypp_g['_rush_yds_allowed'] / def_rush_ypp_g['_rush_n_allowed'], np.nan)

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
        (off_run_sr_g, ['off_run_success_rate_game']),
        (pressure_off_g[['team', 'game_id', 'pbe_score_game']], ['pbe_score_game']),
        (def_rush_ypp_g[['team', 'game_id', 'def_rush_ypp_allowed_game']], ['def_rush_ypp_allowed_game']),
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
    raise SystemExit("No real seasons could be loaded -- nothing to analyze.")

historical_log = pd.concat(all_game_logs, ignore_index=True).sort_values(['team', 'season', 'week']).reset_index(drop=True)
print(f"\nTotal real historical team-game rows: {len(historical_log)} across {len(SEASONS)} real season(s).\n")

# ==============================================================================
# WALK-FORWARD TRANSFORM -- copied verbatim from nfl_quality_stat_matchup_
# analysis.py / nfl_walkforward_backtest.py. Each team-game's own box-score
# stat is replaced with that team's real rolling average from games played
# BEFORE this week, same season only, 2+ real prior games required.
# ==============================================================================
print("Building real pre-game rolling averages (2+ prior real games required, per team per season)...")
prewk_log = historical_log.copy().sort_values(['team', 'season', 'week'])
for col in BACKTESTABLE_COLS:
    prewk_log[col] = (
        prewk_log.groupby(['team', 'season'])[col]
        .transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
    )

# ==============================================================================
# REAL COMPOSITE OFFENSE/DEFENSE SCORE, WALK-FORWARD VERSION -- the same 15
# real NFL_QUALITY_STATS stats, split by side of ball, same win_rate-0.5
# weighting as nfl_matchup_sheet.py's live compute_composite_side_scores.
# The ONLY difference from the live version: instead of z-scoring once
# against the CURRENT full-season team_matrix, this z-scores EACH REAL
# (season, week) row separately, against every other team's OWN pre-game
# rolling number at that exact same real point in the season -- so the
# composite itself never uses future information either.
# (label, game_col, higher_is_better, real_win_rate)
# ==============================================================================
GAME_OFFENSE_STATS = [
    ("Success Rate (Offense)", "off_success_rate_game", True, 0.634),
    ("EPA/Play (Offense)", "off_epa_play_game", True, 0.610),
    ("Pass Yards", "pass_yards", True, 0.594),
    ("Run Success Rate (Offense)", "off_run_success_rate_game", True, 0.592),
    ("Sack % Taken", "sack_pct_offense_game", False, 0.591),
    ("Completion %", "completion_pct_game", True, 0.588),
    ("Pass Block Efficiency (PBE)", "pbe_score_game", True, 0.582),
    ("Pressure Rate Allowed", "pressure_rate_allowed_game", False, 0.578),
    ("Rush Yards", "rush_yards", True, 0.553),
    ("3rd Down %", "third_down_pct_game", True, 0.553),
]
GAME_DEFENSE_STATS = [
    ("Interceptions Gained", "interceptions_gained_game", True, 0.571),
    ("EPA/Play Allowed (Defense)", "def_epa_play_allowed_game", False, 0.559),
    ("Rush Yards Allowed", "rush_yards_allowed", False, 0.559),
    ("Rush Yards Per Play Allowed (Defense)", "def_rush_ypp_allowed_game", False, 0.559),
    ("Pressure Rate Forced", "pressure_rate_forced_game", True, 0.557),
]
assert len(GAME_OFFENSE_STATS) == 10 and len(GAME_DEFENSE_STATS) == 5


def compute_weekly_composite(df_in, stat_list, out_col):
    """Real weighted z-score composite, computed separately within each
    real (season, week) group -- see the note above. Returns df_in with
    out_col added (NaN wherever no real stat had enough real spread/data
    that week to compute anything, honestly, not guessed)."""
    d = df_in.copy()
    score_sum = pd.Series(0.0, index=d.index)
    weight_used = pd.Series(0.0, index=d.index)
    for label, game_col, higher_is_better, win_rate in stat_list:
        if game_col not in d.columns:
            continue
        weight = win_rate - 0.5
        grp = d.groupby(['season', 'week'])[game_col]
        mean = grp.transform('mean')
        std = grp.transform('std')
        z = (d[game_col] - mean) / std
        if not higher_is_better:
            z = -z
        valid = z.notna() & std.notna() & (std != 0)
        score_sum = score_sum.where(~valid, score_sum + (z * weight).fillna(0))
        weight_used = weight_used.where(~valid, weight_used + weight)
    d[out_col] = np.where(weight_used > 0, score_sum / weight_used, np.nan)
    return d


prewk_log = compute_weekly_composite(prewk_log, GAME_OFFENSE_STATS, 'off_composite_prewk')
prewk_log = compute_weekly_composite(prewk_log, GAME_DEFENSE_STATS, 'def_composite_prewk')

# ==============================================================================
# REAL OPPONENT COMPOSITES -- self-merge on game_id to attach each row's
# REAL opponent's own off/def composite (as of that same real week) onto
# this row. One real game = exactly 2 rows sharing a game_id, so matching
# on (game_id, opponent==the other row's team) always finds exactly one.
# ==============================================================================
_opp_lookup = prewk_log[['game_id', 'team', 'off_composite_prewk', 'def_composite_prewk']].rename(
    columns={'team': 'opponent', 'off_composite_prewk': 'opp_off_composite_prewk',
             'def_composite_prewk': 'opp_def_composite_prewk'})
prewk_log = prewk_log.merge(_opp_lookup, on=['game_id', 'opponent'], how='left')

# ==============================================================================
# REAL WALK-FORWARD STRENGTH OF SCHEDULE -- for each team-game row,
# OFF_SOS/DEF_SOS = the average REAL opponent composite from every PRIOR
# real game this season (shift(1) + expanding, same pattern as every other
# rolling stat above -- this week's own opponent is NOT included, only
# opponents already played). A team with no real prior opponent yet gets
# NaN (honestly excluded downstream, not defaulted to 0 the way the live
# version does -- here we want the backtest to only compare adjusted vs raw
# on the SAME real games where an adjustment was actually possible).
# ==============================================================================
prewk_log = prewk_log.sort_values(['team', 'season', 'week'])
prewk_log['off_sos_prewk'] = (
    prewk_log.groupby(['team', 'season'])['opp_def_composite_prewk']
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)
prewk_log['def_sos_prewk'] = (
    prewk_log.groupby(['team', 'season'])['opp_off_composite_prewk']
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)
prewk_log['adj_off_composite_prewk'] = prewk_log['off_composite_prewk'] + prewk_log['off_sos_prewk']
prewk_log['adj_def_composite_prewk'] = prewk_log['def_composite_prewk'] + prewk_log['def_sos_prewk']

# Real opponent's ADJUSTED composite too, for the NET matchup-edge test below.
_opp_adj_lookup = prewk_log[['game_id', 'team', 'adj_off_composite_prewk', 'adj_def_composite_prewk']].rename(
    columns={'team': 'opponent', 'adj_off_composite_prewk': 'opp_adj_off_composite_prewk',
             'adj_def_composite_prewk': 'opp_adj_def_composite_prewk'})
prewk_log = prewk_log.merge(_opp_adj_lookup, on=['game_id', 'opponent'], how='left')


def build_completed_games(df_in):
    """Pairs up every real completed game (clean W/L both sides, ties
    excluded -- same convention as every other backtest script here).
    Returns one dict per real game with both teams' raw/adjusted rows."""
    grouped = {}
    for _, row in df_in.iterrows():
        grouped.setdefault(row['game_id'], []).append(row)
    games = []
    for gid, rows in grouped.items():
        if len(rows) != 2:
            continue
        r0, r1 = rows[0], rows[1]
        if r0['result'] not in ('W', 'L') or r1['result'] not in ('W', 'L') or r0['result'] == r1['result']:
            continue
        winner_row, loser_row = (r0, r1) if r0['result'] == 'W' else (r1, r0)
        games.append((winner_row, loser_row))
    return games


def backtest_head_to_head(games, col_fn, label):
    """Real walk-forward win rate for one stat: whichever team col_fn gives
    the higher value to is favored, checked against the REAL winner. Games
    where either side's value is missing are honestly excluded, not
    guessed."""
    n_decided, n_correct = 0, 0
    for winner_row, loser_row in games:
        wv, lv = col_fn(winner_row), col_fn(loser_row)
        if wv is None or lv is None or pd.isna(wv) or pd.isna(lv) or wv == lv:
            continue
        n_decided += 1
        if wv > lv:
            n_correct += 1
    if n_decided == 0:
        print(f"  {label:55s}   no real decided games -- can't score")
        return None
    win_rate = n_correct / n_decided * 100
    print(f"  {label:55s} {win_rate:6.1f}%   (n={n_decided} real games)")
    return win_rate


all_games = build_completed_games(prewk_log)
print(f"\n{len(all_games)} real completed games with a clean W/L pairing found across {len(SEASONS)} season(s).\n")

print("=" * 78)
print("DOES THE REAL SOS-ADJUSTED COMPOSITE PREDICT BETTER THAN THE RAW ONE?")
print("=" * 78)
print("\nOFFENSE side (higher composite = favored):")
raw_off_wr = backtest_head_to_head(all_games, lambda r: r['off_composite_prewk'], "  Raw OFF_COMPOSITE (no SOS adjustment)")
adj_off_wr = backtest_head_to_head(all_games, lambda r: r['adj_off_composite_prewk'], "  SOS-ADJUSTED OFF_COMPOSITE")

print("\nDEFENSE side (higher composite = favored, i.e. that team's defense picked to 'win' the individual battle):")
raw_def_wr = backtest_head_to_head(all_games, lambda r: r['def_composite_prewk'], "  Raw DEF_COMPOSITE (no SOS adjustment)")
adj_def_wr = backtest_head_to_head(all_games, lambda r: r['adj_def_composite_prewk'], "  SOS-ADJUSTED DEF_COMPOSITE")

print("\nNET MATCHUP EDGE (this team's offense composite minus the REAL opponent's")
print("defense composite -- the game-relevant version, same shape as this sheet's")
print("existing NET EPA (MATCHUP) row):")
prewk_log['raw_net_edge'] = prewk_log['off_composite_prewk'] - prewk_log['opp_def_composite_prewk']
prewk_log['adj_net_edge'] = prewk_log['adj_off_composite_prewk'] - prewk_log['opp_adj_def_composite_prewk']
all_games = build_completed_games(prewk_log)  # rebuild so the two new columns are in each row
raw_net_wr = backtest_head_to_head(all_games, lambda r: r['raw_net_edge'], "  Raw NET EDGE (no SOS adjustment)")
adj_net_wr = backtest_head_to_head(all_games, lambda r: r['adj_net_edge'], "  SOS-ADJUSTED NET EDGE")

print("\n" + "=" * 78)
print("HONEST READ")
print("=" * 78)
for side, raw, adj in [("OFFENSE", raw_off_wr, adj_off_wr), ("DEFENSE", raw_def_wr, adj_def_wr), ("NET EDGE", raw_net_wr, adj_net_wr)]:
    if raw is None or adj is None:
        print(f"  {side}: not enough real decided games to compare.")
    elif adj > raw + 1.0:
        print(f"  {side}: SOS-adjustment REAL improvement -- {adj:.1f}% adjusted vs {raw:.1f}% raw (+{adj-raw:.1f} pts).")
    elif raw > adj + 1.0:
        print(f"  {side}: SOS-adjustment REAL step BACKWARD -- {adj:.1f}% adjusted vs {raw:.1f}% raw ({adj-raw:+.1f} pts). Don't ship this side's adjustment.")
    else:
        print(f"  {side}: no real meaningful difference -- {adj:.1f}% adjusted vs {raw:.1f}% raw ({adj-raw:+.1f} pts). Adjustment isn't earning its complexity here.")

print("\nHonest sample-size note: the adjusted columns need BOTH 2+ real prior")
print("games for a team's own composite AND 1+ real prior opponent whose own")
print("composite was already computable -- that cascades, so the adjusted")
print("backtest's real usable sample starts later in the season and can be")
print("meaningfully smaller than the raw backtest's. Sample sizes are printed")
print("above per row -- check them before trusting a small swing either way.")
