# ==============================================================================
# nfl_stat_gap_analysis.py
# ==============================================================================
# ON-DEMAND TOOL -- not part of the weekly routine. Companion to
# nfl_walkforward_backtest.py, answering a follow-up question about 4 of its
# WEAK stats specifically: Run Stop Rate (Defense), Pass Rush Productivity
# (PRP), Success Rate Allowed (Defense), and Pass Yards Allowed.
#
# The question: if a def run stop can show favor to the D but still lose, is
# there a gap % where the number is actually true and really does lean D,
# backed by winning %? A small gap will always lean O -- at what point does
# it become an actual D win %?
#
# nfl_walkforward_backtest.py reports ONE flat win rate per stat (e.g. Run
# Stop Rate = 53.9%), averaged across every real game regardless of HOW BIG
# the real gap was between the two teams' pre-game numbers. That flat number
# could be hiding exactly what you're describing: games where the two teams'
# real values were nearly identical (a real coin flip, correctly noisy) get
# averaged together with games where one team's real number was miles ahead
# of the other's. If the big-gap games actually win a lot more than the flat
# average, that's the real separation point you're asking about.
#
# This script takes each of those 4 stats, computes the REAL walk-forward
# pre-game gap between the two teams for every real game (same "2+ real
# prior games, no leakage" rule as nfl_walkforward_backtest.py), expresses
# that gap as a real symmetric percent difference, splits games into
# quantile buckets by that gap size, and reports the real win rate of the
# team the stat favored, bucket by bucket -- so you can see, in real
# numbers, whether there's an actual gap-size threshold where the stat
# starts meaning something.
#
# Nothing here is faked: the real per-season pull and walk-forward
# transform below (through building prewk_log) are copied verbatim from
# nfl_walkforward_backtest.py. Keep this in sync with that file if its pull
# logic ever changes.
#
# Usage:
#   python nfl_stat_gap_analysis.py
#       -> tests the 4 most recent real completed seasons
#   python nfl_stat_gap_analysis.py --seasons 2019,2020,2021,2022,2023,2024
# ==============================================================================

import argparse
import datetime
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Does a bigger real pre-game gap on a weak defensive stat actually predict wins better?"
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

print(f"Real stat-gap analysis against real seasons: {SEASONS}\n")

BACKTESTABLE_COLS = [
    'rush_yards', 'rush_yards_allowed', 'pass_yards', 'pass_yards_allowed',
    'off_epa_play_game', 'def_epa_play_allowed_game',
    'off_success_rate_game', 'def_success_rate_allowed_game',
    'third_down_pct_game', 'third_down_pct_allowed_game',
    'sack_pct_offense_game', 'sack_pct_defense_game',
    'pressure_rate_allowed_game', 'pressure_rate_forced_game',
    'completion_pct_game', 'interceptions_thrown_game', 'interceptions_gained_game',
    'off_run_success_rate_game', 'def_run_stop_rate_game',
    'pbe_score_game', 'prp_score_game',
]

# ==============================================================================
# PER-SEASON REAL PER-GAME STAT LOG -- copied verbatim from
# nfl_walkforward_backtest.py (itself copied from nfl_data_pull.py PART
# 3B/3C). See that file for the full explanatory comments.
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

    run_plays_clean = pbp_clean[pbp_clean['play_type'] == 'run']
    off_run_sr_g = run_plays_clean.groupby(['posteam', 'game_id'])['is_success'].mean().reset_index().rename(
        columns={'posteam': 'team', 'is_success': 'off_run_success_rate_game'})
    off_run_sr_g['off_run_success_rate_game'] *= 100
    def_run_sr_g = run_plays_clean.groupby(['defteam', 'game_id'])['is_success'].mean().reset_index().rename(
        columns={'defteam': 'team', 'is_success': 'def_run_stop_rate_game'})
    def_run_sr_g['def_run_stop_rate_game'] = (1 - def_run_sr_g['def_run_stop_rate_game']) * 100

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

    pressure_off_g['pbe_score_game'] = np.where(
        pressure_off_g['_drop'] > 0,
        100 - ((pressure_off_g['_sacks'] + 0.75 * pressure_off_g['_hits']) / pressure_off_g['_drop'] * 100), np.nan)
    pressure_def_g['prp_score_game'] = np.where(
        pressure_def_g['_drop'] > 0,
        (pressure_def_g['_sacks'] + 0.75 * pressure_def_g['_hits']) / pressure_def_g['_drop'] * 100, np.nan)

    comp_g = pbp_clean[pbp_clean['play_type'] == 'pass'].groupby(['posteam', 'game_id']).agg(
        _comp=('complete_pass', 'sum'), _att=('pass_attempt', 'sum'), _int=('interception', 'sum')).reset_index().rename(columns={'posteam': 'team'})
    comp_g['completion_pct_game'] = np.where(comp_g['_att'] > 0, comp_g['_comp'] / comp_g['_att'] * 100, np.nan)
    comp_g = comp_g.rename(columns={'_int': 'interceptions_thrown_game'})
    int_def_g = pbp_clean[pbp_clean['play_type'] == 'pass'].groupby(['defteam', 'game_id'])['interception'].sum().reset_index().rename(
        columns={'defteam': 'team', 'interception': 'interceptions_gained_game'})

    for extra_df, cols in [
        (off_epa_g, ['off_epa_play_game']), (def_epa_g, ['def_epa_play_allowed_game']),
        (off_sr_g, ['off_success_rate_game']), (def_sr_g, ['def_success_rate_allowed_game']),
        (off_run_sr_g, ['off_run_success_rate_game']), (def_run_sr_g, ['def_run_stop_rate_game']),
        (off_3rd_g, ['third_down_pct_game']), (def_3rd_g, ['third_down_pct_allowed_game']),
        (sack_off_g[['team', 'game_id', 'sack_pct_offense_game']], ['sack_pct_offense_game']),
        (sack_def_g[['team', 'game_id', 'sack_pct_defense_game']], ['sack_pct_defense_game']),
        (pressure_off_g[['team', 'game_id', 'pressure_rate_allowed_game']], ['pressure_rate_allowed_game']),
        (pressure_def_g[['team', 'game_id', 'pressure_rate_forced_game']], ['pressure_rate_forced_game']),
        (pressure_off_g[['team', 'game_id', 'pbe_score_game']], ['pbe_score_game']),
        (pressure_def_g[['team', 'game_id', 'prp_score_game']], ['prp_score_game']),
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
    raise SystemExit("No real seasons could be loaded -- nothing to analyze.")

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

n_total_rows = len(prewk_log)
n_burned_in = prewk_log[BACKTESTABLE_COLS].notna().any(axis=1).sum()
print(f" -> {n_total_rows - n_burned_in} of {n_total_rows} team-game row(s) had fewer than 2 real prior games this "
      f"season (no pre-game number possible yet -- correctly excluded, not guessed); {n_burned_in} have at least "
      f"one usable pre-game stat.\n")

# ==============================================================================
# NEW: real gap-bucket analysis. For a given stat, pairs every real
# completed game, computes the real symmetric percent gap between the two
# teams' pre-game numbers, and buckets games by that gap size (quantiles,
# so every bucket gets a similar real sample size regardless of how the
# real gap distribution is shaped).
# ==============================================================================

# The 4 weak/defensive stats, plus their real direction
# (matches NFL_BACKTESTABLE_STATS in nfl_walkforward_backtest.py exactly).
TARGET_STATS = [
    ("Run Stop Rate (Defense)", "def_run_stop_rate_game", True),
    ("Pass Rush Productivity (PRP)", "prp_score_game", True),
    ("Success Rate Allowed (Defense)", "def_success_rate_allowed_game", False),
    ("Pass Yards Allowed", "pass_yards_allowed", False),
]

N_BUCKETS = 5


def build_gap_games(prewk_df, game_col, higher_is_better):
    """Pairs every real completed game (result W/L only, byes/ties/missing
    excluded) and returns one record per game: which team the stat favored
    (by real pre-game value), whether that team actually won, and the real
    symmetric percent gap between the two teams' real pre-game numbers.
    Games with a missing pre-game value on either side, or an exact tie
    (no real lean either way), are honestly excluded, not guessed."""
    records = []
    for game_id, grp in prewk_df.groupby('game_id'):
        grp = grp[grp['result'].isin(['W', 'L'])]
        if len(grp) != 2:
            continue
        r0, r1 = grp.iloc[0], grp.iloc[1]
        v0, v1 = r0.get(game_col), r1.get(game_col)
        if v0 is None or v1 is None or pd.isna(v0) or pd.isna(v1):
            continue
        v0, v1 = float(v0), float(v1)
        if v0 == v1:
            continue
        r0_favored = (v0 > v1) if higher_is_better else (v0 < v1)
        favored_row, other_row = (r0, r1) if r0_favored else (r1, r0)
        favored_val, other_val = (v0, v1) if r0_favored else (v1, v0)
        denom = (abs(favored_val) + abs(other_val)) / 2.0
        if denom == 0:
            continue
        gap_pct = abs(favored_val - other_val) / denom * 100.0
        records.append({
            "game_id": game_id,
            "favored_team": favored_row['team'],
            "other_team": other_row['team'],
            "favored_won": favored_row['result'] == 'W',
            "gap_pct": gap_pct,
        })
    return records


def report_gap_buckets(games, label, n_buckets=N_BUCKETS):
    if not games:
        print(f"{label}: no real usable games -- skipping.")
        return
    df = pd.DataFrame(games).sort_values('gap_pct').reset_index(drop=True)
    n = len(df)
    overall_win_rate = df['favored_won'].mean() * 100
    print(f"\n--- {label} ---")
    print(f"Real games with a usable pre-game gap: {n}")
    print(f"Real overall win rate (flat, same as nfl_walkforward_backtest.py): {overall_win_rate:.1f}%\n")

    # Split into n_buckets equal-SIZE groups by rank (not pd.qcut, which can
    # collapse buckets when many games share the same real gap value) --
    # every bucket gets essentially the same real sample size regardless of
    # how the real gap distribution is shaped.
    bucket_indices = np.array_split(np.arange(n), n_buckets)
    print(f"{'Bucket':<8}{'Real gap % range':<24}{'n':>6}{'Win rate':>12}")
    prior_win_rate = None
    crossover_note = None
    for i, idx in enumerate(bucket_indices):
        if len(idx) == 0:
            continue
        bucket_df = df.iloc[idx]
        lo, hi = bucket_df['gap_pct'].min(), bucket_df['gap_pct'].max()
        wr = bucket_df['favored_won'].mean() * 100
        bn = len(bucket_df)
        print(f"{i + 1:<8}{f'{lo:.1f}% - {hi:.1f}%':<24}{bn:>6}{wr:>11.1f}%")
        if prior_win_rate is not None and crossover_note is None and wr >= overall_win_rate + 5 and wr >= 58:
            crossover_note = f"real win rate first clears {wr:.1f}% (5+ pts above the flat {overall_win_rate:.1f}% average) at a real gap of {lo:.1f}%+"
        prior_win_rate = wr

    if crossover_note:
        print(f"-> {crossover_note}.")
    else:
        print(f"-> no bucket cleared the flat {overall_win_rate:.1f}% average by 5+ real points -- this stat "
              f"doesn't show a real gap-size threshold where it becomes meaningfully more reliable; it stays "
              f"close to a coin flip even at its largest real gaps.")


print("==============================================================================")
print("REAL GAP-SIZE ANALYSIS -- does a bigger real pre-game gap actually predict wins")
print("better, for the 4 weak/defensive stats specifically?")
print("==============================================================================")
for label, game_col, higher_is_better in TARGET_STATS:
    games = build_gap_games(prewk_log, game_col, higher_is_better)
    report_gap_buckets(games, label)

print("\nHonest note: buckets are split by real RANK (equal real sample size per "
      "bucket), not fixed round gap percentages, since each stat's real gap "
      "distribution is a different shape. The printed range for each bucket is "
      "its own real min-max gap %, so you can see exactly what gap size each "
      "bucket actually covers.")
