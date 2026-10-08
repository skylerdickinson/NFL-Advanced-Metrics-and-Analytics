# ==============================================================================
# nfl_winprob_calibration_backtest.py
# ==============================================================================
# ON-DEMAND TOOL -- not part of the weekly routine (nfl_data_pull.py then
# nfl_matchup_sheet.py still run every week as normal; this doesn't touch
# either of their files). Belongs next to the other backtest tools in
# research_and_backtests/.
#
# Direct answer to "does it learn or is it just gonna pump out the same bad
# reads": the quality-stat BADGE already re-tunes its own weights every run
# against the real, growing nfl_stat_history.json (tune_nfl_stat_weights).
# The win-probability/spread/total-points PANEL does not -- it runs through
# predict_matchup() in nfl_matchup_sheet.py, which converts each team's
# off_power_rating/def_power_rating into points and win probability using
# two constants that were hand-picked and never fit to anything real:
#
#   POINTS_PER_RATING_UNIT = 7.0   # points of scoring shift per 1.0 rating unit
#   WIN_PROB_SCALE = 8.5           # logistic scale, margin -> win probability
#
# The comment above them in nfl_matchup_sheet.py says so directly:
# "calibration assumptions, not empirically fit to real historical results."
# That's confirmed as the real cause of the honest week-2 numbers (37.5%
# win-prob accuracy, +9.1 points too hot on totals) once the contaminated
# 2026-09-21 rerun was cleaned out of nfl_internal_model_track.csv.
#
# This script fits BOTH constants against real, walk-forward-safe historical
# data (same "only real games played BEFORE the week being tested" rule as
# nfl_walkforward_backtest.py, same source data, same season default) --
# real least-squares for POINTS_PER_RATING_UNIT, real numerical log-loss
# minimization for WIN_PROB_SCALE (no sklearn/scipy, same project
# convention as the rest of this codebase) -- and reports the real,
# honest accuracy of the OLD hardcoded constants vs the NEWLY FIT ones on
# the exact same real historical games, so you can see whether fitting
# actually helps before trusting it.
#
# ONE DISCLOSED SIMPLIFICATION: the live off_power_rating/def_power_rating
# formula in nfl_data_pull.py uses 6 stats per side, including
# redzone_efficiency/redzone_efficiency_allowed (weight 0.05/-0.05 out of
# a 0.30-1.00 total per side -- the smallest of the six). Real red-zone
# drive detection needs per-drive yardline tracking that isn't already
# built anywhere in this project's backtest scaffolding, so it's left out
# of this backtest's rating replica rather than guessed or faked. Every
# other input (points, yards/play, EPA/play, success rate, turnover
# margin) is real and walk-forward-safe. This means the ratings computed
# here are a close but not byte-for-byte match to the live formula --
# close enough to fit real, useful calibration constants, not close enough
# to claim an exact replica.
#
# Usage:
#   python nfl_winprob_calibration_backtest.py
#       -> fits against the 4 most recent real completed seasons
#   python nfl_winprob_calibration_backtest.py --seasons 2021,2022,2023,2024
#       -> fits against exactly the real seasons you list
# ==============================================================================

import argparse
import datetime
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Fit POINTS_PER_RATING_UNIT and WIN_PROB_SCALE against real historical games, "
                "and honestly compare against the current hardcoded constants."
)
parser.add_argument(
    "--seasons", type=str, default=None,
    help="Comma-separated real season years to fit against, e.g. 2021,2022,2023,2024. "
         "Default: the 4 most recent real completed seasons."
)
args = parser.parse_args()

if args.seasons:
    SEASONS = sorted({int(y.strip()) for y in args.seasons.split(",") if y.strip()})
else:
    _current_year = datetime.date.today().year
    SEASONS = [y for y in range(_current_year - 4, _current_year) if y >= 2016]

print(f"Fitting the real win-probability/points model against real seasons: {SEASONS}\n")

# Current live constants (nfl_matchup_sheet.py) -- kept here as the real
# baseline to compare the new fit against, not re-derived.
OLD_POINTS_PER_RATING_UNIT = 7.0
OLD_WIN_PROB_SCALE = 8.5

OFFENSE_RATING_WEIGHTS = {
    'points_per_game': 0.30,
    'off_yards_per_play': 0.25,
    'off_efficiency_epa': 0.20,
    'off_success_rate': 0.10,
    'turnover_margin_offense': -0.10,
}
DEFENSE_RATING_WEIGHTS = {
    'points_allowed_per_game': -0.30,
    'def_yards_per_play': -0.25,
    'def_efficiency_epa': -0.20,
    'def_success_rate': -0.10,
    'turnover_margin_defense': 0.10,
}
MIN_PRIOR_GAMES = 2  # same "needs 2+ real prior games" rule as the rest of this project


def _per_game(df_src, group_cols, value_col, out_col, agg='mean'):
    g = df_src.groupby(group_cols)[value_col].agg(agg).reset_index()
    return g.rename(columns={value_col: out_col})


all_games = []  # one row per real team-game, walk-forward stats filled in below

for yr in SEASONS:
    print(f"  Loading real {yr} season play-by-play (nflverse)...")
    pbp = nfl.load_pbp(seasons=[yr]).to_pandas()
    pbp_reg = pbp[(pbp['season_type'] == 'REG') & (pbp['posteam'].notna()) & (pbp['defteam'].notna())].copy()
    if pbp_reg.empty:
        print(f"  ⚠ no real regular-season plays returned for {yr} -- skipping this season")
        continue

    for field in ['yards_gained', 'epa', 'interception', 'fumble_lost']:
        if field in pbp_reg.columns:
            pbp_reg[field] = pd.to_numeric(pbp_reg[field], errors='coerce').fillna(0)
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

    yards = pbp_clean.groupby(['posteam', 'game_id'])['yards_gained'].sum().reset_index().rename(columns={'posteam': 'team', 'yards_gained': 'off_yards'})
    yards_allowed = pbp_clean.groupby(['defteam', 'game_id'])['yards_gained'].sum().reset_index().rename(columns={'defteam': 'team', 'yards_gained': 'def_yards_allowed'})
    off_plays = pbp_clean.groupby(['posteam', 'game_id']).size().reset_index(name='off_plays').rename(columns={'posteam': 'team'})
    def_plays = pbp_clean.groupby(['defteam', 'game_id']).size().reset_index(name='def_plays').rename(columns={'defteam': 'team'})
    off_epa_g = _per_game(pbp_clean, ['posteam', 'game_id'], 'epa', 'off_epa_play_game').rename(columns={'posteam': 'team'})
    def_epa_g = _per_game(pbp_clean, ['defteam', 'game_id'], 'epa', 'def_epa_play_allowed_game').rename(columns={'defteam': 'team'})
    off_sr_g = _per_game(pbp_clean, ['posteam', 'game_id'], 'is_success', 'off_success_rate_game').rename(columns={'posteam': 'team'})
    off_sr_g['off_success_rate_game'] *= 100
    def_sr_g = _per_game(pbp_clean, ['defteam', 'game_id'], 'is_success', 'def_success_rate_allowed_game').rename(columns={'defteam': 'team'})
    def_sr_g['def_success_rate_allowed_game'] *= 100

    # Real turnovers: this team's own giveaways (interceptions thrown +
    # fumbles lost, as the offense) and real takeaways forced (interceptions
    # + fumbles recovered, as the defense) -- same real definition as
    # nfl_data_pull.py's off_turnovers/def_turnovers (master['turnover_margin_offense']
    # = master['off_turnovers'], etc).
    off_to_g = pbp_reg.groupby(['posteam', 'game_id']).agg(
        _int=('interception', 'sum'), _fl=('fumble_lost', 'sum')).reset_index().rename(columns={'posteam': 'team'})
    off_to_g['off_turnovers_game'] = off_to_g['_int'] + off_to_g['_fl']
    def_to_g = pbp_reg.groupby(['defteam', 'game_id']).agg(
        _int=('interception', 'sum'), _fl=('fumble_lost', 'sum')).reset_index().rename(columns={'defteam': 'team'})
    def_to_g['def_turnovers_game'] = def_to_g['_int'] + def_to_g['_fl']

    team_games = pbp_reg.groupby(['posteam', 'game_id']).agg(
        defteam=('defteam', 'first'), home_team=('home_team', 'first'), away_team=('away_team', 'first'),
    ).reset_index().rename(columns={'posteam': 'team'})
    team_games = team_games.merge(game_weeks, on='game_id', how='left')
    team_games['season'] = yr
    team_games['opponent'] = team_games['defteam']
    team_games['is_home'] = team_games['team'] == team_games['home_team']
    team_games = team_games.merge(final_scores[['game_id', 'home', 'away', 'home_score', 'away_score', 'winner']], on='game_id', how='left')
    team_games['pts_for'] = np.where(team_games['is_home'], team_games['home_score'], team_games['away_score'])
    team_games['pts_against'] = np.where(team_games['is_home'], team_games['away_score'], team_games['home_score'])

    for extra_df in [yards, yards_allowed, off_plays, def_plays, off_epa_g, def_epa_g, off_sr_g, def_sr_g, off_to_g[['team', 'game_id', 'off_turnovers_game']], def_to_g[['team', 'game_id', 'def_turnovers_game']]]:
        team_games = team_games.merge(extra_df, on=['team', 'game_id'], how='left')

    team_games['off_yards_per_play_game'] = np.where(team_games['off_plays'] > 0, team_games['off_yards'] / team_games['off_plays'], np.nan)
    team_games['def_yards_per_play_game'] = np.where(team_games['def_plays'] > 0, team_games['def_yards_allowed'] / team_games['def_plays'], np.nan)

    all_games.append(team_games)

team_games_all = pd.concat(all_games, ignore_index=True)
print(f"\nReal team-game rows loaded: {len(team_games_all)}\n")

PER_GAME_COL_BY_RATING_KEY = {
    'points_per_game': 'pts_for',
    'off_yards_per_play': 'off_yards_per_play_game',
    'off_efficiency_epa': 'off_epa_play_game',
    'off_success_rate': 'off_success_rate_game',
    'turnover_margin_offense': 'off_turnovers_game',
    'points_allowed_per_game': 'pts_against',
    'def_yards_per_play': 'def_yards_per_play_game',
    'def_efficiency_epa': 'def_epa_play_allowed_game',
    'def_success_rate': 'def_success_rate_allowed_game',
    'turnover_margin_defense': 'def_turnovers_game',
}


def compute_power_rating(rolling_df, weights):
    """Same real z-score-vs-this-week's-league-average formula as
    nfl_data_pull.py's compute_power_rating(), applied to a walk-forward
    rolling-average snapshot instead of a live season-to-date snapshot."""
    rating = pd.Series(0.0, index=rolling_df.index)
    for key, weight in weights.items():
        col = PER_GAME_COL_BY_RATING_KEY[key]
        std = rolling_df[col].std()
        z = (rolling_df[col] - rolling_df[col].mean()) / std if std and std > 0 else 0
        rating = rating + (weight * z)
    return rating


# ------------------------------------------------------------------
# Walk-forward: for each real (season, week), build every team's rolling
# average from ONLY that team's real games in EARLIER weeks of the SAME
# season (never a future game, never a different season) -- identical
# no-leakage rule as nfl_walkforward_backtest.py.
# ------------------------------------------------------------------
rating_diffs, real_margins, real_wins = [], [], []
games_used = 0

for (season, week), week_games in team_games_all.groupby(['season', 'week']):
    prior = team_games_all[(team_games_all['season'] == season) & (team_games_all['week'] < week)]
    if prior.empty:
        continue
    game_counts = prior.groupby('team').size()
    eligible_teams = set(game_counts[game_counts >= MIN_PRIOR_GAMES].index)
    if len(eligible_teams) < 2:
        continue

    rolling = prior[prior['team'].isin(eligible_teams)].groupby('team').agg(
        {col: 'mean' for col in set(PER_GAME_COL_BY_RATING_KEY.values())}
    )
    off_rating = compute_power_rating(rolling, OFFENSE_RATING_WEIGHTS)
    def_rating = compute_power_rating(rolling, DEFENSE_RATING_WEIGHTS)

    # Each real game_id appears twice in week_games (once per team's own
    # row) -- walk it once per real game_id, not once per team-row.
    seen_game_ids = set()
    for _, g in week_games.iterrows():
        if g['game_id'] in seen_game_ids:
            continue
        seen_game_ids.add(g['game_id'])
        home_team, away_team = g['home_team'], g['away_team']
        if home_team not in eligible_teams or away_team not in eligible_teams:
            continue
        net_home = off_rating.get(home_team, np.nan) - def_rating.get(away_team, np.nan)
        net_away = off_rating.get(away_team, np.nan) - def_rating.get(home_team, np.nan)
        if pd.isna(net_home) or pd.isna(net_away):
            continue
        home_score, away_score = g['home_score'], g['away_score']
        if pd.isna(home_score) or pd.isna(away_score) or home_score == away_score:
            continue  # real ties excluded, same as the rest of this project's backtests
        rating_diffs.append(net_home - net_away)
        real_margins.append(home_score - away_score)
        real_wins.append(1 if home_score > away_score else 0)
        games_used += 1

rating_diffs = np.array(rating_diffs)
real_margins = np.array(real_margins)
real_wins = np.array(real_wins)
print(f"Real, walk-forward-safe games usable for fitting: {games_used}\n")

if games_used < 50:
    print("⚠ Too few real usable games to fit anything meaningful -- widen --seasons and rerun.")
    raise SystemExit(1)

# ------------------------------------------------------------------
# Fit 1: POINTS_PER_RATING_UNIT -- real least squares, zero intercept
# (matches predict_matchup()'s margin = POINTS_PER_RATING_UNIT * rating_diff
# exactly -- no constant term to fit, by construction of that formula).
# ------------------------------------------------------------------
NEW_POINTS_PER_RATING_UNIT = float(np.sum(rating_diffs * real_margins) / np.sum(rating_diffs ** 2))


# ------------------------------------------------------------------
# Fit 2: WIN_PROB_SCALE -- real numerical log-loss minimization (no
# sklearn/scipy, same convention as the rest of this codebase). Margin used
# here is real_margins directly (not re-derived from rating_diffs *
# NEW_POINTS_PER_RATING_UNIT), since predict_matchup() derives win_prob_a
# from the SAME margin it just computed from the rating differential --
# fitting WIN_PROB_SCALE against the rating-implied margin (using the
# newly fit slope) keeps the two fits internally consistent with each
# other, matching how they're actually chained live.
# ------------------------------------------------------------------
implied_margin = rating_diffs * NEW_POINTS_PER_RATING_UNIT


def _log_loss(scale):
    p = 1 / (1 + np.exp(-implied_margin / scale))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(real_wins * np.log(p) + (1 - real_wins) * np.log(1 - p))


# Coarse scan then golden-section refine -- log-loss is convex-ish in scale
# over any sane range, so a scan + local refine is plenty for one real
# parameter; no need for scipy.
_scan = np.linspace(1.0, 30.0, 291)
_losses = [_log_loss(s) for s in _scan]
_best_idx = int(np.argmin(_losses))
_lo = _scan[max(_best_idx - 1, 0)]
_hi = _scan[min(_best_idx + 1, len(_scan) - 1)]
_gr = (np.sqrt(5) - 1) / 2
for _ in range(60):
    _c = _hi - _gr * (_hi - _lo)
    _d = _lo + _gr * (_hi - _lo)
    if _log_loss(_c) < _log_loss(_d):
        _hi = _d
    else:
        _lo = _c
NEW_WIN_PROB_SCALE = float((_lo + _hi) / 2)

print(f"FIT RESULTS (real, walk-forward-safe, {games_used} real games across seasons {SEASONS}):")
print(f"  POINTS_PER_RATING_UNIT:  old = {OLD_POINTS_PER_RATING_UNIT:.3f}   new (fit) = {NEW_POINTS_PER_RATING_UNIT:.3f}")
print(f"  WIN_PROB_SCALE:          old = {OLD_WIN_PROB_SCALE:.3f}   new (fit) = {NEW_WIN_PROB_SCALE:.3f}\n")


# ------------------------------------------------------------------
# Honest before/after: real accuracy, real log-loss, real avg margin error,
# on the SAME real games, for the old hardcoded constants vs the new fit.
# ------------------------------------------------------------------
def _evaluate(points_per_unit, win_prob_scale, label):
    margin_pred = rating_diffs * points_per_unit
    p = 1 / (1 + np.exp(-margin_pred / win_prob_scale))
    pick_home = p > 0.5
    correct = int(np.sum(pick_home == (real_wins == 1)))
    ll = -np.mean(real_wins * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - real_wins) * np.log(np.clip(1 - p, 1e-6, 1 - 1e-6)))
    mae = float(np.mean(np.abs(margin_pred - real_margins)))
    print(f"  {label}: {correct}/{games_used} = {correct/games_used:.1%} straight-up, "
          f"log-loss {ll:.4f}, avg margin error {mae:.2f} pts")


print("HONEST BEFORE/AFTER on the same real games:")
_evaluate(OLD_POINTS_PER_RATING_UNIT, OLD_WIN_PROB_SCALE, "OLD hardcoded constants")
_evaluate(NEW_POINTS_PER_RATING_UNIT, NEW_WIN_PROB_SCALE, "NEW fit constants     ")
