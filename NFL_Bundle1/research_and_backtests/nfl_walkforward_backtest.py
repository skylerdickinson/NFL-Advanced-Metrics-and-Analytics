# ==============================================================================
# nfl_walkforward_backtest.py
# ==============================================================================
# ON-DEMAND TOOL -- not part of the weekly routine. Companion to
# nfl_historical_backtest.py, answering a DIFFERENT, harder question.
#
# nfl_historical_backtest.py measures "did the team with the better stat IN
# THAT GAME also win that game" -- a real number, but a same-game
# correlation, closer to "how explanatory is this stat of a game's own box
# score" than to a real pre-game forecast.
#
# This script instead measures what the weekly sheet actually does: for
# each real game, look ONLY at each team's rolling stat average from their
# games played BEFORE that week (exactly what nfl_team_matrix.csv would
# have shown going into that Sunday -- no leakage from the game being
# "predicted"), pick whichever team had the better pre-game number, and
# check how often that call was actually right. A team needs at least 2
# real prior games this season before it gets a pre-game number at all
# (same "needs 2+ games played" rule VOID already uses elsewhere) -- so
# every team's first 1-2 games each season are correctly excluded as
# undecidable, not guessed.
#
# Nothing here is faked: every number comes from real nflverse play-by-play.
# The per-game stat computation block is copied verbatim from nfl_data_pull.py
# PART 3C (same as nfl_historical_backtest.py); the backtest/tuning model is
# copied verbatim from nfl_matchup_sheet.py's model block. Keep both in sync
# with their sources if those ever change.
#
# Usage:
#   python nfl_walkforward_backtest.py
#       -> tests the 4 most recent real completed seasons
#   python nfl_walkforward_backtest.py --seasons 2019,2020,2021,2022,2023,2024
# ==============================================================================

import argparse
import datetime
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Walk-forward test: does each team's PRE-GAME rolling stat average actually call real winners?"
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

print(f"Walk-forward testing the real stat-weighted model against real seasons: {SEASONS}\n")

BACKTESTABLE_COLS = [
    'rush_yards', 'rush_yards_allowed', 'pass_yards', 'pass_yards_allowed',
    'off_epa_play_game', 'def_epa_play_allowed_game',
    'off_success_rate_game', 'def_success_rate_allowed_game',
    'third_down_pct_game', 'third_down_pct_allowed_game',
    'sack_pct_offense_game', 'sack_pct_defense_game',
    'pressure_rate_allowed_game', 'pressure_rate_forced_game',
    'completion_pct_game', 'interceptions_thrown_game', 'interceptions_gained_game',
    # ADDED (Run Success Rate vs Run Stop Rate, PBE vs PRP -- Pressure Rate
    # Allowed/Forced were already in this list above; Interior/Edge Pressure
    # (Tug o' War) is NOT here -- see the note above NFL_BACKTESTABLE_STATS
    # for why that one needs a genuinely different, bigger backtest, not a
    # quick add here).
    'off_run_success_rate_game', 'def_run_stop_rate_game',
    'pbe_score_game', 'prp_score_game',
    # ADDED -- Rush YPP isolates per-snap line-of-scrimmage results from
    # raw Rush Yards (which conflates game script/volume with actual
    # efficiency); Stuff Rate is the "how often does nothing happen" floor
    # measure; TFL Rate is real backfield disruption across ALL plays
    # (run+pass, sacks excluded -- their own stat already), NOT scoped to
    # run-only since that would mostly duplicate Stuff Rate; RYOE is Next
    # Gen Stats' own real context-adjusted rush value (different data
    # source -- NGS weekly data, not pbp -- so it's genuinely new
    # information, not just Rush YPP again); Fumbles Lost/Forced were
    # previously only baked into the combined is_turnover flag, never
    # tested as their own stat.
    'off_rush_ypp_game', 'def_rush_ypp_allowed_game',
    'stuff_rate_offense_game', 'stuff_rate_defense_game',
    'tfl_rate_allowed_game', 'tfl_rate_forced_game',
    'ryoe_per_attempt_game',
    'fumbles_lost_game', 'fumbles_forced_game',
    # ADDED same round -- Rush EPA/Play (run-only version of the existing
    # off_epa_play_game/def_epa_play_allowed_game) and Line Yards (real
    # Football Outsiders formula, not this codebase's own invention).
    'off_rush_epa_game', 'def_rush_epa_allowed_game',
    'off_line_yards_game', 'def_line_yards_allowed_game',
]

# ==============================================================================
# PER-SEASON REAL PER-GAME STAT LOG -- copied verbatim from nfl_data_pull.py
# PART 3B/3C (same as nfl_historical_backtest.py). See that file's copy of
# this block for the full explanatory comments.
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

    numeric_fields = ['yards_gained', 'epa', 'pass_attempt', 'sack', 'complete_pass', 'interception', 'third_down_converted',
                       'tackled_for_loss', 'fumble_lost']
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

    # Run Success Rate (offense) / Run Stop Rate (defense) -- ADDED. Same
    # real is_success (EPA>0) flag as
    # off_success_rate_game/def_success_rate_allowed_game above, copied
    # verbatim from nfl_data_pull.py's own real formula, just filtered to
    # rushing plays only. Defense side is framed as the share of real rush
    # attempts faced that did NOT succeed (1 - success rate), same as
    # nfl_data_pull.py, so BOTH sides read "higher is better" -- matches
    # def_run_stop_rate's real direction in nfl_team_matrix.csv.
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

    # PBE (Pass Block Efficiency) / PRP (Pass Rush Productivity) -- ADDED.
    # Same real weighted formula as nfl_data_pull.py's season-long
    # pbe_score/prp_score (1x sacks, 0.75x QB hits, NOT the same as the plain
    # pressure_rate_allowed/forced_game above), reusing the exact same
    # _sacks/_hits/_drop counts already grouped above for
    # pressure_off_g/pressure_def_g -- no new groupby needed. PBE is framed
    # 100=never pressured (subtracted from 100); PRP is a straight weighted
    # rate (higher = more pressure forced).
    pressure_off_g['pbe_score_game'] = np.where(
        pressure_off_g['_drop'] > 0,
        100 - ((pressure_off_g['_sacks'] + 0.75 * pressure_off_g['_hits']) / pressure_off_g['_drop'] * 100), np.nan)
    pressure_def_g['prp_score_game'] = np.where(
        pressure_def_g['_drop'] > 0,
        (pressure_def_g['_sacks'] + 0.75 * pressure_def_g['_hits']) / pressure_def_g['_drop'] * 100, np.nan)

    # Rush Yards Per Play (offense) / Rush Yards Per Play Allowed (defense)
    # -- ADDED. Same real run_plays_clean already built above for Run
    # Success/Stop Rate, aggregating yards_gained with BOTH sum and count
    # so it can be divided into a true per-snap average -- strips out
    # game-script/volume effects that raw "Rush Yards" (total per game)
    # can't separate from real per-play line-of-scrimmage results.
    off_rush_ypp_g = run_plays_clean.groupby(['posteam', 'game_id'])['yards_gained'].agg(['sum', 'count']).reset_index().rename(
        columns={'posteam': 'team', 'sum': '_rush_yds', 'count': '_rush_n'})
    off_rush_ypp_g['off_rush_ypp_game'] = np.where(off_rush_ypp_g['_rush_n'] > 0, off_rush_ypp_g['_rush_yds'] / off_rush_ypp_g['_rush_n'], np.nan)
    def_rush_ypp_g = run_plays_clean.groupby(['defteam', 'game_id'])['yards_gained'].agg(['sum', 'count']).reset_index().rename(
        columns={'defteam': 'team', 'sum': '_rush_yds_allowed', 'count': '_rush_n_allowed'})
    def_rush_ypp_g['def_rush_ypp_allowed_game'] = np.where(def_rush_ypp_g['_rush_n_allowed'] > 0, def_rush_ypp_g['_rush_yds_allowed'] / def_rush_ypp_g['_rush_n_allowed'], np.nan)

    # Stuff Rate (offense/defense) -- ADDED. Same real run_plays_clean pool,
    # real formula from nfl_data_pull.py (share of rush plays stopped AT OR
    # BEHIND the line, yards_gained<=0). Offense side: lower is better
    # (rarely stuffed). Defense side: higher is better (stuffs opponents a
    # lot).
    stuff_off_g = run_plays_clean.groupby(['posteam', 'game_id'])['yards_gained'].apply(lambda s: (s <= 0).mean() * 100).reset_index().rename(
        columns={'posteam': 'team', 'yards_gained': 'stuff_rate_offense_game'})
    stuff_def_g = run_plays_clean.groupby(['defteam', 'game_id'])['yards_gained'].apply(lambda s: (s <= 0).mean() * 100).reset_index().rename(
        columns={'defteam': 'team', 'yards_gained': 'stuff_rate_defense_game'})

    # Rush EPA/Play (offense) / Rush EPA/Play Allowed (defense) -- ADDED. The
    # existing off_epa_play_game/def_epa_play_allowed_game above are already
    # real EPA/play, but computed across pbp_clean (run+pass combined) --
    # there was no run-only version. Same real run_plays_clean pool as Run
    # Success Rate above, just averaging the real epa column directly instead
    # of the binary is_success flag -- a continuous-value lens on run
    # efficiency alongside Run Success Rate's binary one (Success Rate 63.4%
    # and EPA/Play 61.0% already tested as real-but-different signal for the
    # combined run+pass case, so the run-only pair is worth testing on its
    # own rather than assumed redundant with Run Success Rate).
    off_rush_epa_g = run_plays_clean.groupby(['posteam', 'game_id'])['epa'].mean().reset_index().rename(
        columns={'posteam': 'team', 'epa': 'off_rush_epa_game'})
    def_rush_epa_g = run_plays_clean.groupby(['defteam', 'game_id'])['epa'].mean().reset_index().rename(
        columns={'defteam': 'team', 'epa': 'def_rush_epa_allowed_game'})

    # Line Yards (offense) / Line Yards Allowed (defense) -- ADDED. Real,
    # publicly documented Football Outsiders formula (not "our own" like
    # Stuff Rate/Havoc Rate above) that credits the O-line specifically for
    # getting a run TO the second level, not for what the back does once
    # he's through: losses count 120% (a bad snap is fully on the line),
    # 0-4 real yards count 100%, 4-8 real yards count only 50%
    # (increasingly the back's own ability), and anything past 8 real yards
    # is capped at 0% marginal credit (a real breakaway run is credited to
    # the RB alone, not the blocking) -- so this is a genuinely different
    # lens from raw Rush YPP above, which gives a 60-yard run full credit
    # for every one of its 60 yards.
    _ly_yards = run_plays_clean['yards_gained']
    _ly_value = np.select(
        [_ly_yards < 0, _ly_yards <= 4, _ly_yards <= 8],
        [_ly_yards * 1.2, _ly_yards * 1.0, 4 + (_ly_yards - 4) * 0.5],
        default=6.0,
    )
    line_yards_df = run_plays_clean[['posteam', 'defteam', 'game_id']].copy()
    line_yards_df['_line_yards_value'] = _ly_value
    off_line_yards_g = line_yards_df.groupby(['posteam', 'game_id'])['_line_yards_value'].mean().reset_index().rename(
        columns={'posteam': 'team', '_line_yards_value': 'off_line_yards_game'})
    def_line_yards_g = line_yards_df.groupby(['defteam', 'game_id'])['_line_yards_value'].mean().reset_index().rename(
        columns={'defteam': 'team', '_line_yards_value': 'def_line_yards_allowed_game'})

    # TFL Rate allowed (offense) / forced (defense) -- ADDED. Deliberately
    # NOT scoped to run plays only -- a run-only TFL rate would mostly
    # duplicate Stuff Rate above (any run tackled for an actual LOSS is
    # already inside Stuff Rate's "<=0 yards" bucket). Using the real
    # tackled_for_loss flag across ALL of pbp_clean (run+ pass, sacks
    # excluded -- they have their own real flag/stat already) instead
    # captures broader real backfield disruption, including a blown-up
    # screen or checkdown, that Stuff Rate can't see. Same real column
    # nfl_data_pull.py already uses for its season-long
    # tfl_made/tfl_allowed, expressed here as a rate per total real plays
    # run.
    tfl_off_g = pbp_clean.groupby(['posteam', 'game_id']).agg(
        _tfl=('tackled_for_loss', 'sum'), _plays=('play_id', 'count')).reset_index().rename(columns={'posteam': 'team'})
    tfl_off_g['tfl_rate_allowed_game'] = np.where(tfl_off_g['_plays'] > 0, tfl_off_g['_tfl'] / tfl_off_g['_plays'] * 100, np.nan)
    tfl_def_g = pbp_clean.groupby(['defteam', 'game_id']).agg(
        _tfl=('tackled_for_loss', 'sum'), _plays=('play_id', 'count')).reset_index().rename(columns={'defteam': 'team'})
    tfl_def_g['tfl_rate_forced_game'] = np.where(tfl_def_g['_plays'] > 0, tfl_def_g['_tfl'] / tfl_def_g['_plays'] * 100, np.nan)

    # Fumbles Lost (offense) / Fumbles Forced (defense) -- ADDED. Real
    # fumble_lost flag, same one already folded into is_turnover in
    # nfl_data_pull.py, but never tested as its own real backtestable stat
    # until now (same pattern as Interceptions Thrown/Gained just below).
    fum_off_g = pbp_clean.groupby(['posteam', 'game_id'])['fumble_lost'].sum().reset_index().rename(
        columns={'posteam': 'team', 'fumble_lost': 'fumbles_lost_game'})
    fum_def_g = pbp_clean.groupby(['defteam', 'game_id'])['fumble_lost'].sum().reset_index().rename(
        columns={'defteam': 'team', 'fumble_lost': 'fumbles_forced_game'})

    # RYOE per Attempt -- ADDED. Different real data source than everything
    # else in this file: Next Gen Stats (nfl.load_nextgen_stats), not
    # play-by-play -- a real, context-adjusted rush-value metric (accounts
    # for real blockers-in-box/direction/down-distance via NGS's own
    # expectation model), genuinely different information from the raw Rush
    # YPP above, not just the same thing twice. NGS rushing data is
    # published per RUSHER per real week, so this is averaged (unweighted,
    # same convention nfl_data_pull.py's own season-long ryoe_per_attempt
    # already uses) across a team's real rushers for that week to get one
    # real team-week number, merged onto team_games by (team, week) below --
    # no game_id in the raw NGS data, so this is the one stat in this file
    # joined on week instead of game_id. Single-sided (no real "RYOE
    # allowed" mirror exists in this data), same as how Rush
    # Yards/Completion %/3rd Down % already compare each team's own real
    # number head-to-head rather than needing an opponent side. Wrapped
    # defensively, same pattern as nfl_data_pull.py's _pull_ngs_pfr_metric
    # -- a real loader hiccup or column rename blanks this one season's
    # RYOE, not the whole script.
    try:
        ngs_rush = nfl.load_nextgen_stats(stat_type='rushing', seasons=[yr]).to_pandas()
        if 'week' in ngs_rush.columns:
            ngs_rush = ngs_rush[ngs_rush['week'] > 0]
        ngs_team_col = 'team_abbr' if 'team_abbr' in ngs_rush.columns else ('team' if 'team' in ngs_rush.columns else None)
        if ngs_team_col and 'rush_yards_over_expected_per_att' in ngs_rush.columns and 'week' in ngs_rush.columns:
            ngs_rush['rush_yards_over_expected_per_att'] = pd.to_numeric(ngs_rush['rush_yards_over_expected_per_att'], errors='coerce')
            ryoe_g = ngs_rush.groupby([ngs_team_col, 'week'])['rush_yards_over_expected_per_att'].mean().reset_index().rename(
                columns={ngs_team_col: 'team', 'rush_yards_over_expected_per_att': 'ryoe_per_attempt_game'})
        else:
            print(f"  ⚠ real Next Gen Stats rushing columns not found for {yr} -- RYOE will be blank this season")
            ryoe_g = pd.DataFrame(columns=['team', 'week', 'ryoe_per_attempt_game'])
    except Exception as e:
        print(f"  ⚠ real Next Gen Stats rushing data unavailable for {yr} -- RYOE will be blank this season ({e})")
        ryoe_g = pd.DataFrame(columns=['team', 'week', 'ryoe_per_attempt_game'])

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
        (off_rush_ypp_g[['team', 'game_id', 'off_rush_ypp_game']], ['off_rush_ypp_game']),
        (def_rush_ypp_g[['team', 'game_id', 'def_rush_ypp_allowed_game']], ['def_rush_ypp_allowed_game']),
        (stuff_off_g, ['stuff_rate_offense_game']),
        (stuff_def_g, ['stuff_rate_defense_game']),
        (tfl_off_g[['team', 'game_id', 'tfl_rate_allowed_game']], ['tfl_rate_allowed_game']),
        (tfl_def_g[['team', 'game_id', 'tfl_rate_forced_game']], ['tfl_rate_forced_game']),
        (fum_off_g, ['fumbles_lost_game']),
        (fum_def_g, ['fumbles_forced_game']),
        (off_rush_epa_g, ['off_rush_epa_game']),
        (def_rush_epa_g, ['def_rush_epa_allowed_game']),
        (off_line_yards_g, ['off_line_yards_game']),
        (def_line_yards_g, ['def_line_yards_allowed_game']),
    ]:
        team_games = team_games.merge(extra_df, on=['team', 'game_id'], how='left')

    # RYOE joins on (team, week) instead of (team, game_id) -- see the note
    # above where ryoe_g is built. Printed match rate is an honest real
    # diagnostic: if real team codes between pbp and Next Gen Stats ever
    # drift apart for a franchise, this is what would show it immediately
    # instead of that team silently going all-NaN with no explanation.
    team_games = team_games.merge(ryoe_g, on=['team', 'week'], how='left')
    _ryoe_match_pct = team_games['ryoe_per_attempt_game'].notna().mean() * 100 if len(team_games) else 0.0
    print(f"    -> real RYOE matched for {_ryoe_match_pct:.0f}% of {yr} team-game rows (team+week join to Next Gen Stats)")

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
# WALK-FORWARD TRANSFORM: replace each team-game's OWN box-score stat with
# that team's REAL rolling average from games played BEFORE this week, same
# season only (never crosses into a different season, never peeks at the
# game being "predicted" or any later game). Requires 2+ real prior games
# with a non-missing value for that specific stat, same as VOID's own
# "needs 2+ games played" rule -- otherwise NaN, honestly excluded rather
# than guessed.
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
# BACKTEST / WEIGHT-TUNING MODEL -- copied verbatim from nfl_matchup_sheet.py
# (same as nfl_historical_backtest.py's copy). Run TWICE below: once against
# the real same-game box score (retrospective, for direct comparison) and
# once against the real pre-game rolling averages (the actual walk-forward
# test this script exists for).
# ==============================================================================

NFL_BACKTESTABLE_STATS = [
    ("Rush Yards", "rush_yards", "rushing_yards_per_game", True, "TRENCHES & RUN GAME"),
    ("Rush Yards Allowed", "rush_yards_allowed", "rushing_yards_allowed_per_game", False, "TRENCHES & RUN GAME"),
    # ADDED -- never backtested before now. Both read "higher is better"
    # (def_run_stop_rate is framed as 1 - success-rate-allowed, same as
    # nfl_data_pull.py's real formula).
    ("Run Success Rate (Offense)", "off_run_success_rate_game", "off_run_success_rate", True, "TRENCHES & RUN GAME"),
    ("Run Stop Rate (Defense)", "def_run_stop_rate_game", "def_run_stop_rate", True, "TRENCHES & RUN GAME"),
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
    # ADDED -- same real sack/QB-hit data as Pressure Rate above, just
    # weighted 1x sacks + 0.75x hits (PFF-style) instead of an unweighted
    # rate. PBE framed 100=never pressured (higher is better); PRP a
    # straight weighted rate forced (higher is better for the defense).
    ("Pass Block Efficiency (PBE)", "pbe_score_game", "pbe_score", True, "PASSING & PROTECTION"),
    ("Pass Rush Productivity (PRP)", "prp_score_game", "prp_score", True, "PASSING & PROTECTION"),
    ("Completion %", "completion_pct_game", "completion_pct", True, "QB ACCURACY & EFFICIENCY"),
    ("Interceptions Thrown", "interceptions_thrown_game", "interceptions_thrown_per_game", False, "QB ACCURACY & EFFICIENCY"),
    ("Interceptions Gained", "interceptions_gained_game", "interceptions_gained_per_game", True, "QB ACCURACY & EFFICIENCY"),
    # ADDED (rush YPP / stuff rate / TFL /RYOE / fumbles) -- see the
    # per-game computation block above for the real formula and reasoning
    # behind each one, especially why TFL Rate is deliberately NOT
    # run-scoped (would mostly duplicate Stuff Rate) and why RYOE is a
    # genuinely different data source from Rush YPP, not a duplicate of it.
    ("Rush Yards Per Play (Offense)", "off_rush_ypp_game", "off_rushing_yards_per_play", True, "TRENCHES & RUN GAME"),
    ("Rush Yards Per Play Allowed (Defense)", "def_rush_ypp_allowed_game", "def_rushing_yards_per_play", False, "TRENCHES & RUN GAME"),
    ("Stuff Rate (Offense)", "stuff_rate_offense_game", "stuff_rate_offense", False, "TRENCHES & RUN GAME"),
    ("Stuff Rate (Defense)", "stuff_rate_defense_game", "stuff_rate_defense", True, "TRENCHES & RUN GAME"),
    # team_matrix_col points at the real raw counts (tfl_allowed/tfl_made)
    # -- team_matrix doesn't have a rate version of these yet, only this
    # backtest's own per-game number is expressed as a real rate.
    ("TFL Rate Allowed (Offense)", "tfl_rate_allowed_game", "tfl_allowed", False, "TRENCHES & RUN GAME"),
    ("TFL Rate Forced (Defense)", "tfl_rate_forced_game", "tfl_made", True, "TRENCHES & RUN GAME"),
    ("RYOE per Attempt", "ryoe_per_attempt_game", "ryoe_per_attempt", True, "TRENCHES & RUN GAME"),
    ("Fumbles Lost (Offense)", "fumbles_lost_game", "fumbles_lost_per_game", False, "TRENCHES & RUN GAME"),
    ("Fumbles Forced (Defense)", "fumbles_forced_game", "opponent_fumbles_forced_per_game", True, "TRENCHES & RUN GAME"),
    # ADDED same round. team_matrix_col names below do NOT exist in
    # nfl_team_matrix.csv yet -- these are brand new stat concepts to this
    # whole project, being tested here first before any decision to wire
    # them into the live weekly sheet.
    ("Rush EPA/Play (Offense)", "off_rush_epa_game", "off_rush_epa", True, "TRENCHES & RUN GAME"),
    ("Rush EPA/Play Allowed (Defense)", "def_rush_epa_allowed_game", "def_rush_epa_allowed", False, "TRENCHES & RUN GAME"),
    ("Line Yards (Offense)", "off_line_yards_game", "off_line_yards", True, "TRENCHES & RUN GAME"),
    ("Line Yards Allowed (Defense)", "def_line_yards_allowed_game", "def_line_yards_allowed", False, "TRENCHES & RUN GAME"),
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


def run_and_report(title, game_log_source_df):
    history = {"games": {}}
    n_recorded = record_nfl_completed_games(history, game_log_source_df)
    included, accuracy, correct, decided, total = tune_nfl_stat_weights(history)
    label_by_key = {game_col: label for label, game_col, _, _, _ in NFL_BACKTESTABLE_STATS}

    print("=" * 78)
    print(f"{title} -- {n_recorded} real game(s) with a clean W/L on both sides")
    print("=" * 78)
    if not included:
        print("No stat cleared the model's inclusion bar on this real sample.")
    else:
        print(f"Combined tuned model: {correct}/{decided} decided = {accuracy*100:.1f}% "
              f"({total - decided} of {total} left undecided -- net-zero lean)")
        print("Stats the model kept (greedy, stops when nothing left helps):")
        for label, game_col, weight, win_rate, n in included:
            print(f"  {label:32s}  real win rate {win_rate*100:5.1f}%  (n={n:4d})   weight={weight:+.3f}")

    print("\nEvery real stat's own individual win rate (n=3+), ranked:")
    all_rates = _nfl_individual_stat_win_rates(history)
    included_keys_set = {k for _, k, *_ in included}
    for key, (wr, n) in sorted(all_rates.items(), key=lambda kv: -kv[1][0]):
        tag = "  <- IN TUNED MODEL" if key in included_keys_set else ""
        print(f"  {label_by_key.get(key, key):32s}  real win rate {wr*100:5.1f}%  (n={n:4d}){tag}")
    print()
    return accuracy, decided, total


retro_acc, retro_decided, retro_total = run_and_report(
    "RETROSPECTIVE (same-game box score vs. that game's own outcome)", historical_log)

walk_acc, walk_decided, walk_total = run_and_report(
    "WALK-FORWARD (real pre-game rolling averages vs. the actual outcome)", prewk_log)

print("=" * 78)
print("SIDE BY SIDE")
print("=" * 78)
print(f"Retrospective (same-game, explanatory):  {retro_acc*100:5.1f}%  on {retro_decided}/{retro_total} real games")
print(f"Walk-forward (pre-game, actually predictive): {walk_acc*100:5.1f}%  on {walk_decided}/{walk_total} real games")
print(f"\nThe walk-forward number is the honest one -- it's what the sheet's weekly win-probability calls should "
      f"actually be expected to land near, since that's the same 'only knowable info' constraint the real weekly "
      f"matchup sheet is under. The retrospective number above it is a ceiling on how explanatory these stats are "
      f"of a game's own box score, not a promise about future picks.")
# FIXED: this note used to still list Pass Block Efficiency/Pass Rush
# Productivity as untested even after they were added earlier this
# session -- stale and wrong. Now lists only what's ACTUALLY still
# missing as of this round (Havoc Rate, Red Zone %, Passer Rating, CPOE,
# and full Turnover Margin -- CPOE specifically skipped on purpose, see
# README, since it's a passing-side stat that doesn't bear on the
# trenches/run-game question this round of additions was about).
print("\nNote: VOID, Havoc Rate, Red Zone %, Passer Rating, CPOE, and full Turnover Margin (fumbles lost/forced "
      "are now tested individually above, but not combined into one real turnover-margin number) are still not "
      "in this test -- same PART 3C scope limit as nfl_historical_backtest.py.")
