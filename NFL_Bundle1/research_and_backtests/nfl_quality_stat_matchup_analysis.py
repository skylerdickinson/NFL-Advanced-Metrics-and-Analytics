# ==============================================================================
# nfl_quality_stat_matchup_analysis.py
# ==============================================================================
# ON-DEMAND TOOL -- not part of the weekly routine. Companion to
# nfl_walkforward_backtest.py, answering this question about the quality-stat badge: when the 11/12
# important stats win about 67% of games, were the other 33% games where the
# opponent had several matched stats (e.g. 5/12 vs 8/11)?
#
# (The badge was 12 stats when this was written -- see the UPDATED note below:
# NFL_QUALITY_STATS grew to 15 real stats in a later round, and this file's
# own vote list was refreshed to match.)
#
# Short answer: yes, and it doesn't need a new "how close were the raw
# numbers" threshold (which would be a judgment call) -- the real quality-
# stat badge ALREADY computes, for every real game, exactly how many of
# its stats favored each team pre-game (that's the vote count the badge
# itself is built from). This script reuses that same real vote tally and
# asks: was the badge WRONG more often on games where the vote was close
# to a 50/50 split (e.g. 8-7, 7-6 -- a real toss-up) than on games where
# it was lopsided (e.g. 12-1)?  If accuracy is clearly lower in the
# close-split bucket than the lopsided-split bucket, that's a real, honest
# "yes" to your question -- the misses cluster where the stats themselves
# didn't clearly agree, not randomly across the board.
#
# UPDATED: NFL_QUALITY_STATS in nfl_matchup_sheet.py was refreshed from 12
# to 15 real stats after the "trenches" round tested 13 new candidate
# stats -- 3 of them (Run Success Rate, Pass Block Efficiency, Rush Yards
# Per Play Allowed) cleared the real >55% win-rate bar and already had a
# live nfl_team_matrix.csv column, so they were added to the weekly badge.
# This file's NFL_QUALITY_STATS_FOR_MATCHUP below was refreshed the same
# way, in the same order, so the vote-margin analysis here reflects the
# CURRENT real badge, not the older 12-stat one. (See README.txt for the
# full real backtested win rates behind each stat.)
#
# Data + methodology: this reuses nfl_walkforward_backtest.py's real
# per-game pull and walk-forward pre-game-rolling-average transform
# VERBATIM (same "only knowable info going into that Sunday" constraint,
# same "2+ real prior games needed" burn-in rule, same real nflverse
# play-by-play source) -- restricted down to the SAME 15 real quality
# stats NFL_QUALITY_STATS in nfl_matchup_sheet.py actually uses for the
# weekly badge (NFL_QUALITY_STATS_FOR_MATCHUP below -- keep in sync with
# that list if it ever changes). Keep this file's copied blocks in sync
# with nfl_walkforward_backtest.py/nfl_data_pull.py if those change.
#
# Usage:
#   python nfl_quality_stat_matchup_analysis.py
#       -> tests the 4 most recent real completed seasons
#   python nfl_quality_stat_matchup_analysis.py --seasons 2019,2020,2021,2022,2023,2024
# ==============================================================================

import argparse
import datetime
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Does the 15-quality-stat badge miss more often when the real pre-game vote was close to a 50/50 split?"
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

print(f"Quality-stat vote-margin analysis against real seasons: {SEASONS}\n")

BACKTESTABLE_COLS = [
    'rush_yards', 'rush_yards_allowed', 'pass_yards', 'pass_yards_allowed',
    'off_epa_play_game', 'def_epa_play_allowed_game',
    'off_success_rate_game', 'def_success_rate_allowed_game',
    'third_down_pct_game', 'third_down_pct_allowed_game',
    'sack_pct_offense_game', 'sack_pct_defense_game',
    'pressure_rate_allowed_game', 'pressure_rate_forced_game',
    'completion_pct_game', 'interceptions_thrown_game', 'interceptions_gained_game',
    # 3 new real stats added when NFL_QUALITY_STATS was refreshed from 12
    # to 15 -- same real formulas as nfl_walkforward_backtest.py, copied
    # verbatim (see the computation block below).
    'off_run_success_rate_game', 'pbe_score_game', 'def_rush_ypp_allowed_game',
]

# ==============================================================================
# PER-SEASON REAL PER-GAME STAT LOG -- copied verbatim from
# nfl_walkforward_backtest.py (itself copied verbatim from nfl_data_pull.py
# PART 3B/3C). See those files for the full explanatory comments on each
# step. Keep in sync if either of those change.
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

    # 3 new real stats -- ADDED when NFL_QUALITY_STATS was refreshed from
    # 12 to 15 real stats. Formulas copied verbatim from
    # nfl_walkforward_backtest.py (see that file's own comments for the
    # full real reasoning behind each one).
    run_plays_clean = pbp_clean[pbp_clean['play_type'] == 'run']
    # Run Success Rate (Offense) -- same real is_success (EPA>0) flag as
    # off_success_rate_game above, filtered to rushing plays only.
    off_run_sr_g = run_plays_clean.groupby(['posteam', 'game_id'])['is_success'].mean().reset_index().rename(
        columns={'posteam': 'team', 'is_success': 'off_run_success_rate_game'})
    off_run_sr_g['off_run_success_rate_game'] *= 100
    # Pass Block Efficiency (PBE) -- same real weighted formula as
    # nfl_data_pull.py's season-long pbe_score (1x sacks, 0.75x QB hits),
    # reusing the exact same _sacks/_hits/_drop counts already grouped
    # above for pressure_off_g -- no new groupby needed. Framed
    # 100=never pressured (subtracted from 100), higher is better.
    pressure_off_g['pbe_score_game'] = np.where(
        pressure_off_g['_drop'] > 0,
        100 - ((pressure_off_g['_sacks'] + 0.75 * pressure_off_g['_hits']) / pressure_off_g['_drop'] * 100), np.nan)
    # Rush Yards Per Play Allowed (Defense) -- same real run_plays_clean
    # pool, yards_gained summed and counted per team-game so it can be
    # divided into a true per-snap average (lower is better).
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
# WALK-FORWARD TRANSFORM -- copied verbatim from nfl_walkforward_backtest.py.
# Same "only knowable info going into that Sunday" constraint the real
# weekly badge is under: each team-game's own box-score stat is replaced
# with that team's real rolling average from games played BEFORE this
# week, same season only, 2+ real prior games required.
# ==============================================================================

print("Building real pre-game rolling averages (2+ prior real games required, per team per season)...")
prewk_log = historical_log.copy().sort_values(['team', 'season', 'week'])
for col in BACKTESTABLE_COLS:
    prewk_log[col] = (
        prewk_log.groupby(['team', 'season'])[col]
        .transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
    )

# ==============================================================================
# THE REAL 15 QUALITY STATS -- must stay in sync with NFL_QUALITY_STATS in
# nfl_matchup_sheet.py (that's the exact list the weekly badge counts). REFRESHED
# from 12 to 15 (same order as NFL_QUALITY_STATS) when that list was refreshed
# after the "trenches" round -- the 3 new entries are marked below. (label,
# game_log_col, higher_is_better) -- game_log_col is the walk-forward pre-game
# column above; mirrors NFL_QUALITY_STATS' team_matrix_col one-for-one via
# NFL_BACKTESTABLE_STATS' mapping in nfl_walkforward_backtest.py.
# ==============================================================================
NFL_QUALITY_STATS_FOR_MATCHUP = [
    ("Success Rate (Offense)", "off_success_rate_game", True),
    ("EPA/Play (Offense)", "off_epa_play_game", True),
    ("Pass Yards", "pass_yards", True),
    ("Run Success Rate (Offense)", "off_run_success_rate_game", True),      # NEW
    ("Sack % Taken", "sack_pct_offense_game", False),
    ("Completion %", "completion_pct_game", True),
    ("Pass Block Efficiency (PBE)", "pbe_score_game", True),                # NEW
    ("Pressure Rate Allowed", "pressure_rate_allowed_game", False),
    ("Interceptions Gained", "interceptions_gained_game", True),
    ("EPA/Play Allowed (Defense)", "def_epa_play_allowed_game", False),
    ("Rush Yards Allowed", "rush_yards_allowed", False),
    ("Rush Yards Per Play Allowed (Defense)", "def_rush_ypp_allowed_game", False),  # NEW
    ("Pressure Rate Forced", "pressure_rate_forced_game", True),
    ("Rush Yards", "rush_yards", True),
    ("3rd Down %", "third_down_pct_game", True),
]


def _stat_vote(winner_val, loser_val, higher_is_better):
    """Same real vote logic the badge/backtest already use: +1 if this stat
    favored the eventual winner pre-game, -1 if it favored the loser, None
    if either side has no real pre-game number yet or the two are exactly
    tied (no real lean either way)."""
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


def build_real_completed_games(game_log_source_df):
    """Pairs up each real completed game's two team-rows (needs a clean W/L
    on both sides -- ties excluded, same as the walk-forward backtest) and
    computes each side's real pre-game vote count on the 15 quality stats.
    Returns a list of dicts: one per real game."""
    has_game_id = 'game_id' in game_log_source_df.columns

    def _game_key(row):
        if has_game_id and pd.notna(row.get('game_id')):
            return f"gid_{row['game_id']}"
        return f"wk_{row['week']}_{'-'.join(sorted([str(row['team']), str(row['opponent'])]))}"

    grouped = {}
    for _, row in game_log_source_df.iterrows():
        grouped.setdefault(_game_key(row), []).append(row)

    games = []
    for key, rows in grouped.items():
        if len(rows) != 2:
            continue
        r0, r1 = rows[0], rows[1]
        if r0['result'] not in ('W', 'L') or r1['result'] not in ('W', 'L') or r0['result'] == r1['result']:
            continue
        winner_row, loser_row = (r0, r1) if r0['result'] == 'W' else (r1, r0)

        n_favor_winner = 0
        n_favor_loser = 0
        for label, game_col, higher_is_better in NFL_QUALITY_STATS_FOR_MATCHUP:
            vote = _stat_vote(winner_row.get(game_col), loser_row.get(game_col), higher_is_better)
            if vote == 1:
                n_favor_winner += 1
            elif vote == -1:
                n_favor_loser += 1
            # vote is None (missing pre-game data or exact tie) -> not
            # counted on either side, same as the real badge's own logic.

        n_decided = n_favor_winner + n_favor_loser
        if n_decided == 0:
            continue  # no usable pre-game stat at all yet for this game -- skip, don't guess

        # Real final score margin -- pts_for/pts_against are the actual
        # game's real final score (NOT part of BACKTESTABLE_COLS, so the
        # walk-forward transform never touches them -- these are untouched
        # real box-score numbers), letting us check the "was it basically a
        # FG-or-less finish" question directly, no new data pull needed.
        w_pts, l_pts = winner_row.get('pts_for'), loser_row.get('pts_for')
        score_margin = None
        if pd.notna(w_pts) and pd.notna(l_pts):
            score_margin = float(w_pts) - float(l_pts)

        games.append({
            "week": int(winner_row['week']) if pd.notna(winner_row.get('week')) else None,
            "winner": winner_row['team'], "loser": loser_row['team'],
            "n_favor_winner": n_favor_winner, "n_favor_loser": n_favor_loser,
            "n_decided": n_decided,
            "margin": n_favor_winner - n_favor_loser,   # >0 = badge correctly leaned winner; <0 = badge leaned loser (wrong); 0 = dead-even split
            "score_margin": score_margin,               # real final point differential (winner_pts - loser_pts), None if either score missing
        })
    return games


def report_upset_score_margins(games):
    """Tests the "FG to win" explanation for upsets directly with real final scores we
    already have (the injury half would need a separate real historical
    injury-report pull and matching to specific players' importance, a much
    fuzzier judgment call, not attempted here). For the real 'lopsided
    wrong' games (badge margin 6+, but it still picked the loser -- i.e.
    most of the 15 stats agreed and the team lost anyway), buckets by the
    REAL final score margin to see whether those upsets tended to come down
    to a one-score (or even one-kick) finish, or got run away with. The
    margin-6+ cutoff is an absolute count kept as-is from the original
    12-stat version (not rescaled to the new 15-stat max) -- still a
    meaningful "most stats agreed" bar, just a slightly smaller share of
    the real vote than it used to be."""
    lopsided_wrong = [g for g in games if g["margin"] <= -6 and g["score_margin"] is not None]
    if not lopsided_wrong:
        print("\nNo real lopsided-wrong (upset) games with a real final score to check.")
        return

    print("\n" + "=" * 78)
    print("YOUR FOLLOW-UP: of the real LOPSIDED-WRONG games (most stats agreed, team")
    print("lost anyway) -- were these close, could-have-gone-either-way finishes, or")
    print("real blowouts the other way? Bucketed by REAL final score margin.")
    print("=" * 78)
    score_buckets = [
        (0, 3, "3 pts or less -- literally a FG (or a missed one) from flipping"),
        (4, 8, "4-8 pts -- one-score game (a TD/2pt swing would flip it)"),
        (9, 16, "9-16 pts -- two scores, not a nail-biter"),
        (17, 999, "17+ pts -- a real blowout the other way"),
    ]
    n_total = len(lopsided_wrong)
    for lo, hi, desc in score_buckets:
        bucket = [g for g in lopsided_wrong if lo <= g["score_margin"] <= hi]
        pct = len(bucket) / n_total * 100 if n_total else 0
        print(f"  {desc:60s} {len(bucket):>4d} of {n_total}  ({pct:5.1f}%)")

    n_one_score = sum(1 for g in lopsided_wrong if g["score_margin"] <= 8)
    print(f"\n{n_one_score} of {n_total} real lopsided-wrong games ({n_one_score/n_total*100:.1f}%) were "
          f"decided by 8 points or less (one score) -- despite most of the 15 stats "
          f"pointing the other way. If that share is high, it's a real, honest "
          f"'yes': a lot of these upsets weren't the model being badly fooled, they "
          f"were close games that could have bounced either way, and just happened "
          f"to bounce against the stats.")
    print("\nOn injuries specifically: this can't be checked yet -- would need a real")
    print("historical injury-report pull (nflverse has one, nfl.load_injuries()) matched")
    print("to which specific players sat out or were limited each game, and then a")
    print("judgment call on whether that player mattered enough to swing the game --")
    print("a lot fuzzier than the score-margin check above. Not attempted here.")


def report_vote_margin_analysis(games):
    if not games:
        print("No real games had at least one decided quality stat -- nothing to report.")
        return

    n_correct = sum(1 for g in games if g["margin"] > 0)
    n_wrong = sum(1 for g in games if g["margin"] < 0)
    n_push = sum(1 for g in games if g["margin"] == 0)
    n_total = len(games)

    print("=" * 78)
    print(f"REAL 15-QUALITY-STAT VOTE, {n_total} real completed game(s)")
    print("=" * 78)
    print(f"Badge favored the real winner:  {n_correct}/{n_total}  ({n_correct/n_total*100:.1f}%)")
    print(f"Badge favored the real loser:   {n_wrong}/{n_total}  ({n_wrong/n_total*100:.1f}%)")
    print(f"Dead-even split (no lean):      {n_push}/{n_total}  ({n_push/n_total*100:.1f}%)")

    print("\n" + "=" * 78)
    print("YOUR QUESTION: does accuracy drop when the real vote was close to 50/50")
    print("(e.g. 6-5) vs. lopsided (e.g. 10-1)?  Bucketed by |margin| -- how many")
    print("more stats favored one side than the other, among the real decided ones.")
    print("=" * 78)
    decided_games = [g for g in games if g["margin"] != 0]
    buckets = [(1, 1, "margin 1 (e.g. 6-5, 5-4) -- essentially a coin flip"),
               (2, 2, "margin 2 (e.g. 7-5, 6-4)"),
               (3, 3, "margin 3 (e.g. 8-5, 7-4)"),
               (4, 5, "margin 4-5"),
               (6, 999, "margin 6+ -- lopsided, most stats agreed")]
    print(f"{'BUCKET':55s} {'GAMES':>7s} {'CORRECT':>9s} {'ACCURACY':>10s}")
    for lo, hi, desc in buckets:
        bucket_games = [g for g in decided_games if lo <= abs(g["margin"]) <= hi]
        if not bucket_games:
            print(f"{desc:55s} {'0':>7s} {'--':>9s} {'--':>10s}")
            continue
        bucket_correct = sum(1 for g in bucket_games if g["margin"] > 0)
        acc = bucket_correct / len(bucket_games) * 100
        print(f"{desc:55s} {len(bucket_games):>7d} {bucket_correct:>9d} {acc:>9.1f}%")

    print("\nIf the top rows (close margins) show a clearly lower accuracy % than the")
    print("bottom rows (lopsided margins), that's a real 'yes' to your question --")
    print("the misses really do cluster in the close, mixed-stat games rather than")
    print("being spread evenly across confident and toss-up calls alike.")

    print("\n" + "=" * 78)
    print("SAME QUESTION, FROM THE OTHER DIRECTION: of the real games the badge got")
    print("WRONG, how many were close mixed calls (margin 1-2) vs. real lopsided")
    print("upsets (margin 5+, i.e. most stats pointed to the team that still lost)?")
    print("=" * 78)
    wrong_games = [g for g in games if g["margin"] < 0]
    if not wrong_games:
        print("No wrong picks in this real sample.")
    else:
        for lo, hi, desc in buckets:
            n_in_bucket = sum(1 for g in wrong_games if lo <= abs(g["margin"]) <= hi)
            pct = n_in_bucket / len(wrong_games) * 100
            print(f"  {desc:55s} {n_in_bucket:>4d} of {len(wrong_games)} wrong picks  ({pct:5.1f}%)")

    print("\n" + "=" * 78)
    print("BY REAL VOTE SPLIT (exact n_favor_winner-n_favor_loser, most common first)")
    print("=" * 78)
    from collections import Counter
    split_counts = Counter((g["n_favor_winner"], g["n_favor_loser"]) for g in games)
    for (nw, nl), cnt in sorted(split_counts.items(), key=lambda kv: -kv[1])[:15]:
        outcome = "correct" if nw > nl else ("wrong" if nw < nl else "push")
        print(f"  {nw}-{nl} split ({nw+nl} decided of 15):  {cnt} real game(s)  [{outcome}]")


all_games = build_real_completed_games(prewk_log)
report_vote_margin_analysis(all_games)
report_upset_score_margins(all_games)
