# ==============================================================================
# nfl_data_pull.py
# ==============================================================================
# Step 1 of the weekly routine (see README.txt). Pulls a fresh season of real
# nflverse play-by-play + player-level data via nflreadpy, computes every
# stat the matchup sheet needs, and writes four CSVs into this same folder:
#   nfl_team_matrix.csv        -- one row per team (PART 1-4, unchanged logic)
#   nfl_injury_report.csv      -- current injury report (PART 5, unchanged)
#   nfl_player_stats.csv       -- one row per skill player, fantasy/prop usage (PART 6, NEW)
#   nfl_def_vs_position.csv    -- fantasy points allowed by each defense, by
#                                 position, with a 1-32 matchup rank (PART 6, NEW)
#
# Nothing here is ever faked: every number is computed from real nflverse
# data. If a stat can't be computed (e.g. a rookie with no games yet), the
# row shows NaN, which the PDF builder renders as "--" rather than a guess.
#
# Run this first, every week, before nfl_matchup_sheet.py -- see README.txt.
# ==============================================================================

import argparse
import re
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Pull fresh NFL team, player, and injury data for the matchup sheet.")
parser.add_argument("--season", type=int, default=None, help="Season year to pull (default: nflverse's own current-season default).")
args = parser.parse_args()
SEASON_KWARGS = {"seasons": [args.season]} if args.season else {}

# ==============================================================================
# PART 1: PLAY ARCHIVES LOADING & TEAM RECORD GENERATION
# This section loads the raw play logs using nflreadpy and uses the final
# scores of each game to calculate true team records natively without scraping.
# ==============================================================================

print("Step 1: Loading NFL play-by-play data streams...")
pbp = nfl.load_pbp(**SEASON_KWARGS).to_pandas()

# Filter down to regular season plays and remove blank team rows
pbp_reg = pbp[(pbp['season_type'] == 'REG') & (pbp['posteam'].notna()) & (pbp['defteam'].notna())].copy()

# ==============================================================================
# WALK-FORWARD SAFETY: never let the week currently being predicted leak into
# its own "season-to-date" stats.
# ==============================================================================
# Real bug this fixes: this script has always pulled load_pbp() with no week
# ceiling -- every team-level stat below is built from EVERY real play
# nflverse has for the season SO FAR AT THE MOMENT YOU RUN THIS. That's fine
# on the normal Tuesday-morning cadence (README.txt), because the upcoming
# week's real games genuinely haven't been played yet by Tuesday, so there's
# nothing to leak. But if this ever gets rerun mid-week or after kickoff --
# testing a fix, regenerating a PDF, just curious -- and some of the CURRENT
# week's real games are already final, those results silently flow straight
# into the "pre-game" stats used to build that same week's matchup sheet and
# nfl_internal_model_track.csv log. That's exactly what happened on
# 2026-09-21: a same-day rerun to verify the season/week logging fix pulled
# stats that already included several real week-2 results, and the "week 2"
# predictions it logged (and the PDF stats it built) quietly already knew
# some of week 2's own outcomes. Confirmed directly against the real log:
# the honest, truly pre-game week-2 run (2026-09-20) called 6/16 = 37.5%
# straight-up; the contaminated same-week rerun (2026-09-21, with real final
# scores already sitting in the same log rows) called 11/16 = 68.8% and
# flipped its pick on 5 games versus the earlier run -- all 5 flips landed
# on the real winner, which a genuine pre-game re-forecast essentially never
# does by chance.
#
# Fix: compute the same real "current NFL week" (today's actual Tue-Mon
# window) nfl_matchup_sheet.py already uses to decide which week it's
# building a sheet for, then hard-cap every stat below to weeks STRICTLY
# BEFORE that week -- no matter how many of that week's real games have
# already been played by the time this happens to run. This makes a rerun
# safe on any day of the week, not just Tuesday morning, and doesn't change
# behavior at all on the normal cadence (no plays exist yet for a week that
# hasn't happened, so there's nothing to cut).
try:
    _sched_season = args.season if args.season else pd.Timestamp.now().year
    _sched = nfl.load_schedules(seasons=[_sched_season]).to_pandas()
    _sched = _sched[_sched['game_type'] == 'REG'].copy()
    _sched['gameday'] = pd.to_datetime(_sched['gameday'], errors='coerce')

    _today = pd.Timestamp.now()
    _days_since_tuesday = (_today.weekday() - 1) % 7   # Mon=0 ... Tue=1, same as nfl_matchup_sheet.py
    _week_start = _today - pd.Timedelta(days=_days_since_tuesday)
    _week_end = _week_start + pd.Timedelta(days=6)
    _this_week_sched = _sched[(_sched['gameday'] >= pd.Timestamp(_week_start.date())) &
                               (_sched['gameday'] <= pd.Timestamp(_week_end.date()))]
    if _this_week_sched.empty and not _sched.empty:
        _sched_valid = _sched.dropna(subset=['gameday']).copy()
        _sched_valid['days_from_today'] = (_sched_valid['gameday'] - pd.Timestamp(_today.date())).abs()
        _nearest_week = _sched_valid.loc[_sched_valid['days_from_today'].idxmin(), 'week'] if not _sched_valid.empty else None
        if _nearest_week is not None:
            _this_week_sched = _sched[_sched['week'] == _nearest_week]

    TARGET_WEEK = int(_this_week_sched['week'].iloc[0]) if not _this_week_sched.empty else None
except Exception as e:
    print(f" -> ⚠ Couldn't determine the real current week for the walk-forward safety cutoff ({e}); "
          f"no week filtering applied -- double-check you're only running this before that week's real "
          f"games start.")
    TARGET_WEEK = None

if TARGET_WEEK is not None and 'week' in pbp_reg.columns:
    _pre_cutoff_plays = len(pbp_reg)
    _leaked_weeks = sorted(pbp_reg.loc[pbp_reg['week'] >= TARGET_WEEK, 'week'].unique().tolist())
    pbp_reg = pbp_reg[pbp_reg['week'] < TARGET_WEEK].copy()
    if _leaked_weeks:
        print(f" -> ⚠ Real plays already existed for week(s) {_leaked_weeks} (this week's target is week "
              f"{TARGET_WEEK}) -- dropped {_pre_cutoff_plays - len(pbp_reg)} real play(s) from those weeks "
              f"so they can't leak into this week's 'pre-game' stats. If you're building week {TARGET_WEEK}'s "
              f"sheet on purpose after some of its real games are already final, that's expected -- this is "
              f"working as intended, not a bug.")
    else:
        print(f" -> Walk-forward check OK: no real plays exist yet for week {TARGET_WEEK} (this week's "
              f"target) or later -- nothing to cut.")
else:
    print(" -> ⚠ Walk-forward safety cutoff not applied (couldn't determine a real target week) -- make "
          "sure you're running this before the upcoming week's real games start.")

# Enforce numeric typing across raw value columns to protect calculation steps
numeric_fields = ['yards_gained', 'epa', 'sp', 'posteam_score', 'defteam_score', 'ydstogo', 'down', 'penalty_yards', 'pass_touchdown', 'rush_touchdown']
for field in numeric_fields:
    if field in pbp_reg.columns:
        pbp_reg[field] = pd.to_numeric(pbp_reg[field], errors='coerce').fillna(0)

# Build custom statistical evaluation indicators
pbp_reg['is_success'] = pbp_reg['epa'] > 0
pbp_reg['is_turnover'] = pbp_reg['interception'] + pbp_reg['fumble_lost']
pbp_clean = pbp_reg[(pbp_reg['play_type'].isin(['pass', 'run'])) & (pbp_reg['epa'].notna())].copy()

# Calculate specific macro game counts per team to generate real per-game averages
games_played = pbp_reg.groupby('posteam')['game_id'].nunique().reset_index().rename(columns={'posteam': 'team', 'game_id': 'gp'})

print("Step 2: Calculating team win-loss records from the scoreboards...")
final_scores = pbp_reg.groupby('game_id').agg(
    home=('home_team', 'first'),
    away=('away_team', 'first'),
    home_score=('home_score', 'last'),
    away_score=('away_score', 'last')
).reset_index()

final_scores['winner'] = np.where(final_scores['home_score'] > final_scores['away_score'], final_scores['home'],
                                  np.where(final_scores['away_score'] > final_scores['home_score'], final_scores['away'], 'TIE'))

# FIXED: the original subtracted the league-wide tie count from every team's
# losses, instead of that team's own ties -- undercounting losses for any
# team that wasn't itself involved in whichever games happened to tie.
# Now counts each team's own ties and reports the standard NFL "W-L-T" record.
records_map = {}
for team in games_played['team']:
    team_games = (final_scores['home'] == team) | (final_scores['away'] == team)
    wins = (final_scores['winner'] == team).sum()
    ties = (team_games & (final_scores['winner'] == 'TIE')).sum()
    losses = team_games.sum() - wins - ties
    records_map[team] = f"{wins}-{losses}-{ties}"

tr_data = pd.DataFrame(list(records_map.items()), columns=['team', 'record'])

# ==============================================================================
# END OF PART 1
# ==============================================================================
# ==============================================================================
# PART 2: PASSER RATING FORMULATION, SLICING & COMPREHENSIVE CSV EXPORT
# This section natively computes the official 4-part traditional NFL Passer Rating,
# aggregates advanced situational metrics, and saves the master CSV file.
# ==============================================================================

print("Step 3: Slicing situational drive and scoring data frames...")
drive_stats = pbp_reg.groupby(['game_id', 'drive']).agg(posteam=('posteam', 'first'), defteam=('defteam', 'first'), drive_yards=('yards_gained', 'sum'), fixed_drive_result=('fixed_drive_result', 'first')).reset_index()
off_drive = drive_stats.groupby('posteam').agg(yards_per_drive=('drive_yards', 'mean')).reset_index().rename(columns={'posteam': 'team'})
def_drive = drive_stats.groupby('defteam').agg(yards_per_drive_allowed=('drive_yards', 'mean')).reset_index().rename(columns={'defteam': 'team'})

# POINTS PER DRIVE (2026-10-04 -- same idea as the CFB
# project's PPD). Built off the SAME real fixed_drive_result column this
# file already uses for the red-zone TD check just below -- no new data
# source. Uses the standard TD=7 / FG=3 / Safety=-2 drive-result scoring
# convention (the common industry drive-efficiency-stat simplification --
# does NOT track the real XP/2pt conversion result of each individual
# touchdown, unlike CFB's PPD which used CFBD's real start/end score delta
# directly. Flagging that difference here rather than calling this
# "real" in the same sense CFB's PPD is).
_PTS_MAP = {'Touchdown': 7, 'Field Goal': 3, 'Safety': -2}
drive_stats['drive_points'] = drive_stats['fixed_drive_result'].map(_PTS_MAP).fillna(0)
off_ppd = drive_stats.groupby('posteam').agg(points_per_drive=('drive_points', 'mean')).reset_index().rename(columns={'posteam': 'team'})
def_ppd = drive_stats.groupby('defteam').agg(points_per_drive_allowed=('drive_points', 'mean')).reset_index().rename(columns={'defteam': 'team'})

rz_drives = pbp_reg[pbp_reg['yardline_100'] <= 20].groupby(['game_id', 'drive']).agg(posteam=('posteam', 'first'), defteam=('defteam', 'first'), is_td=('fixed_drive_result', lambda x: (x == 'Touchdown').any())).reset_index()
off_rz = rz_drives.groupby('posteam').agg(rz_drives=('is_td', 'count'), rz_tds=('is_td', 'sum')).reset_index().rename(columns={'posteam': 'team'})
off_rz['redzone_efficiency'] = np.where(off_rz['rz_drives'] > 0, (off_rz['rz_tds'] / off_rz['rz_drives']) * 100, 0)
def_rz = rz_drives.groupby('defteam').agg(rz_drives_allowed=('is_td', 'count'), rz_tds_allowed=('is_td', 'sum')).reset_index().rename(columns={'defteam': 'team'})
def_rz['redzone_efficiency_allowed'] = np.where(def_rz['rz_drives_allowed'] > 0, (def_rz['rz_tds_allowed'] / def_rz['rz_drives_allowed']) * 100, 0)

fg_plays = pbp_reg[pbp_reg['play_type'] == 'field_goal']
off_fg = fg_plays.groupby('posteam').agg(fg_att=('field_goal_result', 'count'), fg_md=('field_goal_result', lambda x: (x == 'made').sum())).reset_index().rename(columns={'posteam': 'team'})
off_fg['fg_pct'] = np.where(off_fg['fg_att'] > 0, (off_fg['fg_md'] / off_fg['fg_att']) * 100, 0)

# FIXED (again -- real-data run surfaced a second bug in this same spot):
# points ALLOWED was being computed by grouping by 'defteam' and reading
# 'defteam_score' -- but defteam_score is the DEFENDING team's OWN running
# score at that moment, not the opponent's score they allowed, so
# points_allowed_per_game was silently coming out identical to
# points_per_game for every team (confirmed on a live pull: ARI showed
# 26.0 points/game AND 26.0 points allowed/game, its own score both times
# -- which is also why AVG SCORING MARGIN was showing +0.0 across the
# board). Rebuilt from the real final score of each game (already computed
# in Part 1's final_scores) instead of re-deriving from a mid-game running-
# score column -- provably correct, not fragile to which column means what.
pts_rows = []
for _, g in final_scores.iterrows():
    pts_rows.append({'team': g['home'], 'game_pts': g['home_score'], 'game_pts_allowed': g['away_score']})
    pts_rows.append({'team': g['away'], 'game_pts': g['away_score'], 'game_pts_allowed': g['home_score']})
pts_df = pd.DataFrame(pts_rows)
off_pts = pts_df.groupby('team')['game_pts'].sum().reset_index().rename(columns={'game_pts': 'tot_pts'})
def_pts = pts_df.groupby('team')['game_pts_allowed'].sum().reset_index().rename(columns={'game_pts_allowed': 'tot_pts_allowed'})

def process_base_metrics(df, team_col, label_prefix):
    grouped = df.groupby(team_col).agg(
        total_yards=('yards_gained', 'sum'), total_plays=('play_id', 'count'), yards_per_play=('yards_gained', 'mean'), efficiency_epa=('epa', 'mean'), success_rate_pct=('is_success', 'mean'), turnovers=('is_turnover', 'sum'), third_downs=('third_down_converted', 'count'), third_down_convs=('third_down_converted', 'sum'), sacks=('sack', 'sum'), tfl=('tackled_for_loss', 'sum')
    ).reset_index().rename(columns={team_col: 'team'})
    
    rush_df = df[df['play_type'] == 'run'].groupby(team_col).agg(rushing_yards=('yards_gained', 'sum'), rush_plays=('play_id', 'count'), rushing_yards_per_play=('yards_gained', 'mean')).reset_index().rename(columns={team_col: 'team'})
    
    pass_df = df[df['play_type'] == 'pass'].groupby(team_col).agg(
        passing_yards=('yards_gained', 'sum'), pass_plays=('play_id', 'count'), yards_per_pass=('yards_gained', 'mean'),
        completions=('complete_pass', 'sum'), attempts=('pass_attempt', 'sum'), touchdowns=('pass_touchdown', 'sum'), interceptions=('interception', 'sum')
    ).reset_index().rename(columns={team_col: 'team'})
    
    # Calculate official 4-part traditional NFL Passer Rating formula natively
    pass_df['a'] = np.clip(((pass_df['completions'] / pass_df['attempts']) - 0.3) * 5, 0, 2.375)
    pass_df['b'] = np.clip(((pass_df['passing_yards'] / pass_df['attempts']) - 3) * 0.25, 0, 2.375)
    pass_df['c'] = np.clip((pass_df['touchdowns'] / pass_df['attempts']) * 20, 0, 2.375)
    pass_df['d'] = np.clip(2.375 - ((pass_df['interceptions'] / pass_df['attempts']) * 25), 0, 2.375)
    pass_df['passer_rating'] = ((pass_df['a'] + pass_df['b'] + pass_df['c'] + pass_df['d']) / 6) * 100
    pass_df = pass_df.drop(columns=['a', 'b', 'c', 'd'])
    
    merged = grouped.merge(rush_df, on='team', how='left').merge(pass_df, on='team', how='left').fillna(0)
    merged.columns = ['team'] + [f'{label_prefix}_{col}' for col in merged.columns if col != 'team']
    return merged

print("Step 4: Merging offensive and defensive array blocks into feature layers...")
offense_features = process_base_metrics(pbp_clean, 'posteam', 'off')
defense_features = process_base_metrics(pbp_clean, 'defteam', 'def')

master = games_played.merge(offense_features, on='team', how='left').merge(defense_features, on='team', how='left').merge(off_drive, on='team', how='left').merge(def_drive, on='team', how='left').merge(off_ppd, on='team', how='left').merge(def_ppd, on='team', how='left').merge(off_rz[['team', 'redzone_efficiency']], on='team', how='left').merge(def_rz[['team', 'redzone_efficiency_allowed']], on='team', how='left').merge(off_fg[['team', 'fg_pct']], on='team', how='left').merge(off_pts, on='team', how='left').merge(def_pts, on='team', how='left').merge(tr_data, on='team', how='left').fillna(0)

# Normalize volume categories into per-game stats metrics for ML models
master['total_yards_per_game'] = master['off_total_yards'] / master['gp']
master['total_yards_per_game_allowed'] = master['def_total_yards'] / master['gp']
master['passing_yards_per_game'] = master['off_passing_yards'] / master['gp']
master['passing_yards_allowed_per_game'] = master['def_passing_yards'] / master['gp']
master['rushing_yards_per_game'] = master['off_rushing_yards'] / master['gp']
master['rushing_yards_allowed_per_game'] = master['def_rushing_yards'] / master['gp']
master['points_per_game'] = master['tot_pts'] / master['gp']
master['points_allowed_per_game'] = master['tot_pts_allowed'] / master['gp']

master['third_down_pct_offense'] = (master['off_third_down_convs'] / master['off_third_downs']) * 100
master['third_down_pct_defense'] = (master['def_third_down_convs'] / master['def_third_downs']) * 100

# --- AVG 3RD DOWN DISTANCE (offense) / AVG 3RD DOWN DISTANCE FACED
# (defense): real 'ydstogo' averaged across the SAME real 3rd-down plays
# already counted for THIRD DOWN % just above (any play with a non-null
# 'third_down_converted' -- a genuine 3rd-down conversion decision, not
# every snap that happened to occur on 3rd down). A team that's efficient
# on 1st/2nd down puts itself in shorter, more manageable 3rd downs --
# lower is better for offense; forcing a LONGER distance is good defense,
# so higher is better there (see lower_is_better_fields in
# nfl_matchup_sheet.py for the offense-side convention).
third_down_plays = pbp_clean[pbp_clean['third_down_converted'].notna()]
avg_3rd_dist_off = third_down_plays.groupby('posteam')['ydstogo'].mean().reset_index().rename(
    columns={'posteam': 'team', 'ydstogo': 'avg_3rd_down_distance_offense'})
avg_3rd_dist_def = third_down_plays.groupby('defteam')['ydstogo'].mean().reset_index().rename(
    columns={'defteam': 'team', 'ydstogo': 'avg_3rd_down_distance_defense'})
master = master.merge(avg_3rd_dist_off, on='team', how='left').merge(avg_3rd_dist_def, on='team', how='left')
# NOTE: added to columns_to_export below, once that list actually exists
# (it's first defined further down, right after this whole block).

master['off_success_rate'] = master['off_success_rate_pct'] * 100
master['def_success_rate'] = master['def_success_rate_pct'] * 100
master['turnover_margin_offense'] = master['off_turnovers']
master['turnover_margin_defense'] = master['def_turnovers']

# Rename columns to maintain your target matrix features
master = master.rename(columns={'off_passer_rating': 'open_source_passer_rating'})

# CLARIFIED (these already existed as off_sacks/off_tfl/def_sacks/def_tfl --
# renamed only, nothing new computed): off_sacks = sacks taken while that
# team's offense was on the field = sacks ALLOWED. off_tfl = same idea for
# tackles for loss ALLOWED. def_sacks/def_tfl are the mirror image, made by
# that team's defense.
master = master.rename(columns={
    'off_sacks': 'sacks_allowed',
    'off_tfl': 'tfl_allowed',
    'def_sacks': 'sacks_made',
    'def_tfl': 'tfl_made',
})

columns_to_export = [
    'team', 'record', 'gp', 'points_per_game', 'points_allowed_per_game', 'total_yards_per_game', 'total_yards_per_game_allowed', 'off_yards_per_play', 'def_yards_per_play', 'passing_yards_per_game', 'passing_yards_allowed_per_game', 'off_yards_per_pass', 'def_yards_per_pass', 'rushing_yards_per_game', 'rushing_yards_allowed_per_game', 'off_rushing_yards_per_play', 'def_rushing_yards_per_play', 'off_efficiency_epa', 'def_efficiency_epa', 'off_success_rate', 'def_success_rate', 'turnover_margin_offense', 'turnover_margin_defense', 'third_down_pct_offense', 'third_down_pct_defense', 'avg_3rd_down_distance_offense', 'avg_3rd_down_distance_defense', 'fg_pct', 'redzone_efficiency', 'redzone_efficiency_allowed', 'sacks_allowed', 'sacks_made', 'tfl_allowed', 'tfl_made', 'yards_per_drive', 'yards_per_drive_allowed', 'points_per_drive', 'points_per_drive_allowed', 'open_source_passer_rating'
]

# ==============================================================================
# PART 2B: 4TH DOWN AGGRESSIVENESS RATE
# Real go-for-it rate on 4th down, in genuine decision situations only --
# excludes clear garbage time (|score differential| > 16) and 4th downs deep
# in a team's own territory (yardline_100 > 50), where punting is close to
# an automatic decision rather than a real choice. Computed across the FULL
# SEASON so far, not just the last game, since 4th-down philosophy is mostly
# a head-coach trait (stable game to game) and attempts are rare enough per
# game that a single-game sample would be too noisy to mean anything.
# ==============================================================================

print("Step 4b: Computing 4th down aggressiveness rates...")

fourth_downs = pbp_reg[
    (pbp_reg['down'] == 4) &
    (pbp_reg['yardline_100'] <= 50) &
    (pbp_reg['score_differential'].abs() <= 16) &
    (pbp_reg['play_type'].isin(['pass', 'run', 'field_goal', 'punt']))
].copy()
fourth_downs['went_for_it'] = fourth_downs['play_type'].isin(['pass', 'run'])

aggressiveness = fourth_downs.groupby('posteam').agg(
    fourth_down_attempts_considered=('went_for_it', 'count'),
    fourth_down_go_count=('went_for_it', 'sum'),
).reset_index().rename(columns={'posteam': 'team'})
aggressiveness['fourth_down_aggressiveness_pct'] = np.where(
    aggressiveness['fourth_down_attempts_considered'] > 0,
    (aggressiveness['fourth_down_go_count'] / aggressiveness['fourth_down_attempts_considered']) * 100,
    0
)

# ==============================================================================
# PART 2C: RAW COUNTING / SITUATIONAL STATS (free, computed directly from the
# same nflverse play-by-play already loaded above -- no extra network calls).
# Covers the "normal stats" half of the original stat list: penalty yards,
# plays/gm, time of possession, scoring margin, rush/pass attempts (+
# allowed), completion %, yards/completion, sack %, interceptions thrown/
# gained, fumbles (own + forced), blocked FGs, stuff rate, havoc rate, and
# the real 4th-down CONVERSION % (distinct from the go-for-it aggressiveness
# rate above -- this is what fraction of those go-for-it attempts actually
# worked). Stuff rate and havoc rate are OUR OWN formulas, clearly labeled
# below and in the README -- there's no single official definition of
# either, so these use a standard, transparent computation (not a
# proprietary one), same spirit as VOID.
# ==============================================================================

print("Step 4c: Computing raw counting and situational stats...")

# --- Penalty yards/gm (charged against this team) ---
if 'penalty_team' in pbp_reg.columns and 'penalty_yards' in pbp_reg.columns:
    pen = pbp_reg[pbp_reg['penalty_team'].notna() & (pbp_reg['penalty_team'] != '')].groupby('penalty_team')['penalty_yards'].sum().reset_index()
    pen.columns = ['team', 'tot_penalty_yards']
else:
    pen = pd.DataFrame(columns=['team', 'tot_penalty_yards'])
master = master.merge(pen, on='team', how='left')
master['tot_penalty_yards'] = master['tot_penalty_yards'].fillna(0)
master['penalty_yards_per_game'] = master['tot_penalty_yards'] / master['gp']

# --- Plays/gm (offense already has this raw count from Part 2's grouping) ---
master['plays_per_game'] = master['off_total_plays'] / master['gp']

# --- Scoring margin (this team's own PPG - PAPG; the "opposing" mirror is
# just its negative -- not a separately sourced number, just the flip side) ---
master['avg_scoring_margin'] = master['points_per_game'] - master['points_allowed_per_game']
master['opposing_avg_scoring_margin'] = -master['avg_scoring_margin']

# --- Rush/pass attempts per game (+ allowed) ---
master['rush_attempts_per_game'] = master['off_rush_plays'] / master['gp']
master['rush_attempts_per_game_allowed'] = master['def_rush_plays'] / master['gp']
master['pass_attempts_per_game'] = master['off_attempts'] / master['gp']
master['pass_attempts_per_game_allowed'] = master['def_attempts'] / master['gp']

# --- PASS PLAY % / RUSH PLAY % (play-calling mix): real play_type counts
# already computed above (off_pass_plays includes sacks/scrambles, same as
# how the industry usually defines "pass rate" -- a sack IS a called pass
# play that failed, not a run). Offense = this team's OWN tendency;
# defense = the tendency of the offenses THIS team's defense has faced.
# Rush % is just the complement (100 - pass %) on both sides.
master['pass_play_pct_offense'] = np.where(
    (master['off_pass_plays'] + master['off_rush_plays']) > 0,
    (master['off_pass_plays'] / (master['off_pass_plays'] + master['off_rush_plays'])) * 100, np.nan)
master['rush_play_pct_offense'] = 100 - master['pass_play_pct_offense']
master['pass_play_pct_defense'] = np.where(
    (master['def_pass_plays'] + master['def_rush_plays']) > 0,
    (master['def_pass_plays'] / (master['def_pass_plays'] + master['def_rush_plays'])) * 100, np.nan)
master['rush_play_pct_defense'] = 100 - master['pass_play_pct_defense']

# --- NET EFFICIENCY (EPA/play): off_efficiency_epa (this team's own
# offense, higher = better) minus def_efficiency_epa (EPA/play ALLOWED by
# this team's defense, lower = better -- already in lower_is_better_fields
# in nfl_matchup_sheet.py). Subtracting a "lower is better" number from a
# "higher is better" one this way means a bigger net is always the better
# outcome on both sides at once -- a real, simple combined signal, same
# spirit as the community's common "Net EPA/play." Not a new model, just
# the two real numbers already on this sheet combined.
master['net_efficiency_epa'] = master['off_efficiency_epa'] - master['def_efficiency_epa']

# --- TRUE Completion % (+ allowed): a QB spiking the ball to stop the
# clock counts as a real pass_attempt in nflverse's raw data but was never
# going to be a completion -- unfairly drags completion % down for teams
# that spike more. Real 'qb_spike' column, no charting needed to exclude
# it. This can't ALSO exclude throwaways or batted-at-the-line passes the
# way the full industry "adjusted completion %" does -- nflverse has no
# structured column for either (checked directly against the real pbp data
# dictionary), same wall as the PFR-sourced version already removed -- see
# README.txt. So this is real, but a NARROWER "true" completion % than the
# full version: spikes excluded from the denominator, throwaways/batted
# passes are not (and can't be, from any free/licensed source). Since
# spikes are rare (a handful per team per season), expect this to move the
# number by a fraction of a point, not reshuffle rankings.
if 'qb_spike' in pbp_clean.columns:
    pbp_clean['qb_spike'] = pd.to_numeric(pbp_clean['qb_spike'], errors='coerce').fillna(0)
else:
    pbp_clean['qb_spike'] = 0
pass_no_spike = pbp_clean[(pbp_clean['play_type'] == 'pass') & (pbp_clean['qb_spike'] == 0)]
true_attempts_off = pass_no_spike.groupby('posteam')['pass_attempt'].sum().reset_index().rename(
    columns={'posteam': 'team', 'pass_attempt': 'off_attempts_true'})
true_attempts_def = pass_no_spike.groupby('defteam')['pass_attempt'].sum().reset_index().rename(
    columns={'defteam': 'team', 'pass_attempt': 'def_attempts_true'})
master = master.merge(true_attempts_off, on='team', how='left').merge(true_attempts_def, on='team', how='left')
master['off_attempts_true'] = master['off_attempts_true'].fillna(0)
master['def_attempts_true'] = master['def_attempts_true'].fillna(0)

master['completion_pct'] = np.where(master['off_attempts_true'] > 0, (master['off_completions'] / master['off_attempts_true']) * 100, 0)
master['completion_pct_allowed'] = np.where(master['def_attempts_true'] > 0, (master['def_completions'] / master['def_attempts_true']) * 100, 0)
master['yards_per_completion'] = np.where(master['off_completions'] > 0, master['off_passing_yards'] / master['off_completions'], 0)
master['yards_per_completion_allowed'] = np.where(master['def_completions'] > 0, master['def_passing_yards'] / master['def_completions'], 0)

# --- MISCUE RATE (+ allowed): a REVERSE BOX SCORE estimate -- attempts
# minus completions minus interceptions minus spikes (spikes already
# excluded via pass_no_spike above) -- i.e. every passing play that didn't
# end in a clean completion, a pick, or a clock-stopping spike. This is
# deliberately NOT labeled "drop %" or "bad throw %": free data has no
# column anywhere that says WHICH of four different things happened on a
# given incompletion -- a receiver drop, an inaccurate throw, an
# intentional throwaway, or a pass batted down at the line. All four
# collapse into this one leftover bucket, on purpose, because separating
# them needs PFR-style manual charting (removed -- see LICENSING in
# README.txt) that no free/licensed source provides. Real numerator/
# denominator, honestly ambiguous label -- see "MISCUE RATE" in
# README.txt. Cross-check it against real CPOE (completion % over
# expected, already on this sheet) in the PDF -- see nfl_matchup_sheet.py.
if 'interception' in pass_no_spike.columns:
    ints_off = pass_no_spike.groupby('posteam')['interception'].sum().reset_index().rename(
        columns={'posteam': 'team', 'interception': 'off_ints_true'})
    ints_def = pass_no_spike.groupby('defteam')['interception'].sum().reset_index().rename(
        columns={'defteam': 'team', 'interception': 'def_ints_true'})
    master = master.merge(ints_off, on='team', how='left').merge(ints_def, on='team', how='left')
else:
    master['off_ints_true'] = 0
    master['def_ints_true'] = 0
master['off_ints_true'] = master['off_ints_true'].fillna(0)
master['def_ints_true'] = master['def_ints_true'].fillna(0)

# Sanity check on the arithmetic assumption that every true pass attempt is
# EXACTLY one of completed / incomplete / intercepted (no overlap) -- this
# has to hold for the subtraction below to mean what it says. Same
# assumption the passer rating formula above already relies on implicitly;
# checked explicitly here so a future nflverse schema change shows up as a
# printed warning instead of a silently wrong number.
_miscue_off_raw = master['off_attempts_true'] - master['off_completions'] - master['off_ints_true']
_miscue_def_raw = master['def_attempts_true'] - master['def_completions'] - master['def_ints_true']
_bad_off = (_miscue_off_raw < -0.5).sum()
_bad_def = (_miscue_def_raw < -0.5).sum()
if _bad_off or _bad_def:
    print(f"  ⚠ MISCUE RATE sanity check: {_bad_off} offense / {_bad_def} defense team-rows show a "
          f"negative leftover before flooring -- the completions+interceptions <= attempts assumption "
          f"didn't hold for those teams this pull; miscue rate floored at 0 for them rather than shown negative.")
master['miscue_count_off'] = _miscue_off_raw.clip(lower=0)
master['miscue_count_def'] = _miscue_def_raw.clip(lower=0)
master['miscue_rate_offense'] = np.where(master['off_attempts_true'] > 0, (master['miscue_count_off'] / master['off_attempts_true']) * 100, 0)
master['miscue_rate_defense'] = np.where(master['def_attempts_true'] > 0, (master['miscue_count_def'] / master['def_attempts_true']) * 100, 0)

# --- Sack % (of dropbacks: attempts + sacks), offense taken / defense forced ---
master['sack_pct_offense'] = np.where((master['off_attempts'] + master['sacks_allowed']) > 0, (master['sacks_allowed'] / (master['off_attempts'] + master['sacks_allowed'])) * 100, 0)
master['sack_pct_defense'] = np.where((master['def_attempts'] + master['sacks_made']) > 0, (master['sacks_made'] / (master['def_attempts'] + master['sacks_made'])) * 100, 0)

# --- QB pressure rate (offense allowed / defense forced): OUR OWN free
# proxy, computed straight from real pbp -- (sacks + QB hits) / dropbacks.
# Replaces a PFR-sourced "pressure %" that used to live here (PFR's own
# pressure charting has no free/resellable license -- see README.txt). This
# will run LOWER than PFR's real number since it only catches pressures
# that ended in a sack or a recorded hit, not near-misses PFR's charters
# also count -- it's a real, honestly-labeled proxy for O-line/D-line
# pressure, not a reproduction of PFR's proprietary metric. ---
if 'qb_hit' in pbp_reg.columns:
    pbp_reg['qb_hit'] = pd.to_numeric(pbp_reg['qb_hit'], errors='coerce').fillna(0)
else:
    pbp_reg['qb_hit'] = 0
pressure_off = pbp_reg.groupby('posteam').agg(
    dropback_attempts=('pass_attempt', 'sum'), sacks_n=('sack', 'sum'), hits_n=('qb_hit', 'sum')
).reset_index().rename(columns={'posteam': 'team'})
pressure_off['dropbacks'] = pressure_off['dropback_attempts'] + pressure_off['sacks_n']
pressure_off['pressure_rate_allowed_pct'] = np.where(
    pressure_off['dropbacks'] > 0, ((pressure_off['sacks_n'] + pressure_off['hits_n']) / pressure_off['dropbacks']) * 100, 0
)
pressure_def = pbp_reg.groupby('defteam').agg(
    dropback_attempts_faced=('pass_attempt', 'sum'), sacks_forced_n=('sack', 'sum'), hits_forced_n=('qb_hit', 'sum')
).reset_index().rename(columns={'defteam': 'team'})
pressure_def['dropbacks_faced'] = pressure_def['dropback_attempts_faced'] + pressure_def['sacks_forced_n']
pressure_def['pressure_rate_forced_pct'] = np.where(
    pressure_def['dropbacks_faced'] > 0, ((pressure_def['sacks_forced_n'] + pressure_def['hits_forced_n']) / pressure_def['dropbacks_faced']) * 100, 0
)
master = master.merge(pressure_off[['team', 'pressure_rate_allowed_pct']], on='team', how='left').merge(
    pressure_def[['team', 'pressure_rate_forced_pct']], on='team', how='left')

# --- "TRENCH BATTLE" panel stats: Pass Blocking Efficiency (PBE, offense)
# vs. Pass Rush Productivity (PRP, defense) -- a SECOND, differently-
# weighted read on the same real sack/QB-hit data used above for
# pressure_rate_allowed_pct/forced_pct, closer to PFF's real published PBE
# weighting (1x sacks, 0.75x QB hits) and framed as an "efficiency" score
# (100 = never pressured) rather than a raw pressure-allowed rate. Hurries
# are NOT included in either number -- same real-data wall as PART 2F's Tug
# o' War / Edge Pressure index: no free/licensed source tags an individual
# hurry to a team in real time (was_pressure is blacked out for the season
# still in progress, and even historically only gives a combined
# sacks+hits+hurries count, not hurries in isolation). Real, honestly
# labeled, but narrower than PFF's real PBE.
pressure_off['weighted_pressure_allowed'] = pressure_off['sacks_n'] + 0.75 * pressure_off['hits_n']
pressure_off['pbe_score'] = np.where(
    pressure_off['dropbacks'] > 0, 100 - (pressure_off['weighted_pressure_allowed'] / pressure_off['dropbacks'] * 100), np.nan
)
pressure_def['weighted_pressure_forced'] = pressure_def['sacks_forced_n'] + 0.75 * pressure_def['hits_forced_n']
pressure_def['prp_score'] = np.where(
    pressure_def['dropbacks_faced'] > 0, (pressure_def['weighted_pressure_forced'] / pressure_def['dropbacks_faced'] * 100), np.nan
)
master = master.merge(pressure_off[['team', 'pbe_score']], on='team', how='left').merge(
    pressure_def[['team', 'prp_score']], on='team', how='left')
columns_to_export += ['pbe_score', 'prp_score']

# --- Run Success Rate (offense) / Run Stop Rate (defense): same EPA>0
# "success" definition already used for off_success_rate/def_success_rate
# above, just filtered to rushing plays only. Defense side is framed as the
# share of rush attempts faced that did NOT succeed, so both sides of this
# pair read "higher is better" -- consistent with the rest of the sheet's
# offense-vs-defense-allowed pairs.
run_plays_off = pbp_clean[pbp_clean['play_type'] == 'run'].groupby('posteam')['is_success'].mean().reset_index().rename(
    columns={'posteam': 'team', 'is_success': 'off_run_success_rate'})
run_plays_def = pbp_clean[pbp_clean['play_type'] == 'run'].groupby('defteam')['is_success'].mean().reset_index().rename(
    columns={'defteam': 'team', 'is_success': 'def_run_success_allowed_rate'})
master = master.merge(run_plays_off, on='team', how='left').merge(run_plays_def, on='team', how='left')
master['off_run_success_rate'] = master['off_run_success_rate'] * 100
master['def_run_stop_rate'] = (1 - master['def_run_success_allowed_rate']) * 100
columns_to_export += ['off_run_success_rate', 'def_run_stop_rate']

# Rush EPA/Play (offense) / Rush EPA/Play Allowed (defense). Same real
# 'epa' column already used for off_efficiency_epa/def_efficiency_epa
# above, same pbp_clean source, just filtered to rushing plays only -- the
# SAME play_type == 'run' filter as Run Success Rate/Run Stop Rate right
# above. Tested a real 56.2% walk-forward win rate (offense side) in the
# earlier "trenches" backtest round (see nfl_walkforward_backtest.py's
# off_rush_epa_game/def_rush_epa_allowed_game and the NFL_QUALITY_STATS
# comment in nfl_matchup_sheet.py) -- cleared the same >55% real bar as Run
# Success Rate/PBE/Rush YPP Allowed did that round, but wasn't exported to
# nfl_team_matrix.csv until now since no real season-to-date column existed
# yet. Defense side (Rush EPA/Play Allowed) was never separately backtested
# -- added here for the real offense/defense crossed-pair shape every other
# stat on the sheet already has, not because it independently cleared the
# quality-stat bar.
off_rush_epa = pbp_clean[pbp_clean['play_type'] == 'run'].groupby('posteam')['epa'].mean().reset_index().rename(
    columns={'posteam': 'team', 'epa': 'off_rush_epa'})
def_rush_epa = pbp_clean[pbp_clean['play_type'] == 'run'].groupby('defteam')['epa'].mean().reset_index().rename(
    columns={'defteam': 'team', 'epa': 'def_rush_epa_allowed'})
master = master.merge(off_rush_epa, on='team', how='left').merge(def_rush_epa, on='team', how='left')
columns_to_export += ['off_rush_epa', 'def_rush_epa_allowed']

# LINE YARDS (offense) / LINE YARDS ALLOWED (defense), 2026-10-04. CFB's own
# "Off Line Yards"/"Def Line Yards" (cfb_working_schedule.py) comes straight
# from CollegeFootballData's API as a precomputed field --
# nflverse/nflreadpy has no equivalent precomputed stat for the NFL, so
# there's no free source to just read this one off of. Built here instead
# from real play-by-play using the standard, publicly documented "Adjusted
# Line Yards" formula (Football Outsiders' published methodology, not
# anyone's proprietary number): on every real rush attempt, credit the
# O-line with a share of the real yards gained --
#   losses                   -> 1.20x the (negative) yards
#   0 to 4 yards              -> 100% credit
#   5 to 10 yards             -> 50% credit (the back's breakaway speed,
#                                not the O-line, explains yardage past 4)
#   11+ yards                 -> credit caps at the 5-10 tier's value --
#                                a long run past that is almost entirely on
#                                the ball carrier, not the line
# QB scrambles are excluded (same real 'qb_scramble' flag nflverse's own
# pbp already carries) -- an O-line's run-blocking has nothing to do with
# a quarterback improvising after a busted pass play. Same pbp_clean/
# play_type == 'run' source as Run Success Rate/Rush EPA just above,
# averaged per real rush attempt, offense side and the opposing defense's
# allowed side.
def _line_yards_credit(y):
    if y < 0:
        return y * 1.20
    elif y <= 4:
        return y
    elif y <= 10:
        return 4 + (y - 4) * 0.50
    else:
        return 4 + (10 - 4) * 0.50  # capped -- same credit as a 10-yard gain


_ly_runs = pbp_clean[pbp_clean['play_type'] == 'run'].copy()
if 'qb_scramble' in _ly_runs.columns:
    _ly_runs = _ly_runs[_ly_runs['qb_scramble'] != 1]
_ly_runs['line_yards_credit'] = _ly_runs['yards_gained'].apply(_line_yards_credit)
off_line_yards = _ly_runs.groupby('posteam')['line_yards_credit'].mean().reset_index().rename(
    columns={'posteam': 'team', 'line_yards_credit': 'off_line_yards'})
def_line_yards = _ly_runs.groupby('defteam')['line_yards_credit'].mean().reset_index().rename(
    columns={'defteam': 'team', 'line_yards_credit': 'def_line_yards_allowed'})
master = master.merge(off_line_yards, on='team', how='left').merge(def_line_yards, on='team', how='left')
columns_to_export += ['off_line_yards', 'def_line_yards_allowed']


# --- SECONDARY LOCK INDEX (defense): 100 minus real yards allowed per real
# target faced (the spike-excluded true-attempts base already built for
# True Completion % above). Real numerator (yards allowed), real
# denominator (targets faced), re-expressed as "100 minus" only so a
# tighter defense reads as a HIGHER number, matching this sheet's
# higher-is-better convention everywhere else. Expect a narrow real range
# (roughly 91-95) -- yards/target allowed itself only ranges ~5-9 across
# the league, and this isn't artificially stretched to look more dramatic
# than the real data actually is.
master['yards_per_target_allowed'] = np.where(master['def_attempts_true'] > 0, master['def_passing_yards'] / master['def_attempts_true'], np.nan)
master['secondary_lock_index'] = 100 - master['yards_per_target_allowed']
columns_to_export += ['secondary_lock_index']

# --- YAC/reception (offense) + YAC allowed/reception (defense) -- computed
# fresh from real pbp's own 'yards_after_catch' column, NOT the NGS-sourced
# 'team_yac_per_reception' already elsewhere on this sheet (PART 6B). Built
# from the SAME real source on both sides, on purpose: mixing an NGS-
# tracking-derived offense number against a pbp-derived defense number
# would be comparing two different measurement methods and calling it one
# stat. This pbp-based pair only exists for the SKILL BATTLE panel
# (nfl_matchup_sheet.py), where an apples-to-apples offense/defense
# comparison matters more than matching the NGS-sourced number elsewhere.
if 'yards_after_catch' in pbp_clean.columns:
    pass_plays_yac = pbp_clean[pbp_clean['play_type'] == 'pass'].copy()
    pass_plays_yac['yards_after_catch'] = pd.to_numeric(pass_plays_yac['yards_after_catch'], errors='coerce').fillna(0)
    yac_off = pass_plays_yac.groupby('posteam').agg(
        tot_yac_off=('yards_after_catch', 'sum'), tot_rec_off=('complete_pass', 'sum')
    ).reset_index().rename(columns={'posteam': 'team'})
    yac_def = pass_plays_yac.groupby('defteam').agg(
        tot_yac_def=('yards_after_catch', 'sum'), tot_rec_def=('complete_pass', 'sum')
    ).reset_index().rename(columns={'defteam': 'team'})
    master = master.merge(yac_off, on='team', how='left').merge(yac_def, on='team', how='left')
else:
    master['tot_yac_off'] = np.nan
    master['tot_rec_off'] = np.nan
    master['tot_yac_def'] = np.nan
    master['tot_rec_def'] = np.nan
master['yac_per_reception_pbp'] = np.where(master['tot_rec_off'] > 0, master['tot_yac_off'] / master['tot_rec_off'], np.nan)
master['yac_allowed_per_reception_pbp'] = np.where(master['tot_rec_def'] > 0, master['tot_yac_def'] / master['tot_rec_def'], np.nan)
columns_to_export += ['yac_per_reception_pbp', 'yac_allowed_per_reception_pbp']

# --- Interceptions thrown/gm (offense) + gained/gm (defense) ---
master['interceptions_thrown_per_game'] = master['off_interceptions'] / master['gp']
master['interceptions_gained_per_game'] = master['def_interceptions'] / master['gp']

# --- Fumbles: own lost/gm (offense) + forced/gm (defense) ---
if 'fumble_lost' in pbp_clean.columns:
    fum_off = pbp_clean.groupby('posteam')['fumble_lost'].sum().reset_index().rename(columns={'posteam': 'team', 'fumble_lost': 'tot_fumbles_lost_off'})
    fum_def = pbp_clean.groupby('defteam')['fumble_lost'].sum().reset_index().rename(columns={'defteam': 'team', 'fumble_lost': 'tot_fumbles_forced_def'})
else:
    fum_off = pd.DataFrame(columns=['team', 'tot_fumbles_lost_off'])
    fum_def = pd.DataFrame(columns=['team', 'tot_fumbles_forced_def'])
master = master.merge(fum_off, on='team', how='left').merge(fum_def, on='team', how='left')
master['tot_fumbles_lost_off'] = master['tot_fumbles_lost_off'].fillna(0)
master['tot_fumbles_forced_def'] = master['tot_fumbles_forced_def'].fillna(0)
master['fumbles_lost_per_game'] = master['tot_fumbles_lost_off'] / master['gp']
master['opponent_fumbles_forced_per_game'] = master['tot_fumbles_forced_def'] / master['gp']

# --- FG blocked/gm ---
if 'field_goal_result' in fg_plays.columns:
    fg_blk = fg_plays[fg_plays['field_goal_result'] == 'blocked'].groupby('posteam').size().reset_index(name='tot_fg_blocked')
    fg_blk.columns = ['team', 'tot_fg_blocked']
else:
    fg_blk = pd.DataFrame(columns=['team', 'tot_fg_blocked'])
master = master.merge(fg_blk, on='team', how='left')
master['tot_fg_blocked'] = master['tot_fg_blocked'].fillna(0)
master['fg_blocked_per_game'] = master['tot_fg_blocked'] / master['gp']

# --- FG blocks FORCED/gm (defense side -- the mirror of FG BLOCKED above,
# which was already one-sided: this team's OWN attempts getting blocked).
# Real 'field_goal_result'=='blocked', grouped by DEFTEAM instead of
# posteam -- how many of the OTHER team's FG attempts did this defense
# block. ---
if 'field_goal_result' in fg_plays.columns:
    fg_blk_forced = fg_plays[fg_plays['field_goal_result'] == 'blocked'].groupby('defteam').size().reset_index(name='tot_fg_blocks_forced')
    fg_blk_forced.columns = ['team', 'tot_fg_blocks_forced']
else:
    fg_blk_forced = pd.DataFrame(columns=['team', 'tot_fg_blocks_forced'])
master = master.merge(fg_blk_forced, on='team', how='left')
master['tot_fg_blocks_forced'] = master['tot_fg_blocks_forced'].fillna(0)
master['fg_blocks_forced_per_game'] = master['tot_fg_blocks_forced'] / master['gp']

# ==============================================================================
# SPECIAL TEAMS: punts/gm, avg punt distance, punt return avg, avg FG
# distance, punt blocks (+ forced), special teams TDs. All real, direct
# from nflverse pbp's own punt/field-goal columns -- 'kick_distance',
# 'punt_blocked', 'return_yards', 'return_touchdown', 'punt_fair_catch',
# 'punt_downed', 'punt_out_of_bounds' are all real, standard, documented
# nflverse fields (verified against nflfastR's own field reference), no
# charting or proprietary model needed for any of this, unlike BLITZ RATE
# and ON-TARGET % below.
# ==============================================================================
punt_plays = pbp_reg[pbp_reg['play_type'] == 'punt'].copy()
# Boolean/indicator flags default to 0 ("no") when a column is entirely
# missing from a pull -- a defensible degrade (same pattern as qb_spike
# elsewhere), since these only ever gate WHICH plays count. Continuous
# yardage columns are handled separately below and NEVER default to 0 --
# a missing 'kick_distance' column defaulting to 0 would silently average
# out to a fake "0.0 yards/punt" for every team instead of a real
# unavailable ('--'). Caught this exact bug in testing before it shipped.
for c in ['punt_blocked', 'return_touchdown', 'punt_fair_catch', 'punt_downed', 'punt_out_of_bounds']:
    if c in punt_plays.columns:
        punt_plays[c] = pd.to_numeric(punt_plays[c], errors='coerce').fillna(0)
    else:
        punt_plays[c] = 0
for c in ['kick_distance', 'return_yards']:
    if c in punt_plays.columns:
        punt_plays[c] = pd.to_numeric(punt_plays[c], errors='coerce')
    else:
        punt_plays[c] = np.nan
# Returning team: real 'return_team' column when nflverse provides it (more
# reliable than assuming defteam always means "the team receiving the
# punt" -- verified against nflfastR's own field reference, same column
# used below for kickoff return attribution); falls back to defteam if a
# future pull is missing it, same defensive pattern as everywhere else.
punt_ret_col = 'return_team' if 'return_team' in punt_plays.columns else 'defteam'

# Punts/gm (kicking team = posteam on a punt play)
punts_off = punt_plays.groupby('posteam').size().reset_index(name='tot_punts').rename(columns={'posteam': 'team'})
master = master.merge(punts_off, on='team', how='left')
master['tot_punts'] = master['tot_punts'].fillna(0)
master['punts_per_game'] = master['tot_punts'] / master['gp']

# Avg punt distance -- excludes blocked punts (no real "distance" for a
# punt that never got off), real kick_distance on the rest.
punt_ok = punt_plays[punt_plays['punt_blocked'] == 0]
avg_punt_dist = punt_ok.groupby('posteam')['kick_distance'].mean().reset_index().rename(
    columns={'posteam': 'team', 'kick_distance': 'avg_punt_distance'})
master = master.merge(avg_punt_dist, on='team', how='left')

# Punt return avg -- the RECEIVING/returning team is 'defteam' on a punt
# play. Restricted to punts that were actually returnable (not a fair
# catch, not downed, not out of bounds, not blocked) -- a real, standard
# filter for "yards per punt return," not an estimate.
punt_returnable = punt_plays[
    (punt_plays['punt_blocked'] == 0) & (punt_plays['punt_fair_catch'] == 0) &
    (punt_plays['punt_downed'] == 0) & (punt_plays['punt_out_of_bounds'] == 0)
]
punt_ret_avg = punt_returnable.groupby(punt_ret_col)['return_yards'].mean().reset_index().rename(
    columns={punt_ret_col: 'team', 'return_yards': 'punt_return_avg'})
master = master.merge(punt_ret_avg, on='team', how='left')

# Punts blocked (this team's OWN punts getting blocked) + forced (this
# team's defense blocking the OTHER team's punt) -- same posteam/defteam
# mirror pattern as FG blocks above.
punt_blk_off = punt_plays[punt_plays['punt_blocked'] == 1].groupby('posteam').size().reset_index(name='tot_punts_blocked').rename(columns={'posteam': 'team'})
punt_blk_def = punt_plays[punt_plays['punt_blocked'] == 1].groupby('defteam').size().reset_index(name='tot_punt_blocks_forced').rename(columns={'defteam': 'team'})
master = master.merge(punt_blk_off, on='team', how='left').merge(punt_blk_def, on='team', how='left')
master['tot_punts_blocked'] = master['tot_punts_blocked'].fillna(0)
master['tot_punt_blocks_forced'] = master['tot_punt_blocks_forced'].fillna(0)
master['punts_blocked_per_game'] = master['tot_punts_blocked'] / master['gp']
master['punt_blocks_forced_per_game'] = master['tot_punt_blocks_forced'] / master['gp']

# Avg FG distance -- ALL real attempts (made, missed, AND blocked all carry
# a real 'kick_distance' for the attempt itself, regardless of outcome) --
# reflects a team's actual field goal range/aggressiveness, not just makes.
avg_fg_dist = fg_plays.copy()
if 'kick_distance' in avg_fg_dist.columns:
    avg_fg_dist['kick_distance'] = pd.to_numeric(avg_fg_dist['kick_distance'], errors='coerce')
    avg_fg_dist = avg_fg_dist.groupby('posteam')['kick_distance'].mean().reset_index().rename(
        columns={'posteam': 'team', 'kick_distance': 'avg_fg_distance'})
else:
    avg_fg_dist = pd.DataFrame(columns=['team', 'avg_fg_distance'])
master = master.merge(avg_fg_dist, on='team', how='left')

# Special teams TDs -- punt return TDs (defteam on a punt play) + kickoff
# return TDs (real 'return_touchdown' on kickoff plays). NOTE: this counts
# RETURN touchdowns only (punt/kickoff returned all the way) -- a TD off a
# recovered/returned BLOCKED kick is a separate, much rarer event that
# isn't cleanly isolated by these same columns, so it's not included here;
# this is a real, slightly narrower "special teams TD" count, not an
# estimate of the broader category.
kickoff_plays = pbp_reg[pbp_reg['play_type'] == 'kickoff'].copy()
if 'return_touchdown' in kickoff_plays.columns:
    kickoff_plays['return_touchdown'] = pd.to_numeric(kickoff_plays['return_touchdown'], errors='coerce').fillna(0)
    # Kickoff returns are credited to whichever team is RECEIVING -- verified
    # against real pbp: 'return_team' identifies the returning team directly
    # on kickoff plays (more reliable than assuming posteam/defteam meaning
    # for kickoffs specifically, since that convention isn't as consistent
    # as it is for punts).
    kret_col = 'return_team' if 'return_team' in kickoff_plays.columns else 'defteam'
    kr_td = kickoff_plays[kickoff_plays['return_touchdown'] == 1].groupby(kret_col).size().reset_index(name='kr_tds').rename(columns={kret_col: 'team'})
else:
    kr_td = pd.DataFrame(columns=['team', 'kr_tds'])
pr_td = punt_plays[punt_plays['return_touchdown'] == 1].groupby(punt_ret_col).size().reset_index(name='pr_tds').rename(columns={punt_ret_col: 'team'})
master = master.merge(kr_td, on='team', how='left').merge(pr_td, on='team', how='left')
master['kr_tds'] = master['kr_tds'].fillna(0)
master['pr_tds'] = master['pr_tds'].fillna(0)
master['special_teams_tds_per_game'] = (master['kr_tds'] + master['pr_tds']) / master['gp']

# --- Stuff rate: rush plays stopped at or behind the line (yards_gained<=0)
# / total rush plays. Offense side = how often THIS team's run game gets
# stuffed (lower is better); defense side = how often it stuffs opponents
# (higher is better). Our own formula (no single official definition). ---
rush_plays_all = pbp_clean[pbp_clean['play_type'] == 'run']
stuff_off = rush_plays_all.groupby('posteam').agg(rush_plays_n=('play_id', 'count'), stuffed_n=('yards_gained', lambda x: (x <= 0).sum())).reset_index().rename(columns={'posteam': 'team'})
stuff_off['stuff_rate_offense'] = np.where(stuff_off['rush_plays_n'] > 0, (stuff_off['stuffed_n'] / stuff_off['rush_plays_n']) * 100, 0)
stuff_def = rush_plays_all.groupby('defteam').agg(rush_plays_faced_n=('play_id', 'count'), stuffed_forced_n=('yards_gained', lambda x: (x <= 0).sum())).reset_index().rename(columns={'defteam': 'team'})
stuff_def['stuff_rate_defense'] = np.where(stuff_def['rush_plays_faced_n'] > 0, (stuff_def['stuffed_forced_n'] / stuff_def['rush_plays_faced_n']) * 100, 0)
master = master.merge(stuff_off[['team', 'stuff_rate_offense']], on='team', how='left').merge(stuff_def[['team', 'stuff_rate_defense']], on='team', how='left')

# --- Havoc rate (defense): (TFL + sacks + INTs gained + fumbles forced) /
# total defensive plays. OUR OWN formula -- there's no single industry-
# standard havoc definition; this is the common "disruptive defensive play"
# combination used by several public analytics writeups. ---
master['havoc_rate_defense'] = np.where(
    master['def_total_plays'] > 0,
    ((master['tfl_made'] + master['sacks_made'] + master['def_interceptions'] + master['tot_fumbles_forced_def']) / master['def_total_plays']) * 100,
    0
)

# --- Plays faced / game (defense side of plays_per_game, added for the
# TEAM RUNDOWN DEFENSE panel) -- real, same def_total_plays already used
# for havoc rate above, just normalized per game. ---
master['def_plays_per_game'] = master['def_total_plays'] / master['gp']

# --- Real 4th-down CONVERSION % (of the go-for-it attempts already isolated
# in Part 2B, what fraction actually picked up the first down / scored) ---
fd_go = fourth_downs[fourth_downs['went_for_it']].copy()
fd_go['converted'] = (fd_go['yards_gained'] >= fd_go['ydstogo']) | (fd_go.get('pass_touchdown', 0) == 1) | (fd_go.get('rush_touchdown', 0) == 1)
fourth_conv = fd_go.groupby('posteam').agg(fourth_down_go_attempts=('converted', 'count'), fourth_down_conversions=('converted', 'sum')).reset_index().rename(columns={'posteam': 'team'})
fourth_conv['fourth_down_conversion_pct'] = np.where(fourth_conv['fourth_down_go_attempts'] > 0, (fourth_conv['fourth_down_conversions'] / fourth_conv['fourth_down_go_attempts']) * 100, 0)
master = master.merge(fourth_conv[['team', 'fourth_down_conversion_pct']], on='team', how='left')

# --- Real 4th-down conversion % ALLOWED (defense side): the SAME real
# go-for-it attempts (fd_go) above, just re-attributed by grouping on
# 'defteam' instead of 'posteam' -- same re-attribution technique already
# used elsewhere in this pipeline (e.g. catch_pct_allowed), not a new model.
fourth_conv_allowed = fd_go.groupby('defteam').agg(
    fourth_down_go_attempts_faced=('converted', 'count'),
    fourth_down_conversions_allowed=('converted', 'sum')
).reset_index().rename(columns={'defteam': 'team'})
fourth_conv_allowed['fourth_down_conversion_pct_allowed'] = np.where(
    fourth_conv_allowed['fourth_down_go_attempts_faced'] > 0,
    (fourth_conv_allowed['fourth_down_conversions_allowed'] / fourth_conv_allowed['fourth_down_go_attempts_faced']) * 100, 0
)
master = master.merge(fourth_conv_allowed[['team', 'fourth_down_conversion_pct_allowed']], on='team', how='left')

# --- Time of possession/gm. nflverse's pbp carries per-drive TOP as a
# "MM:SS" string on 'drive_time_of_possession' -- parsed to seconds here,
# summed per game, then averaged across games played. Some nflverse
# releases omit this column; wrapped so it shows '--' rather than crashing. ---
if 'drive_time_of_possession' in pbp_reg.columns:
    def _top_to_seconds(val):
        try:
            m, s = str(val).split(':')
            return int(m) * 60 + int(s)
        except (ValueError, AttributeError):
            return np.nan
    top_drives = pbp_reg.groupby(['game_id', 'posteam', 'drive'])['drive_time_of_possession'].first().reset_index()
    top_drives['top_seconds'] = top_drives['drive_time_of_possession'].apply(_top_to_seconds)
    top_per_game = top_drives.groupby(['game_id', 'posteam'])['top_seconds'].sum().reset_index()
    top_avg = top_per_game.groupby('posteam')['top_seconds'].mean().reset_index().rename(columns={'posteam': 'team', 'top_seconds': 'time_of_possession_seconds_per_game'})
    master = master.merge(top_avg, on='team', how='left')
else:
    master['time_of_possession_seconds_per_game'] = np.nan
master['time_of_possession_minutes_per_game'] = master['time_of_possession_seconds_per_game'] / 60

columns_to_export += [
    'penalty_yards_per_game', 'plays_per_game', 'avg_scoring_margin', 'opposing_avg_scoring_margin',
    'rush_attempts_per_game', 'rush_attempts_per_game_allowed', 'pass_attempts_per_game', 'pass_attempts_per_game_allowed',
    'pass_play_pct_offense', 'rush_play_pct_offense', 'pass_play_pct_defense', 'rush_play_pct_defense',
    'net_efficiency_epa',
    'completion_pct', 'completion_pct_allowed', 'yards_per_completion', 'yards_per_completion_allowed',
    'miscue_rate_offense', 'miscue_rate_defense',
    'sack_pct_offense', 'sack_pct_defense', 'interceptions_thrown_per_game', 'interceptions_gained_per_game',
    'fumbles_lost_per_game', 'opponent_fumbles_forced_per_game', 'fg_blocked_per_game', 'fg_blocks_forced_per_game',
    'stuff_rate_offense', 'stuff_rate_defense', 'havoc_rate_defense', 'fourth_down_conversion_pct',
    'fourth_down_conversion_pct_allowed', 'def_plays_per_game',
    'time_of_possession_minutes_per_game', 'pressure_rate_allowed_pct', 'pressure_rate_forced_pct',
    'punts_per_game', 'avg_punt_distance', 'punt_return_avg', 'avg_fg_distance',
    'punts_blocked_per_game', 'punt_blocks_forced_per_game', 'special_teams_tds_per_game',
]

# ==============================================================================
# END OF PART 2C
# ==============================================================================
# ==============================================================================
# PART 2D: OFFENSIVE/DEFENSIVE SCHEME TENDENCIES -- PERSONNEL GROUPINGS +
# FORMATION (shotgun vs. under center) + DEFENSIVE PERSONNEL (base/nickel/
# dime). Real, free, nflverse-licensed data (CC BY 4.0) via nflreadpy's
# load_participation() -- NFL's own tracking-derived personnel/formation
# data, republished by nflverse, same license as everything else in this
# pipeline. This covers the same ground as Sharp Football
# Analysis, Sumer Sports, and NFL Savant -- all three explicitly prohibit
# commercial use in their own terms (see README.txt "LICENSING") -- for
# free, with a clean, resellable license, straight from the source those
# sites' own personnel/formation numbers ultimately come from anyway.
#
# offense_personnel is a raw string like "1 RB, 1 TE, 3 WR" -- parsed below
# into the standard "XY personnel" shorthand (X = # RB, Y = # TE) used
# throughout the industry. offense_formation is nflverse's own formation
# tag (SHOTGUN / UNDER CENTER / PISTOL / etc). defense_personnel ("4 DL, 2
# LB, 5 DB") is bucketed into Base/Nickel/Dime by defensive-back count, the
# standard convention (4 DB = Base, 5 = Nickel, 6 = Dime).
# ==============================================================================

print("Step 4d: Computing personnel grouping + formation tendencies from real participation data...")

FOCUS_PERSONNEL_GROUPS = ['11', '12', '21']  # the three you specifically asked about
SCHEME_COLS = ['pct_personnel_11', 'pct_personnel_12', 'pct_personnel_21',
               'pct_shotgun', 'pct_under_center', 'pct_def_base', 'pct_def_nickel', 'pct_def_dime']

try:
    # ROOT CAUSE (found from a real diagnostic pull): load_pbp() and
    # load_participation() do NOT share the same "current season" default.
    # With no seasons= argument, load_pbp() defaulted to 2026 (this year's
    # games) while load_participation() silently defaulted to 2025 (nflverse
    # hasn't published 2026 participation/tracking data yet) -- two
    # completely different seasons' games, so the game_id + play_id join
    # below matched 0% of rows even though nothing else was wrong with it.
    # Fix: always request the SAME season pbp actually loaded (read off
    # pbp_reg itself, the real source of truth), never nflreadpy's default.
    participation_season = int(pbp_reg['season'].max()) if len(pbp_reg) else None
    if participation_season is None:
        raise ValueError("no season could be determined from pbp_reg to request participation data for")
    participation = nfl.load_participation(seasons=[participation_season]).to_pandas()
    if len(participation) == 0:
        raise ValueError(
            f"load_participation() returned 0 rows for season {participation_season} -- nflverse likely "
            f"hasn't published participation/tracking data for this season yet")
    if 'season_type' in participation.columns:
        participation = participation[participation['season_type'] == 'REG'].copy()
    else:
        participation = participation.copy()

    # FIXED after a real live pull: nflreadpy's actual load_participation()
    # output does NOT carry home_team/away_team columns on its own (those
    # only show up in nflverse's own docs because their example joins
    # participation to pbp first) -- confirmed from the real column list a
    # live run printed: nflverse_game_id, old_game_id, play_id,
    # possession_team, offense_formation, offense_personnel,
    # defenders_in_box, defense_personnel, number_of_pass_rushers, and
    # more, but no home_team/away_team. So defteam is now pulled directly
    # from the SAME real pbp already loaded in PART 1 (which already has
    # game_id/play_id/defteam for every play) via a join on game_id +
    # play_id, instead of being derived from columns that don't exist here.
    participation_game_id_col = 'nflverse_game_id' if 'nflverse_game_id' in participation.columns else (
        'game_id' if 'game_id' in participation.columns else 'old_game_id')
    required_cols = ['offense_personnel', 'defense_personnel', 'offense_formation', 'possession_team', 'play_id']
    if participation_game_id_col not in participation.columns or not all(c in participation.columns for c in required_cols):
        raise KeyError(f"participation data missing expected columns -- have {list(participation.columns)}")

    def _parse_personnel_group(s):
        if not isinstance(s, str):
            return None
        rb = re.search(r'(\d+)\s*RB', s)
        te = re.search(r'(\d+)\s*TE', s)
        return f"{rb.group(1)}{te.group(1)}" if rb and te else None

    def _bucket_defense_personnel(s):
        if not isinstance(s, str):
            return None
        db = re.search(r'(\d+)\s*DB', s)
        if not db:
            return None
        n = int(db.group(1))
        return 'Base' if n <= 4 else ('Nickel' if n == 5 else ('Dime' if n == 6 else 'Quarter+'))

    participation['personnel_group'] = participation['offense_personnel'].apply(_parse_personnel_group)
    participation['def_personnel_bucket'] = participation['defense_personnel'].apply(_bucket_defense_personnel)

    # The game_id + play_id join below kept coming back with ZERO matches on
    # real pulls (offense stats, which don't need this join, worked fine --
    # only defense stats, which do, came back blank). nflreadpy loads data
    # through polars -> pandas, which can hand back "arrow-backed" dtypes
    # (e.g. int64 vs double, both backed by pyarrow) for what look like the
    # same column across two different tables. Pandas' merge() hashes the
    # raw dtype representation, so an int64[pyarrow] play_id in one table and
    # a double[pyarrow] play_id in the other can silently match NOTHING even
    # though every value is numerically identical -- no error, no warning,
    # just an all-NaN join. Fix: normalize both join keys to plain built-in
    # strings on both sides before merging, so the comparison is on the
    # actual values, not on however each table happened to encode them.
    def _norm_play_id(series):
        return pd.to_numeric(series, errors='coerce').apply(lambda v: str(int(v)) if pd.notna(v) else None)

    pbp_defteam_key = pbp_reg[['game_id', 'play_id', 'defteam']].drop_duplicates(subset=['game_id', 'play_id']).copy()
    pbp_defteam_key['_game_key'] = pbp_defteam_key['game_id'].astype(str)
    pbp_defteam_key['_play_key'] = _norm_play_id(pbp_defteam_key['play_id'])
    participation['_game_key'] = participation[participation_game_id_col].astype(str)
    participation['_play_key'] = _norm_play_id(participation['play_id'])

    participation = participation.merge(
        pbp_defteam_key[['_game_key', '_play_key', 'defteam']], on=['_game_key', '_play_key'], how='left')
    defense_match_rate = participation['defteam'].notna().mean() if len(participation) else 0
    print(f"  (defense join match rate: {defense_match_rate:.0%} of {len(participation):,} participation rows matched to a defteam)")
    formation_upper = participation['offense_formation'].astype(str).str.upper()
    participation['is_shotgun'] = formation_upper.eq('SHOTGUN')
    participation['is_under_center'] = formation_upper.isin(['UNDER CENTER', 'UNDER_CENTER'])

    all_teams = sorted(master['team'].unique())
    scheme_df = pd.DataFrame({'team': all_teams})

    off_plays = participation[participation['personnel_group'].notna()]
    off_totals = off_plays.groupby('possession_team').size()
    for grp in FOCUS_PERSONNEL_GROUPS:
        counts = off_plays[off_plays['personnel_group'] == grp].groupby('possession_team').size()
        scheme_df[f'pct_personnel_{grp}'] = (counts / off_totals * 100).reindex(all_teams).values

    formation_plays = participation[participation['offense_formation'].notna()]
    formation_totals = formation_plays.groupby('possession_team').size()
    shotgun_counts = formation_plays[formation_plays['is_shotgun']].groupby('possession_team').size()
    uc_counts = formation_plays[formation_plays['is_under_center']].groupby('possession_team').size()
    scheme_df['pct_shotgun'] = (shotgun_counts / formation_totals * 100).reindex(all_teams).values
    scheme_df['pct_under_center'] = (uc_counts / formation_totals * 100).reindex(all_teams).values

    def_plays = participation[participation['def_personnel_bucket'].notna() & participation['defteam'].notna()]
    def_totals = def_plays.groupby('defteam').size()
    for bucket, col in [('Base', 'pct_def_base'), ('Nickel', 'pct_def_nickel'), ('Dime', 'pct_def_dime')]:
        counts = def_plays[def_plays['def_personnel_bucket'] == bucket].groupby('defteam').size()
        scheme_df[col] = (counts / def_totals * 100).reindex(all_teams).values

    master = master.merge(scheme_df, on='team', how='left')
    for c in SCHEME_COLS:
        if c not in columns_to_export:
            columns_to_export.append(c)
    print("  ✓ Personnel groupings (11/12/21), formation (shotgun/under center), defensive personnel (base/nickel/dime) -- real nflverse participation data")
except Exception as e:
    print(f"  ⚠ Personnel/formation tendencies unavailable this pull ('--' in PDF): {e}")
    for c in SCHEME_COLS:
        master[c] = np.nan
        if c not in columns_to_export:
            columns_to_export.append(c)

# ==============================================================================
# END OF PART 2D
# ==============================================================================
# ==============================================================================
# PART 2E: COORDINATOR-ATTRIBUTED COVERAGE TENDENCIES
# nflverse's participation data (man/zone coverage, coverage shell) doesn't
# update in-season -- see PART 2D -- so there is no such thing as "this
# team's real 2026 coverage tendency" yet. Instead of leaving this blank all
# season, this attributes each of the CURRENT 32 teams' offensive/defensive
# coordinator's most recent full season actually calling plays -- even if
# that was with a different team -- using real nflverse historical data
# (2016-2025, CC BY 4.0). Nothing here is faked: every number is that real
# person's real numbers from their real last playcalling season, clearly
# labeled with who and where/when they're from. Coordinator/team mapping
# lives in nfl_coordinator_mapping.csv next to this script -- keep it
# updated as staffs change (see README.txt); rows with no source team/year
# (a true rookie coordinator) are left '--', honestly, rather than guessed.
# ==============================================================================

print("Step 4e: Computing coordinator-attributed coverage tendencies...")

COVERAGE_SHELL_COLS = ['dc_cover0_pct', 'dc_cover1_pct', 'dc_cover2_pct', 'dc_cover3_pct', 'dc_cover4_pct', 'dc_cover6_pct']
COORD_COLS = ['dc_man_pct', 'dc_zone_pct', *COVERAGE_SHELL_COLS, 'dc_source_label', 'dc_sample_plays',
              'oc_epa_vs_man', 'oc_epa_vs_zone', 'oc_success_vs_man', 'oc_success_vs_zone',
              'oc_source_label', 'oc_sample_plays']

def _norm_play_id(series):
    return pd.to_numeric(series, errors='coerce').apply(lambda v: str(int(v)) if pd.notna(v) else None)

try:
    mapping_path = OUT_DIR / 'nfl_coordinator_mapping.csv'
    if not mapping_path.exists():
        raise FileNotFoundError(f"{mapping_path.name} not found next to this script -- coordinator tendencies unavailable this pull")
    coord_map = pd.read_csv(mapping_path)
    for col in ['oc_source_team', 'oc_source_year', 'dc_source_team', 'dc_source_year']:
        if col not in coord_map.columns:
            raise KeyError(f"{mapping_path.name} is missing expected column '{col}'")

    needed_seasons = sorted({
        int(v) for v in pd.concat([coord_map['oc_source_year'], coord_map['dc_source_year']]).dropna().unique()
    })
    if not needed_seasons:
        raise ValueError("no valid source year found anywhere in the coordinator mapping")

    # (team, season) -> defense coverage-shell %'s, and offense EPA/success
    # split by man vs. zone faced, computed once per historical season
    # actually referenced by the mapping (usually just 2025, occasionally
    # older for a coordinator's last playcalling stint elsewhere).
    def_profile_rows = []
    off_profile_rows = []
    for yr in needed_seasons:
        print(f"  Loading real {yr} pbp + participation for coordinator attribution...")
        pbp_yr = nfl.load_pbp(seasons=[yr]).to_pandas()
        pbp_yr_reg = pbp_yr[(pbp_yr['season_type'] == 'REG') & (pbp_yr['posteam'].notna()) & (pbp_yr['defteam'].notna())].copy()
        pbp_yr_reg['epa'] = pd.to_numeric(pbp_yr_reg['epa'], errors='coerce')
        pbp_yr_reg['is_scrimmage_play'] = pbp_yr_reg['play_type'].isin(['pass', 'run'])

        part_yr = nfl.load_participation(seasons=[yr]).to_pandas()
        if 'season_type' in part_yr.columns:
            part_yr = part_yr[part_yr['season_type'] == 'REG'].copy()
        gid_col = 'nflverse_game_id' if 'nflverse_game_id' in part_yr.columns else (
            'game_id' if 'game_id' in part_yr.columns else 'old_game_id')
        if gid_col not in part_yr.columns or 'defense_man_zone_type' not in part_yr.columns:
            print(f"  ⚠ {yr} participation data missing expected coverage columns -- skipping this season")
            continue

        part_yr = part_yr.copy()
        part_yr['_game_key'] = part_yr[gid_col].astype(str)
        part_yr['_play_key'] = _norm_play_id(part_yr['play_id'])
        pbp_key = pbp_yr_reg[['game_id', 'play_id', 'defteam', 'posteam', 'epa', 'is_scrimmage_play']].copy()
        pbp_key['_game_key'] = pbp_key['game_id'].astype(str)
        pbp_key['_play_key'] = _norm_play_id(pbp_key['play_id'])
        joined = part_yr.merge(
            pbp_key[['_game_key', '_play_key', 'defteam', 'posteam', 'epa', 'is_scrimmage_play']],
            on=['_game_key', '_play_key'], how='left')

        zone_upper = joined['defense_man_zone_type'].astype(str).str.upper()
        joined['is_man'] = zone_upper.str.contains('MAN')
        joined['is_zone'] = zone_upper.str.contains('ZONE')
        cov_upper = joined['defense_coverage_type'].astype(str).str.upper() if 'defense_coverage_type' in joined.columns else pd.Series('', index=joined.index)

        # -- defense side: coverage-shell % by defteam --
        def_plays = joined[joined['defteam'].notna() & (joined['is_man'] | joined['is_zone'])]
        def_totals = def_plays.groupby('defteam').size()
        man_counts = def_plays[def_plays['is_man']].groupby('defteam').size()
        zone_counts = def_plays[def_plays['is_zone']].groupby('defteam').size()
        shell_counts = {}
        for shell, tag in [('cover0', 'COVER_0'), ('cover1', 'COVER_1'), ('cover2', 'COVER_2'),
                            ('cover3', 'COVER_3'), ('cover4', 'COVER_4'), ('cover6', 'COVER_6')]:
            shell_counts[shell] = joined[joined['defteam'].notna() & (cov_upper == tag)].groupby('defteam').size()
        for team, total in def_totals.items():
            row = {'team': team, 'season': yr, 'dc_sample_plays': int(total)}
            row['dc_man_pct'] = round((man_counts.get(team, 0) / total) * 100, 1)
            row['dc_zone_pct'] = round((zone_counts.get(team, 0) / total) * 100, 1)
            for shell in ['cover0', 'cover1', 'cover2', 'cover3', 'cover4', 'cover6']:
                row[f'dc_{shell}_pct'] = round((shell_counts[shell].get(team, 0) / total) * 100, 1)
            def_profile_rows.append(row)

        # -- offense side: EPA/success rate vs. man vs. zone by posteam --
        off_plays = joined[joined['posteam'].notna() & joined['is_scrimmage_play'].fillna(False) & (joined['is_man'] | joined['is_zone'])]
        for team, grp in off_plays.groupby('posteam'):
            vs_man = grp[grp['is_man']]
            vs_zone = grp[grp['is_zone']]
            off_profile_rows.append({
                'team': team, 'season': yr, 'oc_sample_plays': int(len(grp)),
                'oc_epa_vs_man': round(vs_man['epa'].mean(), 3) if len(vs_man) else np.nan,
                'oc_epa_vs_zone': round(vs_zone['epa'].mean(), 3) if len(vs_zone) else np.nan,
                'oc_success_vs_man': round((vs_man['epa'] > 0).mean() * 100, 1) if len(vs_man) else np.nan,
                'oc_success_vs_zone': round((vs_zone['epa'] > 0).mean() * 100, 1) if len(vs_zone) else np.nan,
            })

    def_profile = pd.DataFrame(def_profile_rows)
    off_profile = pd.DataFrame(off_profile_rows)
    if def_profile.empty and off_profile.empty:
        raise ValueError("no historical coverage data could be computed for any season in the mapping")

    coord_rows = []
    for _, r in coord_map.iterrows():
        team = r['team']
        row = {'team': team}
        if pd.notna(r.get('dc_source_team')) and pd.notna(r.get('dc_source_year')):
            src_team, src_yr = str(r['dc_source_team']).strip(), int(r['dc_source_year'])
            match = def_profile[(def_profile['team'] == src_team) & (def_profile['season'] == src_yr)]
            if not match.empty:
                m = match.iloc[0]
                for c in ['dc_man_pct', 'dc_zone_pct', *COVERAGE_SHELL_COLS, 'dc_sample_plays']:
                    row[c] = m[c]
                dc_name = r.get('dc_name', '')
                row['dc_source_label'] = f"{dc_name} ({src_team} {src_yr})" if pd.notna(dc_name) and str(dc_name).strip() else f"{src_team} {src_yr}"
        if pd.notna(r.get('oc_source_team')) and pd.notna(r.get('oc_source_year')):
            src_team, src_yr = str(r['oc_source_team']).strip(), int(r['oc_source_year'])
            match = off_profile[(off_profile['team'] == src_team) & (off_profile['season'] == src_yr)]
            if not match.empty:
                m = match.iloc[0]
                for c in ['oc_epa_vs_man', 'oc_epa_vs_zone', 'oc_success_vs_man', 'oc_success_vs_zone', 'oc_sample_plays']:
                    row[c] = m[c]
                oc_name = r.get('oc_name', '')
                row['oc_source_label'] = f"{oc_name} ({src_team} {src_yr})" if pd.notna(oc_name) and str(oc_name).strip() else f"{src_team} {src_yr}"
        coord_rows.append(row)

    coord_df = pd.DataFrame(coord_rows)
    master = master.merge(coord_df, on='team', how='left')
    for c in COORD_COLS:
        if c not in master.columns:
            master[c] = np.nan
        if c not in columns_to_export:
            columns_to_export.append(c)
    matched_dc = master['dc_source_label'].notna().sum() if 'dc_source_label' in master.columns else 0
    matched_oc = master['oc_source_label'].notna().sum() if 'oc_source_label' in master.columns else 0
    print(f"  ✓ Coordinator-attributed coverage tendencies: {matched_dc}/32 teams have a DC match, {matched_oc}/32 have an OC match -- rest are '--' (rookie coordinator or mapping needs an entry)")
except Exception as e:
    print(f"  ⚠ Coordinator-attributed coverage tendencies unavailable this pull ('--' in PDF): {e}")
    for c in COORD_COLS:
        master[c] = np.nan
        if c not in columns_to_export:
            columns_to_export.append(c)

# ==============================================================================
# END OF PART 2E
# ==============================================================================
# ==============================================================================
# PART 2F: INTERIOR / EDGE PRESSURE INDEX ("TUG O' WAR" / "EDGE PRESSURE")
# How much of a team's real, credited pass-rush pressure comes from its
# NT/DT players (Tug o' War) vs. its DE/OLB players (Edge Pressure),
# tracked by the ACTUAL PEOPLE on the 2026 roster rather than by team
# laundry -- same reasoning as the coordinator attribution in PART 2E: a
# D-line/edge group can turn over completely in free agency/the draft, so
# "this team's rate last year" is really "whoever played there last year's
# rate," who may not even be on the roster anymore.
#
# THE HONEST LIMITS (also in README.txt):
#   1. This measures CREDITED pressure -- sacks, half-sacks, and QB hits,
#      every one of them tied to a real player_id in nflverse's pbp -- not
#      hurries. There is no free/licensed source of individual-level hurry
#      data anywhere (checked: not nflverse, not NFL Next Gen Stats' public
#      pages, nothing). nflverse's participation data DOES carry a real
#      team/play-level "was_pressure" flag (sacks+hits+hurries combined,
#      FTN-charted, CC BY 4.0) -- but like everything else in that dataset
#      (PART 2D/2E), it's never published for a season still in progress,
#      and it's never attributed to which specific defender created the
#      pressure, only that pressure happened on the play. So it can't be
#      this season's live number or broken out by player -- it's used
#      instead as the real validation check in backtest_pressure_index.py
#      (run separately, once -- not part of this weekly pull).
#   2. "OLB" isn't a clean edge-rushing tag -- some teams use it for a
#      coverage backer who rarely rushes the passer. No special-casing
#      needed: since every player's weight in the team number is literally
#      his own real pass-rush snap count, a coverage-heavy OLB naturally
#      contributes almost nothing to either side of the ratio instead of
#      skewing it.
#   3. A true rookie (or anyone with zero tracked historical pass-rush
#      snaps) has no personal track record to lean on -- rather than assume
#      he plays like whoever he's replacing, he's assigned the POOLED
#      league-wide historical rate for his position group, at a small
#      nominal snap weight (ASSUMED_ROOKIE_SNAPS below), so he doesn't get
#      zero influence but also can't swing a team's number until he's built
#      a real sample this season.
#
# HOW THE NUMBER IS BUILT: pull the current 2026 roster, filter to NT/DT and
# DE/OLB, and for each of those real players pull their OWN historical
# credited sacks/half-sacks/QB hits and their OWN real pass-rush snap counts
# (from participation's defense_players field) across the last
# HISTORY_WINDOW_YEARS completed seasons. Summing raw counts (not averaging
# rates) across a team's current personnel means the baseline is
# automatically weighted toward whichever players have the deepest real
# track record -- no separate weighting formula needed, and a team that
# churned its whole D-line naturally leans on live 2026 data faster, since
# there's less personnel history to lean on anyway. This season's own
# credited events (straight from pbp_reg, already loaded for every other
# stat on this sheet -- no participation data needed, so no in-season
# blackout problem) get added directly on top. The blended rate is then
# converted to a percentile against a real pooled historical reference pool
# (every current team, every season in the window) -- tier labeling/color
# happens in nfl_matchup_sheet.py, same split as VOID (raw number computed
# here, presentation there).
# ==============================================================================

print("Step 4f: Computing interior/edge pressure index (Tug o' War / Edge Pressure)...")

INTERIOR_POSITIONS = {'DT', 'NT'}
EDGE_POSITIONS = {'DE', 'OLB'}
HISTORY_WINDOW_YEARS = 4          # most recent completed seasons used for personnel baselines + league reference pool
ASSUMED_ROOKIE_SNAPS = 40         # nominal pass-rush-snap weight for a player with zero tracked history

PRESSURE_INDEX_COLS = [
    'interior_pressure_rate', 'interior_pressure_pct', 'interior_pressure_note',
    'edge_pressure_rate', 'edge_pressure_pct', 'edge_pressure_note',
]

CREDIT_COLS = [
    ('sack_player_id', 1.0), ('half_sack_1_player_id', 0.5), ('half_sack_2_player_id', 0.5),
    ('qb_hit_1_player_id', 1.0), ('qb_hit_2_player_id', 1.0),
]

try:
    current_season_pf = int(pbp_reg['season'].max()) if len(pbp_reg) else None
    if current_season_pf is None:
        raise ValueError("no current season determined -- can't compute pressure index")

    # nflverse's participation loader only serves 2016-2025 -- see PART 2D --
    # so the history window is clipped to that real range regardless of what
    # HISTORY_WINDOW_YEARS asks for.
    hist_years = [y for y in range(current_season_pf - HISTORY_WINDOW_YEARS, current_season_pf) if 2016 <= y <= 2025]
    if not hist_years:
        raise ValueError("no historical seasons available in nflverse's 2016-2025 participation range to build a personnel baseline")

    # Roster lookup spans BOTH the current season (who's actually on each
    # 2026 team right now) and every historical season in the window (each
    # historical credited player's position AT THAT TIME -- a guy can
    # change positions over a career, so this is looked up per-season, not
    # assumed static).
    roster_all = nfl.load_rosters(seasons=[current_season_pf, *hist_years]).to_pandas()
    roster_needed_cols = ['season', 'team', 'depth_chart_position', 'gsis_id']
    if not all(c in roster_all.columns for c in roster_needed_cols):
        raise KeyError(f"load_rosters() missing expected columns -- have {list(roster_all.columns)}")
    roster_all = roster_all[roster_all['gsis_id'].notna()].copy()
    roster_all['gsis_id'] = roster_all['gsis_id'].astype(str)
    # A player can show up more than once in a season's roster feed (team
    # changes, practice-squad elevations) -- keep the LAST listed row per
    # (gsis_id, season).
    roster_all = roster_all.drop_duplicates(subset=['gsis_id', 'season'], keep='last')
    # IMPORTANT: nflverse's 'position' column is a broad group ('DL', 'LB',
    # etc.) that lumps interior and edge D-linemen together, and edge OLBs
    # in with inside/middle LBs -- confirmed against a real pull (2021-2025,
    # 15,531 roster rows: 'position' never once produced 'DT'/'NT'/'DE'/
    # 'OLB'). 'depth_chart_position' is the real fine-grained column --
    # fully populated for every DL/LB-group row in that same real pull, with
    # exactly the DT/NT/DE/OLB/ILB/MLB/LB labels this index needs. Use it,
    # not 'position'.
    pos_lookup = roster_all.set_index(['gsis_id', 'season'])['depth_chart_position'].to_dict()
    team_lookup = roster_all.set_index(['gsis_id', 'season'])['team'].to_dict()
    current_roster = roster_all[roster_all['season'] == current_season_pf].copy()

    # Lightweight permanent safety net (not the earlier deep-dive
    # diagnostic, since the real fix above is now confirmed) -- if a future
    # nflverse schema change breaks the depth_chart_position match again,
    # this prints the real values instead of silently computing 0/32 teams.
    n_interior_match = current_roster['depth_chart_position'].isin(INTERIOR_POSITIONS).sum()
    n_edge_match = current_roster['depth_chart_position'].isin(EDGE_POSITIONS).sum()
    if len(current_roster) > 0 and n_interior_match == 0 and n_edge_match == 0:
        print(f"  [diagnostic] neither position set matched real 'depth_chart_position' values -- "
              f"most common first:\n{current_roster['depth_chart_position'].value_counts(dropna=False).head(25).to_string()}")

    # --------------------------------------------------------------------
    # Per-season player-level pass-rush snaps (real participation
    # defense_players lists) + credited sacks/half-sacks/QB hits (real pbp
    # player_id columns) -- built once per historical season, then reused
    # for both (a) the current roster's personnel baseline and (b) the
    # league-wide historical reference pool for percentile scaling.
    # NOTE: this pulls pbp + participation independently per season, same
    # as PART 2D/2E do -- some redundant network calls if seasons overlap,
    # traded for keeping each PART self-contained and independently
    # fail-safe (consistent with the rest of this script).
    # --------------------------------------------------------------------
    player_season_rows = []   # player_id, season, pass_rush_snaps, credited_events
    team_season_rows = []     # team, season, dropbacks_faced, interior_events, edge_events

    for yr in hist_years:
        print(f"  Loading real {yr} pbp + participation for pressure-index attribution...")
        pbp_yr = nfl.load_pbp(seasons=[yr]).to_pandas()
        pbp_yr_reg = pbp_yr[(pbp_yr['season_type'] == 'REG') & (pbp_yr['defteam'].notna())].copy()

        present_credit_cols = [(c, w) for c, w in CREDIT_COLS if c in pbp_yr_reg.columns]
        if not present_credit_cols:
            print(f"  ⚠ {yr} pbp missing all sack/QB-hit player-credit columns -- skipping this season")
            continue

        pbp_yr_reg['pass_attempt'] = pd.to_numeric(pbp_yr_reg.get('pass_attempt', 0), errors='coerce').fillna(0)
        pbp_yr_reg['sack'] = pd.to_numeric(pbp_yr_reg.get('sack', 0), errors='coerce').fillna(0)
        pbp_yr_reg['dropback_flag'] = pbp_yr_reg['pass_attempt'] + pbp_yr_reg['sack']
        drop_yr = pbp_yr_reg.groupby('defteam')['dropback_flag'].sum().reset_index(name='dropbacks_faced')

        # Melt every credited-event column into one long (defteam, player_id, weight) table.
        credit_frames = []
        for col, weight in present_credit_cols:
            sub = pbp_yr_reg[pbp_yr_reg[col].notna()][['defteam', col]].copy()
            sub.columns = ['defteam', 'player_id']
            sub['player_id'] = sub['player_id'].astype(str)
            sub['weight'] = weight
            credit_frames.append(sub)
        credits_long = pd.concat(credit_frames, ignore_index=True) if credit_frames else pd.DataFrame(columns=['defteam', 'player_id', 'weight'])
        credits_long['position'] = credits_long['player_id'].apply(lambda pid: pos_lookup.get((pid, yr)))

        player_credits = credits_long.groupby('player_id')['weight'].sum().reset_index().rename(columns={'weight': 'credited_events'})

        # Real pass-rush snaps per player: explode participation's
        # defense_players list, keep only plays where the offense actually
        # dropped back to pass (play_type == 'pass' on the joined pbp row).
        part_yr = nfl.load_participation(seasons=[yr]).to_pandas()
        if 'season_type' in part_yr.columns:
            part_yr = part_yr[part_yr['season_type'] == 'REG'].copy()
        gid_col = 'nflverse_game_id' if 'nflverse_game_id' in part_yr.columns else (
            'game_id' if 'game_id' in part_yr.columns else 'old_game_id')
        if gid_col not in part_yr.columns or 'defense_players' not in part_yr.columns:
            print(f"  ⚠ {yr} participation data missing defense_players -- snap counts unavailable this season, using credited events only")
            snaps_by_player = pd.DataFrame(columns=['player_id', 'pass_rush_snaps'])
        else:
            part_yr = part_yr.copy()
            part_yr['_game_key'] = part_yr[gid_col].astype(str)
            part_yr['_play_key'] = _norm_play_id(part_yr['play_id'])
            pbp_type_key = pbp_yr_reg[['game_id', 'play_id', 'play_type']].copy()
            pbp_type_key['_game_key'] = pbp_type_key['game_id'].astype(str)
            pbp_type_key['_play_key'] = _norm_play_id(pbp_type_key['play_id'])
            part_yr = part_yr.merge(pbp_type_key[['_game_key', '_play_key', 'play_type']], on=['_game_key', '_play_key'], how='left')
            pass_plays = part_yr[part_yr['play_type'] == 'pass']
            exploded = pass_plays['defense_players'].dropna().astype(str).str.split(';').explode().str.strip()
            exploded = exploded[exploded != '']
            snaps_by_player = exploded.value_counts().reset_index()
            snaps_by_player.columns = ['player_id', 'pass_rush_snaps']

        player_season = player_credits.merge(snaps_by_player, on='player_id', how='outer')
        player_season['credited_events'] = player_season['credited_events'].fillna(0)
        player_season['pass_rush_snaps'] = player_season['pass_rush_snaps'].fillna(0)
        player_season['season'] = yr
        player_season_rows.append(player_season)

        # Team-level interior/edge credited events + SNAPS this historical
        # season, for the league-wide reference pool. IMPORTANT: the
        # reference pool's denominator has to be built the SAME way the
        # personnel baseline's is below (summed individual pass-rush snaps
        # across however many interior/edge players a team actually rotates
        # through), not team-wide dropbacks -- those are different scales
        # (multiple interior/edge players can each log snaps on the SAME
        # dropback, so summed player-snaps run well above single-season
        # team dropbacks). Comparing a personnel-summed rate against a
        # dropback-denominated rate would silently compress every team's
        # percentile toward the bottom of the scale -- caught on the first
        # smoke-test run (every team came back under the 20th percentile),
        # fixed by keeping both sides of the percentile comparison on the
        # same per-player-snap basis.
        credits_long['is_interior'] = credits_long['position'].isin(INTERIOR_POSITIONS)
        credits_long['is_edge'] = credits_long['position'].isin(EDGE_POSITIONS)
        team_interior_events = credits_long[credits_long['is_interior']].groupby('defteam')['weight'].sum()
        team_edge_events = credits_long[credits_long['is_edge']].groupby('defteam')['weight'].sum()

        player_season_pos = player_season.copy()
        player_season_pos['position'] = player_season_pos['player_id'].apply(lambda pid: pos_lookup.get((pid, yr)))
        player_season_pos['team'] = player_season_pos['player_id'].apply(lambda pid: team_lookup.get((pid, yr)))
        team_interior_snaps = player_season_pos[player_season_pos['position'].isin(INTERIOR_POSITIONS)].groupby('team')['pass_rush_snaps'].sum()
        team_edge_snaps = player_season_pos[player_season_pos['position'].isin(EDGE_POSITIONS)].groupby('team')['pass_rush_snaps'].sum()

        all_teams_yr = sorted(set(drop_yr['defteam'].dropna()) | set(team_interior_snaps.index) | set(team_edge_snaps.index))
        team_yr = pd.DataFrame({'team': all_teams_yr})
        team_yr['season'] = yr
        team_yr['interior_events'] = team_yr['team'].map(team_interior_events).fillna(0)
        team_yr['edge_events'] = team_yr['team'].map(team_edge_events).fillna(0)
        team_yr['interior_snaps'] = team_yr['team'].map(team_interior_snaps).fillna(0)
        team_yr['edge_snaps'] = team_yr['team'].map(team_edge_snaps).fillna(0)
        team_season_rows.append(team_yr)

    if not player_season_rows:
        raise ValueError("no historical seasons produced usable pressure-index data")

    player_hist = pd.concat(player_season_rows, ignore_index=True)
    player_hist_totals = player_hist.groupby('player_id').agg(
        hist_events=('credited_events', 'sum'), hist_snaps=('pass_rush_snaps', 'sum')
    ).reset_index()
    # Each player's most-recently-known position across the window (a guy
    # can move DE<->OLB over a career; use his latest listed spot).
    player_hist_totals['position'] = player_hist_totals['player_id'].apply(
        lambda pid: next((pos_lookup.get((pid, yr)) for yr in reversed(hist_years) if pos_lookup.get((pid, yr))), None))

    team_hist = pd.concat(team_season_rows, ignore_index=True)

    # League-wide pooled positional rate -- used ONLY as the fallback for a
    # player with zero tracked history (true rookies, practice-squad-only
    # players), computed from this same real historical data.
    league_interior_pool = player_hist_totals[player_hist_totals['position'].isin(INTERIOR_POSITIONS) & (player_hist_totals['hist_snaps'] > 0)]
    league_edge_pool = player_hist_totals[player_hist_totals['position'].isin(EDGE_POSITIONS) & (player_hist_totals['hist_snaps'] > 0)]
    league_interior_rate = (league_interior_pool['hist_events'].sum() / league_interior_pool['hist_snaps'].sum()) if league_interior_pool['hist_snaps'].sum() > 0 else 0.02
    league_edge_rate = (league_edge_pool['hist_events'].sum() / league_edge_pool['hist_snaps'].sum()) if league_edge_pool['hist_snaps'].sum() > 0 else 0.02

    # --------------------------------------------------------------------
    # Each of the 32 CURRENT teams' personnel-weighted baseline, built from
    # its actual 2026 roster's NT/DT and DE/OLB players' own real history.
    # --------------------------------------------------------------------
    player_hist_lookup = player_hist_totals.set_index('player_id')[['hist_events', 'hist_snaps']].to_dict('index')

    def _personnel_baseline(team, positions, league_rate):
        players = current_roster[(current_roster['team'] == team) & (current_roster['depth_chart_position'].isin(positions))]
        events, snaps, n_players, n_with_history = 0.0, 0.0, 0, 0
        for pid in players['gsis_id']:
            hist = player_hist_lookup.get(pid)
            n_players += 1
            if hist and hist['hist_snaps'] > 0:
                events += hist['hist_events']
                snaps += hist['hist_snaps']
                n_with_history += 1
            else:
                events += league_rate * ASSUMED_ROOKIE_SNAPS
                snaps += ASSUMED_ROOKIE_SNAPS
        return events, snaps, n_players, n_with_history

    # --------------------------------------------------------------------
    # Live 2026 credited events -- straight from pbp_reg, already loaded for
    # every other stat on this sheet, so this doesn't need participation
    # data (and isn't blocked by its in-season blackout). Dropbacks faced
    # reuses pressure_def, already computed in PART 2C.
    # --------------------------------------------------------------------
    pbp_reg_pc = pbp_reg.copy()
    for c, _ in CREDIT_COLS:
        if c not in pbp_reg_pc.columns:
            pbp_reg_pc[c] = np.nan
    live_credit_frames = []
    for col, weight in CREDIT_COLS:
        sub = pbp_reg_pc[pbp_reg_pc[col].notna()][['defteam', col]].copy()
        sub.columns = ['defteam', 'player_id']
        sub['player_id'] = sub['player_id'].astype(str)
        sub['weight'] = weight
        live_credit_frames.append(sub)
    live_credits = pd.concat(live_credit_frames, ignore_index=True) if live_credit_frames else pd.DataFrame(columns=['defteam', 'player_id', 'weight'])
    live_credits['position'] = live_credits['player_id'].apply(lambda pid: pos_lookup.get((pid, current_season_pf)))
    live_interior_events = live_credits[live_credits['position'].isin(INTERIOR_POSITIONS)].groupby('defteam')['weight'].sum()
    live_edge_events = live_credits[live_credits['position'].isin(EDGE_POSITIONS)].groupby('defteam')['weight'].sum()
    live_dropbacks = pressure_def.set_index('team')['dropbacks_faced']

    # --------------------------------------------------------------------
    # League-wide historical reference pool for percentile scaling -- every
    # current team's own real history in the window.
    # --------------------------------------------------------------------
    team_hist['interior_rate'] = np.where(team_hist['interior_snaps'] > 0, team_hist['interior_events'] / team_hist['interior_snaps'], np.nan)
    team_hist['edge_rate'] = np.where(team_hist['edge_snaps'] > 0, team_hist['edge_events'] / team_hist['edge_snaps'], np.nan)
    interior_reference_pool = team_hist['interior_rate'].dropna().values
    edge_reference_pool = team_hist['edge_rate'].dropna().values

    def _percentile(value, pool):
        if pool is None or len(pool) == 0 or pd.isna(value):
            return np.nan
        return float((pool <= value).mean() * 100)

    pressure_rows = []
    for team in sorted(current_roster['team'].dropna().unique()):
        int_base_events, int_base_snaps, int_n, int_n_hist = _personnel_baseline(team, INTERIOR_POSITIONS, league_interior_rate)
        edge_base_events, edge_base_snaps, edge_n, edge_n_hist = _personnel_baseline(team, EDGE_POSITIONS, league_edge_rate)

        live_int = live_interior_events.get(team, 0.0)
        live_edge = live_edge_events.get(team, 0.0)
        live_drop = live_dropbacks.get(team, 0.0)

        int_snaps_final = int_base_snaps + live_drop
        edge_snaps_final = edge_base_snaps + live_drop
        int_rate = ((int_base_events + live_int) / int_snaps_final) if int_snaps_final > 0 else np.nan
        edge_rate = ((edge_base_events + live_edge) / edge_snaps_final) if edge_snaps_final > 0 else np.nan

        pressure_rows.append({
            'team': team,
            'interior_pressure_rate': round(int_rate, 4) if pd.notna(int_rate) else np.nan,
            'interior_pressure_pct': round(_percentile(int_rate, interior_reference_pool), 1) if pd.notna(int_rate) else np.nan,
            'interior_pressure_note': f"{int_n} NT/DT tracked ({int_n_hist} with real history)" if int_n > 0 else np.nan,
            'edge_pressure_rate': round(edge_rate, 4) if pd.notna(edge_rate) else np.nan,
            'edge_pressure_pct': round(_percentile(edge_rate, edge_reference_pool), 1) if pd.notna(edge_rate) else np.nan,
            'edge_pressure_note': f"{edge_n} DE/OLB tracked ({edge_n_hist} with real history)" if edge_n > 0 else np.nan,
        })

    pressure_df_out = pd.DataFrame(pressure_rows)
    master = master.merge(pressure_df_out, on='team', how='left')
    for c in PRESSURE_INDEX_COLS:
        if c not in master.columns:
            master[c] = np.nan
        if c not in columns_to_export:
            columns_to_export.append(c)
    n_matched = master['interior_pressure_pct'].notna().sum() if 'interior_pressure_pct' in master.columns else 0
    print(f"  ✓ Interior/edge pressure index: {n_matched}/32 teams computed (personnel-weighted history + live {current_season_pf} data, scaled vs. a {len(hist_years)}-season league reference pool)")
except Exception as e:
    print(f"  ⚠ Interior/edge pressure index unavailable this pull ('--' in PDF): {e}")
    for c in PRESSURE_INDEX_COLS:
        master[c] = np.nan
        if c not in columns_to_export:
            columns_to_export.append(c)

# ==============================================================================
# END OF PART 2F
# ==============================================================================
# ==============================================================================
# PART 3: LAST-GAME RECAP + IN-SEASON SCHEME TENDENCY PROXIES
# Uses each team's MOST RECENT game only (not a season aggregate), since
# coordinators/schemes can change and recency matters more here than sample
# size. Man/zone coverage labels aren't real-time available in-season --
# nflverse's participation data with that detail is only released after a
# season fully concludes -- so this uses proxies that ARE reliably in the
# core play-by-play data already loaded above: pass/run play-calling mix,
# and pressure rate (sacks + QB hits per dropback).
# ==============================================================================

print("Step 5: Building last-game recap and scheme-tendency proxies...")

if 'qb_hit' in pbp_reg.columns:
    pbp_reg['qb_hit'] = pd.to_numeric(pbp_reg['qb_hit'], errors='coerce').fillna(0)
else:
    pbp_reg['qb_hit'] = 0

last_week_per_team = pbp_reg.groupby('posteam')['week'].max().reset_index().rename(columns={'posteam': 'team', 'week': 'last_week'})

last_game_rows = []
for _, lw_row in last_week_per_team.iterrows():
    team = lw_row['team']
    wk = lw_row['last_week']
    team_off_plays = pbp_reg[(pbp_reg['posteam'] == team) & (pbp_reg['week'] == wk)]
    team_def_plays = pbp_reg[(pbp_reg['defteam'] == team) & (pbp_reg['week'] == wk)]

    if team_off_plays.empty:
        continue

    g_id = team_off_plays['game_id'].iloc[0]
    is_home = (team_off_plays['home_team'].iloc[0] == team)
    opponent = team_off_plays['defteam'].iloc[0]

    final_row = final_scores[final_scores['game_id'] == g_id]
    if not final_row.empty:
        fr = final_row.iloc[0]
        pts_for = fr['home_score'] if is_home else fr['away_score']
        pts_against = fr['away_score'] if is_home else fr['home_score']
        if fr['winner'] == team:
            result = 'W'
        elif fr['winner'] == 'TIE':
            result = 'T'
        else:
            result = 'L'
    else:
        pts_for, pts_against, result = 0, 0, '-'

    off_total = len(team_off_plays)
    off_pass = (team_off_plays['play_type'] == 'pass').sum()
    off_pressure_allowed = team_off_plays['sack'].sum() + team_off_plays['qb_hit'].sum()

    def_total = len(team_def_plays)
    def_pass_faced = (team_def_plays['play_type'] == 'pass').sum()
    def_pressure_forced = team_def_plays['sack'].sum() + team_def_plays['qb_hit'].sum()

    last_game_rows.append({
        'team': team,
        'last_game_opponent': opponent,
        'last_game_result': result,
        'last_game_points_for': pts_for,
        'last_game_points_against': pts_against,
        'last_game_off_pass_rate': round((off_pass / off_total) * 100, 1) if off_total > 0 else 0,
        'last_game_off_pressure_rate_allowed': round((off_pressure_allowed / off_pass) * 100, 1) if off_pass > 0 else 0,
        'last_game_def_pass_rate_faced': round((def_pass_faced / def_total) * 100, 1) if def_total > 0 else 0,
        'last_game_def_pressure_rate_forced': round((def_pressure_forced / def_pass_faced) * 100, 1) if def_pass_faced > 0 else 0,
    })

last_game_df = pd.DataFrame(last_game_rows)
master = master.merge(last_game_df, on='team', how='left')
master = master.merge(aggressiveness[['team', 'fourth_down_aggressiveness_pct']], on='team', how='left')

last_game_cols = [
    'last_game_opponent', 'last_game_result', 'last_game_points_for', 'last_game_points_against',
    'last_game_off_pass_rate', 'last_game_off_pressure_rate_allowed',
    'last_game_def_pass_rate_faced', 'last_game_def_pressure_rate_forced',
]
columns_to_export += last_game_cols + ['fourth_down_aggressiveness_pct']

# ==============================================================================
# PART 3B: FULL-SEASON PER-GAME LOG -- "PREVIOUS SCHEDULE" PANEL
# Every game each team has played so far this season, one row per game (not
# just the most recent one -- that's PART 3 above). Real per-game score, real
# per-game rush/pass yards on offense, and real per-game rush/pass yards
# ALLOWED on defense, all sourced directly from the same pbp_clean/pbp_reg
# already loaded -- no new data source, no season-average blending. Written
# to its own CSV (nfl_game_log.csv) since it's variable-length per team
# (a team with 3 games played has 3 rows, one with 8 has 8), unlike the
# fixed one-row-per-team master matrix.
# ==============================================================================

print("Step 5b: Building full per-game log (every game played, both sides) for the Previous Schedule panel...")

game_weeks = pbp_reg.groupby('game_id')['week'].first().reset_index()

# Real per-game rush/pass yards, offense side -- grouped by (posteam, game_id)
# instead of just posteam, so each game gets its own row instead of a season sum.
off_game_rush = pbp_clean[pbp_clean['play_type'] == 'run'].groupby(['posteam', 'game_id'])['yards_gained'].sum().reset_index().rename(columns={'posteam': 'team', 'yards_gained': 'rush_yards'})
off_game_pass = pbp_clean[pbp_clean['play_type'] == 'pass'].groupby(['posteam', 'game_id'])['yards_gained'].sum().reset_index().rename(columns={'posteam': 'team', 'yards_gained': 'pass_yards'})
# Same re-attribution technique used throughout this pipeline (e.g.
# catch_pct_allowed): the SAME real yardage, grouped by 'defteam' instead of
# 'posteam' -- this team's defense allowed that many rush/pass yards that game.
def_game_rush = pbp_clean[pbp_clean['play_type'] == 'run'].groupby(['defteam', 'game_id'])['yards_gained'].sum().reset_index().rename(columns={'defteam': 'team', 'yards_gained': 'rush_yards_allowed'})
def_game_pass = pbp_clean[pbp_clean['play_type'] == 'pass'].groupby(['defteam', 'game_id'])['yards_gained'].sum().reset_index().rename(columns={'defteam': 'team', 'yards_gained': 'pass_yards_allowed'})

team_games = pbp_reg.groupby(['posteam', 'game_id']).agg(
    defteam=('defteam', 'first'),
    home_team=('home_team', 'first'),
    away_team=('away_team', 'first'),
).reset_index().rename(columns={'posteam': 'team'})
team_games = team_games.merge(game_weeks, on='game_id', how='left')
team_games['opponent'] = team_games['defteam']
team_games['is_home'] = team_games['team'] == team_games['home_team']

team_games = team_games.merge(
    final_scores[['game_id', 'home', 'away', 'home_score', 'away_score', 'winner']],
    on='game_id', how='left'
)
team_games['pts_for'] = np.where(team_games['is_home'], team_games['home_score'], team_games['away_score'])
team_games['pts_against'] = np.where(team_games['is_home'], team_games['away_score'], team_games['home_score'])
team_games['result'] = np.where(team_games['winner'] == team_games['team'], 'W',
                         np.where(team_games['winner'] == 'TIE', 'T', 'L'))

team_games = team_games.merge(off_game_rush, on=['team', 'game_id'], how='left')
team_games = team_games.merge(off_game_pass, on=['team', 'game_id'], how='left')
team_games = team_games.merge(def_game_rush, on=['team', 'game_id'], how='left')
team_games = team_games.merge(def_game_pass, on=['team', 'game_id'], how='left')

# NOTE: unlike the special-teams NO_FILLNA_COLS fix (where a MISSING COLUMN
# would've faked a plausible 0.0), a missing ROW here just means that team
# ran zero rush (or pass) plays in that specific game -- a real, if rare,
# zero, not an unavailable stat. fillna(0) is the honestly correct move here.
yard_cols = ['rush_yards', 'pass_yards', 'rush_yards_allowed', 'pass_yards_allowed']
team_games[yard_cols] = team_games[yard_cols].fillna(0)

game_log_path = OUT_DIR / 'nfl_game_log.csv'

# ==============================================================================
# PART 3C: EXTENDED PER-GAME STAT LOG -- feeds the real, backtested stat-
# Weighted win-prediction model in nfl_matchup_sheet.py (added to "have it
# learn from previous games... give the stats weights and move the weights
# around"). Same technique as PART 3B directly above -- group the SAME real
# pbp_clean/pbp_reg by (team, game_id) instead of just by team, so every
# number here is this team's REAL performance in that ONE game, not a
# season-to-date average. This is what lets the backtest model actually
# measure "did the team that led stat X win that specific game," not "does
# a team with a better season-long number in X tend to win," which would be
# a much weaker (and slower to update) signal.
#
# KNOWN SCOPE LIMIT: this covers EPA/play, success rate, 3rd down %, sack %,
# pressure rate, completion %, and interceptions -- the stats that are
# simple per-play aggregations. VOID, the interior/edge pressure index
# ("Tug o' War"), personnel/formation tendencies, and the NGS advanced
# metrics are NOT included here -- they're either multi-step composite
# formulas (VOID's opponent-adjustment) or come from a separate data pull
# (NGS/participation) not easily re-grouped to single-game granularity in
# this pass. Those stats still show on the sheet every week; they just
# don't yet have real per-game backtest history behind them, so the
# prediction model in nfl_matchup_sheet.py uses documented prior weights
# for them instead of measured ones until a future update extends this
# section to cover them too.
# ==============================================================================

print("Step 5c: Building extended per-game stat log (EPA, success rate, 3rd down, sacks, pressure, completions, INTs) for the prediction model...")

def _per_game(df_src, team_col, value_col, out_col, agg='mean'):
    g = df_src.groupby([team_col, 'game_id'])[value_col].agg(agg).reset_index()
    return g.rename(columns={team_col: 'team', value_col: out_col})

# EPA/play -- offense's own EPA/play that game, and what this team's
# defense ALLOWED that game (mean EPA/play across the opponent's real
# snaps against this defense).
off_epa_g = _per_game(pbp_clean, 'posteam', 'epa', 'off_epa_play_game')
def_epa_g = _per_game(pbp_clean, 'defteam', 'epa', 'def_epa_play_allowed_game')

# Success rate -- same real is_success flag (epa > 0) already used for the
# season-long number, just grouped one level finer.
off_sr_g = _per_game(pbp_clean, 'posteam', 'is_success', 'off_success_rate_game', agg='mean')
off_sr_g['off_success_rate_game'] *= 100
def_sr_g = _per_game(pbp_clean, 'defteam', 'is_success', 'def_success_rate_allowed_game', agg='mean')
def_sr_g['def_success_rate_allowed_game'] *= 100

# 3rd down % -- same real third_down_converted flag used for the season
# number, grouped per game. A team with zero 3rd-down plays that game (rare
# but possible in a blowout) legitimately has no real number here -- NaN,
# not a fake 0%, consistent with how every other "no real plays this
# bucket" case is handled throughout this pipeline.
third_down_plays_g = pbp_clean[pbp_clean['third_down_converted'].notna()]
off_3rd_g = third_down_plays_g.groupby(['posteam', 'game_id'])['third_down_converted'].mean().reset_index().rename(
    columns={'posteam': 'team', 'third_down_converted': 'third_down_pct_game'})
off_3rd_g['third_down_pct_game'] *= 100
def_3rd_g = third_down_plays_g.groupby(['defteam', 'game_id'])['third_down_converted'].mean().reset_index().rename(
    columns={'defteam': 'team', 'third_down_converted': 'third_down_pct_allowed_game'})
def_3rd_g['third_down_pct_allowed_game'] *= 100

# Sack % of dropbacks (attempts + sacks), same real formula as the season
# number -- offense taken / defense forced, per game.
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

# Pressure rate (sacks + QB hits) / dropbacks, per game -- same real free
# proxy already established in PART 2C (pbp_reg['qb_hit'] is already
# numeric/filled by that point in the script), just grouped by game_id too.
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

# Completion % and interceptions thrown/gained, per game.
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

game_log_df = team_games[[
    'team', 'game_id', 'week', 'opponent', 'is_home', 'result', 'pts_for', 'pts_against',
    'rush_yards', 'pass_yards', 'rush_yards_allowed', 'pass_yards_allowed',
    'off_epa_play_game', 'def_epa_play_allowed_game',
    'off_success_rate_game', 'def_success_rate_allowed_game',
    'third_down_pct_game', 'third_down_pct_allowed_game',
    'sack_pct_offense_game', 'sack_pct_defense_game',
    'pressure_rate_allowed_game', 'pressure_rate_forced_game',
    'completion_pct_game', 'interceptions_thrown_game', 'interceptions_gained_game',
]].sort_values(['team', 'week']).reset_index(drop=True)

game_log_df.to_csv(game_log_path, index=False)
print(f" -> Wrote {len(game_log_df)} team-game row(s) ({game_log_df.shape[1]} columns, "
      f"{game_log_df.shape[1] - 6} of them real per-game stats) to {game_log_path.name}.")

# ==============================================================================
# PART 4: WEIGHTED POWER RATINGS FOR ML / SPREAD / TOTAL PREDICTIONS
# These stats carry the most weight: points/game and yards/play get the
# largest share, EPA/play next, then success rate, turnover margin, and red
# zone efficiency get smaller weights. Each team's rating is a weighted sum
# of how many standard deviations above/below the league average they are on
# each stat (a z-score), so the weights are directly comparable regardless
# of each stat's raw scale. This produces off_power_rating/def_power_rating
# per team; the actual matchup-specific spread/total/win-probability get
# computed in the PDF builder script, since those need BOTH teams' ratings
# together, not just one team's own.
# ==============================================================================

print("Step 6: Computing weighted power ratings for predictions...")

OFFENSE_RATING_WEIGHTS = {
    'points_per_game': 0.30,
    'off_yards_per_play': 0.25,
    'off_efficiency_epa': 0.20,
    'off_success_rate': 0.10,
    'turnover_margin_offense': -0.10,    # negative: fewer giveaways is better
    'redzone_efficiency': 0.05,
}
DEFENSE_RATING_WEIGHTS = {
    'points_allowed_per_game': -0.30,    # negative: fewer points allowed is better
    'def_yards_per_play': -0.25,         # negative: fewer yards/play allowed is better
    'def_efficiency_epa': -0.20,         # negative: lower EPA/play allowed is better
    'def_success_rate': -0.10,           # negative: lower success rate allowed is better
    'turnover_margin_defense': 0.10,     # positive: more takeaways is better
    'redzone_efficiency_allowed': -0.05, # negative: lower red zone TD% allowed is better
}

def compute_power_rating(df, weights):
    rating = pd.Series(0.0, index=df.index)
    for stat, weight in weights.items():
        std = df[stat].std()
        z = (df[stat] - df[stat].mean()) / std if std and std > 0 else 0
        rating = rating + (weight * z)
    return rating

master['off_power_rating'] = compute_power_rating(master, OFFENSE_RATING_WEIGHTS)
master['def_power_rating'] = compute_power_rating(master, DEFENSE_RATING_WEIGHTS)
columns_to_export += ['off_power_rating', 'def_power_rating']

final_matrix = master[columns_to_export].copy().fillna(0)

# Export next to this script (not a hardcoded Mac path -- keeps the whole
# project portable, and matchable to whatever folder it's actually run from).
team_matrix_path = OUT_DIR / 'nfl_team_matrix.csv'
final_matrix.to_csv(team_matrix_path, index=False)
print("\n[Done] Matrix generated cleanly with real traditional Passer Ratings!")
print(f"Destination path: '{team_matrix_path}'")

# ==============================================================================
# PART 5: CURRENT INJURY REPORT
# Real nflverse injury data (weekly report status). Saved as its own file
# since it's naturally multiple rows per team (one per injured player), not
# a single per-team row like the main matrix above. NOTE: nflverse's own
# changelog has flagged this loader as unreliable in some recent seasons --
# this is wrapped defensively so a failure here doesn't take down the whole
# script, same as every other live pull in this project.
# ==============================================================================

print("Step 7: Pulling current injury report...")
try:
    injuries = nfl.load_injuries().to_pandas()
    # Keep only players with an actual game-status designation (Questionable/
    # Doubtful/Out) from the most recent reported week -- rows with no
    # report_status are typically early-week practice-only entries, not a
    # real game-availability concern.
    injuries = injuries[injuries['report_status'].notna() & (injuries['report_status'] != '')]
    latest_week = injuries['week'].max()
    injuries = injuries[injuries['week'] == latest_week]
    injury_export = injuries[['team', 'full_name', 'position', 'report_primary_injury', 'report_status']].copy()
    injury_export.columns = ['Team', 'Player', 'Position', 'Injury', 'Status']
    injury_export.to_csv(OUT_DIR / 'nfl_injury_report.csv', index=False)
    print(f"Success! {len(injury_export)} injury report entries saved.")
except Exception as e:
    print(f"⚠ Injury report pull failed, matchup sheet will show 'No injury data available': {e}")
    pd.DataFrame(columns=['Team', 'Player', 'Position', 'Injury', 'Status']).to_csv(OUT_DIR / 'nfl_injury_report.csv', index=False)

# ==============================================================================
# PART 6: PLAYER-LEVEL FANTASY / PROP USAGE + DEFENSE-VS-POSITION MATCHUPS
# NEW -- added for the fantasy/prop angle of the matchup sheet. Uses
# nflverse's own official weekly player_stats table (the same dataset
# FantasyPros/ESPN-style target_share and fantasy_points_ppr numbers come
# from) rather than hand-computing scoring, so this isn't reinventing a
# scoring formula -- it's the real nflverse numbers, season-to-date.
#
# Column names below match nflverse's documented player_stats schema
# (player_display_name, position, team, opponent_team, targets, target_share,
# receptions, receiving_yards, carries, rushing_yards, fantasy_points_ppr,
# etc.) -- wrapped defensively, same as the rest of this script, so a schema
# change in a future nflverse release degrades to "--" in the PDF instead of
# crashing the whole weekly pull.
# ==============================================================================

print("Step 8: Pulling player-level usage stats for the fantasy/prop watch...")

def _first_present(df, candidates, default=None):
    """Returns the first column name in `candidates` that actually exists in
    df, or `default` if none do -- guards against nflverse renaming a column
    between releases."""
    for c in candidates:
        if c in df.columns:
            return c
    return default

# Team-level QB rushing profile (mobile vs. pocket passer) -- filled in
# below if the player-stats pull succeeds; stays this empty shape (renders
# as "--" everywhere) if it doesn't, same fallback pattern as everything else.
qb_profile = pd.DataFrame(columns=['team', 'starting_qb_name', 'qb_rush_attempts_pg', 'qb_rush_yards_pg', 'qb_mobility_tag'])

# Team-level passing-game volume (targets/gm, receptions/gm) -- filled in
# below from the same real nflverse player_stats already loaded for the
# Fantasy Watch panel, so it's consistent with those per-player numbers
# rather than a separately-sourced count. Same empty-shape fallback pattern.
team_targets_receptions = pd.DataFrame(columns=['team', 'targets_per_game', 'receptions_per_game', 'catch_pct', 'catch_pct_allowed'])

try:
    pstats = nfl.load_player_stats(summary_level='week', **SEASON_KWARGS).to_pandas()
    pstats = pstats[pstats.get('season_type', 'REG') == 'REG'] if 'season_type' in pstats.columns else pstats

    name_col = _first_present(pstats, ['player_display_name', 'player_name'])
    team_col = _first_present(pstats, ['team', 'recent_team'])
    pos_col = _first_present(pstats, ['position'])
    week_col = _first_present(pstats, ['week'])

    if not all([name_col, team_col, pos_col, week_col]):
        raise KeyError(f"Couldn't find expected player/team/position/week columns; got {list(pstats.columns)[:20]}...")

    skill = pstats[pstats[pos_col].isin(['QB', 'RB', 'WR', 'TE'])].copy()

    numeric_cols = ['targets', 'target_share', 'receptions', 'receiving_yards', 'receiving_tds',
                     'carries', 'rushing_yards', 'rushing_tds', 'passing_yards', 'passing_tds',
                     'fantasy_points', 'fantasy_points_ppr']
    for c in numeric_cols:
        if c in skill.columns:
            skill[c] = pd.to_numeric(skill[c], errors='coerce').fillna(0)
        else:
            skill[c] = 0.0

    skill = skill.sort_values(week_col)
    # total TDs/game = rush + rec + pass TDs, computed once up front so the
    # main aggregation below can just average it like every other column.
    skill['all_tds'] = skill['rushing_tds'] + skill['receiving_tds'] + skill['passing_tds']

    # ------------------------------------------------------------------
    # TEAM-level targets/gm and receptions/gm -- real nflverse targets/
    # receptions columns (same ones already powering each player's row in
    # nfl_player_stats.csv), summed across every skill player on the team
    # per week, then averaged over the weeks that team actually played
    # (bye weeks naturally drop out since no player has a row for them).
    # This is a straight real aggregation, not a new formula.
    # ------------------------------------------------------------------
    team_week = skill.groupby([team_col, week_col]).agg(
        targets=('targets', 'sum'), receptions=('receptions', 'sum')
    ).reset_index()
    team_targets_receptions = team_week.groupby(team_col).agg(
        targets_per_game=('targets', 'mean'), receptions_per_game=('receptions', 'mean'),
        tot_targets_season=('targets', 'sum'), tot_receptions_season=('receptions', 'sum'),
    ).reset_index().rename(columns={team_col: 'team'})
    # CATCH %: real receptions / real targets, season totals (not an average
    # of weekly rates -- a team's true season catch % isn't the mean of its
    # weekly catch %'s, since weeks with more targets should count for more).
    # Distinct from COMPLETION % elsewhere on this sheet: this is targets-
    # based (nflverse's official per-target credit), completion % is
    # attempts-based (pbp's pass_attempt count) -- independently sourced
    # measures of nearly the same real-world thing, so don't be surprised if
    # they're close but not bit-identical for a given team.
    team_targets_receptions['catch_pct'] = np.where(
        team_targets_receptions['tot_targets_season'] > 0,
        (team_targets_receptions['tot_receptions_season'] / team_targets_receptions['tot_targets_season']) * 100, np.nan)
    team_targets_receptions = team_targets_receptions.round(1)

    def _last3_ppr(group):
        return group.tail(3)['fantasy_points_ppr'].mean()

    agg = skill.groupby([name_col, team_col, pos_col]).agg(
        gp=(week_col, 'nunique'),
        targets_pg=('targets', 'mean'),
        target_share_pct=('target_share', lambda x: x.mean() * 100),
        carries_pg=('carries', 'mean'),
        receptions_pg=('receptions', 'mean'),
        rec_yards_pg=('receiving_yards', 'mean'),
        rush_yards_pg=('rushing_yards', 'mean'),
        pass_yards_pg=('passing_yards', 'mean'),
        total_tds_pg=('all_tds', 'mean'),
        ppr_pts_pg=('fantasy_points_ppr', 'mean'),
    ).reset_index()
    agg.columns = ['player', 'team', 'position', 'gp', 'targets_pg', 'target_share_pct',
                   'carries_pg', 'receptions_pg', 'rec_yards_pg', 'rush_yards_pg', 'pass_yards_pg',
                   'total_tds_pg', 'ppr_pts_pg']

    last3 = skill.groupby([name_col, team_col, pos_col])[[week_col, 'fantasy_points_ppr']].apply(_last3_ppr).reset_index()
    last3.columns = ['player', 'team', 'position', 'ppr_pts_last3_pg']
    agg = agg.merge(last3, on=['player', 'team', 'position'], how='left')

    # Only keep players with meaningful season-long usage -- a single garbage-
    # time snap shouldn't produce a "starter-caliber" looking row.
    agg = agg[agg['gp'] >= 1]

    # ------------------------------------------------------------------
    # QB rushing profile -- mobile vs. pocket passer, using each team's
    # PRIMARY QB (highest PPR pts/gm at the position) this season. "Mobile"
    # is OUR OWN threshold (rush_yards_pg >= 15) -- not an official league
    # classification, just a quick visual tag, documented as such in the
    # README. Combine with time-to-throw (added below) for the fuller
    # pocket-presence picture.
    # ------------------------------------------------------------------
    qb_rows = agg[agg['position'] == 'QB'].sort_values(['team', 'ppr_pts_pg'], ascending=[True, False])
    qb_primary = qb_rows.groupby('team').first().reset_index()
    if not qb_primary.empty:
        qb_profile = qb_primary[['team', 'player', 'carries_pg', 'rush_yards_pg']].rename(
            columns={'player': 'starting_qb_name', 'carries_pg': 'qb_rush_attempts_pg', 'rush_yards_pg': 'qb_rush_yards_pg'}
        )
        qb_profile['qb_mobility_tag'] = np.where(qb_profile['qb_rush_yards_pg'] >= 15, 'Mobile', 'Pocket')

    # Snap % -- separate dataset (Pro Football Reference via nflverse), best-effort.
    try:
        snaps = nfl.load_snap_counts(**SEASON_KWARGS).to_pandas()
        snap_name_col = _first_present(snaps, ['player', 'pfr_player_name', 'full_name'])
        snap_pct_col = _first_present(snaps, ['offense_pct'])
        if snap_name_col and snap_pct_col:
            snap_avg = snaps.groupby(snap_name_col)[snap_pct_col].mean().reset_index()
            snap_avg.columns = ['player', 'snap_pct_avg']
            snap_avg['snap_pct_avg'] = snap_avg['snap_pct_avg'] * 100
            agg = agg.merge(snap_avg, on='player', how='left')
        else:
            agg['snap_pct_avg'] = np.nan
    except Exception as snap_err:
        print(f"⚠ Snap count pull failed, snap_pct_avg will show '--': {snap_err}")
        agg['snap_pct_avg'] = np.nan

    agg = agg.round(1).sort_values(['team', 'position', 'ppr_pts_pg'], ascending=[True, True, False])
    agg.to_csv(OUT_DIR / 'nfl_player_stats.csv', index=False)
    print(f"Success! {len(agg)} skill-position player rows saved to nfl_player_stats.csv.")

    # ----------------------------------------------------------------------
    # Defense-vs-position: how many PPR fantasy points has each team's
    # defense allowed, per game, to each offensive position -- the standard
    # "start/sit vs this defense" matchup signal. Rank 1 = allows the MOST
    # (best matchup for the offense facing them), rank 32 = stingiest.
    # ----------------------------------------------------------------------
    opp_col = _first_present(pstats, ['opponent_team'])
    if opp_col:
        skill_for_def = skill.copy()
        per_game = skill_for_def.groupby([opp_col, pos_col, week_col])['fantasy_points_ppr'].sum().reset_index()
        dvp = per_game.groupby([opp_col, pos_col])['fantasy_points_ppr'].mean().reset_index()
        dvp.columns = ['team', 'position', 'fpts_allowed_pg']
        dvp['rank_worst_matchup_1_to_32'] = dvp.groupby('position')['fpts_allowed_pg'].rank(ascending=False, method='min').astype(int)
        dvp = dvp.round(1).sort_values(['position', 'rank_worst_matchup_1_to_32'])
        dvp.to_csv(OUT_DIR / 'nfl_def_vs_position.csv', index=False)
        print(f"Success! Defense-vs-position matchup table saved ({len(dvp)} team/position rows).")

        # CATCH % ALLOWED (defense) -- same real targets-based methodology
        # as CATCH % above, re-attributed via nflverse's own 'opponent_team'
        # column (already used for the defense-vs-position table just
        # above) instead of 'team': every target/reception counted against
        # the DEFENSE that actually covered it that week, not the offense
        # that gained it. Keeps the offense/defense sides of the SKILL
        # BATTLE panel's catch % on the same real, targets-based footing.
        opp_week_catch = skill.groupby([opp_col, week_col]).agg(
            targets=('targets', 'sum'), receptions=('receptions', 'sum')
        ).reset_index()
        opp_season_catch = opp_week_catch.groupby(opp_col).agg(
            tot_targets_allowed=('targets', 'sum'), tot_receptions_allowed=('receptions', 'sum')
        ).reset_index().rename(columns={opp_col: 'team'})
        opp_season_catch['catch_pct_allowed'] = np.where(
            opp_season_catch['tot_targets_allowed'] > 0,
            (opp_season_catch['tot_receptions_allowed'] / opp_season_catch['tot_targets_allowed']) * 100, np.nan)
        team_targets_receptions = team_targets_receptions.merge(
            opp_season_catch[['team', 'catch_pct_allowed']], on='team', how='left')
    else:
        print("⚠ No opponent_team column found -- skipping defense-vs-position table.")
        pd.DataFrame(columns=['team', 'position', 'fpts_allowed_pg', 'rank_worst_matchup_1_to_32']).to_csv(OUT_DIR / 'nfl_def_vs_position.csv', index=False)

    if 'catch_pct_allowed' not in team_targets_receptions.columns:
        team_targets_receptions['catch_pct_allowed'] = np.nan

except Exception as e:
    print(f"⚠ Player-level fantasy/prop pull failed, matchup sheet's Fantasy Watch panel will show 'No data available': {e}")
    pd.DataFrame(columns=['player', 'team', 'position', 'gp', 'targets_pg', 'target_share_pct',
                           'carries_pg', 'receptions_pg', 'rec_yards_pg', 'rush_yards_pg', 'pass_yards_pg',
                           'total_tds_pg', 'ppr_pts_pg', 'ppr_pts_last3_pg', 'snap_pct_avg']).to_csv(OUT_DIR / 'nfl_player_stats.csv', index=False)
    pd.DataFrame(columns=['team', 'position', 'fpts_allowed_pg', 'rank_worst_matchup_1_to_32']).to_csv(OUT_DIR / 'nfl_def_vs_position.csv', index=False)

master = master.merge(qb_profile, on='team', how='left')
columns_to_export += ['starting_qb_name', 'qb_rush_attempts_pg', 'qb_rush_yards_pg', 'qb_mobility_tag']

master = master.merge(team_targets_receptions[['team', 'targets_per_game', 'receptions_per_game', 'catch_pct', 'catch_pct_allowed']], on='team', how='left')
columns_to_export += ['targets_per_game', 'receptions_per_game', 'catch_pct', 'catch_pct_allowed']

# ==============================================================================
# PART 6B: NEXT GEN STATS ADVANCED METRICS
# NEW -- team-level advanced passing/rushing/receiving metrics from
# nflverse's NGS (Next Gen Stats -- real player-tracking data licensed from
# the NFL, published under nflverse's CC BY 4.0 license). A free, real,
# nflverse-hosted dataset -- not an estimate. Each is aggregated to one
# number per team with a simple average across that team's players/weeks
# (not attempt-weighted) -- a solid team-level signal, not a precise
# weighted total.
#
# Covers: CPOE, NGS time to throw, RYOE per attempt, target separation, and
# team YAC.
#
# STRIPPED OUT (see README.txt "LICENSING -- WHAT WAS REMOVED AND WHY"):
# this section used to also pull Pro Football Reference (PFR) advanced
# stats via nflreadpy's load_pfr_advstats() -- PFR "pocket time"/time to
# throw, QB pressure %, times blitzed/hurried/hit, drop %, adjusted
# completion %, and yards before contact. That data originates from PFR's
# own charting regardless of which library fetches it, and PFR's terms of
# use prohibit building a tool/product on scraped PFR data without their
# permission. Removed entirely rather than sell something built on data we
# don't have clear rights to resell. Time to throw is still covered for
# free below (NGS's own number). QB pressure is now a real, independently
# computed free proxy from raw pbp -- see 'pressure_rate_allowed_pct' /
# 'pressure_rate_forced_pct' in PART 2C. Drop %, adjusted completion %, and
# yards before contact have no free equivalent (they require PFR's manual
# charting or real tracking data) and are simply gone, not faked.
#
# EVERY metric below is independently wrapped: _pull_ngs_pfr_metric() always
# returns a valid (team, stat) frame, filled with real numbers on success or
# all-NaN (renders as "--") on any failure -- so nflverse renaming one
# column, or one of these loaders being temporarily down, never costs you
# the other metrics or takes out the rest of the weekly pull.
# ==============================================================================

print("Step 8b: Pulling Next Gen Stats advanced metrics...")

ngs_pfr_landed = []

def _pull_ngs_pfr_metric(loader_df, team_col_candidates, stat_candidates, out_name, label):
    try:
        d = loader_df.copy()
        team_col = _first_present(d, team_col_candidates)
        stat_col = _first_present(d, stat_candidates)
        if not team_col or not stat_col:
            raise KeyError(f"columns not found in {list(d.columns)[:15]}...")
        d[stat_col] = pd.to_numeric(d[stat_col], errors='coerce')
        out = d.groupby(team_col)[stat_col].mean().reset_index()
        out.columns = ['team', out_name]
        ngs_pfr_landed.append(out_name)
        print(f"  ✓ {label}")
    except Exception as e:
        print(f"  ⚠ {label} unavailable this pull ('--' in PDF): {e}")
        out = pd.DataFrame({'team': master['team'].unique(), out_name: np.nan})
    return out

# --- Next Gen Stats: passing (CPOE, NGS time to throw) ---
try:
    ngs_pass = nfl.load_nextgen_stats(stat_type='passing', **SEASON_KWARGS).to_pandas()
    if 'week' in ngs_pass.columns:
        ngs_pass = ngs_pass[ngs_pass['week'] > 0]  # drop the season-total row so this is a real weekly average
except Exception:
    ngs_pass = pd.DataFrame()
ngs_pass_team_col = _first_present(ngs_pass, ['team_abbr', 'team']) or 'team'
adv_cpoe = _pull_ngs_pfr_metric(ngs_pass, ['team_abbr', 'team'], ['completion_percentage_above_expectation'], 'cpoe', 'CPOE (completion % over expected)')
adv_ngs_ttt = _pull_ngs_pfr_metric(ngs_pass, ['team_abbr', 'team'], ['avg_time_to_throw'], 'ngs_time_to_throw', 'NGS avg time to throw')

# --- Next Gen Stats: rushing (RYOE per attempt) ---
try:
    ngs_rush = nfl.load_nextgen_stats(stat_type='rushing', **SEASON_KWARGS).to_pandas()
    if 'week' in ngs_rush.columns:
        ngs_rush = ngs_rush[ngs_rush['week'] > 0]
except Exception:
    ngs_rush = pd.DataFrame()
adv_ryoe = _pull_ngs_pfr_metric(ngs_rush, ['team_abbr', 'team'], ['rush_yards_over_expected_per_att'], 'ryoe_per_attempt', 'RYOE per attempt')

# --- Next Gen Stats: receiving (target separation, team YAC) ---
try:
    ngs_rec = nfl.load_nextgen_stats(stat_type='receiving', **SEASON_KWARGS).to_pandas()
    if 'week' in ngs_rec.columns:
        ngs_rec = ngs_rec[ngs_rec['week'] > 0]
except Exception:
    ngs_rec = pd.DataFrame()
adv_sep = _pull_ngs_pfr_metric(ngs_rec, ['team_abbr', 'team'], ['avg_separation'], 'target_separation', 'Target separation')
adv_yac = _pull_ngs_pfr_metric(ngs_rec, ['team_abbr', 'team'], ['avg_yac'], 'team_yac_per_reception', 'Team YAC/reception')

# --- Target separation ALLOWED (defense) -- for the SKILL BATTLE panel.
# NGS's real weekly separation numbers are tagged to the RECEIVING team,
# not to whoever covered them, so there's no direct "allowed" column to
# pull. Re-attributed here to whichever defense actually faced that
# offense that week, using the real week-by-week schedule already visible
# in pbp (posteam/defteam per week, no extra network call). Same real NGS
# numbers nflverse already publishes -- just regrouped by opponent instead
# of by own team. Not a new model, not an estimate of separation.
try:
    week_opp = pbp_reg[['week', 'posteam', 'defteam']].drop_duplicates()
    sep_team_col = _first_present(ngs_rec, ['team_abbr', 'team'])
    if sep_team_col and 'week' in ngs_rec.columns and 'avg_separation' in ngs_rec.columns:
        ngs_rec_j = ngs_rec.copy()
        ngs_rec_j['avg_separation'] = pd.to_numeric(ngs_rec_j['avg_separation'], errors='coerce')
        ngs_rec_j = ngs_rec_j.merge(
            week_opp, left_on=['week', sep_team_col], right_on=['week', 'posteam'], how='inner'
        )
        adv_sep_allowed = ngs_rec_j.groupby('defteam')['avg_separation'].mean().reset_index()
        adv_sep_allowed.columns = ['team', 'target_separation_allowed']
        ngs_pfr_landed.append('target_separation_allowed')
        print("  ✓ Target separation allowed (defense, re-attributed via real schedule)")
    else:
        raise KeyError(f"avg_separation/week/team column missing from NGS receiving pull; got {list(ngs_rec.columns)[:15]}...")
except Exception as e:
    print(f"  ⚠ Target separation allowed unavailable this pull ('--' in PDF): {e}")
    adv_sep_allowed = pd.DataFrame({'team': master['team'].unique(), 'target_separation_allowed': np.nan})

for frame in [adv_cpoe, adv_ngs_ttt, adv_ryoe, adv_sep, adv_yac, adv_sep_allowed]:
    master = master.merge(frame, on='team', how='left')
    new_col = [c for c in frame.columns if c != 'team'][0]
    if new_col not in columns_to_export:
        columns_to_export.append(new_col)

print(f"Step 8b complete -- {len(ngs_pfr_landed)}/6 NGS advanced metrics landed this pull: {ngs_pfr_landed}")

# ==============================================================================
# PART 6C: ESPN FOOTBALL POWER INDEX (FPI) -- REMOVED
# This section used to pull ESPN's proprietary Football Power Index from
# ESPN's undocumented API. Removed entirely -- see README.txt "LICENSING --
# WHAT WAS REMOVED AND WHY". ESPN's terms of use (Disney's Terms of Use,
# which cover espn.com) prohibit both scraped/automated access to their
# content and any commercial use of it, with no carve-out for a "hidden"
# API. There's also no independent way to reproduce FPI -- it's ESPN's own
# proprietary blend of offensive/defensive/special-teams efficiency, not
# something with a public formula, unlike DVOA where VOID could be built as
# a genuinely separate computation. VOID (PART 7 below) already fills the
# same "which team is actually good, adjusted for opponent" role on this
# sheet, computed entirely from licensed nflverse data -- nothing is
# missing here, just an unlicensed duplicate.
# ==============================================================================

# ==============================================================================
# PART 7: VOID -- Value Over Interactive Average
# NEW -- an opponent-adjusted, situational-success-based team efficiency
# rating, in the same conceptual family as Football Outsiders/FTN's DVOA,
# but NOT that stat: it's our own independently-built formula, computed
# entirely from real nflverse play-by-play. DVOA itself can't be freely
# reproduced (its exact play-value curve and opponent-adjustment iteration
# are FTN's proprietary IP, refined over 20 years), so this doesn't try to
# match their number -- it borrows the same two ideas (situational success
# value + strength-of-opponent adjustment) using methods that ARE public:
#
#   1. SITUATIONAL VALUE per play -- how much of the yardage needed for a
#      fresh set of downs did this play actually gain, given the down. This
#      mirrors the publicly-described SHAPE of DVOA's success-value concept
#      (1st down needs less of the "to-go" yardage to count as good, 3rd/4th
#      down needs all of it), but our exact thresholds/bonus curve are our
#      own choice, not FTN's published formula.
#   2. OPPONENT ADJUSTMENT -- each team's raw value is adjusted by how
#      tough the defenses/offenses they've actually played have been,
#      using the single-pass "opponent-average" method described publicly
#      by the nflfastR/Open Source Football community (not an iterative
#      solve -- a reasonable, transparent approximation of one).
#
# Positive VOID = better than league average, for BOTH offense and defense
# (this is a deliberate, documented difference from DVOA's own sign
# convention, where defense is "good" when negative -- picked here so VOID
# reads the same intuitive way on both sides of the ball).
#
# KNOWN MATH LIMITATION -- WEEK 1 (or any team with < 2 games played): the
# single-pass opponent adjustment algebraically CANCELS ITSELF OUT to
# exactly the same constant for every team whenever a team has played only
# ONE opponent so far (confirmed on a real live pull) -- because that one
# opponent's own raw number was, by definition, computed from the SAME set
# of plays as this team's raw number (same game, viewed from the other
# side of the ball), so the "how tough was their opponent" adjustment term
# and the team's own raw term cancel exactly. This isn't a guess -- it's an
# algebraic identity of the method, not a data problem, and it only shows
# up this cleanly in a small-sample edge case. Rather than show a
# misleadingly precise (and identical-for-everyone) number, VOID is held
# back to "--" for any team with fewer than 2 games played -- see the gp
# gate right before void_final is built below. It becomes meaningful again
# as soon as a team's opponent pool has more than one team in it.
# ==============================================================================

print("Step 9: Computing VOID (Value Over Interactive Average) ratings...")

VOID_NEED_FRAC = {1: 0.45, 2: 0.60, 3: 1.00, 4: 1.00}

def _play_value(row):
    down = row['down']
    if down not in VOID_NEED_FRAC or row['ydstogo'] <= 0:
        return np.nan
    need_yards = VOID_NEED_FRAC[down] * row['ydstogo']
    value = row['yards_gained'] / need_yards
    value = max(min(value, 1.5), -1.5)  # clip extremes so one huge/awful play can't swing a team's whole rating
    # Explosive-play bonuses, same spirit as DVOA's bonus points for 10/20/40+ yard gains
    if row['yards_gained'] >= 40:
        value += 0.6
    elif row['yards_gained'] >= 20:
        value += 0.4
    elif row['yards_gained'] >= 10:
        value += 0.2
    # Turnovers are always strongly negative regardless of yardage
    if row.get('interception', 0) or row.get('fumble_lost', 0):
        value = min(value, -1.5)
    # A touchdown is always a clearly positive play, even on a small-yardage goal-to-go plunge
    if row.get('touchdown', 0):
        value = max(value, 1.2)
    return value

void_plays = pbp_clean[pbp_clean['down'].isin([1, 2, 3, 4]) & (pbp_clean['ydstogo'] > 0)].copy()
if 'touchdown' not in void_plays.columns:
    void_plays['touchdown'] = void_plays.get('pass_touchdown', 0) + void_plays.get('rush_touchdown', 0)
void_plays['void_value'] = void_plays.apply(_play_value, axis=1)
void_plays = void_plays[void_plays['void_value'].notna()]
print(f"  VOID diagnostic: {len(void_plays)} qualifying plays across {void_plays['game_id'].nunique() if len(void_plays) else 0} game(s).")

raw_off = void_plays.groupby('posteam')['void_value'].mean().reset_index().rename(columns={'posteam': 'team', 'void_value': 'raw_off_void'})
raw_def = void_plays.groupby('defteam')['void_value'].mean().reset_index().rename(columns={'defteam': 'team', 'void_value': 'raw_def_void_allowed'})
void_df = raw_off.merge(raw_def, on='team', how='outer')
print(f"  VOID diagnostic: raw_off_void range [{void_df['raw_off_void'].min():.3f}, {void_df['raw_off_void'].max():.3f}], "
      f"raw_def_void_allowed range [{void_df['raw_def_void_allowed'].min():.3f}, {void_df['raw_def_void_allowed'].max():.3f}] across {len(void_df)} team(s).")
void_df = void_df.fillna(0)

league_avg_off_void = void_df['raw_off_void'].mean()
league_avg_def_void_allowed = void_df['raw_def_void_allowed'].mean()

# Games played per team, from Part 1 -- used just below to gate VOID on a
# real opponent pool (see the KNOWN MATH LIMITATION note above this block).
gp_lookup = games_played.set_index('team')['gp'].to_dict()

# Build each team's list of opponents faced this season from the same
# game-by-game schedule already derived in Part 1 (final_scores).
opponents_faced = {}
for _, g in final_scores.iterrows():
    opponents_faced.setdefault(g['home'], []).append(g['away'])
    opponents_faced.setdefault(g['away'], []).append(g['home'])

void_lookup_off = void_df.set_index('team')['raw_off_void'].to_dict()
void_lookup_def = void_df.set_index('team')['raw_def_void_allowed'].to_dict()

adj_rows = []
for team, opps in opponents_faced.items():
    opp_def_values = [void_lookup_def.get(o, league_avg_def_void_allowed) for o in opps]
    opp_off_values = [void_lookup_off.get(o, league_avg_off_void) for o in opps]
    avg_opp_def_faced = np.mean(opp_def_values) if opp_def_values else league_avg_def_void_allowed
    avg_opp_off_faced = np.mean(opp_off_values) if opp_off_values else league_avg_off_void

    raw_off = void_lookup_off.get(team, league_avg_off_void)
    raw_def_allowed = void_lookup_def.get(team, league_avg_def_void_allowed)

    # Beat tougher-than-average defenses -> bonus to offense. Faced
    # tougher-than-average offenses -> bonus to defense (lowers the
    # allowed-value number, since lower allowed = better defense).
    off_adjustment = league_avg_def_void_allowed - avg_opp_def_faced
    def_adjustment = avg_opp_off_faced - league_avg_off_void

    adj_off_void = raw_off + off_adjustment
    adj_def_void_allowed = raw_def_allowed - def_adjustment

    # Gate: with only 1 game played, the opponent adjustment algebraically
    # cancels itself out to the same constant for every team (see the
    # KNOWN MATH LIMITATION note above) -- real NaN here instead of a
    # falsely-precise, identical-for-everyone number. Needs >=2 games
    # played, which also means >=2 DISTINCT opponents in almost every real
    # case (a team can't play the same opponent twice this early).
    if gp_lookup.get(team, 0) < 2:
        adj_off_void_final, adj_def_void_final = np.nan, np.nan
    else:
        adj_off_void_final = adj_off_void - league_avg_off_void
        adj_def_void_final = league_avg_def_void_allowed - adj_def_void_allowed  # flipped so positive = good defense too

    adj_rows.append({
        'team': team,
        'off_void': adj_off_void_final,
        'def_void': adj_def_void_final,
    })

void_final = pd.DataFrame(adj_rows)
void_final['total_void'] = void_final['off_void'] + void_final['def_void']
# Nullable Int64 (not plain int) -- a team held back by the gp gate above
# has NaN here, and plain .astype(int) would crash on that; Int64 keeps it
# as real missing data, which the PDF renders as "--".
void_final['off_void_rank'] = void_final['off_void'].rank(ascending=False, method='min').astype('Int64')
void_final['def_void_rank'] = void_final['def_void'].rank(ascending=False, method='min').astype('Int64')
void_final['total_void_rank'] = void_final['total_void'].rank(ascending=False, method='min').astype('Int64')
void_final = void_final.round(3)
teams_gated = [t for t in void_final['team'] if gp_lookup.get(t, 0) < 2]
if teams_gated:
    print(f"  VOID: {len(teams_gated)} team(s) held to '--' this week (fewer than 2 games played -- see KNOWN MATH LIMITATION note): {teams_gated}")

master = master.merge(void_final, on='team', how='left')
columns_to_export += ['off_void', 'def_void', 'total_void', 'off_void_rank', 'def_void_rank', 'total_void_rank']

# NOTE on the fillna(0) below: it's correct for every "real, always-computable
# from pbp" column (0 fumbles, 0 penalty yards, etc. are genuine values), but
# WRONG for the NGS columns above -- those go NaN specifically when a pull
# failed or a column wasn't found, and a 0 there would misread as "this team
# had 0% CPOE" instead of "we don't have this number." Those columns are
# excluded here and stay true NaN -- blank in the CSV, "--" in the PDF.
NO_FILLNA_COLS = [
    'cpoe', 'ngs_time_to_throw', 'ryoe_per_attempt', 'target_separation', 'team_yac_per_reception',
    'starting_qb_name', 'qb_mobility_tag',
    # Team-level targets/gm + receptions/gm (from the same player-stats pull
    # as the Fantasy Watch panel) -- 0 is never a real value for a team's
    # season-long targets/gm, so a failed pull must show '--', not a fake 0.
    'targets_per_game', 'receptions_per_game', 'catch_pct', 'catch_pct_allowed',
    # Personnel/formation/defensive-personnel tendencies (PART 2D) -- real
    # NaN when participation data wasn't available that pull, not a
    # genuine 0% usage.
    *SCHEME_COLS,
    # Coordinator-attributed coverage tendencies (PART 2E) -- real NaN for a
    # rookie coordinator with no prior playcalling season, or a team missing
    # from the mapping -- must stay '--', never a fake 0%/0.000.
    *COORD_COLS,
    # Interior/edge pressure index (PART 2F) -- real NaN when the whole
    # computation failed this pull (never a fake 0%/0.000 interior share).
    *PRESSURE_INDEX_COLS,
    # VOID columns, gated to NaN for any team with < 2 games played (see the
    # KNOWN MATH LIMITATION note above PART 7) -- must NOT get fillna(0)'d,
    # or a genuinely-not-computable rating would masquerade as a real
    # "0.000, rank 1/32" instead of showing "--".
    'off_void', 'def_void', 'total_void', 'off_void_rank', 'def_void_rank', 'total_void_rank',
    # SKILL BATTLE panel (receivers vs. DBs) -- secondary lock index, pbp-
    # based YAC/reception (+ allowed), and re-attributed separation allowed
    # all go NaN on a genuine pull failure (missing 'yards_after_catch' or
    # NGS receiving data), never a fake 0 -- a "0.0 yards/target allowed"
    # or "0.0 yards after catch" would misread as a real (impossible)
    # value, not as unavailable data.
    'secondary_lock_index', 'yac_per_reception_pbp', 'yac_allowed_per_reception_pbp', 'target_separation_allowed',
    # Special teams AVERAGES (not counts): a team that never punted or never
    # attempted a FG has no real average, not a real 0.0 -- 0.0 yards/punt
    # would misread as "this team's punts travel zero yards," an impossible
    # value, instead of "no punts this pull." Counting stats (punts/gm,
    # blocks, special teams TDs) stay on the normal fillna(0) path below --
    # a genuine 0 of those IS a real value.
    'avg_punt_distance', 'punt_return_avg', 'avg_fg_distance',
]
final_matrix = master[columns_to_export].copy()
fill_cols = [c for c in columns_to_export if c not in NO_FILLNA_COLS]
final_matrix[fill_cols] = final_matrix[fill_cols].fillna(0)
final_matrix.to_csv(team_matrix_path, index=False)
print(f"Success! VOID ratings + NGS advanced metrics added to {team_matrix_path.name}.")

# ==============================================================================
# END OF PART 7 / END OF SCRIPT
# ==============================================================================
# ==============================================================================
# PART 8: PREMIUM MANUAL-ENTRY TEMPLATES -- REMOVED
# This used to write nfl_premium_team_manual.csv / nfl_premium_player_manual.csv
# as fill-in-by-hand placeholders for real DVOA, pass/run block & pass
# rush/run stop win rate, double-team rate, interior/edge pressure rate,
# Yards Per Route Run, and Juke Rate -- all PFF+ or FTN Fantasy sourced.
# Removed entirely -- see README.txt "LICENSING -- WHAT WAS REMOVED AND WHY".
# Even with a paid PFF+/FTN subscription, their terms explicitly prohibit
# incorporating that data into anything sold to someone else -- a
# subscription buys personal viewing access, not resale rights, so there
# was no legitimate path for this panel to ever be filled in for a product
# you sell. DVOA's role is already covered by VOID (PART 7) for free.
# Interior/edge pressure rate now has a real, independently-computed
# substitute too -- see PART 2F (Tug o' War / Edge Pressure), built from
# player-attributed sacks/hits, not PFF's charting. It's honestly a
# narrower stat than PFF's (no hurries -- see PART 2F's own writeup on why),
# not a reproduction of their number. Win-rate, double-team rate, YPRR, and
# juke rate still have no free or independently-computable substitute (they
# require proprietary player-tracking or manual film charting nobody
# publishes openly) -- still gone, not faked.
# ==============================================================================
