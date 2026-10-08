# ==============================================================================
# nfl_matchup_sheet.py
# ==============================================================================
# Step 2 of the weekly routine (see README.txt). Reads the CSVs nfl_data_pull.py
# just wrote, pulls this week's live schedule from nflverse (load_schedules --
# CC BY 4.0 licensed, same source as everything else in this pipeline; this
# used to pull from ESPN's undocumented scoreboard API, which was removed --
# see README.txt "LICENSING -- WHAT WAS REMOVED AND WHY"), and builds the PDF
# report(s). By default it writes BOTH:
#   nfl_full_slate_week<N>.pdf         -- every game this week, one combined PDF
#   matchups_week<N>/<AWAY>_at_<HOME>.pdf  -- the SAME content, one PDF per game
# so you can sell either the full-week bundle or a single matchup on its own.
# Use --bundle-only / --singles-only to build just one of the two.
#
# The sold PDF is DESCRIPTIVE only -- real stats, the VOID rating, injuries,
# fantasy usage. The model win-probability/spread/total panel is deliberately
# NOT included (a specific-game forecast can be publicly, embarrassingly
# wrong in a way a "here's what this team has actually done" stat can't) --
# it still runs quietly every week into nfl_internal_model_track.csv, for
# your own eyes, so you can build a real accuracy track record before ever
# deciding whether to put it in front of customers. See README.txt.
# ==============================================================================
# SECTION 1: LIVE SCHEDULE PARSING & VISUAL CANVAS SETUP
# PART 1 BOUNDARY LINE START
# ==============================================================================

import argparse
import json
import nflreadpy as nfl
import nfl_delta_grid
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfl_margin_model
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

OUT_DIR = Path(__file__).resolve().parent
# MERGED (2026-10-05): every PDF this script builds goes in generated/ (data/CSVs stay next to the script).
GEN_DIR = OUT_DIR / "generated"
GEN_DIR.mkdir(exist_ok=True)

cli = argparse.ArgumentParser(description="Build the weekly NFL matchup sheet PDF(s) from the CSVs nfl_data_pull.py produced.")
cli.add_argument("--bundle-only", action="store_true", help="Only build the combined full-slate PDF, skip individual matchup PDFs.")
cli.add_argument("--singles-only", action="store_true", help="Only build individual matchup PDFs, skip the combined full-slate PDF.")
cli.add_argument("--season", type=int, default=None, help="Season year for the schedule pull (default: nflverse's own current-season default).")
args = cli.parse_args()
BUILD_BUNDLE = not args.singles_only
BUILD_SINGLES = not args.bundle_only

print("Step 1: Reading local predictive data matrix sheet...")
csv_path = OUT_DIR / 'nfl_team_matrix.csv'
if not csv_path.exists():
    raise FileNotFoundError(f"Missing {csv_path.name}! Run nfl_data_pull.py first (see README.txt).")

df = pd.read_csv(csv_path)
# NET PPD = points/drive the offense scores minus points/drive its defense allows.
# This is the number the PPD row now PRINTS and the teal flag is judged on (so they always agree).
if {'points_per_drive', 'points_per_drive_allowed'} <= set(df.columns):
    df['net_ppd'] = df['points_per_drive'] - df['points_per_drive_allowed']

print("Step 1b: Reading current injury report...")
injury_path = OUT_DIR / 'nfl_injury_report.csv'
if injury_path.exists():
    injuries_df = pd.read_csv(injury_path)
else:
    injuries_df = pd.DataFrame(columns=['Team', 'Player', 'Position', 'Injury', 'Status'])

print("Step 1c: Reading player usage + defense-vs-position tables for the Fantasy Watch panel...")
player_path = OUT_DIR / 'nfl_player_stats.csv'
players_df = pd.read_csv(player_path) if player_path.exists() else pd.DataFrame()
dvp_path = OUT_DIR / 'nfl_def_vs_position.csv'
dvp_df = pd.read_csv(dvp_path) if dvp_path.exists() else pd.DataFrame()

print("Step 1d: Reading full per-game log for the Previous Schedule panel...")
game_log_path = OUT_DIR / 'nfl_game_log.csv'
game_log_df = pd.read_csv(game_log_path) if game_log_path.exists() else pd.DataFrame(
    columns=['team', 'week', 'opponent', 'is_home', 'result', 'pts_for', 'pts_against',
             'rush_yards', 'pass_yards', 'rush_yards_allowed', 'pass_yards_allowed'])

# NOTE: this used to also read nfl_premium_team_manual.csv / nfl_premium_
# player_manual.csv (fill-in-by-hand PFF+/FTN stats) and show a Premium
# Metrics panel built from them -- removed along with those CSVs; see
# README.txt "LICENSING -- WHAT WAS REMOVED AND WHY".

# Calculate dynamic league averages, standard deviations, and elite thresholds from master dataframe
# FIXED: sacks_allowed/tfl_allowed were missing from this list entirely, so
# the heat-map was coloring "more sacks/TFLs allowed" GREEN (as if it were
# good) instead of red -- backwards for an offensive stat. Also renamed to
# match the corrected column names from nfl_ml_dump.py (off_sacks/off_tfl
# no longer exist there).
lower_is_better_fields = [
    'def_efficiency_epa', 'points_allowed_per_game', 'total_yards_per_game_allowed',
    'def_yards_per_play', 'passing_yards_allowed_per_game', 'rushing_yards_allowed_per_game',
    'third_down_pct_defense', 'redzone_efficiency_allowed', 'yards_per_drive_allowed',
    'turnover_margin_offense', 'sacks_allowed', 'tfl_allowed',
    # Rush EPA/Play Allowed (defense) -- NEW (nfl_data_pull.py PART 2). A
    # defense wants LESS real EPA/play allowed on the ground, same direction
    # as def_efficiency_epa above.
    'def_rush_epa_allowed',
    # Line Yards Allowed (defense) -- NEW (nfl_data_pull.py, real Adjusted
    # Line Yards formula off real pbp). A defense wants LESS O-line-credited
    # yardage allowed, same direction as def_rush_epa_allowed just above.
    'def_line_yards_allowed',
    # VOID ranks: 1 = best, 32 = worst, so lower is better here too.
    'off_void_rank', 'def_void_rank', 'total_void_rank',
    # NEW (raw counting/situational + NGS/PFR advanced stats): fields where a
    # SMALLER number is the good outcome -- everything else added this round
    # defaults to "higher is better", which is correct for the rest.
    'completion_pct_allowed', 'yards_per_completion_allowed', 'sack_pct_offense',
    'interceptions_thrown_per_game', 'fumbles_lost_per_game', 'fg_blocked_per_game',
    'stuff_rate_offense', 'penalty_yards_per_game', 'pressure_rate_allowed_pct',
    # 4TH DOWN CONVERSION % ALLOWED (defense) and yards allowed PER PLAY,
    # split rush/pass -- same "defense wants less of this conceded" logic as
    # the *_allowed fields already above. Added when these three joined the
    # OFFENSE vs DEFENSE matchup panel (see MATCHUP_PAIRS below).
    'fourth_down_conversion_pct_allowed', 'def_rushing_yards_per_play', 'def_yards_per_pass',
    # MISCUE RATE (offense side only -- see nfl_data_pull.py): more of a
    # team's own targets going nowhere is bad for that offense. The
    # defense-side mirror is left at the default (higher = better), since a
    # defense forcing MORE of these leftover incompletions is a good
    # outcome for them even though the cause is genuinely ambiguous.
    'miscue_rate_offense',
    # AVG 3RD DOWN DISTANCE (offense side only -- see nfl_data_pull.py): a
    # SHORTER distance to go is the good outcome for the offense (means
    # 1st/2nd down were efficient). Defense side stays default (higher =
    # better -- forcing a longer 3rd down is good defense).
    'avg_3rd_down_distance_offense',
    # SKILL BATTLE "allowed" fields (receivers vs. DBs -- see
    # nfl_data_pull.py): a defense wants LESS of each of these conceded.
    'catch_pct_allowed', 'yac_allowed_per_reception_pbp', 'target_separation_allowed',
    # Special teams: more of THIS team's own punts getting blocked is bad.
    # (punts_per_game, avg_punt_distance, punt_return_avg, avg_fg_distance,
    # fg_blocks_forced_per_game, punt_blocks_forced_per_game, special_teams_
    # tds_per_game, pass_play_pct/rush_play_pct all stay at the default --
    # either genuinely higher-is-better or purely informational tendency
    # with no "good/bad" direction.)
    'punts_blocked_per_game',
    # REAL BUG FOUND AND FIXED (found while auditing every stat on the
    # sheet: last_game_off_pressure_rate_allowed -- the LAST-GAME version of
    # pressure allowed by this team's own O-line -- was missing from this
    # list, so it defaulted to higher-is-better and was coloring a team that
    # got pressured MORE last game as if that were GOOD (green/purple for a
    # high number). The season-long version of this exact same stat,
    # pressure_rate_allowed_pct, was already correctly listed above -- this
    # was the one place its single-game twin got left off. Confirmed with
    # real data (nfl_team_matrix.csv): this field runs ~24% average, 4-48%
    # range, same shape as the season version -- same fix applies.
    'last_game_off_pressure_rate_allowed',
]

league_avgs = df.mean(numeric_only=True)
league_stds = df.std(numeric_only=True)

# Real per-stat league-wide rank lookup (2026-09-27, same real ask/feature
# as CFB's "i need a rank beside every metric... like rush yards per play
# 5.6 (87)", now carried over to NFL across all 3 PDFs). Computed straight
# off this week's real nfl_team_matrix.csv rows (df), reusing
# lower_is_better_fields -- the exact same real, tested direction table
# get_stat_tier()/_qs_favorability() already rely on -- so a stat's rank
# always points the same way its color-grading already does; no separate
# direction judgment call is made here.
def _build_stat_rank_lookup(team_df):
    """Returns {column: {team_name: rank_int_or_None}}, one real league-
    wide rank per team per numeric stat, computed straight from the same
    nfl_team_matrix.csv values already displayed everywhere in this
    report -- not re-scraped or guessed. Ties share the same rank
    (pandas' 'min' method -- the standard sports-ranking convention: two
    teams tied for 2nd both show (2), the next team shows (4)). A team
    with no real value for a given stat gets None (no rank shown) rather
    than being silently counted as if it had one."""
    indexed = team_df.set_index('team')
    rank_lookup = {}
    for col in indexed.select_dtypes(include=[np.number]).columns:
        higher_is_better = col not in lower_is_better_fields
        series = pd.to_numeric(indexed[col], errors='coerce')
        ranks = series.rank(method='min', ascending=not higher_is_better, na_option='keep')
        rank_lookup[col] = {team: (int(r) if pd.notna(r) else None) for team, r in ranks.items()}
    return rank_lookup


def _with_rank(disp_text, column_key, team_name, rank_lookup=None):
    """Appends ' (rank)' to an already-formatted display string, e.g.
    '5.6' -> '5.6 (12)'. Leaves '--' (missing data), a None column key
    (e.g. a computed-on-the-fly field with no real df column, like
    net_epa_matchup), and any stat with no real rank available for this
    team untouched."""
    rank_lookup = rank_lookup if rank_lookup is not None else _STAT_RANK_LOOKUP
    if disp_text in ('--', None) or column_key is None or team_name is None:
        return disp_text
    rank = rank_lookup.get(column_key, {}).get(team_name)
    if rank is None:
        return disp_text
    return f"{disp_text} ({rank})"


# One real computation per run (df doesn't change after this point) --
# reused by every panel below, so "carry over to all 3 pdfs" (full slate
# bundle, singles, and the old man/quick-scan sheet) is automatic: they
# all share these same panel-building functions.
_STAT_RANK_LOOKUP = _build_stat_rank_lookup(df)

elite_thresholds = {}
# WORST_THRESHOLDS -- the mirror image of elite_thresholds: the real bottom
# X% (worst) of this season's 32 real teams for each field, direction-aware
# same as elite. Elite purple marks the best real teams; this marks the
# worst real teams orange, a distinct, rarer signal from the ordinary
# full-box red an outlier already gets -- "this isn't just bad, it's one of
# the worst real teams in the league at this specific stat." Both cutoffs
# tightened from top/bottom 5% to top/bottom 3% -- with only 32 real teams,
# that's close to just the single best/worst team per stat, an
# intentionally rare, high-bar signal.
worst_thresholds = {}

for col in df.select_dtypes(include=[np.number]).columns:
    if col in lower_is_better_fields:
        elite_thresholds[col] = df[col].quantile(0.03) # Top 3% lowest values are elite (was top 5%)
        worst_thresholds[col] = df[col].quantile(0.97) # Bottom 3% = highest values, for a lower-is-better stat
    else:
        elite_thresholds[col] = df[col].quantile(0.97) # Top 3% highest values are elite (was top 5%)
        worst_thresholds[col] = df[col].quantile(0.03) # Bottom 3% = lowest values, for a higher-is-better stat

# BUG FIXED: two teams that were far apart on NET EPA (MATCHUP) both rendered
# purple. NET EPA (MATCHUP)
# isn't a real column in df -- it's a per-matchup cross computation (this
# team's off EPA/play x1.5 minus the SPECIFIC opponent's def EPA/play
# allowed, see VOID_PAIRS/net_epa_matchup below), computed fresh on
# throwaway row copies right before the panel is built. That means it never
# got a row in the for-loop above, so get_stat_tier() was silently falling
# back to its unknown-field defaults (avg=0, elite=0, worst=0) for it --
# which meant ANY positive net_epa_matchup value counted as "elite" (>= 0)
# no matter how small, so two teams that were genuinely far apart (e.g.
# 0.145 vs 0.02) could both show purple just for both being positive.
# Fixed with REAL thresholds, not a guessed number: every real ORDERED pair
# of teams in the league (32 x 31 = 992 real matchups, self-matchups
# excluded) gets its real net_epa_matchup value computed the exact same way
# the sheet does, and elite/worst/avg/std become the real top-3%/bottom-3%/
# mean/std of THAT real distribution -- an honest empirical distribution
# built from real off_efficiency_epa/def_efficiency_epa data, not a normal-
# distribution approximation and not two separate thresholds bolted
# together.
_off_epa_vals = df['off_efficiency_epa'].to_numpy()
_def_epa_vals = df['def_efficiency_epa'].to_numpy()
_net_epa_matchup_universe = [
    _off_epa_vals[_i] * 1.5 - _def_epa_vals[_j]
    for _i in range(len(df)) for _j in range(len(df)) if _i != _j
]
_net_epa_matchup_universe = pd.Series(_net_epa_matchup_universe)
league_avgs['net_epa_matchup'] = _net_epa_matchup_universe.mean()
league_stds['net_epa_matchup'] = _net_epa_matchup_universe.std()
elite_thresholds['net_epa_matchup'] = _net_epa_matchup_universe.quantile(0.97)
worst_thresholds['net_epa_matchup'] = _net_epa_matchup_universe.quantile(0.03)

print("Step 2: Pulling active game schedule from nflverse (real games, CC BY 4.0 licensed)...")
matchups = []
week_number = None
# Kickoff day/time per matchup, keyed by (away, home) -- built from
# nflverse's own gameday/gametime columns. Shown in the PDF header.
schedule_info = {}
try:
    season_for_schedule = args.season if args.season else pd.Timestamp.now().year
    sched = nfl.load_schedules(seasons=[season_for_schedule]).to_pandas()
    sched = sched[sched['game_type'] == 'REG'].copy()
    sched['gameday'] = pd.to_datetime(sched['gameday'], errors='coerce')

    # Same "current NFL week" window logic as before: the actual
    # Tuesday-through-Monday date range, recalculated from today's date
    # every run, so it keeps working correctly every week going forward.
    today = datetime.now()
    days_since_tuesday = (today.weekday() - 1) % 7   # Mon=0 ... Tue=1 in Python's weekday()
    week_start = today - timedelta(days=days_since_tuesday)
    week_end = week_start + timedelta(days=6)          # the following Monday

    this_week = sched[(sched['gameday'] >= pd.Timestamp(week_start.date())) & (sched['gameday'] <= pd.Timestamp(week_end.date()))]

    # FIXED: if there are no games in the current Tue-Mon window (e.g. this
    # is run between seasons, or the schedule hasn't been published that
    # far out yet), fall back to whichever week has games closest to today
    # -- same spirit as the old ESPN "default current week" fallback --
    # instead of silently producing zero matchups.
    if this_week.empty and not sched.empty:
        sched_valid = sched.dropna(subset=['gameday']).copy()
        sched_valid['days_from_today'] = (sched_valid['gameday'] - pd.Timestamp(today.date())).abs()
        nearest_week = sched_valid.loc[sched_valid['days_from_today'].idxmin(), 'week'] if not sched_valid.empty else None
        if nearest_week is not None:
            this_week = sched[sched['week'] == nearest_week]
            print(f" -> No games in the current Tue-Mon window -- using nearest week ({int(nearest_week)}) instead.")

    week_number = int(this_week['week'].iloc[0]) if not this_week.empty else None

    skipped_games = []
    for _, g in this_week.iterrows():
        away_team, home_team = g['away_team'], g['home_team']
        # NOTE: nflverse's schedule data uses the SAME team codes as its own
        # play-by-play data (both come from nflverse) -- unlike the old ESPN
        # pull, there's no LAR/WSH-style code mismatch to patch around here.

        gameday = g.get('gameday')
        gametime = g.get('gametime')
        day_str = gameday.strftime('%a %m/%d') if pd.notna(gameday) else "--"
        day_time_str = f"{day_str} - {gametime}" if pd.notna(gametime) and gametime else day_str

        if away_team in df['team'].values and home_team in df['team'].values:
            matchups.append((away_team, home_team))
            schedule_info[(away_team, home_team)] = day_time_str
        else:
            skipped_games.append(f"{away_team} @ {home_team}")

    print(f" -> Successfully loaded {len(matchups)} real scheduled matchup(s) (week {week_number}).")
    if skipped_games:
        print(f" -> ⚠ Skipped {len(skipped_games)} game(s) with unrecognized team codes: {skipped_games}")
        print(f"    Known team codes in your data: {sorted(df['team'].unique())}")
except Exception as e:
    print(f" -> Schedule pipeline busy ({e}). Loading fallback loop.")

# Whether `matchups` reflects REAL scheduled games. Flows into every PDF
# this run builds -- see the "!! PLACEHOLDER PAIRING !!" header note below.
# Never silently ship arbitrary pairings without this showing.
matchups_are_real = bool(matchups)

if not matchups:
    print(" -> ⚠⚠⚠ WARNING: no real schedule data came back from nflverse this run. Falling back")
    print("    to an ARBITRARY alphabetical pairing of teams -- these are NOT real scheduled")
    print("    games. Every PDF built this run will say so in its header. Re-run once nflverse's")
    print("    schedule data is reachable, or check the error printed above.")
    teams_list = df['team'].unique()
    for i in range(0, len(teams_list)-1, 2):
        matchups.append((teams_list[i], teams_list[i+1]))

print("Step 3: Configuring ReportLab structural canvas styles...")


def new_doc(pdf_path):
    """Each output PDF (the combined bundle, and every individual matchup
    PDF) gets its own SimpleDocTemplate -- same page setup every time."""
    return SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)


styles = getSampleStyleSheet()

# Primary Matchup Header: Team abbreviations stay at 13pt
style_main_hdr = ParagraphStyle('MainHeader', fontName='Helvetica-Bold', fontSize=13, leading=15, textColor=colors.HexColor('#E2E8F0'), alignment=1)

# Flanking Record Headers
style_record_hdr = ParagraphStyle('RecordHeader', fontName='Helvetica-Bold', fontSize=8.6, leading=11, textColor=colors.HexColor('#E2E8F0'), alignment=1)

# REDESIGNED HEADER v2: switched to a serif "Roman" face (ReportLab's
# built-in Times family -- no font embedding needed) instead of Helvetica.
# Team name is now centered over its OWN column with the record stacked
# directly underneath it (one column per team, not a separate outer record
# flank) -- 3 columns total, using panel_widths itself so each team's header
# block sits directly above that same team's stat column in every full-width
# panel below (VOID RATING, TEAM RUNDOWN, etc. all use panel_widths too).
# Turquoise, deliberately, for two specific NEW features that mean something
# different from this sheet's general purple/orange cell-tiering system
# (defined further below): the STAT MISMATCH BREAKDOWN's
# biggest-real-gap-per-category cell, and a team's own name in the header
# when the real backtested model gives it a >=80% win probability. This is
# not a reversal of "turquoise is not the answer" -- that decision was
# specifically about the always-on per-cell outlier tier (settled on
# purple/orange instead). This is a new, separate, deliberate use of the
# color for "the model is confident about this,".
TURQUOISE_HIGHLIGHT = '#2DD4BF'

style_hdr_team_name = ParagraphStyle('HdrTeamName', fontName='Times-Bold', fontSize=24, leading=27, textColor=colors.HexColor('#E2E8F0'), alignment=1)
# Turquoise variant -- used only when the real backtested/hybrid model
# (see STAT-WEIGHTED WIN PREDICTION MODEL below) gives THIS team a >=80%
# real win probability this week.
style_hdr_team_name_hot = ParagraphStyle('HdrTeamNameHot', fontName='Times-Bold', fontSize=24, leading=27, textColor=colors.HexColor(TURQUOISE_HIGHLIGHT), alignment=1)
# BLACK-TEXT variant of style_hdr_team_name -- old man sheet only, once its
# team-name blocks lose their dark fill too (see
# build_matchup_header_elements's ink_saver param). The turquoise >=80%
# win-probability highlight (style_hdr_team_name_hot, just above) is
# untouched either way -- that's a real signal, not decorative fill, and
# turquoise text reads fine on white too.
style_hdr_team_name_dark = ParagraphStyle('HdrTeamNameDark', fontName='Times-Bold', fontSize=24, leading=27, textColor=colors.black, alignment=1)
style_hdr_week_big = ParagraphStyle('HdrWeekBig', fontName='Times-Bold', fontSize=12, leading=14, textColor=colors.HexColor('#E2E8F0'), alignment=1)
# BLACK-TEXT variant of the above, used ONLY on the old man sheet's middle
# WEEK/kickoff/vs/PROJ TOTAL box once its dark fill is removed there (see
# build_matchup_header_elements's ink_saver param). Never used on the full
# sheet/individual PDFs (that header keeps its original look, unchanged).
style_hdr_week_big_dark = ParagraphStyle('HdrWeekBigDark', fontName='Times-Bold', fontSize=12, leading=14, textColor=colors.black, alignment=1)
# QUALITY STAT CONSENSUS badge ("N/M quality stats"), under the record in
# the header. Rendered as an inline <font color=...> span inside the same
# Paragraph as the team name (see build_matchup_elements), not its own
# ParagraphStyle, since ReportLab Paragraphs don't support per-run styles --
# the color (plain grey, or turquoise at QUALITY_CONSENSUS_HOT_THRESHOLD+)
# is computed and applied inline there instead. See the
# NFL_QUALITY_STATS/QUALITY_CONSENSUS_HOT_THRESHOLD comment in the
# STAT-WEIGHTED WIN PREDICTION MODEL section below for where the 12-stat
# list and the threshold came from -- a real backtest, not a guess.
# Deliberately a SEPARATE signal from the >=80% win-probability team-name
# highlight above, not a replacement for it: that one is "the weighted model
# is confident," this one is "a large, independent group of real stats
# happen to agree" -- real testing showed those two don't always fire
# together, so both stay visible rather than merging into one signal.

# Panel Sub-Headers -- SHRUNK slightly to fit a half-width column
style_sub_hdr  = ParagraphStyle('SubHeader', fontName='Helvetica-Bold', fontSize=5.4, leading=7, textColor=colors.HexColor('#E2E8F0'), alignment=1)
# BLACK-TEXT variant of the above -- old man sheet section headers ONLY,
# once their dark fill is removed (see build_quick_scan_panel). Never
# used by build_stat_panel (the full sheet/individual PDFs keep their
# original dark header bars, unchanged).
style_sub_hdr_dark = ParagraphStyle('SubHeaderDark', fontName='Helvetica-Bold', fontSize=5.4, leading=7, textColor=colors.black, alignment=1)

# Center Section Labels -- SHRUNK to fit a narrower half-width label column
style_label    = ParagraphStyle('LabelCell', fontName='Helvetica-Bold', fontSize=5.2, leading=6.6, textColor=colors.white, alignment=1)
# BLACK-TEXT variant -- old man sheet only, once the grey label-column fill
# behind it is removed (see build_quick_scan_panel/build_breakdown_grid).
# build_stat_panel (the full sheet/individual PDFs) keeps the original
# white-on-grey label column, untouched.
style_label_dark = ParagraphStyle('LabelCellDark', fontName='Helvetica-Bold', fontSize=5.2, leading=6.6, textColor=colors.black, alignment=1)

# Stat Box Base Text Style -- SHRUNK to fit a narrower half-width value column
style_val_base = ParagraphStyle('ValBase', fontName='Helvetica-Bold', fontSize=5.6, leading=7, alignment=1)

# TEAM RUNDOWN panels only -- a bit bigger than the shared styles above so
# every row reads as roomy as a short one-line row (e.g. INTERCEPTIONS/
# GAME), not just the ones lucky enough not to wrap. Kept separate from
# style_val_base/style_label so this doesn't change any other panel.
style_rundown_val   = ParagraphStyle('RundownVal', fontName='Helvetica-Bold', fontSize=6.0, leading=7.2, alignment=1)
style_rundown_label = ParagraphStyle('RundownLabel', fontName='Helvetica-Bold', fontSize=5.5, leading=6.6, textColor=colors.white, alignment=1)
# Small team-abbreviation sub-header, centered directly over that team's own
# value column in the TEAM RUNDOWN OFFENSE/DEFENSE grids.
style_rundown_team_hdr = ParagraphStyle('RundownTeamHdr', fontName='Helvetica-Bold', fontSize=7, leading=8.5, textColor=colors.HexColor('#E2E8F0'), alignment=1)
# Panel TITLE ("TEAM RUNDOWN -- OFFENSE"/"-- DEFENSE") -- bigger than the
# team-abbreviation sub-header row underneath it (style_rundown_team_hdr,
# 7pt), so the title reads as the dominant line and DET/BUF reads as the
# secondary line under it.
style_rundown_title = ParagraphStyle('RundownTitle', fontName='Helvetica-Bold', fontSize=8.4, leading=10.2, textColor=colors.HexColor('#E2E8F0'), alignment=1)

# VOID RATING panel only -- deliberately bigger/bolder than every other
# build_stat_panel caller and paired with a much narrower VOID_WIDTHS (see
# below) to shrink the panel's overall footprint while making the actual
# numbers stand out more, not less. SHRUNK AGAIN slightly, right after the
# panel grew to 10 rows with EPA/PLAY (OFF) added) -- still bigger/bolder
# than every other panel's shared styles, just a bit less so than before.
style_void_val   = ParagraphStyle('VoidVal', fontName='Helvetica-Bold', fontSize=6.4, leading=7.6, alignment=1)
style_void_pop   = ParagraphStyle('VoidPop', fontName='Helvetica-Bold', fontSize=7.2, leading=8.6, alignment=1)
style_void_label = ParagraphStyle('VoidLabel', fontName='Helvetica-Bold', fontSize=5.8, leading=7.0, textColor=colors.white, alignment=1)

# NEW: "pops out" style for whichever side has the larger raw number in a
# paired row -- larger + bolder than the base style, combined with a
# thicker box border, to fake a raised/3D look ReportLab tables can't do
# with a true drop shadow. Scaled down to match the smaller half-width base.
style_val_pop  = ParagraphStyle('ValPop', fontName='Helvetica-Bold', fontSize=6.6, leading=8, alignment=1)

# Fantasy Watch panel styles
style_fw_name  = ParagraphStyle('FWName', fontName='Helvetica-Bold', fontSize=6.2, leading=7.5, textColor=colors.black)
style_fw_stat  = ParagraphStyle('FWStat', fontName='Helvetica', fontSize=5.8, leading=7, alignment=1)

# THE CAPTAIN's QB sub-header line (name, mobility tag, rush yd/gm, time to
# throw) -- sits centered directly over each team's own grid column, same
# idea as style_rundown_team_hdr but smaller since it carries a full line of
# text instead of just a team abbreviation.
style_captain_subhdr = ParagraphStyle('CaptainSubHdr', fontName='Helvetica-Bold', fontSize=5.0, leading=6.2, textColor=colors.HexColor('#E2E8F0'), alignment=1)

# Small italic footnote under a grid for a real, asked-about stat that has
# no free/licensed data source (e.g. DROP % -- see build_freebirds_panel) --
# says so plainly instead of leaving it a silent gap.
style_note = ParagraphStyle('Note', fontName='Helvetica-Oblique', fontSize=4.8, leading=6, alignment=1, textColor=colors.HexColor('#555555'))

# HOW TO READ THIS SHEET (glossary appendix) styles -- left-aligned, real
# reading-size prose (unlike the compact centered panel styles above),
# since this is a dedicated reference page meant to actually be read, not a
# tiny caption squeezed under a stat grid.
style_gloss_body = ParagraphStyle('GlossBody', fontName='Helvetica', fontSize=6.3, leading=8.0, alignment=0)


# ==============================================================================
# COLOR TIERING (redesigned: the old version colored a cell unless it was
# within 0.15 STANDARD DEVIATIONS of the league average -- an extremely
# tight band for most real NFL stats, so almost nothing came back white and
# nothing actually stood out. Measured as % deviation from the real league
# average. History: 10%/15% -> 15%/25% -> 20%/35% (elite tightened to
# top/bottom 5%) -> briefly tried 10%/35% -> corrected back to 20%/35% ->
# briefly tried a 4-tier 25/40/60 version with a "mild" pastel tier and
# turquoise for worst -- but turquoise was dropped: orange marks the bottom 3% and purple the
# top 3%, with white/black for 0-25%, green/red stats for 25-55%, and
# green/red cells above 55%. So this is now a clean 3-tier system again (no mild
# pastel tier), just at new numbers, with orange restored:
#   0-25% off average    -> plain white box, black text (near average)
#   25-55% off average   -> just the NUMBER gets colored, ordinary green/red
#   more than 55% off    -> the WHOLE BOX gets colored, ordinary green/red
#   elite (top 3% of the league, elite_thresholds above -- TIGHTENED from
#   top/bottom 5% to top/bottom 3% per this same ask; with only 32 real
#   teams, that's close to just the single best/worst team per stat)
#                         -> always a purple box no matter what the %
#                           threshold says -- an elite number is elite.
#   worst (bottom 3% of the league, worst_thresholds below)
#                         -> always an ORANGE box (back from turquoise),
#                           same logic in reverse.
# % of average breaks down for the handful of stats whose real league
# average sits near zero (net EPA/play, avg scoring margin -- both ~0 BY
# CONSTRUCTION, since one team's edge is another's deficit across a full
# league) -- dividing by a near-zero average blows an ordinary value up into
# a meaningless huge percentage. For those, this falls back to a standard-
# deviation-based measure instead (same real numbers, just an honest
# yardstick that doesn't divide by ~0) -- judged by std/|avg|: when the
# spread is large relative to the average itself, the average is too close
# to zero to divide by safely. STD_INNER_BAND/STD_OUTER_BAND are derived
# from the same real growth rate used for the last few widenings (15% ->
# 0.75, i.e. 0.05 std-devs per 1%, the last rate actually specified by you,
# back at the 15%/25% widening): 25% -> 1.25, 55% -> 2.75. Proportional
# judgment call, flagged as such -- say the word for different std-based
# numbers here specifically.
# ==============================================================================

COLOR_RED    = '#FF4D4D'
COLOR_GREEN  = '#2ECC71'
COLOR_PURPLE = '#9B59B6'
COLOR_ORANGE = '#F5A623'   # worst-3% tier (back from turquoise), same hex PRESSURE_TIERS already uses for its orange tier
COLOR_BLACK  = '#000000'

PCT_MODE_MAX_CV = 5.0     # std/|avg| above this -> avg too close to 0, use std fallback
PCT_INNER_BAND  = 25.0    # +/- this % of average -> no color at all
PCT_OUTER_BAND  = 55.0    # beyond this % of average -> whole box colored
STD_INNER_BAND  = 1.25    # std-dev fallback equivalents, for near-zero-average stats
STD_OUTER_BAND  = 2.75

def get_stat_tier(val, field_name):
    """Returns (box_fill_color_or_None, text_hex_color) for one real stat
    value. box_fill_color is a reportlab Color for the cell's BACKGROUND, or
    None to leave the cell's default white background alone (the "doesn't
    matter" and "worth a glance" tiers -- only the outlier tier fills the
    whole box)."""
    avg = league_avgs.get(field_name, 0)
    std = league_stds.get(field_name, 0.1) or 0.1
    elite = elite_thresholds.get(field_name, 0)
    worst = worst_thresholds.get(field_name, 0)
    is_lower_better = field_name in lower_is_better_fields

    is_elite = (val <= elite) if is_lower_better else (val >= elite)
    if is_elite:
        return colors.HexColor(COLOR_PURPLE), COLOR_BLACK

    is_worst = (val >= worst) if is_lower_better else (val <= worst)
    if is_worst:
        return colors.HexColor(COLOR_ORANGE), COLOR_BLACK

    if abs(avg) > 1e-9 and (std / abs(avg)) <= PCT_MODE_MAX_CV:
        dev = (val - avg) / abs(avg) * 100
        inner, outer = PCT_INNER_BAND, PCT_OUTER_BAND
    else:
        dev = (val - avg) / std
        inner, outer = STD_INNER_BAND, STD_OUTER_BAND

    if is_lower_better:
        dev = -dev   # positive dev always means "better than average" from here on

    abs_dev = abs(dev)
    if abs_dev <= inner:
        return None, COLOR_BLACK
    color_hex = COLOR_GREEN if dev > 0 else COLOR_RED
    if abs_dev <= outer:
        return None, color_hex                        # moderate -- number only
    return colors.HexColor(color_hex), COLOR_BLACK      # outlier -- whole box

def build_centered_table(data_matrix, column_widths, bg_colors=None, border_color='#000000'):
    t = Table(data_matrix, colWidths=column_widths)
    styles_list = [
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.75, colors.HexColor(border_color)),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),     # SHRUNK from 2.0 -- tighter rows now that
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),  # each matchup fits more paired stats
    ]
    if bg_colors:
        for r_idx, row in enumerate(bg_colors):
            for c_idx, color in enumerate(row):
                if color:
                    styles_list.append(('BACKGROUND', (c_idx, r_idx), (c_idx, r_idx), color))
                    
    t.setStyle(TableStyle(styles_list))
    return t

# ==============================================================================
# PART 1 BOUNDARY LINE END
# ==============================================================================
# ==============================================================================
# SECTION 2: MATCHUP PANEL DESIGN & PDF COMPILATION
# PART 2 BOUNDARY LINE START
#
# REDESIGNED: instead of one wide "offense" panel (both teams' own offense
# side by side) followed by one wide "defense" panel (both teams' own
# defense-allowed side by side), each panel now pairs ONE team's offense
# directly against the OPPONENT's defense -- e.g. Team A's points/game next
# to Team B's points allowed/game, in the same row. This is the actual
# football matchup question ("how does this offense fare against that
# defense") rather than two disconnected same-stat comparisons.
# ==============================================================================

print("Step 4: Compiling panel grids into the final matchup PDF...")

center_grey = colors.HexColor('#4A4A4A')
obsidian_black = colors.HexColor('#1A1A1A')
panel_widths = [140, 260, 140]           # full-width panels (header, overall ratings)
side_widths = [56, 92, 56]                # SHRUNK further -- grey label area especially reduced
SIDE_PANEL_TOTAL = sum(side_widths)        # 204 -- narrower than before, on purpose
# TEAM RUNDOWN side-by-side panels specifically (longer labels than the
# other side-by-side panels above, e.g. "RUSH YDS ALLOWED / GM (RUSH PLAY %
# FACED)") -- uses the FULL 270pt half-width instead of the narrower 204,
# so labels wrap less and every row reads as roomy as a short one-line row
# like INTERCEPTIONS/GAME.
RUNDOWN_SIDE_WIDTHS = [56, 148, 56]        # sums to 260 -- 10pt narrower than OUTER_HALF_WIDTH
                                            # (270) on purpose, so CENTER alignment in the outer
                                            # side-by-side table leaves a real gap between the
                                            # offense and defense panels instead of them touching
                                            # -- same ~10pt gap PREVIOUS SCHEDULE's two mini-tables
                                            # already have (their 260pt-wide tables sit inside the
                                            # same 270pt OUTER_HALF_WIDTH column too).
OUTER_HALF_WIDTH = 270                     # kept at exactly half the 540pt page, so the now-
                                            # narrower panel has room to center within it

# VOID RATING grid only -- the "VOID RATING" title bar stays full-width
# (panel_widths, unchanged) but the stat-rows grid underneath it shrinks to
# this much narrower size and centers itself under that header ("shrink the
# middle down... shrink the entire section, its ok if its smaller than the
# other columns, I want it to stand out"). Middle (label) column cut
# hardest since that's what felt oversized; the two value columns stay
# roomy enough for the bigger style_void_val/style_void_pop fonts used here
# (see build_stat_panel's val_style/pop_style params). SHRUNK AGAIN
# slightly, right after the panel grew to 10 rows with EPA/PLAY (OFF)
# added) -- same proportions, just ~8% narrower overall, paired with the
# smaller style_void_val/style_void_pop/style_void_label fonts and tighter
# row_padding below so a now-taller panel doesn't eat more page-1 room.
VOID_GRID_WIDTHS = [108, 64, 108]          # sums to 280 -- was [120, 70, 120] (310) before this ask

# TOP HEADER only -- shrink the grey WEEK/kickoff/vs middle column down to
# as little room as its real text actually needs (measured with
# pdfmetrics.stringWidth: "WEEK 2" at 12pt Times-Bold and the kickoff
# string at 6.8pt Times-Roman both come in under 47pt, so 70pt covers that
# plus normal cell padding with room to spare) and hand the freed width to
# the two team-name columns on either side to shrink the grey middle "as
# small as you can get it and leave the rest black." Sums to the same 540pt
# as panel_widths -- still a full-bleed header, just weighted differently.
HEADER_WIDTHS = [235, 70, 235]

# OFFENSE vs DEFENSE matchup panel -- moved to sit right after NET EPA PER
# MATCHUP (instead of much further down the sheet) and trimmed down to the 8
# rows that actually matter. Net EPA is a single
# crossed number (this offense's EPA vs THIS opponent's defensive EPA) with
# nothing backing it up next to it -- this panel is that same
# crossed-matchup idea broken out row by row, so it now sits right where it
# explains the number above it instead of requiring a flip several panels
# later to find supporting detail. Previously 21 rows; the other 13
# (POINTS/YARDS/DRIVE-level game totals, SUCCESS RATE, TFL, TURNOVERS, raw
# attempt counts, COMPLETION %, SACK %, INTERCEPTIONS, FUMBLES, RUN STUFF
# RATE) aren't gone from the sheet -- they're real, already-computed columns
# still sitting in nfl_team_matrix.csv, just not repeated here since most of
# them are already shown once in TEAM RUNDOWN OFFENSE/DEFENSE above (same
# real numbers, same-team framing there vs. crossed-matchup framing here).
# (left_key, right_key, format, suffix, multiplier, label)
MATCHUP_PAIRS = [
    ('third_down_pct_offense', 'third_down_pct_defense', "{:.1f}", "%", 1, "3RD DOWN CONVERSION %"),
    ('fourth_down_conversion_pct', 'fourth_down_conversion_pct_allowed', "{:.0f}", "%", 1, "4TH DOWN CONVERSION %"),
    ('redzone_efficiency', 'redzone_efficiency_allowed', "{:.1f}", "%", 1, "RED ZONE TD %"),
    ('avg_3rd_down_distance_offense', 'avg_3rd_down_distance_defense', "{:.1f}", "", 1, "AVG TO-GO DISTANCE (3RD DOWN)"),
    # Real rush-only / pass-only yards-per-play splits (off_rushing_yards_per_play
    # / def_rushing_yards_per_play / off_yards_per_pass / def_yards_per_pass) --
    # already computed in nfl_data_pull.py, just not used anywhere on the sheet
    # until now. Distinct from YARDS/PLAY elsewhere, which blends rush+pass.
    ('off_rushing_yards_per_play', 'def_rushing_yards_per_play', "{:.1f}", "", 1, "RUSH YARDS / PLAY"),
    ('off_yards_per_pass', 'def_yards_per_pass', "{:.1f}", "", 1, "PASS YARDS / PLAY"),
    ('off_efficiency_epa', 'def_efficiency_epa', "{:.1f}", "%", 100, "NET EFFICIENCY (EPA/PLAY)"),
    ('sacks_allowed', 'sacks_made', "{:.0f}", "", 1, "SACKS"),
]

# NOTE: the standalone "PASSER RATING" panel (RESIDUAL_OVERALL_PAIRS,
# open_source_passer_rating) is GONE -- that real number is fully
# duplicated in THE CAPTAIN's grid now (see CAPTAIN_PAIRS above), so a
# separate panel for it was pure repetition once THE CAPTAIN existed.

# NOTE: the standalone "ADVANCED METRICS (NGS + FREE PROXIES)" panel
# (ADVANCED_PAIRS) is GONE. What happened to its 6 real fields, so
# nothing quietly vanished:
#   - cpoe -- duplicated in THE CAPTAIN's grid, safe to drop here.
#   - ryoe_per_attempt -- MOVED (not dropped) to THE DAWGS grid, since it's a real rushing efficiency stat.
#   - target_separation (same-key, team-vs-team) -- superseded by the more
#     informative CROSSED version (target_separation vs. target_separation_
#     allowed) already sitting in THE FREEBIRDS grid, real data either way.
#   - pressure_rate_allowed_pct / pressure_rate_forced_pct (same-key) --
#     likewise superseded by the crossed version already in THE FREEBIRDS
#     grid (pressure_rate_allowed_pct vs. pressure_rate_forced_pct).
#   - team_yac_per_reception -- the one real field here with NO surviving
#     home anywhere on the sheet. This is a distinct real source from the
#     yac_per_reception_pbp/yac_allowed_per_reception_pbp pair still shown
#     in THE FREEBIRDS/SKILL BATTLE (see the comment above
#     SKILL_BATTLE_PAIRS for why those two YAC sources are kept
#     deliberately separate) -- still pulled every run in nfl_team_matrix.csv,
#     just not displayed anywhere right now. Flagging honestly rather than
#     silently dropping it; say the word if you want it back somewhere.

# TRENCH BATTLE panel: offense's own O-line/run-game number paired directly
# against the OPPOSING defense's mirrored number, same shape as
# MATCHUP_PAIRS above. Both sides of both rows read "higher is better", so
# no entries need adding to lower_is_better_fields -- get_stat_tier()'s
# default (higher-is-better) is already correct, and build_stat_panel's
# existing "bigger raw number on this row gets bolded/pops" treatment does
# the "who's winning this matchup" comparison for free.
#   PBE (Pass Blocking Efficiency, real, no hurries -- see nfl_data_pull.py)
#   vs. PRP (Pass Rush Productivity, same real weighting, mirrored for D).
#   Run Success Rate (offense, EPA>0) vs. Run Stop Rate (defense, share of
#   rush attempts faced that did NOT succeed).
TRENCH_BATTLE_PAIRS = [
    # comparable=True (default): both sides are real rates on the same
    # 0-100% "rush outcome" scale, so the bigger-number pop is a legitimate
    # at-a-glance read of which side is winning the run game. PBE vs PRP
    # (pass block vs pass rush) is NOT on this same scale -- see
    # build_pbe_prp_panel() below, which nets them properly via z-score
    # instead of raw subtraction/magnitude comparison.
    ('off_run_success_rate', 'def_run_stop_rate', "{:.1f}", "%", 1, "RUN SUCCESS RATE vs RUN STOP RATE"),
]

# SKILL BATTLE panel: receivers-as-a-unit vs. the opposing DBs, same
# offense-vs-defense-allowed shape as MATCHUP_PAIRS/TRENCH_BATTLE_PAIRS
# above. All three rows are real, same-scale on both sides (comparable=True
# is correct, unmodified default). See nfl_data_pull.py for how each is
# sourced -- CATCH % is nflverse's own targets/receptions (re-attributed to
# the covering defense via opponent_team for the "allowed" side); YAC/REC
# is computed fresh from real pbp on BOTH sides on purpose (not the NGS-
# sourced number elsewhere on this sheet) so offense and defense are
# measured the same way; SEPARATION is real NGS data, with "allowed"
# re-attributed to the covering defense via the real weekly schedule.
# Deliberately does NOT include DROP % -- see README.txt "SKILL BATTLE"
# section for why that one isn't isolable from any free data source.
SKILL_BATTLE_PAIRS = [
    ('catch_pct', 'catch_pct_allowed', "{:.1f}", "%", 1, "CATCH % vs CATCH % ALLOWED"),
    ('yac_per_reception_pbp', 'yac_allowed_per_reception_pbp', "{:.1f}", "", 1, "YAC/REC vs YAC ALLOWED/REC"),
    ('target_separation', 'target_separation_allowed', "{:.1f}", "", 1, "SEPARATION vs SEPARATION ALLOWED"),
]

# SECONDARY LOCK INDEX: a defense-vs-defense comparison (not offense-vs-
# defense -- there's no offensive mirror for a coverage-tightness index),
# same same-stat-both-sides shape TENDENCY_PAIRS already uses above. 100
# minus real yards allowed per real target faced -- see nfl_data_pull.py.
SECONDARY_LOCK_PAIR = [
    ('secondary_lock_index', 'secondary_lock_index', "{:.1f}", "", 1, "SECONDARY LOCK INDEX"),
]

# ==============================================================================
# THE FIELD GENERAL / THE POUND -- two more "precision snippet" panels, same
# idea as THE DIRTY BIRD (a curated top-3, not the full grid): real stats,
# deliberately picked for the biggest at-a-glance "who wins this specific
# matchup" read, not new data. (special teams' version of this didn't
# survive -- see the NOTE below THE POUND for why). Like THE DIRTY BIRD
# reuses 2 of FREEBIRDS' 5 rows, these reuse fields already in THE
# CAPTAIN/THE DAWGS grids -- not new numbers, just a curated highlight,
# which you already said you're fine with (it's the whole point of a
# "snippet" -- you approved the overlap for WR, applying the same call
# here). All three (FIELD GENERAL/POUND/DIRTY BIRD) are grouped together at
# the end of the page -- see the comment above skill_away/skill_home further
# down.
#
# THE FIELD GENERAL (QB) -- accuracy, protection, ball security: the three
# things that decide most passing matchups, each a real offense-vs-opposing-
# defense crossed pair (same 3 of the 4 crossed CAPTAIN_PAIRS rows -- YARDS/
# COMPLETION left out as the one row of the four that's more of a supporting
# big-play indicator than a standalone "who wins" signal).
FIELD_GENERAL_PAIRS = [
    ('completion_pct', 'completion_pct_allowed', "{:.1f}", "%", 1, "COMPLETION % vs ALLOWED"),
    ('sack_pct_offense', 'sack_pct_defense', "{:.1f}", "%", 1, "SACK % (PROTECTION vs PASS RUSH)"),
    ('interceptions_thrown_per_game', 'interceptions_gained_per_game', "{:.1f}", "", 1, "INTs THROWN vs GAINED / GM"),
]

# THE POUND (RB) -- production, failure rate, efficiency: same idea for the
# run game. Real offense-vs-opposing-defense crossed pairs (3 of DAWGS_PAIRS'
# 6 crossed rows -- FUMBLES and TFL left out as more turnover-luck/box-score
# noise than a repeatable "who wins the run game" signal).
THE_POUND_PAIRS = [
    ('rushing_yards_per_game', 'rushing_yards_allowed_per_game', "{:.1f}", "", 1, "RUSHING YARDS/GM vs ALLOWED"),
    ('stuff_rate_offense', 'stuff_rate_defense', "{:.1f}", "%", 1, "STUFF RATE (OFF) vs (DEF)"),
    ('off_run_success_rate', 'def_run_stop_rate', "{:.1f}", "%", 1, "RUN SUCCESS RATE vs RUN STOP RATE"),
]

# NOTE: a "SPECIAL TEAMS EDGE" precision snippet briefly lived here (real
# FG/punt block rows + a PUNT RETURN AVERAGE same-key row) -- REMOVED, since
# it turned out to be pure duplication: the block rows are the same real
# numbers as SPECIAL TEAMS BLOCKS, and punt return average was already in
# KICKING & RETURNS. Special teams only has 2 real
# offense-vs-opposing-defense crossed fields (FG/punt blocks, BLOCK_PAIRS
# below) -- no real third crossed stat exists to round out a 3-stat snippet
# without either repeating a number already on the page or faking one, so
# it's just not here as a separate panel; both real numbers are still shown
# exactly once (SPECIAL TEAMS BLOCKS / KICKING & RETURNS).

# GAME PLAN's old standalone panel (PASS PLAY %, RUSH PLAY %, NET
# EFFICIENCY) is GONE -- all 3 fields moved into VOID_PAIRS below,
# alongside LEAGUE RANK/TOTAL VOID and the new NET EPA/SUCCESS RATE rows,
# instead of their own separate panel. Still real, same formulas/columns
# -- just relocated, see VOID_PAIRS.
KICKING_PAIRS = [
    ('punts_per_game', 'punts_per_game', "{:.1f}", "", 1, "PUNTS / GAME"),
    ('avg_punt_distance', 'avg_punt_distance', "{:.1f}", "", 1, "AVG PUNT DISTANCE"),
    ('punt_return_avg', 'punt_return_avg', "{:.1f}", "", 1, "PUNT RETURN AVG"),
    ('avg_fg_distance', 'avg_fg_distance', "{:.1f}", "", 1, "AVG FG DISTANCE (ATTEMPTED)"),
    ('special_teams_tds_per_game', 'special_teams_tds_per_game', "{:.2f}", "", 1, "SPECIAL TEAMS TDs / GAME"),
]
# Block pairs DO have a real offense-vs-opposing-defense shape (this team's
# own kicking-unit vulnerability vs. the OTHER team's block-forcing
# ability) -- same MATCHUP_PAIRS convention as everything else, unlike the
# team-vs-team pairs above.
BLOCK_PAIRS = [
    ('fg_blocked_per_game', 'fg_blocks_forced_per_game', "{:.2f}", "", 1, "FG BLOCKED vs FG BLOCKS FORCED"),
    ('punts_blocked_per_game', 'punt_blocks_forced_per_game', "{:.2f}", "", 1, "PUNTS BLOCKED vs PUNT BLOCKS FORCED"),
]

# NOTE: the standalone "PERSONNEL & FORMATION TENDENCIES" panel
# (SCHEME_TENDENCY_PAIRS -- real nflverse participation data, personnel
# groupings + shotgun/under center + defensive sub-package rates, see
# nfl_data_pull.py PART 2D) is GONE. Unlike ADVANCED_PAIRS above, none of
# these 8 fields have an equivalent shown anywhere else on the sheet --
# they're just not displayed right now. Still real, still pulled every
# run (nfl_data_pull.py's participation-data step is untouched), so
# bringing this panel back later is just re-adding the build_stat_panel
# call, not re-pulling any data.

# VOID -- Value Over Interactive Average (see nfl_data_pull.py Part 7 for the
# full writeup). An opponent-adjusted, situational-success-based efficiency
# rating -- our own formula, in the same spirit as DVOA but independently
# built from free data. Positive = better than league average, for both sides
# of the ball. This is a DESCRIPTIVE rating of how a team has actually
# performed, not a prediction of an upcoming game -- unlike the model
# spread/total/win-probability panel (deliberately left out of this sheet for
# now, see README.txt), a rating of what already happened doesn't carry the
# same risk of being publicly, embarrassingly wrong. NEW ORDER: LEAGUE RANK
# leads, then everything grouped "under league rank" --
# TOTAL VOID, NET EPA, NET EFFICIENCY, SUCCESS RATE, PASS PLAY %, RUSH PLAY %
# -- with OFF VOID/DEF VOID (not on that list) moved to the bottom rather
# than removed. Same real numbers as before, just consolidated into one panel
# instead of three (this used to be split across VOID RATING, the separate
# GAME PLAN panel, and the separate NET EPA PER MATCHUP panel -- both of
# those are gone now, their real fields all live here).
#
# LEAGUE RANK prints as an ordinal ("1st"/"2nd"/"3rd"/...) instead of a
# "X/N" fraction now. The "/N" version (still the real count of teams that
# currently pass the gp>=2 VOID gate, NOT hardcoded /32 -- see
# nfl_data_pull.py Step 9 / KNOWN MATH LIMITATION note) is what caught the
# earlier red-instead-of-purple confusion, so that real gate and its
# early-season small-N caveat are still fully documented in README.txt even
# though the sheet itself now shows just the ordinal, no denominator.
def _ordinal(n):
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', 11-13 -> 'Nth', etc.
    Only ever called on a real, already-non-missing rank value -- the
    missing case ('--') is handled by build_stat_panel before this runs."""
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

# NOTE on 'net_epa_matchup': unlike every other field here, NET EPA isn't a
# plain df column -- it's a real cross-team computation (this team's off
# EPA/play x1.5 minus the SPECIFIC opponent's def EPA/play allowed, see the
# old build_net_epa_panel formula, now folded in here) that needs BOTH
# teams' rows to produce EACH side's number. It's computed fresh right
# before this panel is built (see build_matchup_elements below) and stashed
# under this key on temporary copies of r_a/r_b made just for this one
# panel call -- not a real column in nfl_team_matrix.csv the way
# off_void/off_success_rate/etc. are. Labels below are shortened from their
# original panel's wording (e.g. plain "NET EPA" not "NET EPA ((OFF EPA
# x1.5) - DEF EPA)") to fit VOID_GRID_WIDTHS' narrow 70pt label column --
# the full formula is still in the note under this panel and in the
# glossary, nothing lost, just not spelled out in the label itself here.
#
# NOTE on 'off_efficiency_epa' (EPA/PLAY (OFF) row): this is real quality stat #2 (0.610, second only to Success
# Rate's 0.634) that was ALREADY on the sheet, but only ever combined into
# NET EFFICIENCY (own off epa minus own def epa allowed) and NET EPA
# (MATCHUP) (cross-team combo) -- the raw, unmixed number itself was never
# shown. Unlike net_epa_matchup above, this one IS a real column in
# nfl_team_matrix.csv (same one league_avgs/elite_thresholds/etc. were
# already correctly built from in the real for-loop up top), so it gets
# real, already-correct tier coloring with no extra threshold work needed
# -- this isn't the same bug as the NET EPA purple-coloring one you caught.
VOID_PAIRS = [
    ('total_void_rank', 'total_void_rank', _ordinal, "", 1, "LEAGUE RANK"),
    ('total_void', 'total_void', "{:.3f}", "", 1, "TOTAL VOID"),
    ('net_epa_matchup', 'net_epa_matchup', "{:.3f}", "", 1, "NET EPA (MATCHUP)"),
    ('off_efficiency_epa', 'off_efficiency_epa', "{:.2f}", "", 1, "EPA/PLAY (OFF)"),
    ('net_efficiency_epa', 'net_efficiency_epa', "{:.2f}", "", 1, "NET EFFICIENCY"),
    ('off_success_rate', 'off_success_rate', "{:.1f}", "%", 1, "SUCCESS RATE (OFF)"),
    ('net_ppd', 'net_ppd', "{:.2f}", "", 1, "NET PPD"),
    ('pass_play_pct_offense', 'pass_play_pct_offense', "{:.1f}", "%", 1, "PASS PLAY %"),
    ('rush_play_pct_offense', 'rush_play_pct_offense', "{:.1f}", "%", 1, "RUSH PLAY %"),
    ('off_void', 'off_void', "{:.3f}", "", 1, "OFF VOID"),
    ('def_void', 'def_void', "{:.3f}", "", 1, "DEF VOID"),
]

# NET VOID -- one extra row, appended directly under the VOID RATING grid.
# This is a straight TOTAL VOID gap between the two teams (left minus
# right), shown ONLY on whichever side has the better (higher) TOTAL VOID
# -- the other side is left blank, not zero/duplicated -- and colored red
# when that gap size has actually been the LOSING band so far, black
# otherwise.
#
# Threshold source, honestly: the one clean pre-game check run so far
# (Week 3 2026, n=16 decided games, a real pre-Week-3 VOID snapshot checked
# against real final results). The higher-VOID
# team went 7-9 (43.8%), WORSE than a coin flip. All 9 misses had a gap
# under 0.15 (range 0.047-0.145, avg 0.098); the 7 hits ranged wider
# (0.012-0.21, avg 0.137) but still overlapped that same low band some.
# So NET_VOID_RED_MAX marks the gap range that's been unreliable in the one
# week checked so far -- NOT a proven long-run edge off n=16. Revisit this
# number once nfl_model_scorecard.py has more weeks of real results banked
# -- say the word to tighten/loosen it.
NET_VOID_RED_MAX = 0.15

# ------------------------------------------------------------------------------
# TEAL FLAGS. A stat row gets a teal fill on the BETTER team's cell only when
# the gap between the two teams is big enough to be in the band that has
# actually been winning:
#   TOTAL VOID           gap >= TEAL_VOID_MIN (0.15)  -- wk3-4 2026: 8-2 at 0.147+,
#                        vs 6-15 for smaller gaps (31 games)
#   SUCCESS RATE         NET success rate (off SR minus def SR allowed) gap >=
#                        TEAL_NET_SR_MIN (10.7 pts) -- top third of gaps went 9-1,
#                        the rest 8-21 (31 games)
#   PPD                  NET points/drive (off minus allowed) gap >= TEAL_NET_PPD_MIN
#                        (0.20). Week 4 only (n=15): better net PPD went 13-2 but
#                        gap size did NOT separate hits from misses, so this one is
#                        a loose line, not a proven cutoff.
# All three are tiny early-season samples -- retune here once more weeks are
# banked. Teal fill REPLACES the purple/orange/green tint on that one cell.
# ------------------------------------------------------------------------------
TEAL_VOID_MIN = 0.15
TEAL_NET_SR_MIN = 10.7
TEAL_NET_PPD_MIN = 0.20
COLOR_TEAL_FILL = '#6FD9CE'
_TEAL_CTX = {'a': None, 'b': None}   # set per game by build_delta_scan_elements

def _teal_num(row, key):
    try:
        v = row.get(key) if hasattr(row, 'get') else row[key]
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

def teal_side(label, row_a, row_b):
    """'L' / 'R' / None -- which side of this stat row (if either) gets the
    teal fill. row_a is the LEFT team's row, row_b the RIGHT team's."""
    if row_a is None or row_b is None:
        return None
    if label == "TOTAL VOID":
        a, b = _teal_num(row_a, 'total_void'), _teal_num(row_b, 'total_void')
        gate = TEAL_VOID_MIN
    elif label == "SUCCESS RATE (OFF)":
        ao, ad = _teal_num(row_a, 'off_success_rate'), _teal_num(row_a, 'def_success_rate')
        bo, bd = _teal_num(row_b, 'off_success_rate'), _teal_num(row_b, 'def_success_rate')
        if None in (ao, ad, bo, bd):
            return None
        a, b = ao - ad, bo - bd
        gate = TEAL_NET_SR_MIN
    elif label == "NET PPD":
        ao, ad = _teal_num(row_a, 'points_per_drive'), _teal_num(row_a, 'points_per_drive_allowed')
        bo, bd = _teal_num(row_b, 'points_per_drive'), _teal_num(row_b, 'points_per_drive_allowed')
        if None in (ao, ad, bo, bd):
            return None
        a, b = ao - ad, bo - bd
        gate = TEAL_NET_PPD_MIN
    else:
        return None
    if a is None or b is None:
        return None
    if abs(a - b) < gate:
        return None
    side = 'L' if a > b else 'R'
    # The number printed on the sheet for these rows is the OFFENSE-only stat
    # (PPD / off SR), but the gate above uses the NET stat. When the two
    # disagree the teal used to land on the cell across from the better printed
    # number -- confusing. Only light the teal when the printed number's better
    # side is the same side as the net edge.
    shown_key = {"SUCCESS RATE (OFF)": 'off_success_rate'}.get(label)
    if shown_key:
        sa, sb = _teal_num(row_a, shown_key), _teal_num(row_b, shown_key)
        if sa is not None and sb is not None and sa != sb:
            if ('L' if sa > sb else 'R') != side:
                return None
    return side

def build_net_void_row(r_a, r_b, t_a, t_b, val_style=None, label_style=None, label_bg=True):
    """Single extra row under the VOID RATING grid -- same VOID_GRID_WIDTHS
    so it reads as part of that panel, not a separate one. r_a/r_b are the
    same r_a_void/r_b_void rows already built for VOID RATING (total_void
    is untouched by that copy, still the real column).
    val_style/label_style/label_bg let a caller match its OWN sheet's
    styling instead of the full sheet's (e.g. the quick scan/"old man"
    sheet's plain-black-on-white labels, no grey fill -- see its call in
    build_quick_scan_elements) without hardcoding one look here."""
    val_style = val_style or style_void_pop
    label_style = label_style or style_void_label
    raw_a = r_a.get('total_void') if hasattr(r_a, 'get') else r_a['total_void']
    raw_b = r_b.get('total_void') if hasattr(r_b, 'get') else r_b['total_void']
    a_missing = pd.isna(raw_a)
    b_missing = pd.isna(raw_b)

    teal_l = teal_r = None
    if a_missing or b_missing:
        # Same gate as TOTAL VOID itself (needs 2+ games played) -- if
        # either side is gated, there's no real gap to show yet.
        disp_left, disp_right = "--", "--"
    else:
        net = raw_a - raw_b
        gap = abs(net)
        color = COLOR_RED if gap < NET_VOID_RED_MAX else COLOR_BLACK
        text = f'<font color="{color}">{net:+.3f}</font>'
        _tf = colors.HexColor(COLOR_TEAL_FILL) if gap >= TEAL_VOID_MIN else None
        if net > 0:
            disp_left, disp_right = text, ""
            teal_l = _tf
        elif net < 0:
            disp_left, disp_right = "", text
            teal_r = _tf
        else:
            # True tie (net == 0) -- nobody's "better," show on neither side
            # rather than picking one arbitrarily.
            disp_left, disp_right = "", ""

    matrix = [[
        Paragraph(disp_left, val_style),
        Paragraph("NET VOID", label_style),
        Paragraph(disp_right, val_style),
    ]]
    label_bg_color = center_grey if label_bg else None
    t = build_centered_table(matrix, VOID_GRID_WIDTHS, bg_colors=[[teal_l, label_bg_color, teal_r]], border_color='#000000')
    t.setStyle(TableStyle([
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    return t

# ==============================================================================
# TEAM RUNDOWN: OFFENSE / DEFENSE
# Each team's own numbers side by side (same shape as
# OVERALL_PAIRS/TENDENCY_PAIRS: team-vs-team, not offense-vs-defense-allowed),
# now organized into one real, un-fragmented offense list and defense list
# instead of scattered across several small panels. A handful of rows pack TWO
# related real stats into one cell (e.g. rush yards/game next to rush play %
# on the same line) -- see build_rundown_panel below for how that's rendered;
# each of the two numbers is still its own real column, just shown together
# since they answer the same football question (how much they ran it, and how
# often they chose to).
#
# Row tuple shapes:
#   5-tuple  (key, fmt, suffix, mult, label)                     -- one stat
#   9-tuple  (key1, fmt1, suf1, mult1, key2, fmt2, suf2, mult2, label) -- combo
# ==============================================================================

TEAM_RUNDOWN_OFFENSE_ROWS = [
    ('points_per_game', "{:.1f}", "", 1, "POINTS / GAME"),
    ('avg_scoring_margin', "{:+.1f}", "", 1, "AVG SCORING MARGIN"),
    ('total_yards_per_game', "{:.1f}", "", 1, "YARDS / GAME"),
    ('yards_per_drive', "{:.1f}", "", 1, "YARDS / DRIVE"),
    ('off_yards_per_play', "{:.2f}", "", 1, "YARDS / PLAY"),
    ('rushing_yards_per_game', "{:.1f}", "", 1, 'rush_play_pct_offense', "{:.0f}", "%", 1, "RUSH YARDS / GM  (RUSH PLAY %)"),
    ('passing_yards_per_game', "{:.1f}", "", 1, 'pass_play_pct_offense', "{:.0f}", "%", 1, "PASS YARDS / GM  (PASS PLAY %)"),
    ('plays_per_game', "{:.1f}", "", 1, "PLAYS / GAME"),
    ('time_of_possession_minutes_per_game', "{:.1f}", "", 1, "TIME OF POSSESSION (MIN)"),
    ('third_down_pct_offense', "{:.1f}", "%", 1, "3RD DOWN CONVERSION %"),
    ('fourth_down_conversion_pct', "{:.0f}", "%", 1, "4TH DOWN CONVERSION %"),
    ('avg_3rd_down_distance_offense', "{:.1f}", "", 1, "AVG TO-GO DISTANCE (3RD DOWN)"),
    ('redzone_efficiency', "{:.1f}", "%", 1, "RED ZONE SUCCESS %"),
    ('fg_pct', "{:.1f}", "%", 1, "FIELD GOAL %"),
    ('net_efficiency_epa', "{:.2f}", "", 1, "NET EFFICIENCY (EPA/PLAY)"),
    ('penalty_yards_per_game', "{:.1f}", "", 1, "PENALTY YARDS / GAME"),
    ('interceptions_thrown_per_game', "{:.1f}", "", 1, "INTERCEPTIONS THROWN / GAME"),
]

# Defense mirror. Where a real, honestly-computed defensive counterpart
# exists it's used (yards allowed, rush/pass play % FACED, plays FACED/game,
# 4th down % ALLOWED -- the last two built specifically for this panel, see
# nfl_data_pull.py); AVG SCORING MARGIN, FG %, and NET EFFICIENCY are single
# team-level numbers with no separate offense/defense version (already shown
# once, above) so they're deliberately not repeated here. Sacks, TFL, stuff
# rate, and havoc rate are defense-only counting stats with no offense-side
# row above -- added. Interceptions appear on BOTH lists now (thrown,
# offense side / gained, defense side) -- same real number either way, just
# each team's own half of the same turnover.
TEAM_RUNDOWN_DEFENSE_ROWS = [
    ('points_allowed_per_game', "{:.1f}", "", 1, "POINTS ALLOWED / GAME"),
    ('total_yards_per_game_allowed', "{:.1f}", "", 1, "YARDS ALLOWED / GAME"),
    ('yards_per_drive_allowed', "{:.1f}", "", 1, "YARDS / DRIVE ALLOWED"),
    ('def_yards_per_play', "{:.2f}", "", 1, "YARDS / PLAY ALLOWED"),
    ('rushing_yards_allowed_per_game', "{:.1f}", "", 1, 'rush_play_pct_defense', "{:.0f}", "%", 1, "RUSH YDS ALLOWED / GM  (RUSH PLAY % FACED)"),
    ('passing_yards_allowed_per_game', "{:.1f}", "", 1, 'pass_play_pct_defense', "{:.0f}", "%", 1, "PASS YDS ALLOWED / GM  (PASS PLAY % FACED)"),
    ('def_plays_per_game', "{:.1f}", "", 1, "PLAYS FACED / GAME"),
    ('third_down_pct_defense', "{:.1f}", "%", 1, "3RD DOWN DEFENSE %"),
    ('fourth_down_conversion_pct_allowed', "{:.0f}", "%", 1, "4TH DOWN CONVERSION % ALLOWED"),
    ('avg_3rd_down_distance_defense', "{:.1f}", "", 1, "AVG TO-GO DISTANCE FACED (3RD DOWN)"),
    ('redzone_efficiency_allowed', "{:.1f}", "%", 1, "RED ZONE TD % ALLOWED"),
    ('sacks_made', "{:.0f}", "", 1, "SACKS"),
    ('tfl_made', "{:.0f}", "", 1, "TFL"),
    ('stuff_rate_defense', "{:.1f}", "%", 1, "RUN STUFF RATE"),
    ('interceptions_gained_per_game', "{:.1f}", "", 1, "INTERCEPTIONS GAINED / GAME"),
    ('havoc_rate_defense', "{:.1f}", "%", 1, "HAVOC RATE"),
]

def build_rundown_panel(header_text, rows, left_row, right_row, widths=panel_widths, team_a_label=None, team_b_label=None):
    """Same-stat, team-vs-team panel (like build_stat_panel/OVERALL_PAIRS),
    extended to support an optional SECOND stat packed into the same cell
    (a 9-tuple row instead of 5) -- e.g. rush yards/game shown right next to
    rush play % on one line. Still real get_stat_tier() shading on the
    PRIMARY stat of a combo row; the secondary number rides along as plain
    text (mixing two different stats' color scales in one cell would be
    misleading, not useful). team_a_label/team_b_label, if given, add a
    small sub-header row right under the title bar with each team's
    abbreviation centered directly over its own value column -- so it's
    unambiguous which column is which team even with the panel's main "TEAM
    RUNDOWN" title carrying no team names."""
    header_rows = [[Paragraph(header_text, style_rundown_title), "", ""]]
    header_spans = [('SPAN', (0,0), (-1,0))]
    if team_a_label is not None and team_b_label is not None:
        header_rows.append([
            Paragraph(team_a_label, style_rundown_team_hdr),
            "",
            Paragraph(team_b_label, style_rundown_team_hdr),
        ])
    h_table = Table(header_rows, colWidths=widths)
    h_table.setStyle(TableStyle([
        *header_spans,
        ('BACKGROUND', (0,0), (-1,-1), obsidian_black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))

    matrix = []
    bg_rows = []

    def _cell_text(row_src, key, fmt, suffix, mult, key2=None, fmt2=None, suffix2=None, mult2=None):
        raw = row_src.get(key) if hasattr(row_src, 'get') else row_src[key]
        missing = pd.isna(raw)
        val = (raw * mult) if not missing else None
        # get_stat_tier() takes the RAW (pre-mult) value -- league_avgs/
        # league_stds/elite_thresholds are always computed from the raw
        # dataframe column, so comparing a mult-scaled display value against
        # them would be comparing numbers on two different scales (see the
        # real bug this fixed in NET EFFICIENCY (EPA/PLAY), README.txt).
        box, text_color = (None, COLOR_BLACK) if missing else get_stat_tier(raw, key)
        team_name = row_src.get('team') if hasattr(row_src, 'get') else row_src['team']
        primary = _with_rank("--" if missing else fmt.format(val) + suffix, key, team_name)
        if key2 is not None:
            raw2 = row_src.get(key2) if hasattr(row_src, 'get') else row_src[key2]
            missing2 = pd.isna(raw2)
            val2 = (raw2 * mult2) if not missing2 else None
            secondary = _with_rank("--" if missing2 else fmt2.format(val2) + suffix2, key2, team_name)
            # Same font size for both numbers, "value / value%" instead of
            # value (value%).
            text = f'<font color="{text_color}">{primary} / {secondary}</font>'
        else:
            text = f'<font color="{text_color}">{primary}</font>'
        return text, box

    for row_def in rows:
        if len(row_def) == 9:
            key, fmt, suffix, mult, key2, fmt2, suffix2, mult2, label = row_def
        else:
            key, fmt, suffix, mult, label = row_def
            key2 = fmt2 = suffix2 = mult2 = None

        str_left, c_left = _cell_text(left_row, key, fmt, suffix, mult, key2, fmt2, suffix2, mult2)
        str_right, c_right = _cell_text(right_row, key, fmt, suffix, mult, key2, fmt2, suffix2, mult2)

        matrix.append([
            Paragraph(str_left, style_rundown_val),
            Paragraph(label, style_rundown_label),
            Paragraph(str_right, style_rundown_val),
        ])
        bg_rows.append([c_left, center_grey, c_right])

    t = build_centered_table(matrix, widths, bg_colors=bg_rows, border_color='#000000')
    # Slightly more breathing room than build_centered_table's default
    # (1.5pt) -- trimmed down from 2.6 to shrink this grid's overall
    # footprint (17 rows x 2 panels adds up fast), while still a bit
    # roomier than the tightest panels on the sheet.
    t.setStyle(TableStyle([
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
    ]))
    return [h_table, Spacer(1, 1), t]

# ==============================================================================
# PREVIOUS SCHEDULE: every game played so far this season, one row per game --
# real score, real rush/pass yards (offense), real rush/pass yards allowed
# (defense), straight from nfl_game_log.csv (see nfl_data_pull.py PART 3B).
# A team with 2 games played shows 2 rows; a bye-week team shows exactly what
# it's played, nothing padded in.
# ==============================================================================

style_sched_hdr = ParagraphStyle('SchedHdr', fontName='Helvetica-Bold', fontSize=5.0, leading=6, textColor=colors.white, alignment=1)
style_sched_cell = ParagraphStyle('SchedCell', fontName='Helvetica', fontSize=5.2, leading=6.4, alignment=1)

def _team_schedule_rows(team):
    rows = game_log_df[game_log_df['team'] == team].sort_values('week')
    if rows.empty:
        return None
    out = []
    for _, r in rows.iterrows():
        vs_at = "vs" if bool(r['is_home']) else "@"
        result = r['result'] if pd.notna(r['result']) else "-"
        out.append([
            Paragraph(f"WK {int(r['week'])}", style_sched_cell),
            Paragraph(f"{result} {vs_at} {r['opponent']}", style_sched_cell),
            Paragraph(f"{int(r['pts_for'])}-{int(r['pts_against'])}", style_sched_cell),
            Paragraph(f"{r['rush_yards']:.0f}", style_sched_cell),
            Paragraph(f"{r['pass_yards']:.0f}", style_sched_cell),
            Paragraph(f"{r['rush_yards_allowed']:.0f}", style_sched_cell),
            Paragraph(f"{r['pass_yards_allowed']:.0f}", style_sched_cell),
        ])
    return out

SCHEDULE_COL_LABELS = ["WK", "RESULT", "SCORE", "RUSH YDS", "PASS YDS", "RUSH YDS\nALLOWED", "PASS YDS\nALLOWED"]

def build_schedule_table(team):
    body_rows = _team_schedule_rows(team)
    header_row = [Paragraph(lbl.replace("\n", "<br/>"), style_sched_hdr) for lbl in SCHEDULE_COL_LABELS]
    if body_rows is None:
        data = [header_row, [Paragraph("No games played yet this season", style_sched_cell), "", "", "", "", "", ""]]
        span_style = [('SPAN', (0,1), (-1,1))]
    else:
        data = [header_row] + body_rows
        span_style = []
    col_widths = [22, 62, 32, 32, 32, 40, 40]
    t = Table(data, colWidths=col_widths)
    style = [
        ('BACKGROUND', (0,0), (-1,0), obsidian_black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#000000')),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#EFEFEF')]),
    ] + span_style
    t.setStyle(TableStyle(style))
    return t

def build_previous_schedule_panel(t_a, t_b):
    body_row = [[build_schedule_table(t_a), build_schedule_table(t_b)]]
    body_table = Table(body_row, colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    body_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    return [full_width_header("PREVIOUS SCHEDULE", widths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH]), Spacer(1, 1), body_table]

def build_stat_panel(header_text, pairs, left_row, right_row, widths=panel_widths,
                      val_style=None, pop_style=None, label_style=None, row_padding=None,
                      grid_widths=None):
    """Builds one panel: a full-width header bar, then one row per stat pair,
    with the larger raw value on each row getting a bolder, thicker-bordered
    'pop' treatment so it visually stands out from its neighbor.
    val_style/pop_style/label_style let one specific panel (e.g. VOID RATING)
    use its own larger fonts without changing every other build_stat_panel
    caller -- default to the shared globals when not given. row_padding, if
    given, overrides build_centered_table's default 1.5pt row padding.
    grid_widths, if given, sizes ONLY the stat-rows grid narrower than the
    header bar (widths) -- the header stays full width, the grid centers
    itself underneath it (Table's default hAlign is CENTER) at its own,
    smaller size. Defaults to widths (old behavior: header and grid match)."""
    val_style = val_style or style_val_base
    pop_style = pop_style or style_val_pop
    label_style = label_style or style_label
    grid_widths = grid_widths or widths
    header_row = [[Paragraph(header_text, style_sub_hdr), "", ""]]
    h_table = Table(header_row, colWidths=widths)
    h_table.setStyle(TableStyle([
        ('SPAN', (0,0), (-1,0)),
        ('BACKGROUND', (0,0), (-1,-1), obsidian_black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))

    matrix = []
    bg_rows = []
    pop_cells = []

    for r_idx, row_def in enumerate(pairs):
        # Pairs are normally 6-tuples (left_key, right_key, fmt, suffix,
        # mult, label). An optional 7th element, comparable=False, turns off
        # the "bigger raw number pops" treatment below for a row where the
        # two sides are real numbers but NOT on the same scale/units (e.g.
        # a 0-100 efficiency index vs. a 0-30ish% rate) -- comparing them by
        # raw magnitude would always pick the same side no matter what
        # actually happened on the field, which is a misleading "who's
        # winning" signal, not an honest one. Defaults to True (unchanged
        # behavior) for every existing same-scale pair.
        if len(row_def) == 7:
            left_key, right_key, fmt, suffix, mult, label, comparable = row_def
        else:
            left_key, right_key, fmt, suffix, mult, label = row_def
            comparable = True
        raw_left = left_row.get(left_key) if hasattr(left_row, 'get') else left_row[left_key]
        raw_right = right_row.get(right_key) if hasattr(right_row, 'get') else right_row[right_key]
        # Some NGS-sourced fields go real NaN when that week's pull failed
        # (see NO_FILLNA_COLS in nfl_data_pull.py) -- render those as a
        # plain "--" cell instead of "nan" or a misleading red highlight.
        left_missing = pd.isna(raw_left)
        right_missing = pd.isna(raw_right)

        val_left = (raw_left * mult) if not left_missing else None
        val_right = (raw_right * mult) if not right_missing else None

        # get_stat_tier() takes the RAW (pre-mult) value, not the mult-scaled
        # display value -- league_avgs/league_stds/elite_thresholds are always
        # computed from the raw dataframe column, so a field with mult != 1
        # (e.g. NET EFFICIENCY (EPA/PLAY), mult=100 to show a fraction as a
        # %) would otherwise compare a x100 display number against
        # thresholds computed on the un-scaled x1 data -- which is a real bug
        # that made a merely-decent EPA/play look "elite" every time (any
        # positive value times 100 cleared the raw ~0.25 elite cutoff).
        # Fixed by always tiering off the raw value; val_left/val_right stay
        # mult-scaled for DISPLAY text and for the "bigger number pops"
        # comparison below (both sides share the same mult, so that
        # comparison is unaffected either way).
        c_left, text_left = (None, COLOR_BLACK) if left_missing else get_stat_tier(raw_left, left_key)
        c_right, text_right = (None, COLOR_BLACK) if right_missing else get_stat_tier(raw_right, right_key)
        if label == "NET PPD":
            # no league-percentile tiering exists for points_per_drive -- never let
            # get_stat_tier's missing-threshold defaults paint it purple.
            c_left = c_right = None
        _ts = teal_side(label, left_row, right_row)
        if _ts == 'L':
            c_left, text_left = colors.HexColor(COLOR_TEAL_FILL), COLOR_BLACK
        elif _ts == 'R':
            c_right, text_right = colors.HexColor(COLOR_TEAL_FILL), COLOR_BLACK

        pop_left = comparable and (not left_missing) and (not right_missing) and (val_left > val_right)
        pop_right = comparable and (not left_missing) and (not right_missing) and (val_right > val_left)

        # fmt is normally a "{:...}"-style format string, but can also be a
        # callable (e.g. _ordinal for LEAGUE RANK) for display logic a plain
        # format string can't express -- suffix is ignored in that case.
        # The callable/format branch stays INSIDE the missing-value ternary
        # (not precomputed above it) so it's never called on a None value --
        # val_left/val_right are None exactly when left_missing/right_missing
        # is True, and Python's ternary short-circuits the untaken branch.
        disp_left = "--" if left_missing else (fmt(val_left) if callable(fmt) else fmt.format(val_left) + suffix)
        disp_right = "--" if right_missing else (fmt(val_right) if callable(fmt) else fmt.format(val_right) + suffix)
        if not callable(fmt):
            team_left = left_row.get('team') if hasattr(left_row, 'get') else left_row['team']
            team_right = right_row.get('team') if hasattr(right_row, 'get') else right_row['team']
            disp_left = _with_rank(disp_left, left_key, team_left)
            disp_right = _with_rank(disp_right, right_key, team_right)
        str_left = f'<font color="{text_left}">{disp_left}</font>'
        str_right = f'<font color="{text_right}">{disp_right}</font>'

        matrix.append([
            Paragraph(str_left, pop_style if pop_left else val_style),
            Paragraph(label, label_style),
            Paragraph(str_right, pop_style if pop_right else val_style),
        ])
        bg_rows.append([c_left, center_grey, c_right])
        if pop_left:
            pop_cells.append((r_idx, 0))
        if pop_right:
            pop_cells.append((r_idx, 2))

    t = build_centered_table(matrix, grid_widths, bg_colors=bg_rows, border_color='#000000')
    if row_padding is not None:
        t.setStyle(TableStyle([
            ('TOPPADDING', (0,0), (-1,-1), row_padding),
            ('BOTTOMPADDING', (0,0), (-1,-1), row_padding),
        ]))

    # Layer the "3D pop" box border onto whichever cell had the bigger raw
    # number in each row.
    pop_style_cmds = []
    for (r_idx, c_idx) in pop_cells:
        pop_style_cmds.append(('BOX', (c_idx, r_idx), (c_idx, r_idx), 1.5, colors.black))
    if pop_style_cmds:
        t.setStyle(TableStyle(pop_style_cmds))

    return [h_table, Spacer(1, 1), t]

# ==============================================================================
# QUICK SCAN SHEET ("the old man sheet") -- a separate, condensed panel
# builder used ONLY by build_quick_scan_elements below. This sheet does NOT
# use get_stat_tier (no purple/orange/red/green league-percentile tiering)
# and does NOT use the "pop" bold-border treatment build_stat_panel uses.
# Coloring here is a plain, faded fill on whichever cell real WON that row,
# between just these two teams -- colored by which SIDE OF THE BALL that
# winning number belongs to (light green = offense, light orange = defense),
# not by which team. QUICK_SCAN_DEFENSE_COLUMNS below is the real
# classification of every column used on this sheet -- every column not in it
# defaults to offense (which is every real column used here except the
# explicitly defensive ones). QUICK_SCAN_NEUTRAL_COLUMNS are the whole-team
# composite numbers (VOID/NET ratings) that genuinely mix both sides of the
# ball -- those stay plain/uncolored rather than being forced into an
# offense-or-defense bucket that would misrepresent them. Direction (is a
# bigger or smaller raw number better for this specific column) is decided by
# the SAME real lower_is_better_fields list the main sheet's get_stat_tier
# already uses -- not a separate, guessed table -- so a stat like SACK %
# correctly favors the LOWER number and a stat like CPOE correctly favors the
# HIGHER one.
# ==============================================================================
QUICK_SCAN_FAVOR_GREEN  = colors.HexColor('#C8F0D2')  # faded/pale green  -- offense won that row (much lighter than COLOR_GREEN '#2ECC71')
QUICK_SCAN_FAVOR_ORANGE = colors.HexColor('#FBE0BD')  # faded/pale orange -- defense won that row (much lighter than COLOR_ORANGE '#F5A623')

QUICK_SCAN_DEFENSE_COLUMNS = {
    # THE FIELD GENERAL / THE CAPTAIN
    'completion_pct_allowed', 'sack_pct_defense', 'interceptions_gained_per_game',
    'interior_pressure_rate', 'edge_pressure_rate',
    # THE POUND
    'rushing_yards_allowed_per_game', 'stuff_rate_defense', 'def_run_stop_rate', 'def_rushing_yards_per_play',
    # THE DIRTY BIRD / FREEBIRDS
    'catch_pct_allowed', 'yac_allowed_per_reception_pbp', 'target_separation_allowed', 'pressure_rate_forced_pct',
    # SECONDARY LOCK INDEX -- defense-only, no offensive mirror (see
    # SECONDARY_LOCK_PAIR's original writeup) -- added here so it colors
    # orange, not the 'offense' default, wherever it's now used on this
    # sheet.
    'secondary_lock_index',
    # OFFENSE vs DEFENSE matchup panel (MATCHUP_PAIRS), if ever reused here
    'def_efficiency_epa', 'def_yards_per_pass',
}
# Whole-team composite ratings (mix offense + defense together) -- left
# plain/uncolored on this sheet rather than forced into offense/defense.
QUICK_SCAN_NEUTRAL_COLUMNS = {'total_void_rank', 'total_void', 'net_efficiency_epa', 'net_epa_matchup'}

def _qs_side(col_name):
    """Real classification of one column as 'offense', 'defense', or None
    (whole-team composite, left uncolored) -- see the QUICK SCAN SHEET note
    above."""
    if col_name in QUICK_SCAN_NEUTRAL_COLUMNS:
        return None
    return 'defense' if col_name in QUICK_SCAN_DEFENSE_COLUMNS else 'offense'

def _qs_favorability(val, col_name):
    """Turns one real raw value into a real z-score (std devs from the real
    league average for that column), sign-flipped via the SAME real
    lower_is_better_fields list get_stat_tier uses so a positive result
    always means "good for whichever side this column belongs to" --
    EXACT SAME technique already used for real crossed-but-different-scale
    comparisons elsewhere in this file (see the PBE vs PRP note: "compared
    as z-scores (std devs from real league avg), not raw numbers"). This is
    NOT the get_stat_tier color system -- league_avgs/league_stds are only
    used here internally to make two DIFFERENT columns (e.g. this team's own
    completion % vs the other team's completion % ALLOWED, or sack_pct_
    offense vs sack_pct_defense) fairly comparable head-to-head, never
    rendered as a purple/orange/red tier the way the main sheet does. A
    plain same-scale raw comparison would be WRONG here -- e.g. comparing
    completion_pct (~55-70 range) directly against completion_pct_allowed
    (also ~55-70 range but LOWER is better for that side) without a
    league-relative normalization can't tell you which real number is
    actually the better outcome for its side."""
    avg = league_avgs.get(col_name, 0)
    std = league_stds.get(col_name, 0.1) or 0.1
    z = (val - avg) / std
    return -z if col_name in lower_is_better_fields else z

def build_quick_scan_panel(header_text, pairs, left_row, right_row, widths=panel_widths,
                            grid_widths=None, val_style=None, label_style=None, row_padding=None):
    """Same header-bar-plus-rows shape as build_stat_panel, but for the quick
    scan sheet only -- see the QUICK SCAN SHEET note above for why the
    coloring logic here is deliberately simpler/different. A row's `pairs`
    tuple can be a same-key row (left_key == right_key, e.g. both teams' own
    CPOE) or a crossed row (e.g. PRESSURE RATE ALLOWED vs FORCED) -- either
    way this just reads left_row[left_key] and right_row[right_key], decides
    which one real-favors between just these two teams, and colors that
    winning cell green or orange depending on whether the WINNING column is
    an offense or defense stat (_qs_side), not which team it belongs to."""
    val_style = val_style or style_val_base
    # Old man sheet only -- black text on a plain/white label cell instead
    # of white text on a grey fill (build_stat_panel's grey label column,
    # used by the full sheet/individual PDFs, is untouched).
    label_style = label_style or style_label_dark
    grid_widths = grid_widths or widths
    # Old man sheet only -- NO background fill on this header bar
    # (build_stat_panel's obsidian_black-filled header, used by the full
    # sheet/individual PDFs, is untouched), black text instead of the
    # near-white style_sub_hdr, and a bold black rule under the header row
    # in place of the fill to still mark where each new section starts.
    header_row = [[Paragraph(header_text, style_sub_hdr_dark), "", ""]]
    h_table = Table(header_row, colWidths=widths)
    h_table.setStyle(TableStyle([
        ('SPAN', (0,0), (-1,0)),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LINEBELOW', (0,0), (-1,-1), 1.5, colors.black),
    ]))

    matrix = []
    bg_rows = []
    for row_def in pairs:
        if len(row_def) == 7:
            left_key, right_key, fmt, suffix, mult, label, comparable = row_def
        else:
            left_key, right_key, fmt, suffix, mult, label = row_def
            comparable = True
        raw_left = left_row.get(left_key) if hasattr(left_row, 'get') else left_row[left_key]
        raw_right = right_row.get(right_key) if hasattr(right_row, 'get') else right_row[right_key]
        left_missing = pd.isna(raw_left)
        right_missing = pd.isna(raw_right)
        val_left = (raw_left * mult) if not left_missing else None
        val_right = (raw_right * mult) if not right_missing else None
        str_left = "--" if left_missing else (fmt(val_left) if callable(fmt) else fmt.format(val_left) + suffix)
        str_right = "--" if right_missing else (fmt(val_right) if callable(fmt) else fmt.format(val_right) + suffix)
        if not callable(fmt):
            team_left = left_row.get('team') if hasattr(left_row, 'get') else left_row['team']
            team_right = right_row.get('team') if hasattr(right_row, 'get') else right_row['team']
            str_left = _with_rank(str_left, left_key, team_left)
            str_right = _with_rank(str_right, right_key, team_right)

        # Direction-aware head-to-head favor -- ONLY between these two real
        # teams, not vs. the whole league (that's the whole point of this
        # sheet; league_avgs/league_stds are used only INSIDE
        # _qs_favorability to make two differently-scaled columns fairly
        # comparable, never shown as a tier -- see that function's
        # docstring). Ties, non-comparable/missing rows, and a winning
        # column with no offense/defense side (QUICK_SCAN_NEUTRAL_COLUMNS)
        # all get no fill at all, same as the main sheet leaving a cell
        # plain.
        bg_left = bg_right = None
        if comparable and not left_missing and not right_missing:
            eff_left = _qs_favorability(raw_left, left_key)
            eff_right = _qs_favorability(raw_right, right_key)
            if eff_left > eff_right:
                winner_side = _qs_side(left_key)
                if winner_side == 'offense':
                    bg_left = QUICK_SCAN_FAVOR_GREEN
                elif winner_side == 'defense':
                    bg_left = QUICK_SCAN_FAVOR_ORANGE
            elif eff_right > eff_left:
                winner_side = _qs_side(right_key)
                if winner_side == 'offense':
                    bg_right = QUICK_SCAN_FAVOR_GREEN
                elif winner_side == 'defense':
                    bg_right = QUICK_SCAN_FAVOR_ORANGE

        _ts = teal_side(label, left_row, right_row)
        if _ts == 'L':
            bg_left = colors.HexColor(COLOR_TEAL_FILL)
        elif _ts == 'R':
            bg_right = colors.HexColor(COLOR_TEAL_FILL)

        matrix.append([
            Paragraph(str_left, val_style),
            Paragraph(label, label_style),
            Paragraph(str_right, val_style),
        ])
        # Label cell background: no fill (was center_grey) -- per the same
        # "lose the grey color for stat labels" ask above.
        bg_rows.append([bg_left, None, bg_right])

    t = build_centered_table(matrix, grid_widths, bg_colors=bg_rows, border_color='#000000')
    if row_padding is not None:
        t.setStyle(TableStyle([
            ('TOPPADDING', (0,0), (-1,-1), row_padding),
            ('BOTTOMPADDING', (0,0), (-1,-1), row_padding),
        ]))
    return [h_table, Spacer(1, 1), t]

def _pair_row_cells(left_row, right_row, left_key, right_key, fmt, suffix, mult, label,
                     comparable=True, val_style=None, pop_style=None, label_style=None):
    """Builds one [left_cell, label_cell, right_cell] Paragraph row plus its
    matching [left_color, grey, right_color] background row -- the same
    per-row logic as build_stat_panel()'s inner loop (same get_stat_tier()-
    on-the-RAW-value rule, same 'bigger raw number pops' treatment), just
    factored out so the grid-style position panels below (THE CAPTAIN /
    FREEBIRDS / DAWGS) can mix these plain real stat-pair rows with their
    own special rows (pressure index, PBE/PRP -- real too, just not a plain
    numeric compare) inside ONE shared grid, without duplicating the
    raw-vs-mult-scaled tiering rule that build_stat_panel already got right."""
    val_style = val_style or style_val_base
    pop_style = pop_style or style_val_pop
    label_style = label_style or style_label

    raw_left = left_row.get(left_key) if hasattr(left_row, 'get') else left_row[left_key]
    raw_right = right_row.get(right_key) if hasattr(right_row, 'get') else right_row[right_key]
    left_missing = pd.isna(raw_left)
    right_missing = pd.isna(raw_right)
    val_left = (raw_left * mult) if not left_missing else None
    val_right = (raw_right * mult) if not right_missing else None

    c_left, text_left = (None, COLOR_BLACK) if left_missing else get_stat_tier(raw_left, left_key)
    c_right, text_right = (None, COLOR_BLACK) if right_missing else get_stat_tier(raw_right, right_key)

    pop_left = comparable and (not left_missing) and (not right_missing) and (val_left > val_right)
    pop_right = comparable and (not left_missing) and (not right_missing) and (val_right > val_left)

    team_left = left_row.get('team') if hasattr(left_row, 'get') else left_row['team']
    team_right = right_row.get('team') if hasattr(right_row, 'get') else right_row['team']
    disp_left = _with_rank("--" if left_missing else fmt.format(val_left) + suffix, left_key, team_left)
    disp_right = _with_rank("--" if right_missing else fmt.format(val_right) + suffix, right_key, team_right)
    str_left = f'<font color="{text_left}">{disp_left}</font>'
    str_right = f'<font color="{text_right}">{disp_right}</font>'

    row = [Paragraph(str_left, pop_style if pop_left else val_style),
           Paragraph(label, label_style),
           Paragraph(str_right, pop_style if pop_right else val_style)]
    bg = [c_left, center_grey, c_right]
    return row, bg

# Same-stat, team-vs-team tendency comparison (last game's scheme proxies +
# season-long 4th down aggressiveness). Reuses build_stat_panel exactly like
# OVERALL_PAIRS does, since there's no opponent-side counterpart for these --
# it's each team's own tendency shown side by side. NOTE: the standalone
# "SCHEME TENDENCIES (LAST GAME)" panel built from this list is GONE. Still
# real, still pulled every run -- just not displayed anywhere right now.
TENDENCY_PAIRS = [
    ('last_game_off_pass_rate', 'last_game_off_pass_rate', "{:.1f}", "%", 1, "PASS RATE (LAST GM, OFF)"),
    ('last_game_off_pressure_rate_allowed', 'last_game_off_pressure_rate_allowed', "{:.1f}", "%", 1, "PRESSURE ALLOWED (LAST GM)"),
    ('last_game_def_pass_rate_faced', 'last_game_def_pass_rate_faced', "{:.1f}", "%", 1, "PASS RATE FACED (LAST GM, DEF)"),
    ('last_game_def_pressure_rate_forced', 'last_game_def_pressure_rate_forced', "{:.1f}", "%", 1, "PRESSURE FORCED (LAST GM)"),
    ('fourth_down_aggressiveness_pct', 'fourth_down_aggressiveness_pct', "{:.0f}", "%", 1, "4TH DOWN AGGRESSIVENESS (SEASON)"),
]

# ==============================================================================
# PREDICTION MODEL: win probability, spread, and team totals.
#
# SUPERSEDED 2026-09-27 by nfl_margin_model.py (see that file's own header
# for the full story): the OLD approach below (a hand-weighted power rating
# from OFFENSE_RATING_WEIGHTS/DEFENSE_RATING_WEIGHTS in nfl_data_pull.py --
# "", never fit to any real outcome -- converted to points via the two
# constants just below, which WERE real-fit on 2026-09-22) checked out
# badly against real week-2 results: 6/16 (37.5%) straight-up picks, 17.3
# pt avg margin error, and two games given 80%+/86%+ real win-probability
# that lost outright. The 2026-09-22 fit only calibrated the OUTPUT of the
# hand-picked rating -- the weights deciding which stats mattered, and how
# much, were still a guess.
#
# predict_matchup() now gets its real margin/total/win-probability from
# nfl_margin_model.py's own directly-fit regression instead (20 raw real
# team stats -> real least squares straight to real margin/total, no
# hand-picked weights anywhere in the chain, re-fit automatically as more
# real games complete -- see that file's docstring for the honest
# in-sample AND out-of-sample holdout numbers).
#
# POINTS_PER_RATING_UNIT/WIN_PROB_SCALE below are KEPT only as the
# fallback path's constants (predict_matchup() only reaches them if
# nfl_margin_model.py has never produced a real fit yet, e.g. brand new
# setup with no network available at all) -- not the live model anymore.
# ==============================================================================
POINTS_PER_RATING_UNIT = 3.057   # fallback only -- see note above (real fit, 2026-09-22, 957 games)
WIN_PROB_SCALE = 8.243           # fallback only -- see note above (real fit, 2026-09-22, 957 games)

league_avg_ppg = league_avgs.get('points_per_game', 22.0)

def predict_matchup(row_a, row_b):
    """row_a = away team's real nfl_team_matrix.csv row, row_b = home
    team's. UPDATED 2026-09-27: uses the real, data-fitted margin/total
    model (nfl_margin_model.py) whenever a real fit is available -- see
    that file's own header for the full "why" (real week-2 check: 37.5%
    straight-up, 17.3 pt avg margin error, two 80%+/86%+ confidence picks
    that lost outright) and its honest holdout numbers. Falls back to the
    OLD hand-picked-rating approach only if nfl_margin_model.py has never
    produced a fit yet (e.g. brand new setup, or no network available at
    all) -- run `python nfl_margin_model.py` once by hand to bootstrap it;
    every run after that keeps itself fresh automatically (see
    nfl_margin_model.get_or_refresh_fit's own docstring)."""
    fit = nfl_margin_model.get_or_refresh_fit()
    if fit is not None:
        pred = nfl_margin_model.predict_from_fit(fit, row_a, row_b)
        margin, total, win_prob_a = pred['margin'], pred['total'], pred['win_prob_a']
        exp_pts_a = max((total + margin) / 2, 3.0)
        exp_pts_b = max((total - margin) / 2, 3.0)
        return {'exp_pts_a': exp_pts_a, 'exp_pts_b': exp_pts_b, 'margin': exp_pts_a - exp_pts_b,
                'total': exp_pts_a + exp_pts_b, 'win_prob_a': win_prob_a, 'model_source': 'fitted'}

    # FALLBACK -- only reachable if nfl_margin_model.py has never produced
    # a real fit yet (nfl_margin_model_fit.json doesn't exist and this run
    # couldn't create one, e.g. no network available at all). Same
    # hand-picked power-rating approach this file used before 2026-09-27 --
    # kept only as a safety net so a brand new setup doesn't crash, not the
    # real model anymore.
    net_a = row_a['off_power_rating'] - row_b['def_power_rating']
    net_b = row_b['off_power_rating'] - row_a['def_power_rating']

    exp_pts_a = max(league_avg_ppg + POINTS_PER_RATING_UNIT * net_a, 3.0)
    exp_pts_b = max(league_avg_ppg + POINTS_PER_RATING_UNIT * net_b, 3.0)

    margin = exp_pts_a - exp_pts_b
    total = exp_pts_a + exp_pts_b
    win_prob_a = 1 / (1 + np.exp(-margin / WIN_PROB_SCALE))

    return {'exp_pts_a': exp_pts_a, 'exp_pts_b': exp_pts_b, 'margin': margin,
            'total': total, 'win_prob_a': win_prob_a,
            'model_source': 'fallback (hand-picked rating -- run nfl_margin_model.py)'}

def full_width_header(text, widths=panel_widths):
    row = [[Paragraph(text, style_sub_hdr)] + [""] * (len(widths) - 1)]
    t = Table(row, colWidths=widths)
    style = [
        ('BACKGROUND', (0,0), (-1,-1), obsidian_black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]
    if len(widths) > 1:
        style.append(('SPAN', (0,0), (-1,0)))
    t.setStyle(TableStyle(style))
    return t

def build_note(text, widths=panel_widths):
    """A small italic caption row (style_note, ~4.8pt) -- factored out of
    the DROP % note under THE FREEBIRDS so the same compact explainer
    format can be reused for the EPA/VOID/TUG O' WAR/PASS BLOCK vs PASS
    RUSH/SECONDARY LOCK INDEX explainer captions without repeating this Table/TableStyle
    boilerplate at every call site."""
    row = [[Paragraph(text, style_note)]]
    t = Table(row, colWidths=[sum(widths)])
    t.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    return t

# ==============================================================================
# STAT-WEIGHTED WIN PREDICTION MODEL -- real, backtested against YOUR OWN
# nfl_game_log.csv history (persists across every week you run
# nfl_data_pull.py -- same idea as the CFB project's version, fixed this
# same session to NOT stop the instant backtest accuracy clears 80%; it
# keeps testing every remaining stat against the current best combination
# and only stops when nothing left actually improves it, so it's genuinely
# hunting for the strongest combination rather than settling for "good
# enough").: for right now this uses the FULL stat set on the sheet, not
# just the ones with real per-game history -- a REAL, MEASURED weight for
# any stat nfl_game_log.csv actually has per-game history for (see PART 3C
# in nfl_data_pull.py), and a documented domain-knowledge PRIOR weight for
# every other stat (VOID, pressure index, havoc rate, etc.) until
# nfl_data_pull.py is extended to log those per-game too. Every stat shown
# anywhere from this system says which kind of weight backed it --
# "measured" vs "prior" -- so nothing here pretends to be more rigorously
# tested than it actually is. This whole system feeds two things: the STAT MISMATCH BREAKDOWN per matchup below, and the >=80%
# real-win-probability turquoise team-name highlight in the header (see
# build_matchup_elements).
# ==============================================================================

NFL_STAT_HISTORY_PATH = OUT_DIR / "nfl_stat_history.json"

# Real per-game backtestable stats -- keys are exact nfl_game_log.csv
# columns (see PART 3C, nfl_data_pull.py). Deliberately EXCLUDES pts_for/
# pts_against: those ARE the outcome (the team that scored more won, by
# definition, barring a tie), so including them as a "predictor" would be
# circular, not a real signal.
# (label, game_log_col, team_matrix_col, higher_is_better, category)
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

# Stats with NO real per-game history yet -- PRIOR weights only, grounded
# in public sports-analytics consensus, not measured against your results.
# Net-efficiency/EPA-style ratings and pressure are the strongest known
# forward predictors, so VOID and PBE/PRP get the higher priors; turnover
# margin gets a deliberately LOW prior despite how dramatic it looks in a
# single box score -- it's famously one of the least persistent stats in
# football week to week.
# (label, team_matrix_col, higher_is_better, category, prior_weight)
NFL_PRIOR_STATS = [
    ("VOID (Total)", "total_void", True, "OVERALL EFFICIENCY", 0.22),
    ("VOID (Offense)", "off_void", True, "OVERALL EFFICIENCY", 0.15),
    ("VOID (Defense)", "def_void", True, "OVERALL EFFICIENCY", 0.15),
    ("Red Zone TD %", "redzone_efficiency", True, "OVERALL EFFICIENCY", 0.08),
    ("Red Zone TD % Allowed", "redzone_efficiency_allowed", False, "OVERALL EFFICIENCY", 0.08),
    ("Havoc Rate (Defense)", "havoc_rate_defense", True, "DISRUPTION & TURNOVERS", 0.12),
    ("Run Stuff Rate (Offense side, lower better)", "stuff_rate_offense", False, "TRENCHES & RUN GAME", 0.10),
    ("Run Stuff Rate (Defense side)", "stuff_rate_defense", True, "TRENCHES & RUN GAME", 0.10),
    ("Pass Block Efficiency (PBE)", "pbe_score", True, "PASSING & PROTECTION", 0.10),
    ("Pass Rush Productivity (PRP)", "prp_score", True, "PASSING & PROTECTION", 0.10),
    ("Passer Rating", "open_source_passer_rating", True, "QB ACCURACY & EFFICIENCY", 0.10),
    ("CPOE", "cpoe", True, "QB ACCURACY & EFFICIENCY", 0.08),
    ("Turnover Margin (Offense, giveaways)", "turnover_margin_offense", False, "DISRUPTION & TURNOVERS", 0.04),
    ("Turnover Margin (Defense, takeaways)", "turnover_margin_defense", True, "DISRUPTION & TURNOVERS", 0.04),
]

# ==============================================================================
# QUALITY STAT CONSENSUS. Separate from the hybrid-weighted model above:
# this is a real, unweighted MAJORITY VOTE across a fixed set of "quality"
# stats, checked against each week's current nfl_team_matrix.csv numbers
# for the upcoming matchup.
#
# The stats below and their real win rates are NOT a guess -- they come
# from a real walk-forward backtest (pre-game rolling averages only, no
# leakage) across 4 full real seasons (2022-2025, 1084 real completed
# games), run with nfl_historical_backtest.py / nfl_walkforward_backtest.py
# / nfl_flare_backtest.py (all three still in this folder -- rerun them
# yourself anytime, e.g. once more real seasons are in the books, and
# update this list + the two thresholds below if the numbers move).
# "Quality" = real individual walk-forward win rate > 55% in that backtest
# AND a real season-to-date column already sitting in nfl_team_matrix.csv
# (both conditions required -- see the REFRESHED note below).
#
# REFRESHED after the "trenches" round extended the backtest from 17 to
# 34 total stats tested. Went from 12 to 15 stats: added Run Success
# Rate (Offense) 59.2%, Pass Block Efficiency (PBE) 58.2%, and Rush
# Yards Per Play Allowed (Defense) 55.9% -- all 3 cleared the same >55%
# real bar AND already have a live nfl_team_matrix.csv column
# (off_run_success_rate, pbe_score, def_rushing_yards_per_play), so no
# nfl_data_pull.py changes were needed to add them here.
#
# Real stats that also tested >55% but are deliberately NOT
# added yet: Rush EPA/Play (Offense) 56.2%, Line Yards (Offense) 55.1%,
# RYOE per Attempt 54.4% (just under the bar anyway). None of these three
# has a real season-to-date column in nfl_team_matrix.csv yet -- they only
# exist as PER-GAME numbers inside the backtest scripts. Adding them here
# would need nfl_data_pull.py extended first to compute and export a real
# season-long version, a separate task not done in this round. Also
# deliberately left out: Stuff Rate and TFL Rate, which the backtest
# showed testing at or even below a real coin flip (TFL Rate Forced
# specifically at 48.7% walk-forward, 40.7% retrospective -- likely
# contaminated by real victory-formation kneel-downs counting as tackles
# for loss, an honest open question, not yet fixed).
#
# The 5 stats still deliberately excluded from the original round
# (Interceptions Thrown 52.8%, Success Rate Allowed 51.8%, 3rd Down %
# Allowed 51.5%, Sack % Forced 51.5%, and Pass Yards Allowed 48.6% -- that
# last one actually UNDER a coin flip, a real game-script-contamination
# effect on raw yards-allowed stats) stay out so a handful of noisy stats
# can't pad out a count that looks more confident than it is.
#
# Real backtest result behind the two thresholds (from the original 12-stat
# round): requiring 4+ of the quality stats to agree hit 68.8% accuracy on
# 650 real games (vs. 55-57% with fewer than 4 agreeing) -- a real,
# meaningful signal, not just noise. You picked 8 of 12 (two-thirds) as the
# bar for the turquoise highlight specifically (so the color stays a rare
# "concentrate on this" signal rather than lighting up on 3 of every 4
# games) -- QUALITY_CONSENSUS_HOT_THRESHOLD below is now 10 of 15, the same
# real two-thirds proportion, recalibrated for the larger list rather than
# left at the old raw number (which would have made the hot badge trigger
# more easily just from the list growing, not from any real change in
# evidence).
#
# The badge's "(NN%)" win-prob number is a WEIGHTED vote share, not a plain
# count ratio: each stat's vote counts in proportion to its own real_win_rate
# above rather than as a flat +1, so Success Rate (63.4%) pulls harder than
# 3rd Down % (55.3%) when they disagree. Deliberately NOT a log-odds/naive-
# Bayes stack -- several of these stats (EPA and Success Rate especially)
# are correlated, and treating every vote as independent evidence would
# compound that correlation into overconfident percentages the backtest
# never actually demonstrated. Weighted vote share degrades gracefully to a
# plain ratio when the agreeing stats have similar real win rates (the
# common case), and only leans away from it when a lopsided mix of strong
# vs. marginal stats disagree. This is an honest re-expression of the same
# count you already see (N/M), not a separate prediction -- it will not
# always match a naive N/M arithmetic percentage.
# (label, matrix_col, higher_is_better, real_win_rate)
NFL_QUALITY_STATS = [
    ("Success Rate (Offense)", "off_success_rate", True, 0.634),
    ("EPA/Play (Offense)", "off_efficiency_epa", True, 0.610),
    ("Pass Yards", "passing_yards_per_game", True, 0.594),
    ("Run Success Rate (Offense)", "off_run_success_rate", True, 0.592),
    ("Sack % Taken", "sack_pct_offense", False, 0.591),
    ("Completion %", "completion_pct", True, 0.588),
    ("Pass Block Efficiency (PBE)", "pbe_score", True, 0.582),
    ("Pressure Rate Allowed", "pressure_rate_allowed_pct", False, 0.578),
    ("Interceptions Gained", "interceptions_gained_per_game", True, 0.571),
    ("EPA/Play Allowed (Defense)", "def_efficiency_epa", False, 0.559),
    ("Rush Yards Allowed", "rushing_yards_allowed_per_game", False, 0.559),
    ("Rush Yards Per Play Allowed (Defense)", "def_rushing_yards_per_play", False, 0.559),
    ("Pressure Rate Forced", "pressure_rate_forced_pct", True, 0.557),
    ("Rush Yards", "rushing_yards_per_game", True, 0.553),
    ("3rd Down %", "third_down_pct_offense", True, 0.553),
]
QUALITY_CONSENSUS_HOT_THRESHOLD = 10


def compute_nfl_quality_consensus(r_a, r_b):
    """For THIS week's matchup (r_a/r_b are nfl_team_matrix.csv rows --
    current season-to-date numbers, not a past game), counts how many of
    the 15 real quality stats favor each team, AND a weighted win-prob share
    built from those same votes. A stat with missing data on either side, or
    an exact tie, votes for neither (honestly excluded, not guessed) -- so
    count_a + count_b is the real number of stats that had something to say
    this week, which can be less than 15.

    win_prob_a is each side's total real_win_rate weight divided by the
    combined weight of every stat that voted -- i.e. a vote from a stat with
    a stronger real backtested win rate (e.g. Success Rate at 63.4%) counts
    for more than a vote from a marginal one (e.g. 3rd Down % at 55.3%),
    instead of every agreeing stat counting as a flat +1. Returns None when
    no stat had usable data on both sides."""
    count_a, count_b = 0, 0
    weight_a, weight_b = 0.0, 0.0
    for label, matrix_col, higher_is_better, win_rate in NFL_QUALITY_STATS:
        va, vb = r_a.get(matrix_col), r_b.get(matrix_col)
        if va is None or vb is None or (isinstance(va, float) and pd.isna(va)) or (isinstance(vb, float) and pd.isna(vb)):
            continue
        if va == vb:
            continue
        a_leads = (va > vb) if higher_is_better else (va < vb)
        if a_leads:
            count_a += 1
            weight_a += win_rate
        else:
            count_b += 1
            weight_b += win_rate
    total_weight = weight_a + weight_b
    win_prob_a = (weight_a / total_weight) if total_weight > 0 else None
    return count_a, count_b, win_prob_a


# ==============================================================================
# COMPOSITE OFFENSE/DEFENSE POWER RANK + STRENGTH OF SCHEDULE (SOS) ("can we
# create a sos... label each team off and def with rank based off our
# data... and factor that into games already played and how that effects the
# teams currents stats"). REFERENCE LAYER for now: real ranks get shown on
# the PDF (see "{team} OFFENSE (rank)" headers below), and the real
# opponent-adjusted scores get computed here too, but NEITHER feeds
# compute_nfl_quality_consensus/the live predictions yet. That only happens
# once a real walk-forward backtest (see nfl_sos_backtest.py) proves an
# SOS-adjusted stat actually predicts better -- same "prove it before
# trusting it" standard every other stat here has been held to.
#
# RANK BASIS -- a composite of the SAME real 15 NFL_QUALITY_STATS above,
# split by which side of the ball each one measures (10 offensive, 5
# defensive -- see OFFENSE_QUALITY_STATS/DEFENSE_QUALITY_STATS below), each
# weighted by its own real backtested win rate (same "win_rate - 0.5"
# edge-weight scheme compute_nfl_quality_consensus/tune_nfl_stat_weights
# already use). Not a new, unproven stat list -- the already-proven one,
# just turned into a per-side composite instead of a head-to-head vote.
#
# METHOD:
#   1. z-score every real team's value on each of the 15 stats across the
#      real 32-team league (oriented so higher z = better, flipping sign
#      for stats where higher_is_better is False).
#   2. Weighted-average the 10 offensive stats' z-scores (weight =
#      real_win_rate - 0.5) into one real number per team -- OFF_SCORE.
#      Same for the 5 defensive stats -> DEF_SCORE. A team missing a real
#      value for one stat just doesn't get that term; the average
#      renormalizes over the real weight actually used, so missing data
#      doesn't quietly drag a team's score toward zero.
#   3. Rank all 32 real teams 1-32 on each (1 = best) -> OFF_RANK/DEF_RANK,
#      what actually gets printed on the PDF.
#   4. STRENGTH OF SCHEDULE, from each team's REAL opponents played so far
#      this season (nfl_game_log.csv, already loaded as game_log_df -- no
#      new data pull): OFF_SOS = the average real DEF_SCORE of every
#      opponent faced (how tough the defenses you've actually played have
#      been); DEF_SOS = the average real OFF_SCORE of opponents faced.
#   5. OPPONENT-ADJUSTED SCORE = raw score + SOS -- credit for facing tough
#      opponents and still producing; a raw score built against weak
#      opponents gets pulled back down. This is a single real PASS, not a
#      fully iterative power rating (which would re-solve every team's
#      score against everyone else's ALREADY-adjusted score, repeated to
#      convergence -- the standard SRS/Massey approach) -- deliberately
#      simpler and more transparent for a first real version, especially
#      this early in the season with only 1-2 real games per team so far.
#      A possible real future upgrade once there's more season to work
#      with, not built here.
# ==============================================================================
OFFENSE_QUALITY_STATS = [t for t in NFL_QUALITY_STATS if t[0] in {
    "Success Rate (Offense)", "EPA/Play (Offense)", "Pass Yards",
    "Run Success Rate (Offense)", "Sack % Taken", "Completion %",
    "Pass Block Efficiency (PBE)", "Pressure Rate Allowed", "Rush Yards",
    "3rd Down %",
}]
DEFENSE_QUALITY_STATS = [t for t in NFL_QUALITY_STATS if t[0] in {
    "Interceptions Gained", "EPA/Play Allowed (Defense)", "Rush Yards Allowed",
    "Rush Yards Per Play Allowed (Defense)", "Pressure Rate Forced",
}]
assert len(OFFENSE_QUALITY_STATS) + len(DEFENSE_QUALITY_STATS) == len(NFL_QUALITY_STATS), \
    "every NFL_QUALITY_STATS entry must be classified offense or defense"


def compute_composite_side_scores(team_matrix_df, stat_list):
    """Real weighted z-score composite for one side of the ball (offense or
    defense), across the real full-league team_matrix. Returns a dict
    {team: composite_score}, higher = better on that side regardless of
    each underlying stat's own higher_is_better direction (already folded
    in). A team missing a real value for a given stat just doesn't get that
    stat's term (weights renormalized over what's actually available for
    that team), same honest missing-data handling as
    compute_nfl_quality_consensus."""
    scores = {t: 0.0 for t in team_matrix_df['team']}
    weight_used = {t: 0.0 for t in team_matrix_df['team']}
    for label, matrix_col, higher_is_better, win_rate in stat_list:
        if matrix_col not in team_matrix_df.columns:
            continue
        col = team_matrix_df[matrix_col]
        mean, std = col.mean(), col.std()
        if not std or pd.isna(std) or std == 0:
            continue  # no real spread to rank on for this stat this week
        weight = win_rate - 0.5
        for _, row in team_matrix_df.iterrows():
            val = row.get(matrix_col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            z = (val - mean) / std
            if not higher_is_better:
                z = -z
            scores[row['team']] += z * weight
            weight_used[row['team']] += weight
    return {t: (scores[t] / weight_used[t] if weight_used[t] > 0 else 0.0) for t in scores}


def compute_power_ranks(team_matrix_df):
    """Real OFF_SCORE/DEF_SCORE (composite, continuous) and OFF_RANK/
    DEF_RANK (1-32, 1=best) for every real team in the current
    nfl_team_matrix.csv. Returns a dict keyed by team."""
    off_scores = compute_composite_side_scores(team_matrix_df, OFFENSE_QUALITY_STATS)
    def_scores = compute_composite_side_scores(team_matrix_df, DEFENSE_QUALITY_STATS)
    off_rank_order = sorted(off_scores, key=lambda t: -off_scores[t])
    def_rank_order = sorted(def_scores, key=lambda t: -def_scores[t])
    off_ranks = {t: i + 1 for i, t in enumerate(off_rank_order)}
    def_ranks = {t: i + 1 for i, t in enumerate(def_rank_order)}
    return {
        t: {
            "off_score": off_scores[t], "def_score": def_scores[t],
            "off_rank": off_ranks[t], "def_rank": def_ranks[t],
        }
        for t in off_scores
    }


def compute_sos_and_adjustment(power_ranks, game_log_source_df):
    """Real strength of schedule + a single-pass opponent adjustment, using
    each team's REAL games already played this season (game_log_source_df
    -- same source the rest of the sheet uses, so this is built only from
    real completed games, no projections). OFF_SOS for a team = the average
    real DEF_SCORE of the opponents it has actually played; DEF_SOS = the
    average real OFF_SCORE of opponents faced. A team with no real games
    logged yet gets SOS=0.0 (honestly neutral, not guessed) and its
    adjusted score equals its raw score. Mutates and returns power_ranks
    with 'off_sos'/'def_sos'/'adj_off_score'/'adj_def_score' added."""
    if 'team' in game_log_source_df.columns and 'opponent' in game_log_source_df.columns:
        opponents_by_team = game_log_source_df.groupby('team')['opponent'].apply(list).to_dict()
    else:
        opponents_by_team = {}
    for team, info in power_ranks.items():
        opponents = [o for o in opponents_by_team.get(team, []) if o in power_ranks]
        if opponents:
            off_sos = sum(power_ranks[o]["def_score"] for o in opponents) / len(opponents)
            def_sos = sum(power_ranks[o]["off_score"] for o in opponents) / len(opponents)
        else:
            off_sos, def_sos = 0.0, 0.0
        info["off_sos"] = off_sos
        info["def_sos"] = def_sos
        info["adj_off_score"] = info["off_score"] + off_sos
        info["adj_def_score"] = info["def_score"] + def_sos
    return power_ranks


def _rank_suffix(team, rank_key):
    """" (N)" for the PDF header, or "" (not a guessed number) if this team
    has no real entry in NFL_POWER_RANKS."""
    info = NFL_POWER_RANKS.get(team)
    if not info or rank_key not in info:
        return ""
    return f" ({info[rank_key]})"


print("Step 1e: Computing real composite offense/defense power ranks + strength of schedule...")
NFL_POWER_RANKS = compute_power_ranks(df)
NFL_POWER_RANKS = compute_sos_and_adjustment(NFL_POWER_RANKS, game_log_df)
print(f" -> Real power ranks computed for {len(NFL_POWER_RANKS)} teams.")


def load_nfl_stat_history():
    """Loads the persistent cross-week history from disk. Returns the
    {"games": {}} fallback if the file doesn't exist yet or is unreadable,
    so a first-ever run (or a corrupted file) never crashes the sheet."""
    try:
        with open(NFL_STAT_HISTORY_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "games" not in data:
            return {"games": {}}
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"games": {}}


def save_nfl_stat_history(history):
    """Best-effort write -- a save failure shouldn't take down the sheet."""
    try:
        with open(NFL_STAT_HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
    except OSError as e:
        print(f" -> ⚠ Couldn't save NFL stat history to disk ({e}); this run's newly recorded games won't persist for next time.")


def _nfl_stat_vote(winner_val, loser_val, higher_is_better):
    """+1 if the eventual WINNER led this stat, -1 if the loser led it,
    None if missing or tied (a tie casts no vote either way)."""
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
    """Pairs up nfl_game_log.csv's per-team-per-game rows into real games
    (by game_id when present; falling back to week + sorted team names for
    an older log pulled before game_id was added), and records each
    completed game's per-stat votes, keyed so re-running this script (or
    running it next week) never double-counts an already-recorded game --
    same persistence idea as the CFB project, but reading real per-game
    NFL results directly from your own data pull instead of scraping a
    live scoreboard."""
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
            continue  # tie, unplayed, or bad/duplicate data -- no clean winner to vote on
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

    if new_count:
        save_nfl_stat_history(history)
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
    """Greedy forward-selection, same fixed version as the CFB project:
    keeps adding whichever remaining stat improves combined backtest
    accuracy the most, and stops only when NOTHING left improves it
    further -- it does not stop the moment accuracy clears 80%. Returns
    (included, accuracy, correct, decided, total_games)."""
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


def build_nfl_hybrid_weights(included_backtest_stats):
    """Combines the measured backtest weights with the documented prior
    weights into one list keyed by the nfl_team_matrix.csv column each stat
    lives in (the disparity breakdown reads THIS WEEK's not-yet-played
    matchup from nfl_team_matrix.csv, not the historical game log). A prior
    is skipped if a measured weight already covers that exact stat, so
    nothing gets double-counted."""
    hybrid = []
    measured_matrix_cols = set()
    for label, game_col, matrix_col, higher_is_better, category in NFL_BACKTESTABLE_STATS:
        match = next((row for row in included_backtest_stats if row[1] == game_col), None)
        if match is not None:
            _, _, weight, win_rate, n = match
            hybrid.append({
                'label': label, 'matrix_col': matrix_col, 'higher_is_better': higher_is_better,
                'category': category, 'weight': weight, 'source': 'measured',
                'detail': f"{win_rate*100:.1f}% win rate, {n} real games",
            })
            measured_matrix_cols.add(matrix_col)
    for label, matrix_col, higher_is_better, category, prior_weight in NFL_PRIOR_STATS:
        if matrix_col in measured_matrix_cols:
            continue
        hybrid.append({
            'label': label, 'matrix_col': matrix_col, 'higher_is_better': higher_is_better,
            'category': category, 'weight': prior_weight, 'source': 'prior',
            'detail': "prior estimate, not yet backtested",
        })
    return hybrid


_NFL_CATEGORY_TEMPLATES = {
    "TRENCHES & RUN GAME": "{leader} has the real edge in the trenches on {label} ({a_disp} vs {b_disp}) -- this game may be settled on the ground, with {leader} controlling the line of scrimmage.",
    "PASSING & PROTECTION": "{leader} has the real edge on {label} ({a_disp} vs {b_disp}) -- if {other}'s protection or pass rush can't close this gap, {leader} should get what it wants through the air.",
    "QB ACCURACY & EFFICIENCY": "{leader} has the real edge on {label} ({a_disp} vs {b_disp}) -- worth watching {other}'s QB play closely if this holds up.",
    "OVERALL EFFICIENCY": "{leader} has the real overall efficiency edge on {label} ({a_disp} vs {b_disp}) -- the kind of gap that tends to show up in the final score.",
    "DISRUPTION & TURNOVERS": "{leader} has the real edge on {label} ({a_disp} vs {b_disp}) -- disruption/turnover numbers are noisier than the rest of this breakdown, so treat this one as real but less reliable.",
}


def _nfl_mismatch_sentence(row, t_a, t_b):
    other = t_b if row['leader'] == t_a else t_a
    template = _NFL_CATEGORY_TEMPLATES.get(row['category'], "{leader} has the real edge on {label} ({a_disp} vs {b_disp}).")
    return template.format(leader=row['leader'], other=other, label=row['label'],
                            a_disp=f"{row['val_a']:.1f}", b_disp=f"{row['val_b']:.1f}")


def build_stat_mismatch_breakdown(t_a, t_b, r_a, r_b, hybrid_weights):
    """For THIS matchup: ranks every hybrid-weighted stat by real gap size
    (in real standard deviations, using the same league_avgs/league_stds
    already computed for the sheet's own color-tiering) times how much that
    stat actually matters (its measured-or-prior weight), picks the single
    most telling gap per category, and writes a plain-language sentence
    with the leading team's real number highlighted turquoise -- the whole
    cell. Distinct from the general purple/elite-orange/worst
    cell-tier system elsewhere on the sheet (see the long comment above
    COLOR_RED/COLOR_ORANGE/COLOR_PURPLE) -- turquoise here specifically
    means "the model leans on this stat and it's telling you something.\""""
    scored = []
    for w in hybrid_weights:
        col = w['matrix_col']
        va, vb = r_a.get(col), r_b.get(col)
        if va is None or vb is None or (isinstance(va, float) and pd.isna(va)) or (isinstance(vb, float) and pd.isna(vb)):
            continue
        std = league_stds.get(col) or 0.0
        if std <= 1e-9:
            continue
        gap_z = (va - vb) / std
        if not w['higher_is_better']:
            gap_z = -gap_z  # positive gap_z always means "team A is ahead" from here on
        disparity = abs(gap_z) * max(w['weight'], 0.0)
        if disparity <= 0:
            continue
        leader = t_a if gap_z > 0 else t_b
        scored.append({**w, 'val_a': va, 'val_b': vb, 'gap_z': gap_z, 'disparity': disparity, 'leader': leader})

    if not scored:
        return []

    by_category = {}
    for row in scored:
        cat = row['category']
        if cat not in by_category or row['disparity'] > by_category[cat]['disparity']:
            by_category[cat] = row
    top_rows = sorted(by_category.values(), key=lambda r: r['disparity'], reverse=True)

    els = [full_width_header(f"STAT MISMATCH BREAKDOWN: {t_a} vs {t_b}"), Spacer(1, 1)]
    els.append(build_note(
        "Turquoise = the single biggest real, weighted stat gap in that category. \"measured\" = backtested against "
        "your own nfl_game_log.csv results; \"prior\" = sports-analytics-consensus estimate, not yet backtested."))
    els.append(Spacer(1, 2))

    for row in top_rows:
        cell_a_bg = colors.HexColor(TURQUOISE_HIGHLIGHT) if row['leader'] == t_a else colors.white
        cell_b_bg = colors.HexColor(TURQUOISE_HIGHLIGHT) if row['leader'] == t_b else colors.white
        row_table = Table([[
            Paragraph(f"{row['val_a']:.1f}", ParagraphStyle('mmValA', fontName='Helvetica-Bold', fontSize=9, alignment=1, textColor=colors.HexColor('#000000'))),
            Paragraph(f"<b>{row['category']}</b><br/><font size=7>{row['label']} ({row['source']})</font>",
                      ParagraphStyle('mmLabel', fontName='Helvetica', fontSize=7.5, alignment=1, textColor=colors.white, leading=9)),
            Paragraph(f"{row['val_b']:.1f}", ParagraphStyle('mmValB', fontName='Helvetica-Bold', fontSize=9, alignment=1, textColor=colors.HexColor('#000000'))),
        ]], colWidths=[130, 250, 130])
        row_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), cell_a_bg),
            ('BACKGROUND', (1,0), (1,0), obsidian_black),
            ('BACKGROUND', (2,0), (2,0), cell_b_bg),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#000000')),
        ]))
        sentence_p = Paragraph(f"<font size=7.3 color='#CBD5E0'>{_nfl_mismatch_sentence(row, t_a, t_b)}</font>",
                                ParagraphStyle('mmSentence', fontName='Helvetica', leading=9))
        els.append(row_table)
        els.append(sentence_p)
        els.append(Spacer(1, 3))

    return els

# ==============================================================================
# COORDINATOR-ATTRIBUTED COVERAGE TENDENCIES (nfl_data_pull.py PART 2E)
# nflverse's real coverage/personnel data doesn't update in-season (see
# README.txt), so this is real historical data attributed to each CURRENT
# team's actual OC/DC -- their own numbers from their own last full season
# calling plays, even if that was with a different team. Always shows who
# and which team/season the numbers are actually from, so it's never
# mistaken for this team's current-season tendency. Blank ("--") means a
# true rookie coordinator with no prior playcalling season, or a team not
# yet in nfl_coordinator_mapping.csv -- keep that file updated (see README).
# ==============================================================================

# SPLIT -- this used to be one standalone "COORDINATOR TENDENCIES" panel
# down with SCHEME TENDENCIES, framing each side as "{team} OFF vs
# {opponent} DEF" (this team's OC number paired with the OTHER team's DC
# number in the same cell). Per "move oc tendency to under each team
# offense stats and dc tendency under each teams def stats," OC and DC are
# now SEPARATE strips: OC TENDENCY sits right under TEAM RUNDOWN -- OFFENSE
# (both teams' own OC number side by side, same team-vs-team/own-number
# convention that panel already uses -- not crossed with the opponent's
# DC), and DC TENDENCY sits right under TEAM RUNDOWN -- DEFENSE the same
# way. Still real, still the same nfl_coordinator_mapping.py-sourced
# historical data (see the writeup above), just relocated and reframed to
# match the panel it now lives under instead of crossed.
def _oc_text(row):
    def _f(v, fmt="{:.1f}"):
        return fmt.format(v) if pd.notna(v) else "--"
    label = row.get('oc_source_label')
    if pd.isna(label):
        return "<i>OC vs. man/zone unavailable (rookie coordinator or not yet mapped)</i>"
    n = row.get('oc_sample_plays')
    return (f"OC: <b>{label}</b> ({int(n) if pd.notna(n) else '?'} plays)<br/>"
            f"vs Man: {_f(row.get('oc_epa_vs_man'), '{:.2f}')} EPA/play, {_f(row.get('oc_success_vs_man'), '{:.0f}')}% success"
            f" &ndash; vs Zone: {_f(row.get('oc_epa_vs_zone'), '{:.2f}')} EPA/play, {_f(row.get('oc_success_vs_zone'), '{:.0f}')}% success")

def _dc_text(row):
    def _f(v, fmt="{:.1f}"):
        return fmt.format(v) if pd.notna(v) else "--"
    label = row.get('dc_source_label')
    if pd.isna(label):
        return "<i>DC coverage tendency unavailable (rookie coordinator or not yet mapped)</i>"
    n = row.get('dc_sample_plays')
    return (f"DC: <b>{label}</b> ({int(n) if pd.notna(n) else '?'} plays)<br/>"
            f"Man {_f(row.get('dc_man_pct'), '{:.0f}')}% / Zone {_f(row.get('dc_zone_pct'), '{:.0f}')}%"
            f" &ndash; Cov1 {_f(row.get('dc_cover1_pct'), '{:.0f}')}%,"
            f" Cov2 {_f(row.get('dc_cover2_pct'), '{:.0f}')}%,"
            f" Cov3 {_f(row.get('dc_cover3_pct'), '{:.0f}')}%,"
            f" Cov4 {_f(row.get('dc_cover4_pct'), '{:.0f}')}%")

def build_tendency_strip(label_text, text_a, text_b, widths):
    """A small 2-column (team_a | team_b) text strip with its own thin
    title bar -- used to attach OC TENDENCY under TEAM RUNDOWN -- OFFENSE
    and DC TENDENCY under TEAM RUNDOWN -- DEFENSE, nested in the SAME
    half-width column as the rundown table it belongs to (see
    build_matchup_elements)."""
    header_row = [[Paragraph(label_text, style_sub_hdr), ""]]
    h = Table(header_row, colWidths=widths)
    h.setStyle(TableStyle([
        ('SPAN', (0,0), (-1,0)),
        ('BACKGROUND', (0,0), (-1,-1), obsidian_black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
    ]))
    body_row = [[Paragraph(text_a, style_fw_stat), Paragraph(text_b, style_fw_stat)]]
    b = Table(body_row, colWidths=widths)
    b.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.75, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 1), ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('LEFTPADDING', (0,0), (-1,-1), 4), ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    return [h, b]

# NOTE: build_last_game_panel() (a one-line "X beat Y, 21-14" recap) used to
# live here -- removed now that PREVIOUS SCHEDULE shows every game played,
# including the most recent one, so the standalone recap was pure
# duplication. last_game_* fields are still real, still pulled, and still
# used by build_deep_dive_commentary() below for its prose summary.

# ==============================================================================
# THE CAPTAIN -- QB section, page 2. Now a proper stat GRID (like TEAM
# RUNDOWN / MATCHUP_PAIRS) instead of prose paragraphs, with the real
# per-player usage list KEPT underneath it.
#
# Header sub-line (restored -- this was the old QB PROFILE panel's line 1,
# lost when THE CAPTAIN went prose-only): starting QB name, our own
# qb_mobility_tag ("Pocket"/"Mobile" -- OUR threshold, rush_yards_pg >= 15,
# not an official league classification, documented as such), rush yd/gm
# (+ att/gm), and NGS time to throw -- sitting centered over each team's own
# grid column, same mechanism TEAM RUNDOWN's team_a_label/team_b_label uses.
#
# Grid rows: CPOE and PASSER RATING first, centered (same-key, no real
# opposing-defense counterpart -- each team's own number shown side by
# side, same convention as ADVANCED_PAIRS/RESIDUAL_OVERALL_PAIRS elsewhere
# on this sheet). Then the real offense-vs-opposing-defense crossed pairs:
# true completion %, sack %, yards/completion, interceptions/gm -- the same
# real fields the old prose version showed, now with their real defensive
# counterpart alongside instead of standing alone.
#
# Then TUG O' WAR (Interior) / EDGE PRESSURE -- MOVED here (not duplicated)
# from the old standalone PRESSURE INDEX panel ("under qb add the interior
# menace and edge pressure formula we did"). Framing unchanged (each team's
# own defensive front, {t_a} DEF vs {t_b} DEF) -- it lives here now because
# it's real context for how much pressure each of these QBs is going to
# face, not because the stat itself changed meaning.
#
# On-target % (QB accuracy on catchable throws) still has no free
# substitute -- PFR-exclusive manual charting, same wall as drop % below in
# THE FREEBIRDS; see README.txt "LICENSING". Asked about directly and
# confirmed absent, not overlooked.
# ==============================================================================
CAPTAIN_PAIRS = [
    ('cpoe', 'cpoe', "{:.1f}", "%", 1, "CPOE (COMP % OVER EXPECTED)"),
    ('open_source_passer_rating', 'open_source_passer_rating', "{:.1f}", "", 1, "PASSER RATING"),
    ('completion_pct', 'completion_pct_allowed', "{:.1f}", "%", 1, "TRUE COMPLETION %"),
    ('sack_pct_offense', 'sack_pct_defense', "{:.1f}", "%", 1, "SACK %"),
    ('yards_per_completion', 'yards_per_completion_allowed', "{:.1f}", "", 1, "YARDS / COMPLETION"),
    ('interceptions_thrown_per_game', 'interceptions_gained_per_game', "{:.1f}", "", 1, "INTERCEPTIONS / GM"),
]

def build_captain_panel(t_a, t_b, row_a, row_b, players_df, dvp_df):
    def _f(v, fmt="{:.1f}"):
        return fmt.format(v) if pd.notna(v) else "--"

    def qb_subhdr_text(row):
        name = row.get('starting_qb_name')
        name = name if pd.notna(name) else "No QB data"
        tag = row.get('qb_mobility_tag')
        tag = tag if pd.notna(tag) else "--"
        return (f"<b>{name}</b> ({tag})<br/>"
                f"{_f(row.get('qb_rush_yards_pg'))} rush yd/gm ({_f(row.get('qb_rush_attempts_pg'))} att/gm)"
                f" &ndash; TT: {_f(row.get('ngs_time_to_throw'), '{:.2f}')}s")

    def qb_fantasy_para(team, opponent):
        if players_df is None or players_df.empty:
            return Paragraph("No QB usage data available", style_fw_stat)
        team_rows = players_df[(players_df['team'] == team) & (players_df['position'] == 'QB')] \
            .sort_values('ppr_pts_pg', ascending=False).head(1)
        if team_rows.empty:
            return Paragraph("No QB usage data available", style_fw_stat)
        r = team_rows.iloc[0]
        matchup_str = _dvp_rank_text(dvp_df, opponent, 'QB')
        return Paragraph(
            f"{_f(r.get('total_tds_pg'))} total TD/gm<br/>"
            f"PPR: {_f(r.get('ppr_pts_pg'))} pts/gm (last 3: {_f(r.get('ppr_pts_last3_pg'))})"
            f" &ndash; vs this defense: {matchup_str}",
            style_fw_stat)

    # Title bar + QB name/tag/rush/TT sub-header row, same 2-row header
    # table technique as build_rundown_panel()'s team_a_label/team_b_label.
    header_rows = [
        [Paragraph(f"THE CAPTAIN: {t_a} vs {t_b}", style_sub_hdr), "", ""],
        [Paragraph(qb_subhdr_text(row_a), style_captain_subhdr), "",
         Paragraph(qb_subhdr_text(row_b), style_captain_subhdr)],
    ]
    h_table = Table(header_rows, colWidths=panel_widths)
    h_table.setStyle(TableStyle([
        ('SPAN', (0,0), (-1,0)),
        ('BACKGROUND', (0,0), (-1,-1), obsidian_black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))

    # SPLIT -- CPOE and PASSER RATING are same-key rows (both teams' own
    # real value already sits side by side in one row), so there's no
    # genuine "reverse direction" to show for those -- same for Tug o'
    # War/Edge Pressure below (each team's OWN defensive front, nothing
    # tied to a specific offense). Those stay in ONE shared grid, shown
    # once. TRUE COMPLETION %, SACK %, YARDS/COMPLETION and
    # INTERCEPTIONS/GM ARE genuinely crossed (this team's offense vs the
    # OTHER team's defense) -- before, this grid only ever showed "{t_a}
    # OFF vs {t_b} DEF" for those. Now split into two side-by-side panels,
    # same away_side/home_side pattern as OFFENSE vs DEFENSE/THE DIRTY
    # BIRD/SPECIAL TEAMS BLOCKS elsewhere on the sheet, so BOTH directions
    # show.
    same_pairs = [p for p in CAPTAIN_PAIRS if p[0] == p[1]]
    crossed_pairs = [p for p in CAPTAIN_PAIRS if p[0] != p[1]]

    matrix, bg_rows = _grid_from_pairs(same_pairs, row_a, row_b)
    # Tug o' War (Interior) / Edge Pressure -- MOVED here from the old
    # standalone PRESSURE INDEX panel, see comment above.
    row, bg = _pressure_row_cells(row_a, row_b, 'interior_pressure_rate', 'interior_pressure_pct',
                                   'interior_pressure_note', "TUG O' WAR<br/>(INTERIOR, DEF)")
    matrix.append(row); bg_rows.append(bg)
    row, bg = _pressure_row_cells(row_a, row_b, 'edge_pressure_rate', 'edge_pressure_pct',
                                   'edge_pressure_note', "EDGE PRESSURE<br/>(DEF)")
    matrix.append(row); bg_rows.append(bg)

    shared_grid = build_centered_table(matrix, panel_widths, bg_colors=bg_rows, border_color='#000000')

    # EXPLAINER. Both are real credited-pressure rates (interior vs. edge
    # rushers, from real nflverse pressure charting) for THIS team's own
    # defensive front -- not tied to a specific opposing offense, which is
    # why they sit in the shared grid above rather than a crossed split. The
    # bracket word (ELITE/ABOVE AVG/AVERAGE/BELOW AVG/WEAK) next to each
    # number is that team's real percentile rank vs. a real historical
    # multi-season reference pool of NFL defenses, not vs. just this week's
    # 32 teams.
    pressure_note = build_note(
        "TUG O' WAR (INTERIOR) &amp; EDGE PRESSURE = real credited pressure rate for this team's D-line/edge rushers (nflverse pressure data), shown as this team's own front -- not matched to a specific opposing O-line. "
        "Real league range this season: interior ~2%-6% (avg ~4%), edge ~3%-8% (avg ~5%). "
        "ELITE/ABOVE AVG/AVERAGE/BELOW AVG/WEAK = real percentile rank vs. a historical multi-season reference pool, not just this week's 32 teams.")

    captain_away = build_stat_panel(f"{t_a} OFFENSE  vs  {t_b} DEFENSE", crossed_pairs, row_a, row_b, widths=side_widths)
    captain_home = build_stat_panel(f"{t_b} OFFENSE  vs  {t_a} DEFENSE", crossed_pairs, row_b, row_a, widths=side_widths)
    captain_split = Table([[captain_away, captain_home]], colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    captain_split.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))

    # Real per-player QB usage line, kept underneath the grid -- same
    # real source as before, same two-column layout every other position
    # panel's player list uses.
    list_row = [[qb_fantasy_para(t_a, t_b), qb_fantasy_para(t_b, t_a)]]
    list_table = Table(list_row, colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    list_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.75, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))

    return [h_table, Spacer(1, 1), shared_grid, pressure_note, Spacer(1, 2), captain_split, Spacer(1, 2), list_table]

# ==============================================================================
# NET EPA PER MATCHUP -- your exact formula: (OFF EPA x 1.5) - DEF EPA,
# computed live for THIS specific matchup (each team's offense against the
# opponent's actual defense), not a generic team-level number.
#
# UNUSED as of your "move them under league rank" ask -- this used to be
# its own standalone panel; both its rows (NET EPA and SUCCESS RATE
# (OFFENSE)) now render inside the VOID RATING panel instead (see
# net_epa_matchup in VOID_PAIRS and its note). Kept here, not called
# anywhere, purely for reference/in case you ever want the standalone
# panel back -- same pattern as SCHEME_TENDENCY_PAIRS above.
# ==============================================================================

def build_net_epa_panel(t_a, t_b, row_a, row_b):
    net_a = row_a['off_efficiency_epa'] * 1.5 - row_b['def_efficiency_epa']
    net_b = row_b['off_efficiency_epa'] * 1.5 - row_a['def_efficiency_epa']
    # NEW: SUCCESS RATE (OFFENSE) row, added right under NET EPA -- this
    # is off_success_rate, the exact real stat behind the #1 entry in
    # NFL_QUALITY_STATS (63.4% real walk-forward win rate, the single best
    # individual predictor of the quality stats/badge, now a 15-stat list
    # after the "trenches" round refresh). It was computed and used
    # everywhere already (the model, the badge) but never actually shown
    # to you as its own labeled number anywhere on the sheet -- this is
    # that missing label. Deliberately the DIRECT team-vs-team read (this
    # team's own off_success_rate next to the opponent's own
    # off_success_rate), same as how compute_nfl_quality_consensus
    # actually compares it for the badge -- NOT netted against the
    # opponent's defense the way NET EPA above is, since Success Rate
    # Allowed (Defense) is one of the 5 stats that did NOT clear the real
    # 55% quality bar (51.8%), so pairing offense against defense-allowed
    # here the way the EPA row does would mix a proven stat with an
    # unproven one. success_a/b are already 0-100 (see nfl_data_pull.py --
    # off_success_rate = off_success_rate_pct * 100).
    success_a = row_a.get('off_success_rate')
    success_b = row_b.get('off_success_rate')
    has_success = pd.notna(success_a) and pd.notna(success_b)
    row_data = [[
        Paragraph(f"{net_a:.3f}", style_val_pop if net_a >= net_b else style_val_base),
        Paragraph("NET EPA ((OFF EPA x1.5) - DEF EPA)", style_label),
        Paragraph(f"{net_b:.3f}", style_val_pop if net_b > net_a else style_val_base),
    ]]
    bg_rows = [[None, center_grey, None]]
    if has_success:
        row_data.append([
            Paragraph(f"{success_a:.1f}%", style_val_pop if success_a >= success_b else style_val_base),
            Paragraph("SUCCESS RATE (OFFENSE)", style_label),
            Paragraph(f"{success_b:.1f}%", style_val_pop if success_b > success_a else style_val_base),
        ])
        bg_rows.append([None, center_grey, None])
    t = build_centered_table(row_data, panel_widths, bg_colors=bg_rows, border_color='#000000')
    # EXPLAINER. EPA = Expected Points Added/play, real nflverse
    # play-by-play data -- how much a play shifts a team's expected scoring
    # vs. league average (positive helps, negative hurts). Real per-team
    # range this season is roughly -0.34 to +0.41 per play (league avg
    # ~0.01) -- see off_efficiency_epa/def_efficiency_epa in
    # nfl_team_matrix.csv. NET EPA above is this team's OWN off EPA/play
    # (weighted x1.5, your formula) minus the OPPONENT's def EPA/play
    # allowed -- positive favors this team, negative the opponent, 0 = even.
    # The same real off/def EPA/play numbers (unweighted) also drive NET
    # EFFICIENCY (EPA/PLAY) elsewhere on this sheet (GAME PLAN, TEAM
    # RUNDOWN, OFFENSE vs DEFENSE) -- same metric, same scale, just not
    # multiplied by 1.5 or netted against an opponent there. SUCCESS RATE
    # (OFFENSE) = real share of this team's plays with EPA > 0 (same
    # "success" flag VOID and RUN SUCCESS RATE already use) -- higher is
    # better, and this is the real single strongest predictor of the quality
    # stats behind the badge (now 15). Trimmed to stay terse -- page 1's
    # pagination budget is already tight (see the OC/DC strip fix and VOID
    # note comment above).
    note = build_note(
        "EPA = Expected Points Added/play, real nflverse data (range ~-0.34 to +0.41, avg ~0.01). "
        "NET EPA = this team's off EPA/play (x1.5) minus opponent's def EPA/play allowed; same numbers drive NET EFFICIENCY elsewhere, unweighted. "
        "SUCCESS RATE (OFFENSE) = real share of plays with EPA > 0 -- the single best real predictor behind the quality-stat badge (63.4% real win rate).")
    return [full_width_header("NET EPA PER MATCHUP"), Spacer(1, 1), t, note]

def build_prediction_panel(t_a, t_b, pred):
    win_prob_a_pct = pred['win_prob_a'] * 100
    win_prob_b_pct = 100 - win_prob_a_pct

    if pred['margin'] > 0.05:
        spread_text = f"{t_a} by {pred['margin']:.1f}"
    elif pred['margin'] < -0.05:
        spread_text = f"{t_b} by {-pred['margin']:.1f}"
    else:
        spread_text = "Pick'em"

    stat_rows = [[
        Paragraph(f"{win_prob_a_pct:.0f}%", style_val_pop if win_prob_a_pct >= win_prob_b_pct else style_val_base),
        Paragraph("WIN PROBABILITY", style_label),
        Paragraph(f"{win_prob_b_pct:.0f}%", style_val_pop if win_prob_b_pct > win_prob_a_pct else style_val_base),
    ], [
        Paragraph(f"{pred['exp_pts_a']:.1f}", style_val_base),
        Paragraph("PROJECTED TEAM TOTAL", style_label),
        Paragraph(f"{pred['exp_pts_b']:.1f}", style_val_base),
    ]]
    stat_table = build_centered_table(
        stat_rows, panel_widths,
        bg_colors=[[None, center_grey, None], [None, center_grey, None]],
        border_color='#000000'
    )

    footer_row = [[Paragraph(f"MODEL SPREAD: {spread_text}   |   MODEL TOTAL: {pred['total']:.1f}", style_label)]]
    footer_table = Table(footer_row, colWidths=[sum(panel_widths)])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), center_grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.75, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))

    return [full_width_header(f"MODEL PREDICTION: {t_a} vs {t_b}"), Spacer(1, 1), stat_table, Spacer(1, 1), footer_table]

def build_injury_panel(t_a, t_b, inj_df):
    def team_injury_lines(team):
        team_rows = inj_df[inj_df['Team'] == team]
        if team_rows.empty:
            return [Paragraph("No significant injuries reported", style_val_base)]
        return [Paragraph(f"{r['Player']} ({r['Position']}) &ndash; {r['Injury']} &ndash; {r['Status']}", style_val_base)
                for _, r in team_rows.iterrows()]

    body_row = [[team_injury_lines(t_a), team_injury_lines(t_b)]]
    body_table = Table(body_row, colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    body_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.75, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    return [full_width_header(f"INJURY REPORT: {t_a} vs {t_b}", widths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH]), Spacer(1, 1), body_table]

# ==============================================================================
# POSITION BREAKDOWN PANELS -- THE CAPTAIN (QB, above) / THE DAWGS (RB)
# /THE FREEBIRDS (WR) / THE TANKS (TE). Replaces the old single FANTASY
# /PROP WATCH panel (one mixed-position top-4-by-PPR list) with one themed
# panel per position -- same real source data (nfl_player_stats.csv +
# nfl_def_vs_position.csv, both real nflverse data, nothing fabricated; an
# empty pull says so plainly instead of showing blank/fake rows), just
# organized and named by position. Each panel shows the real
# per-team top-N players at that position by PPR pts/gm, their usage
# (targets/share or carries + snap %), PPR production (season + last-3-game
# trend), and how many PPR points the OPPONENT's defense has allowed that
# position this season (1-32 matchup rank, from dvp_df -- 32 = stingiest, 1
# = most generous).
# ==============================================================================

POSITION_PANEL_THEMES = {
    'RB': ('THE DAWGS', 3),
    'WR': ('THE FREEBIRDS', 4),
    'TE': ('THE TANKS', 2),
}

def _dvp_rank_text(dvp_df, opponent_team, position):
    if dvp_df is None or dvp_df.empty:
        return "--"
    row = dvp_df[(dvp_df['team'] == opponent_team) & (dvp_df['position'] == position)]
    if row.empty:
        return "--"
    rank = int(row.iloc[0]['rank_worst_matchup_1_to_32'])
    fpts = row.iloc[0]['fpts_allowed_pg']
    return f"#{rank}/32 ({fpts:.1f} PPR pts/gm allowed to {position}s)"

# NEW: PROJ THIS WEEK (following "is this a prediction?" -- honest answer
# at the time was no, these panels are all real backward-looking
# production/usage, not a computed number for the upcoming game). This is
# that missing number, added WITHOUT touching target share/usage/snap
# %/last-3 trend, which stay exactly as they were.
#
# Real, transparent formula, NOT a backtested one (unlike the quality-stat
# badge or PROJ TOTAL's underlying model, which at least has real walk-
# forward numbers behind it) -- there's no real per-player historical log
# to backtest a fantasy projection against yet, the same honest gap noted
# when this was first asked about. Treat this as a real-data starting
# formula, not a proven one:
#   baseline  = 50/50 blend of this player's real season PPR pts/gm and
#               real last-3-game PPR pts/gm (equal weight -- recent form
#               matters, but a 3-game sample is noisy, so it's not
#               overweighted without evidence that it should be)
#   matchup   = this specific opponent's real PPR pts/gm allowed to the
#               position, divided by the real 32-team league-average PPR
#               pts/gm allowed to that position this season (>1 = soft
#               matchup, <1 = tough one), clipped to [0.6, 1.6] so a small
#               early-season DVP sample can't swing a player's projection
#               wildly on noise
#   proj      = baseline x matchup
def _project_fantasy_pts(player_row, dvp_df, opponent_team, position):
    season_avg = player_row.get('ppr_pts_pg')
    last3_avg = player_row.get('ppr_pts_last3_pg')
    if pd.isna(season_avg):
        return None
    baseline = season_avg if pd.isna(last3_avg) else (0.5 * season_avg + 0.5 * last3_avg)
    if dvp_df is None or dvp_df.empty:
        return baseline
    pos_rows = dvp_df[dvp_df['position'] == position]
    opp_row = pos_rows[pos_rows['team'] == opponent_team]
    if opp_row.empty or pos_rows.empty:
        return baseline
    league_avg_allowed = pos_rows['fpts_allowed_pg'].mean()
    opp_allowed = opp_row.iloc[0]['fpts_allowed_pg']
    if pd.isna(league_avg_allowed) or league_avg_allowed == 0 or pd.isna(opp_allowed):
        return baseline
    matchup_mult = max(0.6, min(1.6, opp_allowed / league_avg_allowed))
    return baseline * matchup_mult

def _position_player_rows(players_df, dvp_df, team, opponent, position, top_n):
    if players_df is None or players_df.empty:
        return [Paragraph(f"No {position} usage data available", style_fw_stat)]
    team_rows = players_df[(players_df['team'] == team) & (players_df['position'] == position)] \
        .sort_values('ppr_pts_pg', ascending=False).head(top_n)
    if team_rows.empty:
        return [Paragraph(f"No {position} usage data available", style_fw_stat)]
    out = []
    for _, r in team_rows.iterrows():
        usage_bits = []
        if r.get('targets_pg', 0) > 0.3:
            usage_bits.append(f"{r['targets_pg']:.1f} tgt/gm ({r.get('target_share_pct', 0):.0f}% share)")
        if r.get('carries_pg', 0) > 0.3:
            usage_bits.append(f"{r['carries_pg']:.1f} car/gm")
        usage_str = ", ".join(usage_bits) if usage_bits else "limited usage"
        snap = r.get('snap_pct_avg', np.nan)
        snap_str = f"{snap:.0f}% snaps" if pd.notna(snap) else "-- snaps"
        matchup_str = _dvp_rank_text(dvp_df, opponent, position)
        proj_pts = _project_fantasy_pts(r, dvp_df, opponent, position)
        proj_line = f"<br/><font color=\"#94A3B8\">PROJ THIS WEEK: {proj_pts:.1f} PPR pts</font>" if proj_pts is not None else ""
        out.append(Paragraph(
            f"<b>{r['player']}</b> &ndash; {usage_str}, {snap_str}<br/>"
            f"PPR: {r['ppr_pts_pg']:.1f} pts/gm (last 3: {r.get('ppr_pts_last3_pg', float('nan')):.1f})"
            f" &ndash; vs this defense: {matchup_str}"
            f"{proj_line}",
            style_fw_stat))
    return out

def build_position_panel(theme_title, position, t_a, t_b, players_df, dvp_df, top_n):
    """Used for THE TANKS (TE) only now -- THE DAWGS (RB) and THE FREEBIRDS
    (WR) below got their own stat grids; TE didn't, so this stays
    the plain real-player-list panel for that one position."""
    body_row = [[
        _position_player_rows(players_df, dvp_df, t_a, t_b, position, top_n),
        _position_player_rows(players_df, dvp_df, t_b, t_a, position, top_n),
    ]]
    body_table = Table(body_row, colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    body_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.75, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    return [full_width_header(f"{theme_title}: {t_a} vs {t_b}", widths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH]), Spacer(1, 1), body_table]

def _player_list_table(t_a, t_b, position, players_df, dvp_df, top_n):
    """The real per-player usage list (name, usage, snaps, PPR + last-3
    trend, opponent matchup rank) -- shared body used under THE FREEBIRDS/
    DAWGS grids below, same layout build_position_panel already used."""
    body_row = [[
        _position_player_rows(players_df, dvp_df, t_a, t_b, position, top_n),
        _position_player_rows(players_df, dvp_df, t_b, t_a, position, top_n),
    ]]
    body_table = Table(body_row, colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    body_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.75, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    return body_table

def _grid_from_pairs(pairs, row_a, row_b, widths=panel_widths):
    matrix, bg_rows = [], []
    for row_def in pairs:
        left_key, right_key, fmt, suffix, mult, label = row_def
        row, bg = _pair_row_cells(row_a, row_b, left_key, right_key, fmt, suffix, mult, label)
        matrix.append(row)
        bg_rows.append(bg)
    return matrix, bg_rows

# ==============================================================================
# THE FREEBIRDS -- WR section, now a stat grid (like THE CAPTAIN above)
# instead of just the real player list, with that player list KEPT
# underneath.
#
# Real offense-vs-opposing-defense crossed pairs: passing yards/game, pass
# yards/play, target separation, YAC/reception (the crossed pbp-sourced
# pair, not the same-key team_yac_per_reception used elsewhere -- see the
# comment above SKILL_BATTLE_PAIRS for why those two are kept deliberately
# separate), and pressure rate allowed (this team's protection) vs pressure
# rate forced (the opponent's pass rush).
#
# DROP % -- it's genuinely not here: no free/CC-
# licensed data source exists for it (PFR-exclusive manual charting only),
# same wall documented under ADVANCED_PAIRS/README.txt "LICENSING". Shown
# as a plain note under the grid instead of a silent gap, so it reads as
# "not available" rather than "forgotten."
#
# PASS BLOCK (PBE) vs PASS RUSH (PRP) -- MOVED here (not duplicated) from
# the old standalone "TRENCH BATTLE" panel.
#
# SECONDARY LOCK INDEX -- MOVED here (not duplicated) from its own old
# standalone panel -- same anti-duplication call as PBE/PRP above.
# Same-key, DEF vs DEF framing preserved (no offensive mirror for a
# coverage-tightness index -- see the original writeup above
# SECONDARY_LOCK_PAIR).
# ==============================================================================
FREEBIRDS_PAIRS = [
    ('passing_yards_per_game', 'passing_yards_allowed_per_game', "{:.1f}", "", 1, "PASSING YARDS / GAME"),
    ('off_yards_per_pass', 'def_yards_per_pass', "{:.1f}", "", 1, "PASS YARDS / PLAY"),
    ('target_separation', 'target_separation_allowed', "{:.1f}", "", 1, "TARGET SEPARATION (YDS)"),
    ('yac_per_reception_pbp', 'yac_allowed_per_reception_pbp', "{:.1f}", "", 1, "YAC / RECEPTION"),
    ('pressure_rate_allowed_pct', 'pressure_rate_forced_pct', "{:.1f}", "%", 1, "PRESSURE RATE (ALLOWED vs FORCED)"),
]

def build_freebirds_panel(t_a, t_b, row_a, row_b, players_df, dvp_df):
    # SPLIT -- SECONDARY LOCK INDEX is same-key, DEF-vs-DEF (no offense tied
    # to it, no reverse direction). PASS BLOCK (PBE) vs PASS RUSH (PRP)
    # already computes BOTH crossed directions INSIDE one row -- its left
    # cell is {t_a} PBE vs {t_b} PRP, its right cell is {t_b} PBE vs {t_a}
    # PRP (see _pbe_prp_row_cells above) -- so there's no "other direction"
    # left to add for that one either. Both stay in ONE shared grid, shown
    # once. FREEBIRDS_PAIRS (passing yds/gm, pass yds/play, target
    # separation, YAC/reception, pressure rate allowed vs forced) ARE
    # one-directional crossed pairs -- before, this grid only ever showed
    # "{t_a} OFF vs {t_b} DEF" for those. Now split into two side-by-side
    # panels, same pattern as OFFENSE vs DEFENSE/THE DIRTY BIRD/SPECIAL
    # TEAMS BLOCKS elsewhere on the sheet, so BOTH directions show.
    matrix, bg_rows = _grid_from_pairs(SECONDARY_LOCK_PAIR, row_a, row_b)
    row, bg = _pbe_prp_row_cells(t_a, row_a, t_b, row_b)
    matrix.append(row); bg_rows.append(bg)
    shared_grid = build_centered_table(matrix, panel_widths, bg_colors=bg_rows, border_color='#000000')

    # EXPLAINERS. PBE and PRP are real but on DIFFERENT scales (PBE ~0-100
    # index, PRP a ~3%-34% pressure rate), so they're never subtracted
    # directly -- each is first converted to a z-score (how many std devs
    # from the real league mean) and then netted; that net decides LEANS
    # O-LINE / LEANS PASS RUSH / EVEN. SECONDARY LOCK INDEX has no offensive
    # mirror by design (it's a pure coverage-tightness read on this team's
    # own defense).
    pbe_prp_note = build_note(
        "PASS BLOCK (PBE) vs PASS RUSH (PRP) = real indices on different scales (PBE ~0-100, real range ~66-97; PRP a real pressure rate, ~3%-34%) -- compared as z-scores (std devs from real league avg), not raw numbers, to decide LEANS O-LINE / LEANS PASS RUSH / EVEN.")
    secondary_note = build_note(
        "SECONDARY LOCK INDEX = 100 minus real receiving yards allowed per real target faced (nflverse) -- higher = tighter coverage. Real league range this season ~88-96 (avg ~94). Defense-only: no offensive mirror stat exists for coverage tightness.")

    freebirds_away = build_stat_panel(f"{t_a} OFFENSE  vs  {t_b} DEFENSE", FREEBIRDS_PAIRS, row_a, row_b, widths=side_widths)
    freebirds_home = build_stat_panel(f"{t_b} OFFENSE  vs  {t_a} DEFENSE", FREEBIRDS_PAIRS, row_b, row_a, widths=side_widths)
    freebirds_split = Table([[freebirds_away, freebirds_home]], colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    freebirds_split.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))

    note_table = build_note(
        "DROP % &ndash; not shown: no free/CC-licensed data source exists (PFR-exclusive manual charting only). See README.txt “LICENSING”.")

    list_table = _player_list_table(t_a, t_b, 'WR', players_df, dvp_df, POSITION_PANEL_THEMES['WR'][1])

    return [full_width_header(f"THE FREEBIRDS: {t_a} vs {t_b}"), Spacer(1, 1), shared_grid,
            secondary_note, pbe_prp_note, Spacer(1, 2),
            freebirds_split, note_table, Spacer(1, 2), list_table]

# ==============================================================================
# THE DAWGS -- RB section, now a stat grid (like THE CAPTAIN/FREEBIRDS
# above) instead of just the real player list, with that player list
# KEPT underneath.
#
# Real offense-vs-opposing-defense crossed pairs: rushing yards/game, rush
# yards/play, RYOE/rush attempt (real NGS rush-yards-over-expected -- MOVED
# here, not duplicated, from the old standalone ADVANCED METRICS panel now
# that it's gone), stuff rate, fumbles (lost vs forced), TFL (allowed vs
# made), and RUN SUCCESS RATE vs RUN STOP RATE -- MOVED here (not
# duplicated) from the old standalone "TRENCH BATTLE" panel's one
# TRENCH_BATTLE_PAIRS row, same relocation logic as PBE/PRP moving to THE
# FREEBIRDS -- both halves of that old panel now live with the position
# group they're most relevant to, so the standalone panel itself is retired
# rather than left half-full.
#
# "Anything I've missed" -- I looked (rush_attempts_per_game/_allowed is the
# one other real, unused-elsewhere column that fit), but raw rush-attempt
# volume is genuinely ambiguous good/bad for a defense (more carries faced
# can mean a leaky run defense OR just an opponent running out the clock
# with a lead -- same kind of ambiguity this sheet already leaves uncolored
# for miscue_rate's defense side/PASS RATE FACED), so coloring it green/red
# either way would be asserting a signal that isn't really there. Left out
# rather than shown with a misleading direction.
# ==============================================================================
DAWGS_PAIRS = [
    ('rushing_yards_per_game', 'rushing_yards_allowed_per_game', "{:.1f}", "", 1, "RUSHING YARDS / GAME"),
    ('off_rushing_yards_per_play', 'def_rushing_yards_per_play', "{:.1f}", "", 1, "RUSH YARDS / PLAY"),
    ('ryoe_per_attempt', 'ryoe_per_attempt', "{:.2f}", "", 1, "RYOE / RUSH ATTEMPT"),
    # RUSH EPA/PLAY -- NEW. Real EPA/play (nfl_data_pull.py PART 2, same
    # 'epa' column as EPA/PLAY (OFF) elsewhere, filtered to rushing plays
    # only). def_rush_epa_allowed is in lower_is_better_fields (a defense
    # wants LESS EPA/play allowed), same convention as def_efficiency_epa.
    ('off_rush_epa', 'def_rush_epa_allowed', "{:.2f}", "", 1, "RUSH EPA/PLAY (OFF vs DEF ALLOWED)"),
    # LINE YARDS -- NEW. Real Adjusted Line Yards (nfl_data_pull.py,
    # standard Football Outsiders-style credit formula off real pbp rush
    # plays, QB scrambles excluded) -- CFB's equivalent (Off/Def Line Yards)
    # comes straight from CFBD's API as a precomputed field; nflverse has no
    # free equivalent, so this is built from real play-by-play instead of
    # read off a source.
    ('off_line_yards', 'def_line_yards_allowed', "{:.2f}", "", 1, "LINE YARDS (OFF vs DEF ALLOWED)"),
    ('stuff_rate_offense', 'stuff_rate_defense', "{:.1f}", "%", 1, "STUFF RATE"),
    ('fumbles_lost_per_game', 'opponent_fumbles_forced_per_game', "{:.1f}", "", 1, "FUMBLES (LOST vs FORCED)"),
    ('tfl_allowed', 'tfl_made', "{:.1f}", "", 1, "TFL (ALLOWED vs MADE)"),
] + TRENCH_BATTLE_PAIRS  # the RUN SUCCESS RATE vs RUN STOP RATE row, moved here (see comment above)

def build_dawgs_panel(t_a, t_b, row_a, row_b, players_df, dvp_df):
    # SPLIT -- RYOE/rush attempt is same-key (both teams' own real NGS
    # number already sits side by side in one row), no reverse direction,
    # shown once, shared. The rest of DAWGS_PAIRS (rushing yds/gm, rush
    # yds/play, stuff rate, fumbles, TFL, and the RUN SUCCESS RATE vs RUN
    # STOP RATE row folded in from the old TRENCH BATTLE panel) ARE crossed
    # -- before, this grid only ever showed "{t_a} OFF vs {t_b} DEF" for
    # those. Now split into two side-by-side panels, same pattern as OFFENSE
    # vs DEFENSE/THE DIRTY BIRD/SPECIAL TEAMS BLOCKS elsewhere on the sheet,
    # so BOTH directions show.
    same_pairs = [p for p in DAWGS_PAIRS if p[0] == p[1]]
    crossed_pairs = [p for p in DAWGS_PAIRS if p[0] != p[1]]

    matrix, bg_rows = _grid_from_pairs(same_pairs, row_a, row_b)
    shared_grid = build_centered_table(matrix, panel_widths, bg_colors=bg_rows, border_color='#000000')

    dawgs_away = build_stat_panel(f"{t_a} OFFENSE  vs  {t_b} DEFENSE", crossed_pairs, row_a, row_b, widths=side_widths)
    dawgs_home = build_stat_panel(f"{t_b} OFFENSE  vs  {t_a} DEFENSE", crossed_pairs, row_b, row_a, widths=side_widths)
    dawgs_split = Table([[dawgs_away, dawgs_home]], colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    dawgs_split.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))

    list_table = _player_list_table(t_a, t_b, 'RB', players_df, dvp_df, POSITION_PANEL_THEMES['RB'][1])
    return [full_width_header(f"THE DAWGS: {t_a} vs {t_b}"), Spacer(1, 1), shared_grid, Spacer(1, 2),
            dawgs_split, Spacer(1, 2), list_table]

# ==============================================================================
# PREMIUM METRICS PANEL -- PARTIALLY RESTORED (real, not PFF's)
# This used to read nfl_premium_team_manual.csv (DVOA, pass/run block & pass
# rush/run stop win rates, double-team rate, interior/edge pressure rate --
# all fill-in-by-hand PFF+/FTN numbers) and show whatever had been entered.
# Removed entirely -- see README.txt "LICENSING -- WHAT WAS REMOVED AND
# WHY". DVOA's role is already covered by VOID for free. Interior/edge
# pressure rate now has a real substitute below -- "Tug o' War" (interior,
# NT/DT) and "Edge Pressure" (DE/OLB), built in nfl_data_pull.py PART 2F
# from player-attributed sacks/half-sacks/QB hits, tracked to the actual
# players on each 2026 roster. It's an honestly narrower stat than PFF's
# (credited pressure only, no hurries -- see PART 2F's own writeup for why
# that's not available anywhere, free or paid, at the player level), not a
# reproduction of their proprietary number. Win-rate/double-team/YPRR/juke
# rate still have no free substitute and stay gone.
#
# TIER SCALE (our own cutoffs, not anyone's official grading scale):
# percentile is against a real historical league-wide reference pool (every
# current team's own last few real seasons) -- ELITE >=85th pctile, ABOVE
# AVG 65-85, AVERAGE 35-65, BELOW AVG 15-35, WEAK <15th.
# ==============================================================================

PRESSURE_TIERS = [   # (percentile cutoff, label, color) -- checked top-down, first match wins
    (85, 'ELITE',      colors.HexColor('#9B59B6')),
    (65, 'ABOVE AVG',  colors.HexColor('#2ECC71')),
    (35, 'AVERAGE',    colors.HexColor('#FFFFFF')),
    (15, 'BELOW AVG',  colors.HexColor('#F5A623')),
    (0,  'WEAK',       colors.HexColor('#FF4D4D')),
]

def _pressure_tier(pct):
    """Returns (label, color) for a percentile, or (None, None) if unavailable."""
    if pd.isna(pct):
        return None, None
    for cutoff, label, color in PRESSURE_TIERS:
        if pct >= cutoff:
            return label, color
    return 'WEAK', colors.HexColor('#FF4D4D')

# NOTE: this used to be its own standalone "PRESSURE INDEX (TUG O' WAR
# /EDGE)" panel. ("under qb add the interior menace and edge pressure
# formula we did"), it's now MOVED (not duplicated) into THE CAPTAIN's
# grid -- same real percentile/tier logic, same two rows, just living next
# to the QB stats it provides pass-rush context for instead of further
# down the page. Factored into a reusable row-cell builder so THE
# CAPTAIN's grid (below) can pull it in directly.
def _pressure_row_cells(row_a, row_b, rate_col, pct_col, note_col, label):
    def stat_text(row):
        rate = row.get(rate_col)
        pct = row.get(pct_col)
        note = row.get(note_col)
        if pd.isna(rate) or pd.isna(pct):
            return (f"<i>unavailable this pull</i>", None)
        tier_label, color = _pressure_tier(pct)
        text = (f"<b>{rate * 100:.1f}%</b> of credited pressure"
                f"<br/>{pct:.0f}th pctile vs. league history &ndash; <b>{tier_label}</b>"
                f"<br/><font size=\"4.4\">{note if pd.notna(note) else ''}</font>")
        return (text, color)

    a_text, a_color = stat_text(row_a)
    b_text, b_color = stat_text(row_b)
    row = [Paragraph(a_text, style_val_base), Paragraph(label, style_label), Paragraph(b_text, style_val_base)]
    bg = [a_color, center_grey, b_color]
    return row, bg

# Real, this-week z-score of a field against the real 32-team league average
# -- same normalization the model favorite/underdog edge calc uses. Used by
# _pbe_prp_row_cells below (PBE and PRP live on different scales -- see its
# docstring) and available to any future panel needing the same treatment.
def z(val, field_key):
    std = league_stds.get(field_key, 0) or 1
    return (val - league_avgs.get(field_key, 0)) / std

# Real percentile of a value among the 32 teams in THIS week's own pull --
# cross-sectional (this week's field), since PBE/PRP don't carry a
# multi-season reference pool the way the Tug o' War / Edge Pressure percentiles
# above do. Reuses the same PRESSURE_TIERS/_pressure_tier() bracket, so
# "34th pctile -- BELOW AVG" means the same thing here that it means for
# Interior/Edge Pressure.
def field_pctile(val, field_key):
    if field_key not in df.columns:
        return np.nan
    col = df[field_key].dropna()
    return (col <= val).mean() * 100 if len(col) else np.nan

# NOTE: this used to be its own standalone "TRENCH BATTLE (PBE vs PRP)"
# panel. ("move it to freebirds"), it's now MOVED (not duplicated) into THE
# FREEBIRDS grid -- same real z-score net, same logic, just relocated.
# Factored into a reusable row-cell builder for that reason.
def _pbe_prp_row_cells(t_a, row_a, t_b, row_b):
    """PBE (pass blocking, offense) and PRP (pass rush, defense) are both
    real, but live on genuinely different scales -- PBE is a 0-100
    efficiency index, PRP a real ~6-19% pressure-forced rate. A raw
    subtraction would ALWAYS read as a lopsided offensive edge no matter
    what actually happened on the field, purely because PBE's scale is
    structurally bigger -- that's not a real signal, it's a units bug.
    Instead this nets each side's Z-SCORE -- how many standard deviations
    above/below THIS WEEK's real 32-team average each number is, on its own
    scale. That IS a genuinely comparable "who's over/under-performing more"
    signal: positive nets lean the O-line/offense, negative nets lean the
    pass rush/defense.
    """
    def matchup_cell(off_team, off_row, def_team, def_row):
        pbe = off_row.get('pbe_score')
        prp = def_row.get('prp_score')
        if pd.isna(pbe) or pd.isna(prp):
            return "<i>PBE/PRP unavailable this pull</i>", colors.white
        pbe_pct = field_pctile(pbe, 'pbe_score')
        prp_pct = field_pctile(prp, 'prp_score')
        pbe_label, _ = _pressure_tier(pbe_pct)
        prp_label, _ = _pressure_tier(prp_pct)
        net = z(pbe, 'pbe_score') - z(prp, 'prp_score')
        if net > 0.15:
            lean, color = "O-LINE", colors.HexColor('#2ECC71')
        elif net < -0.15:
            lean, color = "PASS RUSH", colors.HexColor('#FF4D4D')
        else:
            lean, color = "EVEN", colors.white
        # IMPORTANT: PBE (~82-93 scale) and PRP (~6-19% scale) are real but
        # NOT directly comparable by raw magnitude -- an O-line's PBE can
        # look like a "bigger number" than a pass rush's PRP and still be
        # the weaker side, because they're measured on different scales.
        # Each number's own percentile (in parentheses) is what actually
        # says whether it's good or bad -- that's what the net below
        # compares, not the raw digits.
        text = (f"<b>{off_team} PBE: {pbe:.1f}</b> ({pbe_pct:.0f}th pctile &ndash; {pbe_label})"
                f"<br/><b>{def_team} PRP: {prp:.1f}</b> ({prp_pct:.0f}th pctile &ndash; {prp_label})"
                f"<br/>Net: {net:+.2f} &ndash; leans <b>{lean}</b>"
                f"<br/><font size=\"4.4\">Different scales -- compare the pctiles, not the raw numbers. Sacks + 0.75×hits/dropback, no hurries (see README)</font>")
        return text, color

    a_text, a_color = matchup_cell(t_a, row_a, t_b, row_b)
    b_text, b_color = matchup_cell(t_b, row_b, t_a, row_a)
    row = [Paragraph(a_text, style_val_base), Paragraph("PASS BLOCK (PBE) vs<br/>PASS RUSH (PRP)", style_label), Paragraph(b_text, style_val_base)]
    bg = [a_color, center_grey, b_color]
    return row, bg

style_commentary = ParagraphStyle('Commentary', fontName='Helvetica', fontSize=7, leading=9.5, alignment=4)

def explain_matchup_edge(fav_off_row, dog_def_row):
    """Identifies whether the favorite's rushing or passing attack has the
    bigger statistical edge against the underdog's defense specifically (not
    just which team is 'better' overall), and phrases it as a concrete
    reason using real numbers -- e.g. 'GB may have trouble stopping the run'
    grounded in actual yards/game figures, not a vague hunch."""
    def edge(off_key, def_key):
        off_std = league_stds.get(off_key, 1) or 1
        def_std = league_stds.get(def_key, 1) or 1
        off_z = (fav_off_row[off_key] - league_avgs.get(off_key, 0)) / off_std
        def_z = (dog_def_row[def_key] - league_avgs.get(def_key, 0)) / def_std
        # def_key here is always an "allowed" stat (lower is better for the
        # defense), so a defense allowing MORE than average (positive def_z)
        # is bad for them -- add the two so both signals point the same way.
        return off_z + def_z

    rush_edge = edge('rushing_yards_per_game', 'rushing_yards_allowed_per_game')
    pass_edge = edge('passing_yards_per_game', 'passing_yards_allowed_per_game')

    if rush_edge >= pass_edge:
        return (f"{dog_def_row['team']} may have trouble containing the run here — "
                f"{fav_off_row['team']} is averaging {fav_off_row['rushing_yards_per_game']:.0f} rush yards/game "
                f"against a {dog_def_row['team']} defense allowing {dog_def_row['rushing_yards_allowed_per_game']:.0f} per game.")
    else:
        return (f"{dog_def_row['team']} may have trouble in the passing game here — "
                f"{fav_off_row['team']} is averaging {fav_off_row['passing_yards_per_game']:.0f} pass yards/game "
                f"against a {dog_def_row['team']} defense allowing {dog_def_row['passing_yards_allowed_per_game']:.0f} per game.")

def build_deep_dive_commentary(t_a, t_b, row_a, row_b):
    """Formulaic (not LLM-generated) prose summary built directly from the
    stats above -- deterministic, free, no API key required. Deliberately
    stays DESCRIPTIVE (what each team has actually done, and which side of
    the ball has the statistical edge) rather than forecasting a score or
    spread for the upcoming game -- see README.txt for why the model
    prediction panel itself was left out of the sold sheet."""
    def team_blurb(row):
        pass_lean = ("pass-heavy" if row['last_game_off_pass_rate'] > 55
                     else "run-heavy" if row['last_game_off_pass_rate'] < 45 else "balanced")
        pressure_note = ("faced heavy pressure" if row['last_game_off_pressure_rate_allowed'] > 25
                          else "protected the passer well")
        agg_note = ("aggressive" if row['fourth_down_aggressiveness_pct'] > 60
                    else "conservative" if row['fourth_down_aggressiveness_pct'] < 30 else "situational")
        result_word = {'W': 'win', 'L': 'loss', 'T': 'tie'}.get(row['last_game_result'], 'result')
        return (f"{row['team']} ({row['record']}) enters off a {result_word} against "
                f"{row['last_game_opponent']} ({int(row['last_game_points_for'])}-{int(row['last_game_points_against'])}). "
                f"Their offense leaned {pass_lean} that game and {pressure_note}. On the season they've been "
                f"{agg_note} on 4th down ({row['fourth_down_aggressiveness_pct']:.0f}% go-for-it rate).")

    # NEW: VOID is genuinely NaN ("--") for any team with fewer than 2 games
    # played (see the KNOWN MATH LIMITATION note in nfl_data_pull.py PART 7)
    # -- guard against that here instead of crashing on int(NaN), and fall
    # back to a plain statistical-edge read (still real numbers, just not
    # VOID-framed) when either team isn't eligible yet.
    if pd.isna(row_a.get('total_void')) or pd.isna(row_b.get('total_void')):
        take = (f"VOID isn't available yet for one or both teams this week (needs 2+ games played -- "
                f"see README.txt). {explain_matchup_edge(row_a, row_b) if row_a['off_yards_per_play'] >= row_b['off_yards_per_play'] else explain_matchup_edge(row_b, row_a)}")
    else:
        void_gap = row_a['total_void'] - row_b['total_void']
        if void_gap > 0.02:
            take = f"By VOID, {t_a} has rated as the stronger team so far this season (#{int(row_a['total_void_rank'])} vs #{int(row_b['total_void_rank'])} league-wide). {explain_matchup_edge(row_a, row_b)}"
        elif void_gap < -0.02:
            take = f"By VOID, {t_b} has rated as the stronger team so far this season (#{int(row_b['total_void_rank'])} vs #{int(row_a['total_void_rank'])} league-wide). {explain_matchup_edge(row_b, row_a)}"
        else:
            take = f"By VOID, these two have rated about evenly so far this season (#{int(row_a['total_void_rank'])} vs #{int(row_b['total_void_rank'])} league-wide)."

    body_text = team_blurb(row_a) + " " + team_blurb(row_b) + " " + take
    body = Paragraph(body_text, style_commentary)
    body_wrap = Table([[body]], colWidths=[sum(panel_widths)])
    body_wrap.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F5F5F5')),
        ('GRID', (0,0), (-1,-1), 0.75, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    return [full_width_header("DEEP DIVE"), Spacer(1, 1), body_wrap]

# ==============================================================================
# HOW TO READ THIS SHEET -- a real glossary appendix, added (this replaced
# the original plan of tiny inline captions under each panel, which page 1's
# pagination budget couldn't absorb -- see the round-4 README entry). Every
# field on this sheet gets an entry: what the number means, why it matters,
# and (for the 6 stats flagged for a formula check) the real formula behind
# it, pulled straight from nfl_data_pull.py, not re-derived or guessed here.
#
# TWO SEPARATE REAL SCALES ARE USED ON THIS SHEET, and this page explains
# both rather than conflating them:
#   1. CELL COLORS -- the live red/white/green/purple coloring every panel
#      already uses (get_stat_tier() above), a %-of-average or std-dev-based
#      deviation band, with purple reserved for the real top-15th percentile
#      league-wide ("elite"). This is what's actually painted on the page.
#   2. POOR / AVERAGE / GOOD below -- a simpler, separate 3-tier scale built
#      just for this glossary: the real bottom/middle/top third of THIS
#      SEASON's 32 real NFL teams for that exact stat (a plain percentile
#      tercile split), direction-aware using the same lower_is_better_fields
#      list the rest of the sheet already uses. Computed fresh from real
#      data (df) every time this PDF is built -- never a fixed, hardcoded
#      number -- so it stays accurate as the season progresses. It's
#      deliberately a different (simpler) method than #1's deviation bands,
#      which is why the exact cutoff numbers won't always match a plain
#      reading of the live cell colors -- both are real, just two different
#      honest lenses on the same real data.
# ==============================================================================

# Fields already given their own full write-up (formula + scale) below --
# excluded from the auto-scraped format table so a stray reuse of one of
# these keys elsewhere on the sheet can't silently generate a second,
# conflicting glossary line for it.
_GLOSSARY_HANDWRITTEN_FIELDS = {
    'off_efficiency_epa', 'def_efficiency_epa', 'net_efficiency_epa',
    'off_void', 'def_void', 'total_void', 'off_void_rank', 'def_void_rank', 'total_void_rank',
    'interior_pressure_rate', 'edge_pressure_rate', 'interior_pressure_pct', 'edge_pressure_pct',
    'pbe_score', 'prp_score', 'secondary_lock_index',
}

def _scrape_field_formats():
    """Real (fmt, suffix) display format per field key, scraped straight
    from the same *_PAIRS/*_ROWS lists every panel already renders from --
    so a glossary scale line always prints numbers formatted exactly the
    way that same field is formatted everywhere else on the sheet. Skips
    any row using a multiplier != 1 (only one real row on the whole sheet
    does -- MATCHUP_PAIRS' NET EFFICIENCY row, mult=100 for that one
    panel's display convention -- and skips the VOID_PAIRS ordinal-function
    row, both irrelevant here since those fields are hand-written above)."""
    fmt_map = {}
    # GAME_PLAN_PAIRS removed from this list -- that list itself no longer
    # exists (its fields moved into VOID_PAIRS, see the note above it).
    # pass_play_pct_offense/rush_play_pct_offense still get scraped fine via
    # TEAM_RUNDOWN_OFFENSE_ROWS/TEAM_RUNDOWN_DEFENSE_ROWS below (they also
    # appear there, e.g. "RUSH YARDS / GM (RUSH PLAY %)"); net_efficiency_epa
    # is hand-written (_GLOSSARY_HANDWRITTEN_FIELDS) either way.
    pair_style_lists = [MATCHUP_PAIRS, TRENCH_BATTLE_PAIRS, SKILL_BATTLE_PAIRS,
                         FIELD_GENERAL_PAIRS, THE_POUND_PAIRS, KICKING_PAIRS, BLOCK_PAIRS,
                         TENDENCY_PAIRS, CAPTAIN_PAIRS, FREEBIRDS_PAIRS, DAWGS_PAIRS]
    for lst in pair_style_lists:
        for row in lst:
            if len(row) == 6:
                key_a, key_b, fmt, suffix, mult, _label = row
                if mult == 1 and not callable(fmt):
                    fmt_map.setdefault(key_a, (fmt, suffix))
                    fmt_map.setdefault(key_b, (fmt, suffix))
    for lst in [TEAM_RUNDOWN_OFFENSE_ROWS, TEAM_RUNDOWN_DEFENSE_ROWS]:
        for row in lst:
            if len(row) == 5:
                key, fmt, suffix, mult, _label = row
                if mult == 1 and not callable(fmt):
                    fmt_map.setdefault(key, (fmt, suffix))
            elif len(row) == 9:
                key, fmt, suffix, mult, key2, fmt2, suffix2, mult2, _label = row
                if mult == 1 and not callable(fmt):
                    fmt_map.setdefault(key, (fmt, suffix))
                if mult2 == 1 and not callable(fmt2):
                    fmt_map.setdefault(key2, (fmt2, suffix2))
    return fmt_map

FIELD_FORMAT = _scrape_field_formats()

def _fmt_gloss_val(field_key, val):
    fmt, suffix = FIELD_FORMAT.get(field_key, ("{:.1f}", ""))
    try:
        return fmt.format(val) + suffix
    except Exception:
        return f"{val:.2f}"

def _tercile_scale_line(scale_specs):
    """scale_specs: list of (side_label_or_None, field_key). Real bottom/
    middle/top third of this season's 32 real teams for that field -- see
    the write-up at the top of this section for why this is a DIFFERENT
    (simpler) real scale than the live cell-color bands.

    SANITY-CHECKED, not just computed and printed blind: this early in a
    real NFL season, several genuinely rare-event real fields (4th down
    attempts, blocked kicks, special-teams TDs) have most of the real
    32 teams sitting at exactly the same value (often 0) -- a plain
    tercile split on real data like that produces a degenerate or
    downright impossible-looking bucket (e.g. "GOOD: above 100.0%",
    which no real team could ever clear). Rather than print a technically-
    computed-but-nonsensical number, this falls back to a plain real
    average + range note for exactly those cases -- still 100% real data,
    just not forced into a 3-bucket shape that doesn't fit it yet."""
    lines = []
    for side_label, field_key in scale_specs:
        prefix = f"{side_label} &ndash; " if side_label else ""
        if field_key not in df.columns:
            lines.append(f"{prefix}data unavailable this pull")
            continue
        s = df[field_key].dropna()
        if len(s) < 8:
            lines.append(f"{prefix}sample too small this week for a reliable real scale")
            continue
        p33, p67, avg = s.quantile(1/3), s.quantile(2/3), s.mean()
        real_min, real_max = s.min(), s.max()
        lb = field_key in lower_is_better_fields
        avg_s = _fmt_gloss_val(field_key, avg)
        # Degenerate cases: no real separation between the tercile cutoffs
        # (most teams tied, usually at 0 this early in the season), or the
        # computed "good" bound sits AT the real observed max / the "poor"
        # bound sits AT the real observed min -- either way that bucket is
        # empty in reality, so a POOR/AVERAGE/GOOD line would be printing a
        # bucket no real team is actually in.
        degenerate = (p33 == p67) or (p67 >= real_max) or (p33 <= real_min)
        if degenerate:
            rng = f"{_fmt_gloss_val(field_key, real_min)} to {_fmt_gloss_val(field_key, real_max)}"
            lines.append(f"{prefix}real teams cluster too unevenly this early in the season for a "
                         f"meaningful POOR/AVERAGE/GOOD split (a common real pattern for rare-event "
                         f"stats early on) &ndash; real avg ~{avg_s}, real range {rng}")
            continue
        lo, hi = _fmt_gloss_val(field_key, p33), _fmt_gloss_val(field_key, p67)
        if lb:
            line = f"GOOD: below {lo}  |  AVERAGE: {lo}-{hi}  |  POOR: above {hi}  (real avg ~{avg_s})"
        else:
            line = f"POOR: below {lo}  |  AVERAGE: {lo}-{hi}  |  GOOD: above {hi}  (real avg ~{avg_s})"
        lines.append(prefix + line)
    return "<br/>".join(lines)

# ==============================================================================
# GLOSSARY_ENTRIES -- one dict per stat concept. 'scale' triggers a live,
# real tercile computed from df (see _tercile_scale_line above) every time
# this PDF is built; 'scale_note' is a hand-written override for stats where
# a fixed poor/avg/good scale would be dishonest (VOID -- too small a real
# sample so far this season) or meaningless (purely informational
# tendencies with no real "good/bad" direction -- same fields already
# flagged that way in lower_is_better_fields's own comments above).
# ==============================================================================
GLOSSARY_ENTRIES = [
    # -------------------- THE SIX ADVANCED METRICS -- FORMULAS --------------------
    {'section': "THE SIX ADVANCED METRICS -- REAL FORMULAS", 'name': "EPA / NET EFFICIENCY / NET EPA PER MATCHUP",
     'body': ("EPA (Expected Points Added) is nflverse's own real published advanced stat -- how much a single "
              "play shifts a team's expected point total, given the down, distance, and field position, not just "
              "the raw yards gained. off_efficiency_epa/def_efficiency_epa on this sheet are the real, unmodified "
              "average of nflverse's play-by-play epa column for that team's plays, offense and defense (EPA "
              "ALLOWED) separately. NET EFFICIENCY is this team's own off EPA/play minus what its defense allows -- "
              "one number for whether this team's plays are worth more than its opponents' plays against it. NET "
              "EPA PER MATCHUP is the same idea computed live for THIS specific pairing instead of a generic "
              "season number."),
     'formula': "NET EFFICIENCY = off_efficiency_epa &minus; def_efficiency_epa.  NET EPA PER MATCHUP = (this team's off_efficiency_epa &times; 1.5) &minus; the opponent's def_efficiency_epa. "
                "Both now render as rows inside the VOID RATING panel at the top of the sheet, not their own separate panel -- moved there, real numbers/formulas unchanged. "
                "The raw off_efficiency_epa number itself (EPA/PLAY (OFF), real quality stat #2 at 61.0%) is also now its own row, right under NET EPA (MATCHUP) -- previously it was only ever shown combined into NET EFFICIENCY/NET EPA, never on its own.",
     'scale': [("OFFENSE EPA/play (higher is better)", 'off_efficiency_epa'),
               ("DEFENSE EPA/play allowed (lower is better)", 'def_efficiency_epa'),
               ("NET EFFICIENCY (off &minus; def, higher is better)", 'net_efficiency_epa')]},

    {'section': "THE SIX ADVANCED METRICS -- REAL FORMULAS", 'name': "VOID (Value Over Interactive Average)",
     'body': ("Our own opponent-adjusted, situational-success efficiency rating -- in the same conceptual family "
              "as DVOA, but independently built entirely from free, real nflverse play-by-play (DVOA's own exact "
              "formula is proprietary and can't be reproduced). Every qualifying play gets a real situational "
              "value based on how much of the yards-to-go it actually gained for that down, with explosive-play "
              "bonuses and turnover/touchdown overrides, then each team's raw average is adjusted for the real "
              "strength of the opponents it has actually played. Positive = better than league average, for BOTH "
              "offense and defense (deliberately different from DVOA's sign convention, so it reads the same way "
              "on both sides of the ball)."),
     'formula': ("Per play: value = yards_gained &divide; (need_frac[down] &times; yards_to_go), clipped to "
                 "[-1.5, 1.5], +0.2/+0.4/+0.6 bonus for 10+/20+/40+ yard gains, forced to &le;-1.5 on a turnover "
                 "and &ge;1.2 on a touchdown. Team VOID = the opponent-adjusted average of that value across all "
                 "real qualifying plays."),
     'scale_note': ("Zero = exactly league average, by construction. No fixed POOR/AVERAGE/GOOD cutoffs are "
                     "printed here -- only 2 real team-weeks have accumulated enough games (2+, the real gate "
                     "this stat needs to be mathematically meaningful -- see README.txt) so far this season, too "
                     "small a real sample to set an honest cutoff. Read it directionally: solidly positive = "
                     "playing above average, solidly negative = below, near zero = roughly average.")},

    {'section': "THE SIX ADVANCED METRICS -- REAL FORMULAS", 'name': "TUG O' WAR (INTERIOR) / EDGE PRESSURE",
     'body': ("Real credited pass-rush pressure rate for this team's own defensive front -- interior (NT/DT) for "
              "Tug o' War, edge rushers (DE/OLB) for Edge Pressure -- built from real player-attributed sacks and "
              "QB hits, personnel-weighted to the actual players on this team's current roster using their own "
              "real historical rates. Not tied to a specific opposing O-line (that comparison lives in PASS BLOCK "
              "vs PASS RUSH below); this is just how much heat this team's front generates on its own."),
     'formula': "rate = credited pressure events (sacks + QB hits) &divide; real pass-block snaps faced, for that position group, personnel-weighted to the current roster.",
     'scale_note': ("The ELITE/ABOVE AVG/AVERAGE/BELOW AVG/WEAK badge shown on the sheet is a real percentile "
                     "rank against a historical MULTI-SEASON reference pool (every current team's own last few "
                     "real seasons) -- ELITE &ge;85th pctile, ABOVE AVG 65-85th, AVERAGE 35-65th, BELOW AVG "
                     "15-35th, WEAK &lt;15th. That reference pool is a different real dataset than this week's 32 "
                     "teams, so a plain tercile of this week's rates alone (like the other entries below) would "
                     "give a different, less accurate answer -- not printed here for that reason. For real "
                     "context only: this week's 32 real teams range roughly 2%-6% interior (avg ~3.6%) and "
                     "3%-8% edge (avg ~5.2%).")},

    {'section': "THE SIX ADVANCED METRICS -- REAL FORMULAS", 'name': "PASS BLOCK (PBE) vs PASS RUSH (PRP)",
     'body': ("PBE (Pass Block Efficiency) is a real 0-100 index for this team's OWN O-line -- 100 means never "
              "pressured. PRP (Pressure Rate Productivity) is a real pressure-forced rate for the OPPOSING "
              "defense's pass rush. Both come from the same real sack/QB-hit data as pressure_rate_allowed_pct/"
              "forced_pct, just weighted differently (PFF's real published PBE weighting: 1&times; sacks, "
              "0.75&times; QB hits). Because they live on genuinely different real scales (PBE ~0-100, PRP a "
              "~3%-34% rate), they're never subtracted directly -- each is converted to a z-score (real standard "
              "deviations from this week's real 32-team average) and THOSE are netted, which is what actually "
              "decides LEANS O-LINE / LEANS PASS RUSH / EVEN on the sheet."),
     'formula': ("PBE = 100 &minus; [(sacks allowed + 0.75 &times; QB hits allowed) &divide; dropbacks] &times; 100.  "
                 "PRP = [(sacks forced + 0.75 &times; QB hits forced) &divide; dropbacks faced] &times; 100.  "
                 "Net = z(PBE) &minus; z(PRP); net &gt; 0.15 leans O-LINE, net &lt; -0.15 leans PASS RUSH, else EVEN."),
     'scale': [("PBE (real, this season, higher is better)", 'pbe_score'),
               ("PRP (real, this season, higher is better for the defense)", 'prp_score')]},

    {'section': "THE SIX ADVANCED METRICS -- REAL FORMULAS", 'name': "SECONDARY LOCK INDEX",
     'body': ("A single real number for how tight a defense's coverage has been PER TARGET faced -- isolates "
              "coverage tightness independent of how many passes a defense faced (a defense seeing fewer targets "
              "doesn't automatically look better or worse). Defense-only: no offensive mirror stat exists for "
              "coverage tightness, so this shows once, not split into a crossed offense-vs-defense panel like "
              "most of the sheet."),
     'formula': "Secondary Lock Index = 100 &minus; (real receiving yards allowed &divide; real targets faced).",
     'scale': [(None, 'secondary_lock_index')]},

    # -------------------- SCORING & OVERALL PRODUCTION --------------------
    {'section': "SCORING & OVERALL PRODUCTION", 'name': "POINTS / GAME (and ALLOWED)",
     'body': "Real points scored/allowed per game. The most basic real box-score number on the sheet, but noisy on its own -- doesn't account for opponent quality, field position, or special-teams/defensive scores mixed in with offense.",
     'scale': [("OFFENSE", 'points_per_game'), ("DEFENSE (allowed, lower is better)", 'points_allowed_per_game')]},
    {'section': "SCORING & OVERALL PRODUCTION", 'name': "AVG SCORING MARGIN",
     'body': "Real average real point differential per game (this team's points minus the opponent's). A quick real gauge of how competitive this team's games have actually been, not just its record.",
     'scale': [(None, 'avg_scoring_margin')]},
    {'section': "SCORING & OVERALL PRODUCTION", 'name': "YARDS / GAME (and ALLOWED)",
     'body': "Real total offensive/defensive yardage per game. Correlates with winning but less tightly than EPA/play, since it doesn't weight WHEN or how efficiently those yards came.",
     'scale': [("OFFENSE", 'total_yards_per_game'), ("DEFENSE (allowed, lower is better)", 'total_yards_per_game_allowed')]},
    {'section': "SCORING & OVERALL PRODUCTION", 'name': "YARDS / DRIVE (and ALLOWED)",
     'body': "Real yards gained/allowed per real offensive drive -- normalizes for pace (a team running more real plays naturally racks up more total yards without necessarily being more efficient per drive).",
     'scale': [("OFFENSE", 'yards_per_drive'), ("DEFENSE (allowed, lower is better)", 'yards_per_drive_allowed')]},
    {'section': "SCORING & OVERALL PRODUCTION", 'name': "YARDS / PLAY (and ALLOWED)",
     'body': "Real yards per offensive/defensive snap -- the classic real efficiency-per-snap number, one level less granular than EPA/play since it ignores down/distance/field position context.",
     'scale': [("OFFENSE", 'off_yards_per_play'), ("DEFENSE (allowed, lower is better)", 'def_yards_per_play')]},

    # -------------------- PLAY-CALLING & PACE --------------------
    {'section': "PLAY-CALLING, PACE & TIME OF POSSESSION", 'name': "RUSH/PASS YARDS PER GAME (with RUSH/PASS PLAY %)",
     'body': ("Real rushing/passing yardage per game, with that side's real share of offensive plays shown in "
              "parentheses. The play-% number is a real tendency, not inherently good or bad on its own (a "
              "run-heavy team protecting a lead looks the same in this column as a run-heavy team that simply "
              "can't throw) -- read the yardage number for production, the % for game plan."),
     'scale': [("RUSHING YDS/GM", 'rushing_yards_per_game'), ("RUSHING YDS/GM ALLOWED (lower is better)", 'rushing_yards_allowed_per_game'),
               ("PASSING YDS/GM", 'passing_yards_per_game'), ("PASSING YDS/GM ALLOWED (lower is better)", 'passing_yards_allowed_per_game')]},
    {'section': "PLAY-CALLING, PACE & TIME OF POSSESSION", 'name': "PLAYS / GAME (and FACED)",
     'body': "Real offensive snaps run per game (or defensive snaps faced). Left uncolored/no fixed scale on this sheet by design -- more plays can mean an efficient no-huddle attack OR a defense that can't get off the field, genuinely ambiguous which it is from this number alone.",
     'scale_note': "No fixed scale -- genuinely ambiguous direction (see body). Real context only: this week's 32 real teams run ~52-68 offensive plays/game (avg ~60)."},
    {'section': "PLAY-CALLING, PACE & TIME OF POSSESSION", 'name': "TIME OF POSSESSION (MIN)",
     'body': "Real minutes of possession per game. Same genuine ambiguity as PLAYS/GAME -- controlling the clock can reflect a dominant, efficient offense or just a run-it-out-the-clock game script; no fixed scale is printed for the same reason.",
     'scale_note': "No fixed scale -- genuinely ambiguous direction (see body). Real context only: this week's 32 real teams average ~24-36 min/gm (avg ~30)."},

    # -------------------- DOWN & DISTANCE / SITUATIONAL --------------------
    {'section': "DOWN & DISTANCE / SITUATIONAL FOOTBALL", 'name': "3RD DOWN CONVERSION % (and DEFENSE)",
     'body': "Real share of 3rd-down plays converted for a new set of downs (offense) or allowed (defense). One of the most repeatable real situational-football numbers -- directly extends or ends real drives.",
     'scale': [("OFFENSE", 'third_down_pct_offense'), ("DEFENSE (allowed, lower is better)", 'third_down_pct_defense')]},
    {'section': "DOWN & DISTANCE / SITUATIONAL FOOTBALL", 'name': "4TH DOWN CONVERSION % (and ALLOWED)",
     'body': "Real share of 4th-down attempts converted/allowed. Small real sample for most teams (most 4th downs are punted), so read this alongside 4TH DOWN AGGRESSIVENESS below rather than alone.",
     'scale': [("OFFENSE", 'fourth_down_conversion_pct'), ("DEFENSE (allowed, lower is better)", 'fourth_down_conversion_pct_allowed')]},
    {'section': "DOWN & DISTANCE / SITUATIONAL FOOTBALL", 'name': "AVG TO-GO DISTANCE (3RD DOWN)",
     'body': "Real average yards needed on 3rd down (offense faced, or defense forced). A shorter real distance for the offense means 1st/2nd down were efficient; a longer real distance forced is good defense.",
     'scale': [("OFFENSE (shorter is better)", 'avg_3rd_down_distance_offense'), ("DEFENSE (longer forced is better)", 'avg_3rd_down_distance_defense')]},
    {'section': "DOWN & DISTANCE / SITUATIONAL FOOTBALL", 'name': "4TH DOWN AGGRESSIVENESS (SEASON)",
     'body': "Real season-long go-for-it rate on 4th down in real go-for-it situations (excludes obvious punt/FG spots). A real coaching-tendency number -- informational, not inherently good or bad (an aggressive team in bad spots isn't automatically smarter than a conservative one in good field position).",
     'scale_note': "No fixed scale -- a real coaching tendency, not a performance grade. Real context only: this week's 32 real teams range roughly 0%-50% (avg ~21%)."},
    {'section': "DOWN & DISTANCE / SITUATIONAL FOOTBALL", 'name': "RED ZONE TD/SUCCESS % (and ALLOWED)",
     'body': "Real share of red-zone trips that end in a touchdown (offense) or are allowed to (defense). Directly measures whether real scoring chances actually turn into points, separate from just reaching the red zone.",
     'scale': [("OFFENSE", 'redzone_efficiency'), ("DEFENSE (allowed, lower is better)", 'redzone_efficiency_allowed')]},
    {'section': "DOWN & DISTANCE / SITUATIONAL FOOTBALL", 'name': "FIELD GOAL %",
     'body': "Real field goal conversion rate. Straightforward, but a small real sample for most teams (a handful of kicks per game), so a single miss or make swings this number more than most other stats on the sheet.",
     'scale': [(None, 'fg_pct')]},

    # -------------------- TURNOVERS, PRESSURE & TRENCHES --------------------
    {'section': "TURNOVERS, PRESSURE & THE TRENCHES", 'name': "INTERCEPTIONS THROWN / GAME (and GAINED)",
     'body': "Real interceptions thrown by this team's offense, or gained by this defense, per game. One of the clearest real turnover-margin drivers on the sheet.",
     'scale': [("OFFENSE (thrown, lower is better)", 'interceptions_thrown_per_game'), ("DEFENSE (gained)", 'interceptions_gained_per_game')]},
    {'section': "TURNOVERS, PRESSURE & THE TRENCHES", 'name': "FUMBLES (LOST vs FORCED)",
     'body': "Real fumbles this team's offense has lost, and real fumbles this team's defense has forced. Fumble recovery itself is famously close to a real coin flip league-wide, so read the FORCED number as the more repeatable real defensive skill; LOST carries more year-to-year randomness.",
     'scale': [("LOST (offense, lower is better)", 'fumbles_lost_per_game'), ("FORCED (defense)", 'opponent_fumbles_forced_per_game')]},
    {'section': "TURNOVERS, PRESSURE & THE TRENCHES", 'name': "SACKS (ALLOWED vs MADE)",
     'body': "Real sacks this team's O-line has allowed, and real sacks this team's defense has made. A cleaner, more familiar real box-score number than PRESSURE RATE below, but noisier -- pressure without a sack (a QB who escapes or throws it away) doesn't show up here at all.",
     'scale': [("ALLOWED (offense, lower is better)", 'sacks_allowed'), ("MADE (defense)", 'sacks_made')]},
    {'section': "TURNOVERS, PRESSURE & THE TRENCHES", 'name': "TFL (ALLOWED vs MADE)",
     'body': "Real tackles for loss this team's offense has allowed, and real TFLs this team's defense has made. A good proxy for which team is winning the line of scrimmage on run plays specifically.",
     'scale': [("ALLOWED (offense, lower is better)", 'tfl_allowed'), ("MADE (defense)", 'tfl_made')]},
    {'section': "TURNOVERS, PRESSURE & THE TRENCHES", 'name': "RUN STUFF RATE",
     'body': "Real share of an offense's own run plays stopped at or behind the line of scrimmage, or a defense's real share of opposing run plays stuffed. The cleanest single real number for who's winning the run-blocking/run-defending trench battle on a given play.",
     'scale': [("OFFENSE (stuffed, lower is better)", 'stuff_rate_offense'), ("DEFENSE (stuffs made)", 'stuff_rate_defense')]},
    {'section': "TURNOVERS, PRESSURE & THE TRENCHES", 'name': "PRESSURE RATE (ALLOWED vs FORCED, SEASON)",
     'body': "Real season-long share of real dropbacks that ended in a sack or QB hit -- allowed by this team's O-line, or forced by this team's pass rush. The team-wide season version of the same real sack/hit data PBE/PRP above use with a different weighting.",
     'scale': [("ALLOWED (offense, lower is better)", 'pressure_rate_allowed_pct'), ("FORCED (defense)", 'pressure_rate_forced_pct')]},
    {'section': "TURNOVERS, PRESSURE & THE TRENCHES", 'name': "HAVOC RATE",
     'body': "Real share of a defense's snaps that produce a TFL, sack, forced fumble, pass breakup, or interception -- a single real number for how often a defense disrupts a play behind or at the line rather than just letting it develop.",
     'scale': [(None, 'havoc_rate_defense')]},
    {'section': "TURNOVERS, PRESSURE & THE TRENCHES", 'name': "PENALTY YARDS / GAME",
     'body': "Real penalty yardage assessed against this team per game (offense and defense combined). Fewer is better -- self-inflicted real yardage that erases real drive progress or extends an opponent's.",
     'scale': [(None, 'penalty_yards_per_game')]},

    # -------------------- QB PASSING --------------------
    {'section': "QB PASSING METRICS", 'name': "CPOE (Completion % Over Expected)",
     'body': "Real nflverse advanced stat -- this QB's actual completion rate minus what a league-average arm would be expected to complete on the exact same real throws (accounting for depth, pressure, receiver separation, etc.). Isolates real accuracy/decision-making from a favorable or unfavorable real schedule of throws.",
     'scale': [(None, 'cpoe')]},
    {'section': "QB PASSING METRICS", 'name': "PASSER RATING",
     'body': "The familiar open-source passer rating formula (completion %, yards/attempt, TD rate, INT rate combined into one real number). Widely recognized but a blunter real instrument than CPOE or EPA -- doesn't account for opponent quality or situational context.",
     'scale': [(None, 'open_source_passer_rating')]},
    {'section': "QB PASSING METRICS", 'name': "TRUE COMPLETION % (vs ALLOWED)",
     'body': "Real completion percentage for this offense's passing attack, or real completion percentage this defense's coverage has allowed.",
     'scale': [("OFFENSE", 'completion_pct'), ("DEFENSE (allowed, lower is better)", 'completion_pct_allowed')]},
    {'section': "QB PASSING METRICS", 'name': "SACK % (PROTECTION vs PASS RUSH)",
     'body': "Real sacks taken as a share of real real dropbacks (offense's protection), or real sacks generated as a share of real dropbacks faced (defense's pass rush).",
     'scale': [("OFFENSE (protection, lower is better)", 'sack_pct_offense'), ("DEFENSE (pass rush)", 'sack_pct_defense')]},
    {'section': "QB PASSING METRICS", 'name': "YARDS / COMPLETION (vs ALLOWED)",
     'body': "Real yards gained per completed pass, offense and defense. A real proxy for how aggressively (and successfully) an offense is pushing the ball downfield, or how much a defense limits yards after the catch and on deep completions.",
     'scale': [("OFFENSE", 'yards_per_completion'), ("DEFENSE (allowed, lower is better)", 'yards_per_completion_allowed')]},

    # -------------------- RUSHING --------------------
    {'section': "RUSHING METRICS", 'name': "RYOE / RUSH ATTEMPT",
     'body': "Real NGS (Next Gen Stats) rush yards over expected, per attempt -- how many more (or fewer) yards a real rusher gained than an average back would be expected to on the exact same real blocking/defender positioning. Isolates real runner skill from real O-line push.",
     'scale': [(None, 'ryoe_per_attempt')]},
    {'section': "RUSHING METRICS", 'name': "RUN SUCCESS RATE vs RUN STOP RATE",
     'body': "Real share of an offense's run plays that gain enough yardage to be a situational success (same success-value logic VOID uses), against the real share of opposing run plays a defense stops short of that same bar. The direct offense-vs-defense read on which side is winning the ground game overall, not just at the line (see RUN STUFF RATE above for that narrower cut).",
     'scale': [("OFFENSE (run success)", 'off_run_success_rate'), ("DEFENSE (run stop)", 'def_run_stop_rate')]},

    # -------------------- RECEIVING / COVERAGE --------------------
    {'section': "RECEIVING & COVERAGE METRICS", 'name': "CATCH % (vs ALLOWED)",
     'body': "Real share of real targets that are caught, for this offense's receivers or against this defense's coverage.",
     'scale': [("OFFENSE", 'catch_pct'), ("DEFENSE (allowed, lower is better)", 'catch_pct_allowed')]},
    {'section': "RECEIVING & COVERAGE METRICS", 'name': "YAC / RECEPTION (vs ALLOWED)",
     'body': "Real yards after the catch per reception, for this offense's pass-catchers or allowed by this defense. A real proxy for playmaking-after-the-catch ability (offense) or real tackling-in-space discipline (defense).",
     'scale': [("OFFENSE", 'yac_per_reception_pbp'), ("DEFENSE (allowed, lower is better)", 'yac_allowed_per_reception_pbp')]},
    {'section': "RECEIVING & COVERAGE METRICS", 'name': "TARGET SEPARATION (vs ALLOWED)",
     'body': "Real average separation (in yards) this offense's receivers create at the time of the throw, or real separation this defense's coverage allows. A cleaner real measure of route-running/coverage skill than catch rate alone, since it's measured independent of whether the pass was actually completed.",
     'scale': [("OFFENSE", 'target_separation'), ("DEFENSE (allowed, lower is better)", 'target_separation_allowed')]},
    {'section': "RECEIVING & COVERAGE METRICS", 'name': "DROP % -- not shown",
     'body': "Not available: no free/CC-licensed real data source exists for it (PFR-exclusive manual charting only). See README.txt \"LICENSING.\"",
     'scale_note': None},

    # -------------------- TENDENCIES (INFORMATIONAL) --------------------
    {'section': "OC/DC TENDENCY METRICS (INFORMATIONAL -- NO FIXED SCALE)", 'name': "PASS/PRESSURE TENDENCY (LAST GAME) & 4TH DOWN AGGRESSIVENESS",
     'body': ("PASS RATE (LAST GM), PRESSURE ALLOWED/FORCED (LAST GM), and PASS RATE FACED (LAST GM) are real "
              "single-game tendency numbers from each team's most recent real game -- useful for spotting a real "
              "recent shift in game plan, but genuinely a mix of real strategy and real game-script noise (a team "
              "trailing big late will pass far more than its normal tendency). The OC/DC TENDENCY strips (real "
              "historical EPA/success rate vs. man and zone coverage, attributed to each team's actual current "
              "coordinator) are the same idea over a longer real historical window."),
     'scale_note': "No fixed poor/avg/good scale for any of these -- real tendencies and single-game snapshots, not performance grades."},

    # -------------------- KICKING, PUNTING & SPECIAL TEAMS --------------------
    {'section': "KICKING, PUNTING & SPECIAL TEAMS", 'name': "PUNTS / GAME",
     'body': "Real punts per game. Left with no fixed scale by design -- fewer punts often means a better offense, but can also just mean more 4th-down aggressiveness or a shorter field; genuinely ambiguous on its own, same reasoning as PLAYS/GAME above.",
     'scale_note': "No fixed scale -- genuinely ambiguous direction (see body). Real context only: this week's 32 real teams average ~2-6 punts/gm (avg ~3.6)."},
    {'section': "KICKING, PUNTING & SPECIAL TEAMS", 'name': "AVG PUNT DISTANCE",
     'body': "Real average yards per punt. Straightforwardly higher-is-better real leg strength/hang-time number.",
     'scale': [(None, 'avg_punt_distance')]},
    {'section': "KICKING, PUNTING & SPECIAL TEAMS", 'name': "PUNT RETURN AVG",
     'body': "Real average yards per punt return. Higher is better, though a real small-sample stat for most teams (a handful of real returns per game).",
     'scale': [(None, 'punt_return_avg')]},
    {'section': "KICKING, PUNTING & SPECIAL TEAMS", 'name': "AVG FG DISTANCE (ATTEMPTED)",
     'body': "Real average distance of this team's field goal attempts. Informational context for FIELD GOAL % above (a team attempting longer real kicks will naturally convert a lower raw % even with a strong real leg) rather than a stat with its own clean good/bad direction.",
     'scale': [(None, 'avg_fg_distance')]},
    {'section': "KICKING, PUNTING & SPECIAL TEAMS", 'name': "SPECIAL TEAMS TDs / GAME",
     'body': "Real special-teams touchdowns (return/blocked-kick scores) per game. Rare, high-variance real events -- most teams sit at 0.00 most weeks, so a nonzero real number here is a genuine outlier, not necessarily a repeatable skill.",
     'scale': [(None, 'special_teams_tds_per_game')]},
    {'section': "KICKING, PUNTING & SPECIAL TEAMS", 'name': "FG BLOCKED vs FG BLOCKS FORCED / PUNTS BLOCKED vs PUNT BLOCKS FORCED",
     'body': "Real blocked-kick counts, this team's own kicks blocked and blocks this team's special teams has forced. Genuinely rare real events league-wide (most teams sit at 0.00 all season), so treat any nonzero real number here as a notable outlier rather than a stable trend.",
     'scale': [("FG BLOCKED (lower is better)", 'fg_blocked_per_game'), ("FG BLOCKS FORCED", 'fg_blocks_forced_per_game'),
               ("PUNTS BLOCKED (lower is better)", 'punts_blocked_per_game'), ("PUNT BLOCKS FORCED", 'punt_blocks_forced_per_game')]},

    # -------------------- PLAYER USAGE --------------------
    {'section': "PLAYER USAGE (QB/RB/WR/TE LISTS)", 'name': "TARGETS-SHARE/CARRIES-SHARE, SNAP %, PPR PTS/GM, TIME TO THROW",
     'body': ("Real per-player usage numbers under THE CAPTAIN/DAWGS/FREEBIRDS/TANKS: real targets or carries per "
              "game (with that player's real share of the team total in parentheses), real offensive snap %, real "
              "PPR fantasy points per game (with the real last-3-game average alongside it for a recency read), "
              "and for QBs, real NGS time-to-throw in seconds. These are single-PLAYER numbers, not team "
              "averages, so a fixed poor/avg/good scale isn't printed here -- the real \"#N/32 PPR pts/gm allowed "
              "to this position\" figure shown right next to each player already IS this sheet's real scale for "
              "that specific matchup: it tells you directly where the opposing defense ranks against that "
              "position, league-wide, in real terms."),
     'scale_note': None},

    # -------------------- NARRATIVE PANELS --------------------
    {'section': "DEEP DIVE & INJURY REPORT", 'name': "DEEP DIVE / INJURY REPORT",
     'body': ("DEEP DIVE is real, formulaic plain-language commentary built from the same real numbers shown "
              "elsewhere on the sheet (recent results, 4th-down tendency, VOID when available, a real yardage "
              "matchup edge) -- not a separate stat, just those real numbers written out in a sentence. INJURY "
              "REPORT lists real reported injuries for each team as of this pull, or says plainly when none are "
              "reported. Neither has a numeric poor/avg/good scale -- they're real narrative/status panels, not "
              "measured stats."),
     'scale_note': None},
]

def _glossary_entry_flowable(entry):
    parts = [f"<b>{entry['name']}</b> &ndash; {entry['body']}"]
    if entry.get('formula'):
        parts.append(f'<font color="#555555"><i>Real formula: {entry["formula"]}</i></font>')
    if entry.get('scale_note'):
        parts.append(f'<font color="#1a5276"><b>{entry["scale_note"]}</b></font>')
    elif entry.get('scale'):
        parts.append(f'<font color="#1a5276"><b>{_tercile_scale_line(entry["scale"])}</b></font>')
    text = "<br/><br/>".join(parts)
    p = Paragraph(text, style_gloss_body)
    t = Table([[p]], colWidths=[sum(panel_widths)])
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#AAAAAA')),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    return t

def build_glossary_pages():
    """The full 'HOW TO READ THIS SHEET' appendix -- one entry per real stat
    concept on the sheet, grouped by section, starting on its own fresh
    page(s) at the very end of every matchup PDF. Same real content on
    every matchup (the definitions/significance are fixed text; the real
    poor/avg/good numbers recompute live from df every run), so this is
    built once per PDF, not per-team."""
    els = [PageBreak(), full_width_header("HOW TO READ THIS SHEET"), Spacer(1, 3)]
    intro_text = (
        "Two separate real scales are used across this sheet -- explained here once instead of repeated under "
        "every panel.<br/><br/>"
        f'<b>CELL COLORS</b> (every panel\'s live coloring): each real number is compared to the real 2026-season '
        f'league average for that exact stat. Plain black = within 25% of average (close). '
        f'<font color="{COLOR_GREEN}"><b>GREEN</b></font> / <font color="{COLOR_RED}"><b>RED</b></font> = 25-55% '
        f'off average (a real, meaningful signal) -- and beyond 55% off, that same green/red fills the WHOLE '
        f'CELL (a real outlier). <font color="{COLOR_PURPLE}"><b>PURPLE</b></font> = real top 3% league-wide '
        f'(elite), <font color="{COLOR_ORANGE}"><b>ORANGE</b></font> = real bottom 3% league-wide '
        f'(one of the worst in the league at this specific stat) -- these two always win out over the %-based '
        f'steps above, no matter how small or large the gap looks. Direction flips automatically for defense/'
        f'"allowed" stats, so green (and purple) always means good, red (and orange) always means bad, for '
        f'whichever side it\'s on. Tug o\' War/Edge Pressure and Pass Block vs Pass Rush use a '
        f'related but separate 5-tier real percentile system (WEAK/BELOW AVG/AVERAGE/ABOVE AVG/ELITE) -- see '
        f'their entries below.<br/><br/>'
        '<b>POOR / AVERAGE / GOOD</b> below: a simpler 3-tier real scale built just for this glossary, computed '
        'fresh every time this PDF is built, from where the real 32 current NFL teams actually fall this season '
        'for that exact stat &ndash; bottom third of real teams = POOR, middle third = AVERAGE, top third = GOOD '
        '(direction-aware, so "lower is better" stats like points allowed flip automatically). These numbers '
        'move as the real season plays out, same as every other number on this sheet -- never a fixed, '
        'hardcoded cutoff.'
    )
    intro_table = Table([[Paragraph(intro_text, style_gloss_body)]], colWidths=[sum(panel_widths)])
    intro_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F5F5F5')),
        ('GRID', (0,0), (-1,-1), 0.75, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    els.append(intro_table)
    els.append(Spacer(1, 4))

    last_section = None
    for entry in GLOSSARY_ENTRIES:
        if entry['section'] != last_section:
            if last_section is not None:
                els.append(Spacer(1, 2))
            els.append(full_width_header(entry['section']))
            els.append(Spacer(1, 1.5))
            last_section = entry['section']
        els.append(_glossary_entry_flowable(entry))
        els.append(Spacer(1, 1))
    return els

def build_matchup_header_elements(t_a, t_b, r_a, r_b, ink_saver=False):
    """Builds ONLY the top header panel (team names/records/quality badges,
    WEEK/kickoff/vs/PROJ TOTAL center stack, the placeholder-pairing warning
    row if needed) -- pulled out of build_matchup_elements() so the QUICK
    SCAN sheet ("the old man sheet", see build_quick_scan_elements below)
    can reuse this EXACT same header, unchanged. A pure
    extraction, not a rewrite -- build_matchup_elements() below just calls
    this now instead of inlining it, so the real full sheet's header is
    byte-for-byte the same as before this split.

    ink_saver=True (old man sheet only, via build_quick_scan_elements) drops
    the dark grey fill on just the MIDDLE (WEEK/kickoff/vs/PROJ TOTAL) column to save ink, leaving it plain white with black text
    instead. The two team-name blocks stay exactly as they are on the full
    sheet either way -- only the middle box changes here."""
    els = []

    # --------------------------------------------------------------------------
    # TOP HEADER PANEL v2: serif "Roman" face throughout, team name centered
    # over its own column with the record stacked directly underneath (one
    # block per team -- no more separate outer record flank), 3 columns using
    # panel_widths so each team's header block sits almost directly above
    # that team's stat column in the full-width panels below. (This used to
    # also show ESPN's FPI to the left of each team name -- removed along
    # with the FPI pull itself; see README.txt "LICENSING -- WHAT WAS REMOVED
    # AND WHY". VOID, shown in its own panel right below this one now, is the
    # real power rating on this sheet.)
    # --------------------------------------------------------------------------
    kickoff_str = schedule_info.get((t_a, t_b), "--")
    week_str = f"WEEK {week_number}" if week_number is not None else "--"
    # NEW: projected total points, added right under "vs" ("any way you can
    # have this thing give me a prediction on total points scored each
    # game... just put it under vs in the header"). This is the SAME
    # model_total already being logged every run to
    # nfl_internal_model_track.csv (Step 4b below) -- exp_pts_a + exp_pts_b
    # from predict_matchup, recomputed here fresh rather than reused from
    # Step 4b's matchup_win_probs, same pattern as the quality badge above.
    # Deliberately plain grey text, not turquoise/colored -- this is a raw
    # number, not a threshold-gated confidence signal like the badge or the
    # >=80% team-name highlight, and it's the SAME power-ratings model that
    # was originally kept off this sheet entirely for being an unproven,
    # specific-number forecast (see README.txt "WHAT'S REAL vs. WHAT'S A
    # MODEL ESTIMATE") -- now visible, but real accuracy tracking for it
    # lives in nfl_model_scorecard.py so you can actually see how it does
    # before trusting it.
    pred_for_header = predict_matchup(r_a, r_b)
    center_stack = (
        f'{week_str}<br/>'
        f'<font face="Times-Roman" size="6.8">{kickoff_str}</font><br/>'
        f'<font face="Times-Italic" size="8.5">vs</font><br/>'
        f'<font face="Helvetica" size="6.2" color="#94A3B8">PROJ TOTAL {pred_for_header["total"]:.1f}</font>'
    )

    # >=80% real win probability -> that team's name renders in turquoise.
    # matchup_win_probs is populated once, module-level, right before the
    # singles/bundle build loops call this function -- see "Step 4b" below.
    # win_prob_a is P(t_a/away wins).
    win_prob_a = matchup_win_probs.get((t_a, t_b))
    # Ink_saver (old man sheet only) swaps the near-white base team-name
    # style for a black-text one, since the dark fill behind it is about to
    # be removed too. The turquoise >=80% highlight itself is left exactly
    # as-is either way -- a real signal, not decorative fill.
    base_team_style = style_hdr_team_name_dark if ink_saver else style_hdr_team_name
    style_a = style_hdr_team_name_hot if (win_prob_a is not None and win_prob_a >= 0.80) else base_team_style
    style_b = style_hdr_team_name_hot if (win_prob_a is not None and (1 - win_prob_a) >= 0.80) else base_team_style

    # QUALITY STAT CONSENSUS badge -- "N/M quality stats (XX%)" for each
    # team, shown every week under the record. M = however many of the 12
    # real quality stats had usable data on both sides this week (honestly
    # less than 12 early in the season). The (XX%) is a weighted vote share
    # built from those same N/M votes (see compute_nfl_quality_consensus's
    # docstring and the NFL_QUALITY_STATS comment above) -- NOT a plain N/M
    # ratio, so don't expect e.g. "7/11" to land on exactly 64%. Turquoise
    # only when a team clears QUALITY_CONSENSUS_HOT_THRESHOLD -- deliberately
    # a DIFFERENT signal from the >=80% win-probability team-name color
    # above, not a replacement for it; see the real backtest numbers behind
    # both thresholds in the NFL_QUALITY_STATS comment above.
    # NOTE: these badges are embedded as inline <font color=...> spans
    # inside the SAME Paragraph as the team name/record (ReportLab has no
    # per-run style objects within one Paragraph), so the color has to be
    # set explicitly here rather than via a separate ParagraphStyle --
    # otherwise it would silently inherit the team-name's own >=80% color
    # and the two signals would get visually merged instead of staying
    # independent, which was the whole point of keeping them separate.
    qcount_a, qcount_b, qwin_prob_a = compute_nfl_quality_consensus(r_a, r_b)
    qtotal = qcount_a + qcount_b
    qbadge_color_a = TURQUOISE_HIGHLIGHT if qcount_a >= QUALITY_CONSENSUS_HOT_THRESHOLD else '#94A3B8'
    qbadge_color_b = TURQUOISE_HIGHLIGHT if qcount_b >= QUALITY_CONSENSUS_HOT_THRESHOLD else '#94A3B8'
    if qtotal and qwin_prob_a is not None:
        qpct_a = round(qwin_prob_a * 100)
        qpct_b = 100 - qpct_a
        qbadge_text_a = f"{qcount_a}/{qtotal} quality stats ({qpct_a}%)"
        qbadge_text_b = f"{qcount_b}/{qtotal} quality stats ({qpct_b}%)"
    else:
        qbadge_text_a = "quality stats: --"
        qbadge_text_b = "quality stats: --"

    header_data = [[
        Paragraph(
            f'{t_a}<br/><font face="Times-Roman" size="8.5">REC {r_a["record"]}</font>'
            f'<br/><font face="Helvetica" size="6.5" color="{qbadge_color_a}">{qbadge_text_a}</font>',
            style_a,
        ),
        Paragraph(center_stack, style_hdr_week_big_dark if ink_saver else style_hdr_week_big),
        Paragraph(
            f'{t_b}<br/><font face="Times-Roman" size="8.5">REC {r_b["record"]}</font>'
            f'<br/><font face="Helvetica" size="6.5" color="{qbadge_color_b}">{qbadge_text_b}</font>',
            style_b,
        ),
    ]]
    top_table = Table(header_data, colWidths=HEADER_WIDTHS)
    if ink_saver:
        # Old man sheet only -- NO background fill anywhere in this header
        # (team-name blocks included, not just the middle box like the
        # prior ask) -- plain white, black text, the real GRID border is
        # all that's left to mark the boundaries.
        top_table_style = [
            ('BACKGROUND', (0,0), (-1,-1), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.75, colors.HexColor('#000000')),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]
    else:
        top_table_style = [
            ('BACKGROUND', (0,0), (-1,-1), obsidian_black),
            # Middle (WEEK/kickoff/vs) column colored the same grey as the
            # label column running down every stat panel below it.
            ('BACKGROUND', (1,0), (1,0), center_grey),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.75, colors.HexColor('#000000')),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]
    top_table.setStyle(TableStyle(top_table_style))
    els.append(top_table)

    # If the schedule pull failed entirely this run, `matchups` is an
    # arbitrary alphabetical pairing, not a real scheduled game -- never let
    # that go out looking identical to a real matchup sheet.
    if not matchups_are_real:
        warn_row = [[Paragraph(
            "⚠ PLACEHOLDER PAIRING -- SCHEDULE PULL FAILED THIS RUN, THIS IS NOT A REAL SCHEDULED GAME",
            ParagraphStyle('PlaceholderWarn', fontName='Helvetica-Bold', fontSize=7.5, leading=9,
                           textColor=colors.white, alignment=1))]]
        warn_table = Table(warn_row, colWidths=[sum(panel_widths)])
        warn_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#B00020')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        els.append(warn_table)

    els.append(Spacer(1, 0))
    return els


def build_matchup_elements(t_a, t_b):
    """Builds the full set of flowables for ONE matchup -- every panel,
    header through Fantasy Watch. Used both for the combined full-slate PDF
    and for that matchup's own standalone PDF, so the two always show
    identical content."""
    r_a = df[df['team'] == t_a].iloc[0]
    r_b = df[df['team'] == t_b].iloc[0]
    els = build_matchup_header_elements(t_a, t_b, r_a, r_b)

    # --------------------------------------------------------------------------
    # VOID RATING -- at the very top of the page. This panel now
    # consolidates what used to be THREE separate panels (VOID RATING, GAME
    # PLAN, and NET EPA PER MATCHUP) into one -- see the
    # real-numbers-unchanged, just-relocated note above VOID_PAIRS for the
    # full explanation.
    #
    # net_epa_matchup is the one field here that isn't a plain df column --
    # a real cross-team computation (see the NOTE above VOID_PAIRS), so it's
    # computed here and stashed on throwaway copies of r_a/r_b (r_a_void/
    # r_b_void) made just for this one panel call. r_a/r_b themselves are
    # untouched, still used unmodified by every other panel below.
    # --------------------------------------------------------------------------
    r_a_void = r_a.copy()
    r_b_void = r_b.copy()
    r_a_void['net_epa_matchup'] = r_a['off_efficiency_epa'] * 1.5 - r_b['def_efficiency_epa']
    r_b_void['net_epa_matchup'] = r_b['off_efficiency_epa'] * 1.5 - r_a['def_efficiency_epa']
    els.extend(build_stat_panel("VOID RATING", VOID_PAIRS, r_a_void, r_b_void,
                                 grid_widths=VOID_GRID_WIDTHS, val_style=style_void_val,
                                 pop_style=style_void_pop, label_style=style_void_label,
                                 row_padding=1.5))  # SHRUNK from 2
    els.append(build_net_void_row(r_a_void, r_b_void, t_a, t_b))  # NEW: NET VOID row, see build_net_void_row's writeup above VOID_PAIRS
    # EXPLAINER -- extended to cover the newly-consolidated fields (EPA/NET
    # EFFICIENCY/NET EPA/SUCCESS RATE/PASS-RUSH %), not just VOID, since
    # they all live in one panel now. Still kept as terse as the real
    # content allows -- page 1's pagination budget is tight (see the OC/DC
    # strip fix above) -- full formulas/writeups stay in the glossary.
    els.append(build_note(
        "VOID = our own opponent-adjusted efficiency rating (DVOA-style) &ndash; positive = better than league avg; needs 2+ games played. "
        "EPA = Expected Points Added/play (range ~-0.34 to +0.41, avg ~0.01). NET EPA (MATCHUP) = (this team's off EPA/play x1.5) minus THIS opponent's def EPA/play allowed. "
        "EPA/PLAY (OFF) = this team's own raw offensive EPA/play, unmixed with anything else &ndash; quality stat #2 (61.0% real win rate), second only to Success Rate. "
        "NET EFFICIENCY = this team's own off EPA/play minus its own def EPA/play allowed (not opponent-specific). "
        "SUCCESS RATE (OFF) = real share of plays with EPA &gt; 0 &ndash; the single best real predictor behind the quality-stat badge (63.4% real win rate). "
        "PASS/RUSH PLAY % = this team's real offensive play-mix split. "
        "NET VOID = the TOTAL VOID gap between these two teams, shown on the side with the better number &ndash; "
        "red when that gap size has actually been LOSING more than winning so far (under 0.15, based on one week's real check, small sample). "
        "TEAL fill = the better team's cell when the gap is big enough to be in the band that has been winning: TOTAL VOID gap 0.15+, "
        "NET success rate (off minus def allowed) gap 10.7+ pts, NET points/drive (PPD) gap 0.20+. Early-season samples (31 games for VOID/SR, 15 for PPD) &ndash; a flag, not a guarantee."))
    els.append(Spacer(1, 0))

    # --------------------------------------------------------------------------
    # TEAM RUNDOWN: OFFENSE (left) and DEFENSE (right) side by side -- same
    # half-width columns (side_widths) already used by TRENCH BATTLE/THE
    # DIRTY BIRD/SPECIAL TEAMS BLOCKS below, same font sizes too, just packed
    # into a narrower column. Cuts this section's vertical footprint roughly
    # in half vs. the two full-width panels stacked -- then the full
    # Previous Schedule game log. This is what used to be the OVERALL
    # RATINGS panel, now expanded into the real, un-fragmented team-level
    # rundown.
    # --------------------------------------------------------------------------
    rundown_off = build_rundown_panel("TEAM RUNDOWN -- OFFENSE", TEAM_RUNDOWN_OFFENSE_ROWS, r_a, r_b, widths=RUNDOWN_SIDE_WIDTHS, team_a_label=t_a, team_b_label=t_b)
    rundown_def = build_rundown_panel("TEAM RUNDOWN -- DEFENSE", TEAM_RUNDOWN_DEFENSE_ROWS, r_a, r_b, widths=RUNDOWN_SIDE_WIDTHS, team_a_label=t_a, team_b_label=t_b)

    # OC TENDENCY / DC TENDENCY -- MOVED here (not duplicated) from the old
    # standalone COORDINATOR TENDENCIES panel. Nested in the SAME half-width
    # column as the rundown table it belongs to, so OC sits directly under
    # OFFENSE and DC directly under DEFENSE -- see build_tendency_strip/the
    # writeup above _oc_text.
    oc_strip = build_tendency_strip("OC TENDENCY (HISTORICAL)", _oc_text(r_a), _oc_text(r_b), widths=[135, 135])
    dc_strip = build_tendency_strip("DC TENDENCY (HISTORICAL)", _dc_text(r_a), _dc_text(r_b), widths=[135, 135])

    rundown_side_by_side = Table([[
        [rundown_off] + oc_strip,
        [rundown_def] + dc_strip,
    ]], colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    rundown_side_by_side.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    els.append(rundown_side_by_side)
    els.append(Spacer(1, 0))
    els.extend(build_previous_schedule_panel(t_a, t_b))
    els.append(Spacer(1, 0))

    # NOTE: the standalone "NET EPA PER MATCHUP" panel that used to sit here
    # is GONE -- folded into the VOID RATING panel at the top of the page
    # (NET EPA (MATCHUP) row). See build_net_epa_panel's old definition
    # above (kept, unused, for reference/rollback) and the note above
    # VOID_PAIRS for the real formula, unchanged.

    # --------------------------------------------------------------------------
    # OFFENSE vs DEFENSE matchup panel -- moved here (right under where NET EPA used to sit)
    # and trimmed to 8 rows, so it backs up the NET EPA number immediately above instead of
    # requiring a flip several panels down to find it. Two FIXED side-by-side columns, left
    # always Away, right always Home -- no swapping. Each column pairs that team's own
    # offense against the OTHER team's defense (allowed), so scanning straight down either
    # side always means the same team the whole way down. See MATCHUP_PAIRS above for what's
    # in it and why the other 13 rows it used to carry were dropped (not gone from the sheet
    # -- see that comment).
    # --------------------------------------------------------------------------
    # Real composite OFF_RANK/DEF_RANK (1-32, 1=best) injected into each
    # header ("label the pdf... so it shows actual rank beside them") -- see
    # NFL_POWER_RANKS above. Reference-layer only: these ranks don't change
    # any number in the panel below them, just label it with real context.
    # _rank_suffix degrades to "" (not a guessed number) for any team
    # NFL_POWER_RANKS doesn't have a real entry for.
    away_side = build_stat_panel(
        f"{t_a} OFFENSE{_rank_suffix(t_a, 'off_rank')}  vs  {t_b} DEFENSE{_rank_suffix(t_b, 'def_rank')}",
        MATCHUP_PAIRS, r_a, r_b, widths=side_widths)
    home_side = build_stat_panel(
        f"{t_b} OFFENSE{_rank_suffix(t_b, 'off_rank')}  vs  {t_a} DEFENSE{_rank_suffix(t_a, 'def_rank')}",
        MATCHUP_PAIRS, r_b, r_a, widths=side_widths)

    matchup_side_by_side = Table([[away_side, home_side]], colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    matchup_side_by_side.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),   # centers each narrower panel within its half,
                                                 # opening a visible gap between the two sides
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    els.append(matchup_side_by_side)
    # NOTE: no trailing Spacer here on purpose -- this is immediately
    # followed by an explicit PageBreak() below, and a Spacer that doesn't
    # quite fit in whatever's left of page 1 gets pushed onto its own new
    # page by itself (nearly blank), with the PageBreak right after it then
    # forcing ANOTHER page on top of that -- a real bug this surfaced while
    # tightening page 1 to fit the new OC/DC tendency strips back on.

    # --------------------------------------------------------------------------
    # PAGE BREAK -- everything above is the "how these two teams have played"
    # page; everything below (position breakdowns, tendencies, trenches,
    # special teams, injuries) starts fresh on its own page. NOTE: the
    # formulaic DEEP DIVE commentary used to sit right here -- MOVED (not
    # duplicated) to the bottom of this page, right under THE FIELD GENERAL /
    # THE POUND / THE DIRTY BIRD precision snippets. Still real, still the
    # same panel, just relocated.
    # --------------------------------------------------------------------------
    els.append(PageBreak())

    # --------------------------------------------------------------------------
    # POSITION BREAKDOWNS -- THE CAPTAIN (QB), THE DAWGS (RB), THE FREEBIRDS
    # (WR), THE TANKS (TE). THE CAPTAIN/DAWGS/FREEBIRDS are now real stat
    # GRIDS (see CAPTAIN_PAIRS/DAWGS_PAIRS/FREEBIRDS_PAIRS above) with the
    # real per-player list kept underneath each; THE TANKS (TE) stays the
    # plain player-list panel -- no grid for TE. PBE/PRP, the Tug
    # o' War/Edge Pressure formula, and SECONDARY LOCK INDEX all MOVED in
    # here from their old standalone panels further down the page -- not
    # duplicated, so you won't see any of those three again below.
    # --------------------------------------------------------------------------
    els.extend(build_captain_panel(t_a, t_b, r_a, r_b, players_df, dvp_df))
    els.append(Spacer(1, 3))
    els.extend(build_dawgs_panel(t_a, t_b, r_a, r_b, players_df, dvp_df))
    els.append(Spacer(1, 3))
    els.extend(build_freebirds_panel(t_a, t_b, r_a, r_b, players_df, dvp_df))
    els.append(Spacer(1, 3))
    theme_te, top_n_te = POSITION_PANEL_THEMES['TE']
    els.extend(build_position_panel(theme_te, 'TE', t_a, t_b, players_df, dvp_df, top_n_te))
    els.append(Spacer(1, 3))

    # --------------------------------------------------------------------------
    # NOTE: the standalone ADVANCED METRICS (NGS + FREE PROXIES) and PASSER
    # RATING panels used to sit here -- both GONE (all their real fields are
    # duplicated or moved elsewhere now that THE CAPTAIN/FREEBIRDS/DAWGS exist
    # as grids -- see the comments above ADVANCED_PAIRS/RESIDUAL_OVERALL_PAIRS
    # for exactly where each field lives now, and the one field --
    # team_yac_per_reception -- that has no home left).
    #
    # Scheme tendencies (last game), current injury report, and the Fantasy
    # / Prop Watch panel.
    # --------------------------------------------------------------------------
    # NOTE: SCHEME TENDENCIES (LAST GAME) (TENDENCY_PAIRS) used to sit here
    # -- GONE. PERSONNEL & FORMATION TENDENCIES (SEASON) was already gone
    # before this (see the comment above the old SCHEME_TENDENCY_PAIRS
    # definition for where that real data still lives). COORDINATOR
    # TENDENCIES also used to sit here -- SPLIT (not duplicated) into OC
    # TENDENCY / DC TENDENCY strips now living right under TEAM RUNDOWN --
    # OFFENSE/DEFENSE on page 1 -- see build_tendency_strip and the comment
    # above it.

    # NOTE: the old standalone PRESSURE INDEX (Tug o' War/Edge) and TRENCH
    # BATTLE (PBE vs PRP, plus the RUN SUCCESS RATE vs RUN STOP RATE row)
    # panels used to sit here. PRESSURE INDEX moved into THE CAPTAIN, PBE vs
    # PRP moved into THE FREEBIRDS, and RUN SUCCESS/STOP RATE moved into THE
    # DAWGS (see the position breakdown section above) -- all three real,
    # all three MOVED not duplicated, so TRENCH BATTLE as a standalone panel
    # is retired now that both of its halves live with the position group
    # each is most relevant to.
    #
    # THE DIRTY BIRD used to sit right here too -- MOVED down to sit with
    # THE FIELD GENERAL / THE POUND at the end of this page (all three are
    # the same curated-3-stat "precision snippet" idea, now grouped
    # together in QB/RB/WR order).
    #
    # NOTE: SECONDARY LOCK INDEX used to have its own standalone panel here
    # -- moved into THE FREEBIRDS grid instead (see the position breakdown
    # section above).

    # KICKING & RETURNS: special teams (punting/FG/return averages, both
    # team-vs-team; block pairs use the real offense-vs-opposing-defense
    # shape). NOTE: GAME PLAN used to sit right above this -- MOVED up to
    # right under VOID RATING (see near the top of this function).
    els.extend(build_stat_panel(f"KICKING & RETURNS: {t_a} vs {t_b}", KICKING_PAIRS, r_a, r_b))
    els.append(Spacer(1, 3))
    block_away = build_stat_panel(f"SPECIAL TEAMS BLOCKS: {t_a} vs {t_b} DEF", BLOCK_PAIRS, r_a, r_b, widths=side_widths)
    block_home = build_stat_panel(f"SPECIAL TEAMS BLOCKS: {t_b} vs {t_a} DEF", BLOCK_PAIRS, r_b, r_a, widths=side_widths)
    block_side_by_side = Table([[block_away, block_home]], colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    block_side_by_side.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    els.append(block_side_by_side)
    els.append(Spacer(1, 3))

    # --------------------------------------------------------------------------
    # THE FIELD GENERAL (QB) / THE POUND (RB) / THE DIRTY BIRD (WR) -- three
    # curated-3-stat "precision snippet" panels, grouped together at the end
    # of this page (see the writeup above FIELD_GENERAL_PAIRS/THE_POUND_PAIRS
    # for exactly which 3 and why; THE DIRTY BIRD reuses SKILL_BATTLE_PAIRS,
    # unchanged from before).
    # --------------------------------------------------------------------------
    fg_away = build_stat_panel(f"THE FIELD GENERAL: {t_a} QB vs {t_b} DEF", FIELD_GENERAL_PAIRS, r_a, r_b, widths=side_widths)
    fg_home = build_stat_panel(f"THE FIELD GENERAL: {t_b} QB vs {t_a} DEF", FIELD_GENERAL_PAIRS, r_b, r_a, widths=side_widths)
    fg_side_by_side = Table([[fg_away, fg_home]], colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    fg_side_by_side.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    els.append(fg_side_by_side)
    els.append(Spacer(1, 3))

    pound_away = build_stat_panel(f"THE POUND: {t_a} RBs vs {t_b} FRONT SEVEN", THE_POUND_PAIRS, r_a, r_b, widths=side_widths)
    pound_home = build_stat_panel(f"THE POUND: {t_b} RBs vs {t_a} FRONT SEVEN", THE_POUND_PAIRS, r_b, r_a, widths=side_widths)
    pound_side_by_side = Table([[pound_away, pound_home]], colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    pound_side_by_side.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    els.append(pound_side_by_side)
    els.append(Spacer(1, 3))

    # THE DIRTY BIRD (WR) -- MOVED here (not duplicated) from further up
    # this page -- grouped with THE FIELD GENERAL/THE POUND above, same
    # QB/RB/WR order as THE CAPTAIN/DAWGS/FREEBIRDS. Real, same-scale
    # pairs, no z-score net needed.
    #
    # NOTE: the SPECIAL TEAMS EDGE panel that used to sit here (added last
    # round) is GONE -- it turned out to be pure duplication: its 2 crossed
    # rows were the exact same real numbers as SPECIAL TEAMS BLOCKS above,
    # and PUNT RETURN AVERAGE was already shown in KICKING & RETURNS above
    # that (see the NOTE above KICKING_PAIRS for the full writeup). Special
    # teams genuinely only has 2 real crossed stats -- rather than pad to 3
    # with a repeat, it's just not here as its own precision snippet;
    # SPECIAL TEAMS BLOCKS/KICKING & RETURNS above still show both real
    # numbers exactly once.
    skill_away = build_stat_panel(f"THE DIRTY BIRD: {t_a} WRs/TEs vs {t_b} DBs", SKILL_BATTLE_PAIRS, r_a, r_b, widths=side_widths)
    skill_home = build_stat_panel(f"THE DIRTY BIRD: {t_b} WRs/TEs vs {t_a} DBs", SKILL_BATTLE_PAIRS, r_b, r_a, widths=side_widths)
    skill_side_by_side = Table([[skill_away, skill_home]], colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    skill_side_by_side.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    els.append(skill_side_by_side)
    els.append(Spacer(1, 3))

    # DEEP DIVE -- MOVED here (not duplicated), right under the three
    # precision snippets above. Wrapped in KeepTogether so its header and
    # body never split across a page boundary.
    els.append(KeepTogether(build_deep_dive_commentary(t_a, t_b, r_a, r_b)))
    els.append(Spacer(1, 3))

    # STAT MISMATCH BREAKDOWN -- NEW, right after DEEP DIVEe the most
    # telling. nfl_hybrid_weights is built once, module-level, from the real
    # backtested + documented-prior model (see STAT-WEIGHTED WIN PREDICTION
    # MODEL above) -- right before the singles/bundle build loops call this
    # function, same as matchup_win_probs above.
    mismatch_els = build_stat_mismatch_breakdown(t_a, t_b, r_a, r_b, nfl_hybrid_weights)
    if mismatch_els:
        # Not wrapped in one big KeepTogether -- this section can run to 5
        # category rows and forcing all of it onto one page could crowd or
        # overflow; let it flow across a page break naturally instead, same
        # reasoning as the CFB project's miss-postmortems section.
        els.extend(mismatch_els)
        els.append(Spacer(1, 3))

    els.extend(build_injury_panel(t_a, t_b, injuries_df))
    # NOTE: the old FANTASY / PROP WATCH panel used to close out the sheet
    # here -- replaced by THE CAPTAIN/DAWGS/FREEBIRDS/TANKS at the top of
    # this page (same real per-player data, split by position instead of
    # one mixed top-4-by-PPR list).

    # NOTE: HOW TO READ THIS SHEET (the glossary appendix) used to be
    # appended right here, inside this function -- which meant the
    # full-slate BUNDLE got it 16 times over, once per matchup (112 pages
    # for a 16-game week). It's no longer built per-matchup: each SINGLE
    # matchup PDF still gets exactly one copy (added once, after this
    # function returns, in the singles-build loop below), and the combined
    # bundle gets exactly ONE copy total, appended once after every
    # matchup's panels -- see the bundle/singles build loops near the
    # bottom of this file. Same real build_glossary_pages content either
    # way, just called from one place instead of inside every matchup's
    # own element list.

    return els


# ==============================================================================
# QUICK SCAN SHEET ("the old man sheet") -- a condensed, one-section-per-game
# PDF for printing. Reuses build_matchup_header_elements UNCHANGED (see that
# function's docstring), then a handful of curated stat rows/grids below it
# -- real numbers pulled from the SAME df/columns as the full sheet, just a
# small hand-picked subset, colored with the simpler offense/defense
# head-to-head scheme (see the QUICK SCAN SHEET note above
# build_quick_scan_panel) instead of the full sheet's league-percentile
# tiering.
#
# SECTIONS, each a curated
# 3-4 row grid immediately followed by the SAME precision-snippet grid the
# full sheet already shows for that position (reused verbatim, just
# re-colored via build_quick_scan_panel instead of build_stat_panel):
#   BASE 5        -- League Rank, Total Void, Net EPA (Matchup), Net
#                    Efficiency, Success Rate (Off) -- your original 5-stat ask.
#   THE CAPTAIN   -- CPOE, Passer Rating, Tug o' War (Interior), Edge
#                    Pressure, then THE FIELD GENERAL grid (FIELD_GENERAL_PAIRS).
#   THE DAWGS     -- Rush Play %, RYOE/Rush Attempt, Rush Yards/Play (off),
#                    Rush EPA/Play (off), then THE POUND grid (THE_POUND_PAIRS).
#   THE FREEBIRDS -- Pass Play %, Pass Yards/Play (off), Pressure Rate
#                    (Allowed vs Forced), then THE DIRTY BIRD grid
#                    (SKILL_BATTLE_PAIRS).
#
# UPDATED -- Rush EPA/Play is now real and added: nfl_data_pull.py PART 2 now
# computes and exports real off_rush_epa/def_rush_epa_allowed season-to-date
# columns (same real 'epa' pbp column as EPA/PLAY (OFF) elsewhere, filtered
# to rushing plays only). It's a CROSSED pair in DAWGS_PAIRS (off vs def
# allowed, same shape as every other DAWGS row), but a same-key row here on
# the quick scan sheet (just this team's own number), matching how RUSH
# YARDS/PLAY (OFF) right above it is already handled on this sheet.
#
# ONE REAL STAT STILL LEFT OUT, NOT FAKED: pass EPA under THE FREEBIRDS -- no real pass-only EPA/play column exists
# in nfl_team_matrix.csv yet (only the whole-offense off_efficiency_epa,
# already folded into NET EPA/NET EFFICIENCY in the base 5, and now the
# rush-only off_rush_epa above). A real pass-only split was never
# backtested at all. Adding it would need nfl_data_pull.py extended the
# same way Rush EPA/Play just was -- not done this round, flagged here
# rather than guessed at or approximated from the whole-offense number
# under a misleading label.
# ==============================================================================
QUICK_SCAN_BASE_PAIRS = [p for p in VOID_PAIRS if p[5] in {
    "LEAGUE RANK", "TOTAL VOID", "NET EPA (MATCHUP)", "NET EFFICIENCY", "SUCCESS RATE (OFF)", "NET PPD"}]

QUICK_SCAN_CAPTAIN_PAIRS = [p for p in CAPTAIN_PAIRS if p[5] in {
    "CPOE (COMP % OVER EXPECTED)", "PASSER RATING"}] + [
    # BUG FIX: interior_pressure_rate/edge_pressure_rate are stored as raw
    # fractions (e.g. 0.0200 = 2.0%), same as everywhere else this column is
    # read (see _pressure_row_cells's "{rate * 100:.1f}%" on the full
    # sheet) -- mult must be 100 here, not 1, or a real 2-8% rate rounds
    # down to a misleading "0.0%"/"0.1%" at 1 decimal place.
    ('interior_pressure_rate', 'interior_pressure_rate', "{:.1f}", "%", 100, "TUG O' WAR (INTERIOR, DEF)"),
    ('edge_pressure_rate', 'edge_pressure_rate', "{:.1f}", "%", 100, "EDGE PRESSURE (DEF)"),
]

QUICK_SCAN_DAWGS_PAIRS = [
    ('rush_play_pct_offense', 'rush_play_pct_offense', "{:.1f}", "%", 1, "RUSH PLAY %"),
] + [p for p in DAWGS_PAIRS if p[5] == "RYOE / RUSH ATTEMPT"] + [
    ('off_rushing_yards_per_play', 'off_rushing_yards_per_play', "{:.1f}", "", 1, "RUSH YARDS / PLAY (OFF)"),
    ('off_rush_epa', 'off_rush_epa', "{:.2f}", "", 1, "RUSH EPA/PLAY (OFF)"),
]

QUICK_SCAN_FREEBIRDS_PAIRS = [
    ('pass_play_pct_offense', 'pass_play_pct_offense', "{:.1f}", "%", 1, "PASS PLAY %"),
    ('off_yards_per_pass', 'off_yards_per_pass', "{:.1f}", "", 1, "PASS YARDS / PLAY (OFF)"),
] + [p for p in FREEBIRDS_PAIRS if p[5] == "PRESSURE RATE (ALLOWED vs FORCED)"] + (
    # NEW -- reuses SECONDARY_LOCK_PAIR verbatim (same-key, DEF-vs-DEF, no
    # offensive mirror -- see its original writeup above).
    SECONDARY_LOCK_PAIR
)


def _qs_side_by_side(left_panel, right_panel):
    """Wraps two build_quick_scan_panel results side by side, same
    OUTER_HALF_WIDTH Table pattern used throughout build_matchup_elements
    for FIELD GENERAL/THE POUND/THE DIRTY BIRD etc."""
    t = Table([[left_panel, right_panel]], colWidths=[OUTER_HALF_WIDTH, OUTER_HALF_WIDTH])
    t.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    return t


# ==============================================================================
# NEW: BREAKDOWN grids -- your "create me a blank grind at the very bottom,
# label it BREAKDOWN, and put one on both sides under everything, label one
# pass, and the other rush... for the away team fill in the stats that favor
# them, on the home side grid, fill in the stats that favor them" ask (ties
# follow-up, get NO value on either side -- same as every other tie on this
# sheet).
#
# This is NOT a two-team head-to-head grid like every other panel here --
# it's a ONE-SIDED "advantage checklist" per side, per game-phase (pass/rush).
# Each row is either filled with that side's own real value (only when this
# specific real stat, compared head-to-head against the OTHER team via the
# exact same _qs_favorability/_qs_side logic already proven out above,
# actually favors this side) or left "--" if it doesn't. See
# build_breakdown_grid's docstring below for exactly how a row is decided.
#
# TUG O' WAR and EDGE PRESSURE are intentionally listed in BOTH the pass and
# rush grid, per the stat list for each -- they're real trench
# pressure numbers that matter for both phases, not a copy/paste mistake.
# SACK % is listed under the RUSH grid only because that's where the stat list puts it, even though it's a pass-protection stat everywhere else on this
# sheet -- the stat list is not overridden.
# ==============================================================================
BREAKDOWN_GRID_WIDTHS = [64, 140]   # value + label, sums to 204 -- same total as side_widths

BREAKDOWN_PASS_PAIRS = [
    ('interior_pressure_rate', 'interior_pressure_rate', "{:.1f}", "%", 100, "TUG O' WAR"),
    ('edge_pressure_rate', 'edge_pressure_rate', "{:.1f}", "%", 100, "EDGE PRESSURE"),
    ('completion_pct', 'completion_pct_allowed', "{:.1f}", "%", 1, "COMPLETION %"),
    ('target_separation', 'target_separation_allowed', "{:.1f}", "", 1, "SEPARATION"),
    ('yac_per_reception_pbp', 'yac_allowed_per_reception_pbp', "{:.1f}", "", 1, "YAC"),
    ('secondary_lock_index', 'secondary_lock_index', "{:.1f}", "", 1, "SECONDARY LOCK"),
    ('off_yards_per_pass', 'def_yards_per_pass', "{:.1f}", "", 1, "YARDS / PASS"),
    ('pressure_rate_allowed_pct', 'pressure_rate_forced_pct', "{:.1f}", "%", 1, "PRESSURE RATE"),
]

BREAKDOWN_RUSH_PAIRS = [
    ('interior_pressure_rate', 'interior_pressure_rate', "{:.1f}", "%", 100, "TUG O' WAR"),
    ('edge_pressure_rate', 'edge_pressure_rate', "{:.1f}", "%", 100, "EDGE PRESSURE"),
    ('sack_pct_offense', 'sack_pct_defense', "{:.1f}", "%", 1, "SACK %"),
    ('stuff_rate_offense', 'stuff_rate_defense', "{:.1f}", "%", 1, "STUFF"),
    ('off_run_success_rate', 'def_run_stop_rate', "{:.1f}", "%", 1, "RUN SUCCESS %"),
    ('off_rushing_yards_per_play', 'def_rushing_yards_per_play', "{:.1f}", "", 1, "YARDS / RUN"),
]


def build_breakdown_grid(header_text, pairs, own_row, opp_row, widths=None):
    """A ONE-SIDED advantage checklist (see the BREAKDOWN note above) --
    own_row/opp_row, not left_row/right_row, since only ONE side's answer
    comes out of a single call. Each pairs tuple reads exactly like a
    crossed pair elsewhere on this sheet (own_key = this side's column,
    opp_key = the other side's column for the same real matchup -- for a
    same-key row like TUG O' WAR own_key == opp_key, same as everywhere else
    that stat is used). Calling this once with (row_a, row_b) gets the away
    side's own answer; calling it again with the SAME pairs but (row_b,
    row_a) gets the home side's -- the same swap THE FIELD GENERAL/THE
    POUND/THE DIRTY BIRD already use for their away/home panel pairs, just
    applied to a single-column grid instead of a two-column one.

    A row shows own_row's real value, colored green/orange via _qs_side
    exactly like every other favored cell on this sheet, ONLY when
    _qs_favorability(own) is STRICTLY greater than _qs_favorability(opp).
    A tie (or a missing value on either side) prints "--" here, same as
    everywhere else on this sheet a comparison can't declare a real winner
    -- only the stats that favor this side are shown; nothing is force-filled on a draw."""
    widths = widths or BREAKDOWN_GRID_WIDTHS
    header_row = [[Paragraph(header_text, style_sub_hdr_dark), ""]]
    h_table = Table(header_row, colWidths=widths)
    h_table.setStyle(TableStyle([
        ('SPAN', (0,0), (-1,0)),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LINEBELOW', (0,0), (-1,-1), 1.5, colors.black),
    ]))

    matrix = []
    bg_rows = []
    for own_key, opp_key, fmt, suffix, mult, label in pairs:
        raw_own = own_row.get(own_key) if hasattr(own_row, 'get') else own_row[own_key]
        raw_opp = opp_row.get(opp_key) if hasattr(opp_row, 'get') else opp_row[opp_key]
        own_missing = pd.isna(raw_own)
        opp_missing = pd.isna(raw_opp)
        bg = None
        str_val = "--"
        if not own_missing and not opp_missing:
            eff_own = _qs_favorability(raw_own, own_key)
            eff_opp = _qs_favorability(raw_opp, opp_key)
            if eff_own > eff_opp:
                val = raw_own * mult
                str_val = fmt(val) if callable(fmt) else fmt.format(val) + suffix
                if not callable(fmt):
                    team_own = own_row.get('team') if hasattr(own_row, 'get') else own_row['team']
                    str_val = _with_rank(str_val, own_key, team_own)
                side = _qs_side(own_key)
                if side == 'offense':
                    bg = QUICK_SCAN_FAVOR_GREEN
                elif side == 'defense':
                    bg = QUICK_SCAN_FAVOR_ORANGE
        # Old man sheet only -- black text (style_label_dark), no fill
        # behind it (was center_grey).
        matrix.append([Paragraph(str_val, style_val_base), Paragraph(label, style_label_dark)])
        bg_rows.append([bg, None])

    t = build_centered_table(matrix, widths, bg_colors=bg_rows, border_color='#000000')
    return [h_table, Spacer(1, 1), t]


def _qs_section_divider(text, widths=panel_widths):
    """A header-only divider bar (bold black line underneath, no grid of its
    own directly beneath it) -- used for the top-level "BREAKDOWN" label
    above the PASS/RUSH grids, same decolored old-man-sheet header look as
    build_quick_scan_panel's header (see that function's comment)."""
    row = [[Paragraph(text, style_sub_hdr_dark), "", ""]]
    t = Table(row, colWidths=widths)
    t.setStyle(TableStyle([
        ('SPAN', (0,0), (-1,0)),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LINEBELOW', (0,0), (-1,-1), 1.5, colors.black),
    ]))
    return t


def build_quick_scan_elements(t_a, t_b):
    """Builds ONE game's worth of flowables for the QUICK SCAN sheet ("the
    old man sheet") -- the SAME real header as the full sheet
    (build_matchup_header_elements), then the curated sections described in
    the QUICK SCAN SHEET note above. No PageBreak between games here (see
    the build loop below) -- letting ReportLab's normal pagination pack
    multiple compact games onto one printed page is the whole point of this
    sheet."""
    r_a = df[df['team'] == t_a].iloc[0]
    r_b = df[df['team'] == t_b].iloc[0]
    _TEAL_CTX['a'], _TEAL_CTX['b'] = r_a, r_b
    els = build_matchup_header_elements(t_a, t_b, r_a, r_b, ink_saver=True)

    # BASE 5 -- your original ask. net_epa_matchup isn't a plain df column
    # (a real cross-team computation, see the NOTE above VOID_PAIRS) --
    # computed fresh here on throwaway copies, exactly like the full sheet's
    # VOID RATING panel does.
    r_a_void = r_a.copy()
    r_b_void = r_b.copy()
    r_a_void['net_epa_matchup'] = r_a['off_efficiency_epa'] * 1.5 - r_b['def_efficiency_epa']
    r_b_void['net_epa_matchup'] = r_b['off_efficiency_epa'] * 1.5 - r_a['def_efficiency_epa']
    els.extend(build_quick_scan_panel("QUICK SCAN", QUICK_SCAN_BASE_PAIRS, r_a_void, r_b_void,
                                       grid_widths=VOID_GRID_WIDTHS, row_padding=1.5))
    # NEW: NET VOID row, carried over from the full/standalone sheets --
    # same build_net_void_row used there, just styled to match this
    # sheet's own plain-black-on-white label convention (no grey fill)
    # instead of the full sheet's dark-grey/white-text label.
    els.append(build_net_void_row(r_a_void, r_b_void, t_a, t_b,
                                   val_style=style_val_base, label_style=style_label_dark, label_bg=False))
    els.append(Spacer(1, 2))

    # THE CAPTAIN (curated) + THE FIELD GENERAL grid, reused verbatim.
    els.extend(build_quick_scan_panel(f"THE CAPTAIN: {t_a} vs {t_b}", QUICK_SCAN_CAPTAIN_PAIRS, r_a, r_b))
    els.append(Spacer(1, 1))
    fg_away = build_quick_scan_panel(f"THE FIELD GENERAL: {t_a} QB vs {t_b} DEF", FIELD_GENERAL_PAIRS, r_a, r_b, widths=side_widths)
    fg_home = build_quick_scan_panel(f"THE FIELD GENERAL: {t_b} QB vs {t_a} DEF", FIELD_GENERAL_PAIRS, r_b, r_a, widths=side_widths)
    els.append(_qs_side_by_side(fg_away, fg_home))
    els.append(Spacer(1, 2))

    # THE DAWGS (curated) + THE POUND grid, reused verbatim.
    els.extend(build_quick_scan_panel(f"THE DAWGS: {t_a} vs {t_b}", QUICK_SCAN_DAWGS_PAIRS, r_a, r_b))
    els.append(Spacer(1, 1))
    pound_away = build_quick_scan_panel(f"THE POUND: {t_a} RBs vs {t_b} FRONT SEVEN", THE_POUND_PAIRS, r_a, r_b, widths=side_widths)
    pound_home = build_quick_scan_panel(f"THE POUND: {t_b} RBs vs {t_a} FRONT SEVEN", THE_POUND_PAIRS, r_b, r_a, widths=side_widths)
    els.append(_qs_side_by_side(pound_away, pound_home))
    els.append(Spacer(1, 2))

    # THE FREEBIRDS (curated) + THE DIRTY BIRD grid, reused verbatim.
    els.extend(build_quick_scan_panel(f"THE FREEBIRDS: {t_a} vs {t_b}", QUICK_SCAN_FREEBIRDS_PAIRS, r_a, r_b))
    els.append(Spacer(1, 1))
    skill_away = build_quick_scan_panel(f"THE DIRTY BIRD: {t_a} WRs/TEs vs {t_b} DBs", SKILL_BATTLE_PAIRS, r_a, r_b, widths=side_widths)
    skill_home = build_quick_scan_panel(f"THE DIRTY BIRD: {t_b} WRs/TEs vs {t_a} DBs", SKILL_BATTLE_PAIRS, r_b, r_a, widths=side_widths)
    els.append(_qs_side_by_side(skill_away, skill_home))
    els.append(Spacer(1, 2))

    # BREAKDOWN -- at the very bottom, under everything else. Away's own
    # PASS/RUSH advantage checklists on the left, home's on the right -- see
    # the BREAKDOWN note above build_breakdown_grid for exactly how a row
    # gets filled vs left "--". Wrapped in KeepTogether (same pattern
    # already used for the Deep Dive commentary block) so the "BREAKDOWN"
    # divider never gets orphaned alone at the bottom of a page with its
    # grids pushed to the next one.
    away_pass = build_breakdown_grid(f"{t_a} -- PASS", BREAKDOWN_PASS_PAIRS, r_a, r_b)
    away_rush = build_breakdown_grid(f"{t_a} -- RUSH", BREAKDOWN_RUSH_PAIRS, r_a, r_b)
    home_pass = build_breakdown_grid(f"{t_b} -- PASS", BREAKDOWN_PASS_PAIRS, r_b, r_a)
    home_rush = build_breakdown_grid(f"{t_b} -- RUSH", BREAKDOWN_RUSH_PAIRS, r_b, r_a)
    away_stack = away_pass + [Spacer(1, 1)] + away_rush
    home_stack = home_pass + [Spacer(1, 1)] + home_rush
    breakdown_block = [
        _qs_section_divider(f"BREAKDOWN: {t_a} vs {t_b}"),
        Spacer(1, 1),
        _qs_side_by_side(away_stack, home_stack),
    ]
    els.append(KeepTogether(breakdown_block))

    return els

# ==============================================================================
# DELTA SCAN SHEET -- your 2026-10-04 pivot ask: "what if you took the exact
# same layout as the old man instead, and just figured everything on that
# page as a delta and put it under the side it favors?... on like run% and
# pass% leave on both sides, and anything else that doesn't need a delta...
# can you add ypp and ppd under quick scan, and anything that you can break
# down like cfb into a decimals please do... no need for the breakdown on
# the delta sheet, and anything that signals a heavy lean to one side
# highlight in green... leave the old man alone, but just use the exact
# same layout."
#
# This is a NEW, separate function/output -- build_quick_scan_elements and
# the real "old man sheet" PDF it feeds are completely untouched (see that
# function above). This one reuses:
#   - THE SAME real header (build_matchup_header_elements), untouched.
#   - THE SAME curated section groupings/order/labels as the old man sheet
#     (QUICK SCAN / THE CAPTAIN+FIELD GENERAL / THE DAWGS+POUND / THE
#     FREEBIRDS+DIRTY BIRD), via the SAME QUICK_SCAN_*_PAIRS / FIELD_
#     GENERAL_PAIRS / THE_POUND_PAIRS / SKILL_BATTLE_PAIRS lists already
#     defined above.
#   - THE SAME already-tuned delta math nfl_delta_grid.py uses (reused
#     verbatim via nfl_delta_grid.compute_delta -- same CPOE/Interception/
#     Run Success Rate/etc. formulas, not a second implementation) for
#     every stat this sheet shares with that grid, plus a handful of new
#     "standard" specs (Completion %, Sack %, Total Rush Matchup) built in
#     the exact same off/def-key shape as that file's existing entries.
# Confirmed via AskUserQuestion: a collapsed cell shows the real COMPUTED
# DELTA/GAP (not the raw favored-side value, unlike BREAKDOWN below it on
# the old man sheet -- which is why BREAKDOWN is left off this sheet
# entirely, and the green "heavy lean" highlight is ALWAYS green (not the
# page's usual green=offense/orange=defense convention).
# ==============================================================================

def _delta_scan_lean(anchor_key, raw_delta):
    """Rough 'is this a heavy lean' flag for this sheet's green highlight.
    Not a backtested threshold -- a plain z-score-style read using the real
    league_stds this file already computes off the whole league (same dict
    get_stat_tier()/_qs_favorability() already rely on), scaled by sqrt(2)
    since a delta is the DIFFERENCE of two teams' own values, so its
    natural spread is wider than one team's alone (assuming independence).
    >=1 std dev of that real matchup-gap spread counts as "heavy lean".
    anchor_key is the one real column whose league_std stands in for the
    delta's scale -- exact for a same-key (team_vs_team) row, an honest
    approximation for a row built from two different crossed columns."""
    if raw_delta is None or anchor_key is None:
        return False
    std = league_stds.get(anchor_key)
    if std is None or pd.isna(std) or std == 0:
        return False
    z = abs(raw_delta) / (std * 1.41421356)
    return z >= 1.0


def _delta_scan_row_base(label, raw_delta, fmt="{:.2f}", decimalize=False, anchor_key=None):
    """One collapsed delta row for the DELTA SCAN sheet -- same 3-column
    [left value | label | right value] shape as every other grid on this
    page, but only ONE side ever gets a real number: positive raw_delta
    favors the LEFT team (t_a/r_a), negative favors the RIGHT team (t_b/
    r_b), a tie prints 0.00 on neither/both per the usual convention, and
    a missing input leaves both sides blank -- same "positive favors the
    first team" convention nfl_delta_grid.py's own _delta_row already
    uses, just rendered with this sheet's (the old man sheet's) existing
    plain-black-on-white fonts instead of that standalone grid's.
    decimalize=True divides by 100 before display, for stats whose raw column is a
    0-100 percentage rather than already a small fraction (EPA, VOID,
    interior/edge pressure rate, RYOE, yards-scale stats, etc. are already
    decimal-scale and never set this).
    Returns (row_cells, bg_cells) -- caller collects these across a
    section's rows and hands the lists to build_centered_table() once."""
    if raw_delta is None:
        row = [Paragraph("--", style_val_base), Paragraph(label, style_label_dark), Paragraph("--", style_val_base)]
        return row, [None, None, None]
    disp = (raw_delta / 100.0) if decimalize else raw_delta
    heavy = _delta_scan_lean(anchor_key, raw_delta)
    bg = QUICK_SCAN_FAVOR_GREEN if heavy else None
    txt = fmt.format(abs(disp))
    if raw_delta > 0:
        return [Paragraph(txt, style_val_base), Paragraph(label, style_label_dark), Paragraph("", style_val_base)], [bg, None, None]
    elif raw_delta < 0:
        return [Paragraph("", style_val_base), Paragraph(label, style_label_dark), Paragraph(txt, style_val_base)], [None, None, bg]
    else:
        zero = fmt.format(0)
        return [Paragraph(zero, style_val_base), Paragraph(label, style_label_dark), Paragraph(zero, style_val_base)], [None, None, None]


def _delta_scan_row(label, raw_delta, fmt="{:.2f}", decimalize=False, anchor_key=None):
    """Wraps _delta_scan_row_base and layers the teal flag (see TEAL FLAGS
    above) on the better team's cell for TOTAL VOID / SUCCESS RATE (OFF) /
    PPD when the gap clears its line. Teal can land on the side whose cell
    shows no number if the SR/PPD delta printed here (offense-only) favors
    the other team than the NET stat does -- by design, the teal side is
    the net-stat edge."""
    row, bg = _delta_scan_row_base(label, raw_delta, fmt, decimalize, anchor_key)
    _ts = teal_side(label, _TEAL_CTX['a'], _TEAL_CTX['b'])
    if _ts == 'L':
        bg = [colors.HexColor(COLOR_TEAL_FILL), bg[1], bg[2]]
    elif _ts == 'R':
        bg = [bg[0], bg[1], colors.HexColor(COLOR_TEAL_FILL)]
    return row, bg


def build_delta_scan_elements(t_a, t_b):
    """Builds ONE game's worth of flowables for the DELTA SCAN sheet -- see
    the module note just above for the full rundown. build_quick_scan_
    elements() is NOT called or modified by this function; the two are
    fully independent outputs that happen to share a header and a stat
    selection."""
    r_a = df[df['team'] == t_a].iloc[0]
    r_b = df[df['team'] == t_b].iloc[0]
    _TEAL_CTX['a'], _TEAL_CTX['b'] = r_a, r_b
    els = build_matchup_header_elements(t_a, t_b, r_a, r_b, ink_saver=True)

    # net_epa_matchup isn't a plain df column -- same throwaway-copy
    # computation build_quick_scan_elements already uses for the real
    # QUICK SCAN panel.
    r_a_void = r_a.copy()
    r_b_void = r_b.copy()
    r_a_void['net_epa_matchup'] = r_a['off_efficiency_epa'] * 1.5 - r_b['def_efficiency_epa']
    r_b_void['net_epa_matchup'] = r_b['off_efficiency_epa'] * 1.5 - r_a['def_efficiency_epa']

    def D(key, row1=None, row2=None):
        spec = {"type": "team_vs_team", "key": key}
        return nfl_delta_grid.compute_delta(spec, row1 if row1 is not None else r_a, row2 if row2 is not None else r_b)

    def D_std(away_off, home_def, home_off, away_def):
        spec = {"type": "standard", "away_off": away_off, "home_def": home_def,
                "home_off": home_off, "away_def": away_def}
        return nfl_delta_grid.compute_delta(spec, r_a, r_b)

    def D_inv(home_def, away_off, away_def, home_off):
        spec = {"type": "inverted", "home_def": home_def, "away_off": away_off,
                "away_def": away_def, "home_off": home_off}
        return nfl_delta_grid.compute_delta(spec, r_a, r_b)

    # -------------------- QUICK SCAN (BASE 5 + YPP/PPD, NEW) --------------
    els.extend(build_quick_scan_panel(
        "QUICK SCAN", [p for p in QUICK_SCAN_BASE_PAIRS if p[5] == "LEAGUE RANK"],
        r_a_void, r_b_void, grid_widths=VOID_GRID_WIDTHS, row_padding=1.5))
    qs_rows, qs_bgs = [], []
    for label, delta, fmt, dec, anchor in [
        ("TOTAL VOID",         D('total_void'),                                 "{:.3f}", False, 'total_void'),
        ("NET EPA (MATCHUP)",  D('net_epa_matchup', r_a_void, r_b_void),        "{:.3f}", False, 'off_efficiency_epa'),
        ("NET EFFICIENCY",     D('net_efficiency_epa'),                          "{:.2f}", False, 'net_efficiency_epa'),
        ("SUCCESS RATE (OFF)", D('off_success_rate'),                            "{:.3f}", True,  'off_success_rate'),
        ("YPP (OFF)",          D('off_yards_per_play'),                          "{:.2f}", False, 'off_yards_per_play'),
        ("NET PPD",            D('net_ppd'),                                    "{:.2f}", False, 'net_ppd'),
    ]:
        row, bg = _delta_scan_row(label, delta, fmt, dec, anchor)
        qs_rows.append(row); qs_bgs.append(bg)
    els.append(build_centered_table(qs_rows, VOID_GRID_WIDTHS, bg_colors=qs_bgs, border_color='#000000'))
    els.append(build_net_void_row(r_a_void, r_b_void, t_a, t_b,
                                   val_style=style_val_base, label_style=style_label_dark, label_bg=False))
    els.append(Spacer(1, 2))

    # -------------------- THE CAPTAIN (curated) + THE FIELD GENERAL -------
    cap_rows, cap_bgs = [], []
    for label, delta, fmt, dec, anchor in [
        ("CPOE (COMP % OVER EXPECTED)", D('cpoe'),                              "{:.3f}", True,  'cpoe'),
        ("PASSER RATING",               D('open_source_passer_rating'),         "{:.1f}", False, 'open_source_passer_rating'),
        ("TUG O' WAR (INTERIOR, DEF)",  D('interior_pressure_rate'),            "{:.3f}", False, 'interior_pressure_rate'),
        ("EDGE PRESSURE (DEF)",         D('edge_pressure_rate'),                "{:.3f}", False, 'edge_pressure_rate'),
        ("COMPLETION % (QB vs DEF)",    D_std('completion_pct', 'completion_pct_allowed',
                                              'completion_pct', 'completion_pct_allowed'),  "{:.3f}", True, 'completion_pct'),
        ("SACK % (PROTECTION vs PASS RUSH)", D_std('sack_pct_offense', 'sack_pct_defense',
                                              'sack_pct_offense', 'sack_pct_defense'),      "{:.3f}", True, 'sack_pct_offense'),
        ("INTERCEPTION DELTA",          D_inv('interceptions_gained_per_game', 'interceptions_thrown_per_game',
                                              'interceptions_gained_per_game', 'interceptions_thrown_per_game'),
                                              "{:.2f}", False, 'interceptions_thrown_per_game'),
    ]:
        row, bg = _delta_scan_row(label, delta, fmt, dec, anchor)
        cap_rows.append(row); cap_bgs.append(bg)
    els.append(_qs_section_divider(f"THE CAPTAIN / FIELD GENERAL: {t_a} vs {t_b}"))
    els.append(Spacer(1, 1))
    els.append(build_centered_table(cap_rows, panel_widths, bg_colors=cap_bgs, border_color='#000000'))
    els.append(Spacer(1, 2))

    # -------------------- THE DAWGS (curated) + THE POUND ------------------
    els.extend(build_quick_scan_panel(
        "RUSH PLAY %", [p for p in QUICK_SCAN_DAWGS_PAIRS if p[5] == "RUSH PLAY %"], r_a, r_b))
    els.append(Spacer(1, 1))
    dawgs_rows, dawgs_bgs = [], []
    for label, delta, fmt, dec, anchor in [
        ("RYOE / RUSH ATTEMPT",         D('ryoe_per_attempt'),                   "{:.2f}", False, 'ryoe_per_attempt'),
        ("RUSH YARDS / PLAY (OFF)",     D('off_rushing_yards_per_play'),         "{:.2f}", False, 'off_rushing_yards_per_play'),
        ("RUSH EPA/PLAY (OFF)",         D('off_rush_epa'),                       "{:.2f}", False, 'off_rush_epa'),
        ("TOTAL RUSH MATCHUP (YDS/GM)", D_std('rushing_yards_per_game', 'rushing_yards_allowed_per_game',
                                              'rushing_yards_per_game', 'rushing_yards_allowed_per_game'),
                                              "{:.1f}", False, 'rushing_yards_per_game'),
        ("LINE YARDS (OFF vs DEF ALLOWED)", D_std('off_line_yards', 'def_line_yards_allowed',
                                              'off_line_yards', 'def_line_yards_allowed'),
                                              "{:.2f}", False, 'off_line_yards'),
        ("STUFF RATE",                  D_inv('stuff_rate_defense', 'stuff_rate_offense',
                                              'stuff_rate_defense', 'stuff_rate_offense'), "{:.3f}", True, 'stuff_rate_offense'),
        ("RUN SUCCESS RATE",            D_std('off_run_success_rate', 'def_run_stop_rate',
                                              'off_run_success_rate', 'def_run_stop_rate'), "{:.3f}", True, 'off_run_success_rate'),
    ]:
        row, bg = _delta_scan_row(label, delta, fmt, dec, anchor)
        dawgs_rows.append(row); dawgs_bgs.append(bg)
    els.append(_qs_section_divider(f"THE DAWGS / THE POUND: {t_a} vs {t_b}"))
    els.append(Spacer(1, 1))
    els.append(build_centered_table(dawgs_rows, panel_widths, bg_colors=dawgs_bgs, border_color='#000000'))
    els.append(Spacer(1, 2))

    # -------------------- THE FREEBIRDS (curated) + THE DIRTY BIRD --------
    els.extend(build_quick_scan_panel(
        "PASS PLAY %", [p for p in QUICK_SCAN_FREEBIRDS_PAIRS if p[5] == "PASS PLAY %"], r_a, r_b))
    els.append(Spacer(1, 1))
    fb_rows, fb_bgs = [], []
    for label, delta, fmt, dec, anchor in [
        ("PASS YARDS / PLAY (OFF)",            D('off_yards_per_pass'),         "{:.2f}", False, 'off_yards_per_pass'),
        ("PRESSURE RATE (ALLOWED vs FORCED)",  D_std('pressure_rate_allowed_pct', 'pressure_rate_forced_pct',
                                              'pressure_rate_allowed_pct', 'pressure_rate_forced_pct'),
                                              "{:.3f}", True, 'pressure_rate_allowed_pct'),
        ("SECONDARY LOCK INDEX",               D('secondary_lock_index'),       "{:.2f}", False, 'secondary_lock_index'),
        ("CATCH %",                            D_std('catch_pct', 'catch_pct_allowed',
                                              'catch_pct', 'catch_pct_allowed'), "{:.3f}", True, 'catch_pct'),
        ("YAC / RECEPTION",                    D_std('yac_per_reception_pbp', 'yac_allowed_per_reception_pbp',
                                              'yac_per_reception_pbp', 'yac_allowed_per_reception_pbp'),
                                              "{:.2f}", False, 'yac_per_reception_pbp'),
        ("SEPARATION (YDS)",                   D_std('target_separation', 'target_separation_allowed',
                                              'target_separation', 'target_separation_allowed'),
                                              "{:.2f}", False, 'target_separation'),
    ]:
        row, bg = _delta_scan_row(label, delta, fmt, dec, anchor)
        fb_rows.append(row); fb_bgs.append(bg)
    els.append(_qs_section_divider(f"THE FREEBIRDS / THE DIRTY BIRD: {t_a} vs {t_b}"))
    els.append(Spacer(1, 1))
    els.append(build_centered_table(fb_rows, panel_widths, bg_colors=fb_bgs, border_color='#000000'))

    return els


def team_folder_name(team):
    return team.replace('/', '-')


week_tag = f"week{week_number}" if week_number is not None else datetime.now().strftime("%Y-%m-%d")

# ==============================================================================
# STAT-WEIGHTED WIN PREDICTION MODEL -- record this run's newly-completed
# games (from the real per-game log nfl_data_pull.py just wrote) into the
# persistent history, tune the hybrid measured+prior weights, and build
# nfl_hybrid_weights -- consumed by build_stat_mismatch_breakdown() and
# (indirectly, via matchup_win_probs below) the >=80% team-name highlight.
# Must run before build_matchup_elements() is ever CALLED (module-level
# globals it reads), which is why this sits here, right before the
# singles/bundle build loops -- see STAT-WEIGHTED WIN PREDICTION MODEL
# section above (function definitions) for the full writeup.
# ==============================================================================

print("Step 4a: Updating the real, backtested stat-weighted win prediction model...")
nfl_stat_history = load_nfl_stat_history()
_nfl_newly_recorded = record_nfl_completed_games(nfl_stat_history, game_log_df)
if _nfl_newly_recorded:
    print(f" -> Recorded {_nfl_newly_recorded} newly completed game(s) into the persistent stat history "
          f"({len(nfl_stat_history['games'])} total game(s) on record).")
else:
    print(f" -> No new completed games to record this run ({len(nfl_stat_history['games'])} total game(s) on record).")
_nfl_included_stats, _nfl_model_acc, _nfl_model_correct, _nfl_model_decided, _nfl_model_total = tune_nfl_stat_weights(nfl_stat_history)
if _nfl_model_total:
    print(f" -> Model backtest: {_nfl_model_correct}/{_nfl_model_decided} = "
          f"{(_nfl_model_correct/_nfl_model_decided*100 if _nfl_model_decided else 0):.1f}% accuracy using "
          f"{len(_nfl_included_stats)} measured stat(s), across {_nfl_model_total} total game(s) on record.")
nfl_hybrid_weights = build_nfl_hybrid_weights(_nfl_included_stats)


def _nfl_game_is_final(week_val, team_a, team_b):
    """True once nfl_stat_history["games"] (built just above by
    record_nfl_completed_games, straight from nfl_game_log.csv's own real
    W/L column) already has a real completed result recorded for THIS
    EXACT real matchup -- i.e. this specific game has actually been played,
    not just that some other game that week is done. Used below to freeze
    a completed game's logged prediction -- see the INTERNAL-ONLY comment
    right below for why that freeze exists."""
    pair = {team_a, team_b}
    for g in nfl_stat_history["games"].values():
        if g.get("week") == week_val and {g.get("winner"), g.get("loser")} == pair:
            return True
    return False


# ==============================================================================
# INTERNAL-ONLY: model prediction tracking. The win-probability/spread/total
# panel is NOT shown anywhere in the sold PDF (see README.txt) -- but the model
# behind it still runs here, quietly, writing to a CSV that's for YOUR eyes
# only. Check nfl_internal_model_track.csv against real final scores over a few
# weeks before ever deciding whether this model is good enough to show
# customers. UPDATED: win_prob_a is now also stashed in matchup_win_probs
# (module-level, read by build_matchup_elements) to drive the >=80% turquoise
# team-name highlight directly in the grind-it-out PDF -- the raw
# spread/total/exp-points numbers themselves still stay internal-only in
# nfl_internal_model_track.csv, only the >=80% team-name flag surfaces on the
# sheet itself.
#
# UPDATED: this log now ALSO records the quality-stat consensus badge's
# count/percentage for both teams (quality_count_away/home, quality_total,
# quality_win_prob_away), not just the older win-prob model -- run
# nfl_model_scorecard.py (separate, on-demand script) anytime to auto-fill
# real final scores from nflverse and print a real accuracy readout for BOTH
# signals.
#
# Also fixed three real problems found while checking your actual file: (1)
# every rerun of this script for the same week used to blindly APPEND a
# fresh batch of rows instead of replacing the old ones -- your real
# nfl_internal_model_track.csv had 704 rows logged for a single 16-game
# week from today's reruns alone. Now a rerun for the same (season, week,
# away, home) REPLACES that game's row instead of duplicating it. (2) any
# actual_away_score/actual_home_score already recorded for a game (by hand
# or by nfl_model_scorecard.py) is now explicitly carried over across a
# rerun instead of getting blanked back out to empty -- refreshing this
# week's predictions should never erase a result you already scored.
#
# (3) FIXED: team names highlighted in turquoise before the games
# disappeared on reruns after the games -- confirmed by comparing the old
# vs. new PDFs (BAL/CHI/JAX were all genuine real >=80% turquoise
# picks before their games, and all three lost). The bug: nfl_team_matrix.csv
# gets refreshed every week with that week's real results baked into both
# teams' season stats, so re-running this script AFTER a game is final
# doesn't produce a fresh forecast for that game anymore -- it's hindsight,
# informed by stats that already include the very outcome it's "predicting".
# The old replace-on-rerun logic above (fix #1) was overwriting each game's
# ORIGINAL pre-game prediction with this contaminated post-game one on every
# rerun, silently erasing the real historical record of what the model
# actually called beforehand -- which is exactly why nfl_model_scorecard.py
# reported a clean 6/6 = 100% for the turquoise picks when the real number,
# BAL/CHI/JAX included, was 6/9 = 66.7%.
#
# Fix: once _nfl_game_is_final() (just above, reading the same real
# nfl_stat_history["games"] record record_nfl_completed_games already
# builds from your real nfl_game_log.csv) says a real matchup has actually
# been played AND a prediction was already logged for it before, that row
# is now left COMPLETELY untouched -- this run's freshly recomputed pred is
# thrown away for that one game rather than overwriting history. A game
# that hasn't been played yet still gets refreshed normally on every rerun,
# same as before -- only an already-decided game's original forecast gets
# frozen. NOTE: this can only protect predictions logged FROM NOW ON --
# BAL/CHI/JAX's real original win probabilities were already overwritten
# before this fix existed and aren't recoverable from nfl_internal_model_
# track.csv anymore (their PDFs are still real proof they were turquoise
# picks that lost, just not the exact logged percentage).
# ==============================================================================

print("Step 4b: Logging model predictions internally (not part of the sold PDF -- see README.txt)...")
internal_rows = []
matchup_win_probs = {}
for t_a, t_b in matchups:
    r_a = df[df['team'] == t_a].iloc[0]
    r_b = df[df['team'] == t_b].iloc[0]
    pred = predict_matchup(r_a, r_b)
    matchup_win_probs[(t_a, t_b)] = pred['win_prob_a']
    q_count_a, q_count_b, q_win_prob_a = compute_nfl_quality_consensus(r_a, r_b)
    internal_rows.append({
        'season': int(season_for_schedule),
        'week': week_number,  # real numeric week (None only in the rare "couldn't determine a week" fallback)
        'week_tag': week_tag,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'away': t_a, 'home': t_b,
        'model_exp_pts_away': round(pred['exp_pts_a'], 1), 'model_exp_pts_home': round(pred['exp_pts_b'], 1),
        'model_margin_away_minus_home': round(pred['margin'], 1), 'model_total': round(pred['total'], 1),
        'model_win_prob_away': round(pred['win_prob_a'], 3),
        'quality_count_away': q_count_a, 'quality_count_home': q_count_b,
        'quality_total': q_count_a + q_count_b,
        'quality_win_prob_away': round(q_win_prob_a, 3) if q_win_prob_a is not None else None,
        'actual_away_score': None, 'actual_home_score': None,  # auto-filled by nfl_model_scorecard.py once real, or fill by hand
    })
internal_path = OUT_DIR / 'nfl_internal_model_track.csv'
internal_df = pd.DataFrame(internal_rows)
KEY_COLS = ['season', 'week', 'away', 'home']
if internal_path.exists():
    prior = pd.read_csv(internal_path)
    # Carry over any real result already recorded for a game this run is
    # about to refresh, so re-running the sheet never wipes out a scored
    # outcome -- matched on (season, week, away, home), not week_tag, since
    # week_tag's format isn't always consistent (numeric weeks vs. a
    # fallback date-string) but season+week+matchup always is.
    if not prior.empty and all(c in prior.columns for c in KEY_COLS):
        prior_keyed = prior.dropna(subset=['week'])
        prior_unmatched = prior[prior['week'].isna()] if 'week' in prior.columns else prior.iloc[0:0]
        prior_existing_keys = set(zip(prior_keyed['season'], prior_keyed['week'],
                                       prior_keyed['away'], prior_keyed['home']))

        # NEW (see the INTERNAL-ONLY comment's item (3) above): a real game
        # that's already FINAL, with a prediction already logged for it
        # before, gets its freshly recomputed row dropped here -- the
        # already-logged (real, pre-game) row for that same key simply
        # never gets removed below, so it survives into the final file
        # completely untouched instead of being overwritten.
        frozen_mask = internal_df.apply(
            lambda r: pd.notna(r['week'])
                      and (r['season'], r['week'], r['away'], r['home']) in prior_existing_keys
                      and _nfl_game_is_final(r['week'], r['away'], r['home']),
            axis=1)
        if frozen_mask.any():
            print(f" -> {int(frozen_mask.sum())} already-logged prediction(s) for now-completed real game(s) "
                  f"left untouched (frozen at their original pre-game forecast, not overwritten with a "
                  f"post-game recompute).")
        internal_df = internal_df[~frozen_mask].reset_index(drop=True)

        # Carry over any real result already recorded for a game this run is
        # about to refresh, so re-running the sheet never wipes out a scored
        # outcome -- matched on (season, week, away, home), not week_tag, since
        # week_tag's format isn't always consistent (numeric weeks vs. a
        # fallback date-string) but season+week+matchup always is.
        prior_actuals = prior_keyed.set_index(KEY_COLS)[['actual_away_score', 'actual_home_score']]
        prior_actuals = prior_actuals[~prior_actuals.index.duplicated(keep='last')]
        for i, row in internal_df.iterrows():
            key = (row['season'], row['week'], row['away'], row['home'])
            if pd.notna(row['week']) and key in prior_actuals.index:
                internal_df.at[i, 'actual_away_score'] = prior_actuals.at[key, 'actual_away_score']
                internal_df.at[i, 'actual_home_score'] = prior_actuals.at[key, 'actual_home_score']
        # Drop prior rows for the SAME real games this run just rebuilt (only
        # when week is known -- a None-week fallback row can't be matched
        # reliably, so those are left alone rather than risking a wrong drop).
        # A frozen game's prior row is never in new_keys anymore (its new row
        # was dropped above), so it's correctly kept here rather than dropped.
        new_keys = set(zip(internal_df['season'], internal_df['week'], internal_df['away'], internal_df['home']))
        prior_keyed = prior_keyed[~prior_keyed.apply(lambda r: (r['season'], r['week'], r['away'], r['home']) in new_keys, axis=1)]
        internal_df = pd.concat([prior_unmatched, prior_keyed, internal_df], ignore_index=True)
    elif not prior.empty:
        # Old-format file (no season/week columns yet, from before this
        # update) -- also a one-time cleanup of the duplicate-row problem:
        # collapse any existing (week_tag, away, home) repeats down to the
        # single most recent one (by generated_at) before appending this
        # run's new-format rows, rather than leaving the old bloat in place.
        if all(c in prior.columns for c in ['week_tag', 'away', 'home']):
            if 'generated_at' in prior.columns:
                prior = prior.sort_values('generated_at')
            prior = prior.drop_duplicates(subset=['week_tag', 'away', 'home'], keep='last')
        internal_df = pd.concat([prior, internal_df], ignore_index=True)
internal_df.to_csv(internal_path, index=False)
print(f"Logged {len(internal_rows)} prediction(s) to {internal_path.name} (internal tracking only, "
      f"reruns for the same real game now replace rather than duplicate).")

print("Step 5: Compiling visual layouts onto your PDF canvas(es)...")

if BUILD_BUNDLE:
    bundle_path = GEN_DIR / f"nfl_full_slate_{week_tag}.pdf"
    bundle_elements = []
    for idx, (t_a, t_b) in enumerate(matchups):
        bundle_elements.extend(build_matchup_elements(t_a, t_b))
        if idx < len(matchups) - 1:
            bundle_elements.append(PageBreak())
    # HOW TO READ THIS SHEET -- ONE copy for the whole bundle, appended once
    # at the very end after every matchup's panels (previously repeated once
    # per matchup -- see the note in build_matchup_elements above). No extra
    # PageBreak needed here -- build_glossary_pages already starts with its
    # own, and the loop above deliberately skips a trailing PageBreak after
    # the LAST matchup so there's exactly one break, not two.
    bundle_elements.extend(build_glossary_pages())
    new_doc(bundle_path).build(bundle_elements)
    print(f"[Done] Full-slate bundle PDF: '{bundle_path}'")

if BUILD_SINGLES:
    singles_dir = GEN_DIR / f"matchups_{week_tag}"
    singles_dir.mkdir(exist_ok=True)
    for t_a, t_b in matchups:
        single_path = singles_dir / f"{team_folder_name(t_a)}_at_{team_folder_name(t_b)}.pdf"
        # HOW TO READ THIS SHEET -- one copy per single-matchup PDF (each
        # one is still its own standalone document a customer might buy on
        # its own, so it keeps its own glossary -- just built once here
        # instead of inside build_matchup_elements() itself).
        new_doc(single_path).build(build_matchup_elements(t_a, t_b) + build_glossary_pages())
    print(f"[Done] {len(matchups)} individual matchup PDF(s) in: '{singles_dir}'")

# --------------------------------------------------------------------------
# QUICK SCAN SHEET ("the old man sheet") -- a condensed print copy --
# see the QUICK SCAN SHEET comment above build_quick_scan_elements for
# exactly what's on it. Always built (not gated behind
# --bundle-only/--singles-only -- this is a third, separate document, not a
# variant of either). UPDATED -- now an explicit PageBreak between games
# (same pattern the full-slate bundle loop above already uses), so each game
# starts on its own printed page instead of ReportLab packing several
# compact games onto one sheet. The same idx < len(matchups) - 1 guard as
# the bundle loop skips a trailing blank page after the last game.
# --------------------------------------------------------------------------
print("Step 5b: Compiling the quick scan sheet ('the old man sheet') for printing...")
quick_scan_path = GEN_DIR / f"nfl_old_man_sheet_{week_tag}.pdf"
quick_scan_elements = []
for idx, (t_a, t_b) in enumerate(matchups):
    quick_scan_elements.extend(build_quick_scan_elements(t_a, t_b))
    if idx < len(matchups) - 1:
        quick_scan_elements.append(PageBreak())
new_doc(quick_scan_path).build(quick_scan_elements)
print(f"[Done] Quick scan sheet ('the old man sheet'): '{quick_scan_path}'")

# --------------------------------------------------------------------------
# DELTA SCAN SHEET -- your 2026-10-04 pivot ask: the old man sheet's exact
# layout, but everything that can be a delta collapses to one number under
# the side it favors (Pass Play %/Rush Play %/League Rank stay raw on both
# sides, no BREAKDOWN section here -- see build_delta_scan_elements above
# for the full rundown). build_quick_scan_elements/the real old man sheet
# above this block are completely untouched.
# --------------------------------------------------------------------------
print("Step 5c: Compiling the delta scan sheet (old man layout, everything collapsed to a delta)...")
delta_scan_path = GEN_DIR / f"nfl_delta_scan_sheet_{week_tag}.pdf"
delta_scan_elements = []
for idx, (t_a, t_b) in enumerate(matchups):
    delta_scan_elements.extend(build_delta_scan_elements(t_a, t_b))
    if idx < len(matchups) - 1:
        delta_scan_elements.append(PageBreak())
new_doc(delta_scan_path).build(delta_scan_elements)
print(f"[Done] Delta scan sheet: '{delta_scan_path}'")


print(f"\n[Done] {len(matchups)} matchup(s) processed.")

# --------------------------------------------------------------------------
# NOTE: the standalone 2x2 "NFL Matchups -- Delta Grid Report" PDF that
# used to build here (nfl_delta_grid.build_delta_grid_pdf) is REMOVED per
# your 2026-10-04 "you can get rid of the 2x2 grid" ask -- fully
# superseded by the delta scan sheet above (same collapsed-delta math,
# the old man sheet's real layout instead of a cramped 2x2 grid). The
# nfl_delta_grid module itself is NOT deleted -- its compute_delta()
# function and SECTIONS specs are still imported (see the top of this
# file) and reused by build_delta_scan_elements() for the delta scan
# sheet's math, so it's now a shared formula library rather than its own
# report.
# --------------------------------------------------------------------------

# ==============================================================================
# PART 2 BOUNDARY LINE END
# ==============================================================================
