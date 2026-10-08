# ==============================================================================
# backtest_pressure_index.py
# ==============================================================================
# ONE-TIME validation tool for the Tug o' War / Edge Pressure index (PART 2F
# of nfl_data_pull.py) -- NOT part of the weekly routine, doesn't touch or
# overwrite any of the weekly CSVs. Run it by hand whenever you want to
# re-check the stat's honesty, e.g. after a few seasons go by.
#
# IMPORTANT CORRECTION:
# there is NO way to check "does interior share of sacks+hits match interior
# share of TRUE total pressure (including hurries)" -- because hurries have
# no player or position attribution anywhere in any licensed dataset. There
# is no such thing as "true interior share of total pressure" to check
# against; that number doesn't exist for anyone to compute, free or paid.
# So this script validates the two things that actually ARE checkable:
#
#   CHECK 1 -- Is credited pressure (sacks+half-sacks+QB hits, what the live
#   index is built from) a reasonable team-level stand-in for TRUE total
#   pressure (sacks+hits+hurries, nflverse's real "was_pressure" flag,
#   historical seasons only)? This is a TEAM-level check, not split by
#   position (it can't be -- hurries aren't attributed to a specific
#   defender). If credited pressure correlates strongly with true pressure
#   at the team level, that's real evidence the live proxy is capturing the
#   right signal even though it structurally undercounts.
#
#   CHECK 2 -- Does the Tug o' War / Edge Pressure index actually predict
#   anything on the field? For each historical season, checks whether teams
#   with a higher interior (or edge) CREDITED pressure rate produced worse
#   outcomes for the offenses they faced (lower opponent EPA/play, lower
#   opponent success rate) than teams with a lower rate. This doesn't need
#   the hurries number at all -- it's checking whether the stat as actually
#   built and shown on the sheet is predictive, which is the real question
#   that matters for selling it.
#
# EXTENDED to also validate the TRENCH BATTLE panel (PBE/PRP + Run
# Success/Stop Rate), same real per-season data, same honesty rules:
#
#   CHECK 3 -- Is the live PBE proxy (sacks + 0.75x QB hits allowed per
#   dropback, no hurries) a reasonable team-level stand-in for a
#   hurries-inclusive version (was_pressure ALLOWED, i.e. this team's own
#   QB got pressured)? Mirrors CHECK 1, offense side.
#
#   CHECK 4 -- Do PBE and PRP actually predict real outcomes? PBE vs. this
#   team's own real offensive EPA/play + success rate; PRP vs. real
#   opponent EPA/play + success rate allowed (same outcomes CHECK 2 already
#   uses for interior/edge).
#
#   CHECK 5 -- Do Run Success Rate (offense) and Run Stop Rate (defense)
#   track real EPA/play outcomes? Since "success" is itself EPA>0, a strong
#   correlation here is expected (they're measuring closely related things,
#   not independent phenomena) -- reported honestly as confirmation, not
#   presented as a surprising find.
#
#   CHECK 6 -- Does the TRENCH BATTLE panel's actual net signal (PBE
#   z-score minus PRP z-score, computed the SAME way the live sheet computes
#   it -- vs. that SEASON's real 32-team field, not pooled across seasons)
#   line up with which side of the ball actually outperformed? Checked
#   against each team's own (off_epa - def_epa_allowed) differential -- the
#   most direct real proxy available for "which side of this team's ball is
#   carrying it." This is the one that most directly answers "does the
#   panel's leans-O-LINE/leans-PASS-RUSH call mean anything real."
#
# Everything here is real nflverse data (CC BY 4.0), same as the rest of
# this project. Nothing is faked; correlations that come back weak are
# reported as weak, not hidden.
# ==============================================================================

from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent

# nflverse's participation loader only serves 2016-2025. was_pressure is
# FTN-charted data, which only goes back to whenever FTN's charting
# partnership started (recent seasons) -- seasons without it are skipped
# below with a printed note, not silently dropped.
BACKTEST_SEASONS = [2021, 2022, 2023, 2024, 2025]

INTERIOR_POSITIONS = {'DT', 'NT'}
EDGE_POSITIONS = {'DE', 'OLB'}


def _norm_play_id(series):
    return pd.to_numeric(series, errors='coerce').apply(lambda v: str(int(v)) if pd.notna(v) else None)


print("Loading real historical data for the pressure-index backtest...")
print(f"Seasons requested: {BACKTEST_SEASONS} (any without was_pressure/FTN data will be skipped, not faked)\n")

# One roster pull spanning every backtest season, same pattern as PART 2F --
# each historical player's position looked up for the specific season he
# was credited in, not assumed static across his career.
roster_all = nfl.load_rosters(seasons=BACKTEST_SEASONS).to_pandas()
roster_all = roster_all[roster_all['gsis_id'].notna()].copy()
roster_all['gsis_id'] = roster_all['gsis_id'].astype(str)
roster_all = roster_all.drop_duplicates(subset=['gsis_id', 'season'], keep='last')
# IMPORTANT: nflverse's 'position' column is a broad group ('DL', 'LB',
# etc.) that lumps interior and edge D-linemen together, and edge OLBs in
# with inside/middle LBs -- confirmed against a real pull (2021-2025,
# 15,531 roster rows: 'position' never once produced 'DT'/'NT'/'DE'/'OLB').
# 'depth_chart_position' is the real fine-grained column -- fully populated
# for every DL/LB-group row in that same real pull, with exactly the
# DT/NT/DE/OLB/ILB/MLB/LB labels this index needs. Use it, not 'position'.
pos_lookup = roster_all.set_index(['gsis_id', 'season'])['depth_chart_position'].to_dict()

# Lightweight permanent safety net -- if a future nflverse schema change
# breaks the depth_chart_position match again, this prints the real values
# instead of silently computing all-zero interior/edge events.
n_interior_match = roster_all['depth_chart_position'].isin(INTERIOR_POSITIONS).sum()
n_edge_match = roster_all['depth_chart_position'].isin(EDGE_POSITIONS).sum()
if len(roster_all) > 0 and n_interior_match == 0 and n_edge_match == 0:
    print(f"[diagnostic] neither position set matched real 'depth_chart_position' values -- most common first:\n"
          f"{roster_all['depth_chart_position'].value_counts(dropna=False).head(25).to_string()}\n")

CREDIT_COLS = [
    ('sack_player_id', 1.0), ('half_sack_1_player_id', 0.5), ('half_sack_2_player_id', 0.5),
    ('qb_hit_1_player_id', 1.0), ('qb_hit_2_player_id', 1.0),
]

rows = []
for yr in BACKTEST_SEASONS:
    try:
        print(f"Processing {yr}...")
        pbp_yr = nfl.load_pbp(seasons=[yr]).to_pandas()
        pbp_reg = pbp_yr[(pbp_yr['season_type'] == 'REG') & (pbp_yr['posteam'].notna()) & (pbp_yr['defteam'].notna())].copy()
        pbp_reg['epa'] = pd.to_numeric(pbp_reg['epa'], errors='coerce')
        pbp_reg['pass_attempt'] = pd.to_numeric(pbp_reg.get('pass_attempt', 0), errors='coerce').fillna(0)
        pbp_reg['sack'] = pd.to_numeric(pbp_reg.get('sack', 0), errors='coerce').fillna(0)
        pbp_reg['is_success'] = pbp_reg['epa'] > 0
        pbp_reg['dropback_flag'] = pbp_reg['pass_attempt'] + pbp_reg['sack']
        # Real 'qb_hit' boolean, same as nfl_data_pull.py's PBE/PRP block --
        # needed for the TRENCH BATTLE checks below (PBE/PRP use raw
        # team-level sacks+hits, not the player-CREDITED events above).
        if 'qb_hit' in pbp_reg.columns:
            pbp_reg['qb_hit'] = pd.to_numeric(pbp_reg['qb_hit'], errors='coerce').fillna(0)
        else:
            pbp_reg['qb_hit'] = 0

        drop_yr = pbp_reg.groupby('defteam')['dropback_flag'].sum().rename('dropbacks_faced')
        # Offense side's OWN dropbacks (this team's own pass protection
        # reps) -- same real formula, just grouped by posteam instead.
        drop_off_yr = pbp_reg.groupby('posteam')['dropback_flag'].sum().rename('dropbacks_off')

        # Real opponent outcomes allowed by each defense this season.
        outcomes = pbp_reg[pbp_reg['play_type'].isin(['pass', 'run'])].groupby('defteam').agg(
            def_epa_allowed=('epa', 'mean'), def_success_rate_allowed=('is_success', 'mean')
        )
        # Real OWN offensive outcomes this season -- needed for CHECK 4/6.
        outcomes_off = pbp_reg[pbp_reg['play_type'].isin(['pass', 'run'])].groupby('posteam').agg(
            off_epa=('epa', 'mean'), off_success_rate=('is_success', 'mean')
        )

        # Raw team-level sacks/QB-hits, both sides -- this is what PBE/PRP
        # are actually built from in nfl_data_pull.py (NOT the player-
        # credited sack_player_id/qb_hit_1_player_id columns used for
        # interior/edge above -- PBE/PRP are team-level only).
        sacks_hits_off = pbp_reg.groupby('posteam').agg(sacks_taken=('sack', 'sum'), hits_taken=('qb_hit', 'sum'))
        sacks_hits_def = pbp_reg.groupby('defteam').agg(sacks_forced=('sack', 'sum'), hits_forced=('qb_hit', 'sum'))

        # Real run-play-only success rate, both sides -- Run Success Rate /
        # Run Stop Rate as shown live (def side inverted to "un-success"
        # rate at the very end, same as nfl_data_pull.py does).
        run_plays = pbp_reg[pbp_reg['play_type'] == 'run']
        run_off = run_plays.groupby('posteam')['is_success'].mean().rename('off_run_success_rate')
        run_def = run_plays.groupby('defteam')['is_success'].mean().rename('def_run_success_allowed_rate')

        present_credit_cols = [(c, w) for c, w in CREDIT_COLS if c in pbp_reg.columns]
        credit_frames = []
        for col, weight in present_credit_cols:
            sub = pbp_reg[pbp_reg[col].notna()][['defteam', col]].copy()
            sub.columns = ['defteam', 'player_id']
            sub['player_id'] = sub['player_id'].astype(str)
            sub['weight'] = weight
            credit_frames.append(sub)
        credits_long = pd.concat(credit_frames, ignore_index=True) if credit_frames else pd.DataFrame(columns=['defteam', 'player_id', 'weight'])
        credits_long['position'] = credits_long['player_id'].apply(lambda pid: pos_lookup.get((pid, yr)))

        credited_total = credits_long.groupby('defteam')['weight'].sum().rename('credited_events_total')
        credited_interior = credits_long[credits_long['position'].isin(INTERIOR_POSITIONS)].groupby('defteam')['weight'].sum().rename('interior_credited_events')
        credited_edge = credits_long[credits_long['position'].isin(EDGE_POSITIONS)].groupby('defteam')['weight'].sum().rename('edge_credited_events')

        # Real TOTAL pressure including hurries -- was_pressure, team-level
        # only (can't be split by position -- see the writeup above).
        part_yr = nfl.load_participation(seasons=[yr]).to_pandas()
        if 'was_pressure' not in part_yr.columns:
            print(f"  ⚠ {yr}: no was_pressure column in this pull -- skipping CHECK 1/CHECK 3 for this season (CHECK 2/4/5/6 still run)")
            true_pressure = pd.Series(dtype=float, name='true_pressure_events')
            true_pressure_off = pd.Series(dtype=float, name='true_pressure_off_events')
        else:
            if 'season_type' in part_yr.columns:
                part_yr = part_yr[part_yr['season_type'] == 'REG'].copy()
            gid_col = 'nflverse_game_id' if 'nflverse_game_id' in part_yr.columns else (
                'game_id' if 'game_id' in part_yr.columns else 'old_game_id')
            part_yr = part_yr.copy()
            part_yr['_game_key'] = part_yr[gid_col].astype(str)
            part_yr['_play_key'] = _norm_play_id(part_yr['play_id'])
            pbp_key = pbp_reg[['game_id', 'play_id', 'posteam', 'defteam', 'play_type']].copy()
            pbp_key['_game_key'] = pbp_key['game_id'].astype(str)
            pbp_key['_play_key'] = _norm_play_id(pbp_key['play_id'])
            joined = part_yr.merge(pbp_key[['_game_key', '_play_key', 'posteam', 'defteam', 'play_type']], on=['_game_key', '_play_key'], how='left')
            pass_plays = joined[joined['play_type'] == 'pass']
            true_pressure = pass_plays[pass_plays['was_pressure'] == True].groupby('defteam').size().rename('true_pressure_events')
            # Same real was_pressure flag, grouped by the OFFENSE (posteam)
            # instead -- how often THIS team's own QB got pressured, for
            # CHECK 3's PBE-proxy validation.
            true_pressure_off = pass_plays[pass_plays['was_pressure'] == True].groupby('posteam').size().rename('true_pressure_off_events')

        season_df = pd.concat([
            drop_yr, drop_off_yr, outcomes, outcomes_off, credited_total, credited_interior, credited_edge,
            true_pressure, true_pressure_off, sacks_hits_off, sacks_hits_def, run_off, run_def,
        ], axis=1).reset_index()
        season_df = season_df.rename(columns={'index': 'team'})
        season_df['season'] = yr
        rows.append(season_df)
    except Exception as e:
        print(f"  ⚠ {yr} failed entirely, skipping this season: {e}")

if not rows:
    raise SystemExit("No usable seasons -- can't run the backtest. Check your nflverse/network access and try again.")

df = pd.concat(rows, ignore_index=True).fillna(0)
df['credited_pressure_rate'] = np.where(df['dropbacks_faced'] > 0, df['credited_events_total'] / df['dropbacks_faced'], np.nan)
df['true_pressure_rate'] = np.where(df['dropbacks_faced'] > 0, df['true_pressure_events'] / df['dropbacks_faced'], np.nan)
df['interior_credited_rate'] = np.where(df['dropbacks_faced'] > 0, df['interior_credited_events'] / df['dropbacks_faced'], np.nan)
df['edge_credited_rate'] = np.where(df['dropbacks_faced'] > 0, df['edge_credited_events'] / df['dropbacks_faced'], np.nan)

# TRENCH BATTLE columns -- same real formulas as nfl_data_pull.py's PBE/PRP
# block (sacks + 0.75x QB hits, no hurries), computed here from the same
# raw team-level sack/qb_hit columns.
df['pbe_score'] = np.where(
    df['dropbacks_off'] > 0, 100 - ((df['sacks_taken'] + 0.75 * df['hits_taken']) / df['dropbacks_off'] * 100), np.nan)
df['prp_score'] = np.where(
    df['dropbacks_faced'] > 0, (df['sacks_forced'] + 0.75 * df['hits_forced']) / df['dropbacks_faced'] * 100, np.nan)
# Hurries-inclusive version of the SAME offense-side rate, for CHECK 3.
df['true_pressure_off_rate'] = np.where(df['dropbacks_off'] > 0, df['true_pressure_off_events'] / df['dropbacks_off'], np.nan)
df['true_pbe_score'] = 100 - (df['true_pressure_off_rate'] * 100)
# Run Success Rate (offense) / Run Stop Rate (defense) -- same real EPA>0
# definition as nfl_data_pull.py.
df['off_run_success_rate'] = df['off_run_success_rate'] * 100
df['def_run_stop_rate'] = (1 - df['def_run_success_allowed_rate']) * 100

# The TRENCH BATTLE panel's actual net signal -- PBE z-score minus PRP
# z-score, computed WITHIN each season (vs. that season's real 32-team
# field), exactly matching how the live sheet computes it vs. that week's
# real field. Matchup-level (row_a's PBE vs row_b's PRP) isn't
# reconstructable from team-season data alone without real schedules, so
# this checks each team's OWN net against its OWN real outcome
# differential instead -- see CHECK 6.
season_pbe_mean = df.groupby('season')['pbe_score'].transform('mean')
season_pbe_std = df.groupby('season')['pbe_score'].transform('std').replace(0, np.nan)
season_prp_mean = df.groupby('season')['prp_score'].transform('mean')
season_prp_std = df.groupby('season')['prp_score'].transform('std').replace(0, np.nan)
df['pbe_z'] = (df['pbe_score'] - season_pbe_mean) / season_pbe_std
df['prp_z'] = (df['prp_score'] - season_prp_mean) / season_prp_std
df['trench_net'] = df['pbe_z'] - df['prp_z']
df['epa_net'] = df['off_epa'] - df['def_epa_allowed']

df.to_csv(OUT_DIR / 'nfl_pressure_index_backtest_raw.csv', index=False)
print(f"\nSaved {len(df)} team-season rows to nfl_pressure_index_backtest_raw.csv for your own inspection.\n")

print("=" * 78)
print("CHECK 1 -- does credited pressure (what the live index is built from)")
print("track TRUE total pressure (sacks+hits+hurries, from was_pressure)?")
print("(Team-level only -- hurries can't be split by interior/edge, see header note)")
print("=" * 78)
c1 = df[(df['true_pressure_events'] > 0)][['credited_pressure_rate', 'true_pressure_rate']].dropna()
if len(c1) >= 5:
    r1 = c1['credited_pressure_rate'].corr(c1['true_pressure_rate'])
    print(f"n = {len(c1)} team-seasons -- correlation (r) = {r1:.3f}")
    if r1 >= 0.6:
        print("-> Strong positive relationship: credited pressure is a solid team-level stand-in for true total pressure.")
    elif r1 >= 0.3:
        print("-> Moderate relationship: credited pressure tracks true pressure somewhat, but leaves real gaps.")
    else:
        print("-> Weak relationship: credited pressure alone is not a reliable stand-in for true total pressure at the team level.")
else:
    print(f"Only {len(c1)} usable team-season(s) with was_pressure data -- not enough to report a correlation. Add more seasons to BACKTEST_SEASONS or re-run once nflverse publishes more.")

print()
print("=" * 78)
print("CHECK 2 -- does a higher interior/edge CREDITED pressure rate actually")
print("predict worse outcomes for the offenses faced? (opponent EPA/play,")
print("opponent success rate -- lower is better defense, so expect NEGATIVE r)")
print("=" * 78)
for label, col in [("Interior (Tug o' War)", 'interior_credited_rate'), ("Edge Pressure", 'edge_credited_rate')]:
    sub = df[[col, 'def_epa_allowed', 'def_success_rate_allowed']].replace(0, np.nan).dropna()
    if len(sub) < 10:
        print(f"{label}: only {len(sub)} usable team-seasons -- not enough to report a correlation.")
        continue
    r_epa = sub[col].corr(sub['def_epa_allowed'])
    r_sr = sub[col].corr(sub['def_success_rate_allowed'])
    print(f"{label} (n={len(sub)}): r vs. opponent EPA/play allowed = {r_epa:+.3f}, r vs. opponent success rate allowed = {r_sr:+.3f}")
    if r_epa <= -0.2 or r_sr <= -0.2:
        print(f"  -> Negative correlation as expected: more {label.split(' ')[0].lower()} pressure DOES line up with worse offensive outcomes against that defense.")
    else:
        print(f"  -> No meaningful negative relationship found -- treat {label} as descriptive (real, but not shown to be predictive) until this improves with more data.")

print()
print("=" * 78)
print("CHECK 3 -- does the live PBE proxy (sacks + 0.75x hits allowed per")
print("dropback, no hurries) track a hurries-inclusive version (was_pressure")
print("ALLOWED, this team's own QB pressured)? Offense-side mirror of CHECK 1.")
print("=" * 78)
c3 = df[(df['true_pressure_off_events'] > 0)][['pbe_score', 'true_pbe_score']].dropna()
if len(c3) >= 5:
    r3 = c3['pbe_score'].corr(c3['true_pbe_score'])
    print(f"n = {len(c3)} team-seasons -- correlation (r) = {r3:.3f}")
    if r3 >= 0.6:
        print("-> Strong positive relationship: the live PBE proxy is a solid stand-in for a hurries-inclusive version.")
    elif r3 >= 0.3:
        print("-> Moderate relationship: PBE tracks the true (hurries-inclusive) number somewhat, but leaves real gaps.")
    else:
        print("-> Weak relationship: PBE alone is not a reliable stand-in for a hurries-inclusive pass-block number.")
else:
    print(f"Only {len(c3)} usable team-season(s) with was_pressure data -- not enough to report a correlation.")

print()
print("=" * 78)
print("CHECK 4 -- do PBE and PRP predict real outcomes? PBE vs. this team's")
print("own real offensive EPA/play + success rate; PRP vs. real opponent")
print("EPA/play + success rate allowed (PBE: expect POSITIVE r; PRP: expect")
print("NEGATIVE r, since more pressure created should mean a better defense)")
print("=" * 78)
c4_off = df[['pbe_score', 'off_epa', 'off_success_rate']].replace(0, np.nan).dropna()
if len(c4_off) >= 10:
    r_epa = c4_off['pbe_score'].corr(c4_off['off_epa'])
    r_sr = c4_off['pbe_score'].corr(c4_off['off_success_rate'])
    print(f"PBE (n={len(c4_off)}): r vs. own offensive EPA/play = {r_epa:+.3f}, r vs. own offensive success rate = {r_sr:+.3f}")
    print("  -> Positive as expected: better pass protection lines up with a better offense." if r_epa >= 0.2 or r_sr >= 0.2
          else "  -> No meaningful positive relationship found -- treat PBE as descriptive until this improves with more data.")
else:
    print(f"PBE: only {len(c4_off)} usable team-seasons -- not enough to report a correlation.")
c4_def = df[['prp_score', 'def_epa_allowed', 'def_success_rate_allowed']].replace(0, np.nan).dropna()
if len(c4_def) >= 10:
    r_epa = c4_def['prp_score'].corr(c4_def['def_epa_allowed'])
    r_sr = c4_def['prp_score'].corr(c4_def['def_success_rate_allowed'])
    print(f"PRP (n={len(c4_def)}): r vs. opponent EPA/play allowed = {r_epa:+.3f}, r vs. opponent success rate allowed = {r_sr:+.3f}")
    print("  -> Negative as expected: more pressure created DOES line up with worse offensive outcomes against that defense." if r_epa <= -0.2 or r_sr <= -0.2
          else "  -> No meaningful negative relationship found -- treat PRP as descriptive until this improves with more data.")
else:
    print(f"PRP: only {len(c4_def)} usable team-seasons -- not enough to report a correlation.")

print()
print("=" * 78)
print("CHECK 5 -- do Run Success Rate / Run Stop Rate track real EPA/play")
print("outcomes? Expected to correlate strongly (success = EPA>0 by")
print("definition), so this is confirmation the pieces agree, not an")
print("independent surprise finding.")
print("=" * 78)
for label, col, outcome_col, outcome_label in [
    ("Run Success Rate (offense)", 'off_run_success_rate', 'off_epa', 'own offensive EPA/play'),
    ("Run Stop Rate (defense)", 'def_run_stop_rate', 'def_epa_allowed', 'opponent EPA/play allowed'),
]:
    sub = df[[col, outcome_col]].replace(0, np.nan).dropna()
    if len(sub) < 10:
        print(f"{label}: only {len(sub)} usable team-seasons -- not enough to report a correlation.")
        continue
    r = sub[col].corr(sub[outcome_col])
    print(f"{label} (n={len(sub)}): r vs. {outcome_label} = {r:+.3f}")

print()
print("=" * 78)
print("CHECK 6 -- the one that matters most: does the TRENCH BATTLE panel's")
print("actual net signal (PBE z-score minus PRP z-score, computed the SAME")
print("way the live sheet computes it) line up with which side of the ball")
print("really carried that team (off_epa - def_epa_allowed)? Expect POSITIVE")
print("r -- a net that leans O-LINE should show up as offense outperforming")
print("defense, and vice versa.")
print("=" * 78)
c6 = df[['trench_net', 'epa_net']].replace(0, np.nan).dropna()
if len(c6) >= 10:
    r6 = c6['trench_net'].corr(c6['epa_net'])
    print(f"n = {len(c6)} team-seasons -- correlation (r) = {r6:.3f}")
    if r6 >= 0.3:
        print("-> Real positive relationship: the panel's leans-O-LINE/leans-PASS-RUSH call does track which side of the ball actually carried that team.")
    elif r6 >= 0.1:
        print("-> Weak-to-moderate relationship: some real signal, but treat the net as a rough descriptive lean, not a strong predictor.")
    else:
        print("-> No meaningful relationship found -- the net (as currently built) does not reliably track which side carried the team. Treat it as descriptive only.")
else:
    print(f"Only {len(c6)} usable team-seasons -- not enough to report a correlation.")

print("\nDone. Raw team-season data saved for your own review in nfl_pressure_index_backtest_raw.csv.")
