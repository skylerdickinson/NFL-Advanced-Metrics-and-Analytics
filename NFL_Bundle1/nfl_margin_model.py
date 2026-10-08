# ==============================================================================
# nfl_margin_model.py
# ==============================================================================
# Real, data-fitted margin/total/win-probability model for the NFL personal
# sheet's score predictions (the "PROJ TOTAL" header number, the >=80% real
# win-probability turquoise team-name highlight, and the WIN PROBABILITY /
# PROJECTED TEAM TOTAL / MODEL SPREAD panel -- all fed by predict_matchup()
# in nfl_matchup_sheet.py).
#
# WHY THIS EXISTS -- 2026-09-27: predict_matchup used to convert a
# hand-weighted power rating (OFFENSE_RATING_WEIGHTS/DEFENSE_RATING_WEIGHTS,
# still in nfl_data_pull.py -- ", these stats carry the most weight", never
# fit to any real outcome) into points via two constants that WERE real-fit
# on 2026-09-22 (research_and_backtests/nfl_winprob_calibration_backtest.py)
# -- but those two constants only calibrate the OUTPUT of an already
# hand-picked rating; the weights deciding which stats matter, and how much,
# were still a guess. Real week-2 result, checked against
# nfl_internal_model_track.csv: 6/16 (37.5%) straight-up picks -- worse than
# a coin flip -- 17.3 pt average margin error, and two games given 80%+/86%+
# real win-probability confidence that lost outright (JAX 95.2%, NYG 86.4%).
# That's also very likely why NO team hit the >=80% turquoise-highlight bar
# in week 3's real run -- the 2026-09-22 constant recalibration alone made
# margins/win-probs noticeably more conservative, without fixing the weights
# feeding them.
#
# THIS MODULE replaces the whole rating-then-convert chain with ONE real,
# directly-fit regression: 20 raw team-level inputs (this game's away
# team's and home team's own real season-to-date PPG, yards/play, EPA/play,
# success rate, and turnover margin -- offense AND defense side, the exact
# same 10 categories the old hand-picked rating used, just never collapsed
# into one hand-weighted number first) -> real least squares, fit straight
# to real margin (away_score - home_score) and real total
# (away_score + home_score), against real walk-forward-safe historical
# games (a team's inputs for a given game are only ever built from games
# that team already played BEFORE that game -- same no-leakage rule as
# nfl_winprob_calibration_backtest.py / nfl_walkforward_backtest.py).
#
# TRAINING DATA: the 5 most recently completed real NFL seasons (pulled
# once via nflreadpy and cached to disk -- see HISTORICAL_CACHE_PATH -- so
# this doesn't re-pull ~1,000+ real games of play-by-play every week) PLUS
# this season's own real games so far (pulled fresh every time this runs --
# cheap, same real nflreadpy source nfl_data_pull.py already pulls weekly
# anyway). That means the model keeps folding in real 2026 results as the
# season goes -- not a one-time static backtest applied forever to new
# weekly data on the other projects (CFB/MLB).
#
# HONEST HOLDOUT CHECK: since every historical season happened before this
# one, fitting ONLY on the historical seasons and then checking accuracy
# against THIS season's own already-completed real games is a genuine
# out-of-sample test -- the model never saw any 2026 result while fitting
# those weights. Both that holdout number and the final production fit's
# in-sample number (which DOES include 2026-so-far, for the best live
# predictions) get printed and saved -- see fit_and_save().
#
# USAGE:
#   python nfl_margin_model.py                    # fit + save (~1-2 min
#                                                   # after the first run --
#                                                   # historical seasons
#                                                   # cache to disk below)
#   python nfl_margin_model.py --refresh-historical   # force re-pull of the
#                                                   # cached historical
#                                                   # seasons too (only
#                                                   # needed ~once a year,
#                                                   # once a season wraps)
#
# nfl_matchup_sheet.py calls get_or_refresh_fit() itself automatically (see
# predict_matchup()) -- it reuses the cached fit as long as it's under
# MAX_FIT_AGE_DAYS old, so a normal weekly PDF run does NOT re-pull the
# historical seasons; it only refreshes the cheap current-season portion.
# Run this file by hand any time you want to force an update sooner.
# ==============================================================================

import argparse
import datetime
import json
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
FIT_PATH = OUT_DIR / "nfl_margin_model_fit.json"
HISTORICAL_CACHE_PATH = OUT_DIR / "nfl_margin_historical_cache.csv"
MAX_FIT_AGE_DAYS = 7
MIN_PRIOR_GAMES = 2  # same "needs 2+ real prior games" rule as the rest of this project

# Raw per-game stat categories -- the same 10 (5 offense + 5 defense) the
# old hand-picked OFFENSE_RATING_WEIGHTS/DEFENSE_RATING_WEIGHTS used, just
# never collapsed into one hand-weighted number here -- each stays its own
# real regressor and gets its own real fit coefficient instead. No redzone
# efficiency (same disclosed limitation as nfl_winprob_calibration_backtest.py
# -- real redzone drive detection needs per-drive yardline tracking not
# built in this walk-forward pipeline).
# (label, per_game_col, season-to-date col in nfl_team_matrix.csv)
STAT_CATEGORIES = [
    ("PPG", "pts_for", "points_per_game"),
    ("Yards/Play (Off)", "off_yards_per_play_game", "off_yards_per_play"),
    ("EPA/Play (Off)", "off_epa_play_game", "off_efficiency_epa"),
    ("Success Rate (Off)", "off_success_rate_game", "off_success_rate"),
    ("Turnover Margin (Off, giveaways)", "off_turnovers_game", "turnover_margin_offense"),
    ("Points Allowed/Game", "pts_against", "points_allowed_per_game"),
    ("Yards/Play Allowed (Def)", "def_yards_per_play_game", "def_yards_per_play"),
    ("EPA/Play Allowed (Def)", "def_epa_play_allowed_game", "def_efficiency_epa"),
    ("Success Rate Allowed (Def)", "def_success_rate_allowed_game", "def_success_rate"),
    ("Turnover Margin (Def, takeaways)", "def_turnovers_game", "turnover_margin_defense"),
]
PER_GAME_COLS = [c[1] for c in STAT_CATEGORIES]


def _per_game(df_src, group_cols, value_col, out_col, agg='mean'):
    g = df_src.groupby(group_cols)[value_col].agg(agg).reset_index()
    return g.rename(columns={value_col: out_col})


def _load_season_team_games(yr):
    """Real per-team-game rows for one real season (yr), with the 10 real
    per-game stat columns in STAT_CATEGORIES filled in -- same computation
    as nfl_winprob_calibration_backtest.py's loader, factored out here so
    it's shared by BOTH the historical cache and the live current season
    without duplicating the pbp-parsing logic in two files."""
    print(f"  Loading real {yr} season play-by-play (nflverse)...")
    pbp = nfl.load_pbp(seasons=[yr]).to_pandas()
    pbp_reg = pbp[(pbp['season_type'] == 'REG') & (pbp['posteam'].notna()) & (pbp['defteam'].notna())].copy()
    if pbp_reg.empty:
        print(f"  (no real regular-season plays returned for {yr} yet -- skipping)")
        return pd.DataFrame()

    for field in ['yards_gained', 'epa', 'interception', 'fumble_lost']:
        if field in pbp_reg.columns:
            pbp_reg[field] = pd.to_numeric(pbp_reg[field], errors='coerce').fillna(0)
    pbp_reg['is_success'] = pbp_reg['epa'] > 0
    pbp_clean = pbp_reg[(pbp_reg['play_type'].isin(['pass', 'run'])) & (pbp_reg['epa'].notna())].copy()

    final_scores = pbp_reg.groupby('game_id').agg(
        home=('home_team', 'first'), away=('away_team', 'first'),
        home_score=('home_score', 'last'), away_score=('away_score', 'last'),
    ).reset_index()
    game_weeks = pbp_reg.groupby('game_id')['week'].first().reset_index()

    yards = pbp_clean.groupby(['posteam', 'game_id'])['yards_gained'].sum().reset_index().rename(
        columns={'posteam': 'team', 'yards_gained': 'off_yards'})
    yards_allowed = pbp_clean.groupby(['defteam', 'game_id'])['yards_gained'].sum().reset_index().rename(
        columns={'defteam': 'team', 'yards_gained': 'def_yards_allowed'})
    off_plays = pbp_clean.groupby(['posteam', 'game_id']).size().reset_index(name='off_plays').rename(columns={'posteam': 'team'})
    def_plays = pbp_clean.groupby(['defteam', 'game_id']).size().reset_index(name='def_plays').rename(columns={'defteam': 'team'})
    off_epa_g = _per_game(pbp_clean, ['posteam', 'game_id'], 'epa', 'off_epa_play_game').rename(columns={'posteam': 'team'})
    def_epa_g = _per_game(pbp_clean, ['defteam', 'game_id'], 'epa', 'def_epa_play_allowed_game').rename(columns={'defteam': 'team'})
    off_sr_g = _per_game(pbp_clean, ['posteam', 'game_id'], 'is_success', 'off_success_rate_game').rename(columns={'posteam': 'team'})
    off_sr_g['off_success_rate_game'] *= 100
    def_sr_g = _per_game(pbp_clean, ['defteam', 'game_id'], 'is_success', 'def_success_rate_allowed_game').rename(columns={'defteam': 'team'})
    def_sr_g['def_success_rate_allowed_game'] *= 100

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
    team_games['is_home'] = team_games['team'] == team_games['home_team']
    team_games = team_games.merge(
        final_scores[['game_id', 'home', 'away', 'home_score', 'away_score']], on='game_id', how='left')
    team_games['pts_for'] = np.where(team_games['is_home'], team_games['home_score'], team_games['away_score'])
    team_games['pts_against'] = np.where(team_games['is_home'], team_games['away_score'], team_games['home_score'])

    for extra_df in [
        yards, yards_allowed, off_plays, def_plays, off_epa_g, def_epa_g, off_sr_g, def_sr_g,
        off_to_g[['team', 'game_id', 'off_turnovers_game']], def_to_g[['team', 'game_id', 'def_turnovers_game']],
    ]:
        team_games = team_games.merge(extra_df, on=['team', 'game_id'], how='left')

    team_games['off_yards_per_play_game'] = np.where(
        team_games['off_plays'] > 0, team_games['off_yards'] / team_games['off_plays'], np.nan)
    team_games['def_yards_per_play_game'] = np.where(
        team_games['def_plays'] > 0, team_games['def_yards_allowed'] / team_games['def_plays'], np.nan)

    keep_cols = ['team', 'game_id', 'season', 'week', 'home_team', 'away_team',
                 'is_home', 'home_score', 'away_score'] + PER_GAME_COLS
    return team_games[keep_cols]


def _walk_forward_games(team_games_all):
    """Turns real per-team-game rows into real walk-forward matchup rows:
    for each real (season, week), every team's rolling average uses ONLY
    that team's own real games in STRICTLY EARLIER weeks of the SAME
    season (never a future game, never a different season -- same
    no-leakage rule as nfl_winprob_calibration_backtest.py /
    nfl_walkforward_backtest.py). Returns one row per real completed game,
    with away_<cat>/home_<cat> = that team's real rolling per-game average
    walking INTO this game, plus the real final away_score/home_score."""
    if team_games_all.empty:
        return pd.DataFrame()
    rows = []
    for (season, week), week_games in team_games_all.groupby(['season', 'week']):
        prior = team_games_all[(team_games_all['season'] == season) & (team_games_all['week'] < week)]
        if prior.empty:
            continue
        game_counts = prior.groupby('team').size()
        eligible = set(game_counts[game_counts >= MIN_PRIOR_GAMES].index)
        if len(eligible) < 2:
            continue
        rolling = prior[prior['team'].isin(eligible)].groupby('team')[PER_GAME_COLS].mean()

        seen = set()
        for _, g in week_games.iterrows():
            if g['game_id'] in seen:
                continue
            seen.add(g['game_id'])
            home_team, away_team = g['home_team'], g['away_team']
            if home_team not in eligible or away_team not in eligible:
                continue
            home_score, away_score = g['home_score'], g['away_score']
            if pd.isna(home_score) or pd.isna(away_score) or home_score == away_score:
                continue  # real ties excluded, same as the rest of this project's backtests
            row = {'season': season, 'week': week, 'away': away_team, 'home': home_team,
                   'away_score': away_score, 'home_score': home_score}
            for cat in PER_GAME_COLS:
                row[f'away_{cat}'] = rolling.loc[away_team, cat]
                row[f'home_{cat}'] = rolling.loc[home_team, cat]
            rows.append(row)
    return pd.DataFrame(rows)


def _feature_cols():
    return [f'away_{c}' for c in PER_GAME_COLS] + [f'home_{c}' for c in PER_GAME_COLS]


def _build_design_matrix(games_df):
    X = games_df[_feature_cols()].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(X)), X])  # intercept first


def _fit_ols(X, y):
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coefs


def _fit_win_prob_scale(margin_pred, real_away_win):
    """Real numerical log-loss minimization (no sklearn/scipy, same
    convention as the rest of this codebase) -- win_prob_a = P(away wins),
    matching predict_matchup()'s own convention."""
    def _log_loss(scale):
        p = 1 / (1 + np.exp(-margin_pred / scale))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(real_away_win * np.log(p) + (1 - real_away_win) * np.log(1 - p))

    _scan = np.linspace(1.0, 30.0, 291)
    _losses = [_log_loss(s) for s in _scan]
    _best = int(np.argmin(_losses))
    _lo, _hi = _scan[max(_best - 1, 0)], _scan[min(_best + 1, len(_scan) - 1)]
    _gr = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        _c, _d = _hi - _gr * (_hi - _lo), _lo + _gr * (_hi - _lo)
        if _log_loss(_c) < _log_loss(_d):
            _hi = _d
        else:
            _lo = _c
    return float((_lo + _hi) / 2)


def _default_historical_seasons():
    yr = datetime.date.today().year
    return [y for y in range(yr - 5, yr) if y >= 2016]


def _load_historical_cache(seasons, refresh=False):
    if HISTORICAL_CACHE_PATH.exists() and not refresh:
        cached = pd.read_csv(HISTORICAL_CACHE_PATH)
        if set(seasons).issubset(set(cached['season'].unique())):
            print(f"Using cached real historical games ({HISTORICAL_CACHE_PATH.name}, {len(cached)} team-game rows).")
            return cached[cached['season'].isin(seasons)]
    print(f"Pulling/caching real historical seasons {seasons} (one-time cost -- "
          f"cached to {HISTORICAL_CACHE_PATH.name} after this)...")
    frames = [_load_season_team_games(yr) for yr in seasons]
    frames = [f for f in frames if not f.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(HISTORICAL_CACHE_PATH, index=False)
    return combined


def fit_and_save(historical_seasons=None, refresh_historical=False):
    historical_seasons = historical_seasons or _default_historical_seasons()
    current_year = datetime.date.today().year

    historical_team_games = _load_historical_cache(historical_seasons, refresh=refresh_historical)
    print(f"Pulling real {current_year} season-to-date play-by-play (current season, refreshed every run)...")
    current_team_games = _load_season_team_games(current_year)

    historical_matchups = _walk_forward_games(historical_team_games)
    current_matchups = _walk_forward_games(current_team_games)

    print(f"\nReal walk-forward-safe games: {len(historical_matchups)} historical ({historical_seasons}), "
          f"{len(current_matchups)} from {current_year} so far.\n")

    if len(historical_matchups) < 100:
        print("Too few real historical games to fit anything meaningful -- widen historical_seasons and rerun.")
        return None

    # ------------------------------------------------------------------
    # HONEST HOLDOUT: fit on historical seasons ONLY, then check real
    # accuracy against this season's OWN already-completed real games --
    # a genuine out-of-sample test, since none of this season's results
    # were anywhere near the training data.
    # ------------------------------------------------------------------
    holdout = None
    if len(current_matchups) >= 5:
        X_hist = _build_design_matrix(historical_matchups)
        y_margin_hist = (historical_matchups['away_score'] - historical_matchups['home_score']).to_numpy()
        coefs_holdout = _fit_ols(X_hist, y_margin_hist)
        X_cur = _build_design_matrix(current_matchups)
        margin_pred_cur = X_cur @ coefs_holdout
        actual_margin_cur = (current_matchups['away_score'] - current_matchups['home_score']).to_numpy()
        pick_correct = (margin_pred_cur > 0) == (actual_margin_cur > 0)
        holdout = {
            'n': int(len(current_matchups)),
            'straight_up_accuracy': float(np.mean(pick_correct)),
            'mae': float(np.mean(np.abs(margin_pred_cur - actual_margin_cur))),
        }
        print(f"HONEST HOLDOUT (fit on {historical_seasons} ONLY, tested on {current_year}'s "
              f"{holdout['n']} real completed game(s) it never saw): "
              f"{int(np.sum(pick_correct))}/{holdout['n']} = {holdout['straight_up_accuracy']:.1%} straight-up, "
              f"avg margin error {holdout['mae']:.2f} pts.\n")
    else:
        print(f"Fewer than 5 real completed {current_year} games so far -- skipping the holdout check this run.\n")

    # ------------------------------------------------------------------
    # PRODUCTION FIT: historical + current season combined -- the one
    # actually used for live predictions, using every real game available.
    # ------------------------------------------------------------------
    all_matchups = pd.concat([historical_matchups, current_matchups], ignore_index=True) \
        if len(current_matchups) else historical_matchups
    X_all = _build_design_matrix(all_matchups)
    y_margin_all = (all_matchups['away_score'] - all_matchups['home_score']).to_numpy()
    y_total_all = (all_matchups['away_score'] + all_matchups['home_score']).to_numpy()
    coefs_margin = _fit_ols(X_all, y_margin_all)
    coefs_total = _fit_ols(X_all, y_total_all)

    margin_pred_all = X_all @ coefs_margin
    real_away_win = (all_matchups['away_score'] > all_matchups['home_score']).astype(float).to_numpy()
    win_prob_scale = _fit_win_prob_scale(margin_pred_all, real_away_win)

    p_all = np.clip(1 / (1 + np.exp(-margin_pred_all / win_prob_scale)), 1e-6, 1 - 1e-6)
    pick_correct_all = (margin_pred_all > 0) == (real_away_win == 1)
    in_sample = {
        'n': int(len(all_matchups)),
        'straight_up_accuracy': float(np.mean(pick_correct_all)),
        'mae': float(np.mean(np.abs(margin_pred_all - y_margin_all))),
        'log_loss': float(-np.mean(real_away_win * np.log(p_all) + (1 - real_away_win) * np.log(1 - p_all))),
    }
    print(f"PRODUCTION FIT (in-sample, {in_sample['n']} real games, historical + {current_year}-so-far combined): "
          f"{in_sample['straight_up_accuracy']:.1%} straight-up, avg margin error {in_sample['mae']:.2f} pts, "
          f"log-loss {in_sample['log_loss']:.4f}, win_prob_scale={win_prob_scale:.2f}\n")

    fit = {
        'fit_date': datetime.datetime.now().isoformat(),
        'historical_seasons': historical_seasons,
        'n_historical_games': int(len(historical_matchups)),
        'n_current_season_games': int(len(current_matchups)),
        'feature_cols': _feature_cols(),
        'coefs_margin': coefs_margin.tolist(),
        'coefs_total': coefs_total.tolist(),
        'win_prob_scale': win_prob_scale,
        'in_sample': in_sample,
        'holdout': holdout,
    }
    with open(FIT_PATH, 'w') as f:
        json.dump(fit, f, indent=2)
    print(f"Saved real fitted margin/total model to {FIT_PATH.name}.")
    fit['coefs_margin'] = coefs_margin
    fit['coefs_total'] = coefs_total
    return fit


def predict_from_fit(fit, away_row, home_row):
    """away_row/home_row: nfl_team_matrix.csv rows (real season-to-date
    stats) for the away/home team in this matchup. Returns
    {'margin', 'total', 'win_prob_a'} using the real fitted model --
    margin is away-minus-home, win_prob_a is P(away wins), matching
    predict_matchup()'s own convention in nfl_matchup_sheet.py."""
    x = [1.0]
    for _, _, col in STAT_CATEGORIES:
        val = away_row.get(col) if hasattr(away_row, 'get') else away_row[col]
        x.append(float(val) if pd.notna(val) else 0.0)
    for _, _, col in STAT_CATEGORIES:
        val = home_row.get(col) if hasattr(home_row, 'get') else home_row[col]
        x.append(float(val) if pd.notna(val) else 0.0)
    x = np.array(x)
    coefs_margin = np.array(fit['coefs_margin'])
    coefs_total = np.array(fit['coefs_total'])
    margin = float(np.dot(x, coefs_margin))
    total = float(np.dot(x, coefs_total))
    win_prob_a = float(1 / (1 + np.exp(-margin / fit['win_prob_scale'])))
    return {'margin': margin, 'total': total, 'win_prob_a': win_prob_a}


def load_fit():
    if not FIT_PATH.exists():
        return None
    try:
        with open(FIT_PATH) as f:
            fit = json.load(f)
        return fit
    except Exception as e:
        print(f"  (couldn't load {FIT_PATH.name}: {e} -- falling back)")
        return None


_cached_fit = None
_cached_fit_loaded = False


def get_or_refresh_fit(max_age_days=MAX_FIT_AGE_DAYS, historical_seasons=None):
    """Loads the cached real fit if it's under max_age_days old; otherwise
    tries to refresh it (real network pull of this season's games -- the
    historical seasons stay cached on disk, see _load_historical_cache).
    Cached within the PROCESS after the first call (module-level
    _cached_fit) so calling this once per game/prediction, like
    predict_matchup() does, doesn't re-check/re-fit dozens of times in one
    PDF run. Any failure (no network, an nflreadpy error, etc.) falls back
    to whatever cached fit already exists on disk rather than crashing the
    weekly PDF run -- predict_matchup() only falls back to the OLD
    hand-picked rating approach if there is truly no cached fit at all yet
    (e.g. this has never been run successfully)."""
    global _cached_fit, _cached_fit_loaded
    if _cached_fit_loaded:
        return _cached_fit

    stale = True
    if FIT_PATH.exists():
        age_days = (datetime.datetime.now().timestamp() - FIT_PATH.stat().st_mtime) / 86400
        stale = age_days > max_age_days
    if not stale:
        _cached_fit = load_fit()
        _cached_fit_loaded = True
        return _cached_fit

    try:
        fit_and_save(historical_seasons=historical_seasons)
    except Exception as e:
        print(f"  (real margin-model refresh failed this run: {e} -- using last cached fit, if any)")
    _cached_fit = load_fit()
    _cached_fit_loaded = True
    return _cached_fit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit the real margin/total/win-probability model against real historical + current-season NFL games.")
    parser.add_argument("--refresh-historical", action="store_true",
                         help="Force re-pull of the cached historical seasons too (only needed ~once a year).")
    parser.add_argument("--seasons", type=str, default=None,
                         help="Comma-separated historical season years, e.g. 2021,2022,2023,2024.")
    args = parser.parse_args()
    seasons = sorted({int(y.strip()) for y in args.seasons.split(",") if y.strip()}) if args.seasons else None
    fit_and_save(historical_seasons=seasons, refresh_historical=args.refresh_historical)
