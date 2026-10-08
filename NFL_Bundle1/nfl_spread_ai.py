#!/usr/bin/env python3
# ==============================================================================
# nfl_spread_ai.py  --  "THE SPREAD AI": NFL picks against the Vegas spread, and nothing else.
# ==============================================================================
#   python3 nfl_spread_ai.py                 # weekly run: refresh data, (re)train, honest backtest, this week's picks -> PDF + CSV
#   python3 nfl_spread_ai.py --backtest      # just the walk-forward backtest report
#   python3 nfl_spread_ai.py --week 6        # picks for a specific week
#   python3 nfl_spread_ai.py --rebuild       # throw the cache away and rebuild everything from nflverse
#   python3 nfl_spread_ai.py --refresh-lines # force a fresh pull of live Vegas lines (costs 1 Odds API credit)
#   python3 nfl_spread_ai.py --fast          # start history at 2016 instead of 2012 (quicker first run)
#
# WHAT IT DOES
#   The Vegas closing line is the hardest number in sports to beat, so this does NOT try to out-predict the final score from
#   scratch. It predicts only how far the real result will land from the line:  TARGET = (home margin) - (Vegas spread).
#   Positive = the home team covers.  Every input is a number that was known BEFORE kickoff:
#     * the line itself and the total (the market's own opinion goes in as a feature, so the model only learns where it is wrong)
#     * opponent-adjusted, recency-weighted team ratings built from nflverse play-by-play (EPA, success rate, pass/rush split,
#       explosive plays, sacks, early downs, turnovers) for offense AND defense, carried over from last season and regressed
#     * starting-QB quality (shrunk toward average) and QB changes, rest/short week/bye, division game, dome/wind/cold,
#       how each team has done against the number lately
#   Three model families vote (ridge regression, an ensemble of small neural nets written in plain numpy -- no GPU or torch needed --
#   and gradient boosting if scikit-learn is installed). The blend is then SHRUNK by the slope it actually earned out-of-sample,
#   because NFL residuals are mostly noise and an un-shrunk model over-bets.
#
# THE HONEST PART
#   Everything is judged by WALK-FORWARD backtest: for each 4-week block since 2018 the models are trained only on games that
#   happened earlier and then predict the block. No game is ever predicted by a model that has seen it or anything after it.
#   The program prints the real out-of-sample ATS record by season and by confidence tier, with 95% intervals, against the
#   52.38% you need to beat -110 juice. A tier is only called PROVEN if its out-of-sample record clears 54% on 100+ games.
#   If nothing clears the bar the program says so and marks the picks UNPROVEN instead of pretending. No model can promise
#   to beat the spread; this one tells you how it has actually done.
#
# LEARNS OVER TIME: every run retrains on every completed game (including this season's), grades last week's logged picks
#   (nfl_spread_picks_log.csv) and folds the new results into the shrinkage and the tier thresholds.
#
# DATA: nflverse only (nflreadpy, CC BY 4.0 -- credit nflverse if you ever publish this). Live lines: The Odds API (key in
#   novig_config.json, 1 credit per refresh, cached 3 hours) -> falls back to nflverse's own spread_line -> or your own
#   nfl_spread_manual_lines.csv (columns: home, away, home_spread  where home_spread is the sportsbook number, -3.5 = home favored).
# ==============================================================================
import os
import sys
import json
import math
import time
import argparse
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
CACHE = HERE / "nfl_spread_cache"
GEN = HERE / "generated"
PICKS_CSV = HERE / "nfl_spread_picks.csv"
LOG_CSV = HERE / "nfl_spread_picks_log.csv"
MANUAL_LINES = HERE / "nfl_spread_manual_lines.csv"
CONFIG = HERE / "novig_config.json"

FIRST_SEASON = 2012          # play-by-play history starts here (2012 is only a warm-up year; features are valid from 2013)
FIRST_TEST_SEASON = 2018     # first season that is predicted out-of-sample
BLOCK_WEEKS = [(1, 4), (5, 8), (9, 12), (13, 22)]
BREAKEVEN = 0.5238           # win rate needed to beat -110
PROVEN_MIN_ATS = 0.54
PROVEN_MIN_N = 100
PROVEN_MAX_P = 0.10          # and the record must beat break-even with p < 0.10 (a lucky 54% on 150 games does not qualify)
SIGMA_DEFAULT = 13.2         # std dev of (margin - spread) in the NFL
OOS_VERSION = 5              # bump when the MODEL INPUTS change (forces the walk-forward backtest to be rebuilt)
VERSION = 4                  # bump to invalidate caches when the feature code changes

TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS", "JAC": "JAX"}
TEAM_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    "Washington Football Team": "WAS", "Washington Redskins": "WAS", "Oakland Raiders": "LV",
}
DOMES = {"dome", "closed"}


def fix_team(t):
    t = str(t).strip().upper()
    return TEAM_FIX.get(t, t)


def current_season():
    d = datetime.date.today()
    return d.year if d.month >= 3 else d.year - 1


def say(msg=""):
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------------ data loading
def _nfl():
    import nflreadpy as nfl
    return nfl


def _numpy_dtypes(df):
    """nflreadpy -> pandas can hand back arrow-backed columns; turn them into plain numpy/object columns so every pandas op behaves."""
    for c in df.columns:
        dt = str(df[c].dtype)
        if "pyarrow" in dt or "[pyarrow]" in dt:
            if "string" in dt:
                df[c] = df[c].astype(object)
            else:
                try:
                    df[c] = df[c].astype("float64")
                except Exception:
                    df[c] = df[c].astype(object)
    return df


def load_schedule_df(seasons):
    """nflverse schedule/results/lines for the given seasons, regular season only, home-perspective spread (+ = home favored)."""
    nfl = _nfl()
    s = _numpy_dtypes(nfl.load_schedules(seasons=list(seasons)).to_pandas())
    return s


def load_pbp_df(season):
    """One season of nflverse play-by-play, trimmed to the columns this program uses."""
    nfl = _nfl()
    pl = nfl.load_pbp(seasons=[season])
    want = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "success", "wp", "qb_dropback",
            "sack", "interception", "fumble_lost", "yards_gained", "down", "passer_player_id", "passer_id", "passer_player_name",
            "passer", "rush", "pass", "two_point_attempt", "qb_scramble", "ydstogo", "touchdown", "pass_touchdown", "rush_touchdown",
            "drive", "fixed_drive_result", "third_down_converted", "third_down_failed", "qb_hit", "tackled_for_loss"]
    have = [c for c in want if c in pl.columns]
    return _numpy_dtypes(pl.select(have).to_pandas())


def normalize_schedule(s):
    s = s.copy()
    s = s[s["game_type"].astype(str).eq("REG")] if "game_type" in s.columns else s
    for c in ("home_team", "away_team"):
        s[c] = s[c].map(fix_team)
    for c in ("spread_line", "total_line", "home_score", "away_score", "home_rest", "away_rest", "temp", "wind"):
        if c not in s.columns:
            s[c] = np.nan
        s[c] = pd.to_numeric(s[c], errors="coerce")
    for c in ("roof", "div_game", "gameday", "gametime"):
        if c not in s.columns:
            s[c] = np.nan
    s["result"] = s["home_score"] - s["away_score"]
    s["season"] = s["season"].astype(int)
    s["week"] = s["week"].astype(int)
    s["gameday"] = pd.to_datetime(s["gameday"], errors="coerce")
    s = s.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
    return s


# ------------------------------------------------------------------------------------------------ pbp -> team-game table
OFF_M = ["epa", "succ", "pass_epa", "rush_epa", "expl", "sack", "early", "to", "void", "ppd", "third", "press", "havoc", "int"]


def reduce_pbp(p):
    """Play-by-play for ONE season -> one row per (game, team) with that team's offense numbers and the defense numbers it
    produced/allowed. Garbage time (win probability under 5% or over 95%) is left out of everything except turnovers."""
    p = p.copy()
    if "season_type" in p.columns:
        p = p[p["season_type"].astype(str).eq("REG")]
    p_all = p
    if "two_point_attempt" in p.columns:
        p = p[p["two_point_attempt"].fillna(0) != 1]
    p = p[p["posteam"].notna() & p["defteam"].notna() & p["play_type"].isin(["pass", "run"]) & p["epa"].notna()].copy()
    for c in ("posteam", "defteam"):
        p[c] = p[c].map(fix_team)
    for c in ("success", "qb_dropback", "sack", "interception", "fumble_lost", "yards_gained"):
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0) if c in p.columns else 0.0
    if "wp" not in p.columns:
        p["wp"] = 0.5
    p["wp"] = pd.to_numeric(p["wp"], errors="coerce").fillna(0.5)
    p["db"] = p["qb_dropback"] == 1
    p["to_play"] = ((p["interception"] == 1) | (p["fumble_lost"] == 1)).astype(float)
    p["early_dn"] = p["down"].isin([1, 2]) if "down" in p.columns else False
    ng = p[(p["wp"] >= 0.05) & (p["wp"] <= 0.95)].copy()
    ng["pass_expl"] = (ng["db"] & (ng["yards_gained"] >= 20) & (ng["sack"] == 0)).astype(float)
    ng["run_expl"] = ((~ng["db"]) & (ng["yards_gained"] >= 10)).astype(float)
    ng["expl_play"] = ng[["pass_expl", "run_expl"]].max(axis=1)

    def agg(df, key, pre):
        g = df.groupby(["game_id", key])
        out = pd.DataFrame({
            f"{pre}epa": g["epa"].mean(),
            f"{pre}succ": g["success"].mean(),
            f"{pre}expl": g["expl_play"].mean(),
            f"{pre}n": g["epa"].size(),
        })
        d = df[df["db"]].groupby(["game_id", key])
        out[f"{pre}pass_epa"] = d["epa"].mean()
        out[f"{pre}sack"] = d["sack"].mean()
        r = df[~df["db"]].groupby(["game_id", key])
        out[f"{pre}rush_epa"] = r["epa"].mean()
        e = df[df["early_dn"]].groupby(["game_id", key])
        out[f"{pre}early"] = e["epa"].mean()
        return out

    off = agg(ng, "posteam", "off_")
    dfn = agg(ng, "defteam", "def_")
    # ---- extra measures that mirror the matchup sheet: VOID (bo's own value-per-play), points per drive, 3rd down, pressure, havoc, INT rate
    for c in ("ydstogo", "touchdown", "pass_touchdown", "rush_touchdown", "third_down_converted", "third_down_failed", "qb_hit", "tackled_for_loss"):
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0) if c in p.columns else 0.0
    p["down"] = pd.to_numeric(p["down"], errors="coerce") if "down" in p.columns else np.nan
    need = p["down"].map({1: 0.45, 2: 0.60, 3: 1.0, 4: 1.0})
    val = (p["yards_gained"] / (need * p["ydstogo"].where(p["ydstogo"] > 0))).clip(-1.5, 1.5)
    yg = p["yards_gained"]
    val = val + np.where(yg >= 40, 0.6, np.where(yg >= 20, 0.4, np.where(yg >= 10, 0.2, 0.0)))
    val = np.where(p["to_play"] == 1, np.minimum(val, -1.5), val)
    td = (p["touchdown"] == 1) | (p["pass_touchdown"] == 1) | (p["rush_touchdown"] == 1)
    val = np.where(td, np.maximum(val, 1.2), val)
    p["void_v"] = np.where(p["down"].isin([1, 2, 3, 4]) & (p["ydstogo"] > 0), val, np.nan)
    p["pressure"] = ((p["sack"] == 1) | (p["qb_hit"] == 1)).astype(float)
    p["havoc_play"] = ((p["sack"] == 1) | (p["tackled_for_loss"] == 1) | (p["interception"] == 1) | (p["fumble_lost"] == 1)).astype(float)

    def agg2(key, pre):
        g = p[p["void_v"].notna()].groupby(["game_id", key])
        out = pd.DataFrame({f"{pre}void": g["void_v"].mean()})
        t3 = p.groupby(["game_id", key]).agg(c=("third_down_converted", "sum"), f=("third_down_failed", "sum"))
        out[f"{pre}third"] = t3["c"] / (t3["c"] + t3["f"]).replace(0, np.nan)
        d = p[p["db"]].groupby(["game_id", key])
        out[f"{pre}press"] = d["pressure"].mean()
        out[f"{pre}int"] = d["interception"].mean()
        out[f"{pre}havoc"] = p.groupby(["game_id", key])["havoc_play"].mean()
        return out

    o2, d2 = agg2("posteam", "off_"), agg2("defteam", "def_")
    pts_map = {"touchdown": 7.0, "field goal": 3.0, "safety": -2.0}
    if "drive" in p_all.columns and "fixed_drive_result" in p_all.columns:
        dv = p_all[p_all["posteam"].notna() & p_all["drive"].notna()]
        dv = dv.groupby(["game_id", "drive"]).agg(posteam=("posteam", "first"), defteam=("defteam", "first"), res=("fixed_drive_result", "first")).reset_index()
        dv["dp"] = dv["res"].astype(str).str.lower().map(pts_map).fillna(0.0)
        dv["posteam"] = dv["posteam"].map(fix_team)
        dv["defteam"] = dv["defteam"].map(fix_team)
        o2["off_ppd"] = dv.groupby(["game_id", "posteam"])["dp"].mean()
        d2["def_ppd"] = dv.groupby(["game_id", "defteam"])["dp"].mean()
    else:
        o2["off_ppd"], d2["def_ppd"] = np.nan, np.nan
    off = off.join(o2)
    dfn = dfn.join(d2)
    to_off = p.groupby(["game_id", "posteam"])["to_play"].mean().rename("off_to")
    to_def = p.groupby(["game_id", "defteam"])["to_play"].mean().rename("def_to")
    off = off.join(to_off)
    dfn = dfn.join(to_def)
    off.index.names = ["game_id", "team"]
    dfn.index.names = ["game_id", "team"]
    tg = off.join(dfn, how="outer").reset_index()

    # starting QB = the passer with the most dropbacks in that game (all plays, garbage time included)
    pid = "passer_player_id" if "passer_player_id" in p.columns else ("passer_id" if "passer_id" in p.columns else None)
    if pid is not None:
        d = p[p["db"] & p[pid].notna()]
        q = d.groupby(["game_id", "posteam", pid]).agg(n=("epa", "size"), esum=("epa", "sum")).reset_index()
        q = q.sort_values(["game_id", "posteam", "n"], ascending=[True, True, False]).drop_duplicates(["game_id", "posteam"])
        q = q.rename(columns={"posteam": "team", pid: "qb_id", "n": "qb_n", "esum": "qb_esum"})
        tg = tg.merge(q[["game_id", "team", "qb_id", "qb_n", "qb_esum"]], on=["game_id", "team"], how="left")
    else:
        tg["qb_id"], tg["qb_n"], tg["qb_esum"] = np.nan, np.nan, np.nan
    meta = p.groupby("game_id").agg(season=("season", "first"), week=("week", "first")).reset_index()
    tg = tg.merge(meta, on="game_id", how="left")
    return tg


def get_team_games(first_season, last_season, rebuild=False, cur=None):
    """Team-game table for every season in range. Past seasons are reduced once and cached; the current season is re-pulled
    every run so new results flow in."""
    CACHE.mkdir(exist_ok=True)
    cur = cur if cur is not None else current_season()
    parts = []
    for yr in range(first_season, last_season + 1):
        f = CACHE / f"teamgames_{yr}_v{VERSION}.csv.gz"
        if f.exists() and not rebuild and yr < cur:
            parts.append(pd.read_csv(f))
            continue
        say(f"  pulling {yr} play-by-play from nflverse ...")
        try:
            t = reduce_pbp(load_pbp_df(yr))
        except Exception as e:
            if yr == cur:
                say(f"  !! could not pull {yr} play-by-play ({e}); continuing without it")
                if f.exists():
                    parts.append(pd.read_csv(f))
                continue
            raise
        t.to_csv(f, index=False, compression="gzip")
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ------------------------------------------------------------------------------------------------ pre-game features
RATE_M = OFF_M + ["pts"]          # every rating tracked, for offense ("off_x") and defense ("def_x" = what it allowed / made)
HALF_LIFE = {"to": 14.0, "int": 14.0, "pts": 8.0, "ppd": 8.0, "third": 10.0, "press": 9.0, "havoc": 9.0}
DEFAULT_HALF = 7.0
PRIOR_W = 3.0                      # a new season starts worth ~3 games of last season's (regressed) rating
CARRY = 0.5                        # share of last season's final rating carried into the next season
QB_K = 150.0                       # dropbacks of "average QB" mixed into every QB rating (shrinkage)
QB_DECAY = 0.985
ATS_HALF = 16.0
ATS_VEN_HALF = 24.0
ATS_CARRY = 0.3
ATS_SHRINK = 6.0                   # an ATS record is pulled toward 50% as if it had 6 extra coin-flip games


def _shr(r):
    """Shrunk cover rate minus 0.5 from a [w, l, push] record (so +0.1 = covering 60% after shrinkage)."""
    w, l = r[0], r[1]
    return (w + 0.5 * ATS_SHRINK) / (w + l + ATS_SHRINK) - 0.5


def _rs(r):
    return f"{r[0]}-{r[1]}" + (f"-{r[2]}" if r[2] else "")


def _decay(key):
    return 0.5 ** (1.0 / HALF_LIFE.get(key.split("_", 1)[1], DEFAULT_HALF))


class _Team:
    def __init__(self):
        self.season = None
        self.gp = 0
        self.r = {}                 # "off_epa" -> [value, weight]
        self.ats = [0.0, 2.0]
        self.rec = {"all": [0, 0, 0], "home": [0, 0, 0], "away": [0, 0, 0]}      # this season's record against the spread: W-L-push
        self.vm = {"home": [0.0, 2.0], "away": [0.0, 2.0]}                      # recency-weighted margin vs the number, by venue
        self.last_qb = None


def build_games(sched, tg, qb_override=None):
    """Walk through every game in time order. For each game, record both teams' PRE-GAME ratings, then (if it has been
    played) fold its result in. Nothing a game contributes can reach a feature of that game or an earlier one.
    qb_override = {team: qb_id or None} sets the expected starter for UNPLAYED games (None = replacement-level QB)."""
    qb_override = qb_override or {}
    tgi = {}
    if len(tg):
        for rec in tg.to_dict("records"):
            tgi[(rec["game_id"], rec["team"])] = rec
    teams = {}
    lsum = {m: 0.0 for m in RATE_M}
    lcnt = {m: 0 for m in RATE_M}
    qbs = {}                         # qb_id -> [decayed epa sum, decayed dropbacks]
    qb_tot = [0.0, 0.0]
    rows = []
    keys = [f"{s}_{m}" for s in ("off", "def") for m in RATE_M]

    def lmean(m):
        return lsum[m] / lcnt[m] if lcnt[m] else 0.0

    def get_team(t, season):
        st = teams.get(t)
        if st is None:
            st = _Team()
            teams[t] = st
        if st.season != season:
            for k in keys:
                m = k.split("_", 1)[1]
                old = st.r.get(k)
                base = lmean(m)
                st.r[k] = [CARRY * old[0] + (1 - CARRY) * base if old else base, PRIOR_W]
            st.ats = [0.0, 2.0]
            st.rec = {"all": [0, 0, 0], "home": [0, 0, 0], "away": [0, 0, 0]}
            st.vm = {k: [ATS_CARRY * v[0], 2.0] for k, v in st.vm.items()}
            st.season, st.gp = season, 0
        return st

    def qb_rating(qid):
        lm = qb_tot[0] / qb_tot[1] if qb_tot[1] > 0 else 0.0
        if qid is None or (isinstance(qid, float) and math.isnan(qid)) or qid not in qbs:
            return -0.10, 0.0        # unknown / replacement-level quarterback
        e, n = qbs[qid]
        return (e + QB_K * lm) / (n + QB_K) - lm, n

    for g in sched.itertuples(index=False):
        h, a = g.home_team, g.away_team
        hs, as_ = get_team(h, g.season), get_team(a, g.season)
        played = not (pd.isna(g.result))
        rec_h = tgi.get((g.game_id, h))
        rec_a = tgi.get((g.game_id, a))
        have_tg = rec_h is not None and rec_a is not None
        # starting QBs
        if played and have_tg:
            qh, qa = rec_h.get("qb_id"), rec_a.get("qb_id")
        else:
            qh = qb_override.get(h, hs.last_qb)
            qa = qb_override.get(a, as_.last_qb)
        rh, nh = qb_rating(qh)
        ra, na = qb_rating(qa)
        row = {"game_id": g.game_id, "season": g.season, "week": g.week, "home": h, "away": a, "gameday": g.gameday,
               "gametime": g.gametime, "spread": g.spread_line, "total": g.total_line, "result": g.result if played else np.nan,
               "home_qb": qh, "away_qb": qa, "qb_h": rh, "qb_a": ra, "qb_n_h": math.log1p(nh), "qb_n_a": math.log1p(na),
               "qb_chg_h": float(hs.last_qb is not None and qh is not None and qh == qh and qh != hs.last_qb),
               "qb_chg_a": float(as_.last_qb is not None and qa is not None and qa == qa and qa != as_.last_qb),
               "gp_h": hs.gp, "gp_a": as_.gp, "ats_h": hs.ats[0], "ats_a": as_.ats[0],
               "ats_pct_all_h": _shr(hs.rec["all"]), "ats_pct_all_a": _shr(as_.rec["all"]),
               "ats_pct_ven_h": _shr(hs.rec["home"]), "ats_pct_ven_a": _shr(as_.rec["away"]),
               "ats_ven_h": hs.vm["home"][0], "ats_ven_a": as_.vm["away"][0],
               "rec_h_all": _rs(hs.rec["all"]), "rec_h_ven": _rs(hs.rec["home"]),
               "rec_a_all": _rs(as_.rec["all"]), "rec_a_ven": _rs(as_.rec["away"])}
        for k in keys:
            row["h_" + k] = hs.r[k][0]
            row["a_" + k] = as_.r[k][0]
        # context
        hr = g.home_rest if not pd.isna(g.home_rest) else 7.0
        ar = g.away_rest if not pd.isna(g.away_rest) else 7.0
        dome = str(g.roof).lower() in DOMES
        wind = 0.0 if dome else (g.wind if not pd.isna(g.wind) else 6.0)
        temp = 70.0 if dome else (g.temp if not pd.isna(g.temp) else 60.0)
        row.update({"rest_diff": float(np.clip(hr - ar, -7, 7)), "short_h": float(hr <= 5), "short_a": float(ar <= 5),
                    "bye_h": float(hr >= 10), "bye_a": float(ar >= 10), "div": float(g.div_game == 1), "dome": float(dome),
                    "wind": float(wind), "cold": float((not dome) and temp < 35), "week_n": float(g.week)})
        rows.append(row)

        # ---- fold the result in (only for played games with play-by-play)
        if played and have_tg:
            obs = {}
            for t, rec, opp_st, opp_pts in ((h, rec_h, as_, g.away_score), (a, rec_a, hs, g.home_score)):
                pts_for = g.home_score if t == h else g.away_score
                pts_ag = g.away_score if t == h else g.home_score
                o = {f"off_{m}": rec.get(f"off_{m}") for m in OFF_M}
                o.update({f"def_{m}": rec.get(f"def_{m}") for m in OFF_M})
                o["off_pts"], o["def_pts"] = float(pts_for), float(pts_ag)
                obs[t] = (o, opp_st)
            updates = []                 # compute BOTH teams' updates from the PRE-game states first, then apply (no cross-contamination)
            for t, st in ((h, hs), (a, as_)):
                o, opp = obs[t]
                for k in keys:
                    x = o.get(k)
                    if x is None or (isinstance(x, float) and math.isnan(x)):
                        continue
                    side, m = k.split("_", 1)
                    other = ("def_" if side == "off" else "off_") + m
                    # strip out how good the opponent already looked: an off number is judged against the opponent's defense, etc.
                    x_adj = x - (opp.r[other][0] - lmean(m))
                    s_, w = st.r[k]
                    d = _decay(k)
                    w2 = w * d + 1.0
                    updates.append((st, k, [(s_ * w * d + x_adj) / w2, w2]))
            for st, k, val in updates:
                st.r[k] = val
            for m in RATE_M:
                for t in (h, a):
                    x = obs[t][0].get(f"off_{m}")
                    if x is not None and not (isinstance(x, float) and math.isnan(x)):
                        lsum[m] += float(x)
                        lcnt[m] += 1
            # QBs
            for rec in (rec_h, rec_a):
                qid, n, es = rec.get("qb_id"), rec.get("qb_n"), rec.get("qb_esum")
                if qid is not None and qid == qid and n and n == n:
                    cur = qbs.get(qid, [0.0, 0.0])
                    qbs[qid] = [cur[0] * QB_DECAY + float(es), cur[1] * QB_DECAY + float(n)]
                    qb_tot[0] += float(es)
                    qb_tot[1] += float(n)
            # how each team did against the number
            if not pd.isna(g.spread_line):
                cover = float(g.result - g.spread_line)
                for st, sgn, ven in ((hs, 1.0, "home"), (as_, -1.0, "away")):
                    d = 0.5 ** (1.0 / ATS_HALF)
                    s, w = st.ats
                    w2 = w * d + 1.0
                    st.ats = [(s * w * d + sgn * cover) / w2, w2]
                    dv = 0.5 ** (1.0 / ATS_VEN_HALF)
                    sv, wv = st.vm[ven]
                    wv2 = wv * dv + 1.0
                    st.vm[ven] = [(sv * wv * dv + sgn * cover) / wv2, wv2]
                    k = 2 if cover == 0 else (0 if cover * sgn > 0 else 1)
                    st.rec["all"][k] += 1
                    st.rec[ven][k] += 1
            hs.last_qb = rec_h.get("qb_id") if rec_h.get("qb_id") == rec_h.get("qb_id") else hs.last_qb
            as_.last_qb = rec_a.get("qb_id") if rec_a.get("qb_id") == rec_a.get("qb_id") else as_.last_qb
            hs.gp += 1
            as_.gp += 1
    df = pd.DataFrame(rows)
    df = add_derived(df)
    return df, {"qb_rating": qb_rating, "teams": teams}


def add_derived(df):
    d = df
    d["imp_h"] = (d["total"] + d["spread"]) / 2.0
    d["imp_a"] = (d["total"] - d["spread"]) / 2.0
    for m in ("epa", "pass_epa", "rush_epa", "succ", "expl", "early", "pts", "void", "ppd", "third"):
        d[f"hedge_{m}"] = d[f"h_off_{m}"] - d[f"a_def_{m}"]
        d[f"aedge_{m}"] = d[f"a_off_{m}"] - d[f"h_def_{m}"]
        d[f"net_{m}"] = d[f"hedge_{m}"] - d[f"aedge_{m}"]
    d["net_sack"] = (d["h_def_sack"] + d["a_off_sack"]) - (d["a_def_sack"] + d["h_off_sack"])
    d["net_to"] = (d["h_def_to"] + d["a_off_to"]) - (d["a_def_to"] + d["h_off_to"])
    for m in ("press", "havoc", "int"):
        d[f"net_{m}"] = (d[f"h_def_{m}"] + d[f"a_off_{m}"]) - (d[f"a_def_{m}"] + d[f"h_off_{m}"])
    d["qb_diff"] = d["qb_h"] - d["qb_a"]
    d["ats_diff"] = d["ats_h"] - d["ats_a"]
    d["gp_avg"] = (d["gp_h"] + d["gp_a"]) / 2.0
    d["target"] = d["result"] - d["spread"]
    return d


FEATS_COMPACT = ["spread", "total", "net_epa", "net_pass_epa", "net_rush_epa", "net_succ", "net_expl", "net_early", "net_sack",
                 "net_to", "net_pts", "hedge_pass_epa", "aedge_pass_epa", "hedge_rush_epa", "aedge_rush_epa", "qb_diff", "qb_h",
                 "qb_a", "qb_chg_h", "qb_chg_a", "qb_n_h", "qb_n_a", "rest_diff", "short_h", "short_a", "bye_h", "bye_a", "div",
                 "dome", "wind", "cold", "ats_diff", "gp_avg"]
# 2026-10-07: the walk-forward ablation (--ablation) showed VOID / points-per-drive / 3rd down / pressure / havoc / INT rate and the
# home-road ATS records did NOT improve the out-of-sample result (they added noise), so they are NOT model inputs. They are still
# computed (and the ATS records still print on the PDF); put names from NEW_STAT / NEW_ATS back into FEATS_COMPACT to re-test them.
_CORE_M = ["epa", "succ", "pass_epa", "rush_epa", "expl", "sack", "early", "to", "pts"]
FEATS_FULL = FEATS_COMPACT + [f"{p}_{s}_{m}" for p in ("h", "a") for s in ("off", "def") for m in _CORE_M] + ["week_n"]
LABELS = {
    "net_epa": "overall efficiency (EPA) edge", "net_pass_epa": "pass-game efficiency edge", "net_rush_epa": "run-game efficiency edge",
    "net_succ": "success-rate edge", "net_expl": "explosive-play edge", "net_early": "early-down efficiency edge",
    "net_sack": "sack/pass-protection edge", "net_to": "turnover-tendency edge", "net_pts": "scoring-margin edge",
    "hedge_pass_epa": "home pass offense vs away pass D", "aedge_pass_epa": "away pass offense vs home pass D",
    "hedge_rush_epa": "home run game vs away run D", "aedge_rush_epa": "away run game vs home run D",
    "qb_diff": "quarterback gap", "qb_h": "home QB quality", "qb_a": "away QB quality", "qb_chg_h": "home QB change",
    "qb_chg_a": "away QB change", "rest_diff": "rest advantage", "short_h": "home short week", "short_a": "away short week",
    "bye_h": "home off a bye", "bye_a": "away off a bye", "div": "division game", "dome": "dome game", "wind": "wind", "cold": "cold",
    "ats_diff": "recent record vs the number", "spread": "the line itself (favorite/dog bias)", "total": "the game total",
    "net_void": "VOID edge (value per play)", "net_ppd": "points-per-drive edge", "net_third": "third-down edge",
    "net_press": "pressure-rate edge", "net_havoc": "havoc-rate edge", "net_int": "interception-rate edge",
    "ats_pct_all_h": "home team's record vs the spread", "ats_pct_all_a": "away team's record vs the spread",
    "ats_pct_ven_h": "home team's HOME record vs the spread", "ats_pct_ven_a": "away team's ROAD record vs the spread",
    "ats_ven_h": "home team's margin vs the number at home", "ats_ven_a": "away team's margin vs the number on the road",
    "qb_n_h": "home QB experience", "qb_n_a": "away QB experience", "gp_avg": "sample size this season",
}


# ------------------------------------------------------------------------------------------------ models
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


class Scaler:
    def fit(self, X):
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.mu = np.where(np.isnan(self.mu), 0.0, self.mu)
        self.sd = np.where((self.sd < 1e-9) | np.isnan(self.sd), 1.0, self.sd)
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        Z = np.clip(Z, -5, 5)
        return np.nan_to_num(Z, nan=0.0)


def ridge_fit(X, y, alpha):
    ym = y.mean()
    A = X.T @ X + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ (y - ym))
    return w, ym


def ridge_pick_alpha(X, y, alphas=(30, 100, 300, 1000, 3000, 10000)):
    n = len(y)
    cut = int(n * 0.75)
    best, ba = 1e18, alphas[-1]
    for a in alphas:
        w, ym = ridge_fit(X[:cut], y[:cut], a)
        err = float(np.mean((X[cut:] @ w + ym - y[cut:]) ** 2))
        if err < best:
            best, ba = err, a
    return ba


class MLP:
    """Small feed-forward net in plain numpy: inputs -> 24 tanh -> 12 tanh -> 1. Adam, L2 weight decay, input noise, early stopping
    on the most recent 15% of the training games. One of these is a weak learner; the ensemble of seeds is the model."""

    def __init__(self, seed=0, h1=24, h2=12, l2=4e-3, lr=2e-3, epochs=600, patience=60, noise=0.20):
        self.seed, self.h1, self.h2, self.l2, self.lr, self.epochs, self.patience, self.noise = seed, h1, h2, l2, lr, epochs, patience, noise

    def _init(self, d, rng):
        self.p = {"W1": rng.normal(0, 1 / math.sqrt(d), (d, self.h1)), "b1": np.zeros(self.h1),
                  "W2": rng.normal(0, 1 / math.sqrt(self.h1), (self.h1, self.h2)), "b2": np.zeros(self.h2),
                  "W3": rng.normal(0, 1 / math.sqrt(self.h2), (self.h2, 1)) * 0.1, "b3": np.zeros(1)}

    def _fwd(self, X, p=None):
        p = p or self.p
        a1 = np.tanh(X @ p["W1"] + p["b1"])
        a2 = np.tanh(a1 @ p["W2"] + p["b2"])
        return a1, a2, (a2 @ p["W3"] + p["b3"]).ravel()

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.ys = float(np.std(y)) or 1.0
        t = (y - y.mean()) / self.ys
        self.ym = float(y.mean())
        cut = int(n * 0.85)
        Xtr, ttr, Xva, tva = X[:cut], t[:cut], X[cut:], t[cut:]
        self._init(d, rng)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(v_) for k, v_ in self.p.items()}
        best, best_p, bad = 1e18, None, 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for ep in range(1, self.epochs + 1):
            Xn = Xtr + self.noise * rng.normal(size=Xtr.shape)
            a1, a2, out = self._fwd(Xn)
            g = (2.0 / len(ttr)) * (out - ttr)[:, None]
            grads = {"W3": a2.T @ g + self.l2 * self.p["W3"], "b3": g.sum(0)}
            da2 = (g @ self.p["W3"].T) * (1 - a2 ** 2)
            grads["W2"] = a1.T @ da2 + self.l2 * self.p["W2"]
            grads["b2"] = da2.sum(0)
            da1 = (da2 @ self.p["W2"].T) * (1 - a1 ** 2)
            grads["W1"] = Xn.T @ da1 + self.l2 * self.p["W1"]
            grads["b1"] = da1.sum(0)
            for k in self.p:
                m[k] = b1 * m[k] + (1 - b1) * grads[k]
                v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                self.p[k] -= self.lr * (m[k] / (1 - b1 ** ep)) / (np.sqrt(v[k] / (1 - b2 ** ep)) + eps)
            if ep % 5 == 0:
                _, _, pv = self._fwd(Xva)
                e = float(np.mean((pv - tva) ** 2))
                if e < best - 1e-5:
                    best, bad = e, 0
                    best_p = {k: a.copy() for k, a in self.p.items()}
                else:
                    bad += 5
                    if bad >= self.patience:
                        break
        if best_p is not None:
            self.p = best_p
        return self

    def predict(self, X):
        return self._fwd(X)[2] * self.ys + self.ym


class Bundle:
    """The three model families, fit on one training set."""

    def __init__(self, n_mlp=7):
        self.n_mlp = n_mlp

    def fit(self, tr):
        y = np.clip(tr["target"].to_numpy(float), -28, 28)
        Xc = tr[FEATS_COMPACT].to_numpy(float)
        self.sc = Scaler().fit(Xc)
        Z = self.sc.transform(Xc)
        self.alpha = ridge_pick_alpha(Z, y)
        self.w, self.ym = ridge_fit(Z, y, self.alpha)
        self.mlps = [MLP(seed=s).fit(Z, y) for s in range(self.n_mlp)]
        self.gbm = None
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=180, min_samples_leaf=40,
                                                     l2_regularization=5.0, random_state=1)
            self.gbm.fit(tr[FEATS_FULL].to_numpy(float), y)
        except Exception:
            self.gbm = None
        return self

    def predict(self, df):
        Z = self.sc.transform(df[FEATS_COMPACT].to_numpy(float))
        out = {"ridge": Z @ self.w + self.ym}
        if self.mlps:
            out["mlp"] = np.mean([m.predict(Z) for m in self.mlps], axis=0)
        if self.gbm is not None:
            out["gbm"] = self.gbm.predict(df[FEATS_FULL].to_numpy(float))
        out["blend"] = np.mean([out[k] for k in out], axis=0)
        return out

    def reasons(self, row, k=3):
        z = self.sc.transform(row[FEATS_COMPACT].to_numpy(float).reshape(1, -1))[0]
        contrib = z * self.w
        order = np.argsort(-np.abs(contrib))
        res = []
        for i in order:
            f = FEATS_COMPACT[i]
            if f in ("spread", "total", "gp_avg") or abs(contrib[i]) < 0.05:
                continue
            res.append((LABELS.get(f, f), float(contrib[i])))
            if len(res) >= k:
                break
        return res


# ------------------------------------------------------------------------------------------------ walk-forward backtest
def usable(games):
    """Played games with a line and enough history behind their ratings."""
    g = games[games["result"].notna() & games["spread"].notna() & games["total"].notna() & (games["season"] > FIRST_SEASON)]
    return g.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)


def walk_forward(games, cur, verbose=True, n_mlp=7):
    """Predict every game since FIRST_TEST_SEASON with models trained ONLY on earlier games, in 4-week blocks. Cached on disk;
    only blocks that gained new games are recomputed."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"oos_v{OOS_VERSION}_{FIRST_SEASON}.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    if len(old) and "pred_gbm" not in old.columns:
        try:
            import sklearn  # noqa: F401
            say("  scikit-learn is installed now but the cached backtest was built without boosting -> rebuilding the backtest with all 3 models")
            old = pd.DataFrame()
        except Exception:
            pass
    done_ids = set(old["game_id"]) if len(old) else set()
    ug = usable(games)
    new_parts = []
    for season in range(FIRST_TEST_SEASON, cur + 1):
        for lo, hi in BLOCK_WEEKS:
            te = ug[(ug["season"] == season) & (ug["week"] >= lo) & (ug["week"] <= hi)]
            if te.empty or set(te["game_id"]) <= done_ids:
                continue
            tr = ug[(ug["season"] < season) | ((ug["season"] == season) & (ug["week"] < lo))]
            if len(tr) < 600:
                continue
            t0 = time.time()
            b = Bundle(n_mlp).fit(tr)
            pr = b.predict(te)
            part = te[["game_id", "season", "week", "home", "away", "spread", "total", "result", "target"]].copy()
            for k, v in pr.items():
                part[f"pred_{k}"] = v
            new_parts.append(part)
            if verbose:
                say(f"  walk-forward {season} wk{lo}-{min(hi, 18)}: trained on {len(tr)} games, predicted {len(te)} ({time.time() - t0:.0f}s)")
    if new_parts:
        fresh = pd.concat(new_parts)
        old = old[~old["game_id"].isin(fresh["game_id"])] if len(old) else old
        oos = pd.concat([old, fresh], ignore_index=True)
        oos.to_csv(f, index=False)
    else:
        oos = old
    return oos.sort_values(["season", "week", "game_id"]).reset_index(drop=True) if len(oos) else oos


def ats_outcome(pred_side_home, target):
    """+1 win, -1 loss, 0 push for a bet on home (pred_side_home True) or away, given target = margin - spread."""
    t = np.asarray(target, float)
    s = np.where(np.asarray(pred_side_home), 1.0, -1.0)
    return np.sign(t) * s


def tier_cutoffs(oos, upto_season=None):
    """|prediction| cutoffs for tiers C/B/A = 70th/85th/95th percentile of out-of-sample |prediction| (earlier seasons only if given)."""
    d = oos if upto_season is None else oos[oos["season"] < upto_season]
    if len(d) < 200:
        d = oos
    a = np.abs(d["pred_blend"].to_numpy(float))
    return {"C": float(np.quantile(a, 0.70)), "B": float(np.quantile(a, 0.85)), "A": float(np.quantile(a, 0.95))}


def tier_of(absv, cuts):
    return "A" if absv >= cuts["A"] else ("B" if absv >= cuts["B"] else ("C" if absv >= cuts["C"] else "PASS"))


def evaluate(oos, quiet=False):
    """Prints the backtest and returns the numbers the picks page needs."""
    out = {"ok": False}
    if oos is None or len(oos) < 100:
        if not quiet:
            say("  not enough out-of-sample games yet for a backtest report")
        return out
    d = oos.copy()
    d["side_home"] = d["pred_blend"] > 0
    d["res"] = ats_outcome(d["side_home"], d["target"])
    # expanding shrink slope: each season uses only earlier seasons' predictions
    c_by_season, tiers = {}, []
    for s in sorted(d["season"].unique()):
        prev = d[d["season"] < s]
        if len(prev) < 300:
            prev = d
        v = float(np.var(prev["pred_blend"]))
        c_by_season[s] = float(np.clip(np.cov(prev["pred_blend"], prev["target"])[0, 1] / v, 0, 1)) if v > 0 else 0.0
        cuts = tier_cutoffs(d, s)
        sub = d[d["season"] == s]
        tiers += [tier_of(abs(x), cuts) for x in sub["pred_blend"]]
    d["tier"] = tiers
    allv = float(np.var(d["pred_blend"]))
    c_all = float(np.clip(np.cov(d["pred_blend"], d["target"])[0, 1] / allv, 0, 1)) if allv > 0 else 0.0
    sigma = float(np.std(d["target"] - c_all * d["pred_blend"]))
    corr = float(np.corrcoef(d["pred_blend"], d["target"])[0, 1])

    def rec(x):
        w, l, p = int((x == 1).sum()), int((x == -1).sum()), int((x == 0).sum())
        n = w + l
        lo, hi = wilson(w, n)
        pbe = (1.0 - _phi((w / n - BREAKEVEN) / math.sqrt(BREAKEVEN * (1 - BREAKEVEN) / n))) if n else 1.0
        return {"w": w, "l": l, "p": p, "n": n, "pct": (w / n if n else float("nan")), "lo": lo, "hi": hi, "pbe": pbe,
                "roi": ((w * 100 / 110.0 - l) / n if n else float("nan"))}

    overall = rec(d["res"])
    per_model = {}
    for k in ("ridge", "mlp", "gbm", "blend"):
        if f"pred_{k}" in d.columns:
            per_model[k] = rec(ats_outcome(d[f"pred_{k}"] > 0, d["target"]))
    by_season = {int(s): rec(d[d["season"] == s]["res"]) for s in sorted(d["season"].unique())}
    by_tier = {t: rec(d[d["tier"] == t]["res"]) for t in ("A", "B", "C")}
    cum = {t: rec(d[d["tier"].isin({"A": ["A"], "B": ["A", "B"], "C": ["A", "B", "C"]}[t])]["res"]) for t in ("A", "B", "C")}
    pts_tbl = {}
    for thr in (1, 2, 3, 4, 5, 6):
        pts_tbl[thr] = rec(d[np.abs(d["pred_blend"]) >= thr]["res"])
    fav = d["spread"] != 0
    sides = {"picked home": rec(d[d["side_home"]]["res"]), "picked away": rec(d[~d["side_home"]]["res"]),
             "picked the favorite": rec(d[(d["side_home"] == (d["spread"] > 0)) & fav]["res"]),
             "picked the underdog": rec(d[(d["side_home"] != (d["spread"] > 0)) & fav]["res"])}
    mae_line = float(np.mean(np.abs(d["target"])))
    mae_model = float(np.mean(np.abs(d["target"] - c_all * d["pred_blend"])))
    proven = {t: bool(by_tier[t]["n"] >= PROVEN_MIN_N and by_tier[t]["pct"] >= PROVEN_MIN_ATS and by_tier[t]["pbe"] < PROVEN_MAX_P) for t in by_tier}
    out.update({"ok": True, "n": len(d), "overall": overall, "per_model": per_model, "by_season": by_season, "by_tier": by_tier,
                "cum": cum, "pts": pts_tbl, "sides": sides, "c": c_all, "sigma": sigma, "corr": corr, "proven": proven,
                "mae_line": mae_line, "mae_model": mae_model, "cuts": tier_cutoffs(d), "seasons": (int(d["season"].min()), int(d["season"].max()))})
    if quiet:
        return out
    f = lambda r: f"{r['pct'] * 100:5.1f}%  ({r['w']}-{r['l']}{'-' + str(r['p']) if r['p'] else ''}, n={r['n']}, 95% CI {r['lo'] * 100:.1f}-{r['hi'] * 100:.1f}%, ROI {r['roi'] * 100:+.1f}%, p={r['pbe']:.2f} vs break-even)"
    say("")
    say("=" * 100)
    say(f"WALK-FORWARD BACKTEST  --  {out['seasons'][0]}-{out['seasons'][1]}, {len(d)} games, every one predicted by models that had never seen it")
    say("=" * 100)
    say(f"Break-even at -110 is {BREAKEVEN * 100:.2f}%.  Correlation of model vs real (margin - line): {corr:+.3f}   shrink slope earned: {c_all:.2f}")
    say(f"Average miss: the line alone {mae_line:.2f} pts -> line + shrunk model {mae_model:.2f} pts   (lower is better)")
    say("")
    say("Every game, blended model's side:     " + f(overall))
    for k, r in per_model.items():
        say(f"   {k:<6} alone:                       " + f(r))
    say("")
    say("By season:")
    for s, r in by_season.items():
        say(f"   {s}:  " + f(r))
    say("")
    say("By confidence tier (tier cutoffs use earlier seasons only):")
    for t in ("A", "B", "C"):
        say(f"   Tier {t}:  " + f(by_tier[t]) + ("   <- VALIDATED" if proven[t] else ""))
    say("By how many points the model disagreed with Vegas (raw):")
    for thr, r in pts_tbl.items():
        say(f"   >= {thr} pts:  " + f(r))
    say("Sides:")
    for k, r in sides.items():
        say(f"   {k:<22} " + f(r))
    say("")
    best = max(by_tier.values(), key=lambda r: (r["pct"] if r["n"] >= 40 else 0))
    if not any(proven.values()):
        say("VERDICT: no confidence tier has been VALIDATED out-of-sample (needs 54%+ on 100+ games AND p<0.10 against the 52.4% break-even). "
            "Treat the picks as UNPROVEN leans, not a betting edge. (The NFL spread is the most efficient market in sports; an honest model "
            "usually lands near 50-53%.)")
    else:
        ok = [t for t, v in proven.items() if v]
        say(f"VERDICT: tier(s) {', '.join(ok)} VALIDATED out-of-sample (54%+ on 100+ games, p<0.10 vs break-even). That is evidence, not a "
            f"guarantee: the tiers were defined from this same history, so size bets small and let the live log (nfl_spread_picks_log.csv) confirm it.")
    return out


# ------------------------------------------------------------------------------------------------ which inputs help?
NEW_STAT = ["net_void", "net_ppd", "net_third", "net_press", "net_havoc", "net_int"]
NEW_ATS = ["ats_pct_all_h", "ats_pct_all_a", "ats_pct_ven_h", "ats_pct_ven_a", "ats_ven_h", "ats_ven_a"]


def ablation(games, cur):
    """Same walk-forward test (ridge + boosting, quick) run with different input sets, so you can see which groups of numbers
    actually helped against the spread and which did nothing."""
    global FEATS_COMPACT, FEATS_FULL
    base = [f for f in FEATS_COMPACT if f not in NEW_STAT and f not in NEW_ATS]
    sets = {"original inputs": base,
            "+ sheet stats (VOID, points/drive, 3rd down, pressure, havoc, INT)": base + NEW_STAT,
            "+ ATS records (overall, home/road)": base + NEW_ATS,
            "everything": base + NEW_STAT + NEW_ATS}
    save = (FEATS_COMPACT, FEATS_FULL)
    ug = usable(games)
    say("")
    say("=" * 100)
    say("WHICH INPUTS HELP?  (same walk-forward test, ridge + boosting, every game since %d)" % FIRST_TEST_SEASON)
    say("=" * 100)
    try:
        for name, feats in sets.items():
            FEATS_COMPACT = FEATS_FULL = feats
            parts = []
            for season in range(FIRST_TEST_SEASON, cur + 1):
                for lo, hi in BLOCK_WEEKS:
                    te = ug[(ug["season"] == season) & (ug["week"] >= lo) & (ug["week"] <= hi)]
                    tr = ug[(ug["season"] < season) | ((ug["season"] == season) & (ug["week"] < lo))]
                    if te.empty or len(tr) < 600:
                        continue
                    pr = Bundle(0).fit(tr).predict(te)
                    part = te[["season", "target", "spread"]].copy()
                    part["pred"] = pr["blend"]
                    parts.append(part)
            d = pd.concat(parts)
            res = ats_outcome(d["pred"] > 0, d["target"])
            w, l = int((res == 1).sum()), int((res == -1).sum())
            top = d[np.abs(d["pred"]) >= np.quantile(np.abs(d["pred"]), 0.85)]
            rt = ats_outcome(top["pred"] > 0, top["target"])
            tw, tl = int((rt == 1).sum()), int((rt == -1).sum())
            corr = float(np.corrcoef(d["pred"], d["target"])[0, 1])
            say(f"  {name:<68} all picks {100 * w / max(1, w + l):5.1f}%   top 15% {100 * tw / max(1, tw + tl):5.1f}% (n={tw + tl})   corr {corr:+.3f}")
    finally:
        FEATS_COMPACT, FEATS_FULL = save
    say("  (corr = how well the model's disagreement with Vegas predicts the real miss; higher is better, 0 = nothing)")


# ------------------------------------------------------------------------------------------------ live lines
def _api_key():
    try:
        with open(CONFIG) as f:
            k = (json.load(f).get("api_key") or "").strip()
            if k:
                return k
    except Exception:
        pass
    return os.environ.get("ODDS_API_KEY", "").strip()


def parse_odds_events(events):
    """The Odds API 'spreads' events -> {(away, home): {"med": median home point, "home": [[book, point, price]...], "away": [...]}}
    (points use sportsbook sign: -3.5 = favored by 3.5)."""
    out = {}
    for ev in events:
        h, a = TEAM_NAMES.get(ev.get("home_team")), TEAM_NAMES.get(ev.get("away_team"))
        if not h or not a:
            continue
        det = {"home": [], "away": []}
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                if m.get("key") != "spreads":
                    continue
                for o in m.get("outcomes", []):
                    if o.get("point") is None:
                        continue
                    side = "home" if o.get("name") == ev.get("home_team") else ("away" if o.get("name") == ev.get("away_team") else None)
                    if side:
                        det[side].append([b.get("title") or b.get("key"), float(o["point"]), o.get("price")])
        if det["home"]:
            out[(a, h)] = {"med": float(np.median([x[1] for x in det["home"]])), "home": det["home"], "away": det["away"]}
    return out


def best_line(detail, side):
    """Best number for a bet on `side` ('home'/'away') across books: highest point, ties broken by the better price. -> (book, point, price)"""
    rows = (detail or {}).get(side) or []
    if not rows:
        return None
    return max(rows, key=lambda x: (x[1], x[2] if x[2] is not None else -999))


def fetch_odds_lines(refresh=False):
    """Vegas spreads from The Odds API (1 credit), cached 3 hours. -> {(away, home): {"med", "home": [...], "away": [...]}}"""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "odds_lines.json"
    if f.exists() and not refresh and time.time() - f.stat().st_mtime < 3 * 3600:
        try:
            raw = json.load(open(f))
            cached = {tuple(k.split("|")): v for k, v in raw.items()}
            if all(isinstance(v, dict) and "med" in v for v in cached.values()):
                return cached
        except Exception:
            pass
    key = _api_key()
    if not key:
        return {}
    import urllib.request
    import urllib.parse
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?" + urllib.parse.urlencode(
        {"apiKey": key, "regions": "us", "markets": "spreads", "oddsFormat": "american"})
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            events = json.loads(r.read().decode())
    except Exception as e:
        say(f"  (live lines unavailable: {e})")
        return {}
    out = parse_odds_events(events)
    try:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in out.items()}, open(f, "w"))
    except Exception:
        pass
    return out


DETAIL = {}          # game_id -> per-book lines from the Odds API, filled by assemble_lines()


def assemble_lines(up, refresh=False):
    """Pick the best available line for each upcoming game. Returns {game_id: (spread_home_positive, source)}."""
    manual = {}
    if MANUAL_LINES.exists():
        try:
            m = pd.read_csv(MANUAL_LINES)
            for r in m.itertuples(index=False):
                manual[(fix_team(r.away), fix_team(r.home))] = float(r.home_spread)
        except Exception as e:
            say(f"  (could not read {MANUAL_LINES.name}: {e})")
    odds = fetch_odds_lines(refresh)
    res = {}
    for r in up.itertuples(index=False):
        k = (r.away, r.home)
        if k in manual:
            res[r.game_id] = (-manual[k], "manual file")
        elif k in odds:
            res[r.game_id] = (-odds[k]["med"], "Odds API consensus")
            DETAIL[r.game_id] = odds[k]
        elif not pd.isna(r.spread):
            res[r.game_id] = (float(r.spread), "nflverse schedule")
    return res


# ------------------------------------------------------------------------------------------------ QB injuries
def qb_injury_override(tg, cur, wk, up):
    try:
        inj = _nfl().load_injuries(seasons=[cur]).to_pandas()
        inj = inj[(inj["week"] == wk) & (inj["position"] == "QB") & (inj["report_status"].astype(str).str.lower() == "out")]
        out_ids = set(inj["gsis_id"].dropna())
    except Exception:
        return {}
    ov = {}
    for r in up.itertuples(index=False):
        for team, qid in ((r.home, r.home_qb), (r.away, r.away_qb)):
            if qid in out_ids:
                t = tg[(tg["season"] == cur) & (tg["team"] == team) & (tg["qb_id"] != qid) & tg["qb_id"].notna()]
                bk = t.groupby("qb_id")["qb_n"].sum().sort_values(ascending=False)
                ov[team] = bk.index[0] if len(bk) else None
    return ov


def apply_qb_override(up, ov, helper):
    up = up.copy()
    for i, r in up.iterrows():
        for side, team in (("h", r["home"]), ("a", r["away"])):
            if team in ov:
                rt, n = helper["qb_rating"](ov[team])
                up.at[i, f"qb_{side}"], up.at[i, f"qb_n_{side}"], up.at[i, f"qb_chg_{side}"] = rt, math.log1p(n), 1.0
                up.at[i, f"{'home' if side == 'h' else 'away'}_qb"] = ov[team]
    return add_derived(up)


# ------------------------------------------------------------------------------------------------ picks
def fmt_line(team, pts):
    """Sportsbook style: BUF -6.5 / NYJ +6.5 / BUF PK"""
    if abs(pts) < 1e-9:
        return f"{team} PK"
    return f"{team} {pts:+.1f}".replace(".0", ".0")


def make_picks(up, bundle, stats, srcs):
    pr = bundle.predict(up)
    raw = pr["blend"]
    ok = stats.get("ok")
    c = stats["c"] if ok else 0.30
    sigma = stats["sigma"] if ok else SIGMA_DEFAULT
    cuts = stats["cuts"] if ok else {"C": 1.0, "B": 1.8, "A": 2.8}
    rows = []
    for i, r in enumerate(up.reset_index(drop=True).itertuples(index=False)):
        edge = c * raw[i]                                   # shrunk points; + = home covers
        side_home = raw[i] > 0
        pick_team = r.home if side_home else r.away
        vegas_home = -float(r.spread)                       # sportsbook sign for the home team
        pick_pts = vegas_home if side_home else -vegas_home
        tier = tier_of(abs(raw[i]), cuts)
        pcov = _phi(abs(edge) / sigma)
        margin = float(r.spread) + edge
        tot = float(r.total)
        tr = stats["by_tier"].get(tier) if ok and tier != "PASS" else None
        proven = bool(ok and tier != "PASS" and stats["proven"].get(tier))
        bl = best_line(DETAIL.get(r.game_id), "home" if side_home else "away")
        reasons = bundle.reasons(up.reset_index(drop=True).iloc[[i]].iloc[0])
        why = "" if tier == "PASS" else "; ".join(f"{lab} (favors {'home' if v > 0 else 'away'})" for lab, v in reasons[:3])
        rows.append({
            "game_id": r.game_id, "season": r.season, "week": r.week, "kickoff": str(r.gameday)[:10], "away": r.away, "home": r.home,
            "vegas_home": vegas_home, "line_source": srcs.get(r.game_id, ("", ""))[1],
            "model_home_line": -margin, "pick_team": pick_team, "pick_line": pick_pts, "pick_side": "home" if side_home else "away",
            "edge_pts": abs(edge), "raw_pred": float(raw[i]), "cover_prob": pcov, "tier": tier, "proven": proven,
            "tier_oos": (f"{tr['pct'] * 100:.1f}% on {tr['n']}" if tr and tr["n"] else ""),
            "pred_home": (tot + margin) / 2.0, "pred_away": (tot - margin) / 2.0,
            "qb_home_change": int(r.qb_chg_h), "qb_away_change": int(r.qb_chg_a), "why": why,
            "best_book": bl[0] if bl else "", "best_point": bl[1] if bl else np.nan, "best_price": bl[2] if bl else np.nan,
            "ats_home": f"{r.home} {r.rec_h_all} (home {r.rec_h_ven})", "ats_away": f"{r.away} {r.rec_a_all} (road {r.rec_a_ven})",
            "ridge": float(pr["ridge"][i]), "mlp": float(pr["mlp"][i]) if "mlp" in pr else np.nan, "gbm": float(pr["gbm"][i]) if "gbm" in pr else np.nan})
    p = pd.DataFrame(rows)
    order = {"A": 0, "B": 1, "C": 2, "PASS": 3}
    p["_o"] = p["tier"].map(order)
    return p.sort_values(["_o", "edge_pts"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def grade_and_log(picks, sched):
    """Append this week's picks to the running log, grade any logged pick whose game is final, return the live record."""
    cols = ["game_id", "season", "week", "away", "home", "logged_at", "vegas_home", "pick_team", "pick_side", "edge_pts", "raw_pred",
            "tier", "proven", "result_margin", "ats", "pick_line", "best_book", "best_point", "close_home", "clv"]
    log = pd.read_csv(LOG_CSV) if LOG_CSV.exists() else pd.DataFrame(columns=cols)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    if picks is not None and len(picks):
        keep = log[~log["game_id"].isin(picks["game_id"]) | log["ats"].isin(["W", "L", "P"])]
        new = picks.copy()
        new["logged_at"] = now
        new["result_margin"] = np.nan
        new["ats"] = ""
        new["close_home"], new["clv"] = np.nan, np.nan
        log = pd.concat([keep, new[cols]], ignore_index=True)
    for c in ("pick_line", "best_book", "best_point", "close_home", "clv"):
        if c not in log.columns:
            log[c] = np.nan
    res = sched.set_index("game_id")["result"].to_dict()
    closing = sched.set_index("game_id")["spread_line"].to_dict()
    for i, r in log.iterrows():
        if r["ats"] in ("W", "L", "P"):
            if pd.isna(r.get("clv")) and not pd.isna(closing.get(r["game_id"], np.nan)):
                ch = -float(closing[r["game_id"]])
                q0 = float(r["vegas_home"]) * (1 if r["pick_side"] == "home" else -1)
                q1 = ch * (1 if r["pick_side"] == "home" else -1)
                log.at[i, "close_home"], log.at[i, "clv"] = ch, q0 - q1
            continue
        m = res.get(r["game_id"])
        if m is None or pd.isna(m):
            continue
        t = m - (-float(r["vegas_home"]))                   # margin minus the home-positive spread
        side = 1.0 if r["pick_side"] == "home" else -1.0
        log.at[i, "result_margin"] = m
        log.at[i, "ats"] = "P" if t == 0 else ("W" if np.sign(t) * side > 0 else "L")
        if not pd.isna(closing.get(r["game_id"], np.nan)):
            ch = -float(closing[r["game_id"]])                 # closing home number, sportsbook sign
            q0 = float(r["vegas_home"]) * (1 if r["pick_side"] == "home" else -1)
            q1 = ch * (1 if r["pick_side"] == "home" else -1)
            log.at[i, "close_home"], log.at[i, "clv"] = ch, q0 - q1    # + = the number we logged beat the closing number
    log.to_csv(LOG_CSV, index=False)
    g = log[log["ats"].isin(["W", "L"]) & log["tier"].isin(["A", "B", "C"])]      # NO PLAY games are logged but never counted
    live = {}
    for t in ("A", "B", "C", "ALL"):
        s = g if t == "ALL" else g[g["tier"] == t]
        live[t] = (int((s["ats"] == "W").sum()), int((s["ats"] == "L").sum()))
    cv = log[log["tier"].isin(["A", "B", "C"]) & log["clv"].notna()]["clv"].astype(float)
    live["clv"] = (len(cv), float(cv.mean()) if len(cv) else 0.0, float((cv > 0).mean()) if len(cv) else 0.0)
    return live


# ------------------------------------------------------------------------------------------------ PDF
def write_pdf(picks, stats, live, wk, cur, path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    ss = ParagraphStyle("s", fontName="Helvetica", fontSize=7.6, leading=9.2)
    sb = ParagraphStyle("sb", parent=ss, fontName="Helvetica-Bold")
    sc = ParagraphStyle("sc", parent=ss, alignment=1)
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16, leading=19)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=3)
    note = ParagraphStyle("n", fontName="Helvetica-Oblique", fontSize=7, leading=8.6, textColor=colors.HexColor("#555555"))
    doc = SimpleDocTemplate(str(path), pagesize=landscape(letter), leftMargin=28, rightMargin=28, topMargin=26, bottomMargin=26,
                            title="NFL Spread AI")
    story = [Paragraph(f"THE SPREAD AI -- NFL Week {wk}, {cur}", h1),
             Paragraph("Picks against the Vegas spread only. Model = ridge + neural-net ensemble + gradient boosting on opponent-adjusted "
                       "play-by-play ratings, QB, rest and context, trained to predict how far the result lands from the line.", note)]
    ok = stats.get("ok")
    if ok:
        proven = [t for t, v in stats["proven"].items() if v]
        o = stats["overall"]
        banner = (f"BACKTEST (walk-forward, {stats['seasons'][0]}-{stats['seasons'][1]}, {stats['n']} games never seen in training): "
                  f"all picks {o['pct'] * 100:.1f}% ATS (95% CI {o['lo'] * 100:.1f}-{o['hi'] * 100:.1f}%), break-even 52.4%. ")
        banner += (f"Tier(s) {', '.join(proven)} VALIDATED out-of-sample (54%+ on 100+ games, p<0.10 vs break-even)." if proven else
                   "NO tier is VALIDATED out-of-sample (needs 54%+ on 100+ games and p<0.10 vs break-even): treat every pick as an UNPROVEN lean, not a bankable edge.")
        bg = "#E8F5E9" if proven else "#FFF3E0"
    else:
        banner, bg = "No backtest is available yet (run the program once with network access to nflverse). All picks are UNPROVEN.", "#FFF3E0"
    bt = Table([[Paragraph(banner, ss)]], colWidths=[736])
    bt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#999999")),
                            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [Spacer(1, 5), bt, Spacer(1, 6)]
    hdr = ["TIER", "GAME", "DATE", "VEGAS", "MODEL", "PICK", "EDGE", "COVER", "PROJ SCORE", "ATS THIS SEASON", "WHY (biggest drivers)"]
    data = [[Paragraph(f"<b>{h}</b>", sc) for h in hdr]]
    tcol = {"A": "#2E7D32", "B": "#66BB6A", "C": "#C8E6C9", "PASS": "#EEEEEE"}
    cmds = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9D9D9")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8)]
    for i, r in enumerate(picks.itertuples(index=False), start=1):
        tl = r.tier if r.tier == "PASS" else f"{r.tier}{' *' if r.proven else ''}"
        pickc = "NO PLAY" if r.tier == "PASS" else f"<b>{fmt_line(r.pick_team, r.pick_line)}</b>"
        if r.tier != "PASS" and isinstance(r.best_book, str) and r.best_book and not pd.isna(r.best_point):
            pr_ = "" if pd.isna(r.best_price) else f" ({int(r.best_price):+d})"
            pickc += f"<br/><font size=5.8>best: {r.best_book} {r.best_point:+.1f}{pr_}</font>"
        qbn = (" QB chg" if (r.qb_home_change or r.qb_away_change) else "")
        data.append([Paragraph(f"<b>{tl}</b>" + (f"<br/><font size=5.8>{r.tier_oos}</font>" if r.tier_oos else ""), sc),
                     Paragraph(f"{r.away} @ {r.home}{qbn}", sb), Paragraph(r.kickoff[5:], sc),
                     Paragraph(fmt_line(r.home, r.vegas_home) + f"<br/><font size=5.8>{r.line_source}</font>", sc),
                     Paragraph(fmt_line(r.home, r.model_home_line), sc), Paragraph(pickc, sc),
                     Paragraph(f"{r.edge_pts:.1f}", sc), Paragraph(f"{r.cover_prob * 100:.0f}%", sc),
                     Paragraph(f"{r.away} {r.pred_away:.0f}, {r.home} {r.pred_home:.0f}", sc),
                     Paragraph(f"{r.ats_home}<br/>{r.ats_away}", ParagraphStyle("a", parent=ss, fontSize=6.3, leading=7.6)),
                     Paragraph(r.why or "", ss)])
        cmds.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(tcol[r.tier])))
        if r.tier == "A":
            cmds.append(("TEXTCOLOR", (0, i), (0, i), colors.white))
    t = Table(data, colWidths=[46, 80, 40, 62, 52, 66, 34, 38, 76, 104, 138], repeatRows=1)
    t.setStyle(TableStyle(cmds))
    story += [t, Spacer(1, 5),
              Paragraph("TIER = how large the model's disagreement with Vegas is (A = top 5% of out-of-sample disagreements, B = top 15%, C = top 30%; "
                        "below that = NO PLAY). * = that tier is VALIDATED out-of-sample (54%+ on 100+ games, p<0.10 vs break-even); the small print under a tier is its real "
                        "backtest ATS record. EDGE = shrunk points of value vs the line. COVER % = estimated chance the pick covers (a normal-curve "
                        "estimate; ignores key-number effects at 3 and 7). MODEL = the program's fair line, PROJ SCORE splits the market total. "
                        "ATS THIS SEASON = each team\'s record against the spread (W-L-push), overall and at this game\'s venue. Betting 1 unit flat at -110 needs 52.4% to break even.", note)]
    story.append(PageBreak())
    story.append(Paragraph("HOW IT HAS REALLY DONE", h1))
    if ok:
        r_ = lambda r: f"{r['pct'] * 100:.1f}%  ({r['w']}-{r['l']}{'-' + str(r['p']) if r['p'] else ''}, n={r['n']})  95% CI {r['lo'] * 100:.1f}-{r['hi'] * 100:.1f}%   ROI {r['roi'] * 100:+.1f}%"
        rows2 = [["", "Out-of-sample ATS record"]]
        rows2.append(["All picks", r_(stats["overall"])])
        for k, v in stats["per_model"].items():
            rows2.append([f"  {k} alone", r_(v)])
        for tt in ("A", "B", "C"):
            rows2.append([f"Tier {tt}" + ("  (VALIDATED)" if stats["proven"][tt] else ""), r_(stats["by_tier"][tt])])
        for s, v in stats["by_season"].items():
            rows2.append([f"Season {s}", r_(v)])
        tb = Table([[Paragraph(str(a), sb if i == 0 else ss), Paragraph(b, ss)] for i, (a, b) in enumerate(rows2)], colWidths=[140, 560])
        tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9D9D9"))]))
        story += [tb, Spacer(1, 6),
                  Paragraph(f"Correlation of the model with (real margin - line): {stats['corr']:+.3f}. Shrink slope earned out-of-sample: {stats['c']:.2f} "
                            f"(1.00 would mean the raw model is perfectly calibrated; NFL models usually earn well under 0.5). "
                            f"Average miss: line alone {stats['mae_line']:.2f} pts, line + shrunk model {stats['mae_model']:.2f} pts.", note)]
    lw = live.get("ALL", (0, 0))
    story += [Paragraph("LIVE RECORD (every pick this program has logged since you started running it)", h2)]
    if lw[0] + lw[1] == 0:
        story.append(Paragraph("No graded picks yet -- they grade themselves the next time you run it after the games are final.", ss))
    else:
        txt = "   |   ".join(f"{k}: {v[0]}-{v[1]} ({100 * v[0] / max(1, v[0] + v[1]):.0f}%)" for k, v in live.items() if k != "clv" and v[0] + v[1] > 0)
        story.append(Paragraph(txt, ss))
    cl = live.get("clv", (0, 0.0, 0.0))
    if cl[0]:
        story.append(Paragraph(f"CLOSING-LINE VALUE: across {cl[0]} graded plays the number logged at pick time was on average {cl[1]:+.2f} points "
                               f"{'better' if cl[1] >= 0 else 'worse'} than the closing number, and beat the close {cl[2] * 100:.0f}% of the time. "
                               f"(Consistently positive CLV is the best early sign of a real edge, long before win-loss settles.)", ss))
    story += [Spacer(1, 8), Paragraph("Data: nflverse (CC BY 4.0). Live lines: The Odds API consensus (median of US books) unless a manual file is present. "
                                      "This is analysis, not a guarantee. Past out-of-sample results do not promise future ones.", note)]
    doc.build(story)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    global FIRST_SEASON, FIRST_TEST_SEASON
    ap = argparse.ArgumentParser(description="NFL spread picks")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--week", type=int)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--refresh-lines", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--ablation", action="store_true", help="test which groups of inputs help")
    ap.add_argument("--mlps", type=int, default=7, help="number of neural nets in the ensemble")
    a = ap.parse_args(argv)
    if a.fast:
        FIRST_SEASON, FIRST_TEST_SEASON = 2016, 2020
    cur = current_season()
    say("=" * 100)
    say(f"THE SPREAD AI  --  {cur} season, history from {FIRST_SEASON}")
    say("=" * 100)
    if a.rebuild:
        for f in CACHE.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
    say("Loading schedules + lines ...")
    sched = normalize_schedule(load_schedule_df(range(FIRST_SEASON, cur + 1)))
    say("Building team ratings from play-by-play (first run downloads ~%d seasons, later runs are fast) ..." % (cur - FIRST_SEASON + 1))
    tg = get_team_games(FIRST_SEASON, cur, rebuild=a.rebuild, cur=cur)
    games, helper = build_games(sched, tg)
    say(f"  {len(games)} games, {int(games['result'].notna().sum())} played")
    say("Walk-forward backtest (cached; only new blocks are recomputed) ...")
    oos = walk_forward(games, cur, n_mlp=a.mlps)
    stats = evaluate(oos)
    if a.ablation:
        ablation(games, cur)
        if a.backtest:
            return
    if a.backtest:
        return
    played_cur = games[(games["season"] == cur) & games["result"].notna()]
    todo = games[(games["season"] == cur) & games["result"].isna()]
    if todo.empty:
        say("\nNo unplayed games left in this season's schedule -- nothing to pick.")
        return
    wk = a.week or int(todo["week"].min())
    up = todo[todo["week"] == wk].copy().reset_index(drop=True)
    say(f"\nPicking Week {wk}: {len(up)} games")
    ov = qb_injury_override(tg, cur, wk, up)
    if ov:
        say("  QB out per the injury report -> using the backup: " + ", ".join(ov))
    srcs = assemble_lines(up, a.refresh_lines)
    keep = []
    for i, r in up.iterrows():
        if r["game_id"] in srcs:
            up.at[i, "spread"] = srcs[r["game_id"]][0]
            if pd.isna(r["total"]):
                up.at[i, "total"] = 45.0
            keep.append(i)
        else:
            say(f"  !! no line for {r['away']} @ {r['home']} -- skipped (add it to nfl_spread_manual_lines.csv)")
    up = up.loc[keep].reset_index(drop=True)
    if up.empty:
        say("No lines available, so no picks. Put your Odds API key in novig_config.json or fill nfl_spread_manual_lines.csv.")
        return
    up = add_derived(up)
    if ov:
        up = apply_qb_override(up, ov, helper)
    say("Training the final models on every completed game ...")
    bundle = Bundle(a.mlps).fit(usable(games))
    picks = make_picks(up, bundle, stats, srcs)
    picks.to_csv(PICKS_CSV, index=False)
    live = grade_and_log(picks, sched)
    GEN.mkdir(exist_ok=True)
    pdf = GEN / f"NFL_Spread_Picks_week{wk}.pdf"
    write_pdf(picks, stats, live, wk, cur, pdf)
    say("")
    say("=" * 100)
    say(f"WEEK {wk} PICKS AGAINST THE SPREAD")
    say("=" * 100)
    for r in picks.itertuples(index=False):
        flag = "NO PLAY" if r.tier == "PASS" else f"{fmt_line(r.pick_team, r.pick_line):<10} tier {r.tier}{'*' if r.proven else ' '} edge {r.edge_pts:4.1f} cover {r.cover_prob * 100:3.0f}%"
        say(f"  {r.away:>3} @ {r.home:<3}  Vegas {fmt_line(r.home, r.vegas_home):<9} model {fmt_line(r.home, r.model_home_line):<9}  -> {flag}")
    say("")
    say(f"  CSV: {PICKS_CSV.name}   log: {LOG_CSV.name}   PDF: generated/{pdf.name}")
    lw = live["ALL"]
    if lw[0] + lw[1]:
        say(f"  Live record so far: {lw[0]}-{lw[1]} ATS (plays only)")
    cl = live.get("clv", (0, 0.0, 0.0))
    if cl[0]:
        say(f"  Closing-line value so far: {cl[1]:+.2f} pts on average over {cl[0]} plays (beat the close {cl[2] * 100:.0f}% of the time)")


if __name__ == "__main__":
    main()
