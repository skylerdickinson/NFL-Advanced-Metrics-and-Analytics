#!/usr/bin/env python3
"""
nfl_workhorse_tracker.py - the "workhorse tracker": follows each team's lead back, top pass-catchers and QB
the way you follow a pitcher (last-3 form, results vs expected, wear-and-tear traces) and answers
"how much does this team need him?" when he pops up on the injury list.

Per tracked player (RB1, top 3 WR/TE by targets, QB1 for every team) it shows:
  USAGE        season avg vs last 3 games (touches / targets / pass attempts, snap %, share of team opportunities)
  EFFICIENCY   season vs last 3: rush yards over expected per carry (NGS), yds/carry, separation, YAC over
               expectation, EPA per dropback -- whichever apply to his position and exist in the data
  VS EXPECTED  fantasy points above/below what his opportunities should have produced (the NFL "ERA minus FIP":
               high = running hot on touchdowns/luck, due to cool off; low = running cold, due to pop)
  TEAM W/O HIM team points per game in games he played vs games he missed (this + last season), with sample size
  FLAGS        WORKLOAD (heavy / spike / pace), DECLINE (efficiency fell vs his own baseline), HOT / COLD,
               INJURY (status from nfl_injury_report.csv + what the team scores without him + who absorbs his work)

Reads:   nfl_injury_report.csv (from the weekly data pull) + nflverse (via nflreadpy) for the player/NGS/snap/
         schedule data; everything downloaded is cached in nfl_workhorse_cache/ so --no-pull works offline.
Writes:  generated/NFL_Workhorse_Tracker_<weektag>.pdf   and   nfl_workhorse_tracker.csv

Run:     python3 nfl_workhorse_tracker.py [--season 2026] [--no-pull]        (normally a step of nfl_run_all.py)
Test:    python3 nfl_workhorse_tracker.py --selftest        (math + flags on made-up players, no internet)

HONESTY NOTES
  * "Mechanical breakdown" is inferred from performance traces (NGS efficiency/separation, yards per carry,
    EPA) falling against the player's own baseline. It is a smoke detector, not a diagnosis.
  * Thresholds are judgment calls (the constants below), not backtested. Tune them here.
  * Team-without-him numbers come from few games; the sample size is always printed. n < 2 shows "n/a".
"""
import sys
sys.dont_write_bytecode = True
import argparse
import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GEN_DIR = HERE / "generated"
CACHE_DIR = HERE / "nfl_workhorse_cache"

# ------------------------------------------------------------------ thresholds (judgment calls - tune freely)
RB_HEAVY_TOUCHES_L3 = 24.0      # avg touches (carries + receptions) over last 3 -> HEAVY
RB_SPIKE_RATIO = 1.20           # last-3 touches vs his own season-before-that avg ...
RB_SPIKE_MIN = 18.0             # ... and at least this many -> SPIKE (sudden jump in workload)
RB_PACE_TOUCHES = 340.0         # season pace (touches/game x 17) -> HIGH PACE (RB seasons past ~350-370 historically fall off)
RB_SNAP_HEAVY = 75.0            # offensive snap % with 20+ touches
WR_HEAVY_TARGETS_L3 = 11.0
RYOE_DROP = 0.6                 # NGS rush yards over expected per attempt, last 3 below baseline by this much
YPC_DROP = 1.0
REPA_DROP = 0.15                # rushing EPA per carry
SEP_DROP = 0.30                 # NGS avg separation (yards)
YACOE_DROP = 0.8                # NGS YAC above expectation
TGT_SHARE_DROP = 6.0            # target share, percentage points
QB_EPA_DROP = 0.20              # EPA per dropback
HOT_POE = 6.0                   # last-3 fantasy points per game ABOVE expected
COLD_POE = -6.0
DEP_DELTA = 4.0                 # team scores this many fewer points per game without him -> TEAM DEPENDENT
DEP_MIN_WITHOUT = 2             # need at least this many games without him
MIN_BASE_GAMES = 4              # need this many baseline games before trusting a "decline" comparison

POS_OK = {"QB", "RB", "WR", "TE", "FB"}
TEAM_FIX = {"LAR": "LA", "STL": "LA", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "SDG": "LAC",
            "LVR": "LV", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SF": "SF", "GNB": "GB",
            "KAN": "KC", "NOR": "NO", "NWE": "NE", "SFO": "SF", "TAM": "TB"}


# ================================================================== small helpers
def norm_name(s):
    s = str(s).lower()
    s = re.sub(r"[.'`’\-]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fix_team(t):
    t = str(t).strip().upper()
    return TEAM_FIX.get(t, t)


def first_present(df, cands, default=None):
    for c in cands:
        if c in df.columns:
            return c
    return default


def num(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def num0(df, col):
    return num(df, col).fillna(0.0)


def sdiv(a, b):
    return float(a) / float(b) if b not in (0, 0.0, None) and not pd.isna(b) and float(b) != 0 else np.nan


def fmt(v, f="{:.1f}", dash="--"):
    return dash if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def fmts(v, f="{:+.1f}", dash="--"):
    return fmt(v, f, dash)


# ================================================================== loading (nflreadpy on the user's Mac, cached)
def _save_cache(name, df):
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(CACHE_DIR / f"{name}.csv.gz", index=False, compression="gzip")


def _load_cache(name):
    p = CACHE_DIR / f"{name}.csv.gz"
    return pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()


def load_raw(season, pull=True):
    """Return dict of raw frames: weekly, snaps, ngs_rush, ngs_rec, sched. Falls back to cache per table."""
    raw = {}
    seasons = [season - 1, season]
    nfl = None
    if pull:
        try:
            import nflreadpy as nfl  # noqa
        except Exception as e:
            print(f"[workhorse] nflreadpy not importable ({e}); using cached data if any")
            nfl = None
    loaders = {
        "weekly": lambda: nfl.load_player_stats(seasons=seasons, summary_level="week"),
        "snaps": lambda: nfl.load_snap_counts(seasons=seasons),
        "ngs_rush": lambda: nfl.load_nextgen_stats(stat_type="rushing", seasons=seasons),
        "ngs_rec": lambda: nfl.load_nextgen_stats(stat_type="receiving", seasons=seasons),
        "sched": lambda: nfl.load_schedules(seasons=seasons),
    }
    required = {"weekly", "sched"}
    for name, fn in loaders.items():
        df = pd.DataFrame()
        if nfl is not None:
            try:
                df = fn().to_pandas()
                _save_cache(name, df)
                print(f"[workhorse] pulled {name}: {len(df)} rows")
            except Exception as e:
                print(f"[workhorse] pull of {name} failed ({e}); trying cache")
        if df.empty:
            df = _load_cache(name)
            if not df.empty:
                print(f"[workhorse] using cached {name}: {len(df)} rows")
        if df.empty and name in required:
            raise SystemExit(f"[workhorse] no {name} data (no internet and nothing cached yet). "
                             f"Run once with internet: python3 nfl_workhorse_tracker.py")
        raw[name] = df
    return raw


# ================================================================== standardising the tables
def std_weekly(w):
    w = w.copy()
    if "season_type" in w.columns:
        w = w[w["season_type"].astype(str).str.upper() == "REG"]
    nm = first_present(w, ["player_display_name", "player_name"])
    tm = first_present(w, ["team", "recent_team"])
    pos = first_present(w, ["position", "pos"])
    out = pd.DataFrame({
        "pid": w["player_id"].astype(str) if "player_id" in w.columns else w[nm].astype(str),
        "player": w[nm].astype(str),
        "team": w[tm].map(fix_team),
        "pos": w[pos].astype(str).str.upper().replace({"FB": "RB"}),
        "season": num(w, "season").astype(int),
        "week": num(w, "week").astype(int),
    }, index=w.index)
    for c in ["carries", "rushing_yards", "rushing_tds", "rushing_epa", "targets", "receptions", "receiving_yards",
              "receiving_tds", "receiving_epa", "attempts", "passing_yards", "passing_tds", "passing_epa"]:
        out[c] = num0(w, c)
    ic = first_present(w, ["passing_interceptions", "interceptions"])
    out["ints"] = num0(w, ic) if ic else 0.0
    sc = first_present(w, ["sacks_suffered", "sacks"])
    out["sacks"] = num0(w, sc) if sc else 0.0
    out["target_share"] = num(w, "target_share") * 100.0
    out["ay_share"] = num(w, "air_yards_share") * 100.0
    out = out[out["pos"].isin({"QB", "RB", "WR", "TE"})].copy()
    out["nname"] = out["player"].map(norm_name)
    out["touches"] = out["carries"] + out["receptions"]
    out["opps"] = out["carries"] + out["targets"]
    out["ppr_act"] = (0.1 * out["rushing_yards"] + 6 * out["rushing_tds"] + out["receptions"] + 0.1 * out["receiving_yards"]
                      + 6 * out["receiving_tds"] + 0.04 * out["passing_yards"] + 4 * out["passing_tds"] - 2 * out["ints"])
    return out.reset_index(drop=True)


def std_snaps(s):
    if s is None or s.empty:
        return pd.DataFrame(columns=["nname", "season", "week", "snap_pct"])
    s = s.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"].astype(str).str.upper() == "REG"]
    nm = first_present(s, ["player", "player_name", "full_name"])
    pc = first_present(s, ["offense_pct"])
    if not nm or not pc:
        return pd.DataFrame(columns=["nname", "season", "week", "snap_pct"])
    o = pd.DataFrame({"nname": s[nm].map(norm_name), "season": num(s, "season"), "week": num(s, "week"),
                      "snap_pct": num(s, pc) * 100.0}).dropna()
    o["season"] = o["season"].astype(int); o["week"] = o["week"].astype(int)
    return o.groupby(["nname", "season", "week"], as_index=False)["snap_pct"].max()


def std_ngs(d, kind):
    cols = {"rush": ["ryoe_att", "tlos", "rush_att"], "rec": ["sep", "yacoe"]}[kind]
    if d is None or d.empty:
        return pd.DataFrame(columns=["nname", "season", "week"] + cols)
    d = d.copy()
    nm = first_present(d, ["player_display_name", "player_name"])
    wk = num(d, "week")
    d = d[wk > 0]
    if kind == "rush":
        o = pd.DataFrame({"nname": d[nm].map(norm_name), "season": num(d, "season"), "week": num(d, "week"),
                          "ryoe_att": num(d, "rush_yards_over_expected_per_att"), "tlos": num(d, "avg_time_to_los"),
                          "rush_att": num(d, "rush_attempts")})
    else:
        o = pd.DataFrame({"nname": d[nm].map(norm_name), "season": num(d, "season"), "week": num(d, "week"),
                          "sep": num(d, "avg_separation"), "yacoe": num(d, "avg_yac_above_expectation")})
    o = o.dropna(subset=["season", "week"])
    o["season"] = o["season"].astype(int); o["week"] = o["week"].astype(int)
    return o.groupby(["nname", "season", "week"], as_index=False).first()


def std_sched(s):
    s = s.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"].astype(str).str.upper() == "REG"]
    rows = []
    for g in s.itertuples():
        a, h = fix_team(g.away_team), fix_team(g.home_team)
        asc, hsc = getattr(g, "away_score", np.nan), getattr(g, "home_score", np.nan)
        played = pd.notna(asc) and pd.notna(hsc)
        gd = getattr(g, "gameday", None)
        gt = getattr(g, "gametime", None)
        rows.append(dict(season=int(g.season), week=int(g.week), team=a, opp=h, home=False, played=played,
                         pts_for=float(asc) if played else np.nan, pts_against=float(hsc) if played else np.nan,
                         gameday=gd, gametime=gt))
        rows.append(dict(season=int(g.season), week=int(g.week), team=h, opp=a, home=True, played=played,
                         pts_for=float(hsc) if played else np.nan, pts_against=float(asc) if played else np.nan,
                         gameday=gd, gametime=gt))
    return pd.DataFrame(rows)


def build_tables(raw):
    wk = std_weekly(raw["weekly"])
    sn = std_snaps(raw.get("snaps"))
    nr = std_ngs(raw.get("ngs_rush"), "rush")
    nc = std_ngs(raw.get("ngs_rec"), "rec")
    key = ["nname", "season", "week"]
    wk = wk.merge(sn, on=key, how="left").merge(nr, on=key, how="left").merge(nc, on=key, how="left")
    wk["played"] = ((wk["opps"] > 0) | (wk["attempts"] > 0) | (wk["snap_pct"].fillna(0) > 0))
    sched = std_sched(raw["sched"])
    return wk, sched


# ================================================================== expected fantasy points (opportunity-based)
def expected_coefs(wk):
    """League-average PPR points per carry / per target / per pass attempt, by position, from all loaded games."""
    co = {}
    for pos, g in wk.groupby("pos"):
        rush_pts = (0.1 * g["rushing_yards"] + 6 * g["rushing_tds"]).sum()
        rec_pts = (g["receptions"] + 0.1 * g["receiving_yards"] + 6 * g["receiving_tds"]).sum()
        pass_pts = (0.04 * g["passing_yards"] + 4 * g["passing_tds"] - 2 * g["ints"]).sum()
        co[pos] = dict(per_carry=sdiv(rush_pts, g["carries"].sum()), per_tgt=sdiv(rec_pts, g["targets"].sum()),
                       per_att=sdiv(pass_pts, g["attempts"].sum()))
    return co


def add_expected(wk, co):
    def f(r):
        c = co.get(r["pos"], {})
        e = 0.0
        for k, col in (("per_carry", "carries"), ("per_tgt", "targets"), ("per_att", "attempts")):
            v = c.get(k, np.nan)
            if not pd.isna(v):
                e += v * r[col]
        return e
    wk = wk.copy()
    wk["ppr_exp"] = wk.apply(f, axis=1)
    wk["poe"] = wk["ppr_act"] - wk["ppr_exp"]
    return wk


# ================================================================== per-player profile
def agg_rows(g, pos):
    """Rate metrics for a set of games."""
    if g.empty:
        return {}
    n = len(g)
    d = dict(n=n, touches=g["touches"].mean(), carries=g["carries"].mean(), targets=g["targets"].mean(),
             attempts=g["attempts"].mean(), snap=g["snap_pct"].mean(), tshare=g["target_share"].mean(),
             ayshare=g["ay_share"].mean(), poe=g["poe"].mean(),
             ypc=sdiv(g["rushing_yards"].sum(), g["carries"].sum()),
             repa=sdiv(g["rushing_epa"].sum(), g["carries"].sum()),
             ypt=sdiv(g["receiving_yards"].sum(), g["targets"].sum()),
             ryoe=g["ryoe_att"].mean(), tlos=g["tlos"].mean(), sep=g["sep"].mean(), yacoe=g["yacoe"].mean(),
             qbepa=sdiv(g["passing_epa"].sum(), (g["attempts"] + g["sacks"]).sum()),
             ypa=sdiv(g["passing_yards"].sum(), g["attempts"].sum()),
             ppr=g["ppr_act"].mean())
    return d


def team_share_series(wk, team, season):
    """Per week: team carries + targets, used for the share-of-team-opportunities number."""
    t = wk[(wk.team == team) & (wk.season == season)].groupby("week")[["carries", "targets"]].sum()
    return (t["carries"] + t["targets"]).to_dict()


def dependence(pid, team, wk, sched):
    """Team points per game with / without this player, over the loaded games he was (or could have been) on this team."""
    tg = sched[(sched.team == team) & (sched.played)].sort_values(["season", "week"]).reset_index(drop=True)
    if tg.empty:
        return dict(with_ppg=np.nan, without_ppg=np.nan, n_with=0, n_without=0, delta=np.nan)
    mine = wk[(wk.pid == pid) & (wk.team == team) & (wk.played)]
    played = set(zip(mine.season, mine.week))
    present = np.array([(s, w) in played for s, w in zip(tg.season, tg.week)])
    if not present.any():
        return dict(with_ppg=np.nan, without_ppg=np.nan, n_with=0, n_without=0, delta=np.nan)
    start = int(np.argmax(present))
    tg2, pr2 = tg.iloc[start:], present[start:]
    w_ = tg2.loc[pr2, "pts_for"]; o_ = tg2.loc[~pr2, "pts_for"]
    wp = w_.mean() if len(w_) else np.nan
    op = o_.mean() if len(o_) else np.nan
    return dict(with_ppg=wp, without_ppg=op, n_with=int(len(w_)), n_without=int(len(o_)),
                delta=(op - wp) if len(o_) and len(w_) else np.nan)


def profile(p, wk, sched, season):
    """p = dict(pid, player, team, pos). Return a flat dict of metrics + flags."""
    g = wk[(wk.pid == p["pid"]) & (wk.played)].sort_values(["season", "week"])
    cur = g[g.season == season]
    l3 = cur.tail(3)
    prior_cur = cur.iloc[:max(len(cur) - 3, 0)]
    base = g.drop(l3.index)                       # everything he has done that is NOT the last 3 (this + last season)
    szn, rec, bas = agg_rows(cur, p["pos"]), agg_rows(l3, p["pos"]), agg_rows(base, p["pos"])
    pre = agg_rows(prior_cur, p["pos"])
    tm = team_share_series(wk, p["team"], season)
    tot_opps_l3 = sum(tm.get(w, 0) for w in l3.week)
    opp_share_l3 = sdiv(l3["opps"].sum(), tot_opps_l3) * 100 if tot_opps_l3 else np.nan
    tot_opps_s = sum(tm.get(w, 0) for w in cur.week)
    opp_share_szn = sdiv(cur["opps"].sum(), tot_opps_s) * 100 if tot_opps_s else np.nan
    dep = dependence(p["pid"], p["team"], wk, sched)
    # games he missed this season while the team played (informational)
    tg_cur = sched[(sched.team == p["team"]) & (sched.season == season) & (sched.played)]
    missed = int(len(tg_cur) - len(cur))

    flags = []   # (code, level, text)
    pos = p["pos"]
    nb = bas.get("n", 0)
    if pos == "RB" and rec:
        t3 = rec["touches"]
        if t3 >= RB_HEAVY_TOUCHES_L3:
            flags.append(("HEAVY", "amber", f"{t3:.0f} touches/gm last 3"))
        ref = pre.get("touches") if pre else np.nan
        if not pd.isna(ref) and ref > 0 and t3 >= RB_SPIKE_MIN and t3 >= RB_SPIKE_RATIO * ref and t3 < RB_HEAVY_TOUCHES_L3:
            flags.append(("SPIKE", "amber", f"touches {ref:.0f} to {t3:.0f}"))
        elif not pd.isna(ref) and ref > 0 and t3 >= RB_SPIKE_MIN and t3 >= RB_SPIKE_RATIO * ref:
            flags[-1] = ("HEAVY+SPIKE", "amber", f"touches {ref:.0f} to {t3:.0f}/gm")
        if szn and szn["n"] >= 3 and szn["touches"] * 17 >= RB_PACE_TOUCHES:
            flags.append(("HIGH PACE", "amber", f"{szn['touches'] * 17:.0f}-touch season pace"))
        if not pd.isna(rec.get("snap", np.nan)) and rec["snap"] >= RB_SNAP_HEAVY and t3 >= 20:
            flags.append(("SNAPS", "amber", f"{rec['snap']:.0f}% of snaps"))
    if pos in ("WR", "TE") and rec:
        if rec["targets"] >= WR_HEAVY_TARGETS_L3:
            flags.append(("HEAVY", "amber", f"{rec['targets']:.0f} targets/gm last 3"))
    decline = []
    if nb >= MIN_BASE_GAMES and rec:
        def drop(key, thr, label, fmtstr):
            a, b = rec.get(key, np.nan), bas.get(key, np.nan)
            if not pd.isna(a) and not pd.isna(b) and (b - a) >= thr:
                decline.append(f"{label} {fmtstr.format(b)} to {fmtstr.format(a)}")
        if pos == "RB":
            drop("ryoe", RYOE_DROP, "RYOE/att", "{:+.1f}")
            drop("ypc", YPC_DROP, "YPC", "{:.1f}")
            drop("repa", REPA_DROP, "rush EPA/car", "{:+.2f}")
        elif pos in ("WR", "TE"):
            drop("sep", SEP_DROP, "separation", "{:.1f}")
            drop("yacoe", YACOE_DROP, "YAC over exp", "{:+.1f}")
            drop("tshare", TGT_SHARE_DROP, "tgt share", "{:.0f}%")
        elif pos == "QB":
            drop("qbepa", QB_EPA_DROP, "EPA/dropback", "{:+.2f}")
    if decline:
        flags.append(("DECLINE", "red", "; ".join(decline)))
    if rec:
        if rec["poe"] >= HOT_POE:
            flags.append(("HOT", "amber", f"{rec['poe']:+.1f} pts/gm above expected (regression risk)"))
        elif rec["poe"] <= COLD_POE:
            flags.append(("COLD", "teal", f"{rec['poe']:+.1f} pts/gm below expected (due to pop)"))
    if not pd.isna(dep["delta"]) and dep["n_without"] >= DEP_MIN_WITHOUT and dep["delta"] <= -DEP_DELTA:
        flags.append(("DEPENDENT", "red", f"team {dep['delta']:+.1f} pts/gm without him (n={dep['n_without']})"))

    out = dict(pid=p["pid"], player=p["player"], team=p["team"], pos=pos, games=szn.get("n", 0), missed=missed,
               opp_share_szn=opp_share_szn, opp_share_l3=opp_share_l3)
    for pre_, d in (("szn_", szn), ("l3_", rec), ("base_", bas)):
        for k in ("touches", "carries", "targets", "attempts", "snap", "tshare", "ayshare", "poe", "ypc", "repa", "ypt",
                  "ryoe", "tlos", "sep", "yacoe", "qbepa", "ypa", "ppr"):
            out[pre_ + k] = d.get(k, np.nan) if d else np.nan
    out.update(dict(with_ppg=dep["with_ppg"], without_ppg=dep["without_ppg"], n_with=dep["n_with"],
                    n_without=dep["n_without"], dep_delta=dep["delta"]))
    out["flaglist"] = flags
    return out


def tracked_players(wk, season):
    """RB1 (most carries), top 3 WR/TE by targets, QB1 (most attempts) for every team, current season."""
    cur = wk[wk.season == season]
    last_team = cur.sort_values("week").groupby("pid")["team"].last()
    agg = cur.groupby(["pid", "player", "pos"], as_index=False).agg(carries=("carries", "sum"), targets=("targets", "sum"),
                                                                     attempts=("attempts", "sum"))
    agg["team"] = agg["pid"].map(last_team)
    picks = []
    for team, t in agg.groupby("team"):
        rb = t[t.pos == "RB"].sort_values(["carries", "targets"], ascending=False).head(1)
        rc = t[t.pos.isin(["WR", "TE"])].sort_values("targets", ascending=False).head(3)
        qb = t[t.pos == "QB"].sort_values("attempts", ascending=False).head(1)
        for part in (qb, rb, rc):
            for r in part.itertuples():
                picks.append(dict(pid=r.pid, player=r.player, team=team, pos=r.pos))
    return picks


def next_man_up(wk, season, team, pos, exclude_pid):
    cur = wk[(wk.season == season) & (wk.team == team) & (wk.played)]
    grp = {"RB": ["RB"], "WR": ["WR", "TE"], "TE": ["WR", "TE"], "QB": ["QB"]}[pos]
    c = cur[(cur.pos.isin(grp)) & (cur.pid != exclude_pid)]
    if c.empty:
        return None, np.nan
    best = None
    for pid, g in c.groupby("pid"):
        v = g.sort_values("week").tail(3)["opps" if pos != "QB" else "attempts"].mean()
        if best is None or v > best[1]:
            best = (g["player"].iloc[0], v)
    return best


def load_injuries(path):
    p = Path(path)
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    out = {}
    for r in d.itertuples(index=False):
        team, player, pos, inj, status = (list(r) + [None] * 5)[:5]
        out[(norm_name(player), fix_team(team))] = (str(status), str(inj))
    return out


def run(wk, sched, season, inj):
    co = expected_coefs(wk)
    wk = add_expected(wk, co)
    rows = []
    for p in tracked_players(wk, season):
        r = profile(p, wk, sched, season)
        st = inj.get((norm_name(p["player"]), p["team"]))
        r["inj_status"], r["inj_detail"] = (st if st else ("", ""))
        if r["inj_status"] and r["inj_status"].lower() not in ("", "nan", "none"):
            nmu = next_man_up(wk, season, p["team"], p["pos"], p["pid"])
            r["next_man"], r["next_man_opps"] = (nmu if nmu and nmu[0] is not None else ("", np.nan))
            lvl = "red"
            dtxt = ""
            if r["n_without"] >= DEP_MIN_WITHOUT and not pd.isna(r["dep_delta"]):
                dtxt = f"; team {r['dep_delta']:+.1f} pts/gm without him (n={r['n_without']})"
            nx = f"; next man up {r['next_man']}" if r["next_man"] else ""
            r["flaglist"].insert(0, ("INJURY", lvl, f"{r['inj_status']} ({r['inj_detail']}){dtxt}{nx}"))
        else:
            r["next_man"], r["next_man_opps"] = "", np.nan
        rows.append(r)
    return pd.DataFrame(rows), wk



# ================================================================== next-game matchup outlook (all teams)
K_SHRINK = 3.0            # pseudo-games of "league average" blended in, so 3-4 game samples don't overreact
SOFT_RANK, TOUGH_RANK = 22, 8      # defense rank (1 = best of 32): >= SOFT is a soft D, <= TOUGH is a tough D


def load_outlook_inputs(matrix_path, log_path):
    mp, lp = Path(matrix_path), Path(log_path)
    if not mp.exists() or not lp.exists():
        return None, None
    m, g = pd.read_csv(mp), pd.read_csv(lp)
    m["team"] = m["team"].map(fix_team)
    g["team"] = g["team"].map(fix_team)
    g["opponent"] = g["opponent"].map(fix_team)
    return m, g


def d_ranks(m):
    """Defense ranks, 1 = best of 32. rush = EPA per rush allowed, pass = yards per pass allowed,
    sack = defensive sack rate (higher better), havoc = havoc rate (higher better)."""
    def col(cands):
        c = first_present(m, cands)
        return pd.to_numeric(m[c], errors="coerce") if c else pd.Series(np.nan, index=m.index)
    r = pd.DataFrame({"team": m["team"]})
    r["rush_rank"] = col(["def_rush_epa_allowed", "def_rushing_yards_per_play"]).rank(ascending=True, method="min")
    r["pass_rank"] = col(["def_yards_per_pass", "passing_yards_allowed_per_game"]).rank(ascending=True, method="min")
    r["sack_rank"] = col(["sack_pct_defense"]).rank(ascending=False, method="min")
    r["havoc_rank"] = col(["havoc_rate_defense"]).rank(ascending=False, method="min")
    return r.set_index("team")


def _shrunk_ratio(vals, k=K_SHRINK):
    v = [x for x in vals if not pd.isna(x) and np.isfinite(x)]
    n = len(v)
    return (sum(v) + k) / (n + k) if n else 1.0, n


def _shrunk_diff(vals, k=K_SHRINK):
    v = [x for x in vals if not pd.isna(x) and np.isfinite(x)]
    n = len(v)
    return sum(v) / (n + k) if n else 0.0


def outlook_one(team, opp, g, ranks):
    """How `team`'s offense has done against the defenses it has faced (each opponent's OTHER games as the yardstick,
    so a game never grades itself), then projected against `opp`'s defense."""
    gt = g[g.team == team]
    fr, faced = [], []
    for r in gt.itertuples():
        o = g[(g.team == r.opponent) & (g.week != r.week)]
        if o.empty:
            o = g[g.team == r.opponent]
        fr.append(dict(
            f_rush=sdiv(r.rush_yards, o["rush_yards_allowed"].mean()),
            f_pass=sdiv(r.pass_yards, o["pass_yards_allowed"].mean()),
            f_pts=sdiv(r.pts_for, o["pts_against"].mean()),
            d_sack=r.sack_pct_offense_game - o["sack_pct_defense_game"].mean()))
        faced.append(r.opponent)
    fr = pd.DataFrame(fr)
    go = g[g.team == opp]
    if fr.empty or go.empty:
        return None
    f_rush, n = _shrunk_ratio(fr["f_rush"]); f_pass, _ = _shrunk_ratio(fr["f_pass"]); f_pts, _ = _shrunk_ratio(fr["f_pts"])
    d_sack = _shrunk_diff(fr["d_sack"])
    base = dict(rush=go["rush_yards_allowed"].mean(), pas=go["pass_yards_allowed"].mean(), pts=go["pts_against"].mean(),
                sack=go["sack_pct_defense_game"].mean())
    orank = ranks.loc[opp] if opp in ranks.index else pd.Series(dtype=float)
    frank = ranks.reindex(faced).mean() if len(ranks) else pd.Series(dtype=float)
    out = dict(team=team, opp=opp, n_games=n,
               idx_rush=f_rush, idx_pass=f_pass, idx_pts=f_pts, sack_over=d_sack,
               proj_rush=base["rush"] * f_rush, proj_pass=base["pas"] * f_pass, proj_pts=base["pts"] * f_pts,
               proj_sack=max(base["sack"] + d_sack, 0.0),
               opp_rush_rank=orank.get("rush_rank", np.nan), opp_pass_rank=orank.get("pass_rank", np.nan),
               opp_sack_rank=orank.get("sack_rank", np.nan), opp_havoc_rank=orank.get("havoc_rank", np.nan),
               faced_rush_rank=frank.get("rush_rank", np.nan), faced_pass_rank=frank.get("pass_rank", np.nan),
               faced_sack_rank=frank.get("sack_rank", np.nan), faced_havoc_rank=frank.get("havoc_rank", np.nan))
    out["read"] = outlook_read(out)
    return out


def outlook_read(o):
    bits = []
    rr, pr, sr, hr = o["opp_rush_rank"], o["opp_pass_rank"], o["opp_sack_rank"], o["opp_havoc_rank"]
    if not pd.isna(rr):
        if rr >= SOFT_RANK:
            bits.append(f"soft run D (#{rr:.0f}): lean on the ground" + (" - he has been running well" if o["idx_rush"] >= 1.05 else ""))
        elif rr <= TOUGH_RANK:
            bits.append(f"tough run D (#{rr:.0f}): rushing should be harder than usual")
    if not pd.isna(pr):
        if pr >= SOFT_RANK:
            bits.append(f"soft pass D (#{pr:.0f}): passing game in a good spot" + (" (has produced vs its schedule)" if o["idx_pass"] >= 1.05 else ""))
        elif pr <= TOUGH_RANK:
            bits.append(f"tough pass D (#{pr:.0f})")
    disruptive = (not pd.isna(sr) and sr <= TOUGH_RANK) or (not pd.isna(hr) and hr <= TOUGH_RANK)
    if disruptive:
        bits.append(f"disruptive front (sack #{sr:.0f}, havoc #{hr:.0f})" + ("; protection already leaking" if o["sack_over"] > 0.5 else ""))
    for k, lab in (("faced_rush_rank", "run"), ("faced_pass_rank", "pass")):
        v = o[k]
        if not pd.isna(v) and v >= 19:
            bits.append(f"has faced soft {lab} Ds so far (avg #{v:.0f}): numbers may be padded")
        elif not pd.isna(v) and v <= 11:
            bits.append(f"has faced tough {lab} Ds so far (avg #{v:.0f}): numbers are earned")
    if o["n_games"] < 3:
        bits.append(f"only {o['n_games']} game(s) of data")
    return "; ".join(bits) if bits else "no strong edge either way"


def build_outlook(sched, season, matrix_path, log_path):
    m, g = load_outlook_inputs(matrix_path, log_path)
    if m is None:
        print("[workhorse] nfl_team_matrix.csv / nfl_game_log.csv not found: matchup outlook skipped")
        return {}
    ranks = d_ranks(m)
    cur = sched[(sched.season == season) & (~sched.played)]
    res = {}
    if cur.empty:
        return res
    nxt = cur.sort_values(["week", "gameday"]).groupby("team", as_index=False).first()
    for r in nxt.itertuples():
        o = outlook_one(r.team, r.opp, g, ranks)
        if o is not None:
            o["week"] = int(r.week)
            res[r.team] = o
    return res

# ================================================================== PDF
def build_pdf(df, sched, season, path, week_label, as_of_week, outlook=None):
    outlook = outlook or {}
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, KeepTogether

    INK, GREY, RED, AMBER, TEAL = "#111111", "#666666", "#B3261E", "#B26A00", "#0F8F83"
    TEALFILL = "#6FD9CE"
    colmap = {"red": RED, "amber": AMBER, "teal": TEAL}
    st = lambda n, **k: ParagraphStyle(n, fontName=k.pop("f", "Helvetica"), **k)
    H1 = st("h1", fontSize=20, leading=24, textColor=colors.HexColor(INK), f="Helvetica-Bold")
    SUB = st("sub", fontSize=8.5, leading=11.5, textColor=colors.HexColor(GREY))
    GH = st("gh", fontSize=11, leading=14, textColor=colors.white, f="Helvetica-Bold")
    TH = st("th", fontSize=6.8, leading=8.2, textColor=colors.HexColor(GREY), f="Helvetica-Bold")
    C = st("c", fontSize=7, leading=8.6)
    CB = st("cb", fontSize=7.3, leading=8.8, f="Helvetica-Bold")
    NOTE = st("note", fontSize=7.4, leading=10, textColor=colors.HexColor("#333333"))

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def usage(r):
        if r.pos == "RB":
            a = f"{fmt(r.szn_touches)} touches/gm<br/>last3 <b>{fmt(r.l3_touches)}</b>"
            b = f"<br/>snaps {fmt(r.l3_snap, '{:.0f}%')} &middot; {fmt(r.opp_share_l3, '{:.0f}%')} of team opps"
        elif r.pos in ("WR", "TE"):
            a = f"{fmt(r.szn_targets)} tgts/gm<br/>last3 <b>{fmt(r.l3_targets)}</b>"
            b = f"<br/>tgt share {fmt(r.szn_tshare, '{:.0f}%')} / <b>{fmt(r.l3_tshare, '{:.0f}%')}</b>"
        else:
            a = f"{fmt(r.szn_attempts)} att/gm<br/>last3 <b>{fmt(r.l3_attempts)}</b>"
            b = ""
        return Paragraph(a + b, C)

    def eff(r):
        if r.pos == "RB":
            if not pd.isna(r.l3_ryoe):
                a = f"RYOE/att {fmts(r.base_ryoe)} base<br/>last3 <b>{fmts(r.l3_ryoe)}</b>"
            else:
                a = f"YPC {fmt(r.base_ypc)} base<br/>last3 <b>{fmt(r.l3_ypc)}</b>"
            b = f"<br/>YPC {fmt(r.l3_ypc)} &middot; TLOS {fmt(r.l3_tlos, '{:.2f}')}"
        elif r.pos in ("WR", "TE"):
            if not pd.isna(r.l3_sep):
                a = f"sep {fmt(r.base_sep)} base<br/>last3 <b>{fmt(r.l3_sep)}</b>"
            else:
                a = f"Y/tgt {fmt(r.base_ypt)} base<br/>last3 <b>{fmt(r.l3_ypt)}</b>"
            b = f"<br/>YAC+ {fmts(r.l3_yacoe)}"
        else:
            a = f"EPA/db {fmts(r.base_qbepa, '{:+.2f}')} base<br/>last3 <b>{fmts(r.l3_qbepa, '{:+.2f}')}</b>"
            b = f"<br/>Y/A {fmt(r.l3_ypa)}"
        return Paragraph(a + b, C)

    def poe(r):
        v = r.l3_poe
        if pd.isna(v):
            return Paragraph("--", C)
        colr = AMBER if v >= HOT_POE else (TEAL if v <= COLD_POE else INK)
        return Paragraph(f"<font color='{colr}'><b>{v:+.1f}</b></font>", C)

    def wo(r):
        if r.n_without >= DEP_MIN_WITHOUT and not pd.isna(r.dep_delta):
            colr = RED if r.dep_delta <= -DEP_DELTA else INK
            return Paragraph(f"<font color='{colr}'><b>{r.dep_delta:+.1f}</b></font> pts/gm<br/>"
                             f"{fmt(r.without_ppg)} vs {fmt(r.with_ppg)}<br/>n={r.n_without} games", C)
        return Paragraph(f"n/a<br/>(missed {r.n_without})", C)

    def flagcell(r):
        if not r.flaglist:
            return Paragraph("<font color='#999999'>steady</font>", C)
        parts = [f"<font color='{colmap[l]}'><b>{esc(c)}</b></font> {esc(t)}" for c, l, t in r.flaglist]
        return Paragraph("<br/>".join(parts), C)

    def status(r):
        if r.inj_status and r.inj_status not in ("", "nan"):
            return Paragraph(f"<font color='{RED}'><b>{esc(r.inj_status)}</b></font><br/>{esc(r.inj_detail)}", C)
        return Paragraph("<font color='#999999'>--</font>", C)


    def rank_cell(v, soft_good=True):
        if pd.isna(v):
            return "--", None
        v = float(v)
        bg = None
        if v >= SOFT_RANK:
            bg = colors.HexColor(TEALFILL)
        elif v <= TOUGH_RANK:
            bg = colors.HexColor("#F6C9C4")
        return f"#{v:.0f}", bg

    def outlook_table(teams):
        rows = [o for o in (outlook.get(t) for t in teams) if o]
        if not rows:
            return None
        hd = ["OFFENSE", "FACED SO FAR<br/>avg D rank R / P / S / H", "NEXT D rank<br/>R / P / S / H", "PROJ RUSH<br/>yds", "PROJ PASS<br/>yds",
              "PROJ SACK<br/>% taken", "PROJ<br/>PTS*", "HOW IT MAY GO"]
        data = [[Paragraph(h, TH) for h in hd]]
        st_ = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F1F1")), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
               ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#E3E3E3")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
               ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
               ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
        for i, o in enumerate(rows, start=1):
            rk = [rank_cell(o[k]) for k in ("opp_rush_rank", "opp_pass_rank", "opp_sack_rank", "opp_havoc_rank")]
            fr = "  /  ".join((f"{o[k]:.0f}" if not pd.isna(o[k]) else "--") for k in ("faced_rush_rank", "faced_pass_rank", "faced_sack_rank", "faced_havoc_rank"))
            nxt = "  /  ".join(r[0] for r in rk)
            data.append([Paragraph(f"<b>{o['team']}</b><br/>vs {o['opp']}", C), Paragraph(fr, C), Paragraph(f"<b>{nxt}</b>", C),
                         Paragraph(f"<b>{o['proj_rush']:.0f}</b><br/>idx {o['idx_rush']:.2f}", C),
                         Paragraph(f"<b>{o['proj_pass']:.0f}</b><br/>idx {o['idx_pass']:.2f}", C),
                         Paragraph(f"<b>{o['proj_sack']:.1f}%</b>", C),
                         Paragraph(f"<b>{o['proj_pts']:.1f}</b><br/>idx {o['idx_pts']:.2f}", C),
                         Paragraph(esc(o["read"]), C)])
            # tint the NEXT D cell by softest/toughest single rank
            vals = [o[k] for k in ("opp_rush_rank", "opp_pass_rank", "opp_sack_rank", "opp_havoc_rank") if not pd.isna(o[k])]
            if vals and max(vals[:2] or [0]) >= SOFT_RANK:
                st_.append(("BACKGROUND", (2, i), (2, i), colors.HexColor("#BFEFEA")))
            elif vals and min(vals[:2] or [99]) <= TOUGH_RANK:
                st_.append(("BACKGROUND", (2, i), (2, i), colors.HexColor("#F6D5D1")))
        t = Table(data, colWidths=[w * inch for w in [0.7, 1.15, 0.95, 0.65, 0.65, 0.65, 0.6, 2.35]])
        t.setStyle(TableStyle(st_))
        return t

    widths = [0.32, 1.0, 0.7, 1.2, 1.25, 0.5, 0.88, 1.85]
    heads = ["POS", "PLAYER", "STATUS", "USAGE (season / last 3)", "EFFICIENCY (base / last 3)", "VS EXP", "TEAM W/O HIM", "FLAGS"]

    def team_table(team, sub):
        order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
        sub = sub.assign(_o=sub.pos.map(order)).sort_values(["_o", "szn_targets"], ascending=[True, False])
        data = [[Paragraph(h, TH) for h in heads]]
        styles = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F1F1")), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#BBBBBB")),
                  ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                  ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                  ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#E3E3E3")),
                  ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB"))]
        for i, r in enumerate(sub.itertuples(), start=1):
            data.append([Paragraph(r.pos, CB), Paragraph(esc(r.player), CB), status(r), usage(r), eff(r), poe(r), wo(r), flagcell(r)])
            lv = [l for _, l, _ in r.flaglist]
            if r.inj_status and r.inj_status not in ("", "nan"):
                styles.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FDECEA")))
            elif "red" in lv:
                styles.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FFF4F2")))
            elif "teal" in lv:
                styles.append(("BACKGROUND", (7, i), (7, i), colors.HexColor("#D3F3F0")))
        t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
        t.setStyle(TableStyle(styles))
        return t

    story = [Paragraph("NFL WORKHORSE TRACKER", H1),
             Paragraph(f"{esc(week_label)} &middot; data through Week {as_of_week} of {season} &middot; lead back, top 3 pass-catchers and QB for every team. "
                       f"Descriptive, not a pick: it shows who is being leaned on, who is slipping and what the team does without him.", SUB),
             Spacer(1, 5),
             Paragraph("<b>How to read it.</b> <font color='%s'><b>INJURY</b></font> = on the injury report (with what the team scores without him and who absorbs his work). "
                       "<font color='%s'><b>DECLINE</b></font> = his efficiency over the last 3 games fell against his own baseline (this + last season): the data trace of fatigue or a nagging injury. "
                       "<font color='%s'><b>HEAVY / SPIKE / HIGH PACE</b></font> = workload. "
                       "<font color='%s'><b>HOT</b></font> = scoring above what his opportunities should produce (cools off); <font color='%s'><b>COLD</b></font> = below (due to pop). "
                       "<font color='%s'><b>DEPENDENT</b></font> = team scores %.0f+ fewer points a game without him. VS EXP = last-3 fantasy points per game above/below expected from his carries, targets and attempts." %
                       (RED, RED, AMBER, AMBER, TEAL, RED, DEP_DELTA), NOTE),
             Paragraph("*PROJ columns: the team's production against each defense it has faced, measured against what that defense gives up to everyone else "
                       "(the opponent's other games, so a game never grades itself), blended toward league average with %.0f games of padding, then applied to the "
                       "next opponent's season averages. It is a stat read of the matchup, not the model's pick." % K_SHRINK, NOTE),
             Spacer(1, 7)]

    # alerts
    def sev(r):
        s = 0
        if r.inj_status and r.inj_status not in ("", "nan"): s += 100
        for c, l, _ in r.flaglist:
            s += {"red": 10, "amber": 4, "teal": 3}[l]
        return s
    alerts = df[df.apply(lambda r: len(r["flaglist"]) > 0, axis=1)].copy()
    alerts["_s"] = alerts.apply(sev, axis=1)
    alerts = alerts.sort_values("_s", ascending=False).head(24)
    if len(alerts):
        story.append(Table([[Paragraph("WATCH LIST - highest-priority flags league-wide", GH)]], colWidths=[7.7 * inch],
                           style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#222222")), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)])))
        rows = []
        for r in alerts.itertuples():
            fl = "<br/>".join(f"<font color='{colmap[l]}'><b>{esc(c)}</b></font> {esc(t)}" for c, l, t in r.flaglist)
            rows.append([Paragraph(f"<b>{esc(r.player)}</b><br/>{r.team} {r.pos}", C), Paragraph(fl, C)])
        t = Table(rows, colWidths=[1.6 * inch, 6.1 * inch])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
                               ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB"))]))
        story.append(t)
        story.append(PageBreak())

    # game sections for the upcoming (unplayed) week, else by team
    cur = sched[(sched.season == season)]
    unplayed = cur[~cur.played]
    teams_done = set()
    games = []
    if not unplayed.empty:
        wk_next = int(unplayed.week.min())
        gm = unplayed[(unplayed.week == wk_next) & (unplayed.home == False)].copy()
        gm["_k"] = gm["gameday"].astype(str) + gm["gametime"].astype(str)
        for g in gm.sort_values("_k").itertuples():
            games.append((g.team, g.opp, f"{g.team} @ {g.opp}", str(g.gameday)))
            teams_done.update([g.team, g.opp])
    rest = [t for t in sorted(df.team.unique()) if t not in teams_done]
    n = 0
    for away, home, title, when in games:
        block = [Table([[Paragraph(f"{esc(title)} &nbsp;&nbsp;<font size='8'>{esc(when)}</font>", GH)]], colWidths=[7.7 * inch],
                       style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#222222")), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))]
        ot = outlook_table([away, home])
        if ot is not None:
            block += [Paragraph("<b>MATCHUP OUTLOOK</b> &nbsp;<font size='6.5' color='#666666'>ranks: 1 = best defense of 32 &middot; R=rush, P=pass, S=sack rate, H=havoc &middot; "
                                "teal = soft D (22+), red = tough D (1-8)</font>", NOTE), ot, Spacer(1, 3)]
        for tm in (away, home):
            sub = df[df.team == tm]
            if len(sub):
                block += [Paragraph(f"<b>{tm}</b>", NOTE), team_table(tm, sub), Spacer(1, 3)]
        story.append(KeepTogether(block + [Spacer(1, 6)]))
        n += 1
    if rest:
        if n:
            story.append(PageBreak())
        story.append(Paragraph("Teams not on the upcoming slate (bye / already played)", NOTE))
        for tm in rest:
            sub = df[df.team == tm]
            if len(sub):
                story.append(KeepTogether([Paragraph(f"<b>{tm}</b>", NOTE), team_table(tm, sub), Spacer(1, 5)]))

    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.4 * inch, rightMargin=0.4 * inch, topMargin=0.4 * inch, bottomMargin=0.4 * inch,
                            title="NFL Workhorse Tracker")
    doc.build(story)


# ================================================================== main
def current_week_label(sched, season):
    cur = sched[sched.season == season]
    pl = cur[cur.played]
    as_of = int(pl.week.max()) if not pl.empty else 0
    un = cur[~cur.played]
    nxt = int(un.week.min()) if not un.empty else as_of
    return f"Week {nxt}", f"week{nxt}", as_of


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--no-pull", action="store_true", help="use the cached nflverse tables, do not download")
    ap.add_argument("--injuries", default=str(HERE / "nfl_injury_report.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    now = datetime.date.today()
    season = a.season or (now.year if now.month >= 3 else now.year - 1)
    raw = load_raw(season, pull=not a.no_pull)
    wk, sched = build_tables(raw)
    inj = load_injuries(a.injuries)
    df, wk = run(wk, sched, season, inj)
    label, tag, as_of = current_week_label(sched, season)
    GEN_DIR.mkdir(exist_ok=True)
    outlook = build_outlook(sched, season, HERE / "nfl_team_matrix.csv", HERE / "nfl_game_log.csv")
    if outlook:
        pd.DataFrame(list(outlook.values())).to_csv(HERE / "nfl_matchup_outlook.csv", index=False)
    pdf = GEN_DIR / f"NFL_Workhorse_Tracker_{tag}.pdf"
    build_pdf(df, sched, season, pdf, label, as_of, outlook)
    out = df.copy()
    out["flaglist"] = out["flaglist"].map(lambda fl: " | ".join(f"{c}: {t}" for c, _, t in fl))
    out.to_csv(HERE / "nfl_workhorse_tracker.csv", index=False)
    nflag = int((df["flaglist"].map(len) > 0).sum())
    print(f"[workhorse] {len(df)} players tracked, {nflag} with flags -> {pdf.name}, nfl_workhorse_tracker.csv, nfl_matchup_outlook.csv ({len(outlook)} teams)")
    return 0


# ================================================================== self-test on made-up players
def _synthetic():
    rng = np.random.default_rng(7)
    rows, sched = [], []
    teams = {"AAA": "BBB", "CCC": "DDD"}
    # (season, week) lists: 2025 weeks 1-12, 2026 weeks 1-4 played, week 5 unplayed
    seas = [(2025, w) for w in range(1, 13)] + [(2026, w) for w in range(1, 5)]
    pts = {}
    for (s, w) in seas:
        for t, o in teams.items():
            miss_star = (s == 2025 and w in (5, 6, 7))      # AAA's star RB missed these
            p_t = 14 if (t == "AAA" and miss_star) else (28 if t == "AAA" else 21)
            p_o = 20
            sched.append(dict(season=s, week=w, game_type="REG", away_team=t, home_team=o, away_score=p_t + rng.integers(-1, 2),
                              home_score=p_o, gameday=f"{s}-10-0{min(w,9)}", gametime="13:00"))
    sched.append(dict(season=2026, week=5, game_type="REG", away_team="AAA", home_team="BBB", away_score=np.nan, home_score=np.nan,
                      gameday="2026-10-11", gametime="13:00"))
    sched.append(dict(season=2026, week=5, game_type="REG", away_team="CCC", home_team="DDD", away_score=np.nan, home_score=np.nan,
                      gameday="2026-10-11", gametime="16:00"))

    def add(pid, name, team, pos, s, w, **k):
        base = dict(player_id=pid, player_display_name=name, team=team, position=pos, season=s, week=w, season_type="REG",
                    carries=0, rushing_yards=0, rushing_tds=0, rushing_epa=0.0, targets=0, receptions=0, receiving_yards=0,
                    receiving_tds=0, receiving_epa=0.0, attempts=0, passing_yards=0, passing_tds=0, passing_epa=0.0,
                    target_share=0.00, air_yards_share=0.0)
        base.update(k)
        rows.append(base)
    ngs_r, ngs_c, snaps = [], [], []
    for (s, w) in seas:
        l3 = (s == 2026 and w in (2, 3, 4))
        star_out = (s == 2025 and w in (5, 6, 7))
        # AAA star RB: baseline 19 touches, 5.0 ypc, RYOE +0.9 ; last 3: 27 touches, 3.9 ypc, RYOE -0.2
        if not star_out:
            car, ypc, ryoe = (24, 3.9, -0.2) if l3 else (16, 5.0, 0.9)
            add("RB_STAR", "Star Runner", "AAA", "RB", s, w, carries=car, rushing_yards=int(car * ypc), rushing_epa=car * (0.0 if l3 else 0.08),
                targets=4, receptions=3, receiving_yards=20, target_share=0.15)
            ngs_r.append(dict(player_display_name="Star Runner", season=s, week=w, season_type="REG", rush_yards_over_expected_per_att=ryoe,
                              avg_time_to_los=2.7, rush_attempts=car))
            snaps.append(dict(player="Star Runner", season=s, week=w, game_type="REG", offense_pct=0.82 if l3 else 0.65))
        else:
            add("RB_BACK", "Backup Back", "AAA", "RB", s, w, carries=15, rushing_yards=60, targets=2, receptions=2, receiving_yards=12, target_share=0.08)
        add("RB_BACK", "Backup Back", "AAA", "RB", s, w, carries=5, rushing_yards=20, targets=1, receptions=1, receiving_yards=5, target_share=0.04) if not star_out else None
        # AAA WR1: separation drops 3.1 -> 2.4 over last 3 ; steady targets
        add("WR_SEP", "Sep Drop", "AAA", "WR", s, w, targets=8, receptions=5, receiving_yards=70, receiving_tds=0, target_share=0.26)
        ngs_c.append(dict(player_display_name="Sep Drop", season=s, week=w, season_type="REG", avg_separation=2.4 if l3 else 3.1,
                          avg_yac_above_expectation=-0.5 if l3 else 0.6))
        # AAA QB steady
        add("QB_A", "Steady QB", "AAA", "QB", s, w, attempts=34, passing_yards=250, passing_tds=1, passing_epa=34 * 0.10)
        # CCC: stable RB, stable WR, never missing
        add("RB_C", "Stable Back", "CCC", "RB", s, w, carries=17, rushing_yards=75, targets=3, receptions=2, receiving_yards=14, target_share=0.10)
        add("WR_C", "Stable Wideout", "CCC", "WR", s, w, targets=7, receptions=5, receiving_yards=60, target_share=0.24)
        add("QB_C", "Stable QB", "CCC", "QB", s, w, attempts=32, passing_yards=240, passing_tds=1, passing_epa=32 * 0.05)
        add("TE_C", "Hot TE", "CCC", "TE", s, w, targets=5, receptions=3, receiving_yards=30, receiving_tds=2 if l3 else 0, target_share=0.14)
    raw = dict(weekly=pd.DataFrame(rows), snaps=pd.DataFrame(snaps), ngs_rush=pd.DataFrame(ngs_r), ngs_rec=pd.DataFrame(ngs_c),
               sched=pd.DataFrame(sched))
    return raw


def selftest():
    raw = _synthetic()
    wk, sched = build_tables(raw)
    inj = {(norm_name("Star Runner"), "AAA"): ("Questionable", "Hamstring")}
    df, wk2 = run(wk, sched, 2026, inj)
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{info}]" if info else ""))
        ok &= bool(cond)

    r = df[df.player == "Star Runner"].iloc[0]
    codes = {c for c, _, _ in r["flaglist"]}
    check("tracked RB1 is the star runner (not backup)", len(df[(df.team == "AAA") & (df.pos == "RB")]) == 1)
    check("last-3 touches = 27 (24 carries + 3 rec)", abs(r["l3_touches"] - 27) < 1e-9, r["l3_touches"])
    check("HEAVY or HEAVY+SPIKE workload flag", bool(codes & {"HEAVY", "HEAVY+SPIKE"}), sorted(codes))
    check("DECLINE flag from RYOE/YPC drop", "DECLINE" in codes, sorted(codes))
    check("INJURY flag present and first", r["flaglist"][0][0] == "INJURY")
    check("games he missed = 3 in 2025 -> n_without 3", r["n_without"] == 3, r["n_without"])
    check("team scores ~14 without him vs ~28 with: delta ~ -14", -16 < r["dep_delta"] < -12, round(r["dep_delta"], 2))
    check("DEPENDENT flag", "DEPENDENT" in codes)
    check("next man up is Backup Back", r["next_man"] == "Backup Back", r["next_man"])
    w = df[df.player == "Sep Drop"].iloc[0]
    wc = {c for c, _, _ in w["flaglist"]}
    check("WR separation drop flagged DECLINE", "DECLINE" in wc, sorted(wc))
    check("WR base sep 3.1 and last-3 2.4", abs(w["base_sep"] - 3.1) < 1e-9 and abs(w["l3_sep"] - 2.4) < 1e-9, (w["base_sep"], w["l3_sep"]))
    s = df[df.player == "Stable Back"].iloc[0]
    check("stable player has no flags", len(s["flaglist"]) == 0, s["flaglist"])
    q = df[df.player == "Steady QB"].iloc[0]
    check("steady QB no DECLINE", "DECLINE" not in {c for c, _, _ in q["flaglist"]})
    t = df[df.player == "Hot TE"].iloc[0]
    tc = {c for c, _, _ in t["flaglist"]}
    check("TE with 2 TDs/gm over last 3 flagged HOT (above expected)", "HOT" in tc, (sorted(tc), round(t["l3_poe"], 1)))
    check("opportunity share of team computed", 0 < r["opp_share_l3"] < 100, r["opp_share_l3"])
    # expected points sanity: league-average player has ~0 total over/under
    check("expected points balance to ~0 league-wide", abs(wk2["poe"].sum()) < 1e-6 * len(wk2) + 1.0, round(wk2["poe"].sum(), 3))
    # leak check: a player's baseline excludes his own last-3 games
    check("baseline excludes last-3 games (base touches = 19 for star)", abs(r["base_touches"] - (16 + 3)) < 0.01, r["base_touches"])

    # ---- next-game matchup outlook on a made-up 32-team table + 2-week game log
    mt = [dict(team=t, def_rush_epa_allowed=0.0 + i * 0.001, def_yards_per_pass=6.5 + i * 0.01, sack_pct_defense=6.0 - i * 0.05,
               havoc_rate_defense=12.0 - i * 0.1) for i, t in enumerate([f"X{n:02d}" for n in range(28)])]
    mt += [dict(team="AAA", def_rush_epa_allowed=-0.30, def_yards_per_pass=5.0, sack_pct_defense=9.0, havoc_rate_defense=16.0),   # best D
           dict(team="BBB", def_rush_epa_allowed=0.40, def_yards_per_pass=9.0, sack_pct_defense=1.0, havoc_rate_defense=6.0),     # worst D
           dict(team="CCC", def_rush_epa_allowed=0.01, def_yards_per_pass=6.6, sack_pct_defense=5.0, havoc_rate_defense=11.0),
           dict(team="DDD", def_rush_epa_allowed=0.02, def_yards_per_pass=6.7, sack_pct_defense=5.0, havoc_rate_defense=11.0)]
    mat = pd.DataFrame(mt)

    def gl(team, week, opp, rush, rush_allowed):
        return dict(team=team, week=week, opponent=opp, pts_for=20, pts_against=20, rush_yards=rush, pass_yards=200,
                    rush_yards_allowed=rush_allowed, pass_yards_allowed=200, sack_pct_offense_game=5.0, sack_pct_defense_game=5.0)
    glog = pd.DataFrame([gl("AAA", 1, "CCC", 150, 100), gl("CCC", 1, "AAA", 100, 150),
                         gl("BBB", 1, "DDD", 100, 140), gl("DDD", 1, "BBB", 140, 100),
                         gl("AAA", 2, "DDD", 150, 100), gl("DDD", 2, "AAA", 100, 150),
                         gl("BBB", 2, "CCC", 100, 140), gl("CCC", 2, "BBB", 140, 100)])
    rk = d_ranks(mat)
    check("defense ranks: BBB is worst (32) in rush/pass/sack/havoc, AAA best (1)",
          (rk.loc["BBB"] == 32).all() and (rk.loc["AAA"] == 1).all(), rk.loc[["AAA", "BBB"]].values.tolist())
    o = outlook_one("AAA", "BBB", glog, rk)
    check("AAA ran 150 vs defenses that otherwise allow 100: raw factor 1.5, shrunk with 3 pseudo-games over n=2 -> 1.20",
          abs(o["idx_rush"] - 1.20) < 1e-9, o["idx_rush"])
    check("projected rush vs BBB (allows 140): 140 x 1.20 = 168", abs(o["proj_rush"] - 168.0) < 1e-9, o["proj_rush"])
    check("pass and points factors neutral -> 200 yds, 20 pts", abs(o["proj_pass"] - 200) < 1e-9 and abs(o["proj_pts"] - 20) < 1e-9,
          (o["proj_pass"], o["proj_pts"]))
    check("read calls BBB a soft run D and notes he has been running well", "soft run D (#32)" in o["read"] and "running well" in o["read"], o["read"])
    check("read flags the 2-game sample", "only 2 game" in o["read"], o["read"])
    ob = outlook_one("BBB", "AAA", glog, rk)
    check("BBB vs AAA (best D, rank 1): tough run + disruptive front flagged", "tough run D (#1)" in ob["read"] and "disruptive front" in ob["read"], ob["read"])
    outlook_demo = {"AAA": {**o, "week": 5}, "BBB": {**ob, "week": 5}}
    # PDF renders
    import tempfile
    d = Path(tempfile.mkdtemp())
    label, tag, as_of = current_week_label(sched, 2026)
    build_pdf(df, sched, 2026, d / "t.pdf", label, as_of, outlook_demo)
    check("PDF built and non-trivial", (d / "t.pdf").stat().st_size > 5000, (d / "t.pdf").stat().st_size)
    check("label points to upcoming week 5, as-of week 4", (label, as_of) == ("Week 5", 4), (label, as_of))
    print("\nALL SELF-TESTS PASSED" if ok else "\nSELF-TEST FAILURES")
    print("pdf:", d / "t.pdf")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
