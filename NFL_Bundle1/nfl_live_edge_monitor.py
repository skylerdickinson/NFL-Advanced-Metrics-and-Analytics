# ==============================================================================
# nfl_live_edge_monitor.py
# ==============================================================================
# Live, IN-GAME trench-battle alert tool: runs in the background pulling live
# data for all the games and pops up badges when there is a definitive edge
# (O-line getting pushed back, TFL and stuff rate -- one team taking control),
# before it shows up on the scoreboard.
#
# THIS IS A DIFFERENT KIND OF TOOL than every other script in this folder:
# those all pull nflverse data via nflreadpy AFTER games are final, once a
# week. nflreadpy/nflverse has NO live/in-progress mode at all (checked --
# it's a batch downloader around nflverse's post-game data releases, not a
# real-time feed). So this script uses a different, LIVE source instead:
# ESPN's public, unauthenticated scoreboard + game-summary API -- the same
# one your OWN older personal sheet used to pull from before
# nfl_matchup_sheet.py switched to nflverse (see README.txt "LICENSING --
# WHAT WAS REMOVED AND WHY"). That switch was specifically about the SOLD
# PDF -- reselling ESPN data isn't allowed under their terms. This tool
# never resells anything, it just reads your own screen -- same personal-
# use case your old sheet always had.
#
# HONEST CAVEAT, READ THIS FIRST: ESPN's live API is unofficial and
# undocumented -- no official refresh-rate guarantee, no official schema
# guarantee, can change without notice. The parsing below follows the JSON
# shape the ESPN hobbyist/fantasy-tool community has documented and used for
# years, but it has NOT been tested against an actual live in-progress game.
# So: run this during a live game, watch the console output, and if the
# TFL/STUFF signals look wrong or empty while SCORE/YARDS look fine, check
# the one-time raw-shape warning it prints the first time something looks
# off and adjust the field mapping. The box-score-level signals (yards/
# carry, sacks, completion %) are the more battle-tested ESPN fields and
# should "just work"; the play-by-play TFL/stuff-rate parsing is the part
# most likely to need one real fix-up round.
#
# WHAT IT DOES, every POLL_SECONDS (default 45s) while any real NFL game
# is live:
#   1. Pulls every live game's real box-score team totals (rush yards/
#      attempt, pass yards/attempt, sacks, turnovers) AND, best-effort,
#      real per-play detail (to compute TFL rate / stuffed-run rate --
#      "o-line getting pushed back" -- which isn't in the plain box score).
#   2. Compares each team's TONIGHT rate against that SAME team's real
#      SEASON-TO-DATE rate already sitting in nfl_team_matrix.csv (the
#      same file nfl_data_pull.py builds every week for the rest of this
#      project -- no new/separate baseline data needed).
#   3. Only calls it a real edge once there's enough live sample AND a real
#      gap (see MIN_* / GAP_* constants below) -- a team getting stuffed on
#      2 carries by halftime isn't a signal, it's noise.
#   4. Fires a native Mac notification (osascript, no extra install) the
#      FIRST time a real edge crosses the bar for a given team/category/
#      game -- never repeats the same one every 45 seconds; only fires
#      again if that same signal escalates to the next, harder bar.
#
# USAGE:
#   python3 nfl_live_edge_monitor.py              # run it, Ctrl+C to stop
#   python3 nfl_live_edge_monitor.py --quiet       # console only, no real
#                                                   # notifications (for
#                                                   # testing/watching it
#                                                   # work before trusting
#                                                   # it to interrupt you)
#   python3 nfl_live_edge_monitor.py --poll-seconds 30
#
# Run nfl_data_pull.py at least once this week first (same as every other
# script here) -- this tool reads nfl_team_matrix.csv for real season
# baselines and won't have anything to compare against without it.
# ==============================================================================

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

try:
    from curl_cffi import requests as _cffi_requests  # optional -- see fetch_json()
except ImportError:
    _cffi_requests = None

OUT_DIR = Path(__file__).resolve().parent
TEAM_MATRIX_PATH = OUT_DIR / "nfl_team_matrix.csv"
ALERT_STATE_PATH = OUT_DIR / "nfl_live_alert_state.json"

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"
# UPDATED 2026-09-27 (round 2): the real error body from the first run's
# 403 turned out to be an Akamai edge-block page ("Access Denied ...
# Reference #..."), not a plain "missing User-Agent" rejection -- so the
# real desktop-Chrome UA string alone (round 1's fix) can't get past it.
# Akamai's bot filtering here works off the TLS/HTTP handshake fingerprint
# itself, which plain urllib/requests can't fake no matter what headers
# you send (Python's SSL stack has a different signature than a real
# browser's at the protocol level, before any header is even read).
#
# REAL FIX: curl_cffi (pip install curl_cffi) wraps a real curl build that
# can impersonate an actual Chrome TLS handshake byte-for-byte -- the
# standard, well-established fix for exactly this class of Akamai/
# Cloudflare block, used throughout the scraping community for sites like
# this. fetch_json() below uses it when installed, and falls back to plain
# urllib (which you now know gets blocked here) with a clear message
# telling you to install it if it's missing -- this script can't reliably
# work against ESPN's live API without it.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/",
    "Origin": "https://www.espn.com",
}

# ------------------------------------------------------------------
# Edge thresholds -- real sample-size and real gap-size floors, so a
# couple of live plays can't trigger a "definitive edge." Tune these once
# you've watched a few real live games and seen how often each fires.
# ------------------------------------------------------------------
MIN_RUSH_ATTEMPTS_FOR_RUN_SIGNAL = 6
MIN_PASS_ATTEMPTS_FOR_PASS_SIGNAL = 8
STUFF_RATE_GAP_PTS = 20.0     # live stuff rate allowed this many+ pts above this team's real season rate
TFL_RATE_GAP_PTS = 15.0
SACK_RATE_GAP_PTS = 12.0
YPC_GAP = 2.2                 # live yards/carry this many+ yards above/below season average
YPA_GAP = 2.0                 # live yards/pass-attempt this many+ yards above/below season average

POLL_SECONDS_DEFAULT = 45
_warned_shapes = set()  # one-time-per-run warning tracker, see _warn_once()


def _warn_once(key, msg):
    if key not in _warned_shapes:
        _warned_shapes.add(key)
        print(f"  [!] {msg}")


_warned_no_curl_cffi = False


def _fetch_json_cffi(url, retries):
    for attempt in range(retries + 1):
        try:
            resp = _cffi_requests.get(url, headers=HEADERS, impersonate="chrome124", timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if attempt < retries:
                time.sleep(2)
                continue
            print(f"  [!] Failed to fetch {url}: HTTP {resp.status_code} -- body: {resp.text[:300]}")
            return None
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            print(f"  [!] Failed to fetch {url} (curl_cffi): {e}")
            return None


def _fetch_json_urllib(url, retries):
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if attempt < retries:
                time.sleep(2)
                continue
            # Real diagnostic detail on a real block -- prints the status
            # code and the first bit of ESPN's own response body (often a
            # real explanation, e.g. an Akamai/bot-block page) so a repeat
            # 403/429 is fast to diagnose without guessing blind again.
            body = ""
            try:
                body = e.read(300).decode("utf-8", errors="replace")
            except Exception:
                pass
            print(f"  [!] Failed to fetch {url}: HTTP {e.code} {e.reason}" + (f" -- body: {body}" if body else ""))
            return None
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            print(f"  [!] Failed to fetch {url}: {e}")
            return None


def fetch_json(url, retries=1):
    """Prefers curl_cffi (real Chrome TLS-handshake impersonation) since
    ESPN's live API real-403'd plain urllib even with a full browser
    User-Agent -- see the HEADERS comment above. Falls back to urllib only
    if curl_cffi isn't installed, with a one-time warning telling you to
    install it (this fallback WILL still hit the same Akamai block)."""
    global _warned_no_curl_cffi
    if _cffi_requests is not None:
        return _fetch_json_cffi(url, retries)
    if not _warned_no_curl_cffi:
        _warned_no_curl_cffi = True
        print("  [!] curl_cffi isn't installed -- falling back to plain urllib, which already real-403'd "
              "against ESPN's live API once (Akamai TLS-fingerprint block, not just a header check). "
              "Run: pip3 install curl_cffi --break-system-packages   (or without that flag if you're in a venv), "
              "then rerun this script.")
    return _fetch_json_urllib(url, retries)


def get_live_games():
    """Real live NFL games right now, via ESPN's public scoreboard. Returns
    a list of {event_id, away, home, period, clock, display}."""
    data = fetch_json(SCOREBOARD_URL)
    if not data:
        return []
    live = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {})
        if status.get("state") != "in":  # "pre" = not started, "post" = final
            continue
        competitors = comp.get("competitors", [])
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        if not away or not home:
            continue
        live.append({
            "event_id": ev.get("id"),
            "away": away["team"]["abbreviation"],
            "home": home["team"]["abbreviation"],
            "away_score": away.get("score"),
            "home_score": home.get("score"),
            "display": status.get("shortDetail", ""),
        })
    return live


def _parse_boxscore_stats(boxscore, team_abbr):
    """Real live team-total stats from ESPN's boxscore.teams[].statistics[]
    array (rushing yards/attempts, passing yards/attempts, sacks,
    turnovers) -- the well-documented, battle-tested part of ESPN's live
    feed. Returns a plain dict of whatever it can find; missing fields are
    just absent, never guessed."""
    out = {}
    for team_block in boxscore.get("teams", []):
        if team_block.get("team", {}).get("abbreviation") != team_abbr:
            continue
        stat_by_name = {s.get("name"): s.get("displayValue") for s in team_block.get("statistics", [])}
        try:
            if "rushingAttempts" in stat_by_name:
                out["rush_att"] = float(stat_by_name["rushingAttempts"])
            if "rushingYards" in stat_by_name:
                out["rush_yds"] = float(stat_by_name["rushingYards"])
            if "completionAttempts" in stat_by_name and "/" in stat_by_name["completionAttempts"]:
                comp, att = stat_by_name["completionAttempts"].split("/")
                out["pass_comp"], out["pass_att"] = float(comp), float(att)
            if "netPassingYards" in stat_by_name:
                out["pass_yds"] = float(stat_by_name["netPassingYards"])
            if "sacksYardsLost" in stat_by_name and "-" in stat_by_name["sacksYardsLost"]:
                sacks, _ = stat_by_name["sacksYardsLost"].split("-")
                out["sacks_allowed"] = float(sacks)
            if "turnovers" in stat_by_name:
                out["turnovers"] = float(stat_by_name["turnovers"])
        except (ValueError, KeyError):
            _warn_once(f"boxscore_parse_{team_abbr}", f"boxscore stat parsing hit an unexpected value for {team_abbr} -- skipping that field this poll.")
    return out


def _parse_play_level_stats(game_json, away_abbr, home_abbr):
    """Best-effort real per-play detail (TFL / stuffed-run counts, which
    aren't in the plain box score) from ESPN's drives structure. This is
    the part most likely to need a fix once tested against a real live
    game -- see the file header. Returns {team_abbr: {rush_att, stuffed,
    tfl, pass_att, sacks}}."""
    out = {away_abbr: {"rush_att": 0, "stuffed": 0, "tfl": 0, "pass_att": 0, "sacks": 0},
           home_abbr: {"rush_att": 0, "stuffed": 0, "tfl": 0, "pass_att": 0, "sacks": 0}}
    drives = game_json.get("drives", {})
    all_plays = []
    for key in ("previous", "current"):
        block = drives.get(key)
        if isinstance(block, list):
            for d in block:
                all_plays.extend(d.get("plays", []))
        elif isinstance(block, dict):
            all_plays.extend(block.get("plays", []))

    if not all_plays:
        _warn_once("no_plays_found", "No plays found under game['drives'] this poll -- ESPN's live JSON shape "
                                      "may differ from what this script expects. TFL/stuff-rate signals will stay "
                                      "empty until this is fixed -- box-score signals (yards/carry, sacks) still work.")
        return out

    for play in all_plays:
        try:
            play_type = (play.get("type") or {}).get("text", "") or ""
            team_abbr = ((play.get("start") or {}).get("team") or {}).get("abbreviation") \
                or (play.get("team") or {}).get("abbreviation")
            yards = play.get("statYardage")
            text = (play.get("text") or "").lower()
            if team_abbr not in out:
                continue
            if "rush" in play_type.lower():
                out[team_abbr]["rush_att"] += 1
                if yards is not None and yards <= 0:
                    out[team_abbr]["stuffed"] += 1
                if (yards is not None and yards < 0) or "for a loss" in text or "tackled for a loss" in text:
                    out[team_abbr]["tfl"] += 1
            elif "sack" in play_type.lower() or "sacked" in text:
                out[team_abbr]["pass_att"] += 1  # a sack counts as a pass attempt for rate purposes
                out[team_abbr]["sacks"] += 1
            elif "pass" in play_type.lower():
                out[team_abbr]["pass_att"] += 1
        except Exception:
            _warn_once("play_parse_error", "A play object didn't match the expected shape -- skipping it. "
                                            "If this keeps happening, paste the console output back so the field mapping can be fixed.")
            continue
    return out


def get_game_signals(event_id, away_abbr, home_abbr):
    game_json = fetch_json(SUMMARY_URL.format(event_id=event_id))
    if not game_json:
        return None
    boxscore = game_json.get("boxscore", {})
    signals = {}
    for abbr in (away_abbr, home_abbr):
        signals[abbr] = _parse_boxscore_stats(boxscore, abbr)
    play_level = _parse_play_level_stats(game_json, away_abbr, home_abbr)
    for abbr in (away_abbr, home_abbr):
        signals[abbr].update({f"pbp_{k}": v for k, v in play_level.get(abbr, {}).items()})
    return signals


def load_season_baselines():
    """Real season-to-date rates from nfl_team_matrix.csv (already built
    weekly by nfl_data_pull.py) -- same source every other script here
    uses, so this tool's baselines never drift from the rest of the
    project's numbers."""
    if not TEAM_MATRIX_PATH.exists():
        print(f"  [!] {TEAM_MATRIX_PATH.name} not found -- run nfl_data_pull.py first so there's a real "
              f"season baseline to compare live stats against.")
        return None
    df = pd.read_csv(TEAM_MATRIX_PATH, index_col=0)
    return df


def _rate(numer, denom):
    return (numer / denom * 100.0) if denom else None


def detect_edges(event, signals, baselines):
    """Real live-vs-season comparisons -- only returns an edge once both a
    minimum live sample AND a minimum real gap are met (see the constants
    at the top of this file). Returns a list of (team_abbr, category,
    message, severity_key) tuples."""
    edges = []
    away, home = event["away"], event["home"]
    opp = {away: home, home: away}

    for team in (away, home):
        s = signals.get(team, {})
        if team not in baselines.index:
            continue
        base = baselines.loc[team]

        # RUSHING: yards/carry vs season off_rushing_yards_per_play, AND
        # (best-effort) stuffed-run / TFL rate vs season stuff_rate_defense
        # of the team ALLOWING (i.e. the opponent's real season stuff rate
        # forced) -- "o-line getting pushed back" = THIS team's live
        # stuff-rate-allowed is way above their own real season number.
        rush_att = s.get("rush_att") or s.get("pbp_rush_att") or 0
        if rush_att >= MIN_RUSH_ATTEMPTS_FOR_RUN_SIGNAL:
            if s.get("rush_yds") is not None:
                live_ypc = s["rush_yds"] / rush_att
                season_ypc = base.get("off_rushing_yards_per_play")
                if season_ypc is not None and pd.notna(season_ypc):
                    if live_ypc <= season_ypc - YPC_GAP:
                        edges.append((team, "RUN GAME STALLED", f"{team} rushing {live_ypc:.1f} yd/carry tonight "
                                       f"vs their real {season_ypc:.1f} season avg ({rush_att:.0f} carries so far) "
                                       f"-- O-line is losing the trench battle.", "run_stalled"))
                    elif live_ypc >= season_ypc + YPC_GAP:
                        edges.append((team, "RUN GAME ROLLING", f"{team} rushing {live_ypc:.1f} yd/carry tonight "
                                       f"vs their real {season_ypc:.1f} season avg ({rush_att:.0f} carries so far) "
                                       f"-- taking control on the ground.", "run_rolling"))

            pbp_rush = s.get("pbp_rush_att", 0)
            if pbp_rush >= MIN_RUSH_ATTEMPTS_FOR_RUN_SIGNAL:
                live_stuff = _rate(s.get("pbp_stuffed", 0), pbp_rush)
                live_tfl = _rate(s.get("pbp_tfl", 0), pbp_rush)
                season_stuff_allowed = base.get("stuff_rate_offense")  # this team's OWN real season stuff-rate-allowed (offense side)
                if live_stuff is not None and season_stuff_allowed is not None and pd.notna(season_stuff_allowed):
                    if live_stuff >= season_stuff_allowed + STUFF_RATE_GAP_PTS:
                        edges.append((team, "O-LINE GETTING PUSHED BACK",
                                       f"{team} stuffed on {live_stuff:.0f}% of carries tonight vs their real "
                                       f"{season_stuff_allowed:.0f}% season rate ({pbp_rush:.0f} carries) -- real edge for the defense.",
                                       "stuff_edge"))
                if live_tfl is not None and live_tfl >= TFL_RATE_GAP_PTS:
                    edges.append((team, "TFL RATE SPIKING",
                                   f"{team} taken for a loss on {live_tfl:.0f}% of carries tonight ({pbp_rush:.0f} carries) "
                                   f"-- front seven is winning.", "tfl_edge"))

        # PASSING: yards/attempt vs season, sack rate allowed vs season
        pass_att = s.get("pass_att") or s.get("pbp_pass_att") or 0
        if pass_att >= MIN_PASS_ATTEMPTS_FOR_PASS_SIGNAL:
            if s.get("pass_yds") is not None:
                live_ypa = s["pass_yds"] / pass_att
                season_ypa = base.get("off_yards_per_pass")
                if season_ypa is not None and pd.notna(season_ypa):
                    if live_ypa <= season_ypa - YPA_GAP:
                        edges.append((team, "PASS GAME STALLED", f"{team} averaging {live_ypa:.1f} yd/attempt tonight "
                                       f"vs their real {season_ypa:.1f} season avg ({pass_att:.0f} attempts) -- "
                                       f"coverage/pressure winning.", "pass_stalled"))
                    elif live_ypa >= season_ypa + YPA_GAP:
                        edges.append((team, "PASS GAME ROLLING", f"{team} averaging {live_ypa:.1f} yd/attempt tonight "
                                       f"vs their real {season_ypa:.1f} season avg ({pass_att:.0f} attempts) -- "
                                       f"taking control through the air.", "pass_rolling"))

            pbp_pass = s.get("pbp_pass_att", 0)
            if pbp_pass >= MIN_PASS_ATTEMPTS_FOR_PASS_SIGNAL:
                live_sack_rate = _rate(s.get("pbp_sacks", 0), pbp_pass)
                season_sack_rate = base.get("sack_pct_offense")
                if live_sack_rate is not None and season_sack_rate is not None and pd.notna(season_sack_rate):
                    if live_sack_rate >= season_sack_rate + SACK_RATE_GAP_PTS:
                        edges.append((team, "PASS PROTECTION BREAKING DOWN",
                                       f"{team} sacked on {live_sack_rate:.0f}% of drop-backs tonight vs their real "
                                       f"{season_sack_rate:.0f}% season rate ({pbp_pass:.0f} attempts) -- {opp[team]}'s "
                                       f"pass rush has real live pressure.", "sack_edge"))
    return edges


def send_notification(title, message, quiet=False, sticky=True):
    print(f"  >>> EDGE: [{title}] {message}")
    if quiet:
        return
    # UPDATED 2026-09-27 (round 3): plain `display notification` (a Banner)
    # always auto-dismisses after a few seconds -- that's a macOS
    # notification-style setting (System Settings -> Notifications -> <the
    # app osascript is running under> -> switch Banners to Alerts), NOT
    # something this AppleScript command itself controls, so a script can't
    # force it to stick around that way. Defaulting to a real `display
    # dialog` instead -- a modal popup that's GUARANTEED to stay up until
    # you click OK, regardless of any notification setting. Tradeoff: it
    # steals focus and, if several edges fire close together, you'll get
    # that many dialogs stacked up needing individual clicks -- pass
    # sticky=False (--banner on the CLI) to go back to the old
    # auto-dismissing style if that turns out to be more annoying than
    # useful on a busy Sunday. Fired via Popen (non-blocking) so the poll
    # loop keeps running and checking other live games while a dialog sits
    # there waiting on you.
    if sticky:
        script = (f'display dialog {json.dumps(message)} with title {json.dumps(title)} '
                  f'buttons {{"OK"}} default button "OK" with icon caution')
    else:
        script = f'display notification {json.dumps(message)} with title {json.dumps(title)} sound name "Glass"'
    try:
        subprocess.Popen(["osascript", "-e", script])
    except Exception as e:
        print(f"  [!] Couldn't fire a Mac notification ({e}) -- printed above instead.")


def load_alert_state():
    if ALERT_STATE_PATH.exists():
        try:
            with open(ALERT_STATE_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_alert_state(state):
    with open(ALERT_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def run(poll_seconds=POLL_SECONDS_DEFAULT, quiet=False, sticky=True):
    baselines = load_season_baselines()
    if baselines is None:
        return
    alert_state = load_alert_state()
    style = "sticky dialogs (click OK to dismiss)" if sticky else "auto-dismissing banners"
    print(f"Live NFL edge monitor started -- polling every {poll_seconds}s, alerts as {style}. Ctrl+C to stop.\n")

    try:
        while True:
            live_games = get_live_games()
            if not live_games:
                print("No real live games right now.")
            for event in live_games:
                key_prefix = event["event_id"]
                print(f"  {event['away']} {event['away_score']} @ {event['home']} {event['home_score']} "
                      f"({event['display']})")
                signals = get_game_signals(event["event_id"], event["away"], event["home"])
                if not signals:
                    continue
                edges = detect_edges(event, signals, baselines)
                for team, category, message, sev_key in edges:
                    alert_key = f"{key_prefix}:{team}:{sev_key}"
                    if alert_state.get(alert_key):
                        continue  # already fired this exact edge for this game -- don't spam
                    alert_state[alert_key] = True
                    send_notification(f"{team}: {category}", message, quiet=quiet, sticky=sticky)
            save_alert_state(alert_state)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live in-game NFL trench-battle edge monitor with Mac notifications.")
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS_DEFAULT,
                         help=f"How often to poll ESPN's live feed (default {POLL_SECONDS_DEFAULT}s -- don't go much lower, this is an unofficial API).")
    parser.add_argument("--quiet", action="store_true", help="Print edges to console only, skip real Mac notifications (for testing).")
    parser.add_argument("--banner", action="store_true",
                         help="Use the old auto-dismissing notification banners instead of sticky click-to-dismiss dialogs "
                              "(dialogs are the default now -- see the send_notification() comment for why banners can't "
                              "be made to stick around from inside this script).")
    parser.add_argument("--reset", action="store_true", help="Clear the alert-dedup state file before starting (see it re-fire everything fresh).")
    args = parser.parse_args()
    if args.reset and ALERT_STATE_PATH.exists():
        ALERT_STATE_PATH.unlink()
        print("Cleared alert state.")
    run(poll_seconds=args.poll_seconds, quiet=args.quiet, sticky=not args.banner)
