"""
nfl_run_all.py -- the ONE command for the whole NFL "grind it out" project:

    python3 nfl_run_all.py              # the normal weekly run
    python3 nfl_run_all.py --no-pull    # skip the data pull, just rebuild the PDFs from the data already here
    python3 nfl_run_all.py --live       # same weekly run, then start the live in-game edge monitor (runs until you stop it)
    python3 nfl_run_all.py --live-only  # just the live monitor
    python3 nfl_run_all.py --check      # only check everything is in place, run nothing
    python3 nfl_run_all.py --spread-only # only the Spread AI (picks against the Vegas spread -> generated/NFL_Spread_Picks_week<N>.pdf)

What it runs, in order:
  1. nfl_data_pull.py        fresh team/player/injury/game-log data (nflverse) -> the nfl_*.csv files in this folder
                             (then saves a dated copy of the team matrix + game log into history/ -- that's your
                              pre-game record for backtesting VOID / PPD / success rate later)
  2. nfl_matchup_sheet.py    builds every sheet from that data -> generated/:
                               nfl_full_slate_<week>.pdf, matchups_<week>/ (one PDF per game),
                               nfl_old_man_sheet_<week>.pdf, nfl_delta_scan_sheet_<week>.pdf
                             and logs this week's model picks into nfl_internal_model_track.csv
  3. nfl_superlatives.py     the League Superlatives PDF: best/worst team in every category, plus the model's
                             player-level predictions for every game of the latest week where BOTH teams are a
                             #1 best/worst on the sheet -> generated/NFL_League_Superlatives_<week>.pdf
  4. nfl_workhorse_tracker.py the Workhorse Tracker: each team's lead back, top 3 pass-catchers and QB (last-3 usage,
                             efficiency vs their own baseline, results vs expected, team scoring without him, injury
                             impact) plus every team's next-game matchup outlook vs the opponent's rush/pass/sack/havoc
                             D ranks -> generated/NFL_Workhorse_Tracker_<weektag>.pdf
  6. nfl_spread_ai.py        THE SPREAD AI: walk-forward-backtested picks against the Vegas spread (ridge + neural-net ensemble + boosting),
                             grades its own logged picks, retrains on every completed game -> generated/NFL_Spread_Picks_week<N>.pdf,
                             nfl_spread_picks.csv, nfl_spread_picks_log.csv  (runs last, pulls its own history into nfl_spread_cache/)
  5. nfl_model_scorecard.py  scores every logged pick whose game is final and prints the accuracy readout
                             (so run it after the games are done to see the latest week's results)

Supporting files that must stay in this folder (the scripts find them next to themselves):
  nfl_delta_grid.py, nfl_margin_model.py (+ nfl_margin_model_fit.json, nfl_margin_historical_cache.csv),
  nfl_coordinator_mapping.csv (OC/DC tendencies -- the data pull reads it), nfl_stat_history.json,
  nfl_internal_model_track.csv.
NOTE: nfl_coordinator_mapping.csv had been sitting in junk/, which is why the Week 3 and Week 4 sheets had no
OC/DC tendency data. It's back in this folder, so the next data pull fills those in again.

Old scripts, backups, backtests, previous weeks' sheets and one-off PDFs are in junk/.
"""
import os
import sys
import shutil
import datetime
import importlib.util
import subprocess

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

NO_PULL = "--no-pull" in sys.argv
LIVE = "--live" in sys.argv
LIVE_ONLY = "--live-only" in sys.argv
CHECK_ONLY = "--check" in sys.argv

SCRIPTS = ["nfl_spread_ai.py", "nfl_data_pull.py", "nfl_matchup_sheet.py", "nfl_superlatives.py", "nfl_workhorse_tracker.py", "nfl_model_scorecard.py", "nfl_delta_grid.py",
           "nfl_margin_model.py", "nfl_live_edge_monitor.py"]
NEEDED_FILES = ["nfl_margin_model_fit.json", "nfl_margin_historical_cache.csv", "nfl_stat_history.json",
                "nfl_internal_model_track.csv", "nfl_coordinator_mapping.csv"]
DATA_FILES = ["nfl_team_matrix.csv", "nfl_game_log.csv", "nfl_player_stats.csv", "nfl_injury_report.csv",
              "nfl_def_vs_position.csv"]


def preflight():
    ok = True
    print("=" * 70)
    print("PRE-FLIGHT")
    print("=" * 70)
    for s in SCRIPTS:
        if not os.path.exists(s):
            print(f"  !! missing script: {s}")
            ok = False
    for f in NEEDED_FILES:
        if not os.path.exists(f):
            print(f"  !! missing {f} (must sit in this folder -- see the top of this file)")
            ok = False
    for f in DATA_FILES:
        if not os.path.exists(f):
            print(f"  ..  {f} not here yet (the data pull creates it)")
    if importlib.util.find_spec("nflreadpy") is None:
        print("  !! the 'nflreadpy' Python package isn't installed in this Python -- pip3 install -r requirements.txt")
        ok = False
    if ok:
        print("  everything it needs is in place")
    return ok


def run(label, args):
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)
    rc = subprocess.run([sys.executable] + args, cwd=HERE).returncode
    if rc != 0:
        print(f"!! {label} exited with code {rc}")
    return rc == 0


def snapshot():
    """Dated copy of the freshly-pulled matrix + game log: the pre-game numbers each week's games were
    actually played against (the matrix itself gets overwritten every pull)."""
    os.makedirs("history", exist_ok=True)
    stamp = datetime.date.today().strftime("%Y-%m-%d")
    for f in ("nfl_team_matrix.csv", "nfl_game_log.csv"):
        if os.path.exists(f):
            dest = os.path.join("history", f.replace(".csv", f"_{stamp}.csv"))
            shutil.copyfile(f, dest)
    print(f"  saved history/nfl_team_matrix_{stamp}.csv and nfl_game_log_{stamp}.csv")


def main():
    ok = preflight()
    if CHECK_ONLY:
        print("\nAll set." if ok else "\nFix the items marked !! above, then run it again.")
        return
    if LIVE_ONLY:
        run("LIVE EDGE MONITOR (nfl_live_edge_monitor.py) -- Ctrl+C to stop", ["nfl_live_edge_monitor.py"])
        return
    os.makedirs("generated", exist_ok=True)
    failures = []
    if "--spread-only" in sys.argv:
        run("SPREAD AI (nfl_spread_ai.py)", ["nfl_spread_ai.py"] + [a for a in sys.argv[1:] if a not in ("--spread-only",)])
        return

    pull_ok = True
    if NO_PULL:
        print("\n(--no-pull: reusing the data already in this folder)")
    else:
        pull_ok = run("STEP 1/5 -- DATA PULL (nfl_data_pull.py)", ["nfl_data_pull.py"])
        if pull_ok:
            snapshot()
        else:
            failures.append("nfl_data_pull.py")

    if not pull_ok:
        print("\n!! The data pull failed, so the sheets would be built from OLD data -- skipping them.")
        print("   (Run with --no-pull if you really do want to rebuild from what's here.)")
    else:
        if not run("STEP 2/5 -- ALL SHEETS (nfl_matchup_sheet.py)", ["nfl_matchup_sheet.py"]):
            failures.append("nfl_matchup_sheet.py")
        # needs the picks the sheet step just logged, so it runs right after it
        if not run("STEP 3/5 -- LEAGUE SUPERLATIVES + HEADLINE-GAME PREDICTIONS (nfl_superlatives.py)", ["nfl_superlatives.py"]):
            failures.append("nfl_superlatives.py")
        # workhorse tracker: lead back / top receivers / QB per team + next-game matchup outlook (needs the fresh matrix, log, injuries)
        if not run("STEP 4/5 -- WORKHORSE TRACKER + NEXT-GAME OUTLOOK (nfl_workhorse_tracker.py)",
                   ["nfl_workhorse_tracker.py"] + (["--no-pull"] if NO_PULL else [])):
            failures.append("nfl_workhorse_tracker.py")

    if not run("STEP 5/5 -- SCORECARD (nfl_model_scorecard.py)", ["nfl_model_scorecard.py"]):
        failures.append("nfl_model_scorecard.py")

    # extra step: picks against the Vegas spread (pulls its own nflverse history, so it runs even with --no-pull)
    if not run("STEP 6 -- SPREAD AI: PICKS AGAINST THE VEGAS SPREAD (nfl_spread_ai.py)", ["nfl_spread_ai.py"]):
        failures.append("nfl_spread_ai.py")

    print()
    print("=" * 70)
    print("Done. Everything is in generated/ :")
    try:
        for name in sorted(os.listdir("generated")):
            if name.startswith("."):
                continue
            print("  - " + name + ("/" if os.path.isdir(os.path.join("generated", name)) else ""))
    except OSError:
        pass
    if failures:
        print("!! These steps had a problem -- scroll up for the error: " + ", ".join(failures))
    print("=" * 70)

    if LIVE:
        run("LIVE EDGE MONITOR (nfl_live_edge_monitor.py) -- Ctrl+C to stop", ["nfl_live_edge_monitor.py"])


if __name__ == "__main__":
    main()
