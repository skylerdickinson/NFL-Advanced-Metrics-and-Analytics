research_and_backtests/ -- ON-DEMAND validation tools, moved here 2026-09-22
==============================================================================
None of these run as part of your weekly routine (that's still just
nfl_data_pull.py then nfl_matchup_sheet.py, from the main "grind it out"
folder -- see README.txt there). Every script in this subfolder was a
one-time or run-whenever-curious tool used to validate a stat, a model
weight, or an idea before it went into the live sheet. They don't read any
of the weekly CSVs and nothing else in the project reads their output, so
moving them here doesn't change how anything behaves -- they're just out of
the way now.

To run one, cd into THIS subfolder first (each writes its own output next
to itself, same as before):

    cd ~/Desktop/"grind it out"/research_and_backtests
    python3 nfl_historical_backtest.py

What's here:

  backtest_pressure_index.py            One-time validator for the Tug o'
                                         War / Edge Pressure index. Writes
                                         nfl_pressure_index_backtest_raw.csv
                                         (also here) for your own review.

  nfl_historical_backtest.py            Tests the stat-weighted win model
                                         against several full past seasons
                                         at once (the in-season model only
                                         has 1-2 games/team to learn from
                                         early on -- this checks it against
                                         a bigger, already-settled sample).

  nfl_walkforward_backtest.py           The honest pre-game version of the
                                         above: only uses each team's rolling
                                         average from games played BEFORE
                                         the week being tested, no leakage.

  nfl_flare_backtest.py                 Checks whether games where MORE
                                         stats agree pre-game are actually
                                         won more often (consensus strength).

  nfl_quality_stat_matchup_analysis.py  Checks whether the badge's misses
                                         cluster in close-vote (toss-up)
                                         games vs. lopsided ones.

  nfl_sos_backtest.py                   Proves out the opponent-adjusted
                                         strength-of-schedule composite
                                         before it's trusted in a live pick.

  nfl_stat_gap_analysis.py              For weaker stats, checks whether a
                                         BIGGER pre-game gap between the two
                                         teams is actually more predictive
                                         than a small one.

Every one of these has its own detailed docstring at the top explaining
exactly what it does and why it was built -- read that first if you want
the full story on any of them.

THE ONE SCRIPT NOT MOVED, ON PURPOSE: nfl_model_scorecard.py
==============================================================================
This is the "run it every week so the model learns from its picks" script
you were trying to remember -- it stayed in the main "grind it out" folder,
not in here, because it reads/writes nfl_internal_model_track.csv which
lives there too.

It's technically also "on-demand" rather than auto-run by the weekly
pipeline (nfl_data_pull.py / nfl_matchup_sheet.py don't call it for you),
but it's the one you actually need to run yourself on a cadence -- once a
week's real games are final -- so it can pull real final scores and score
your model's past picks against them, building a real accuracy track
record over time instead of a static one-time backtest.

    cd ~/Desktop/"grind it out"
    python3 nfl_model_scorecard.py
