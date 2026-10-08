"""
NFL DELTA GRID -- condensed one-number-per-row matchup sheet, same idea as
the CFB delta's/new.py grid: every delta row collapses to ONE number,
printed under whichever team it favors (blank on the other side) -- same as the CFB version.

Reads straight out of nfl_team_matrix.csv (bo confirmed this, 2026-10-04)
-- no PDF text-scraping, no regex, no rank-suffix parsing bugs.

v2 (2026-10-04, same day):
  - League rank added to every TEAM INFO row (reuses total_void_rank/
    off_void_rank/def_void_rank where the matrix already has them; computed
    on the fly for everything else via a real full-df pandas rank, same
    "min" method CFB's own rank lookup uses).
  - Added to TEAM INFO: Points/Game, Yards/Game, Pass Yds/Play (PYPP),
    Rush Yds/Play (RYPP), Points/Drive (PPD -- see nfl_data_pull.py's
    2026-10-04 points_per_drive addition; this is a TD=7/FG=3/Safety=-2
    drive-result approximation, NOT a real score-delta like CFB's PPD --
    flagged honestly rather than called "real" in the same sense).
  - Added delta rows: Fumble Delta, Tug o' War (Interior) Delta, Edge
    Pressure Delta, Separation Delta, Catch% Delta, Pass Rush vs Pass
    Block Delta (PBE vs PRP), CPOE Delta. New "team_vs_team" delta type
    added for the ones with no real off/def-allowed split in the matrix
    (Tug o' War, Edge Pressure, CPOE -- matches how nfl_matchup_sheet.py's
    own Tug o' War/Edge Pressure panels already compare "each team's OWN
    defensive front" head to head, not matched to a specific opposing
    O-line -- see that file's own comment at the TUG O' WAR block).
  - Team-name header now highlights turquoise for whichever team
    matchup_win_probs favors (same model/color nfl_matchup_sheet.py uses
    for its own >=80% win-prob highlight -- here it's just "the favorite",
    no threshold gate).
  - Layout shrunk to fit 4 matchups per page (2x2). Two single-matchup tables are placed
    side by side per "quadrant-row"; reportlab's own natural pagination
    packs two quadrant-rows per page once there's room (page 1 fits fewer
    since the title/note block at the top eats into its space -- normal,
    not a bug). An earlier version forced an explicit page break every 2
    quadrant-rows, which actually fought the natural flow and wasted
    space; removed once the shrink made 2-per-page fit on its own (see the
    comment above the build loop in build_delta_grid_pdf).

HOW TO WIRE THIS IN: nfl_matchup_sheet.py's Step 6 hook (added at the very
end of that file) already calls:

    nfl_delta_grid.build_delta_grid_pdf(df, matchups, OUT_DIR, week_tag, matchup_win_probs)

reusing the SAME df/matchups/week_tag/matchup_win_probs that script
already computes earlier in its own run -- no second nflverse pull, no
second command for bo to remember.
"""

import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------
# STYLES -- shrunk for a 2x2 (4-per-page) layout.
# ---------------------------------------------------------------------
HIGHLIGHT_COLOR = colors.HexColor("#0C8C8C")  # same turquoise family nfl_matchup_sheet.py uses for its win-prob highlight
SECTION_BG = colors.HexColor("#1F2A3C")
SECTION_FG = colors.white
GRID_LINE = colors.HexColor("#CCCCCC")
HEADER_BG = colors.HexColor("#E8E8E8")

_styles = getSampleStyleSheet()
TEAM_HDR_STYLE = ParagraphStyle("team_hdr", parent=_styles["Normal"], fontName="Helvetica-Bold", fontSize=6.5, alignment=TA_CENTER, leading=7.2)
TEAM_HDR_STYLE_HOT = ParagraphStyle("team_hdr_hot", parent=TEAM_HDR_STYLE, textColor=HIGHLIGHT_COLOR)
SECTION_HDR_STYLE = ParagraphStyle("section_hdr", parent=_styles["Normal"], fontName="Helvetica-Bold", fontSize=4.0, alignment=TA_CENTER, textColor=SECTION_FG, leading=4.5)
# TEAM INFO (the "top grid") sized UP a bit, delta sections (the "bottom
# grid") sized DOWN a bit -- the two
# pairs converge on close to the same effective row height so the sheet
# reads as one consistent size top to bottom, not two different grids.
LABEL_INFO_STYLE = ParagraphStyle("label_info", parent=_styles["Normal"], fontName="Helvetica", fontSize=4.3, alignment=TA_CENTER, leading=4.8)
RAW_STYLE = ParagraphStyle("raw", parent=_styles["Normal"], fontName="Helvetica", fontSize=4.3, alignment=TA_CENTER, leading=4.8)
LABEL_STYLE = ParagraphStyle("label", parent=_styles["Normal"], fontName="Helvetica", fontSize=3.4, alignment=TA_CENTER, leading=3.9)
VAL_STYLE = ParagraphStyle("val", parent=_styles["Normal"], fontName="Helvetica-Bold", fontSize=3.9, alignment=TA_CENTER, leading=4.4)
VAL_STYLE_HL = ParagraphStyle("val_hl", parent=VAL_STYLE, textColor=HIGHLIGHT_COLOR, fontSize=4.3, leading=4.8)

COL_WIDTHS = [48, 108, 48]  # away | label | home -- one quadrant's width in the 2x2 grid
QUAD_GUTTER = 10


def _num(row, key):
    """Safe float pull from a nfl_team_matrix.csv row. Returns None (never
    raises) for a missing/blank/NaN cell -- early-season teams are
    routinely missing a stat for the first week or two."""
    if row is None or key not in row.index:
        return None
    v = row[key]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None
    return v


# =====================================================================
# METRIC SCHEMA -- see module docstring for the full v2 changelog.
# =====================================================================
SECTIONS = {
    "TEAM INFO": [
        {"name": "Yards/Play", "type": "raw_pair", "key": "off_yards_per_play", "fmt": "{:.2f}", "rank": True, "higher_is_better": True},
        {"name": "Pass Play %", "type": "raw_pair", "key": "pass_play_pct_offense", "fmt": "{:.1f}", "suffix": "%", "rank": True, "higher_is_better": True},
        {"name": "Rush Play %", "type": "raw_pair", "key": "rush_play_pct_offense", "fmt": "{:.1f}", "suffix": "%", "rank": True, "higher_is_better": True},
        {"name": "VOID (Total)", "type": "raw_pair", "key": "total_void", "fmt": "{:.3f}", "rank": True, "rank_key": "total_void_rank"},
        {"name": "Points/Game", "type": "raw_pair", "key": "points_per_game", "fmt": "{:.1f}", "rank": True, "higher_is_better": True},
        {"name": "Yards/Game", "type": "raw_pair", "key": "total_yards_per_game", "fmt": "{:.1f}", "rank": True, "higher_is_better": True},
        {"name": "Pass Yds/Play", "type": "raw_pair", "key": "off_yards_per_pass", "fmt": "{:.2f}", "rank": True, "higher_is_better": True},
        {"name": "Rush Yds/Play", "type": "raw_pair", "key": "off_rushing_yards_per_play", "fmt": "{:.2f}", "rank": True, "higher_is_better": True},
        {"name": "Points/Drive", "type": "raw_pair", "key": "points_per_drive", "fmt": "{:.2f}", "rank": True, "higher_is_better": True},
    ],
    "TEAM DELTAS": [
        {
            "name": "Net EPA Delta", "type": "standard",
            "away_off": "off_efficiency_epa", "home_def": "def_efficiency_epa",
            "home_off": "off_efficiency_epa", "away_def": "def_efficiency_epa",
        },
        {
            "name": "Net Success Rate Delta", "type": "net_stat_sub",
            "off_key": "off_success_rate", "def_key": "def_success_rate",
        },
        {
            "name": "Fumble Delta", "type": "inverted",
            "home_def": "opponent_fumbles_forced_per_game", "away_off": "fumbles_lost_per_game",
            "away_def": "opponent_fumbles_forced_per_game", "home_off": "fumbles_lost_per_game",
        },
    ],
    "LINE DELTAS": [
        # Sacks/TFL/Stuff Rate/Pressure Rate moved out (stuff rate and TFL under rush, sacks and
        # pressure under pass) -- LINE DELTAS now holds just the two
        # team-vs-team trenches stats that didn't fit either direction.
        {"name": "Tug o' War (Interior) Delta", "type": "team_vs_team", "key": "interior_pressure_rate"},
        {"name": "Edge Pressure Delta", "type": "team_vs_team", "key": "edge_pressure_rate"},
    ],
    "PASS DELTAS": [
        # Sacks Delta and Pressure Rate Delta moved in from LINE DELTAS
        # (2026-10-04) -- sacks/pressure happen on pass plays, grouped here
        # next to Protection vs Pass Rush Delta (same trenches-on-dropbacks idea).
        {
            "name": "Sacks Delta", "type": "net_margin",
            "away_def": "sacks_made", "away_off": "sacks_allowed",
            "home_def": "sacks_made", "home_off": "sacks_allowed",
        },
        {
            "name": "Pressure Rate Delta", "type": "standard",
            "away_off": "pressure_rate_allowed_pct", "home_def": "pressure_rate_forced_pct",
            "home_off": "pressure_rate_allowed_pct", "away_def": "pressure_rate_forced_pct",
        },
        {
            "name": "Protection vs Pass Rush Delta", "type": "standard",
            "away_off": "pbe_score", "home_def": "prp_score",
            "home_off": "pbe_score", "away_def": "prp_score",
        },
        {
            "name": "Total Pass Matchup Delta", "type": "standard",
            "away_off": "passing_yards_per_game", "home_def": "passing_yards_allowed_per_game",
            "home_off": "passing_yards_per_game", "away_def": "passing_yards_allowed_per_game",
        },
        {
            "name": "Passing Explosiveness Delta", "type": "standard",
            "away_off": "yards_per_completion", "home_def": "yards_per_completion_allowed",
            "home_off": "yards_per_completion", "away_def": "yards_per_completion_allowed",
        },
        {
            "name": "YAC Delta", "type": "standard",
            "away_off": "yac_per_reception_pbp", "home_def": "yac_allowed_per_reception_pbp",
            "home_off": "yac_per_reception_pbp", "away_def": "yac_allowed_per_reception_pbp",
        },
        {
            "name": "Separation Delta", "type": "standard",
            "away_off": "target_separation", "home_def": "target_separation_allowed",
            "home_off": "target_separation", "away_def": "target_separation_allowed",
        },
        {
            "name": "Catch% Delta", "type": "standard",
            "away_off": "catch_pct", "home_def": "catch_pct_allowed",
            "home_off": "catch_pct", "away_def": "catch_pct_allowed",
        },
        {
            "name": "Interception Delta", "type": "inverted", "highlight": True,
            "home_def": "interceptions_gained_per_game", "away_off": "interceptions_thrown_per_game",
            "away_def": "interceptions_gained_per_game", "home_off": "interceptions_thrown_per_game",
        },
        {"name": "CPOE Delta", "type": "team_vs_team", "key": "cpoe"},
    ],
    "RUSH DELTAS": [
        # TFL Delta and Stuff Rate Delta moved in from LINE DELTAS
        # (2026-10-04) -- both are run-stopping/run-blocking stats, grouped
        # here next to the rest of the rushing picture.
        {
            "name": "TFL Delta", "type": "net_margin",
            "away_def": "tfl_made", "away_off": "tfl_allowed",
            "home_def": "tfl_made", "home_off": "tfl_allowed",
        },
        {
            "name": "Stuff Rate Delta", "type": "inverted",
            "home_def": "stuff_rate_defense", "away_off": "stuff_rate_offense",
            "away_def": "stuff_rate_defense", "home_off": "stuff_rate_offense",
        },
        {
            "name": "Rushing Reality Delta", "type": "ratio",
            "away_vol": "rushing_yards_per_game", "home_vol_allowed": "rushing_yards_allowed_per_game",
            "away_eff": "off_rushing_yards_per_play", "home_eff_allowed": "def_rushing_yards_per_play",
            "home_vol": "rushing_yards_per_game", "away_vol_allowed": "rushing_yards_allowed_per_game",
            "home_eff": "off_rushing_yards_per_play", "away_eff_allowed": "def_rushing_yards_per_play",
        },
        {
            "name": "Run Success Rate Delta", "type": "standard", "highlight": True,
            "away_off": "off_run_success_rate", "home_def": "def_run_stop_rate",
            "home_off": "off_run_success_rate", "away_def": "def_run_stop_rate",
        },
        # Line Yards Delta -- NEW (2026-10-04). Real Adjusted Line Yards (nfl_data_pull.py,
        # standard Football Outsiders-style credit formula off real pbp rush
        # plays, QB scrambles excluded) -- CFB's own Line Yards comes straight
        # from CFBD's API as a precomputed field; nflverse has no free
        # equivalent, so this one is built from real play-by-play instead.
        {
            "name": "Line Yards Delta", "type": "standard",
            "away_off": "off_line_yards", "home_def": "def_line_yards_allowed",
            "home_off": "off_line_yards", "away_def": "def_line_yards_allowed",
        },
    ],
}


# ---------------------------------------------------------------------
# DELTA MATH -- positive always means "favors away" throughout. Every
# type below returns None if any required input is missing, so a
# half-missing-data team just prints a blank row instead of crashing.
# ---------------------------------------------------------------------

def _standard(spec, row_away, row_home):
    away_off = _num(row_away, spec["away_off"])
    home_def = _num(row_home, spec["home_def"])
    home_off = _num(row_home, spec["home_off"])
    away_def = _num(row_away, spec["away_def"])
    if None in (away_off, home_def, home_off, away_def):
        return None
    return (away_off - home_def) - (home_off - away_def)


def _inverted(spec, row_away, row_home):
    # Same shape as "standard" but for stats where a HIGHER offense number
    # is actually bad (gets stuffed more / throws more picks / fumbles
    # more) -- sign flipped relative to "standard" using identical term
    # positions.
    home_def = _num(row_home, spec["home_def"])
    away_off = _num(row_away, spec["away_off"])
    away_def = _num(row_away, spec["away_def"])
    home_off = _num(row_home, spec["home_off"])
    if None in (home_def, away_off, away_def, home_off):
        return None
    return (home_def - away_off) - (away_def - home_off)


def _net_margin(spec, row_away, row_home):
    away_def = _num(row_away, spec["away_def"])
    away_off = _num(row_away, spec["away_off"])
    home_def = _num(row_home, spec["home_def"])
    home_off = _num(row_home, spec["home_off"])
    if None in (away_def, away_off, home_def, home_off):
        return None
    away_net = away_def - away_off
    home_net = home_def - home_off
    return away_net - home_net


def _net_stat_sub(spec, row_away, row_home):
    away_off = _num(row_away, spec["off_key"])
    away_def = _num(row_away, spec["def_key"])
    home_off = _num(row_home, spec["off_key"])
    home_def = _num(row_home, spec["def_key"])
    if None in (away_off, away_def, home_off, home_def):
        return None
    away_net = away_off - away_def
    home_net = home_off - home_def
    return away_net - home_net


def _ratio(spec, row_away, row_home):
    avol = _num(row_away, spec["away_vol"])
    h_va = _num(row_home, spec["home_vol_allowed"])
    aeff = _num(row_away, spec["away_eff"])
    h_ea = _num(row_home, spec["home_eff_allowed"])
    hvol = _num(row_home, spec["home_vol"])
    a_va = _num(row_away, spec["away_vol_allowed"])
    heff = _num(row_home, spec["home_eff"])
    a_ea = _num(row_away, spec["away_eff_allowed"])
    if None in (avol, h_va, aeff, h_ea, hvol, a_va, heff, a_ea):
        return None
    d1 = aeff - h_ea
    d2 = heff - a_ea
    if d1 == 0 or d2 == 0:
        return None
    return ((avol - h_va) / d1) - ((hvol - a_va) / d2)


def _team_vs_team(spec, row_away, row_home):
    # For stats with no real off/def-allowed split in the matrix (Tug o'
    # War, Edge Pressure, CPOE) -- straight head-to-head on the SAME
    # column, matching how nfl_matchup_sheet.py's own Tug o' War/Edge
    # Pressure panels already compare "each team's OWN defensive front",
    # not matched to a specific opposing O-line.
    a = _num(row_away, spec["key"])
    h = _num(row_home, spec["key"])
    if None in (a, h):
        return None
    return a - h


_DELTA_FNS = {
    "standard": _standard,
    "inverted": _inverted,
    "net_margin": _net_margin,
    "net_stat_sub": _net_stat_sub,
    "ratio": _ratio,
    "team_vs_team": _team_vs_team,
}


def compute_delta(spec, row_away, row_home):
    """Returns the signed delta value (positive favors away), or None if
    any required stat is missing for this matchup."""
    fn = _DELTA_FNS[spec["type"]]
    return fn(spec, row_away, row_home)


# ---------------------------------------------------------------------
# LEAGUE RANK -- computed once per build_delta_grid_pdf() call across the
# WHOLE real df (all current teams), same "min" ranking method CFB's own
# rank lookup uses. VOID rows skip this and use the matrix's own
# precomputed total_void_rank/off_void_rank/def_void_rank columns instead
# (via "rank_key" on the spec) rather than recomputing.
# ---------------------------------------------------------------------

def _build_rank_lookups(df):
    lookups = {}
    df_idx = df.set_index("team", drop=False) if "team" in df.columns else df
    for specs in SECTIONS.values():
        for spec in specs:
            if spec.get("type") == "raw_pair" and spec.get("rank") and not spec.get("rank_key"):
                key = spec["key"]
                if key not in lookups and key in df_idx.columns:
                    higher = spec.get("higher_is_better", True)
                    s = pd.to_numeric(df_idx[key], errors="coerce")
                    lookups[key] = s.rank(method="min", ascending=not higher, na_option="keep")
    return lookups


# ---------------------------------------------------------------------
# RENDERING -- single-number-per-row, placed under whichever team it
# favors, blank on the other side.
# ---------------------------------------------------------------------

def _delta_row(spec):
    val = spec["_computed_value"]
    hl = bool(spec.get("highlight"))
    style = VAL_STYLE_HL if hl else VAL_STYLE
    if val is None:
        return ["", Paragraph(f"<b>{spec['name']}</b>" if hl else spec["name"], LABEL_STYLE), ""]
    label_txt = f"<font color='#0C8C8C'><b>{spec['name']}</b></font>" if hl else spec["name"]
    if val > 0:
        return [Paragraph(f"{val:.2f}", style), Paragraph(label_txt, LABEL_STYLE), ""]
    elif val < 0:
        return ["", Paragraph(label_txt, LABEL_STYLE), Paragraph(f"{abs(val):.2f}", style)]
    else:
        return [Paragraph("0.00", style), Paragraph(label_txt, LABEL_STYLE), Paragraph("0.00", style)]


def _raw_pair_row(spec, row_away, row_home, rank_lookups):
    fmt = spec.get("fmt", "{:.1f}")
    suffix = spec.get("suffix", "")
    key = spec["key"]
    a = _num(row_away, key)
    h = _num(row_home, key)
    a_txt = (fmt.format(a) + suffix) if a is not None else "N/A"
    h_txt = (fmt.format(h) + suffix) if h is not None else "N/A"
    if spec.get("rank"):
        rank_key = spec.get("rank_key")
        if rank_key:
            a_rank = _num(row_away, rank_key)
            h_rank = _num(row_home, rank_key)
        else:
            ranks = rank_lookups.get(key)
            a_rank = ranks.get(row_away.name) if (ranks is not None and row_away is not None) else None
            h_rank = ranks.get(row_home.name) if (ranks is not None and row_home is not None) else None
        if a is not None and a_rank is not None and not pd.isna(a_rank):
            a_txt += f" ({int(a_rank)})"
        if h is not None and h_rank is not None and not pd.isna(h_rank):
            h_txt += f" ({int(h_rank)})"
    return [Paragraph(a_txt, RAW_STYLE), Paragraph(spec["name"], LABEL_INFO_STYLE), Paragraph(h_txt, RAW_STYLE)]


def build_single_matchup_table(away_team, home_team, row_away, row_home, rank_lookups, win_prob_away=None):
    """Returns a single Table flowable for one matchup -- 3-column (away |
    label | home) shape. win_prob_away = model's win probability for the
    AWAY team (matchup_win_probs.get((away_team, home_team)) from
    nfl_matchup_sheet.py) -- whichever side is >0.5 gets its name
    highlighted turquoise. None (model couldn't produce a prediction) means
    neither side is highlighted."""
    away_style = TEAM_HDR_STYLE
    home_style = TEAM_HDR_STYLE
    if win_prob_away is not None:
        if win_prob_away >= 0.5:
            away_style = TEAM_HDR_STYLE_HOT
        else:
            home_style = TEAM_HDR_STYLE_HOT

    data = [[Paragraph(away_team, away_style), "", Paragraph(home_team, home_style)]]
    row_styles = []
    r = 1
    for section_name, specs in SECTIONS.items():
        # The section-name Paragraph has to sit in column 0 (the span's
        # anchor cell) -- reportlab's SPAN only ever renders the content of
        # the TOP-LEFT cell of the merged range.
        data.append([Paragraph(section_name, SECTION_HDR_STYLE), "", ""])
        row_styles.append(("SPAN", (0, r), (-1, r)))
        row_styles.append(("BACKGROUND", (0, r), (-1, r), SECTION_BG))
        r += 1
        for spec in specs:
            if spec["type"] == "raw_pair":
                data.append(_raw_pair_row(spec, row_away, row_home, rank_lookups))
            else:
                spec = dict(spec)  # don't mutate the shared SECTIONS schema
                spec["_computed_value"] = compute_delta(spec, row_away, row_home)
                data.append(_delta_row(spec))
            r += 1

    t = Table(data, colWidths=COL_WIDTHS, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, GRID_LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 0.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
    ] + row_styles
    t.setStyle(TableStyle(style_cmds))
    return t


def _make_quadrant_row(tables):
    """Lays out 1 or 2 matchup tables side by side (the 2x2 grid's one
    row). A trailing odd matchup gets the left slot with the right slot
    blank."""
    if len(tables) == 2:
        row = [tables[0], "", tables[1]]
    else:
        row = [tables[0], "", ""]
    outer = Table([row], colWidths=[sum(COL_WIDTHS), QUAD_GUTTER, sum(COL_WIDTHS)])
    outer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return outer


def build_delta_grid_pdf(df, matchups, out_dir, week_tag="", matchup_win_probs=None):
    """Main entry point. df = the real nfl_team_matrix.csv DataFrame,
    matchups = list of (away_team, home_team) tuples, out_dir = Path to
    write the PDF into, week_tag = e.g. "week4", matchup_win_probs =
    nfl_matchup_sheet.py's own {(away,home): win_prob_away} dict (pass
    None or {} to skip the favored-team highlight entirely)."""
    matchup_win_probs = matchup_win_probs or {}
    out_dir = Path(out_dir)
    suffix = f"_{week_tag}" if week_tag else ""
    out_path = out_dir / f"All_NFL_Matchups_Grid_Report{suffix}.pdf"

    df_indexed = df.set_index("team", drop=False) if "team" in df.columns else df
    rank_lookups = _build_rank_lookups(df)

    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter,
        leftMargin=0.3 * inch, rightMargin=0.3 * inch,
        topMargin=0.25 * inch, bottomMargin=0.25 * inch,
    )
    elements = []
    # Trimmed further (2026-10-04) so
    # page 1 -- the only page carrying this title/note block -- has enough
    # reclaimed space to also fit 2 quadrant-rows (4 matchups) like every
    # other page, instead of just 1.
    title_style = ParagraphStyle("title", parent=_styles["Title"], fontSize=9, leading=10.5, spaceAfter=1)
    elements.append(Paragraph("NFL Matchups -- Delta Grid Report", title_style))
    note_style = ParagraphStyle("note", parent=_styles["Normal"], fontSize=4.6, leading=5.3, textColor=colors.grey)
    elements.append(Paragraph(
        "Every row collapses to one number under the team it favors. "
        "<font color='#0C8C8C'><b>Teal rows</b></font> (Interception/Run Success Rate Delta) carry real predictive signal; "
        "<font color='#0C8C8C'><b>teal team name</b></font> is whichever side the model favors.",
        note_style,
    ))
    elements.append(Spacer(1, 2))

    # No manual PageBreak counter here on purpose -- an earlier version
    # forced one every 2 quadrant-rows (4 matchups), but that double-counts
    # against reportlab's OWN natural overflow: page 1's title/note block
    # steals some vertical space, so row 2 often already overflows to page
    # 2 on its own -- then the manual break fired right after it anyway,
    # wasting page 2's remaining room instead of letting row 3 join it.
    # Each quadrant-row now genuinely fits 2-per-page in the shrunk 4.3-7.5pt
    # layout (confirmed in the smoke-test preview), so plain natural flow
    # already produces a clean, fully-packed 2x2 grid -- just slower to
    # fill on page 1 specifically, which is normal and not a bug.
    missing_teams = []
    pending = []
    n = len(matchups)
    for idx, (away_team, home_team) in enumerate(matchups):
        row_away = df_indexed.loc[away_team] if away_team in df_indexed.index else None
        row_home = df_indexed.loc[home_team] if home_team in df_indexed.index else None
        if row_away is None:
            missing_teams.append(away_team)
        if row_home is None:
            missing_teams.append(home_team)
        win_prob_away = matchup_win_probs.get((away_team, home_team))
        tbl = build_single_matchup_table(away_team, home_team, row_away, row_home, rank_lookups, win_prob_away)
        pending.append(tbl)

        is_last = idx == n - 1
        if len(pending) == 2 or is_last:
            elements.append(_make_quadrant_row(pending))
            pending = []

    if missing_teams:
        print(f" -> ⚠ No nfl_team_matrix.csv row found for: {sorted(set(missing_teams))} -- those rows printed blank.")

    doc.build(elements)
    print(f"[Done] NFL delta grid report: '{out_path}'")
    return out_path


if __name__ == "__main__":
    # Standalone smoke-test path -- reads nfl_team_matrix.csv directly and
    # builds the grid using every team paired alphabetically, since there's
    # no real schedule pull in THIS file on purpose (the real integration
    # reuses nfl_matchup_sheet.py's own `matchups`/`matchup_win_probs`).
    here = Path(__file__).resolve().parent
    csv_path = here / "nfl_team_matrix.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found -- run nfl_data_pull.py first.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    teams = sorted(df["team"].dropna().unique().tolist())
    demo_matchups = [(teams[i], teams[i + 1]) for i in range(0, len(teams) - 1, 2)]
    # Fake win probs for the smoke test only, so the highlight logic gets exercised.
    demo_win_probs = {(a, b): (0.63 if i % 2 == 0 else 0.41) for i, (a, b) in enumerate(demo_matchups)}
    print(f" -> SMOKE TEST: pairing {len(demo_matchups)} alphabetical matchup(s) from nfl_team_matrix.csv (NOT a real schedule).")
    build_delta_grid_pdf(df, demo_matchups, here, week_tag="smoketest", matchup_win_probs=demo_win_probs)
