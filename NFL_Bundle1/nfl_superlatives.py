#!/usr/bin/env python3
"""
nfl_superlatives.py - weekly "League Superlatives" PDF.

Best / worst team in every category (top 3 and bottom 3 per card), then the
model's player-level predictions for every game of the latest week where BOTH
teams are a #1 best/worst on the sheet.

Reads (all in this folder):  nfl_team_matrix.csv, nfl_def_vs_position.csv,
  nfl_game_log.csv, nfl_player_stats.csv, nfl_injury_report.csv,
  nfl_internal_model_track.csv
Writes:  generated/NFL_League_Superlatives_<weektag>.pdf   (e.g. ..._week5.pdf)

Run on its own:  python3 nfl_superlatives.py [--week week5]
Normally run for you as a step of nfl_run_all.py (after the matchup sheets).
"""
import sys
sys.dont_write_bytecode = True
import re, math, argparse
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, KeepTogether

HERE = Path(__file__).resolve().parent
GEN_DIR = HERE / "generated"

GREEN, RED, INK, GREY = '#1E8E4E', '#C0392B', '#111111', '#666666'
st = lambda n, **k: ParagraphStyle(n, fontName=k.pop('f', 'Helvetica'), **k)
H1 = st('h1', fontSize=22, leading=26, textColor=colors.HexColor(INK), f='Helvetica-Bold')
SUB = st('sub', fontSize=9.5, leading=13, textColor=colors.HexColor(GREY))
SEC = st('sec', fontSize=13, leading=16, textColor=colors.white, f='Helvetica-Bold')
CT = st('ct', fontSize=10.5, leading=13, f='Helvetica-Bold', textColor=colors.HexColor(INK))
CN = st('cn', fontSize=7.5, leading=9.5, textColor=colors.HexColor(GREY))
CELL = st('cell', fontSize=9, leading=11.5)
SM = st('sm', fontSize=7.8, leading=10, textColor=colors.HexColor(GREY))
NOTE = st('note', fontSize=8.5, leading=11.5, textColor=colors.HexColor('#333333'))

def fm(v, f): return f.format(v)

def side(cat, best):
    key, hb = cat['key'], cat['higher_best']
    asc = not hb if best else hb
    s = df.sort_values(key, ascending=asc).head(3)
    col = GREEN if best else RED
    lines = []
    for i, (t, r) in enumerate(s.iterrows(), 1):
        main = f"<b>{i}. <font color='{col}'>{t}</font></b> &nbsp;{fm(r[key], cat['fmt'])}"
        if i == 1:
            sup = ' &middot; '.join(f"{lab} {fm(r[k], f)}" for lab, k, f in cat['support'])
            main = f"<font size='11'>{main}</font><br/><font size='7.5' color='{GREY}'>{sup}</font>"
        lines.append(Paragraph(main, CELL))
    return lines

def card(cat):
    hdr = [Paragraph(cat['title'], CT), '']
    note = [Paragraph(cat['note'], CN), '']
    b = side(cat, True); w = side(cat, False)
    lab = [Paragraph(f"<font color='{GREEN}'><b>BEST</b></font>", SM), Paragraph(f"<font color='{RED}'><b>WORST</b></font>", SM)]
    body = [[b[0], w[0]], [b[1], w[1]], [b[2], w[2]]]
    t = Table([hdr, note, lab] + body, colWidths=[3.55*inch, 3.55*inch])
    t.setStyle(TableStyle([
        ('SPAN', (0,0), (1,0)), ('SPAN', (0,1), (1,1)),
        ('BOX', (0,0), (-1,-1), 0.6, colors.HexColor('#BBBBBB')),
        ('BACKGROUND', (0,0), (-1,1), colors.HexColor('#F3F3F3')),
        ('LINEBELOW', (0,2), (-1,2), 0.4, colors.HexColor('#DDDDDD')),
        ('LINEAFTER', (0,2), (0,-1), 0.4, colors.HexColor('#DDDDDD')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
    ]))
    return KeepTogether([t, Spacer(1, 7)])

def section(name):
    t = Table([[Paragraph(name, SEC)]], colWidths=[7.1*inch])
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor(INK)),
                           ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5), ('LEFTPADDING', (0,0), (-1,-1), 8)]))
    return [Spacer(1, 4), t, Spacer(1, 6)]


GREEN, RED, INK, GREY = '#1E8E4E', '#C0392B', '#111111', '#666666'
def S(n, **k): return ParagraphStyle(n, fontName=k.pop('f', 'Helvetica'), **k)
GH = S('gh', fontSize=15, leading=18, f='Helvetica-Bold', textColor=colors.HexColor(INK))
GS = S('gs', fontSize=8.5, leading=11, textColor=colors.HexColor(GREY))
TH = S('th', fontSize=8, leading=10, f='Helvetica-Bold', textColor=colors.white)
TD = S('td', fontSize=8.5, leading=10.5)
TDB = S('tdb', fontSize=8.5, leading=10.5, f='Helvetica-Bold')
SUBH = S('subh', fontSize=10, leading=12, f='Helvetica-Bold', textColor=colors.HexColor(INK))
NT = S('nt', fontSize=7.8, leading=10, textColor=colors.HexColor('#444444'))
BIG = S('big', fontSize=22, leading=26, f='Helvetica-Bold', alignment=1)

def shrink(x, mean, w): return w * x + (1 - w) * mean
def clip(x, lo, hi): return max(lo, min(hi, x))
def p_atleast1(lam): return 1 - math.exp(-lam)

def build_games(team, players, inj, track, headline, week_tag):
    """team: team matrix indexed by team. headline: {team: [labels]}"""
    lg = {
        'dyp': team['def_yards_per_pass'].mean(), 'drp': team['def_rushing_yards_per_play'].mean(),
        'int': team['interceptions_thrown_per_game'].mean(),
        'sk_off': team['sack_pct_offense'].mean(), 'sk_def': team['sack_pct_defense'].mean(),
        'qb_ypg': players[(players.position == 'QB') & (players.pass_yards_pg > 100)]['pass_yards_pg'].mean(),
    }
    out_names = set(inj[inj.Status.str.lower() == 'out']['Player'])
    q_names = set(inj[inj.Status.str.lower() == 'questionable']['Player'])
    els = []
    w4 = track[track.week_tag == week_tag]
    games = [(r.away, r.home, r) for r in w4.itertuples() if r.away in headline and r.home in headline]

    def proj_team(t, opp, pts, opp_pts, is_away):
        tr, orow = team.loc[t], team.loc[opp]
        margin = pts - opp_pts
        pass_f = clip(1 - 0.012 * margin, 0.9, 1.1)
        rush_f = clip(1 + 0.012 * margin, 0.9, 1.1)
        pass_adj = 0.5 + 0.5 * (orow['def_yards_per_pass'] / lg['dyp'])
        rush_adj = 0.5 + 0.5 * (orow['def_rushing_yards_per_play'] / lg['drp'])
        team_td = pts * 0.8 / 7
        pass_share = clip(tr['passing_yards_per_game'] / (tr['passing_yards_per_game'] + tr['rushing_yards_per_game']), 0.5, 0.75)
        pass_td, rush_td = team_td * pass_share, team_td * (1 - pass_share)
        sk_rate = shrink(0.5 * (tr['sack_pct_offense'] + orow['sack_pct_defense']) / 100, lg['sk_off'] / 100, 0.75)
        dropbacks = tr['pass_attempts_per_game'] * pass_f
        sacks = sk_rate * dropbacks
        int_l = shrink(0.5 * (tr['interceptions_thrown_per_game'] + orow['interceptions_gained_per_game']), lg['int'], 0.5)
        roster = players[(players.team == t) & (~players.player.isin(out_names))]
        notes = []
        # ---- QB
        qbs = roster[roster.position == 'QB'].sort_values('snap_pct_avg', ascending=False)
        starter = tr['starting_qb_name']
        qb_out = starter in out_names
        if qb_out:
            q = {'name': f"Backup (starter {starter} OUT)", 'ypg': lg['qb_ypg'] * 0.85, 'rush': 8.0, 'lowconf': True}
            notes.append(f"{starter} is listed OUT. The only other QB on the roster has no passing sample, so this line is a league-average-starter baseline cut 15% and is low confidence.")
            int_l *= 1.2
        else:
            qr = qbs[qbs.player == starter]
            qr = qr.iloc[0] if len(qr) else qbs.iloc[0]
            q = {'name': qr.player + (' (Q)' if qr.player in q_names else ''), 'ypg': qr.pass_yards_pg, 'rush': qr.rush_yards_pg, 'lowconf': False}
        qb = dict(name=q['name'], pass_yds=q['ypg'] * pass_adj * pass_f, pass_td=pass_td, rush_yds=q['rush'],
                  int_l=int_l, sacks=sacks, low=q['lowconf'])
        # ---- skill players
        sk = roster[roster.position.isin(['RB', 'WR', 'TE'])]
        out_t = players[(players.team == t) & (players.player.isin(out_names)) & (players.position.isin(['RB', 'WR', 'TE']))]
        team_tgts = players[(players.team == t) & (players.position.isin(['RB', 'WR', 'TE']))]['targets_pg'].sum()
        lost = out_t['targets_pg'].sum()
        boost = clip(team_tgts / max(team_tgts - lost, 1e-6), 1, 1.25) if lost > 0 else 1
        if lost > 0:
            notes.append("OUT: " + ", ".join(f"{r.player} ({r.position})" for r in out_t.itertuples() if r.targets_pg > 0 or r.carries_pg > 0) + ". Their targets are spread to the rest of the group (x%.2f)." % boost)
        team_carries = players[(players.team == t) & (players.position == 'RB')]['carries_pg'].sum()
        rbs = sk[sk.position == 'RB'].sort_values('carries_pg', ascending=False).head(2)
        rb_rows = []
        for r in rbs.itertuples():
            n = r.carries_pg * r.gp
            ypc_own = r.rush_yards_pg / r.carries_pg if r.carries_pg > 0 else 4.3
            ypc = shrink(ypc_own, 4.3, n / (n + 30)) * rush_adj
            carries = r.carries_pg * rush_f
            share = r.carries_pg / team_carries if team_carries else 0
            rec = r.receptions_pg * pass_f; recy = r.rec_yards_pg * pass_adj * pass_f * boost
            rb_td = rush_td * share + pass_td * (r.targets_pg / team_tgts if team_tgts else 0) * boost
            rb_rows.append([r.player + (' (Q)' if r.player in q_names else ''), carries, carries * ypc, rec, recy, p_atleast1(rb_td)])
        wrs = sk[sk.position == 'WR'].assign(tt=lambda d: d.targets_pg * d.gp).sort_values('tt', ascending=False).head(3)
        wr_rows = []
        for r in wrs.itertuples():
            tg = r.targets_pg * boost * pass_f
            rec = r.receptions_pg * boost * pass_f
            ypr_adj = pass_adj
            wr_rows.append([r.player + (' (Q)' if r.player in q_names else ''), tg, rec, r.rec_yards_pg * boost * pass_f * ypr_adj,
                            p_atleast1(pass_td * (r.targets_pg / team_tgts if team_tgts else 0) * boost)])
        return qb, rb_rows, wr_rows, notes

    def tbl(rows, head, widths):
        t = Table([[Paragraph(h, TH) for h in head]] + rows, colWidths=[w * inch for w in widths])
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(INK)),
                               ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
                               ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#BBBBBB')),
                               ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                               ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]))
        t.hAlign = 'LEFT'
        return t

    for away, home, r in games:
        pa, ph = r.model_exp_pts_away, r.model_exp_pts_home
        wp = r.model_win_prob_away
        fav = away if wp >= 0.5 else home
        els.append(PageBreak())
        els.append(Paragraph(f"{away} at {home}", GH))
        why = '; '.join(f"<b>{t}</b>: " + ', '.join(headline[t][:4]) for t in (away, home))
        els.append(Paragraph("Why it made the list (each is a #1 best or worst team in): " + why, GS))
        els.append(Spacer(1, 5))
        score = Table([[Paragraph(f"{away} {pa:.0f}", BIG), Paragraph("-", BIG), Paragraph(f"{home} {ph:.0f}", BIG)]],
                      colWidths=[2.6 * inch, 0.5 * inch, 2.6 * inch])
        score.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#BBBBBB')), ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F3F3F3')),
                                   ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
        els.append(score)
        els.append(Paragraph(f"Model projected score {away} {pa:.1f}, {home} {ph:.1f} &middot; total {r.model_total:.1f} &middot; "
                             f"{fav} favored by {abs(r.model_margin_away_minus_home):.1f} &middot; {away} win prob {wp*100:.0f}%", GS))
        els.append(Spacer(1, 6))
        for t, opp, p1, p2 in ((away, home, pa, ph), (home, away, ph, pa)):
            qb, rbs, wrs, notes = proj_team(t, opp, p1, p2, t == away)
            els.append(Paragraph(f"{t} offense vs {opp} defense", SUBH))
            qrow = [[Paragraph(qb['name'] + (' *' if qb['low'] else ''), TDB), Paragraph(f"{qb['pass_yds']:.0f}", TD), Paragraph(f"{qb['pass_td']:.1f}", TD),
                     Paragraph(f"{qb['rush_yds']:.0f}", TD), Paragraph(f"{qb['int_l']:.2f} ({p_atleast1(qb['int_l'])*100:.0f}%)", TD),
                     Paragraph(f"{qb['sacks']:.1f}", TD)]]
            els.append(tbl(qrow, ["QB", "Pass yds", "Pass TD", "Rush yds", "INT exp (P&ge;1)", "Sacks taken"], [2.2, 0.8, 0.8, 0.8, 1.3, 1.0]))
            els.append(Spacer(1, 3))
            rrows = [[Paragraph(n, TDB), Paragraph(f"{c:.0f}", TD), Paragraph(f"{y:.0f}", TD), Paragraph(f"{rc:.1f}", TD), Paragraph(f"{ry:.0f}", TD), Paragraph(f"{td*100:.0f}%", TD)]
                     for n, c, y, rc, ry, td in rbs]
            els.append(tbl(rrows, ["RB", "Carries", "Rush yds", "Rec", "Rec yds", "Anytime TD"], [2.2, 0.8, 0.8, 0.8, 1.1, 1.2]))
            els.append(Spacer(1, 3))
            wrows = [[Paragraph(n, TDB), Paragraph(f"{tg:.1f}", TD), Paragraph(f"{rc:.1f}", TD), Paragraph(f"{ry:.0f}", TD), Paragraph(f"{td*100:.0f}%", TD)]
                     for n, tg, rc, ry, td in wrs]
            els.append(tbl(wrows, ["WR", "Targets", "Rec", "Rec yds", "Anytime TD"], [2.2, 1.1, 1.1, 1.2, 1.3]))
            for n in notes: els.append(Paragraph("&bull; " + n, NT))
            els.append(Spacer(1, 7))
        els.append(Paragraph("Q = listed Questionable. * = low-confidence baseline. Individual pass rushers and defensive backs are not in the player data, so sacks and "
                             "interceptions are projected per quarterback from both teams' sack and takeaway rates, not by defender.", NT))
    return els, len(games)


SEC_DEFS = [
 ("POWER INDEX", [
  dict(title="Power Index (Total: Offense + Defense)", key='total_power', higher_best=True, fmt="{:+.2f}",
       note="Sum of the project's weighted offense and defense power ratings (z-score blends of points, yards/play, EPA, success rate, turnovers, red zone). Higher = better.",
       support=[("Off", 'off_power_rating', "{:+.2f}"), ("Def", 'def_power_rating', "{:+.2f}"), ("VOID", 'total_void', "{:+.3f}")]),
  dict(title="Offensive Power Index", key='off_power_rating', higher_best=True, fmt="{:+.2f}",
       note="Offense-only rating from the same weighted blend.",
       support=[("PPG", 'points_per_game', "{:.1f}"), ("YPP", 'off_yards_per_play', "{:.2f}"), ("EPA/play", 'off_efficiency_epa', "{:+.3f}")]),
  dict(title="Defensive Power Index", key='def_power_rating', higher_best=True, fmt="{:+.2f}",
       note="Defense-only rating; sign-flipped so higher is always better.",
       support=[("PA/G", 'points_allowed_per_game', "{:.1f}"), ("YPP allowed", 'def_yards_per_play', "{:.2f}"), ("EPA/play allowed", 'def_efficiency_epa', "{:+.3f}")]),
 ]),
 ("OVERALL OFFENSE &amp; DEFENSE", [
  dict(title="Offense", key='off_efficiency_epa', higher_best=True, fmt="{:+.3f} EPA/play",
       note="Ranked by offensive EPA per play, the project's strongest single quality stat.",
       support=[("PPG", 'points_per_game', "{:.1f}"), ("YPP", 'off_yards_per_play', "{:.2f}"), ("Success", 'off_success_rate', "{:.1f}%")]),
  dict(title="Defense", key='def_efficiency_epa', higher_best=False, fmt="{:+.3f} EPA/play allowed",
       note="Ranked by defensive EPA per play allowed. Lower = better.",
       support=[("PA/G", 'points_allowed_per_game', "{:.1f}"), ("YPP allowed", 'def_yards_per_play', "{:.2f}"), ("Success allowed", 'def_success_rate', "{:.1f}%")]),
 ]),
 ("THE DAWGS / THE POUND: RUN GAME", [
  dict(title="Rushing Offense", key='off_rush_epa', higher_best=True, fmt="{:+.3f} EPA/rush",
       note="Ranked by rush EPA per play.",
       support=[("Yds/carry", 'off_rushing_yards_per_play', "{:.2f}"), ("RYOE", 'ryoe_per_attempt', "{:+.2f}"), ("Yds/G", 'rushing_yards_per_game', "{:.0f}")]),
  dict(title="Rushing Defense", key='def_rush_epa_allowed', higher_best=False, fmt="{:+.3f} EPA/rush allowed",
       note="Ranked by rush EPA allowed per play. Lower = better.",
       support=[("Yds/carry allowed", 'def_rushing_yards_per_play', "{:.2f}"), ("Run stop", 'def_run_stop_rate', "{:.1f}%"), ("Yds/G allowed", 'rushing_yards_allowed_per_game', "{:.0f}")]),
  dict(title="Yards Before Contact (proxy: Line Yards)", key='off_line_yards', higher_best=True, fmt="{:.2f} line yds/carry",
       note="True YBC is not tracked in this project. Adjusted Line Yards is the closest stat: it credits the offensive line for the early part of each run.",
       support=[("Yds/carry", 'off_rushing_yards_per_play', "{:.2f}"), ("Stuff rate", 'stuff_rate_offense', "{:.1f}%"), ("Run success", 'off_run_success_rate', "{:.1f}%")]),
 ]),
 ("THE FREEBIRDS / THE DIRTY BIRD: PASS GAME", [
  dict(title="Passing Offense", key='off_yards_per_pass', higher_best=True, fmt="{:.2f} yds/pass play",
       note="Ranked by passing yards per play.",
       support=[("CPOE", 'cpoe', "{:+.1f}"), ("Rating", 'open_source_passer_rating', "{:.1f}"), ("Pass Yds/G", 'passing_yards_per_game', "{:.0f}")]),
  dict(title="Passing Defense", key='def_yards_per_pass', higher_best=False, fmt="{:.2f} yds/pass play allowed",
       note="Ranked by passing yards allowed per play. Lower = better.",
       support=[("Comp% allowed", 'completion_pct_allowed', "{:.1f}%"), ("Pass Yds/G allowed", 'passing_yards_allowed_per_game', "{:.0f}"), ("INT/G", 'interceptions_gained_per_game', "{:.2f}")]),
  dict(title="Yards After Catch (YAC)", key='yac_per_reception_pbp', higher_best=True, fmt="{:.2f} YAC/rec",
       note="Play-by-play YAC per reception, the same source as the delta sheet.",
       support=[("Yds/completion", 'yards_per_completion', "{:.1f}"), ("Separation", 'target_separation', "{:.2f} yds"), ("Catch%", 'catch_pct', "{:.1f}%")]),
 ]),
 ("PASS RUSH &amp; SECONDARY", [
  dict(title="Edge Rushers", key='edge_pressure_rate', higher_best=True, fmt="{:.3f} edge pressure rate",
       note="Edge pressure rate generated by the defense.",
       support=[("Pressure forced", 'pressure_rate_forced_pct', "{:.1f}%"), ("Sacks", 'sacks_made', "{:.0f}"), ("Sack%", 'sack_pct_defense', "{:.1f}%")]),
  dict(title="Up-the-Middle Pressure (Tug o' War)", key='interior_pressure_rate', higher_best=True, fmt="{:.3f} interior pressure rate",
       note="Interior pressure rate generated by the defense (DT/guard-center battle).",
       support=[("Pressure forced", 'pressure_rate_forced_pct', "{:.1f}%"), ("Sacks", 'sacks_made', "{:.0f}"), ("TFL", 'tfl_made', "{:.0f}")]),
  dict(title="Sacks: Pass Rush (getting to the QB)", key='sack_pct_defense', higher_best=True, fmt="{:.1f}% sack rate",
       note="Defense sack % of dropbacks. Best = most sacks per dropback, worst = can't finish the rush.",
       support=[("Sacks", 'sacks_made', "{:.0f}"), ("Pressure forced", 'pressure_rate_forced_pct', "{:.1f}%"), ("Edge", 'edge_pressure_rate', "{:.3f}")]),
  dict(title="Sacks: Protection (keeping the QB clean)", key='sack_pct_offense', higher_best=False, fmt="{:.1f}% sacked",
       note="Offense sack % taken. Lower = better protection; the worst side is the team giving up the most sacks.",
       support=[("Sacks allowed", 'sacks_allowed', "{:.0f}"), ("Pressure allowed", 'pressure_rate_allowed_pct', "{:.1f}%"), ("PBE", 'pbe_score', "{:.1f}")]),
  dict(title="Defensive Backs", key='secondary_lock_index', higher_best=True, fmt="{:.2f} Lock Index",
       note="Secondary Lock Index, the project's blended coverage grade.",
       support=[("Catch% allowed", 'catch_pct_allowed', "{:.1f}%"), ("Yds/pass allowed", 'def_yards_per_pass', "{:.2f}"), ("Sep. allowed", 'target_separation_allowed', "{:.2f} yds")]),
 ]),
 ("MATCHUP LEAKS: TIGHT ENDS &amp; MOBILE QBs", [
  dict(title="Tight Ends: Fantasy Points Allowed", key='te_fpts', higher_best=False, fmt="{:.1f} PPR pts/G to TEs",
       note="Yardage allowed to TEs is not tracked in this project; PPR fantasy points allowed to the position is the closest stat. Best = stingiest vs TEs, worst = gives up the most.",
       support=[("Pass Yds/G allowed", 'passing_yards_allowed_per_game', "{:.0f}"), ("Comp% allowed", 'completion_pct_allowed', "{:.1f}%"), ("Sep. allowed", 'target_separation_allowed', "{:.2f} yds")]),
  dict(title="Covering Mobile QBs", key='mob_epa', higher_best=False, fmt="{:+.3f} EPA/play allowed",
       note="Not tracked directly. Built from game logs: each defense's EPA/play allowed ONLY in games vs a team the matrix tags as a Mobile QB. Just 1-3 games per team, and 5 teams have not faced one yet, so treat as directional.",
       support=[("Games", 'mob_n', "{:.0f}"), ("Pts allowed", 'mob_pa', "{:.1f}"), ("Pass Yds allowed", 'mob_pyd', "{:.0f}")]),
 ]),
 ("RED ZONE", [
  dict(title="Red Zone Offense", key='redzone_efficiency', higher_best=True, fmt="{:.1f}% efficiency",
       note="Red zone scoring efficiency. Only ~3 games of trips so far, so small samples swing this hard.",
       support=[("PPG", 'points_per_game', "{:.1f}"), ("PPD", 'points_per_drive', "{:.2f}")]),
  dict(title="Red Zone Defense (bonus)", key='redzone_efficiency_allowed', higher_best=False, fmt="{:.1f}% allowed",
       note="Red zone efficiency allowed. Lower = better. Same small-sample warning.",
       support=[("PA/G", 'points_allowed_per_game', "{:.1f}"), ("PPD allowed", 'points_per_drive_allowed', "{:.2f}")]),
 ]),
]


def latest_week_tag(track):
    """Latest week in the model-track CSV (week_tag like 'week5')."""
    best, best_n = None, -1
    for tg in track['week_tag'].dropna().unique():
        mm = re.search(r'(\d+)', str(tg))
        n = int(mm.group(1)) if mm else -1
        if n > best_n:
            best, best_n = tg, n
    return best, best_n


def third_down_cards(df):
    """3rd-down cards, only if the matrix values look like real percentages.
    (Older data pulls divided by every play, giving 5-11% instead of 30-50%.)"""
    need = ['third_down_pct_offense', 'third_down_pct_defense']
    if not all(c in df.columns for c in need):
        return None
    med_o = df['third_down_pct_offense'].median()
    med_d = df['third_down_pct_defense'].median()
    if not (med_o > 20 and med_d > 20):
        return None
    sup_o = [("PPG", 'points_per_game', "{:.1f}")]
    sup_d = [("PA/G", 'points_allowed_per_game', "{:.1f}")]
    if 'avg_3rd_down_distance_offense' in df.columns:
        sup_o.append(("Avg dist", 'avg_3rd_down_distance_offense', "{:.1f} yds"))
    if 'avg_3rd_down_distance_defense' in df.columns:
        sup_d.append(("Avg dist faced", 'avg_3rd_down_distance_defense', "{:.1f} yds"))
    return ("3RD DOWN", [
        dict(title="3rd Down Offense", key='third_down_pct_offense', higher_best=True, fmt="{:.1f}% converted",
             note="Share of 3rd-down plays that moved the chains.", support=sup_o),
        dict(title="3rd Down Defense", key='third_down_pct_defense', higher_best=False, fmt="{:.1f}% allowed",
             note="Share of opponent 3rd downs converted. Lower = better.", support=sup_d)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--week', help="week tag to build for (default: latest in nfl_internal_model_track.csv)")
    args = ap.parse_args()

    df = pd.read_csv(HERE / 'nfl_team_matrix.csv')
    df['total_power'] = df['off_power_rating'] + df['def_power_rating']
    d = pd.read_csv(HERE / 'nfl_def_vs_position.csv'); g = pd.read_csv(HERE / 'nfl_game_log.csv')
    te = d[d.position == 'TE'].set_index('team')['fpts_allowed_pg']
    df = df.copy(); df['te_fpts'] = df['team'].map(te)
    tag = df.set_index('team')['qb_mobility_tag']
    g['opp_mobile'] = g['opponent'].map(tag) == 'Mobile'
    m = g[g.opp_mobile].groupby('team').agg(mob_n=('game_id', 'count'), mob_epa=('def_epa_play_allowed_game', 'mean'),
                                           mob_pa=('pts_against', 'mean'), mob_pyd=('pass_yards_allowed', 'mean'))
    df = df.join(m, on='team')
    df = df.set_index('team', drop=False)
    players = pd.read_csv(HERE / 'nfl_player_stats.csv'); inj = pd.read_csv(HERE / 'nfl_injury_report.csv')
    track = pd.read_csv(HERE / 'nfl_internal_model_track.csv')

    week_tag = args.week
    if week_tag:
        mm = re.search(r'(\d+)', week_tag); week_n = int(mm.group(1)) if mm else 0
    else:
        week_tag, week_n = latest_week_tag(track)
    if not week_tag:
        print("No week found in nfl_internal_model_track.csv - nothing to build."); return 1
    wk_label = f"Week {week_n}" if week_n else week_tag
    gp = int(round(df['gp'].median())) if 'gp' in df.columns else week_n - 1
    n_teams = len(df)

    sec_defs = [list(s) for s in SEC_DEFS]
    td = third_down_cards(df)
    if td:
        sec_defs.append(list(td))

    # drop any card whose columns are missing instead of crashing
    clean = []
    for name, cats in sec_defs:
        ok = []
        for c in cats:
            if c['key'] not in df.columns or df[c['key']].notna().sum() < 4:
                print(f"  skipped card (no data): {c['title']}"); continue
            c = dict(c); c['support'] = [s for s in c['support'] if s[1] in df.columns]
            ok.append(c)
        if ok: clean.append((name, ok))
    sec_defs = clean
    # make module-level df visible to the card helpers
    globals()['df'] = df

    GEN_DIR.mkdir(exist_ok=True)
    out_path = GEN_DIR / f"NFL_League_Superlatives_{week_tag}.pdf"
    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            leftMargin=0.7*inch, rightMargin=0.7*inch, topMargin=0.6*inch, bottomMargin=0.6*inch,
                            title=f"NFL League Superlatives - {wk_label}")
    els = [Paragraph("NFL League Superlatives", H1),
           Paragraph(f"{wk_label}. Best and worst team in every category, grouped by where it lives on the sheets. Top 3 and bottom 3 shown, "
                     f"with the #1 team's supporting numbers. Built from this season's real team data, {n_teams} teams, about {gp} games played each.", SUB),
           Spacer(1, 6)]
    for name, cats in sec_defs:
        first = True
        for c in cats:
            if first:
                els.append(KeepTogether(section(name) + card(c).__dict__['_content']))
                first = False
            else:
                els.append(card(c))

    short = {}
    for sec, cats in sec_defs:
        for c in cats:
            ss = df[df[c['key']].notna()].sort_values(c['key'], ascending=not c['higher_best'])
            nm = c['title'].split(' (')[0]
            short.setdefault(ss.index[0], []).append(f"best {nm}")
            short.setdefault(ss.index[-1], []).append(f"worst {nm}")

    gels, ng = build_games(df, players, inj, track, short, week_tag)
    n_week = int((track.week_tag == week_tag).sum())
    els += section(f"{wk_label.upper()} GAMES BETWEEN HEADLINE TEAMS: PLAYER PREDICTIONS")
    if ng:
        els.append(Paragraph(f"{ng} of the {n_week} {wk_label} games have two teams that each hold a #1 best or worst spot in a category above. "
                             "Projections for each follow, one page per game.", NOTE))
    else:
        els.append(Paragraph(f"None of the {n_week} {wk_label} games has two teams that each hold a #1 best or worst spot this week.", NOTE))
    els += gels
    els.append(PageBreak())
    els += section("NOT RANKED: DATA NOT IN THE PROJECT YET")
    els.append(Paragraph(
        "<b>Passing yards allowed at 0-5 and 10-20 yards:</b> the project does not store pass depth (air yards / pass length) per defense, so these "
        "are not ranked without inventing numbers. Adding them means pulling air_yards from the play-by-play in nfl_data_pull.py and bucketing "
        "by depth.", NOTE))
    if not td:
        els.append(Spacer(1, 6))
        els += section("3RD DOWN SUCCESS: HELD BACK")
        els.append(Paragraph(
            "<b>Best/worst 3rd-down offense and defense are not ranked here on purpose.</b> The 3rd-down % in nfl_team_matrix.csv currently looks wrong "
            "(values run about 5% to 11%, while real NFL 3rd-down rates sit near 30% to 50%): the conversion flag is divided by every play, not just "
            "3rd-down attempts. This report turns the 3rd-down cards on automatically once the matrix values look like real percentages. "
            "The fix is in nfl_data_pull.py (restrict to down == 3 before the average).", NOTE))
    doc.build(els)
    print(f"wrote {out_path}  ({ng} headline game page(s), {wk_label})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
