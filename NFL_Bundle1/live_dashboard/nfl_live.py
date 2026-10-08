import streamlit as st
import nflreadpy as nfl
import pandas as pd
import time

st.set_page_config(layout="wide", page_title="Live NFL Dashboard")

# 1. Fetch official team metadata to extract team primary colors
@st.cache_data(ttl=86400) # Cache for 24 hours since team colors don't change
def get_team_colors():
    teams = nfl.load_teams().to_pandas()
    # Build a rapid lookup dictionary for team colors
    color_dict = {}
    for _, row in teams.iterrows():
        color_dict[row['team_abbr']] = {
            'primary': row['team_color'],
            'secondary': row['team_color2'] if pd.notna(row['team_color2']) else '#ffffff'
        }
    return color_dict

# 2. Fetch the active weekly schedule
@st.cache_data(ttl=300)
def get_live_games_list():
    sched = nfl.load_schedules().to_pandas()
    # Filter for the most recent active regular season week
    current_week = sched[sched['game_type'] == 'REG']['week'].max() 
    weekly_games = sched[sched['week'] == current_week].copy()
    weekly_games['game_label'] = weekly_games['away_team'] + " @ " + weekly_games['home_team']
    return weekly_games[['game_id', 'game_label', 'home_team', 'away_team']]

try:
    colors = get_team_colors()
    games_df = get_live_games_list()
    
    # 3. Game Selector Interface
    selected_game_label = st.selectbox("Select the game you are watching:", games_df['game_label'].tolist())
    
    # Extract structural constraints for the selected matchup
    match_info = games_df[games_df['game_label'] == selected_game_label].iloc[0]
    game_id = match_info['game_id']
    home_team = match_info['home_team']
    away_team = match_info['away_team']

    # Fetch dynamic colors for the chosen teams (Fallback to slate gray if missing)
    home_color = colors.get(home_team, {'primary': '#1e293b'})['primary']
    away_color = colors.get(away_team, {'primary': '#475569'})['primary']

    # Injecting Dynamic CSS based on Team Selections
    st.markdown(f"""
        <style>
        .home-box {{
            background-color: {home_color};
            padding: 15px;
            border-radius: 8px;
            color: white;
            text-shadow: 1px 1px 2px black;
        }}
        .away-box {{
            background-color: {away_color};
            padding: 15px;
            border-radius: 8px;
            color: white;
            text-shadow: 1px 1px 2px black;
        }}
        </style>
    """, unsafe_with_html=True)

    # 4. Fetch the real-time live play-by-play stream
    @st.cache_data(ttl=30)
    def get_live_game_data(target_game_id):
        pbp = nfl.load_pbp().to_pandas()
        return pbp[pbp['game_id'] == target_game_id].copy()

    df = get_live_game_data(game_id)
    valid_plays = df[(df['play_type'].isin(['pass', 'run'])) & (df['yards_gained'].notna())].copy()

    if valid_plays.empty:
        st.warning(f"Kickoff pending for {selected_game_label}. Awaiting active play transmissions...")
    else:
        # --- PART A: DYNAMIC COLOR-CODED BANNER METRICS ---
        team_stats = valid_plays.groupby('posteam').agg(
            epa=('epa', 'mean'),
            success=('success', 'mean'),
            pass_yds=('yards_gained', lambda x: valid_plays.loc[x.index, 'yards_gained'][valid_plays['play_type'] == 'pass'].sum()),
            rush_yds=('yards_gained', lambda x: valid_plays.loc[x.index, 'yards_gained'][valid_plays['play_type'] == 'run'].sum())
        ).reset_index()

        col1, col2 = st.columns(2)
        
        # Home Team Banner Block
        with col1:
            home_data = team_stats[team_stats['posteam'] == home_team]
            if not home_data.empty:
                row = home_data.iloc[0]
                st.markdown(f"""
                    <div class='home-box'>
                        <h2>🏠 {home_team} Box Score</h2>
                        <p><b>EPA / Play:</b> {row['epa']:.2f}</p>
                        <p><b>Success Rate:</b> {row['success']*100:.1f}%</p>
                        <p><b>Total Yards:</b> {int(row['pass_yds'] + row['rush_yds'])} Yds (Pass: {int(row['pass_yds'])} / Rush: {int(row['rush_yds'])})</p>
                    </div>
                """, unsafe_with_html=True)

        # Away Team Banner Block
        with col2:
            away_data = team_stats[team_stats['posteam'] == away_team]
            if not away_data.empty:
                row = away_data.iloc[0]
                st.markdown(f"""
                    <div class='away-box'>
                        <h2>✈️ {away_team} Box Score</h2>
                        <p><b>EPA / Play:</b> {row['epa']:.2f}</p>
                        <p><b>Success Rate:</b> {row['success']*100:.1f}%</p>
                        <p><b>Total Yards:</b> {int(row['pass_yds'] + row['rush_yds'])} Yds (Pass: {int(row['pass_yds'])} / Rush: {int(row['rush_yds'])})</p>
                    </div>
                """, unsafe_with_html=True)

        # --- PART B: QUARTER-BY-QUARTER TRENDING ANALYSIS ---
        st.subheader("📊 Operational Trends (Quarter-by-Quarter Efficiency)")
        
        # Group metrics cleanly by both possessing team and game quarter
        qtr_trends = valid_plays.groupby(['qtr', 'posteam']).agg(
            epa_per_play=('epa', 'mean'),
            success_rate=('success', 'mean'),
            total_plays=('play_id', 'count')
        ).reset_index()

        # Format layout into 4 small column frames (one per quarter)
        q_cols = st.columns(4)
        for quarter in [1, 2, 3, 4]:
            with q_cols[quarter-1]:
                st.markdown(f"**Quarter {quarter}**")
                q_data = qtr_trends[qtr_trends['qtr'] == quarter]
                
                if q_data.empty:
                    st.caption("No data logged.")
                else:
                    for _, row in q_data.iterrows():
                        st.write(f"**{row['posteam']}** ({int(row['total_plays'])} plays)")
                        st.write(f"└ EPA: `{row['epa_per_play']:.2f}` | SR: `{row['success_rate']*100:.0f}%`")

except Exception as e:
    st.error(f"Error executing visual layout engines: {e}")

# Live UI Rerun Execution Loop
time.sleep(30)
st.rerun()
