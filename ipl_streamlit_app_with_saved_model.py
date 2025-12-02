import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- 1. Configuration & Assets ---
st.set_page_config(page_title="IPL Match Prediction", page_icon="🏏", layout="wide")

# Add Custom CSS for Background Image & Better Component Visibility
page_bg_img = """
<style>
/* Background Image */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?q=80&w=2667&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* General Text Visibility - Targeted to avoid breaking widgets */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
    color: #ffffff !important;
    text-shadow: 2px 2px 4px #000000;
}

/* Fix for Dropdown (Selectbox) Input Field */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(255, 255, 255, 0.95) !important;
    color: black !important;
    border-radius: 5px;
}

/* Fix for Dropdown Menu Options (The list that pops up) */
div[data-baseweb="popover"] {
    background-color: white !important;
}
div[data-baseweb="popover"] div, div[data-baseweb="popover"] span, div[data-baseweb="menu"] li {
    color: black !important;
    text-shadow: none !important; /* Remove shadow for clean read */
}

/* Fix for Team Logos - Add a backing card so they pop */
div[data-testid="stImage"] img {
    background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white backing */
    padding: 10px;
    border-radius: 12px; /* Rounded corners */
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); /* Subtle shadow for depth */
}

/* Style Expander Headers */
div[data-testid="stExpander"] {
    background-color: rgba(0, 0, 0, 0.6);
    border-radius: 10px;
    color: white;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Team Logo Mapping
TEAM_LOGOS = {
    'Chennai Super Kings': 'https://upload.wikimedia.org/wikipedia/en/thumb/2/2b/Chennai_Super_Kings_Logo.svg/500px-Chennai_Super_Kings_Logo.svg.png',
    'Delhi Capitals': 'https://upload.wikimedia.org/wikipedia/en/thumb/2/2f/Delhi_Capitals_Logo.svg/300px-Delhi_Capitals_Logo.svg.png',
    'Gujarat Titans': 'https://upload.wikimedia.org/wikipedia/en/thumb/0/09/Gujarat_Titans_Logo.svg/500px-Gujarat_Titans_Logo.svg.png',
    'Kolkata Knight Riders': 'https://upload.wikimedia.org/wikipedia/en/thumb/4/4c/Kolkata_Knight_Riders_Logo.svg/500px-Kolkata_Knight_Riders_Logo.svg.png',
    'Lucknow Super Giants': 'https://upload.wikimedia.org/wikipedia/en/thumb/a/a9/Lucknow_Super_Giants_IPL_Logo.svg/500px-Lucknow_Super_Giants_IPL_Logo.svg.png',
    'Mumbai Indians': 'https://upload.wikimedia.org/wikipedia/en/thumb/c/cd/Mumbai_Indians_Logo.svg/500px-Mumbai_Indians_Logo.svg.png',
    'Punjab Kings': 'https://upload.wikimedia.org/wikipedia/en/thumb/d/d4/Punjab_Kings_Logo.svg/500px-Punjab_Kings_Logo.svg.png',
    'Rajasthan Royals': 'https://upload.wikimedia.org/wikipedia/en/thumb/6/60/Rajasthan_Royals_Logo.svg/300px-Rajasthan_Royals_Logo.svg.png',
    'Royal Challengers Bangalore': 'https://upload.wikimedia.org/wikipedia/en/thumb/4/4e/Royal_Challengers_Bengaluru_Logo.svg/300px-Royal_Challengers_Bengaluru_Logo.svg.png',
    'Sunrisers Hyderabad': 'https://upload.wikimedia.org/wikipedia/en/thumb/8/81/Sunrisers_Hyderabad.svg/300px-Sunrisers_Hyderabad.svg.png'
}

# Active Teams List
ACTIVE_TEAMS = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

# --- 2. Data Loading ---
@st.cache_data
def load_data():
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # List of files to load
        files = {
            'matches': 'ipl_matches_data_cleaned.csv',
            'teams': 'teams_data_cleaned.csv',
            'players': 'players_data_cleaned.csv',
            'deliveries': 'cleaned_ball_by_ball_data.csv'
        }
        
        # Try loading from script directory first
        data_dict = {}
        for key, filename in files.items():
            filepath = os.path.join(script_dir, filename)
            if os.path.exists(filepath):
                data_dict[key] = filepath
            else:
                # Try current working directory
                if os.path.exists(filename):
                    data_dict[key] = filename
                else:
                    st.warning(f"⚠️ File not found: {filename}")
                    data_dict[key] = None
        
        # Load data
        matches = pd.read_csv(data_dict['matches']) if data_dict['matches'] else None
        teams = pd.read_csv(data_dict['teams']) if data_dict['teams'] else None
        players = pd.read_csv(data_dict['players']) if data_dict['players'] else None
        deliveries = pd.read_csv(data_dict['deliveries']) if data_dict['deliveries'] else None
        
        if matches is None:
            return None, None, None, None
        
        # Preprocess Matches
        data = matches.dropna(subset=['match_winner'])
        data = data[data['result'] != 'no result']
        
        return data, teams, players, deliveries
    except FileNotFoundError as e:
        st.error(f"FileNotFoundError: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# --- 3. Load Pre-trained Model ---
@st.cache_resource
def load_trained_model():
    """Load the pre-trained model and encoders from disk"""
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'trained_models')
        
        # Try current directory if script_dir doesn't work
        if not os.path.exists(models_dir):
            models_dir = 'trained_models'
        
        # Debug: Check if directory exists
        if not os.path.exists(models_dir):
            st.error(f"⚠️ Model directory not found at: {models_dir}")
            st.write(f"Current working directory: {os.getcwd()}")
            st.write(f"Script directory: {script_dir}")
            st.write(f"Available directories: {os.listdir('.')}")
            return None, None, None
        
        model_path = os.path.join(models_dir, 'ipl_model.pkl')
        encoders_path = os.path.join(models_dir, 'ipl_encoders.pkl')
        metadata_path = os.path.join(models_dir, 'model_metadata.pkl')
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return model, encoders, metadata
    except FileNotFoundError as e:
        st.error(f"FileNotFoundError: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# --- 4. Helpers ---
def get_top_players(team_name, deliveries_df):
    team_batting = deliveries_df[deliveries_df['team_batting'] == team_name]
    if team_batting.empty:
        return [], []
        
    top_batters = team_batting.groupby('batter')['batter_runs'].sum().sort_values(ascending=False).head(3).index.tolist()
    
    team_bowling = deliveries_df[deliveries_df['team_bowling'] == team_name]
    valid_wickets = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
    wickets = team_bowling[team_bowling['wicket_kind'].isin(valid_wickets)]
    
    if wickets.empty:
        top_bowlers = []
    else:
        top_bowlers = wickets['bowler'].value_counts().head(3).index.tolist()
    
    return top_batters, top_bowlers

def get_head_to_head_stats(df, team_a, team_b):
    matches = df[((df['team1'] == team_a) & (df['team2'] == team_b)) | ((df['team1'] == team_b) & (df['team2'] == team_a))]
    wins = matches['match_winner'].value_counts()
    return wins.get(team_a, 0), wins.get(team_b, 0)

# --- 5. Main Application ---
data, teams_df, players_df, deliveries_df = load_data()

if data is None:
    st.error("Error: Could not load data files. Please ensure 'ipl_matches_data_cleaned.csv' and other data files are in the same directory.")
    st.stop()

# Load trained model
model, encoders, metadata = load_trained_model()

if model is None:
    st.error("Error: Could not load trained model. Please run 'train_and_save_model.py' first to train and save the model.")
    st.stop()

# --- Display Model Info ---
with st.sidebar:
    st.subheader("📊 Model Information")
    st.info(f"**Model Type:** {metadata['model_type']}\n\n**Accuracy:** {metadata['accuracy']*100:.2f}%\n\n**Status:** ✓ Ready for Prediction")

# --- NEW TITLE ---
st.title("🏆 IPL 2025: Ultimate Victory Predictor")

# --- Team Selection Section with Logos ---
st.subheader("Select Teams")

col1, col_vs, col2 = st.columns([4, 1, 4])

with col1:
    default_index_1 = ACTIVE_TEAMS.index('Mumbai Indians') if 'Mumbai Indians' in ACTIVE_TEAMS else 0
    team1 = st.selectbox("Home Team", ACTIVE_TEAMS, index=default_index_1)
    
    c1_logo, c1_text = st.columns([1, 3])
    with c1_logo:
        if team1 in TEAM_LOGOS:
            st.image(TEAM_LOGOS[team1], width=80)
    with c1_text:
        st.markdown(f"### {team1}")

with col_vs:
    st.markdown("<h1 style='text-align: center; padding-top: 20px;'>VS</h1>", unsafe_allow_html=True)

with col2:
    default_index_2 = ACTIVE_TEAMS.index('Chennai Super Kings') if 'Chennai Super Kings' in ACTIVE_TEAMS else 1
    team2 = st.selectbox("Away Team", ACTIVE_TEAMS, index=default_index_2)
    
    c2_logo, c2_text = st.columns([1, 3])
    with c2_logo:
        if team2 in TEAM_LOGOS:
             st.image(TEAM_LOGOS[team2], width=80)
    with c2_text:
        st.markdown(f"### {team2}")

st.divider()

# --- Match Conditions ---
st.subheader("Match Conditions")
c1, c2, c3 = st.columns(3)
with c1:
    venue = st.selectbox("Match Venue", encoders['venue'].classes_)
with c2:
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
with c3:
    toss_decision = st.radio("Toss Decision", ["field", "bat"])

# --- Prediction Button ---
if st.button("🔮 Predict Future Match Outcome", type="primary"):
    
    try:
        # Create a DataFrame for prediction
        input_data = pd.DataFrame({
            'team1': [encoders['team1'].transform([team1])[0]],
            'team2': [encoders['team2'].transform([team2])[0]],
            'venue': [encoders['venue'].transform([venue])[0]],
            'toss_winner': [encoders['toss_winner'].transform([toss_winner])[0]],
            'toss_decision': [encoders['toss_decision'].transform([toss_decision])[0]]
        })
        
        # Get prediction probabilities
        all_probs = model.predict_proba(input_data)[0]
        
        # Create a dictionary mapping Team Name -> Probability
        team_probs_dict = dict(zip(encoders['match_winner'].classes_, all_probs))
        
        # Extract raw probabilities for the two selected teams
        p1_raw = team_probs_dict.get(team1, 0.0)
        p2_raw = team_probs_dict.get(team2, 0.0)
        
        # Normalize probabilities so they sum to 100% between these two teams
        total_prob = p1_raw + p2_raw
        
        # Handle edge case where model predicts 0 for both (very unlikely)
        if total_prob == 0:
            winner_prob = 0.5
            predicted_winner = team1 # Default fallback
        else:
            p1_norm = p1_raw / total_prob
            p2_norm = p2_raw / total_prob
            
            if p1_norm >= p2_norm:
                predicted_winner = team1
                winner_prob = p1_norm
            else:
                predicted_winner = team2
                winner_prob = p2_norm
        
        # Head-to-Head Stats
        wins_t1, wins_t2 = get_head_to_head_stats(data, team1, team2)

        # --- Display Prediction ---
        st.markdown("### Prediction Report")
        
        with st.container():
            r_col1, r_col2 = st.columns([1, 4])
            
            with r_col1:
                 if predicted_winner in TEAM_LOGOS:
                    st.image(TEAM_LOGOS[predicted_winner], width=100)
                 else:
                    st.write("🏆")
            
            with r_col2:
                st.success(f"## 🏆 Predicted Winner: {predicted_winner}")
                
                st.write(f"**Model Confidence:** {winner_prob:.1%}")
                st.progress(int(winner_prob * 100))
                
                if winner_prob < 0.55:
                    st.warning(f"⚠️ **This is a close call!** The model is only {winner_prob:.1%} sure.")
                elif winner_prob > 0.75:
                    st.info(f"✅ **Strong Favorite:** Historical data heavily favors {predicted_winner}.")
                
                st.write(f"**Head-to-Head History:** {team1} ({wins_t1} wins) - {team2} ({wins_t2} wins)")

        # --- Impact Analysis ---
        st.subheader("🧠 Key Players & Team Strength")
        st.caption("These players have historically been the top performers for their teams.")
        
        with st.expander(f"View {team1} Key Players", expanded=True):
            p_col1, p_col2 = st.columns([1, 4])
            if team1 in TEAM_LOGOS:
                p_col1.image(TEAM_LOGOS[team1], width=60)
            batters, bowlers = get_top_players(team1, deliveries_df)
            if batters:
                p_col2.write(f"**🏏 Batting Core:** {', '.join(batters)}")
                p_col2.write(f"**⚾ Bowling Core:** {', '.join(bowlers)}")
            else:
                p_col2.write("No historical player data available.")

        with st.expander(f"View {team2} Key Players", expanded=True):
            p_col1, p_col2 = st.columns([1, 4])
            if team2 in TEAM_LOGOS:
                p_col1.image(TEAM_LOGOS[team2], width=60)
            batters, bowlers = get_top_players(team2, deliveries_df)
            if batters:
                p_col2.write(f"**🏏 Batting Core:** {', '.join(batters)}")
                p_col2.write(f"**⚾ Bowling Core:** {', '.join(bowlers)}")
            else:
                p_col2.write("No historical player data available.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
