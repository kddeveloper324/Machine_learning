import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings

# --- Suppress Warnings ---
# This prevents sklearn version warnings from appearing in the console
warnings.filterwarnings("ignore")

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

/* Add semi-transparent overlay for better text readability */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.35);
    pointer-events: none;
    z-index: -1;
}

/* Main headings - White with subtle shadow for readability */
h1, h2, h3 {
    color: #ffffff !important;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7) !important;
    font-weight: 700 !important;
}

/* Paragraph text - Readable white */
p, label {
    color: #f0f0f0 !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5) !important;
    font-size: 16px !important;
}

/* Subheader text */
.stSubheader {
    color: #ffffff !important;
}

/* --- DROPDOWN (SELECTBOX) STYLING FIXES --- */

/* 1. Reset background for the main selectbox container */
.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border-radius: 8px !important;
    border: 2px solid #667eea !important;
}

/* 2. FORCE BLACK TEXT for the selected value container */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* 3. Deep selector to ensure the specific text node is black (overriding global white text) */
.stSelectbox div[data-baseweb="select"] div {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important; /* Webkit override */
}

/* 4. Fix the SVG icon (arrow) color */
.stSelectbox div[data-baseweb="select"] svg {
    fill: #000000 !important;
    color: #000000 !important;
}

/* Selectbox text when opened */
.stSelectbox [role="combobox"] {
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* Selectbox input text - make it bold and dark */
input[type="text"], 
div[contenteditable="true"],
.stSelectbox [role="button"] {
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* Dropdown menu options - Clear and readable */
div[data-baseweb="popover"] {
    background-color: #ffffff !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
    border: 2px solid #667eea !important;
}

div[data-baseweb="popover"] li, 
div[data-baseweb="popover"] div[role="option"],
div[role="listbox"] li,
li[data-baseweb="menu-item"] {
    color: #000000 !important;
    background-color: #ffffff !important;
    padding: 12px 14px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    text-shadow: none !important;
}

div[data-baseweb="popover"] li:hover,
li[data-baseweb="menu-item"]:hover {
    background-color: #e8f0ff !important;
    color: #667eea !important;
    font-weight: 700 !important;
}

/* Menu - Ensure options are readable */
div[data-baseweb="menu"] li {
    color: #000000 !important;
    background-color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 14px !important;
}

div[data-baseweb="menu"] li:hover {
    background-color: #e8f0ff !important;
    color: #667eea !important;
}

/* Radio button labels */
.stRadio label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8) !important;
}

.stRadio > div {
    color: #ffffff !important;
}

/* Radio button options - text visibility */
.stRadio span {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.6) !important;
}

/* Fix for Team Logos - Add a backing card so they pop */
div[data-testid="stImage"] img {
    background-color: rgba(255, 255, 255, 0.85) !important;
    padding: 10px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4) !important;
}

/* Style Expander Headers */
div[data-testid="stExpander"] {
    background-color: rgba(20, 20, 50, 0.8) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
}

div[data-testid="stExpander"] > button {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Success/Info/Warning boxes */
.stSuccess, .stInfo, .stWarning {
    background-color: rgba(50, 50, 50, 0.9) !important;
    color: #ffffff !important;
}

/* Better button styling */
.stButton > button {
    background-color: #667eea !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 10px 20px !important;
    border-radius: 6px !important;
}

.stButton > button:hover {
    background-color: #764ba2 !important;
}

/* Sidebar styling */
.stSidebar {
    background-color: rgba(30, 30, 60, 0.95) !important;
}

.stSidebar > div {
    background-color: rgba(30, 30, 60, 0.95) !important;
}

/* Text in sidebar */
.stSidebar p, .stSidebar label {
    color: #ffffff !important;
}

/* Divider styling */
.stHorizontalBlock > hr {
    background-color: rgba(255, 255, 255, 0.3) !important;
}

/* Better contrast for all text */
* {
    text-rendering: optimizeLegibility !important;
    -webkit-font-smoothing: antialiased !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Team Logo Mapping
TEAM_LOGOS = {
    'Chennai Super Kings': 'https://upload.wikimedia.org/wikipedia/en/thumb/2/2b/Chennai_Super_Kings_Logo.svg/500px-Chennai_Super_Kings_Logo.svg.png',
    'Gujarat Titans': 'https://upload.wikimedia.org/wikipedia/en/thumb/0/09/Gujarat_Titans_Logo.svg/500px-Gujarat_Titans_Logo.svg.png',
    'Kolkata Knight Riders': 'https://upload.wikimedia.org/wikipedia/en/thumb/4/4c/Kolkata_Knight_Riders_Logo.svg/500px-Kolkata_Knight_Riders_Logo.svg.png',
    'Lucknow Super Giants': 'https://upload.wikimedia.org/wikipedia/en/thumb/a/a9/Lucknow_Super_Giants_IPL_Logo.svg/500px-Lucknow_Super_Giants_IPL_Logo.svg.png',
    'Mumbai Indians': 'https://upload.wikimedia.org/wikipedia/en/thumb/c/cd/Mumbai_Indians_Logo.svg/500px-Mumbai_Indians_Logo.svg.png',
    'Punjab Kings': 'https://upload.wikimedia.org/wikipedia/en/thumb/d/d4/Punjab_Kings_Logo.svg/500px-Punjab_Kings_Logo.svg.png',
    'Rajasthan Royals': 'https://upload.wikimedia.org/wikipedia/en/thumb/6/60/Rajasthan_Royals_Logo.svg/300px-Rajasthan_Royals_Logo.svg.png',
    'Delhi Capitals': 'https://upload.wikimedia.org/wikipedia/en/thumb/2/2f/Delhi_Capitals_Logo.svg/300px-Delhi_Capitals_Logo.svg.png',
    'Royal Challengers Bangalore': 'https://upload.wikimedia.org/wikipedia/en/thumb/4/4e/Royal_Challengers_Bengaluru_Logo.svg/300px-Royal_Challengers_Bengaluru_Logo.svg.png',
    'Sunrisers Hyderabad': 'https://upload.wikimedia.org/wikipedia/en/thumb/8/81/Sunrisers_Hyderabad.svg/300px-Sunrisers_Hyderabad.svg.png'
}

# Active Teams List
ACTIVE_TEAMS = [
    'Chennai Super Kings',
    
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

# --- 2. Data Loading ---
@st.cache_data
def load_data():
    try:
        matches = pd.read_csv('ipl_matches_data_cleaned.csv')
        teams = pd.read_csv('teams_data_cleaned.csv')
        players = pd.read_csv('players_data_cleaned.csv')
        deliveries = pd.read_csv('cleaned_ball_by_ball_data.csv') 
        
        # Preprocess Matches
        data = matches.dropna(subset=['match_winner'])
        data = data[data['result'] != 'no result']
        
        return data, teams, players, deliveries
    except FileNotFoundError as e:
        return None, None, None, None

# --- 3. Load Pre-trained Model ---
@st.cache_resource
def load_trained_model():
    """Load the pre-trained model and encoders from disk"""
    try:
        # Try multiple possible locations for the model files
        possible_paths = [
            'trained_models',  # Local directory (expected)
            './trained_models',  # Current directory
            '../trained_models',  # Parent directory
            os.path.expanduser('~/trained_models'),  # Home directory
        ]
        
        models_dir = None
        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'ipl_model.pkl')):
                models_dir = path
                break
        
        if models_dir is None:
            st.error("❌ **Model files not found!**")
            st.info("Please ensure `trained_models/` folder exists with `ipl_model.pkl`, `ipl_encoders.pkl`, and `model_metadata.pkl` files.")
            return None, None, None
        
        with open(os.path.join(models_dir, 'ipl_model.pkl'), 'rb') as f:
            model = pickle.load(f)
            
        # Try to suppress verbosity in the loaded model if possible
        if hasattr(model, 'verbose'):
            model.verbose = 0
        
        with open(os.path.join(models_dir, 'ipl_encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
        
        with open(os.path.join(models_dir, 'model_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        return model, encoders, metadata
    except FileNotFoundError as e:
        st.error(f"❌ **Model Loading Error:** {str(e)}")
        st.info("To fix this:\n1. Run `train_and_save_model.py` to generate model files\n2. Create `trained_models/` folder in the app directory\n3. Ensure files are: `ipl_model.pkl`, `ipl_encoders.pkl`, `model_metadata.pkl`")
        return None, None, None
    except Exception as e:
        st.error(f"❌ **Unexpected Error:** {str(e)}")
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
    st.error("❌ **Failed to Load Model**")
    st.markdown("""
    ### How to Fix This:
    
    **Option 1: Local Setup**
    1. Ensure you're running this app in the `Machine_learning` directory
    2. Run `train_and_save_model.py` to generate model files
    3. Check that `trained_models/` folder contains:
       - `ipl_model.pkl`
       - `ipl_encoders.pkl`
       - `model_metadata.pkl`
    
    **Option 2: Streamlit Cloud**
    1. Push the `trained_models/` folder to your GitHub repository
    2. Make sure `.gitignore` doesn't exclude `.pkl` files
    3. Or upload model files to GitHub Releases and download them
    
    **Option 3: Check Current Directory**
    """)
    
    # Show current working directory for debugging
    st.code(f"Current directory: {os.getcwd()}", language="python")
    st.code(f"Files in current dir: {os.listdir('.')[:20]}", language="python")
    
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
    st.markdown("<p style='color: #ffffff; font-weight: 700; text-shadow: 1px 1px 3px #000; font-size: 16px;'>🏠 Home Team</p>", unsafe_allow_html=True)
    # CHANGED: Default index set to 5 for Mumbai Indians
    default_index_1 = 5
    team1 = st.selectbox("Home Team", ACTIVE_TEAMS, index=default_index_1, label_visibility="collapsed", key="team1_select")
    
    c1_logo, c1_text = st.columns([1, 3])
    with c1_logo:
        if team1 in TEAM_LOGOS:
            st.image(TEAM_LOGOS[team1], width=80)
    with c1_text:
        st.markdown(f"<h3 style='color: #ffffff; text-shadow: 1px 1px 3px #000;'>{team1}</h3>", unsafe_allow_html=True)

with col_vs:
    st.markdown("<h1 style='text-align: center; padding-top: 20px; color: #ffffff; text-shadow: 2px 2px 5px #000;'>VS</h1>", unsafe_allow_html=True)

with col2:
    st.markdown("<p style='color: #ffffff; font-weight: 700; text-shadow: 1px 1px 3px #000; font-size: 16px;'>🏟️ Away Team</p>", unsafe_allow_html=True)
    default_index_2 = 1 if len(ACTIVE_TEAMS) > 1 else 0
    team2 = st.selectbox("Away Team", ACTIVE_TEAMS, index=default_index_2, label_visibility="collapsed", key="team2_select")
    
    c2_logo, c2_text = st.columns([1, 3])
    with c2_logo:
        if team2 in TEAM_LOGOS:
            st.image(TEAM_LOGOS[team2], width=80)
    with c2_text:
        st.markdown(f"<h3 style='color: #ffffff; text-shadow: 1px 1px 3px #000;'>{team2}</h3>", unsafe_allow_html=True)

st.divider()

# --- Match Conditions ---
st.subheader("Match Conditions")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<p style='color: #ffffff; font-weight: 700; text-shadow: 1px 1px 3px #000; font-size: 15px;'>🏟️ Match Venue</p>", unsafe_allow_html=True)
    venue_list = sorted(list(encoders['venue'].classes_))
    venue = st.selectbox("Match Venue", venue_list, label_visibility="collapsed", key="venue_select")

with c2:
    st.markdown("<p style='color: #ffffff; font-weight: 700; text-shadow: 1px 1px 3px #000; font-size: 15px;'>🪙 Toss Winner</p>", unsafe_allow_html=True)
    toss_list = [team1, team2]
    toss_winner = st.selectbox("Toss Winner", toss_list, label_visibility="collapsed", key="toss_select")

with c3:
    st.markdown("<p style='color: #ffffff; font-weight: 700; text-shadow: 1px 1px 3px #000; font-size: 15px;'>⚔️ Toss Decision</p>", unsafe_allow_html=True)
    toss_decision = st.radio("Toss Decision", ["bat", "field"], label_visibility="collapsed", key="decision_select")

# --- Prediction Button ---
if st.button("🔮 Predict Future Match Outcome", type="primary"):
    
    try:
        # Validate team selection
        if team1 == team2:
            st.error("❌ Please select different teams for home and away!")
        else:
            # Create a DataFrame for prediction with proper encoding
            input_data = pd.DataFrame({
                'team1': [encoders['team1'].transform([str(team1)])[0]],
                'team2': [encoders['team2'].transform([str(team2)])[0]],
                'venue': [encoders['venue'].transform([str(venue)])[0]],
                'toss_winner': [encoders['toss_winner'].transform([str(toss_winner)])[0]],
                'toss_decision': [encoders['toss_decision'].transform([str(toss_decision)])[0]]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            all_probs = model.predict_proba(input_data)[0]
            
            # Create a dictionary mapping Team Name -> Probability
            team_probs_dict = dict(zip(encoders['match_winner'].classes_, all_probs))
            
            # Decode prediction
            predicted_winner = encoders['match_winner'].inverse_transform([prediction])[0]
            winner_prob = team_probs_dict.get(predicted_winner, 0.5)
            
            # Head-to-Head Stats
            wins_t1, wins_t2 = get_head_to_head_stats(data, team1, team2)

            # --- Display Prediction ---
            st.markdown("### 🏆 Prediction Report")
            
            with st.container():
                r_col1, r_col2 = st.columns([1, 4])
                
                with r_col1:
                    if predicted_winner in TEAM_LOGOS:
                        st.image(TEAM_LOGOS[predicted_winner], width=100)
                    else:
                        st.write("🏆")
                
                with r_col2:
                    st.success(f"## 🏆 Predicted Winner: {predicted_winner}")
                    
                    st.markdown(f"<p style='color: #ffffff; font-size: 16px; font-weight: 600;'>**Model Confidence:** {winner_prob:.1%}</p>", unsafe_allow_html=True)
                    st.progress(int(winner_prob * 100))
                    
                    if winner_prob < 0.55:
                        st.warning(f"⚠️ **This is a close call!** The model is only {winner_prob:.1%} sure.")
                    elif winner_prob > 0.75:
                        st.info(f"✅ **Strong Favorite:** Historical data heavily favors {predicted_winner}.")
                    
                    st.markdown(f"<p style='color: #ffffff; font-size: 15px;'><b>Head-to-Head History:</b> {team1} ({wins_t1} wins) - {team2} ({wins_t2} wins)</p>", unsafe_allow_html=True)

            # --- Impact Analysis ---
            st.subheader("🧠 Key Players & Team Strength")
            st.markdown("<p style='color: #d0d0d0; font-size: 14px;'>These players have historically been the top performers for their teams.</p>", unsafe_allow_html=True)
            
            with st.expander(f"View {team1} Key Players", expanded=True):
                p_col1, p_col2 = st.columns([1, 4])
                if team1 in TEAM_LOGOS:
                    p_col1.image(TEAM_LOGOS[team1], width=60)
                batters, bowlers = get_top_players(team1, deliveries_df)
                if batters:
                    p_col2.markdown(f"<p style='color: #ffffff; font-weight: 600;'>🏏 Batting Core:</p><p style='color: #e0e0e0;'>{', '.join(batters)}</p>", unsafe_allow_html=True)
                    p_col2.markdown(f"<p style='color: #ffffff; font-weight: 600;'>⚾ Bowling Core:</p><p style='color: #e0e0e0;'>{', '.join(bowlers)}</p>", unsafe_allow_html=True)
                else:
                    p_col2.markdown("<p style='color: #b0b0b0;'>No historical player data available.</p>", unsafe_allow_html=True)

            with st.expander(f"View {team2} Key Players", expanded=True):
                p_col1, p_col2 = st.columns([1, 4])
                if team2 in TEAM_LOGOS:
                    p_col1.image(TEAM_LOGOS[team2], width=60)
                batters, bowlers = get_top_players(team2, deliveries_df)
                if batters:
                    p_col2.markdown(f"<p style='color: #ffffff; font-weight: 600;'>🏏 Batting Core:</p><p style='color: #e0e0e0;'>{', '.join(batters)}</p>", unsafe_allow_html=True)
                    p_col2.markdown(f"<p style='color: #ffffff; font-weight: 600;'>⚾ Bowling Core:</p><p style='color: #e0e0e0;'>{', '.join(bowlers)}</p>", unsafe_allow_html=True)
                else:
                    p_col2.markdown("<p style='color: #b0b0b0;'>No historical player data available.</p>", unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"❌ **Encoding Error:** {str(e)}")
        st.info("Please ensure all values are properly selected from the dropdowns.")
    except Exception as e:
        st.error(f"❌ **Prediction Error:** {str(e)}")