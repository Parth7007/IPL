import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and encoders
model = joblib.load('xgboost_ipl_model.pkl')
le_batting = joblib.load('batting_team_encoder.pkl')
le_bowling = joblib.load('bowling_team_encoder.pkl')
le_venue = joblib.load('venue_encoder.pkl')

# Load data to get precomputed averages
df = pd.read_csv('ipl_cleaned.csv')

# Get unique team and venue names from encoders
batting_teams = le_batting.classes_.tolist()
bowling_teams = le_bowling.classes_.tolist()
venues = le_venue.classes_.tolist()

# Precompute team and venue averages
team_avg_score = df.groupby('batting_team_encoded')['final_score'].mean().to_dict()
bowling_avg_conceded = df.groupby('bowling_team_encoded')['final_score'].mean().to_dict()
venue_avg_score = df.groupby('venue_encoded')['final_score'].mean().to_dict()

# Streamlit app
st.title("IPL Final Score Predictor")
st.write("Enter match details to predict the final score!")

# User inputs with real names
batting_team = st.selectbox("Batting Team", options=batting_teams)
bowling_team = st.selectbox("Bowling Team", options=bowling_teams)
venue = st.selectbox("Venue", options=venues)
inning = st.selectbox("Inning", options=[1, 2])
over = st.slider("Overs Completed", min_value=0, max_value=19, value=10)
ball = st.slider("Balls in Current Over", min_value=0, max_value=5, value=0)
current_runs = st.number_input("Current Runs", min_value=0, value=100)
current_wickets = st.number_input("Current Wickets", min_value=0, max_value=10, value=3)

# Calculate derived features
balls_bowled = over * 6 + ball
balls_remaining = 120 - balls_bowled
current_run_rate = current_runs / (balls_bowled / 6) if balls_bowled > 0 else 0
runs_total = 0  # Placeholder, not used in prediction but included for consistency

# Encode user inputs
batting_team_encoded = le_batting.transform([batting_team])[0]
bowling_team_encoded = le_bowling.transform([bowling_team])[0]
venue_encoded = le_venue.transform([venue])[0]

# Prepare input for prediction (match model features)
input_data = pd.DataFrame({
    'inning': [inning],
    'balls_bowled': [balls_bowled],
    'runs_total': [runs_total],  # Not critical for prediction, but included
    'current_runs': [current_runs],
    'current_wickets': [current_wickets],
    'batting_team_encoded': [batting_team_encoded],
    'bowling_team_encoded': [bowling_team_encoded],
    'venue_encoded': [venue_encoded],
    'balls_remaining': [balls_remaining],
    'current_run_rate': [current_run_rate],
    'team_avg_score': [team_avg_score[batting_team_encoded]],
    'bowling_avg_conceded': [bowling_avg_conceded[bowling_team_encoded]],
    'venue_avg_score': [venue_avg_score[venue_encoded]]
})

# Predict
if st.button("Predict Final Score"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Final Score: **{prediction:.0f} runs**")
    st.write(f"Note: Prediction has an average error of ~24 runs based on model performance.")

# Display feature importance (optional)
if st.checkbox("Show Feature Importance"):
    importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.write(importance)