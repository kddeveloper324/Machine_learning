import pandas as pd
import pickle
import joblib
import os

print("=" * 70)
print("IPL MATCH PREDICTION - LOADING MODEL & MAKING PREDICTIONS")
print("=" * 70)

# ===== 1. LOAD MODEL & ENCODERS =====
print("\n[STEP 1] Loading saved model and encoders...")

models_dir = 'trained_models'

try:
    # Load using pickle
    with open(os.path.join(models_dir, 'ipl_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(models_dir, 'ipl_encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    
    with open(os.path.join(models_dir, 'model_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print("✓ Model loaded successfully (pickle format)")
    print("✓ Encoders loaded successfully")
    print("✓ Metadata loaded successfully")
    
except FileNotFoundError as e:
    print(f"✗ Error loading model: {e}")
    print(f"  Please run 'train_and_save_model.py' first to train the model.")
    exit()

# ===== 2. DISPLAY MODEL INFO =====
print("\n[STEP 2] Model Information:")
print(f"  - Model Type: {metadata['model_type']}")
print(f"  - Accuracy on Test Set: {metadata['accuracy']*100:.2f}%")
print(f"  - Features: {metadata['features']}")
print(f"  - Number of Classes: {metadata['n_classes']}")
print(f"  - Classes: {metadata['classes']}")

# ===== 3. LOAD DATA FOR EXAMPLE =====
print("\n[STEP 3] Loading IPL data for example predictions...")
try:
    matches = pd.read_csv('ipl_matches_data_cleaned.csv')
    print(f"✓ Data loaded: {len(matches)} matches found")
except FileNotFoundError:
    print("✗ Could not load data file")
    exit()

# ===== 4. MAKE PREDICTIONS =====
print("\n[STEP 4] Making Predictions...")
print("=" * 70)

# Get sample matches
sample_size = min(5, len(matches))
sample_matches = matches.sample(n=sample_size, random_state=42)

features = metadata['features']

for idx, (i, row) in enumerate(sample_matches.iterrows(), 1):
    print(f"\n[PREDICTION {idx}]")
    print(f"  Team 1: {row['team1']} (Home)")
    print(f"  Team 2: {row['team2']} (Away)")
    print(f"  Venue: {row['venue']}")
    print(f"  Toss Winner: {row['toss_winner']}")
    print(f"  Toss Decision: {row['toss_decision']}")
    print(f"  Actual Winner: {row['match_winner']}")
    
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'team1': [encoders['team1'].transform([row['team1']])[0]],
            'team2': [encoders['team2'].transform([row['team2']])[0]],
            'venue': [encoders['venue'].transform([row['venue']])[0]],
            'toss_winner': [encoders['toss_winner'].transform([row['toss_winner']])[0]],
            'toss_decision': [encoders['toss_decision'].transform([row['toss_decision']])[0]]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        predicted_winner = encoders['match_winner'].inverse_transform([prediction])[0]
        
        print(f"  🔮 Predicted Winner: {predicted_winner}")
        print(f"  Confidence: {max(probabilities)*100:.2f}%")
        
        # Show all probabilities
        print(f"  Probability Distribution:")
        for team, prob in zip(encoders['match_winner'].classes_, probabilities):
            bar = "█" * int(prob * 20)
            print(f"    {team}: {prob*100:5.2f}% {bar}")
        
        # Check if prediction is correct
        if predicted_winner == row['match_winner']:
            print(f"  ✓ CORRECT PREDICTION!")
        else:
            print(f"  ✗ Wrong prediction (actual: {row['match_winner']})")
    
    except Exception as e:
        print(f"  ✗ Error making prediction: {e}")

print("\n" + "=" * 70)
print("PREDICTION EXAMPLES COMPLETED")
print("=" * 70)
print("\n✓ Model is ready for deployment!")
print("\nTo use the model in your Streamlit app:")
print("  1. Load model: model = pickle.load(open('trained_models/ipl_model.pkl', 'rb'))")
print("  2. Load encoders: encoders = pickle.load(open('trained_models/ipl_encoders.pkl', 'rb'))")
print("  3. Prepare input and predict")
print("=" * 70)
