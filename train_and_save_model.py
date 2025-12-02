import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# ===== 1. DATA LOADING =====
print("=" * 70)
print("IPL MATCH PREDICTION - MODEL TRAINING & SAVING")
print("=" * 70)

print("\n[STEP 1] Loading data files...")
try:
    matches = pd.read_csv('ipl_matches_data_cleaned.csv')
    teams = pd.read_csv('teams_data_cleaned.csv')
    players = pd.read_csv('players_data_cleaned.csv')
    deliveries = pd.read_csv('cleaned_ball_by_ball_data.csv')
    print("✓ All data files loaded successfully")
except FileNotFoundError as e:
    print(f"✗ Error loading data: {e}")
    exit()

# ===== 2. DATA PREPROCESSING =====
print("\n[STEP 2] Preprocessing data...")
data = matches.dropna(subset=['match_winner'])
data = data[data['result'] != 'no result']
print(f"✓ Data preprocessed: {len(data)} valid matches found")
print(f"  - Removed null values and 'no result' matches")

# ===== 3. FEATURE ENGINEERING =====
print("\n[STEP 3] Feature Engineering...")
features = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
target = 'match_winner'

encoders = {}
print(f"  - Features: {features}")
print(f"  - Target: {target}")

# Comprehensive encoding to handle all potential labels
for col in features + [target]:
    le = LabelEncoder()
    if col in ['team1', 'team2', 'toss_winner', 'match_winner']:
        all_teams = pd.concat([data['team1'], data['team2'], data['toss_winner'], data['match_winner']]).unique()
        le.fit(all_teams.astype(str))
    else:
        le.fit(data[col].astype(str))
    encoders[col] = le

print(f"✓ Label encoders created for all features")

# ===== 4. DATA TRANSFORMATION =====
print("\n[STEP 4] Transforming data...")
df_model = data.copy()
for col in features:
    df_model[col] = encoders[col].transform(data[col].astype(str))
df_model[target] = encoders[target].transform(data[target].astype(str))

X = df_model[features]
y = df_model[target]

print(f"✓ Data transformed successfully")
print(f"  - X shape: {X.shape}")
print(f"  - y shape: {y.shape}")

# ===== 5. TRAIN-TEST SPLIT =====
print("\n[STEP 5] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✓ Data split completed")
print(f"  - Training set: {len(X_train)} samples")
print(f"  - Test set: {len(X_test)} samples")

# ===== 6. MODEL TRAINING =====
print("\n[STEP 6] Training Random Forest Model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
model.fit(X_train, y_train)
print(f"✓ Model training completed")

# ===== 7. MODEL EVALUATION =====
print("\n[STEP 7] Evaluating Model Performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✓ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n" + "-" * 70)
print("CLASSIFICATION REPORT:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=encoders['match_winner'].classes_))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nFEATURE IMPORTANCE:")
for feature, importance in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {importance:.4f}")

# ===== 8. SAVE MODEL & ENCODERS =====
print("\n[STEP 8] Saving Model & Encoders...")

# Create models directory if it doesn't exist
models_dir = 'trained_models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"✓ Created '{models_dir}' directory")

# Save using pickle
pickle_model_path = os.path.join(models_dir, 'ipl_model.pkl')
pickle_encoders_path = os.path.join(models_dir, 'ipl_encoders.pkl')

with open(pickle_model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✓ Model saved (pickle): {pickle_model_path}")

with open(pickle_encoders_path, 'wb') as f:
    pickle.dump(encoders, f)
print(f"✓ Encoders saved (pickle): {pickle_encoders_path}")

# Save using joblib (more efficient for large sklearn models)
joblib_model_path = os.path.join(models_dir, 'ipl_model.joblib')
joblib_encoders_path = os.path.join(models_dir, 'ipl_encoders.joblib')

joblib.dump(model, joblib_model_path)
print(f"✓ Model saved (joblib): {joblib_model_path}")

joblib.dump(encoders, joblib_encoders_path)
print(f"✓ Encoders saved (joblib): {joblib_encoders_path}")

# Save metadata
metadata = {
    'features': features,
    'target': target,
    'accuracy': float(accuracy),
    'n_classes': len(encoders['match_winner'].classes_),
    'classes': list(encoders['match_winner'].classes_),
    'model_type': 'RandomForestClassifier',
    'n_estimators': 100,
    'test_size': 0.2,
    'random_state': 42
}

metadata_path = os.path.join(models_dir, 'model_metadata.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Metadata saved: {metadata_path}")

# ===== 9. SUMMARY =====
print("\n" + "=" * 70)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 70)
print(f"\n✓ Model successfully trained and saved!")
print(f"\nFiles saved in '{models_dir}/' directory:")
print(f"  1. ipl_model.pkl - Model (pickle format)")
print(f"  2. ipl_encoders.pkl - Encoders (pickle format)")
print(f"  3. ipl_model.joblib - Model (joblib format)")
print(f"  4. ipl_encoders.joblib - Encoders (joblib format)")
print(f"  5. model_metadata.pkl - Model metadata")
print(f"\n✓ Model Accuracy on Test Set: {accuracy*100:.2f}%")
print(f"✓ Total Training Samples: {len(X_train)}")
print(f"✓ Total Test Samples: {len(X_test)}")
print(f"\n✓ Ready for deployment!")
print("=" * 70)
