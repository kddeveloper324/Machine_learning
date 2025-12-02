"""
IPL Model Loading Utility
Quick helper to load and use the trained model in your own code
"""

import pickle
import joblib
import pandas as pd
import os
from typing import Tuple, Dict, Any

class IPLModelLoader:
    """Utility class to load and use the trained IPL prediction model"""
    
    def __init__(self, models_dir: str = 'trained_models', use_joblib: bool = False):
        """
        Initialize the model loader
        
        Args:
            models_dir: Directory containing the saved models
            use_joblib: If True, use joblib format; otherwise use pickle
        """
        self.models_dir = models_dir
        self.use_joblib = use_joblib
        self.model = None
        self.encoders = None
        self.metadata = None
        
    def load_model(self) -> bool:
        """
        Load the trained model and encoders
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.use_joblib:
                self.model = joblib.load(os.path.join(self.models_dir, 'ipl_model.joblib'))
                self.encoders = joblib.load(os.path.join(self.models_dir, 'ipl_encoders.joblib'))
            else:
                with open(os.path.join(self.models_dir, 'ipl_model.pkl'), 'rb') as f:
                    self.model = pickle.load(f)
                with open(os.path.join(self.models_dir, 'ipl_encoders.pkl'), 'rb') as f:
                    self.encoders = pickle.load(f)
            
            with open(os.path.join(self.models_dir, 'model_metadata.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)
            
            print("✓ Model loaded successfully")
            return True
            
        except FileNotFoundError as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.metadata is None:
            return {}
        return {
            'model_type': self.metadata.get('model_type'),
            'accuracy': self.metadata.get('accuracy'),
            'n_classes': self.metadata.get('n_classes'),
            'classes': self.metadata.get('classes'),
            'features': self.metadata.get('features')
        }
    
    def predict(self, team1: str, team2: str, venue: str, 
                toss_winner: str, toss_decision: str) -> Tuple[str, float, Dict]:
        """
        Make a prediction for a match
        
        Args:
            team1: Home team name
            team2: Away team name
            venue: Match venue
            toss_winner: Team that won the toss
            toss_decision: 'bat' or 'field'
        
        Returns:
            Tuple of (predicted_winner, confidence, probability_dict)
        """
        if self.model is None or self.encoders is None:
            print("Model not loaded. Call load_model() first.")
            return None, None, None
        
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'team1': [self.encoders['team1'].transform([team1])[0]],
                'team2': [self.encoders['team2'].transform([team2])[0]],
                'venue': [self.encoders['venue'].transform([venue])[0]],
                'toss_winner': [self.encoders['toss_winner'].transform([toss_winner])[0]],
                'toss_decision': [self.encoders['toss_decision'].transform([toss_decision])[0]]
            })
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            probabilities = self.model.predict_proba(input_data)[0]
            
            # Decode prediction
            predicted_winner = self.encoders['match_winner'].inverse_transform([prediction])[0]
            confidence = max(probabilities)
            
            # Create probability dictionary
            prob_dict = dict(zip(self.encoders['match_winner'].classes_, probabilities))
            
            return predicted_winner, confidence, prob_dict
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None, None
    
    def predict_between_teams(self, team1: str, team2: str, venue: str,
                              toss_winner: str, toss_decision: str) -> Tuple[str, float]:
        """
        Make a prediction and normalize confidence between the two selected teams
        
        Args:
            Same as predict()
        
        Returns:
            Tuple of (predicted_winner, normalized_confidence)
        """
        predicted_winner, _, prob_dict = self.predict(
            team1, team2, venue, toss_winner, toss_decision
        )
        
        if predicted_winner is None:
            return None, None
        
        # Get probabilities for the two teams
        p1 = prob_dict.get(team1, 0.0)
        p2 = prob_dict.get(team2, 0.0)
        
        # Normalize between the two teams
        total = p1 + p2
        if total == 0:
            return team1, 0.5
        
        p1_norm = p1 / total
        p2_norm = p2 / total
        
        if p1_norm >= p2_norm:
            return team1, p1_norm
        else:
            return team2, p2_norm


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("IPL MODEL LOADING UTILITY - EXAMPLE")
    print("=" * 70)
    
    # Initialize loader (using pickle format)
    loader = IPLModelLoader(use_joblib=False)
    
    # Load model
    print("\n[STEP 1] Loading model...")
    if not loader.load_model():
        print("Failed to load model. Exiting.")
        exit()
    
    # Display model info
    print("\n[STEP 2] Model Information:")
    info = loader.get_model_info()
    print(f"  Model Type: {info['model_type']}")
    print(f"  Accuracy: {info['accuracy']*100:.2f}%")
    print(f"  Classes: {len(info['classes'])} teams")
    
    # Make predictions
    print("\n[STEP 3] Making Predictions:")
    
    predictions = [
        ("Mumbai Indians", "Chennai Super Kings", "Wankhede Stadium", "Mumbai Indians", "bat"),
        ("Kolkata Knight Riders", "Royal Challengers Bangalore", "Eden Gardens", "Kolkata Knight Riders", "field"),
        ("Delhi Capitals", "Sunrisers Hyderabad", "Arun Jaitley Stadium", "Sunrisers Hyderabad", "bat"),
    ]
    
    for team1, team2, venue, toss_winner, toss_decision in predictions:
        print(f"\n  {team1} vs {team2}")
        print(f"  Venue: {venue}, Toss: {toss_winner} decides to {toss_decision}")
        
        winner, confidence, probs = loader.predict(team1, team2, venue, toss_winner, toss_decision)
        
        if winner:
            print(f"  ✓ Predicted Winner: {winner}")
            print(f"  ✓ Confidence: {confidence*100:.2f}%")
            
            # Show top 3 probabilities
            top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top Predictions:")
            for team, prob in top_3:
                print(f"    - {team}: {prob*100:.2f}%")
    
    # Try normalized prediction between two teams
    print("\n[STEP 4] Normalized Prediction (between two teams only):")
    winner, confidence = loader.predict_between_teams(
        "Mumbai Indians", "Chennai Super Kings", "Wankhede Stadium", "Mumbai Indians", "bat"
    )
    if winner:
        print(f"  Winner: {winner}")
        print(f"  Confidence: {confidence*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("✓ Utility script completed successfully!")
    print("=" * 70)
