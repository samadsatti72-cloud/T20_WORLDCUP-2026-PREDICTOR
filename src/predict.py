# predict.py

import pandas as pd
import numpy as np


def predict_probability(model, team1, team2, venue, elo_lookup):
    """
    Predict probability that team1 beats team2
    
    Args:
        model: Trained sklearn model
        team1: Name of first team
        team2: Name of second team
        venue: Venue name (not used in current model but kept for compatibility)
        elo_lookup: Dictionary mapping team names to ELO ratings
        
    Returns:
        float: Probability that team1 wins (between 0 and 1)
    """

    # Get ELO ratings (default 1500 if missing)
    elo1 = elo_lookup.get(team1, 1500)
    elo2 = elo_lookup.get(team2, 1500)

    # Feature engineering (MUST match training)
    X = pd.DataFrame([{
        "elo_diff": elo1 - elo2,
        "form_diff": 0.0  # neutral form for simulation
    }])

    # Predict probability
    prob = model.predict_proba(X)[0, 1]
    return float(prob)