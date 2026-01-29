# train_model.py

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings

from data_processing import (
    load_and_standardize,
    compute_elo_history,
    add_recent_form,
)

warnings.simplefilter("ignore")


def prepare_training(paths):
    """
    Load data, compute ELO + recent form features,
    and return training matrix + final ELO lookup.
    """
    print(" Loading and processing training data...")
    df = load_and_standardize(paths)

    print(f"   Loaded {len(df)} matches from {len(paths)} file(s)")

    # Drop incomplete rows
    df = df.dropna(subset=["team1", "team2", "winner"])
    print(f"   After cleaning: {len(df)} valid matches")

    # Binary target: did team1 win?
    df["team1_won"] = (
        df["winner"].str.lower() == df["team1"].str.lower()
    ).astype(int)

    # Compute ELO history
    print("   Computing ELO ratings...")
    df_with_elo, final_elos = compute_elo_history(df)

    # Add recent form features
    print("   Computing recent form...")
    df_feat = add_recent_form(df_with_elo)

    # Feature engineering
    df_feat["elo_diff"] = df_feat["elo_team1"] - df_feat["elo_team2"]
    df_feat["form_diff"] = (
        df_feat["recent_form_team1"] - df_feat["recent_form_team2"]
    )

    # Keep rows with valid features
    df_train = df_feat.dropna(subset=["elo_diff", "form_diff"])

    X = df_train[["elo_diff", "form_diff"]]
    y = df_train["team1_won"]

    print(f"   Training samples: {len(X)}")
    print(f"   Unique teams: {len(final_elos)}")

    return X, y, final_elos


def train_and_save(paths, output_dir="../outputs"):
    """
    Train model and save artifacts
    
    Args:
        paths: List of CSV file paths containing historical match data
        output_dir: Directory to save model and ELO lookup
    """
    print("=" * 60)
    print(" T20 WORLD CUP 2026 PREDICTOR - MODEL TRAINING")
    print("=" * 60)

    X, y, final_elos = prepare_training(paths)

    print(f"\n Training samples: {len(X)}")
    print(f" Features: {list(X.columns)}")

    # Train Random Forest model
    print("\n Training Random Forest model...")
    clf = RandomForestClassifier(
        n_estimators=100,   # faster & stable
        n_jobs=1,
        random_state=42,
    )

    # Cross-validation
    print("   Performing 5-fold cross-validation...")
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    print(f"    CV ROC-AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Final training on all data
    print("\n   Training final model on all data...")
    clf.fit(X, y)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(clf, model_path)
    print(f" Saved model to {model_path}")

    # Save ELO lookup
    elo_path = os.path.join(output_dir, "elo_lookup.joblib")
    joblib.dump(final_elos, elo_path)
    print(f" Saved ELO lookup to {elo_path}")
    
    # Print top teams by ELO
    print("\n TOP 10 TEAMS BY ELO RATING:")
    sorted_teams = sorted(final_elos.items(), key=lambda x: x[1], reverse=True)
    for i, (team, elo) in enumerate(sorted_teams[:10], 1):
        print(f"   {i:2d}. {team:20s} - {elo:.1f}")

    print("\n" + "=" * 60)
    print(" TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


def main():
    """
    Main training function
    Paths are relative to src/ directory
    """
    
    # Paths relative to src/ directory (where this script runs from)
    paths = [
        r"..\data\all_teams_world_t20_results.csv",
        r"..\data\Recent_t20I_matches.csv",
    ]
    
    # Check if files exist
    missing_files = [p for p in paths if not os.path.exists(p)]
    if missing_files:
        print(" ERROR: The following data files were not found:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure your data files are in the correct location.")
        print("Current working directory:", os.getcwd())
        print("\nExpected structure:")
        print("Expected structure:")
        print("T20_WORLDCUP-2026-PREDICTOR/")
        print("|-- data/")
        print("|   |-- all_teams_world_t20_results.csv")
        print("|   `-- Recent_t20I_matches.csv")
        print("`-- src/")

        return
    
    train_and_save(paths)


if __name__ == "__main__":
    main()