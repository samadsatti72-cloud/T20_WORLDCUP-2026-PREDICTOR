# src/simulate_tournament.py

import pandas as pd
import argparse
import joblib
import time
import os
import warnings
from collections import Counter
import numpy as np

from predict import predict_probability

warnings.simplefilter("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# Pre-determined Super 8 seeding (ICC official)
SUPER8_SEEDING = {
    'Group A': ['X1', 'Y3'],  # India (X1), Pakistan (Y3) if they qualify
    'Group B': ['X2', 'Y4'],  # Australia (X2), Sri Lanka (Y4) if they qualify
    'Group C': ['X3', 'Y1'],  # West Indies (X3), England (Y1) if they qualify
    'Group D': ['X4', 'Y2']   # South Africa (X4), New Zealand (Y2) if they qualify
}

SEEDED_TEAMS = {
    'India': 'X1',
    'Australia': 'X2', 
    'West Indies': 'X3',
    'South Africa': 'X4',
    'England': 'Y1',
    'New Zealand': 'Y2',
    'Pakistan': 'Y3',
    'Sri Lanka': 'Y4'
}


def simulate_group_stage(model, elo_lookup, group_matches):
    """Simulate group stage and return top 2 from each group"""
    
    # Initialize points table
    points = {}
    
    for _, match in group_matches.iterrows():
        team_a = match['teamA']
        team_b = match['teamB']
        
        # Skip if teams are not real (placeholder)
        if '_' in team_a or '_' in team_b:
            continue
            
        # Initialize teams in points table
        if team_a not in points:
            points[team_a] = {'played': 0, 'won': 0, 'lost': 0, 'points': 0, 'nrr': 0}
        if team_b not in points:
            points[team_b] = {'played': 0, 'won': 0, 'lost': 0, 'points': 0, 'nrr': 0}
        
        # Predict match outcome
        prob_a_wins = predict_probability(
            model, team_a, team_b, match.get('Venue', ''), elo_lookup
        )
        
        # Determine winner based on probability
        winner = team_a if np.random.random() < prob_a_wins else team_b
        loser = team_b if winner == team_a else team_a
        
        # Update points
        points[winner]['won'] += 1
        points[winner]['points'] += 2
        points[loser]['lost'] += 1
        points[winner]['played'] += 1
        points[loser]['played'] += 1
        
        # Simple NRR simulation (winner gets +0.5, loser gets -0.5)
        points[winner]['nrr'] += 0.5
        points[loser]['nrr'] -= 0.5
    
    return points


def get_group_qualifiers(points_table):
    """Get top 2 teams from a group based on points and NRR"""
    
    # Sort by points (descending), then by NRR (descending)
    sorted_teams = sorted(
        points_table.items(),
        key=lambda x: (x[1]['points'], x[1]['nrr']),
        reverse=True
    )
    
    # Return top 2
    return [team[0] for team in sorted_teams[:2]]


def map_to_super8_groups(qualifiers_by_group):
    """Map qualified teams to Super 8 groups based on pre-determined seeding"""
    
    super8_groups = {'X': [], 'Y': []}
    
    for group_name, teams in qualifiers_by_group.items():
        # Get seeding for this group
        seeds = SUPER8_SEEDING[group_name]
        
        for i, team in enumerate(teams):
            # Assign to Super 8 group based on seeding
            # If team is a seeded team, use their pre-determined position
            # If not, they inherit the position of the seeded team from that group
            
            seed_position = seeds[i]  # X1/X2/X3/X4 or Y1/Y2/Y3/Y4
            
            if seed_position.startswith('X'):
                super8_groups['X'].append(team)
            else:
                super8_groups['Y'].append(team)
    
    return super8_groups


def simulate_super8(model, elo_lookup, super8_groups):
    """Simulate Super 8 stage and return top 2 from each group"""
    
    results = {}
    
    for group_name, teams in super8_groups.items():
        points = {team: {'points': 0, 'nrr': 0} for team in teams}
        
        # Each team plays every other team once
        for i, team_a in enumerate(teams):
            for team_b in teams[i+1:]:
                # Predict match
                prob_a_wins = predict_probability(
                    model, team_a, team_b, '', elo_lookup
                )
                
                winner = team_a if np.random.random() < prob_a_wins else team_b
                loser = team_b if winner == team_a else team_a
                
                points[winner]['points'] += 2
                points[winner]['nrr'] += 0.5
                points[loser]['nrr'] -= 0.5
        
        # Get top 2
        sorted_teams = sorted(
            points.items(),
            key=lambda x: (x[1]['points'], x[1]['nrr']),
            reverse=True
        )
        
        results[group_name] = [team[0] for team in sorted_teams[:2]]
    
    return results


def simulate_knockout(model, elo_lookup, semi_finalists):
    """Simulate semifinals and final"""
    
    # Semi-final 1: Winner of Super8 Group X vs Winner of Super8 Group Y
    sf1_team1 = semi_finalists['X'][0]
    sf1_team2 = semi_finalists['Y'][0]
    
    prob_sf1 = predict_probability(model, sf1_team1, sf1_team2, '', elo_lookup)
    sf1_winner = sf1_team1 if np.random.random() < prob_sf1 else sf1_team2
    
    # Semi-final 2: Runner-up of Super8 Group X vs Runner-up of Super8 Group Y  
    sf2_team1 = semi_finalists['X'][1]
    sf2_team2 = semi_finalists['Y'][1]
    
    prob_sf2 = predict_probability(model, sf2_team1, sf2_team2, '', elo_lookup)
    sf2_winner = sf2_team1 if np.random.random() < prob_sf2 else sf2_team2
    
    # Final
    prob_final = predict_probability(model, sf1_winner, sf2_winner, '', elo_lookup)
    champion = sf1_winner if np.random.random() < prob_final else sf2_winner
    runner_up = sf2_winner if champion == sf1_winner else sf1_winner
    
    return champion, runner_up, (sf1_winner, sf2_winner)


def monte_carlo(fixtures_path, model_path, n=300):
    print("=" * 60)
    print(" T20 WORLD CUP 2026 SIMULATION STARTED")
    print(f" Monte Carlo runs: {n}")
    print("=" * 60)

    # Load fixtures
    fixtures = pd.read_csv(fixtures_path)
    
    # Load trained model and ELO lookup
    model = joblib.load(model_path)
    elo_lookup = joblib.load(os.path.join(os.path.dirname(model_path), "elo_lookup.joblib"))


    # Separate group stage matches
    group_matches = fixtures[fixtures['stage'].str.contains('Group', na=False)]
    
    finalist_counter = Counter()
    champion_counter = Counter()
    semi_finalists_counter = Counter()
    
    start_time = time.time()

    for i in range(1, n + 1):
        # Simulate group stage
        groups = {}
        for group_name in ['Group A', 'Group B', 'Group C', 'Group D']:
            group_df = group_matches[group_matches['stage'] == group_name]
            points_table = simulate_group_stage(model, elo_lookup, group_df)
            groups[group_name] = get_group_qualifiers(points_table)
        
        # Map to Super 8 groups
        super8_groups = map_to_super8_groups(groups)
        
        # Simulate Super 8
        super8_results = simulate_super8(model, elo_lookup, super8_groups)
        
        # Simulate knockouts
        champion, runner_up, finalists = simulate_knockout(model, elo_lookup, super8_results)
        
        # Record results
        finalist_pair = tuple(sorted([champion, runner_up]))
        finalist_counter[finalist_pair] += 1
        champion_counter[champion] += 1
        
        # Record all 4 semi-finalists
        all_semi_finalists = tuple(sorted([
            super8_results['X'][0], super8_results['X'][1],
            super8_results['Y'][0], super8_results['Y'][1]
        ]))
        semi_finalists_counter[all_semi_finalists] += 1

        if i % 50 == 0 or i == n:
            elapsed = int(time.time() - start_time)
            print(f"⏳ Completed {i}/{n} simulations ({elapsed}s elapsed)")

    print("=" * 60)
    print(" SIMULATION COMPLETED")
    print("=" * 60)

    return finalist_counter, champion_counter, semi_finalists_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True, help="Path to fixtures.csv")
    parser.add_argument("--model", default="outputs/model.joblib")
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    finalist_results, champion_results, semi_results = monte_carlo(
        args.fixtures, args.model, args.n
    )

    print("\n MOST LIKELY CHAMPIONS\n")
    total = sum(champion_results.values())
    for team, count in champion_results.most_common(10):
        probability = count / total
        print(f"{team:20s} → {probability:6.2%} ({count} times)")

    print("\n MOST LIKELY FINALISTS\n")
    total = sum(finalist_results.values())
    for teams, count in finalist_results.most_common(10):
        probability = count / total
        print(f"{teams[0]:15s} vs {teams[1]:15s} → {probability:6.2%}")

    print("\n MOST LIKELY SEMI-FINALISTS\n")
    total = sum(semi_results.values())
    for teams, count in semi_results.most_common(5):
        probability = count / total
        print(f"{', '.join(teams):50s} → {probability:6.2%}")


if __name__ == "__main__":
    main()