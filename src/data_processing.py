# data_processing.py
import pandas as pd
import numpy as np
from elo import Elo

def load_and_standardize(paths):

    """
    
    Inputs: list of file paths (csv)
    Returns: combined pandas DataFrame with standardized columns:
      ['date', 'team1', 'team2', 'winner', 'venue', 'city', ...]
    """
    dfs = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        # basic heuristics to map columns
        # Try known column names first
        colmap = {}
        # winner
        for c in df.columns:
            if 'winner' in c.lower() or 'winning' in c.lower():
                colmap['Winner'] = c
                break
        # team1/team2
        for pattern in ['team1','team_1','team 1','team a','home_team']:
            for c in df.columns:
                if pattern in c.lower():
                    colmap['Team1'] = c
                    break
            if 'Team1' in colmap:
                break
        for pattern in ['team2','team_2','team 2','team b','away_team']:
            for c in df.columns:
                if pattern in c.lower():
                    colmap['Team2'] = c
                    break
            if 'Team2' in colmap:
                break
        # date
        for c in df.columns:
            if 'date' in c.lower():
                colmap['Date'] = c
                break
        # venue / city
        for c in df.columns:
            if 'venue' in c.lower() or 'ground' in c.lower() or 'stadium' in c.lower():
                colmap['Venue'] = c
                break
        # rename as standardized
        df_std = pd.DataFrame()
        df_std['winner'] = df[colmap['Winner']] if 'Winner' in colmap else None
        df_std['team1'] = df[colmap['Team1']] if 'Team1' in colmap else None
        df_std['team2'] = df[colmap['Team2']] if 'Team2' in colmap else None
        df_std['date'] = pd.to_datetime(df[colmap['Date']], dayfirst=True, errors='coerce') if 'Date' in colmap else None
        df_std['venue'] = df[colmap['Venue']] if 'Venue' in colmap else None
        for c in df.columns:
            if 'toss' in c.lower():
                df_std['toss'] = df[c]
                break
        dfs.append(df_std)
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    # normalize strings
    for c in ['team1','team2','winner','venue']:
        if c in combined.columns:
            combined[c] = combined[c].astype(str).str.strip()
    return combined

def compute_elo_history(matches_df, k=20, base=1500):
    """
    Input: matches_df sorted by date ascending with columns team1, team2, team1_won (binary)
    Returns: matches_df with elo_team1, elo_team2 at time of match and final Elo dict
    """
    # ensure sorted
    m = matches_df.sort_values('date', na_position='first').reset_index(drop=True).copy()
    elo = Elo(base=base, k=k)
    elo_t1 = []
    elo_t2 = []
    for _, row in m.iterrows():
        t1 = row['team1']
        t2 = row['team2']
        r1 = elo.get(t1)
        r2 = elo.get(t2)
        elo_t1.append(r1)
        elo_t2.append(r2)
        # update based on result
        score_a = 1.0 if row.get('team1_won', 0) == 1 else 0.0
        elo.update(t1, t2, score_a)
    m['elo_team1'] = elo_t1
    m['elo_team2'] = elo_t2
    return m, elo.ratings

def add_recent_form(m, window=5):
    """
    For each match, compute last `window` matches win rate for team1 and team2 prior to the match.
    """
    m = m.copy()
    m['recent_form_team1'] = 0.5
    m['recent_form_team2'] = 0.5
    history = {}
    for idx, row in m.iterrows():
        t1 = row['team1']; t2 = row['team2']
        hist1 = history.get(t1, [])
        hist2 = history.get(t2, [])
        m.at[idx,'recent_form_team1'] = sum(hist1)/len(hist1) if hist1 else 0.5
        m.at[idx,'recent_form_team2'] = sum(hist2)/len(hist2) if hist2 else 0.5
        # append current match result for future matches
        res1 = 1 if row.get('team1_won', 0)==1 else 0
        history.setdefault(t1, []).append(res1)
        history.setdefault(t2, []).append(1-res1)
        # truncate
        history[t1] = history[t1][-window:]
        history[t2] = history[t2][-window:]
    return m