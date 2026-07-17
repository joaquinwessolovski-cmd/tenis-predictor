import pandas as pd
import numpy as np
import pickle
import sqlite3
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append('/Users/wesso/Downloads/ten/tenis')
from train_model import load_data, build_dataset

def main():
    print("Loading data up to 2024 for archetype modeling...")
    df = load_data()
    df['year'] = df['tourney_date'].astype(str).str[:4].astype(int)
    
    # Use data up to 2024 to prevent data leakage in 2025/2026
    df_train = df[df['year'] <= 2024].copy()
    df_train = df_train.sort_values('tourney_date').reset_index(drop=True)
    
    print("Building dataset to get final player stats up to 2024...")
    _, elo_sys, player_stats = build_dataset(df_train)

    print("Extracting features for clustering...")
    player_features = []
    player_ids = []
    
    for pid, stats in player_stats.items():
        svpt = stats.get('svpt', 0)
        if svpt >= 500: # only cluster players with enough data (500 serve points)
            serve_elo = getattr(elo_sys, 'serve_elo', {}).get(pid, elo_sys.default_elo)
            return_elo = getattr(elo_sys, 'return_elo', {}).get(pid, elo_sys.default_elo)
            
            ace_rate = stats.get('ace', 0) / svpt
            df_rate = stats.get('df', 0) / svpt
            first_win_rate = stats.get('1stWon', 0) / max(stats.get('1stIn', 1), 1)
            
            player_features.append([serve_elo, return_elo, ace_rate, df_rate, first_win_rate])
            player_ids.append(pid)
            
    X = np.array(player_features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    print("Cluster Centers (Scaled):")
    print(kmeans.cluster_centers_)
    
    with open('archetype_model.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler, 'kmeans': kmeans}, f)
        
    print("Archetype model saved to archetype_model.pkl")

if __name__ == '__main__':
    main()
