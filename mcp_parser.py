import pandas as pd
import numpy as np
import pickle
import os

MCP_DIR = "data/tennis_MatchChartingProject"
PLAYER_NAMES_PKL = "player_names.pkl"
OUTPUT_PKL = "player_style_profiles.pkl"

def main():
    print("Loading Match Charting Project data...")
    
    # Load Overview stats
    overview_path = os.path.join(MCP_DIR, "charting-m-stats-Overview.csv")
    df_overview = pd.read_csv(overview_path, low_memory=False)
    # We only care about full match totals
    df_overview = df_overview[df_overview['set'] == 'Total'].copy()
    
    # Calculate Total Points
    df_overview['total_pts'] = df_overview['serve_pts'] + df_overview['return_pts']
    
    # Load NetPoints stats
    net_path = os.path.join(MCP_DIR, "charting-m-stats-NetPoints.csv")
    df_net = pd.read_csv(net_path, low_memory=False)
    # We only care about overall NetPoints row
    df_net = df_net[df_net['row'] == 'NetPoints'].copy()
    
    # Merge Overview and NetPoints
    df_merged = pd.merge(df_overview, df_net, on=['match_id', 'player'], how='left')
    
    # Group by player to get career totals in charted matches
    # Summing all relevant columns
    cols_to_sum = [
        'total_pts', 'winners', 'unforced', 'winners_fh', 'winners_bh', 
        'net_pts', 'total_shots'
    ]
    
    # Ensure columns are numeric
    for col in cols_to_sum:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)
        
    grouped = df_merged.groupby('player')[cols_to_sum].sum().reset_index()
    
    # Calculate Style Indices
    # Filter out players with very few charted points (e.g., < 100 points) to avoid noisy stats
    grouped = grouped[grouped['total_pts'] >= 100].copy()
    
    grouped['aggressiveness'] = grouped['winners'] / grouped['total_pts']
    grouped['ue_rate'] = grouped['unforced'] / grouped['total_pts']
    
    # Forehand preference: FH winners / (FH winners + BH winners)
    grouped['fh_preference'] = grouped['winners_fh'] / (grouped['winners_fh'] + grouped['winners_bh']).replace(0, 1)
    
    # Net tendency: net points / total shots (if total shots is 0, use total pts as proxy or 1)
    grouped['net_tendency'] = grouped['net_pts'] / grouped['total_shots'].replace(0, 1)
    
    # Cap / clean infinite or crazy values just in case
    grouped['aggressiveness'] = grouped['aggressiveness'].clip(0, 1)
    grouped['ue_rate'] = grouped['ue_rate'].clip(0, 1)
    grouped['fh_preference'] = grouped['fh_preference'].clip(0, 1)
    grouped['net_tendency'] = grouped['net_tendency'].clip(0, 1)
    
    # Load player IDs to map name to ID
    if os.path.exists(PLAYER_NAMES_PKL):
        with open(PLAYER_NAMES_PKL, 'rb') as f:
            id_to_name = pickle.load(f)
            name_to_id = {v: k for k, v in id_to_name.items()}
    else:
        name_to_id = {}
        
    # Build dictionary
    profiles = {}
    for _, row in grouped.iterrows():
        player_name = row['player']
        # Very basic normalization for names if needed, usually MCP matches ATP names well
        # Sometimes 'Alex De Minaur' vs 'Alex de Minaur'
        pid = name_to_id.get(player_name)
        if not pid:
            # Try case-insensitive
            pid = next((k for n, k in name_to_id.items() if n.lower() == player_name.lower()), None)
            
        if pid:
            profiles[pid] = {
                'aggressiveness': row['aggressiveness'],
                'ue_rate': row['ue_rate'],
                'fh_preference': row['fh_preference'],
                'net_tendency': row['net_tendency']
            }
            
    print(f"Generated style profiles for {len(profiles)} players.")
    
    # Calculate medians for imputation
    medians = {
        'aggressiveness': grouped['aggressiveness'].median(),
        'ue_rate': grouped['ue_rate'].median(),
        'fh_preference': grouped['fh_preference'].median(),
        'net_tendency': grouped['net_tendency'].median()
    }
    
    final_output = {
        'profiles': profiles,
        'medians': medians
    }
    
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(final_output, f)
        
    print(f"Saved profiles to {OUTPUT_PKL}")
    print("Medians for imputation:", medians)

if __name__ == "__main__":
    main()
