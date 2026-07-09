import pandas as pd
import sqlite3
import pickle
import os

class EloSystem:
    def __init__(self, k=32, surface_k=32):
        self.overall_elo = {}
        self.surface_elo = {'Hard': {}, 'Clay': {}, 'Grass': {}, 'Carpet': {}}
        self.k = k
        self.surface_k = surface_k
        self.default_elo = 1500
        
    def get_elo(self, player_id, surface=None):
        if surface and surface in self.surface_elo:
            return self.surface_elo[surface].get(player_id, self.default_elo)
        return self.overall_elo.get(player_id, self.default_elo)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, winner_id, loser_id, surface):
        win_elo = self.get_elo(winner_id)
        los_elo = self.get_elo(loser_id)
        expected_win = self.expected_score(win_elo, los_elo)
        self.overall_elo[winner_id] = win_elo + self.k * (1 - expected_win)
        self.overall_elo[loser_id] = los_elo + self.k * (0 - (1 - expected_win))
        if surface in self.surface_elo:
            win_surf_elo = self.get_elo(winner_id, surface)
            los_surf_elo = self.get_elo(loser_id, surface)
            exp_win_surf = self.expected_score(win_surf_elo, los_surf_elo)
            self.surface_elo[surface][winner_id] = win_surf_elo + self.surface_k * (1 - exp_win_surf)
            self.surface_elo[surface][loser_id] = los_surf_elo + self.surface_k * (0 - (1 - exp_win_surf))


def update_elo_and_db():
    print("Extracting 2025 and 2026 matches from MCP...")
    matches_df = pd.read_csv("data_MCP/charting-m-matches.csv", low_memory=False)
    matches_df['Date'] = pd.to_numeric(matches_df['Date'], errors='coerce')
    matches_df = matches_df[matches_df['Date'] >= 20250101].copy()
    
    if matches_df.empty:
        print("No matches found for 2025/2026.")
        return
        
    print(f"Found {len(matches_df)} matches in MCP for 2025+.")
    
    # Load points to find the winner
    print("Parsing points to determine winners...")
    points_df = pd.read_csv("data_MCP/charting-m-points-2020s.csv", low_memory=False)
    points_df = points_df[points_df['match_id'].isin(matches_df['match_id'])].copy()
    
    # Group by match_id and get the last point played to determine winner
    points_df['Pt'] = pd.to_numeric(points_df['Pt'], errors='coerce')
    last_points = points_df.loc[points_df.groupby('match_id')['Pt'].idxmax()]
    winner_map = dict(zip(last_points['match_id'], last_points['PtWinner']))
    
    # Map names to IDs via SQLite
    conn = sqlite3.connect('tennis_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, full_name FROM Players")
    name_to_id = {row[1].lower(): row[0] for row in cursor.fetchall()}
    
    new_matches = []
    
    # For player stats
    print("Loading Overview stats for player stats...")
    overview_df = pd.read_csv("data_MCP/charting-m-stats-Overview.csv", low_memory=False)
    overview_df = overview_df[(overview_df['match_id'].isin(matches_df['match_id'])) & (overview_df['set'] == 'Total')]
    stats_dict = {}
    for _, row in overview_df.iterrows():
        mid = row['match_id']
        pname = str(row['player']).lower()
        if mid not in stats_dict:
            stats_dict[mid] = {}
        
        # safely parse to numeric
        try:
            stats_dict[mid][pname] = {
                'ace': int(row['aces']),
                'df': int(row['dfs']),
                'svpt': int(row['serve_pts']),
                '1stIn': int(row['first_in']),
                '1stWon': int(row['first_won']),
                '2ndWon': int(row['second_won'])
            }
        except:
            stats_dict[mid][pname] = {'ace':0, 'df':0, 'svpt':0, '1stIn':0, '1stWon':0, '2ndWon':0}

    # Prepare data for insertion
    matches_df = matches_df.sort_values(by='Date')
    
    with open('tennis_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        elo_sys = model_data['elo_sys']
        player_stats = model_data['player_stats']
        model = model_data['model']
        
    def update_stats(pid, stats_obj):
        if pid not in player_stats:
            player_stats[pid] = {'matches': 0, 'ace': 0, 'df': 0, 'svpt': 0, '1stIn': 0, '1stWon': 0, '2ndWon': 0}
        player_stats[pid]['matches'] += 1
        for k in ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon']:
            player_stats[pid][k] += stats_obj.get(k, 0)
    
    for _, row in matches_df.iterrows():
        mid = row['match_id']
        winner_num = str(winner_map.get(mid, '1')) # Default to 1 if missing
        
        p1_name = str(row['Player 1']).strip()
        p2_name = str(row['Player 2']).strip()
        
        if winner_num == '1':
            winner_name = p1_name
            loser_name = p2_name
        else:
            winner_name = p2_name
            loser_name = p1_name
            
        win_id = name_to_id.get(winner_name.lower())
        los_id = name_to_id.get(loser_name.lower())
        
        if win_id and los_id:
            surf = row['Surface']
            date = int(row['Date'])
            tourney = row['Tournament']
            
            # 1. Update Elo and Stats
            elo_sys.update(win_id, los_id, surf)
            
            win_stats = stats_dict.get(mid, {}).get(winner_name.lower(), {})
            los_stats = stats_dict.get(mid, {}).get(loser_name.lower(), {})
            
            update_stats(win_id, win_stats)
            update_stats(los_id, los_stats)
            
            # 2. Append to SQL batch
            new_matches.append((
                tourney, surf, date, win_id, los_id,
                win_stats.get('ace'), win_stats.get('df'), win_stats.get('svpt'), win_stats.get('1stIn'), win_stats.get('1stWon'), win_stats.get('2ndWon'),
                los_stats.get('ace'), los_stats.get('df'), los_stats.get('svpt'), los_stats.get('1stIn'), los_stats.get('1stWon'), los_stats.get('2ndWon')
            ))
            
    if new_matches:
        print(f"Inserting {len(new_matches)} matches into SQLite...")
        cursor.executemany('''
            INSERT INTO Matches (
                tourney_name, surface, tourney_date, winner_id, loser_id,
                w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon,
                l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', new_matches)
        conn.commit()
        
        print("Saving updated ELO to tennis_model.pkl...")
        with open('tennis_model.pkl', 'wb') as f:
            pickle.dump({
                'model': model,
                'elo_sys': elo_sys,
                'player_stats': player_stats
            }, f)
            
    conn.close()
    print("Update complete!")

if __name__ == "__main__":
    update_elo_and_db()
