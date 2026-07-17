import requests
from bs4 import BeautifulSoup
import random
import pandas as pd
import streamlit as st
import sqlite3
import os
import numpy as np

class TournamentEngine:
    def __init__(self, model, elo_sys, player_stats, name_to_id, player_names):
        self.model = model
        self.elo_sys = elo_sys
        self.player_stats = player_stats
        self.name_to_id = name_to_id
        self.player_names = player_names
        
    def get_profile(self, p_id):
        if p_id not in self.player_stats or self.player_stats[p_id]['matches'] == 0:
            return [0, 0, 0, 0, 0]
        st_data = self.player_stats[p_id]
        svpt = max(st_data['svpt'], 1)
        ace_rate = st_data['ace'] / svpt
        df_rate = st_data['df'] / svpt
        first_win_rate = st_data['1stWon'] / max(st_data['1stIn'], 1)
        second_win_rate = st_data['2ndWon'] / max(svpt - st_data['1stIn'], 1)
        bp_saved_rate = st_data.get('bpSaved', 0) / max(st_data.get('bpFaced', 1), 1)
        return [ace_rate, df_rate, first_win_rate, second_win_rate, bp_saved_rate]

    def get_form(self, p_id, surf):
        if p_id not in self.player_stats or 'form' not in self.player_stats[p_id]:
            return [0.5, 0.5]
        form = self.player_stats[p_id]['form']
        all_form = sum(form['all']) / len(form['all']) if form['all'] else 0.5
        surf_form = sum(form['surf'].get(surf, [])) / len(form['surf'].get(surf, [])) if form['surf'].get(surf) else 0.5
        return [all_form, surf_form]

    def predict_match_prob(self, name1, name2, surface):
        if name1 == "Bye": return 0.0
        if name2 == "Bye": return 1.0
        
        id1 = self.name_to_id.get(name1)
        id2 = self.name_to_id.get(name2)
        
        if not id1 or not id2:
            return 0.5  # Fallback if unknown player
            
        elo1 = self.elo_sys.get_elo(id1)
        elo2 = self.elo_sys.get_elo(id2)
        surf_elo1 = self.elo_sys.get_elo(id1, surface)
        surf_elo2 = self.elo_sys.get_elo(id2, surface)
        
        prof1 = self.get_profile(id1)
        prof2 = self.get_profile(id2)
        
        form1 = self.get_form(id1, surface)
        form2 = self.get_form(id2, surface)
        
        import sqlite3
        conn = sqlite3.connect('tennis_database.db')
        c = conn.cursor()
        c.execute("SELECT aggressiveness, ue_rate, fh_preference, net_tendency FROM Players WHERE id=?", (id1,))
        mcp1 = c.fetchone()
        if not mcp1 or mcp1[0] is None:
            mcp1 = (0.15, 0.35, 0.60, 0.10)
            
        c.execute("SELECT aggressiveness, ue_rate, fh_preference, net_tendency FROM Players WHERE id=?", (id2,))
        mcp2 = c.fetchone()
        if not mcp2 or mcp2[0] is None:
            mcp2 = (0.15, 0.35, 0.60, 0.10)
        conn.close()
        
        # H2H logic
        h2h_records = self.player_stats.get('GLOBAL_H2H_RECORDS', {})
        pair_key = tuple(sorted([id1, id2]))
        w_h2h_wins = h2h_records.get(pair_key, {}).get(id1, 0)
        l_h2h_wins = h2h_records.get(pair_key, {}).get(id2, 0)
        total_h2h = w_h2h_wins + l_h2h_wins
        id1_h2h_rate = w_h2h_wins / total_h2h if total_h2h > 0 else 0.5
        id2_h2h_rate = l_h2h_wins / total_h2h if total_h2h > 0 else 0.5
        
        features = [elo1, elo2, surf_elo1, surf_elo2, id1_h2h_rate, id2_h2h_rate] + prof1 + prof2 + form1 + form2 + list(mcp1) + list(mcp2)
        cols = ['A_elo', 'B_elo', 'A_surf_elo', 'B_surf_elo', 'A_h2h', 'B_h2h', 'A_ace', 'A_df', 'A_1w', 'A_2w', 'A_bp', 'B_ace', 'B_df', 'B_1w', 'B_2w', 'B_bp', 'A_form_all', 'A_form_surf', 'B_form_all', 'B_form_surf', 'A_agg', 'A_ue', 'A_fh', 'A_net', 'B_agg', 'B_ue', 'B_fh', 'B_net']
        df_feat = pd.DataFrame([features], columns=cols)
        
        prob_A = self.model.predict_proba(df_feat)[0][1]
        return prob_A

    def simulate_match(self, name1, name2, surface):
        prob_A = self.predict_match_prob(name1, name2, surface)
        
        rand_val = random.random()
        winner_name = name1 if rand_val < prob_A else name2
        winner_prob = prob_A if rand_val < prob_A else (1 - prob_A)
        
        if winner_prob > 0.65:
            est_sets = "2-0"
        else:
            est_sets = "2-1"
        
        return winner_name, winner_prob, est_sets
        
    def simulate_round(self, players, surface):
        next_round = []
        results = []
        for i in range(0, len(players), 2):
            if i+1 < len(players):
                p1 = players[i]
                p2 = players[i+1]
                winner, prob, est_sets = self.simulate_match(p1, p2, surface)
                next_round.append(winner)
                results.append((p1, p2, winner, prob, est_sets))
            else:
                next_round.append(players[i]) # Bye
        return next_round, results
        
    def simulate_tournament(self, players, surface):
        current_players = list(players)
        rounds = []
        while len(current_players) > 1:
            current_players, results = self.simulate_round(current_players, surface)
            rounds.append(results)
        return rounds, current_players[0]

    def simulate_monte_carlo(self, players, surface, runs=10000):
        import numpy as np
        import pandas as pd
        import sqlite3
        
        # Pad players list to the next power of 2 with "Bye"
        def next_power_of_2(x):
            return 1 if x == 0 else 2**(x - 1).bit_length()
            
        padded_players = list(players)
        target_len = next_power_of_2(len(padded_players))
        while len(padded_players) < target_len:
            padded_players.append("Bye")
            
        n = len(padded_players)
        P = np.zeros((n, n))
        
        # 1. Fetch all DB stats in one go
        ids = [self.name_to_id.get(p) for p in padded_players if self.name_to_id.get(p)]
        mcp_data = {}
        if ids:
            id_str = ','.join('?' for _ in ids)
            conn = sqlite3.connect('tennis_database.db')
            c = conn.cursor()
            c.execute(f"SELECT id, aggressiveness, ue_rate, fh_preference, net_tendency FROM Players WHERE id IN ({id_str})", ids)
            for row in c.fetchall():
                # Replace None with defaults just in case
                mcp_data[row[0]] = [
                    row[1] if row[1] is not None else 0.15,
                    row[2] if row[2] is not None else 0.35,
                    row[3] if row[3] is not None else 0.60,
                    row[4] if row[4] is not None else 0.10
                ]
            conn.close()
            
        # 2. Pre-compute player features
        player_dict = {}
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tennis_database.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        for p in padded_players:
            if p == "Bye": continue
            p_id = self.name_to_id.get(p)
            if not p_id:
                continue
                
            row = conn.execute("""
                SELECT winner_age, winner_ht, winner_rank, loser_age, loser_ht, loser_rank, winner_id
                FROM Matches 
                WHERE winner_id = ? OR loser_id = ?
                ORDER BY tourney_date DESC LIMIT 1
            """, (p_id, p_id)).fetchone()
            
            if row:
                if row['winner_id'] == p_id:
                    age, ht, rank = float(row['winner_age'] or 25.0), float(row['winner_ht'] or 185.0), int(row['winner_rank'] or 100)
                else:
                    age, ht, rank = float(row['loser_age'] or 25.0), float(row['loser_ht'] or 185.0), int(row['loser_rank'] or 100)
            else:
                age, ht, rank = 25.0, 185.0, 100
                
            player_dict[p] = {
                'elo': self.elo_sys.get_elo(p_id),
                'surf_elo': self.elo_sys.get_elo(p_id, surface),
                'prof': self.get_profile(p_id),
                'form': self.get_form(p_id, surface),
                'mcp': mcp_data.get(p_id, [0.15, 0.35, 0.60, 0.10]),
                'age': age,
                'ht': ht,
                'rank': rank
            }
        conn.close()
            
        # 3. Build batch features
        features_list = []
        indices_list = []
        for i in range(n):
            for j in range(i+1, n):
                p1 = padded_players[i]
                p2 = padded_players[j]
                
                if p1 == "Bye":
                    P[i, j] = 0.0
                    P[j, i] = 1.0
                    continue
                if p2 == "Bye":
                    P[i, j] = 1.0
                    P[j, i] = 0.0
                    continue
                
                d1 = player_dict.get(p1)
                d2 = player_dict.get(p2)
                if not d1 or not d2:
                    P[i, j] = 0.5
                    P[j, i] = 0.5
                    continue
                    
                id1 = self.name_to_id.get(p1)
                id2 = self.name_to_id.get(p2)
                h2h_records = self.player_stats.get('GLOBAL_H2H_RECORDS', {})
                pair_key = tuple(sorted([id1, id2]))
                w_h2h_wins = h2h_records.get(pair_key, {}).get(id1, 0)
                l_h2h_wins = h2h_records.get(pair_key, {}).get(id2, 0)
                total_h2h = w_h2h_wins + l_h2h_wins
                id1_h2h_rate = w_h2h_wins / total_h2h if total_h2h > 0 else 0.5
                id2_h2h_rate = l_h2h_wins / total_h2h if total_h2h > 0 else 0.5
                    
                indoor = 1 if surface == 'Carpet' else 0
                delta_elo = d1['elo'] - d2['elo']
                delta_rank = d1['rank'] - d2['rank']
                
                features = [
                    d1['elo'], d2['elo'], d1['surf_elo'], d2['surf_elo'], delta_elo,
                    id1_h2h_rate, id2_h2h_rate, id1_h2h_rate, id2_h2h_rate,
                    d1['age'], d2['age'], d1['ht'], d2['ht'], d1['rank'], d2['rank'], delta_rank,
                    indoor, 0, 0
                ] + d1['prof'] + d2['prof'] + d1['form'] + d2['form'] + d1['mcp'] + d2['mcp']
                features_list.append(features)
                indices_list.append((i, j))
                
        # 4. Batch Predict
        if features_list:
            cols = ['A_elo', 'B_elo', 'A_surf_elo', 'B_surf_elo', 'delta_elo',
                    'A_h2h', 'B_h2h', 'A_surf_h2h', 'B_surf_h2h',
                    'A_age', 'B_age', 'A_ht', 'B_ht', 'A_rank', 'B_rank', 'delta_rank',
                    'indoor', 'A_streak', 'B_streak',
                    'A_ace', 'A_df', 'A_1w', 'A_2w', 'A_bp', 
                    'B_ace', 'B_df', 'B_1w', 'B_2w', 'B_bp', 
                    'A_form_all', 'A_form_surf', 'B_form_all', 'B_form_surf', 
                    'A_agg', 'A_ue', 'A_fh', 'A_net', 
                    'B_agg', 'B_ue', 'B_fh', 'B_net']
            df_feat = pd.DataFrame(features_list, columns=cols)
            probs = self.model.predict_proba(df_feat)[:, 1]
            
            for k, (i, j) in enumerate(indices_list):
                P[i, j] = probs[k]
                P[j, i] = 1 - probs[k]
                
        # To simulate a knockout tournament vectorized
        # players are represented by indices 0 to n-1
        # we can run 'runs' simulations simultaneously
        
        current_indices = np.tile(np.arange(n), (runs, 1))
        
        while current_indices.shape[1] > 1:
            next_indices = np.zeros((runs, current_indices.shape[1] // 2), dtype=int)
            for i in range(current_indices.shape[1] // 2):
                # match between current_indices[:, 2*i] and current_indices[:, 2*i + 1]
                p1_idx = current_indices[:, 2*i]
                p2_idx = current_indices[:, 2*i+1]
                
                # Fetch probabilities using advanced indexing
                probs = P[p1_idx, p2_idx]
                rands = np.random.rand(runs)
                
                p1_wins = rands < probs
                
                next_indices[:, i] = np.where(p1_wins, p1_idx, p2_idx)
                
            current_indices = next_indices
            
        champions = current_indices[:, 0]
        champ_counts = np.bincount(champions, minlength=n)
        
        results = []
        for i in range(n):
            if champ_counts[i] > 0 and padded_players[i] != "Bye":
                results.append((padded_players[i], champ_counts[i] / runs))
                
        results.sort(key=lambda x: x[1], reverse=True)
        return results

def scrape_tournaments():
    """Attempt to scrape ATP tour page, fallback if blocked."""
    url = "https://www.atptour.com/en/tournaments"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    
    tournaments = []
    try:
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        if "Just a moment..." in r.text or r.status_code != 200:
            raise Exception("Cloudflare blocking or bad status.")
            
        # Very basic extraction logic depending on the structure
        table = soup.find('table', class_='tournaments-table-class')
        if table:
            rows = table.find_all('tr')
            for row in rows[1:6]:
                cols = row.find_all('td')
                if len(cols) > 2:
                    name = cols[1].text.strip()
                    surface = "Hard" # default
                    tournaments.append({"name": name, "surface": surface})
    except Exception as e:
        print(f"Scraping failed: {e}. Using fallback data.")
        pass
        
    # Fallback to simulated current tournaments if scraping fails or is empty
    if not tournaments:
        tournaments = [
            {"name": "Wimbledon", "surface": "Grass"},
            {"name": "Hamburg Open", "surface": "Clay"},
            {"name": "Swedish Open", "surface": "Clay"},
            {"name": "Hall of Fame Open", "surface": "Grass"}
        ]
        
    return tournaments

def generate_mock_draw(players_list, size=16):
    """Generates a mock draw using available known players."""
    # Pick a random subset of known players
    valid_players = [p for p in players_list if p != "Error"]
    if len(valid_players) < size:
        size = len(valid_players)
    draw = random.sample(valid_players, size)
    return draw
