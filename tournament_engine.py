import requests
from bs4 import BeautifulSoup
import random
import pandas as pd
import streamlit as st
import sqlite3
import os
import numpy as np
import pickle
import glob
from markov_model import predict_match
from train_model import shin_probabilities

class TournamentEngine:
    def __init__(self, model, elo_sys, player_stats, name_to_id, player_names):
        self.models = {}
        if os.path.exists('ensembles'):
            for p in glob.glob('ensembles/*.pkl'):
                name = os.path.basename(p).replace('.pkl', '')
                with open(p, 'rb') as f:
                    self.models[name] = pickle.load(f)
                    
        self.archetype_model = None
        if os.path.exists('archetype_model.pkl'):
            with open('archetype_model.pkl', 'rb') as f:
                self.archetype_model = pickle.load(f)
        
        # Load universal stats
        if os.path.exists('tennis_elo_system.pkl'):
            with open('tennis_elo_system.pkl', 'rb') as f:
                data = pickle.load(f)
                self.elo_sys = data['elo_sys']
                self.player_stats = data['player_stats']
        else:
            self.elo_sys = elo_sys
            self.player_stats = player_stats
            
        self.fatigue = {} # track fatigue during tournament
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

    
    def _get_court_speed(self, surface):
        if surface == 'Grass': return 0.10
        elif surface == 'Hard': return 0.08
        else: return 0.05

    def _build_features(self, id1, id2, surface, level, fatigue_A, fatigue_B, p1_odds, p2_odds, altitude=0):
        elo1 = self.elo_sys.get_elo(id1)
        elo2 = self.elo_sys.get_elo(id2)
        surf_elo1 = self.elo_sys.get_elo(id1, surface)
        surf_elo2 = self.elo_sys.get_elo(id2, surface)
        
        prof1 = self.get_profile(id1)
        prof2 = self.get_profile(id2)
        
        form1 = self.get_form(id1, surface)
        form2 = self.get_form(id2, surface)
        
        conn = sqlite3.connect('tennis_database.db')
        c = conn.cursor()
        c.execute("SELECT aggressiveness, ue_rate, fh_preference, net_tendency FROM Players WHERE id=?", (id1,))
        mcp1 = c.fetchone() or (0.15, 0.35, 0.60, 0.10)
        c.execute("SELECT aggressiveness, ue_rate, fh_preference, net_tendency FROM Players WHERE id=?", (id2,))
        mcp2 = c.fetchone() or (0.15, 0.35, 0.60, 0.10)
        
        def get_info(pid):
            c.execute("SELECT winner_age, winner_ht, winner_rank, loser_age, loser_ht, loser_rank, winner_id, tourney_date FROM Matches WHERE winner_id = ? OR loser_id = ? ORDER BY tourney_date DESC LIMIT 1", (pid, pid))
            row = c.fetchone()
            if row:
                if row[6] == pid: return float(row[0] or 25.0), float(row[1] or 185.0), int(row[2] or 100), 2.0
                else: return float(row[3] or 25.0), float(row[4] or 185.0), int(row[5] or 100), 2.0
            return 25.0, 185.0, 100, 2.0
            
        age1, ht1, rank1, rest1 = get_info(id1)
        age2, ht2, rank2, rest2 = get_info(id2)
        conn.close()
        
        delta_elo = elo1 - elo2
        delta_rank = rank1 - rank2
        
        serve_elo1 = getattr(self.elo_sys, 'serve_elo', {}).get(id1, self.elo_sys.default_elo)
        return_elo1 = getattr(self.elo_sys, 'return_elo', {}).get(id1, self.elo_sys.default_elo)
        serve_elo2 = getattr(self.elo_sys, 'serve_elo', {}).get(id2, self.elo_sys.default_elo)
        return_elo2 = getattr(self.elo_sys, 'return_elo', {}).get(id2, self.elo_sys.default_elo)
        
        h2h_records = self.player_stats.get('GLOBAL_H2H_RECORDS', {})
        surf_h2h_records = self.player_stats.get('SURF_H2H_RECORDS', {})
        pair_key = tuple(sorted([id1, id2]))
        
        w_h2h_wins = h2h_records.get(pair_key, {}).get(id1, 0)
        l_h2h_wins = h2h_records.get(pair_key, {}).get(id2, 0)
        total_h2h = w_h2h_wins + l_h2h_wins
        id1_h2h_rate = w_h2h_wins / total_h2h if total_h2h > 0 else 0.5
        id2_h2h_rate = l_h2h_wins / total_h2h if total_h2h > 0 else 0.5
        
        w_surf_wins = surf_h2h_records.get(surface, {}).get(pair_key, {}).get(id1, 0)
        l_surf_wins = surf_h2h_records.get(surface, {}).get(pair_key, {}).get(id2, 0)
        total_surf_h2h = w_surf_wins + l_surf_wins
        id1_surf_h2h_rate = w_surf_wins / total_surf_h2h if total_surf_h2h > 0 else 0.5
        id2_surf_h2h_rate = l_surf_wins / total_surf_h2h if total_surf_h2h > 0 else 0.5
        
        w_winrate_vs_l_arch = 0.5
        l_winrate_vs_w_arch = 0.5
        
        if self.archetype_model:
            def get_archetype(pid):
                stats = self.player_stats.get(pid, {})
                if stats.get('svpt', 0) < 100: return -1
                s_elo = getattr(self.elo_sys, 'serve_elo', {}).get(pid, self.elo_sys.default_elo)
                r_elo = getattr(self.elo_sys, 'return_elo', {}).get(pid, self.elo_sys.default_elo)
                ace = stats.get('ace', 0) / max(stats.get('svpt', 1), 1)
                df = stats.get('df', 0) / max(stats.get('svpt', 1), 1)
                fw = stats.get('1stWon', 0) / max(stats.get('1stIn', 1), 1)
                scaled = self.archetype_model['scaler'].transform([[s_elo, r_elo, ace, df, fw]])
                return self.archetype_model['kmeans'].predict(scaled)[0]
                
            arch1 = get_archetype(id1)
            arch2 = get_archetype(id2)
            
            global_arch_records = self.player_stats.get('GLOBAL_ARCH_RECORDS', {})
            if arch1 != -1 and arch2 != -1:
                r1 = global_arch_records.get((arch1, arch2), {'wins': 0, 'matches': 0})
                if r1['matches'] > 0: w_winrate_vs_l_arch = r1['wins'] / r1['matches']
                r2 = global_arch_records.get((arch2, arch1), {'wins': 0, 'matches': 0})
                if r2['matches'] > 0: l_winrate_vs_w_arch = r2['wins'] / r2['matches']
        
        court_speed = self._get_court_speed(surface)
        
        imp_prob1, imp_prob2 = shin_probabilities(p1_odds or 1.9, p2_odds or 1.9)
        
        level_map = {'G': 5, 'M': 4, 'A': 3, 'C': 2, 'F': 6, 'D': 1}
        t_level_int = level_map.get(level, 0)
        
        indoor = 1 if surface == 'Carpet' else 0
        
        best_of = 5 if level == 'G' else 3
        p_w = prof1[2]
        p_l = prof2[2]
        prob_markov = predict_match(p_w, p_l, prof1[4], prof2[4], best_of=best_of)
        
        features = [
            elo1, elo2, surf_elo1, surf_elo2, delta_elo,
            serve_elo1, return_elo1, serve_elo2, return_elo2,
            id1_h2h_rate, id2_h2h_rate, id1_surf_h2h_rate, id2_surf_h2h_rate,
            age1, age2, ht1, ht2, rank1, rank2, delta_rank,
            indoor, 0, 0, altitude, court_speed, rest1, rest2, imp_prob1, imp_prob2,
            fatigue_A, fatigue_B, 0, 0,
            w_winrate_vs_l_arch, l_winrate_vs_w_arch
        ] + prof1 + prof2 + form1 + form2 + list(mcp1) + list(mcp2) + ['mock_date', t_level_int, 1, p1_odds or np.nan, p2_odds or np.nan, prob_markov, "A", "B", surface]
        
        columns = ['A_elo', 'B_elo', 'A_surf_elo', 'B_surf_elo', 'delta_elo',
            'A_serve_elo', 'A_return_elo', 'B_serve_elo', 'B_return_elo',
            'A_h2h', 'B_h2h', 'A_surf_h2h', 'B_surf_h2h',
            'A_age', 'B_age', 'A_ht', 'B_ht', 'A_rank', 'B_rank', 'delta_rank',
            'indoor', 'A_streak', 'B_streak', 'altitude', 'court_speed', 'A_rest_days', 'B_rest_days', 'implied_prob_A', 'implied_prob_B',
            'A_fatigue', 'B_fatigue', 'A_inj', 'B_inj',
            'A_winrate_vs_B_arch', 'B_winrate_vs_A_arch'
        ] + [f'A_prof_{i}' for i in range(5)] + [f'B_prof_{i}' for i in range(5)] + \
        ['A_form_winrate', 'A_form_surf_winrate', 'B_form_winrate', 'B_form_surf_winrate'] + \
        ['A_style_agg', 'A_style_err', 'A_style_fb', 'A_style_net'] + \
        ['B_style_agg', 'B_style_err', 'B_style_fb', 'B_style_net'] + \
        ['tourney_date', 'tourney_level', 'target', 'b365_A', 'b365_B', 'prob_markov_A', 'A_name', 'B_name', 'surface']
        
        return pd.DataFrame([features], columns=columns)

    def predict_match_prob(self, name1, name2, surface, level='G', fatigue_A=0, fatigue_B=0, p1_odds=None, p2_odds=None, altitude=0):
        if name1 == "Bye": return 0.0
        if name2 == "Bye": return 1.0
        
        id1 = self.name_to_id.get(name1)
        id2 = self.name_to_id.get(name2)
        if not id1 or not id2: return 0.5
        
        df_AB = self._build_features(id1, id2, surface, level, fatigue_A, fatigue_B, p1_odds, p2_odds, altitude)
        df_BA = self._build_features(id2, id1, surface, level, fatigue_B, fatigue_A, p2_odds, p1_odds, altitude)
        
        drop_cols = ['target', 'tourney_date', 'b365_A', 'b365_B', 'A_name', 'B_name', 'surface']
        X_AB = df_AB.drop(columns=[c for c in drop_cols if c in df_AB.columns])
        X_BA = df_BA.drop(columns=[c for c in drop_cols if c in df_BA.columns])
        
        models_to_average = []
        if f"{surface}_Ensemble" in self.models:
            models_to_average.append(self.models[f"{surface}_Ensemble"])
        if "Global_Ensemble" in self.models:
            models_to_average.append(self.models["Global_Ensemble"])
            
        if not models_to_average and self.models:
            models_to_average.append(list(self.models.values())[0])
            
        if not models_to_average:
            return self.elo_sys.expected_score(self.elo_sys.get_elo(id1), self.elo_sys.get_elo(id2))
            
        prob_AB = np.mean([m.predict_proba(X_AB)[0, 1] for m in models_to_average])
        prob_BA = np.mean([m.predict_proba(X_BA)[0, 1] for m in models_to_average])
        
        # Symmetric prediction
        return (prob_AB + (1 - prob_BA)) / 2.0
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
        # Pad players list to the next power of 2 with "Bye"
        def next_power_of_2(x):
            return 1 if x == 0 else 2**(x - 1).bit_length()
            
        padded_players = list(players)
        target_len = next_power_of_2(len(padded_players))
        while len(padded_players) < target_len:
            padded_players.append("Bye")
            
        n = len(padded_players)
        P = np.zeros((n, n))
        
        # 1. Fetch all DB stats
        ids = [self.name_to_id.get(p) for p in padded_players if self.name_to_id.get(p)]
        mcp_data = {}
        if ids:
            id_str = ','.join('?' for _ in ids)
            conn = sqlite3.connect('tennis_database.db')
            c = conn.cursor()
            c.execute(f"SELECT id, aggressiveness, ue_rate, fh_preference, net_tendency FROM Players WHERE id IN ({id_str})", ids)
            for row in c.fetchall():
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
            if not p_id: continue
            row = conn.execute("SELECT winner_age, winner_ht, winner_rank, loser_age, loser_ht, loser_rank, winner_id FROM Matches WHERE winner_id = ? OR loser_id = ? ORDER BY tourney_date DESC LIMIT 1", (p_id, p_id)).fetchone()
            
            rest = 2.0
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
                'serve_elo': getattr(self.elo_sys, 'serve_elo', {}).get(p_id, self.elo_sys.default_elo),
                'return_elo': getattr(self.elo_sys, 'return_elo', {}).get(p_id, self.elo_sys.default_elo),
                'prof': self.get_profile(p_id),
                'form': self.get_form(p_id, surface),
                'mcp': mcp_data.get(p_id, [0.15, 0.35, 0.60, 0.10]),
                'age': age,
                'ht': ht,
                'rank': rank,
                'rest': rest
            }
        conn.close()
            
        # 3. Build batch features
        features_list = []
        indices_list = []
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = padded_players[i], padded_players[j]
                
                if p1 == "Bye":
                    P[i, j], P[j, i] = 0.0, 1.0
                    continue
                if p2 == "Bye":
                    P[i, j], P[j, i] = 1.0, 0.0
                    continue
                
                d1, d2 = player_dict.get(p1), player_dict.get(p2)
                if not d1 or not d2:
                    continue
                    
                id1 = self.name_to_id.get(p1)
                id2 = self.name_to_id.get(p2)
                h2h_records = self.player_stats.get('GLOBAL_H2H_RECORDS', {})
                surf_h2h_records = self.player_stats.get('SURF_H2H_RECORDS', {})
                pair_key = tuple(sorted([id1, id2]))
                
                w_h2h_wins = h2h_records.get(pair_key, {}).get(id1, 0)
                l_h2h_wins = h2h_records.get(pair_key, {}).get(id2, 0)
                total_h2h = w_h2h_wins + l_h2h_wins
                id1_h2h_rate = w_h2h_wins / total_h2h if total_h2h > 0 else 0.5
                id2_h2h_rate = l_h2h_wins / total_h2h if total_h2h > 0 else 0.5
                
                w_surf_wins = surf_h2h_records.get(surface, {}).get(pair_key, {}).get(id1, 0)
                l_surf_wins = surf_h2h_records.get(surface, {}).get(pair_key, {}).get(id2, 0)
                total_surf_h2h = w_surf_wins + l_surf_wins
                id1_surf_h2h_rate = w_surf_wins / total_surf_h2h if total_surf_h2h > 0 else 0.5
                id2_surf_h2h_rate = l_surf_wins / total_surf_h2h if total_surf_h2h > 0 else 0.5
                    
                indoor = 1 if surface == 'Carpet' else 0
                altitude = 0
                delta_elo = d1['elo'] - d2['elo']
                delta_rank = d1['rank'] - d2['rank']
                
                imp_prob1 = self.elo_sys.expected_score(d1['elo'], d2['elo'])
                imp_prob2 = self.elo_sys.expected_score(d2['elo'], d1['elo'])
                
                features = [
                    d1['elo'], d2['elo'], d1['surf_elo'], d2['surf_elo'], delta_elo,
                    d1['serve_elo'], d1['return_elo'], d2['serve_elo'], d2['return_elo'],
                    id1_h2h_rate, id2_h2h_rate, id1_surf_h2h_rate, id2_surf_h2h_rate,
                    d1['age'], d2['age'], d1['ht'], d2['ht'], d1['rank'], d2['rank'], delta_rank,
                    indoor, 0, 0, altitude, d1['rest'], d2['rest'], imp_prob1, imp_prob2
                ] + d1['prof'] + d2['prof'] + d1['form'] + d2['form'] + d1['mcp'] + d2['mcp']
                
                features_list.append(features)
                indices_list.append((i, j))
                
        # 4. Batch Predict
        if features_list:
            cols = ['A_elo', 'B_elo', 'A_surf_elo', 'B_surf_elo', 'delta_elo',
                'A_serve_elo', 'A_return_elo', 'B_serve_elo', 'B_return_elo',
                'A_h2h', 'B_h2h', 'A_surf_h2h', 'B_surf_h2h',
                'A_age', 'B_age', 'A_ht', 'B_ht', 'A_rank', 'B_rank', 'delta_rank',
                'indoor', 'A_streak', 'B_streak', 'altitude', 'A_rest_days', 'B_rest_days', 'implied_prob_A', 'implied_prob_B',
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
