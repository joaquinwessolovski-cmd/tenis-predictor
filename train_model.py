import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import sqlite3
import argparse
import joblib
import json
import re
import pickle
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from thefuzz import process
from markov_model import predict_match

DATA_DIR = "data 2/tennis_atp"

def shin_probabilities(odds_A, odds_B):
    if odds_A <= 1 or odds_B <= 1:
        if odds_A > 0 and odds_B > 0:
            pA = 1/odds_A; pB = 1/odds_B
            return pA/(pA+pB), pB/(pA+pB)
        return 0.5, 0.5
        
    inv_A = 1.0 / odds_A
    inv_B = 1.0 / odds_B
    sum_inv = inv_A + inv_B
    if sum_inv <= 1.0:
        return inv_A/sum_inv, inv_B/sum_inv
        
    def calc_sum(z):
        pA = ((z**2 + 4*(1-z)*(inv_A**2 / sum_inv))**0.5 - z) / (2*(1-z))
        pB = ((z**2 + 4*(1-z)*(inv_B**2 / sum_inv))**0.5 - z) / (2*(1-z))
        return pA + pB
        
    low, high = 0.0, 0.999
    for _ in range(20):
        mid = (low + high) / 2
        if calc_sum(mid) > 1.0:
            low = mid
        else:
            high = mid
            
    z = (low + high) / 2
    pA = ((z**2 + 4*(1-z)*(inv_A**2 / sum_inv))**0.5 - z) / (2*(1-z))
    pB = ((z**2 + 4*(1-z)*(inv_B**2 / sum_inv))**0.5 - z) / (2*(1-z))
    
    return pA/(pA+pB), pB/(pA+pB)

def load_data(start_year=2000, end_year=2026):
    print("Loading data from SQLite database...")
    import sqlite3
    conn = sqlite3.connect('tennis_database.db')
    
    query = f"""
        SELECT m.*, pw.full_name as winner_name, pl.full_name as loser_name
        FROM Matches m
        JOIN Players pw ON m.winner_id = pw.id
        JOIN Players pl ON m.loser_id = pl.id
        WHERE CAST(SUBSTR(CAST(m.tourney_date AS TEXT), 1, 4) AS INTEGER) BETWEEN {start_year} AND {end_year}
        ORDER BY m.tourney_date ASC, m.id ASC
    """
    
    frame = pd.read_sql_query(query, conn)
    conn.close()
    
    if not frame.empty:
        # Ensure tourney_date is datetime
        frame['tourney_date'] = pd.to_datetime(frame['tourney_date'], errors='coerce')
        frame = frame.dropna(subset=['winner_id', 'loser_id'])
        
    return frame

class EloSystem:
    def __init__(self, k=32, surface_k=32, inactivity_decay=0.1):
        self.overall_elo = {}
        self.surface_elo = {'Hard': {}, 'Clay': {}, 'Grass': {}, 'Carpet': {}}
        self.serve_elo = {}
        self.return_elo = {}
        self.last_played = {}
        self.k = k
        self.surface_k = surface_k
        self.default_elo = 1500
        self.inactivity_decay = inactivity_decay # Points lost per day after 30 days
        
    def _apply_decay(self, elo_val, last_date, current_date):
        if not last_date or not current_date:
            return elo_val
        dt = (current_date - last_date).days
        if dt > 180:
            return max(self.default_elo, elo_val - (dt - 180) * self.inactivity_decay)
        return elo_val
        
    def get_elo(self, player_id, surface=None, current_date=None):
        last_date = self.last_played.get(player_id)
        
        if surface and surface in self.surface_elo:
            elo = self.surface_elo[surface].get(player_id, self.default_elo)
            if current_date:
                return self._apply_decay(elo, last_date, current_date)
            return elo
            
        elo = self.overall_elo.get(player_id, self.default_elo)
        if current_date:
            return self._apply_decay(elo, last_date, current_date)
        return elo

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, winner_id, loser_id, surface, current_date, w_svpt=0, w_won=0, l_svpt=0, l_won=0, tourney_level='A'):
        win_elo = self.get_elo(winner_id, current_date=current_date)
        los_elo = self.get_elo(loser_id, current_date=current_date)
        
        current_k = self.k
        current_surf_k = self.surface_k
        
        expected_win = self.expected_score(win_elo, los_elo)
        
        self.overall_elo[winner_id] = win_elo + current_k * (1 - expected_win)
        self.overall_elo[loser_id] = los_elo + current_k * (0 - (1 - expected_win))
        
        # Serve & Return ELO logic
        w_serve_elo = self.serve_elo.get(winner_id, self.default_elo)
        w_return_elo = self.return_elo.get(winner_id, self.default_elo)
        l_serve_elo = self.serve_elo.get(loser_id, self.default_elo)
        l_return_elo = self.return_elo.get(loser_id, self.default_elo)
        
        if w_svpt > 0 and l_svpt > 0:
            w_serve_rate = w_won / w_svpt
            l_serve_rate = l_won / l_svpt
            
            exp_w_serve = self.expected_score(w_serve_elo, l_return_elo)
            exp_l_serve = self.expected_score(l_serve_elo, w_return_elo)
            
            self.serve_elo[winner_id] = w_serve_elo + current_k * (w_serve_rate - exp_w_serve)
            self.return_elo[loser_id] = l_return_elo + current_k * ((1 - w_serve_rate) - (1 - exp_w_serve))
            
            self.serve_elo[loser_id] = l_serve_elo + current_k * (l_serve_rate - exp_l_serve)
            self.return_elo[winner_id] = w_return_elo + current_k * ((1 - l_serve_rate) - (1 - exp_l_serve))
            
        if surface in self.surface_elo:
            last_date_w = self.last_played.get(winner_id)
            last_date_l = self.last_played.get(loser_id)
            
            win_surf = self.surface_elo[surface].get(winner_id, self.default_elo)
            los_surf = self.surface_elo[surface].get(loser_id, self.default_elo)
            
            win_surf = self._apply_decay(win_surf, last_date_w, current_date)
            los_surf = self._apply_decay(los_surf, last_date_l, current_date)
            
            exp_win_surf = self.expected_score(win_surf, los_surf)
            
            self.surface_elo[surface][winner_id] = win_surf + current_surf_k * (1 - exp_win_surf)
            self.surface_elo[surface][loser_id] = los_surf + current_surf_k * (0 - (1 - exp_win_surf))
            
        self.last_played[winner_id] = current_date
        self.last_played[loser_id] = current_date

def build_dataset(df, skip_challenger=False):
    elo_sys = EloSystem(k=32, surface_k=32, inactivity_decay=0.5)
    dataset = []
    elo_history = []
    stats_cols = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'bpSaved', 'bpFaced', 'ret_pt', 'ret_won']
    player_stats = {}
    player_form = {}
    h2h_records = {}
    surf_h2h_records = {}
    streak = {}
    delta_time = 0.005
    
    def update_stats(p_id, prefix, opp_prefix, row, current_date, is_win, surf):
        if p_id not in player_stats:
            player_stats[p_id] = {col: 0.0 for col in stats_cols}
            player_stats[p_id]['matches'] = 0
            player_stats[p_id]['last_date'] = current_date
            player_stats[p_id]['rank_points'] = 0 # Default points for unranked
            player_form[p_id] = {'all': [], 'surf': {'Hard': [], 'Clay': [], 'Grass': [], 'Carpet': []}}
            
        # Apply EMA decay to stats
        dt = (current_date - player_stats[p_id]['last_date']).days if player_stats[p_id]['last_date'] else 0
        decay = np.exp(-delta_time * dt)
        
        for col in ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'bpSaved', 'bpFaced']:
            val = row.get(f'{prefix}_{col}')
            if not pd.isna(val):
                try:
                    player_stats[p_id][col] = player_stats[p_id][col] * decay + float(val)
                except (ValueError, TypeError):
                    pass
                    
        # Update return stats using opponent's serve stats
        opp_svpt = row.get(f'{opp_prefix}_svpt')
        opp_1stWon = row.get(f'{opp_prefix}_1stWon')
        opp_2ndWon = row.get(f'{opp_prefix}_2ndWon')
        
        if pd.notna(opp_svpt) and opp_svpt > 0:
            ret_pt = float(opp_svpt)
            opp_pts_won = 0
            if pd.notna(opp_1stWon) and pd.notna(opp_2ndWon):
                opp_pts_won = float(opp_1stWon) + float(opp_2ndWon)
            ret_won = ret_pt - opp_pts_won
            
            player_stats[p_id]['ret_pt'] = player_stats[p_id]['ret_pt'] * decay + ret_pt
            player_stats[p_id]['ret_won'] = player_stats[p_id]['ret_won'] * decay + ret_won
                
        player_stats[p_id]['matches'] = player_stats[p_id]['matches'] * decay + 1
        player_stats[p_id]['last_date'] = current_date
        
        # Update Form
        player_form[p_id]['all'].append(1 if is_win else 0)
        if len(player_form[p_id]['all']) > 10:
            player_form[p_id]['all'].pop(0)
            
        if surf in player_form[p_id]['surf']:
            player_form[p_id]['surf'][surf].append(1 if is_win else 0)
            if len(player_form[p_id]['surf'][surf]) > 10:
                player_form[p_id]['surf'][surf].pop(0)
        
    def get_profile(p_id):
        if p_id not in player_stats or player_stats[p_id]['matches'] < 1:
            return [0, 0, 0, 0, 0]
            
        st = player_stats[p_id]
        svpt = max(st['svpt'], 1)
        
        ace_rate = st['ace'] / svpt
        df_rate = st['df'] / svpt
        first_win_rate = st['1stWon'] / max(st['1stIn'], 1)
        second_win_rate = st['2ndWon'] / max(svpt - st['1stIn'], 1)
        
        # BP conversion rates
        bp_saved_rate = st['bpSaved'] / max(st['bpFaced'], 1)
        
        return [ace_rate, df_rate, first_win_rate, second_win_rate, bp_saved_rate]
        
    def get_form(p_id, surf):
        if p_id not in player_form:
            return [0.5, 0.5] # Default to 50% form
        all_form = np.mean(player_form[p_id]['all']) if player_form[p_id]['all'] else 0.5
        surf_form = np.mean(player_form[p_id]['surf'].get(surf, [])) if player_form[p_id]['surf'].get(surf) else 0.5
        return [all_form, surf_form]
        
    style_profiles = {}
    style_medians = {}
    if os.path.exists('../player_style_profiles.pkl'):
        with open('../player_style_profiles.pkl', 'rb') as f:
            mcp_data = pickle.load(f)
            style_profiles = mcp_data['profiles']
            style_medians = mcp_data['medians']
            
    def_agg = style_medians.get('aggressiveness', 0.15)
    def_ue = style_medians.get('ue_rate', 0.18)
    def_fh = style_medians.get('fh_preference', 0.66)
    def_net = style_medians.get('net_tendency', 0.17)
    
    def get_style(pid):
        sp = style_profiles.get(pid, {})
        return [
            sp.get('aggressiveness', def_agg),
            sp.get('ue_rate', def_ue),
            sp.get('fh_preference', def_fh),
            sp.get('net_tendency', def_net)
        ]
        
    injury_tracker = {}
    player_fatigue_log = {}
    global_arch_records = {}
    tourney_aces = {}
    
    level_map = {'G': 5, 'M': 4, 'A': 3, 'C': 2, 'F': 6, 'D': 1}
    
    def get_7day_fatigue(pid, date):
        if pid not in player_fatigue_log: return 0
        player_fatigue_log[pid] = [(d, m) for d, m in player_fatigue_log[pid] if (date - d).days <= 7]
        return sum(m for d, m in player_fatigue_log[pid])
        
    def add_fatigue(pid, date, mins):
        if pid not in player_fatigue_log: player_fatigue_log[pid] = []
        player_fatigue_log[pid].append((date, mins))
    import re
    
    archetype_model = None
    if os.path.exists('archetype_model.pkl'):
        with open('archetype_model.pkl', 'rb') as f:
            archetype_model = pickle.load(f)
            
    archetype_records = {} # {pid: {arch_id: {'wins': 0, 'matches': 0}}}
    
    def get_archetype(pid):
        if not archetype_model or pid not in player_stats or player_stats[pid]['svpt'] < 100:
            return -1
        stats = player_stats[pid]
        s_elo = elo_sys.serve_elo.get(pid, elo_sys.default_elo)
        r_elo = elo_sys.return_elo.get(pid, elo_sys.default_elo)
        ace = stats['ace'] / max(stats['svpt'], 1)
        df = stats['df'] / max(stats['svpt'], 1)
        fw = stats['1stWon'] / max(stats['1stIn'], 1)
        scaled = archetype_model['scaler'].transform([[s_elo, r_elo, ace, df, fw]])
        return archetype_model['kmeans'].predict(scaled)[0]
        
    for idx, row in df.iterrows():
        w_id = row['winner_id']
        l_id = row['loser_id']
        surf = row['surface']
        current_date = row['tourney_date']
        tourney_level = row.get('tourney_level', 'A')
        
        w_elo = elo_sys.get_elo(w_id, current_date=current_date)
        l_elo = elo_sys.get_elo(l_id, current_date=current_date)
        w_surf_elo = elo_sys.get_elo(w_id, surf, current_date=current_date)
        l_surf_elo = elo_sys.get_elo(l_id, surf, current_date=current_date)
        
        w_arch = get_archetype(w_id)
        l_arch = get_archetype(l_id)
        
        w_winrate_vs_l_arch = 0.5
        l_winrate_vs_w_arch = 0.5
        
        if w_arch != -1 and l_arch != -1:
            rec_w = global_arch_records.get((w_arch, l_arch), {'wins': 0, 'matches': 0})
            if rec_w['matches'] > 0: w_winrate_vs_l_arch = rec_w['wins'] / rec_w['matches']
            
            rec_l = global_arch_records.get((l_arch, w_arch), {'wins': 0, 'matches': 0})
            if rec_l['matches'] > 0: l_winrate_vs_w_arch = rec_l['wins'] / rec_l['matches']
                
        w_serve_elo = elo_sys.serve_elo.get(w_id, elo_sys.default_elo)
        w_return_elo = elo_sys.return_elo.get(w_id, elo_sys.default_elo)
        l_serve_elo = elo_sys.serve_elo.get(l_id, elo_sys.default_elo)
        l_return_elo = elo_sys.return_elo.get(l_id, elo_sys.default_elo)
        
        tourney_name = row.get('tourney_name', '')
        court_speed = 0.08
        if tourney_name in tourney_aces and len(tourney_aces[tourney_name]) > 10:
            t_sv = sum(s for s, a in tourney_aces[tourney_name])
            t_a = sum(a for s, a in tourney_aces[tourney_name])
            if t_sv > 0: court_speed = t_a / t_sv
            
        t_level_int = level_map.get(tourney_level, 0)
        
        # Calculate rest days
        w_rest = 7 if not elo_sys.last_played.get(w_id) else max(0, (current_date - elo_sys.last_played[w_id]).days)
        l_rest = 7 if not elo_sys.last_played.get(l_id) else max(0, (current_date - elo_sys.last_played[l_id]).days)
        
        altitude = float(row.get('altitude') or 0.0)
        
        b365_w = float(row.get('b365_w') or np.nan)
        b365_l = float(row.get('b365_l') or np.nan)
        
        implied_prob_w, implied_prob_l = shin_probabilities(b365_w, b365_l)
        
        w_fatigue = get_7day_fatigue(w_id, current_date)
        l_fatigue = get_7day_fatigue(l_id, current_date)
        
        w_inj = 1 if injury_tracker.get(w_id) and (current_date - injury_tracker[w_id]).days < 45 else 0
        l_inj = 1 if injury_tracker.get(l_id) and (current_date - injury_tracker[l_id]).days < 45 else 0
        
        score_str = str(row.get('score', ''))
        if pd.notna(row.get('minutes')) and row['minutes'] > 0:
            match_mins = row['minutes']
        else:
            games = sum([int(g) for g in re.findall(r'\d+', score_str)])
            match_mins = games * 4
            
        add_fatigue(w_id, current_date, match_mins)
        add_fatigue(l_id, current_date, match_mins)
        
        if 'RET' in score_str or 'W/O' in score_str:
            injury_tracker[l_id] = current_date
        
        indoor = 1 if row.get('indoor') == 'I' else 0
        w_age = row.get('winner_age', np.nan)
        l_age = row.get('loser_age', np.nan)
        w_ht = row.get('winner_ht', np.nan)
        l_ht = row.get('loser_ht', np.nan)
        w_rank = row.get('winner_rank', np.nan)
        l_rank = row.get('loser_rank', np.nan)
        
        w_prof = get_profile(w_id)
        l_prof = get_profile(l_id)
        
        w_form = get_form(w_id, surf)
        l_form = get_form(l_id, surf)
        
        w_style = get_style(w_id)
        l_style = get_style(l_id)        
        
        # update global archetype records
        if w_arch != -1 and l_arch != -1:
            if (w_arch, l_arch) not in global_arch_records: global_arch_records[(w_arch, l_arch)] = {'wins': 0, 'matches': 0}
            global_arch_records[(w_arch, l_arch)]['wins'] += 1
            global_arch_records[(w_arch, l_arch)]['matches'] += 1
            
            if (l_arch, w_arch) not in global_arch_records: global_arch_records[(l_arch, w_arch)] = {'wins': 0, 'matches': 0}
            global_arch_records[(l_arch, w_arch)]['matches'] += 1
            
        # Update H2H
        pair_key = f"{min(w_id, l_id)}_{max(w_id, l_id)}"
        surf_pair_key = f"{pair_key}_{surf}"
        
        if pair_key in h2h_records:
            total_h2h = h2h_records[pair_key].get(w_id, 0) + h2h_records[pair_key].get(l_id, 0)
            w_h2h_rate = h2h_records[pair_key].get(w_id, 0) / total_h2h if total_h2h > 0 else 0.5
            l_h2h_rate = h2h_records[pair_key].get(l_id, 0) / total_h2h if total_h2h > 0 else 0.5
        else:
            w_h2h_rate, l_h2h_rate = 0.5, 0.5
            
        if surf_pair_key in surf_h2h_records:
            total_surf_h2h = surf_h2h_records[surf_pair_key].get(w_id, 0) + surf_h2h_records[surf_pair_key].get(l_id, 0)
            w_surf_h2h_rate = surf_h2h_records[surf_pair_key].get(w_id, 0) / total_surf_h2h if total_surf_h2h > 0 else 0.5
            l_surf_h2h_rate = surf_h2h_records[surf_pair_key].get(l_id, 0) / total_surf_h2h if total_surf_h2h > 0 else 0.5
        else:
            w_surf_h2h_rate, l_surf_h2h_rate = 0.5, 0.5
            
        w_streak = streak.get(w_id, 0)
        l_streak = streak.get(l_id, 0)
        
        if player_stats.get(w_id, {}).get('matches', 0) > 2 and player_stats.get(l_id, {}).get('matches', 0) > 2:
            if not skip_challenger or tourney_level != 'C':
                # Convert odds to probabilities, default to ELO-based prob if missing
                implied_prob_w = (1/b365_w) if pd.notna(b365_w) and b365_w > 1 else elo_sys.expected_score(w_elo, l_elo)
                implied_prob_l = (1/b365_l) if pd.notna(b365_l) and b365_l > 1 else elo_sys.expected_score(l_elo, w_elo)
                
                # Calculate Markov model probabilities
                w_serve_win = (player_stats[w_id]['1stWon'] + player_stats[w_id]['2ndWon']) / max(player_stats[w_id]['svpt'], 1)
                l_ret_win = player_stats[l_id]['ret_won'] / max(player_stats[l_id]['ret_pt'], 1)
                
                l_serve_win = (player_stats[l_id]['1stWon'] + player_stats[l_id]['2ndWon']) / max(player_stats[l_id]['svpt'], 1)
                w_ret_win = player_stats[w_id]['ret_won'] / max(player_stats[w_id]['ret_pt'], 1)
                
                p_w = (w_serve_win + (1 - l_ret_win)) / 2.0
                p_l = (l_serve_win + (1 - w_ret_win)) / 2.0
                
                p_w = min(max(p_w, 0.4), 0.8)
                p_l = min(max(p_l, 0.4), 0.8)
                
                best_of = 5 if tourney_level == 'G' else 3
                
                w_bp_saved = player_stats[w_id].get('bpSaved', 0) / max(player_stats[w_id].get('bpFaced', 1), 1)
                l_bp_saved = player_stats[l_id].get('bpSaved', 0) / max(player_stats[l_id].get('bpFaced', 1), 1)
                
                if np.random.rand() > 0.5:
                    # w_id is A, l_id is B
                    prob_markov_A = predict_match(p_w, p_l, w_bp_saved, l_bp_saved, best_of=best_of)
                    delta_elo = w_elo - l_elo
                    delta_rank = w_rank - l_rank
                    features = [
                        w_elo, l_elo, w_surf_elo, l_surf_elo, delta_elo, 
                        w_serve_elo, w_return_elo, l_serve_elo, l_return_elo,
                        w_h2h_rate, l_h2h_rate, w_surf_h2h_rate, l_surf_h2h_rate,
                        w_age, l_age, w_ht, l_ht, w_rank, l_rank, delta_rank,
                        indoor, w_streak, l_streak, altitude, court_speed, w_rest, l_rest, implied_prob_w, implied_prob_l,
                        w_fatigue, l_fatigue, w_inj, l_inj,
                        w_winrate_vs_l_arch, l_winrate_vs_w_arch
                    ] + w_prof + l_prof + w_form + l_form + w_style + l_style + [current_date, t_level_int, 1, b365_w, b365_l, prob_markov_A, row['winner_name'], row['loser_name'], surf]
                else:
                    # l_id is A, w_id is B
                    prob_markov_A = predict_match(p_l, p_w, l_bp_saved, w_bp_saved, best_of=best_of)
                    delta_elo = l_elo - w_elo
                    delta_rank = l_rank - w_rank
                    features = [
                        l_elo, w_elo, l_surf_elo, w_surf_elo, delta_elo,
                        l_serve_elo, l_return_elo, w_serve_elo, w_return_elo,
                        l_h2h_rate, w_h2h_rate, l_surf_h2h_rate, w_surf_h2h_rate,
                        l_age, w_age, l_ht, w_ht, l_rank, w_rank, delta_rank,
                        indoor, l_streak, w_streak, altitude, court_speed, l_rest, w_rest, implied_prob_l, implied_prob_w,
                        l_fatigue, w_fatigue, l_inj, w_inj,
                        l_winrate_vs_w_arch, w_winrate_vs_l_arch
                    ] + l_prof + w_prof + l_form + w_form + l_style + w_style + [current_date, t_level_int, 0, b365_l, b365_w, prob_markov_A, row['loser_name'], row['winner_name'], surf]
                    
                dataset.append(features)
            
        if pair_key not in h2h_records:
            h2h_records[pair_key] = {w_id: 0, l_id: 0}
        h2h_records[pair_key][w_id] = h2h_records[pair_key].get(w_id, 0) + 1
        
        if surf_pair_key not in surf_h2h_records:
            surf_h2h_records[surf_pair_key] = {w_id: 0, l_id: 0}
        surf_h2h_records[surf_pair_key][w_id] = surf_h2h_records[surf_pair_key].get(w_id, 0) + 1
        
        streak[w_id] = w_streak + 1
        streak[l_id] = 0
            
        w_svpt = float(row.get('w_svpt') or 0)
        w_won = float(row.get('w_1stWon') or 0) + float(row.get('w_2ndWon') or 0)
        w_ace = float(row.get('w_ace') or 0)
        l_ace = float(row.get('l_ace') or 0)
        l_svpt = float(row.get('l_svpt') or 0)
        l_won = float(row.get('l_1stWon') or 0) + float(row.get('l_2ndWon') or 0)
        if tourney_name not in tourney_aces: tourney_aces[tourney_name] = []
        tourney_aces[tourney_name].append((w_svpt + l_svpt, w_ace + l_ace))
        
        elo_sys.update(w_id, l_id, surf, current_date, w_svpt, w_won, l_svpt, l_won, tourney_level)
        update_stats(w_id, 'w', 'l', row, current_date, True, surf)
        update_stats(l_id, 'l', 'w', row, current_date, False, surf)
        
        # Track ELO history
        dt_str = str(current_date)
        elo_history.append((dt_str, w_id, elo_sys.get_elo(w_id), elo_sys.get_elo(w_id, 'Hard'), elo_sys.get_elo(w_id, 'Clay'), elo_sys.get_elo(w_id, 'Grass')))
        elo_history.append((dt_str, l_id, elo_sys.get_elo(l_id), elo_sys.get_elo(l_id, 'Hard'), elo_sys.get_elo(l_id, 'Clay'), elo_sys.get_elo(l_id, 'Grass')))
        
    import sqlite3
    try:
        conn = sqlite3.connect('tennis_database.db')
        c = conn.cursor()
        print("Guardando historial de ELO en SQLite...")
        c.execute('DROP TABLE IF EXISTS EloHistory')
        c.execute('''CREATE TABLE EloHistory (
            date TEXT,
            player_id INTEGER,
            elo_global REAL,
            elo_hard REAL,
            elo_clay REAL,
            elo_grass REAL
        )''')
        c.executemany('INSERT INTO EloHistory VALUES (?,?,?,?,?,?)', elo_history)
        c.execute('CREATE INDEX idx_elo_hist_pid ON EloHistory(player_id)')
        c.execute('CREATE INDEX idx_elo_hist_date ON EloHistory(date)')
        conn.commit()
        conn.close()
        print("¡Historial guardado!")
    except Exception as e:
        print(f"Error al guardar EloHistory: {e}")
        
    # We must save player_form as part of player_stats so UI can use it
    for pid in player_stats:
        player_stats[pid]['form'] = player_form.get(pid, {'all': [], 'surf': {}})
        
    player_stats['GLOBAL_H2H_RECORDS'] = h2h_records
    player_stats['ARCHETYPE_RECORDS'] = archetype_records
        
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
    
    df = pd.DataFrame(dataset, columns=columns)
    return df, elo_sys, player_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', type=int, default=None, help='Year to backtest on (e.g. 2026)')
    parser.add_argument('--backtest-date', type=str, default=None, help='YYYYMMDD date to split train/test')
    parser.add_argument('--no-challenger', action='store_true', help='Skip Challenger matches for training/testing')
    args = parser.parse_args()

    print("Loading data...")
    df = load_data() # Loads Kaggle + Scraped data
    
    if df.empty:
        print("No matches found. Ensure the dataset exists.")
        return
        
    print(f"Loaded {len(df)} matches. Building dataset...")
    df_feat, elo_sys, player_stats = build_dataset(df, skip_challenger=args.no_challenger)
    
    print(f"Generated {len(df_feat)} training samples.")
    
    if args.backtest or args.backtest_date:
        if args.backtest_date:
            print(f"Backtesting from date {args.backtest_date}...")
            # Convert tourney_date to string to do lexical comparison
            df_feat['tourney_date_str'] = df_feat['tourney_date'].dt.strftime('%Y%m%d')
            df_train = df_feat[df_feat['tourney_date_str'] < args.backtest_date].copy()
            df_test = df_feat[df_feat['tourney_date_str'] >= args.backtest_date].copy()
            df_train.drop('tourney_date_str', axis=1, inplace=True)
            df_test.drop('tourney_date_str', axis=1, inplace=True)
        else:
            print(f"Backtesting on year {args.backtest}...")
            df_train = df_feat[df_feat['tourney_date'].dt.year < args.backtest].copy()
            df_test = df_feat[df_feat['tourney_date'].dt.year == args.backtest].copy()
        
        X_train = df_train.drop(['target', 'tourney_date', 'tourney_level', 'odds_A', 'odds_B', 'markov_prob_A', 'A_name', 'B_name', 'surface'], axis=1, errors='ignore')
        y_train = df_train['target']
        
        X_test = df_test.drop(['target', 'tourney_date', 'tourney_level', 'odds_A', 'odds_B', 'markov_prob_A', 'A_name', 'B_name', 'surface'], axis=1, errors='ignore')
        y_test = df_test['target']
        
        if len(X_test) == 0:
            print("No test data found for backtest.")
            return
    else:
        X = df_feat.drop(['target', 'tourney_date', 'tourney_level', 'odds_A', 'odds_B', 'markov_prob_A', 'A_name', 'B_name', 'surface'], axis=1, errors='ignore')
        y = df_feat['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost with RandomizedSearchCV...")
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    }
    
    base_model = xgb.XGBClassifier(eval_metric='logloss')
    search = RandomizedSearchCV(base_model, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    
    model = search.best_estimator_
    print(f"Best params: {search.best_params_}")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, probs)
    ll = log_loss(y_test, probs)
    print(f"Overall Accuracy: {acc:.4f}, Brier Score: {brier:.4f}, Log-Loss: {ll:.4f}")
    
    if args.backtest:
        markov_probs = df_test['markov_prob_A']
        markov_preds = (markov_probs > 0.5).astype(int)
        m_acc = accuracy_score(y_test, markov_preds)
        m_brier = brier_score_loss(y_test, markov_probs)
        m_ll = log_loss(y_test, markov_probs)
        print(f"\n[MARKOV MODEL] Accuracy: {m_acc:.4f}, Brier: {m_brier:.4f}, Log-Loss: {m_ll:.4f}\n")
    
    if args.backtest:
        df_test['preds'] = preds
        df_test['probs'] = probs
        print("\n--- Accuracy by Tournament Level ---")
        for lvl in df_test['tourney_level'].unique():
            lvl_df = df_test[df_test['tourney_level'] == lvl]
            lvl_acc = accuracy_score(lvl_df['target'], lvl_df['preds'])
            print(f"Level '{lvl}': {lvl_acc:.4f} (N={len(lvl_df)})")
            
        print("\n--- Value Betting ROI Simulation ---")
        edges = [0.05, 0.025, 0.01, 0.005]
        for edge in edges:
            bets_placed = 0
            profit = 0.0
            
            for idx, row in df_test.iterrows():
                prob_A = row['probs']
                prob_B = 1 - prob_A
                odds_A = row['odds_A']
                odds_B = row['odds_B']
                imp_A = row['implied_prob_A']
                imp_B = row['implied_prob_B']
                target = row['target']
                
                if pd.notna(odds_A) and pd.notna(imp_A) and prob_A > (imp_A + edge):
                    bets_placed += 1
                    if target == 1:
                        profit += (odds_A - 1)
                    else:
                        profit -= 1
                elif pd.notna(odds_B) and pd.notna(imp_B) and prob_B > (imp_B + edge):
                    bets_placed += 1
                    if target == 0:
                        profit += (odds_B - 1)
                    else:
                        profit -= 1
                        
            roi = (profit / bets_placed) * 100 if bets_placed > 0 else 0
            print(f"Edge {edge*100:.1f}% -> Bets: {bets_placed} | Profit: {profit:.2f}u | ROI: {roi:.2f}%")
        print("------------------------------------\n")
    
    # Create player index mapping for the UI
    player_names = {}
    for idx, row in df.iterrows():
        player_names[row['winner_id']] = row['winner_name']
        player_names[row['loser_id']] = row['loser_name']
        
    print("Loading Tennis Abstract Elo from Scraper...")
    try:
        import sys
        sys.path.append('..')
        import api_data
        ta_df = api_data.fetch_tennis_abstract_elo()
        name_to_id = {v: k for k, v in player_names.items()}
        
        if not ta_df.empty:
            for idx, row in ta_df.iterrows():
                ta_name = row['Player']
                if pd.isna(ta_name) or not isinstance(ta_name, str):
                    continue
                ta_name = ta_name.replace('\xa0', ' ')
                p_id = name_to_id.get(ta_name)
                if not p_id:
                    match = process.extractOne(ta_name, list(name_to_id.keys()))
                    if match and match[1] >= 85:
                        p_id = name_to_id[match[0]]
                
                if p_id:
                    overall = float(row['Elo'])
                    elo_sys.overall_elo[p_id] = overall
                    
                    if not pd.isna(row.get('hElo')): elo_sys.surface_elo['Hard'][p_id] = float(row['hElo'])
                    if not pd.isna(row.get('cElo')): elo_sys.surface_elo['Clay'][p_id] = float(row['cElo'])
                    if not pd.isna(row.get('gElo')): elo_sys.surface_elo['Grass'][p_id] = float(row['gElo'])
    except Exception as e:
        print("Could not load Tennis Abstract Elo via scraper:", e)
        
    with open('tennis_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'elo_sys': elo_sys,
            'player_stats': player_stats
        }, f)
        
    with open('player_names.pkl', 'wb') as f:
        pickle.dump(player_names, f)
        
    print("Model and metadata saved.")

if __name__ == "__main__":
    main()
