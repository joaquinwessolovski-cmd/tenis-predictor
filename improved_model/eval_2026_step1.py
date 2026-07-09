import sqlite3
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os

class EloSystem:
    def __init__(self, k=32, surface_k=32, inactivity_decay=0.1):
        self.overall_elo = {}
        self.surface_elo = {'Hard': {}, 'Clay': {}, 'Grass': {}, 'Carpet': {}}
        self.last_played = {}
        self.k = k
        self.surface_k = surface_k
        self.default_elo = 1500
        self.inactivity_decay = inactivity_decay
        
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

    def update(self, winner_id, loser_id, surface, current_date, tourney_level='A'):
        pass

with open('tennis_model_step1.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    elo_sys = model_data['elo_sys']
    player_stats = model_data['player_stats']
    
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
        
def get_profile(p_id):
    if p_id not in player_stats or player_stats[p_id]['matches'] < 1:
        return [0, 0, 0, 0]
    st = player_stats[p_id]
    svpt = max(st['svpt'], 1)
    ace_rate = st['ace'] / svpt
    df_rate = st['df'] / svpt
    first_win_rate = st['1stWon'] / max(st['1stIn'], 1)
    second_win_rate = st['2ndWon'] / max(svpt - st['1stIn'], 1)
    return [ace_rate, df_rate, first_win_rate, second_win_rate]
    
def get_form(p_id, surf):
    st = player_stats.get(p_id, {})
    player_form = st.get('form', {'all': [], 'surf': {}})
    all_form = np.mean(player_form['all']) if player_form['all'] else 0.5
    surf_form = np.mean(player_form.get('surf', {}).get(surf, [])) if player_form.get('surf', {}).get(surf) else 0.5
    return [all_form, surf_form]
    
conn = sqlite3.connect('../tennis_database.db')
df = pd.read_sql_query("SELECT * FROM Matches WHERE tourney_date >= 20260101", conn)

features_list = []
targets = []

for idx, row in df.iterrows():
    w_id = row['winner_id']
    l_id = row['loser_id']
    surf = row['surface']
    
    w_elo = elo_sys.get_elo(w_id, current_date=None)
    l_elo = elo_sys.get_elo(l_id, current_date=None)
    w_surf_elo = elo_sys.get_elo(w_id, surf, current_date=None)
    l_surf_elo = elo_sys.get_elo(l_id, surf, current_date=None)
    
    w_prof = get_profile(w_id)
    l_prof = get_profile(l_id)
    
    w_form = get_form(w_id, surf)
    l_form = get_form(l_id, surf)
    
    w_style = get_style(w_id)
    l_style = get_style(l_id)
    
    if np.random.rand() > 0.5:
        feat = [w_elo, l_elo, w_surf_elo, l_surf_elo] + w_prof + l_prof + w_form + l_form + w_style + l_style
        target = 1
    else:
        feat = [l_elo, w_elo, l_surf_elo, w_surf_elo] + l_prof + w_prof + l_form + w_form + l_style + w_style
        target = 0
        
    features_list.append(feat)
    targets.append(target)
    
X = pd.DataFrame(features_list, columns=['A_elo', 'B_elo', 'A_surf_elo', 'B_surf_elo',
            'A_ace', 'A_df', 'A_1w', 'A_2w',
            'B_ace', 'B_df', 'B_1w', 'B_2w',
            'A_form_all', 'A_form_surf', 'B_form_all', 'B_form_surf',
            'A_agg', 'A_ue', 'A_fh', 'A_net',
            'B_agg', 'B_ue', 'B_fh', 'B_net'])
            
preds = model.predict(X)
acc = accuracy_score(targets, preds)
print(f"Precision: {acc:.4f} en {len(df)} partidos")
