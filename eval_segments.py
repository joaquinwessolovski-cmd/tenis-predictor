import sqlite3
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os

class EloSystem:
    def __init__(self, k=32, surface_k=32):
        self.elo_ratings = {}
        self.surface_elo = {}
        self.k = k
        self.surface_k = surface_k
    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    def get_elo(self, player_id, surface=None, current_date=None):
        if surface:
            return self.surface_elo.get(surface, {}).get(player_id, 1500)
        return self.elo_ratings.get(player_id, 1500)
    def update(self, winner_id, loser_id, surface, current_date, tourney_level='A'):
        pass
        
with open('improved_model/tennis_model_step2.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    elo_sys = model_data['elo_sys']
    player_stats = model_data['player_stats']
    
with open('improved_model/player_names_step2.pkl', 'rb') as f:
    player_names = pickle.load(f)

# Load rankings
rankings = {}
try:
    df_rank = pd.read_csv('rankingatp.csv', sep=';')
    for _, row in df_rank.iterrows():
        name = str(row['Player']).replace('\xa0', ' ')
        rank = row.iloc[-2]  # ATP Rank is second to last
        rankings[name] = float(rank) if not pd.isna(rank) else 999
except Exception as e:
    print(e)

style_profiles = {}
style_medians = {}
if os.path.exists('player_style_profiles.pkl'):
    with open('player_style_profiles.pkl', 'rb') as f:
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
        return [0, 0, 0, 0, 0]
    st = player_stats[p_id]
    svpt = max(st['svpt'], 1)
    ace_rate = st['ace'] / svpt
    df_rate = st['df'] / svpt
    first_win_rate = st['1stWon'] / max(st['1stIn'], 1)
    second_win_rate = st['2ndWon'] / max(svpt - st['1stIn'], 1)
    bp_saved_rate = st.get('bpSaved', 0) / max(st.get('bpFaced', 1), 1)
    return [ace_rate, df_rate, first_win_rate, second_win_rate, bp_saved_rate]
    
def get_form(p_id, surf):
    st = player_stats.get(p_id, {})
    player_form = st.get('form', {'all': [], 'surf': {}})
    all_form = np.mean(player_form['all']) if player_form['all'] else 0.5
    surf_form = np.mean(player_form.get('surf', {}).get(surf, [])) if player_form.get('surf', {}).get(surf) else 0.5
    return [all_form, surf_form]
    
conn = sqlite3.connect('tennis_database.db')
df = pd.read_sql_query("SELECT * FROM Matches WHERE tourney_date >= 20260101", conn)

results = []

for idx, row in df.iterrows():
    w_id = row['winner_id']
    l_id = row['loser_id']
    surf = row['surface']
    tourney_level = row.get('tourney_level', 'Unknown')
    round_name = row.get('round', 'Unknown')
    
    w_name = player_names.get(w_id, 'Unknown')
    l_name = player_names.get(l_id, 'Unknown')
    
    w_rank = rankings.get(w_name, 999)
    l_rank = rankings.get(l_name, 999)
    avg_rank = (w_rank + l_rank) / 2
    if avg_rank <= 20:
        rank_bucket = 'Top 20'
    elif avg_rank <= 50:
        rank_bucket = 'Top 21-50'
    elif avg_rank <= 100:
        rank_bucket = 'Top 51-100'
    else:
        rank_bucket = 'Outside Top 100'
    
    curr_date_str = str(row['tourney_date'])
    current_date = f"{curr_date_str[:4]}-{curr_date_str[4:6]}-{curr_date_str[6:]}"

    w_elo = elo_sys.get_elo(w_id, current_date=current_date)
    l_elo = elo_sys.get_elo(l_id, current_date=current_date)
    w_surf_elo = elo_sys.get_elo(w_id, surf, current_date=current_date)
    l_surf_elo = elo_sys.get_elo(l_id, surf, current_date=current_date)
    
    w_prof = get_profile(w_id)
    l_prof = get_profile(l_id)
    
    w_form = get_form(w_id, surf)
    l_form = get_form(l_id, surf)
    
    w_style = get_style(w_id)
    l_style = get_style(l_id)
    
    h2h_records = player_stats.get('GLOBAL_H2H_RECORDS', {})
    pair_key = tuple(sorted([w_id, l_id]))
    w_h2h_wins = h2h_records.get(pair_key, {}).get(w_id, 0)
    l_h2h_wins = h2h_records.get(pair_key, {}).get(l_id, 0)
    total_h2h = w_h2h_wins + l_h2h_wins
    if total_h2h > 0:
        w_h2h_rate = w_h2h_wins / total_h2h
        l_h2h_rate = l_h2h_wins / total_h2h
    else:
        w_h2h_rate = 0.5
        l_h2h_rate = 0.5
    
    if np.random.rand() > 0.5:
        feat = [w_elo, l_elo, w_surf_elo, l_surf_elo, w_h2h_rate, l_h2h_rate] + w_prof + l_prof + w_form + l_form + w_style + l_style
        target = 1
    else:
        feat = [l_elo, w_elo, l_surf_elo, w_surf_elo, l_h2h_rate, w_h2h_rate] + l_prof + w_prof + l_form + w_form + l_style + w_style
        target = 0
        
    results.append({
        'features': feat,
        'target': target,
        'surface': surf,
        'level': tourney_level,
        'round': round_name,
        'rank_bucket': rank_bucket
    })

res_df = pd.DataFrame(results)
X = pd.DataFrame(res_df['features'].tolist(), columns=['A_elo', 'B_elo', 'A_surf_elo', 'B_surf_elo', 'A_h2h', 'B_h2h',
            'A_ace', 'A_df', 'A_1w', 'A_2w', 'A_bp',
            'B_ace', 'B_df', 'B_1w', 'B_2w', 'B_bp',
            'A_form_all', 'A_form_surf', 'B_form_all', 'B_form_surf',
            'A_agg', 'A_ue', 'A_fh', 'A_net',
            'B_agg', 'B_ue', 'B_fh', 'B_net'])
X = X.fillna(0).astype(float)
            
preds = model.predict(X)
res_df['pred'] = preds
res_df['correct'] = res_df['target'] == res_df['pred']

print('\n=== Overall ===')
print(f"Accuracy: {res_df['correct'].mean():.4f} ({len(res_df)} matches)")

print('\n=== By Surface ===')
grouped = res_df.groupby('surface')['correct'].agg(['mean', 'count'])
for idx, row in grouped.iterrows():
    print(f"{idx}: {row['mean']:.4f} ({int(row['count'])} matches)")

print('\n=== By Tournament Type ===')
grouped = res_df.groupby('level')['correct'].agg(['mean', 'count'])
for idx, row in grouped.iterrows():
    print(f"{idx}: {row['mean']:.4f} ({int(row['count'])} matches)")

print('\n=== By Round ===')
grouped = res_df.groupby('round')['correct'].agg(['mean', 'count'])
for idx, row in grouped.iterrows():
    print(f"{idx}: {row['mean']:.4f} ({int(row['count'])} matches)")
    
print('\n=== By Ranking Bucket ===')
grouped = res_df.groupby('rank_bucket')['correct'].agg(['mean', 'count'])
for idx, row in grouped.iterrows():
    print(f"{idx}: {row['mean']:.4f} ({int(row['count'])} matches)")
