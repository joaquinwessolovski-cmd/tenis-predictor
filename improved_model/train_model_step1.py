import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss
from thefuzz import process

DATA_DIR = "../data 2/tennis_atp"

def load_data(start_year=1968, end_year=2024):
    all_files = glob.glob(os.path.join(DATA_DIR, "atp_matches_[12]*.csv"))
    li = []
    for filename in all_files:
        try:
            year = int(filename.split('_')[-1].split('.')[0])
            if start_year <= year <= end_year:
                df = pd.read_csv(filename, parse_dates=['tourney_date'], low_memory=False)
                li.append(df)
        except ValueError:
            pass
            
    if not li:
        return pd.DataFrame()
        
    frame = pd.concat(li, axis=0, ignore_index=True)
    frame = frame.sort_values(by=['tourney_date', 'match_num'])
    return frame

class EloSystem:
    def __init__(self, k=32, surface_k=32, inactivity_decay=0.1):
        self.overall_elo = {}
        self.surface_elo = {'Hard': {}, 'Clay': {}, 'Grass': {}, 'Carpet': {}}
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

    def update(self, winner_id, loser_id, surface, current_date, tourney_level='A'):
        win_elo = self.get_elo(winner_id, current_date=current_date)
        los_elo = self.get_elo(loser_id, current_date=current_date)
        
        current_k = self.k
        current_surf_k = self.surface_k
        
        expected_win = self.expected_score(win_elo, los_elo)
        
        self.overall_elo[winner_id] = win_elo + current_k * (1 - expected_win)
        self.overall_elo[loser_id] = los_elo + current_k * (0 - (1 - expected_win))
        
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

def build_dataset(df):
    elo_sys = EloSystem(k=32, surface_k=32, inactivity_decay=0.5)
    dataset = []
    stats_cols = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'bpSaved', 'bpFaced']
    player_stats = {}
    player_form = {} # Tracks recent results (1 for win, 0 for loss)
    delta_time = 0.005 # Decay factor for stats
    
    def update_stats(p_id, prefix, row, current_date, is_win, surf):
        if p_id not in player_stats:
            player_stats[p_id] = {col: 0.0 for col in stats_cols}
            player_stats[p_id]['matches'] = 0
            player_stats[p_id]['last_date'] = current_date
            player_stats[p_id]['rank_points'] = 0 # Default points for unranked
            player_form[p_id] = {'all': [], 'surf': {'Hard': [], 'Clay': [], 'Grass': [], 'Carpet': []}}
            
        # Apply EMA decay to stats
        dt = (current_date - player_stats[p_id]['last_date']).days if player_stats[p_id]['last_date'] else 0
        decay = np.exp(-delta_time * dt)
        
        # Remove rank points logic
        
        for col in stats_cols:
            val = row.get(f'{prefix}_{col}')
            if not pd.isna(val):
                player_stats[p_id][col] = player_stats[p_id][col] * decay + val
                
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
            return [0, 0, 0, 0]
            
        st = player_stats[p_id]
        svpt = max(st['svpt'], 1)
        
        ace_rate = st['ace'] / svpt
        df_rate = st['df'] / svpt
        first_win_rate = st['1stWon'] / max(st['1stIn'], 1)
        second_win_rate = st['2ndWon'] / max(svpt - st['1stIn'], 1)
        
        return [ace_rate, df_rate, first_win_rate, second_win_rate]
        
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
        
        w_prof = get_profile(w_id)
        l_prof = get_profile(l_id)
        
        w_form = get_form(w_id, surf)
        l_form = get_form(l_id, surf)
        
        w_style = get_style(w_id)
        l_style = get_style(l_id)
        
        if player_stats.get(w_id, {}).get('matches', 0) > 2 and player_stats.get(l_id, {}).get('matches', 0) > 2:
            if np.random.rand() > 0.5:
                features = [w_elo, l_elo, w_surf_elo, l_surf_elo] + w_prof + l_prof + w_form + l_form + w_style + l_style + [1]
            else:
                features = [l_elo, w_elo, l_surf_elo, w_surf_elo] + l_prof + w_prof + l_form + w_form + l_style + w_style + [0]
                
            dataset.append(features)
            
        elo_sys.update(w_id, l_id, surf, current_date, tourney_level)
        update_stats(w_id, 'w', row, current_date, True, surf)
        update_stats(l_id, 'l', row, current_date, False, surf)
        
    # We must save player_form as part of player_stats so UI can use it
    for pid in player_stats:
        player_stats[pid]['form'] = player_form.get(pid, {'all': [], 'surf': {}})
        
    cols = ['A_elo', 'B_elo', 'A_surf_elo', 'B_surf_elo',
            'A_ace', 'A_df', 'A_1w', 'A_2w',
            'B_ace', 'B_df', 'B_1w', 'B_2w',
            'A_form_all', 'A_form_surf', 'B_form_all', 'B_form_surf',
            'A_agg', 'A_ue', 'A_fh', 'A_net',
            'B_agg', 'B_ue', 'B_fh', 'B_net',
            'Target']
            
    df_feat = pd.DataFrame(dataset, columns=cols)
    return df_feat, elo_sys, player_stats

def main():
    print("Loading data...")
    df = load_data() # Loads Kaggle + Scraped data
    
    if df.empty:
        print("No matches found. Ensure the dataset exists.")
        return
        
    print(f"Loaded {len(df)} matches. Building dataset...")
    df_feat, elo_sys, player_stats = build_dataset(df)
    
    print(f"Generated {len(df_feat)} training samples.")
    
    X = df_feat.drop('Target', axis=1)
    y = df_feat['Target']
    
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
    bs = brier_score_loss(y_test, probs)
    print(f"Accuracy: {acc:.4f}, Brier Score: {bs:.4f}")
    
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
        
    # Save the model and required states
    with open('tennis_model_step1.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'elo_sys': elo_sys,
            'player_stats': player_stats
        }, f)
        
    with open('player_names_step1.pkl', 'wb') as f:
        pickle.dump(player_names, f)
        
    print("Model and metadata saved.")

if __name__ == "__main__":
    main()
