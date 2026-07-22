import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_model import load_data, build_dataset

def main():
    print("Loading data...")
    df = load_data()
    if df.empty:
        print("No matches found.")
        return
        
    print("Building base dataset with advanced features...")
    df_feat, elo_sys, player_stats = build_dataset(df)
    
    with open('tennis_elo_system.pkl', 'wb') as f:
        pickle.dump({'elo_sys': elo_sys, 'player_stats': player_stats}, f)
        
    models_dir = 'ensembles'
    os.makedirs(models_dir, exist_ok=True)
    
    drop_cols = ['target', 'tourney_date', 'b365_A', 'b365_B', 'A_name', 'B_name', 'surface']
    
    results = []
    segments = ['Global', 'Hard', 'Clay', 'Grass']
    
    for seg in segments:
        segment_name = f"{seg}_Ensemble"
        print(f"\n--- Training Ensemble for {segment_name} ---")
        
        if seg == 'Global':
            df_seg = df_feat.copy()
        else:
            df_seg = df_feat[df_feat['surface'] == seg].copy()
            
        if len(df_seg) < 200:
            print(f"Skipping {segment_name} due to low sample size ({len(df_seg)} matches).")
            continue
            
        X = df_seg.drop(columns=[c for c in drop_cols if c in df_seg.columns])
        y = df_seg['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        estimators = [
            ('xgb', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')),
            ('lgb', LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, n_jobs=-1, verbose=-1)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1))
        ]
        
        from sklearn.linear_model import LogisticRegression
        clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3, n_jobs=1)
        
        print(f"Fitting {segment_name} ({len(X_train)} train samples)...")
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)
        ll = log_loss(y_test, y_proba)
        
        print(f"{segment_name} -> Acc: {acc:.4f}, Brier: {brier:.4f}, LogLoss: {ll:.4f}")
        results.append({'segment': segment_name, 'accuracy': acc, 'brier': brier, 'logloss': ll, 'n_samples': len(df_seg)})
        
        with open(f'{models_dir}/{segment_name}.pkl', 'wb') as f:
            pickle.dump(clf, f)
            
    print("\n--- Final Ensemble Results ---")
    for r in results:
        print(f"{r['segment']}: Acc {r['accuracy']:.4f} | Samples: {r['n_samples']}")

if __name__ == '__main__':
    main()
