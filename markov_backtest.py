import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from train_model import load_data
from markov_model import predict_match

def run_backtest():
    print("Loading data...")
    df = load_data()
    
    # Sort chronologically
    df = df.sort_values(by=['tourney_date', 'match_num'])
    
    # We will build EMA stats on the fly
    player_stats = {}
    delta_time = 0.005
    
    test_results = []
    
    # April 1st 2026 to July 31 2026
    start_date = pd.to_datetime('2026-04-01')
    end_date = pd.to_datetime('2026-07-31')
    
    print("Simulating through matches and calculating EMA...")
    for idx, row in df.iterrows():
        current_date = row['tourney_date']
        w_id = row['winner_id']
        l_id = row['loser_id']
        tourney_level = row.get('tourney_level', 'A')
        
        # Initialize stats if not present
        for p_id in [w_id, l_id]:
            if p_id not in player_stats:
                player_stats[p_id] = {
                    'svpt': 0.0,
                    'pts_won': 0.0,
                    'ret_pt': 0.0,
                    'ret_won': 0.0,
                    'last_date': current_date,
                    'matches': 0
                }
        
        # Determine if this match is in our backtest period
        if start_date <= current_date <= end_date and tourney_level != 'C':
            # Only predict if we have at least some matches for both players
            if player_stats[w_id]['matches'] > 2 and player_stats[l_id]['matches'] > 2:
                # Calculate p_w (winner win point on serve)
                w_serve_win = player_stats[w_id]['pts_won'] / max(player_stats[w_id]['svpt'], 1)
                # Opponent (loser) return win
                l_ret_win = player_stats[l_id]['ret_won'] / max(player_stats[l_id]['ret_pt'], 1)
                
                # Calculate p_l (loser win point on serve)
                l_serve_win = player_stats[l_id]['pts_won'] / max(player_stats[l_id]['svpt'], 1)
                # Opponent (winner) return win
                w_ret_win = player_stats[w_id]['ret_won'] / max(player_stats[w_id]['ret_pt'], 1)
                
                # Combine serve and return stats (average them)
                # If player A serves, prob A wins = (A serve win + (1 - B ret win)) / 2
                p_w = (w_serve_win + (1 - l_ret_win)) / 2.0
                p_l = (l_serve_win + (1 - w_ret_win)) / 2.0
                
                # Ensure probabilities are reasonable
                p_w = min(max(p_w, 0.4), 0.8)
                p_l = min(max(p_l, 0.4), 0.8)
                
                best_of = 5 if tourney_level == 'G' else 3
                
                # For our prediction, we'll assign player A = winner, player B = loser to see if it predicts > 0.5
                # But to avoid bias in metrics, we should randomize who is A and who is B
                is_w_first = np.random.rand() > 0.5
                if is_w_first:
                    p_a, p_b = p_w, p_l
                    target = 1
                else:
                    p_a, p_b = p_l, p_w
                    target = 0
                    
                prob_a = predict_match(p_a, p_b, best_of=best_of)
                test_results.append({
                    'target': target,
                    'prob': prob_a
                })

        # Update EMA stats for both players after the match (so no data leakage)
        for p_id, prefix in [(w_id, 'w'), (l_id, 'l')]:
            dt = (current_date - player_stats[p_id]['last_date']).days if player_stats[p_id]['last_date'] else 0
            decay = np.exp(-delta_time * dt)
            
            # Serve stats
            svpt = row.get(f'{prefix}_svpt')
            pts_won = 0
            if not pd.isna(row.get(f'{prefix}_1stWon')) and not pd.isna(row.get(f'{prefix}_2ndWon')):
                pts_won = row.get(f'{prefix}_1stWon') + row.get(f'{prefix}_2ndWon')
                
            if not pd.isna(svpt) and svpt > 0:
                player_stats[p_id]['svpt'] = player_stats[p_id]['svpt'] * decay + float(svpt)
                player_stats[p_id]['pts_won'] = player_stats[p_id]['pts_won'] * decay + float(pts_won)
                
            # Return stats
            opp_prefix = 'l' if prefix == 'w' else 'w'
            opp_svpt = row.get(f'{opp_prefix}_svpt')
            opp_pts_won = 0
            if not pd.isna(row.get(f'{opp_prefix}_1stWon')) and not pd.isna(row.get(f'{opp_prefix}_2ndWon')):
                opp_pts_won = row.get(f'{opp_prefix}_1stWon') + row.get(f'{opp_prefix}_2ndWon')
                
            if not pd.isna(opp_svpt) and opp_svpt > 0:
                ret_pt = float(opp_svpt)
                ret_won = ret_pt - float(opp_pts_won)
                player_stats[p_id]['ret_pt'] = player_stats[p_id]['ret_pt'] * decay + ret_pt
                player_stats[p_id]['ret_won'] = player_stats[p_id]['ret_won'] * decay + ret_won
                
            player_stats[p_id]['matches'] = player_stats[p_id]['matches'] * decay + 1
            player_stats[p_id]['last_date'] = current_date

    print(f"\nBacktest completed on {len(test_results)} matches.")
    
    if len(test_results) > 0:
        y_true = [res['target'] for res in test_results]
        y_prob = [res['prob'] for res in test_results]
        y_pred = [1 if p > 0.5 else 0 for p in y_prob]
        
        acc = accuracy_score(y_true, y_pred)
        ll = log_loss(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Log-Loss: {ll:.4f}")
        print(f"Brier Score: {brier:.4f}")
    else:
        print("No matches found in the specified period.")

if __name__ == "__main__":
    run_backtest()
