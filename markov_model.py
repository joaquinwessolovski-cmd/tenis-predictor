import numpy as np

def calc_game_prob(p_serve, p_bp_saved):
    """
    Probability of winning a service game using a point-by-point Markov chain,
    incorporating Break Point saved %.
    """
    memo = {}
    def game_recursion(s, r):
        if s == 4 and r < 3: return 1.0
        if r == 4 and s < 3: return 0.0
        if s == 3 and r == 3: # Deuce
            p_ad_in = p_serve
            p_ad_out = 1 - p_serve
            
            p_win_from_ad_in = p_serve
            p_win_from_ad_out = p_bp_saved
            p_lose_from_ad_out = 1 - p_bp_saved
            
            p_win = p_ad_in * p_win_from_ad_in
            p_lose = p_ad_out * p_lose_from_ad_out
            
            if p_win + p_lose == 0: return 0.5
            return p_win / (p_win + p_lose)
            
        state = (s, r)
        if state in memo: return memo[state]
        
        # Is it a break point? (returner has 3 or Ad, server has less)
        if r == 3 and s < 3:
            p = p_bp_saved
        else:
            p = p_serve
            
        prob = p * game_recursion(s + 1, r) + (1 - p) * game_recursion(s, r + 1)
        memo[state] = prob
        return prob
        
    return game_recursion(0, 0)

def calc_tiebreak_prob(p_a, p_b):
    """
    Calculate the probability that player A wins a tiebreak.
    p_a = Prob(A wins point on A's serve)
    p_b = Prob(B wins point on B's serve)
    Assuming A serves first.
    """
    memo = {}
    
    def tb_recursion(score_a, score_b, points_played):
        if score_a >= 7 and score_a - score_b >= 2:
            return 1.0
        if score_b >= 7 and score_b - score_a >= 2:
            return 0.0
            
        if score_a == 6 and score_b == 6:
            # From 6-6, A and B take turns serving pairs of points.
            # The prob A wins the next 2 points is p_a * (1 - p_b)
            # The prob B wins the next 2 points is (1 - p_a) * p_b
            # The prob they split is the rest.
            p_a_wins_2 = p_a * (1 - p_b)
            p_b_wins_2 = (1 - p_a) * p_b
            if p_a_wins_2 + p_b_wins_2 == 0:
                return 0.5
            return p_a_wins_2 / (p_a_wins_2 + p_b_wins_2)
            
        state = (score_a, score_b, points_played % 2)
        if state in memo:
            return memo[state]
            
        rem = points_played % 4
        if rem == 0 or rem == 3:
            p_win_point = p_a
        else:
            p_win_point = 1 - p_b
            
        prob = p_win_point * tb_recursion(score_a + 1, score_b, points_played + 1) + \
               (1 - p_win_point) * tb_recursion(score_a, score_b + 1, points_played + 1)
               
        memo[state] = prob
        return prob

    return tb_recursion(0, 0, 0)

def calc_set_prob(p_game_a, p_game_b, p_tb_a, a_serves_first=True):
    """
    Probability that A wins the set.
    """
    memo = {}
    
    def set_recursion(games_a, games_b, a_serving):
        if games_a == 6 and games_b <= 4:
            return 1.0
        if games_b == 6 and games_a <= 4:
            return 0.0
        if games_a == 7 and games_b == 5:
            return 1.0
        if games_b == 7 and games_a == 5:
            return 0.0
        if games_a == 6 and games_b == 6:
            if a_serving:
                return p_tb_a
            else:
                # We expect p_tb_b to be calculated with B serving first
                # For simplicity, if B serves first, it's very close to 1 - p_tb_b
                # Let's approximate or just use a symmetric call
                return 1 - calc_tiebreak_prob(p_b=p_game_b, p_a=p_game_a) 
                
        state = (games_a, games_b, a_serving)
        if state in memo:
            return memo[state]
            
        if a_serving:
            prob = p_game_a * set_recursion(games_a + 1, games_b, False) + \
                   (1 - p_game_a) * set_recursion(games_a, games_b + 1, False)
        else:
            prob = (1 - p_game_b) * set_recursion(games_a + 1, games_b, True) + \
                   p_game_b * set_recursion(games_a, games_b + 1, True)
                   
        memo[state] = prob
        return prob

    return set_recursion(0, 0, a_serves_first)

def predict_match(p_a, p_b, p_bp_saved_a, p_bp_saved_b, best_of=3):
    """
    Predict probability of player A winning the match.
    """
    p_game_a = calc_game_prob(p_a, p_bp_saved_a)
    p_game_b = calc_game_prob(p_b, p_bp_saved_b)
    
    p_tb_a_serves_first = calc_tiebreak_prob(p_a, p_b)
    p_tb_b_serves_first = 1 - calc_tiebreak_prob(p_b, p_a)
    
    p_set_a_serves = calc_set_prob(p_game_a, p_game_b, p_tb_a_serves_first, a_serves_first=True)
    p_set_b_serves = calc_set_prob(p_game_a, p_game_b, p_tb_b_serves_first, a_serves_first=False)
    
    p_set_a = (p_set_a_serves + p_set_b_serves) / 2.0
    
    p = p_set_a
    q = 1 - p
    
    if best_of == 3:
        return p**2 + 2 * (p**2) * q
    elif best_of == 5:
        return p**3 + 3 * (p**3) * q + 6 * (p**3) * (q**2)
    else:
        raise ValueError("best_of must be 3 or 5")
