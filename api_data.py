import requests
import json
from thefuzz import process

def fetch_current_tournaments():
    """
    Fetches currently active ATP tournaments and their matches using the ESPN API.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"
    tournaments = []
    
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        events = data.get('events', [])
        
        for event in events:
            tournament_name = event.get('name', 'Unknown Tournament')
            matches = []
            
            groupings = event.get('groupings', [])
            for group in groupings:
                if group.get('grouping', {}).get('slug') != 'mens-singles':
                    continue
                
                competitions = group.get('competitions', [])
                for comp in competitions:
                    competitors = comp.get('competitors', [])
                    if len(competitors) >= 2:
                        p1 = competitors[0].get('athlete', {}).get('displayName', '')
                        p2 = competitors[1].get('athlete', {}).get('displayName', '')
                        round_name = comp.get('round', {}).get('displayName', 'Unknown Round')
                        status = comp.get('status', {}).get('type', {}).get('description', 'Scheduled')
                        
                        if p1 and p2:
                            matches.append({
                                'player1': p1, 
                                'player2': p2, 
                                'round': round_name, 
                                'status': status,
                                'p1_winner': competitors[0].get('winner', False),
                                'p2_winner': competitors[1].get('winner', False)
                            })
            
            if matches:
                surface = 'Hard'
                name_lower = tournament_name.lower()
                
                clay_keywords = ['roland', 'french', 'monte', 'madrid', 'rome', 'hamburg', 'bastad', 'nordea', 'umag', 'gstaad', 'buenos aires', 'rio', 'santiago', 'estoril', 'munich', 'geneva', 'lyon']
                grass_keywords = ['wimbledon', 'halle', 'stuttgart', 'eastbourne', 'queens', 'hertogenbosch', 'mallorca', 'newport']
                
                if any(kw in name_lower for kw in grass_keywords):
                    surface = 'Grass'
                elif any(kw in name_lower for kw in clay_keywords):
                    surface = 'Clay'
                    
                tournaments.append({
                    'name': tournament_name,
                    'surface': surface,
                    'matches': matches
                })
                
    except Exception as e:
        print(f"Error fetching from ESPN API: {e}")
        
    return tournaments
_fuzzy_cache = {}

def fuzzy_match_player(api_name, db_player_list, threshold=80):
    """
    Matches an API player name to a database player name using fuzzy string matching with caching.
    """
    if not api_name or not db_player_list:
        return None
        
    cache_key = api_name
    if cache_key in _fuzzy_cache:
        return _fuzzy_cache[cache_key]
        
    # Quick exact match
    if api_name in db_player_list:
        return api_name
        
    # Attempt fuzzy match
    match = process.extractOne(api_name, db_player_list)
    if match and match[1] >= threshold:
        _fuzzy_cache[cache_key] = match[0]
        return match[0]
        
    # Heuristics for missing names (e.g. "C. Alcaraz" -> "Carlos Alcaraz")
    for db_name in db_player_list:
        if api_name.lower() in db_name.lower():
            _fuzzy_cache[cache_key] = db_name
            return db_name
            
    _fuzzy_cache[cache_key] = None
    return None

import os
import time
import pandas as pd

def fetch_tennis_abstract_elo():
    """
    Fetches the latest ATP Elo ratings from Tennis Abstract.
    Caches the result locally for 7 days to avoid unnecessary network requests.
    """
    cache_file = "ta_elo_cache.csv"
    # Check if cache exists and is less than 7 days old
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < 7 * 24 * 3600:
            try:
                return pd.read_csv(cache_file)
            except:
                pass # Fall through to fetch if cache reading fails
                
    url = "https://tennisabstract.com/reports/atp_elo_ratings.html"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0"}
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        tables = pd.read_html(r.text, match='Player')
        if tables:
            df = tables[0]
            df.to_csv(cache_file, index=False)
            return df
    except Exception as e:
        print(f"Error fetching from Tennis Abstract: {e}")
        # Try to return stale cache if network fails
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)
            
    return pd.DataFrame()
