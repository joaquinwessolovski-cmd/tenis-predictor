import pandas as pd
import glob
import json
import time
import requests
import os

print("Extracting unique ATP level tournaments...")
files = glob.glob('data 2/tennis_atp/atp_matches_*.csv') + glob.glob('data 2/20*.csv')
names = set()

for f in files:
    try:
        df = pd.read_csv(f, low_memory=False)
        # Filter for ATP tour levels only
        if 'tourney_level' in df.columns:
            df = df[df['tourney_level'].isin(['G', 'M', 'A', 'F'])]
        if 'tourney_name' in df.columns:
            names.update(df['tourney_name'].dropna().unique())
    except:
        pass

names = list(names)
print(f"Found {len(names)} unique ATP tournaments.")

altitudes = {}
if os.path.exists('tourney_altitudes.json'):
    with open('tourney_altitudes.json', 'r') as f:
        altitudes = json.load(f)

headers = {'User-Agent': 'TennisPredictor/1.0'}

for name in names:
    if name in altitudes:
        continue
    
    # Try geocoding
    try:
        # Some cleanup
        clean_name = name.replace("Masters", "").replace("Open", "").replace("ATP", "").strip()
        
        geo_url = f"https://nominatim.openstreetmap.org/search?q={clean_name}&format=json&limit=1"
        res = requests.get(geo_url, headers=headers).json()
        
        if res:
            lat = res[0]['lat']
            lon = res[0]['lon']
            
            ele_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
            ele_res = requests.get(ele_url).json()
            if 'elevation' in ele_res:
                alt = ele_res['elevation'][0]
                altitudes[name] = alt
                print(f"{name} -> {alt}m")
            else:
                altitudes[name] = 0
        else:
            altitudes[name] = 0
            
    except Exception as e:
        print(f"Error on {name}: {e}")
        altitudes[name] = 0
        
    time.sleep(1.2) # Rate limit for Nominatim
    
with open('tourney_altitudes.json', 'w') as f:
    json.dump(altitudes, f, indent=4)
print("Saved altitudes to tourney_altitudes.json")
