import pandas as pd
import glob
import os
files = glob.glob('data 2/tennis_atp/atp_matches_*.csv') + glob.glob('data 2/20*.csv')
names = set()
for f in files:
    try:
        df = pd.read_csv(f, low_memory=False)
        if 'tourney_name' in df.columns:
            names.update(df['tourney_name'].dropna().unique())
    except:
        pass
print(list(names)[:10])
print(len(names))
