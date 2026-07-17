import pandas as pd
import requests
import io
import os

print("Downloading odds from tennis-data.co.uk...")

years = list(range(2001, 2025))
all_odds = []

for y in years:
    print(f"Downloading {y}...")
    try:
        url = f"http://www.tennis-data.co.uk/{y}/{y}.xlsx"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200:
            url = f"http://www.tennis-data.co.uk/{y}/{y}.xls"
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code != 200:
                url = f"http://www.tennis-data.co.uk/{y}/{y}.csv"
                r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                if r.status_code == 200:
                    df = pd.read_csv(io.StringIO(r.text))
                else:
                    print(f"Failed {y}")
                    continue
            else:
                df = pd.read_excel(io.BytesIO(r.content))
        else:
            df = pd.read_excel(io.BytesIO(r.content))
            
        # We need Date, Winner, Loser, B365W, B365L
        if 'B365W' in df.columns and 'B365L' in df.columns:
            subset = df[['Date', 'Winner', 'Loser', 'B365W', 'B365L']].copy()
            subset['Year'] = y
            all_odds.append(subset)
    except Exception as e:
        print(f"Exception on {y}: {e}")

if all_odds:
    df_odds = pd.concat(all_odds, ignore_index=True)
    df_odds = df_odds.dropna(subset=['B365W', 'B365L'])
    print(f"Downloaded {len(df_odds)} matches with Bet365 odds.")
    df_odds.to_csv("data_odds_b365.csv", index=False)
    print("Saved to data_odds_b365.csv")
else:
    print("No odds downloaded.")
