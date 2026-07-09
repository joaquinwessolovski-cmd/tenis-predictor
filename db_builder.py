import sqlite3
import pandas as pd
import os
import glob
import numpy as np

DB_NAME = "tennis_database.db"
DATA_DIR = "data 2/tennis_atp"
MCP_DIR = "data_MCP"

def build_mcp_styles():
    print("Calculating player styles from MCP...")
    df_overview = pd.read_csv(os.path.join(MCP_DIR, "charting-m-stats-Overview.csv"), low_memory=False)
    df_overview = df_overview[df_overview['set'] == 'Total'].copy()
    df_overview['total_pts'] = df_overview['serve_pts'] + df_overview['return_pts']
    
    df_net = pd.read_csv(os.path.join(MCP_DIR, "charting-m-stats-NetPoints.csv"), low_memory=False)
    df_net = df_net[df_net['row'] == 'NetPoints'].copy()
    
    df_merged = pd.merge(df_overview, df_net, on=['match_id', 'player'], how='left')
    cols_to_sum = ['total_pts', 'winners', 'unforced', 'winners_fh', 'winners_bh', 'net_pts', 'total_shots']
    
    for col in cols_to_sum:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)
        
    grouped = df_merged.groupby('player')[cols_to_sum].sum().reset_index()
    grouped = grouped[grouped['total_pts'] >= 100].copy()
    
    grouped['aggressiveness'] = (grouped['winners'] / grouped['total_pts']).clip(0, 1)
    grouped['ue_rate'] = (grouped['unforced'] / grouped['total_pts']).clip(0, 1)
    grouped['fh_preference'] = (grouped['winners_fh'] / (grouped['winners_fh'] + grouped['winners_bh']).replace(0, 1)).clip(0, 1)
    grouped['net_tendency'] = (grouped['net_pts'] / grouped['total_shots'].replace(0, 1)).clip(0, 1)
    
    return grouped

def main():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Create Tables
    cursor.execute('''
        CREATE TABLE Players (
            id INTEGER PRIMARY KEY,
            full_name TEXT,
            hand TEXT,
            dob TEXT,
            ioc TEXT,
            aggressiveness REAL,
            ue_rate REAL,
            fh_preference REAL,
            net_tendency REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE Matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tourney_id TEXT,
            tourney_name TEXT,
            surface TEXT,
            tourney_level TEXT,
            tourney_date INTEGER,
            winner_id INTEGER,
            loser_id INTEGER,
            score TEXT,
            best_of INTEGER,
            round TEXT,
            w_ace REAL, w_df REAL, w_svpt REAL, w_1stIn REAL, w_1stWon REAL, w_2ndWon REAL, w_bpSaved REAL, w_bpFaced REAL,
            l_ace REAL, l_df REAL, l_svpt REAL, l_1stIn REAL, l_1stWon REAL, l_2ndWon REAL, l_bpSaved REAL, l_bpFaced REAL,
            FOREIGN KEY (winner_id) REFERENCES Players(id),
            FOREIGN KEY (loser_id) REFERENCES Players(id)
        )
    ''')
    
    # 2. Process MCP Styles
    mcp_df = build_mcp_styles()
    med_agg = mcp_df['aggressiveness'].median()
    med_ue = mcp_df['ue_rate'].median()
    med_fh = mcp_df['fh_preference'].median()
    med_net = mcp_df['net_tendency'].median()
    
    # Map MCP names to dict
    mcp_dict = {}
    for _, row in mcp_df.iterrows():
        name = row['player']
        mcp_dict[name.lower()] = {
            'agg': row['aggressiveness'], 'ue': row['ue_rate'],
            'fh': row['fh_preference'], 'net': row['net_tendency']
        }
        
    # 3. Insert Players
    print("Inserting players...")
    players_df = pd.read_csv(os.path.join(DATA_DIR, "atp_players.csv"), low_memory=False)
    players_to_insert = []
    
    for _, row in players_df.iterrows():
        pid = int(row['player_id'])
        # Handle NaN strings
        f_name = str(row['name_first']) if pd.notna(row['name_first']) else ""
        l_name = str(row['name_last']) if pd.notna(row['name_last']) else ""
        full_name = f"{f_name} {l_name}".strip()
        
        # Match MCP style by name
        style = mcp_dict.get(full_name.lower())
        if style:
            agg, ue, fh, net = style['agg'], style['ue'], style['fh'], style['net']
        else:
            agg, ue, fh, net = med_agg, med_ue, med_fh, med_net
            
        players_to_insert.append((
            pid, full_name, str(row['hand']), str(row['dob']), str(row['ioc']),
            agg, ue, fh, net
        ))
        
    cursor.executemany('''
        INSERT INTO Players (id, full_name, hand, dob, ioc, aggressiveness, ue_rate, fh_preference, net_tendency)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', players_to_insert)
    
    # 4. Insert Matches
    print("Inserting matches...")
    all_files = glob.glob(os.path.join(DATA_DIR, "atp_matches_[12]*.csv"))
    matches_inserted = 0
    
    for filename in all_files:
        df_matches = pd.read_csv(filename, low_memory=False)
        cols_to_extract = [
            'tourney_id', 'tourney_name', 'surface', 'tourney_level', 'tourney_date',
            'winner_id', 'loser_id', 'score', 'best_of', 'round',
            'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_bpSaved', 'w_bpFaced',
            'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_bpSaved', 'l_bpFaced'
        ]
        # Keep only columns that exist
        avail_cols = [c for c in cols_to_extract if c in df_matches.columns]
        df_subset = df_matches[avail_cols].copy()
        
        # Ensure all columns exist in subset, fill missing with NaN
        for c in cols_to_extract:
            if c not in df_subset.columns:
                df_subset[c] = np.nan
                
        # Drop rows with no winner or loser
        df_subset = df_subset.dropna(subset=['winner_id', 'loser_id'])
        
        # Convert to records
        records = df_subset.to_records(index=False).tolist()
        # Some floats might be nan, SQLite handles None better
        cleaned_records = []
        for rec in records:
            cleaned_rec = [None if pd.isna(x) else x for x in rec]
            # Ensure integer IDs
            cleaned_rec[5] = int(cleaned_rec[5]) # winner_id
            cleaned_rec[6] = int(cleaned_rec[6]) # loser_id
            cleaned_records.append(tuple(cleaned_rec))
            
        cursor.executemany(f'''
            INSERT INTO Matches ({", ".join(cols_to_extract)})
            VALUES ({", ".join(["?"] * len(cols_to_extract))})
        ''', cleaned_records)
        matches_inserted += len(cleaned_records)
        
    # Create indexes
    cursor.execute('CREATE INDEX idx_winner ON Matches(winner_id)')
    cursor.execute('CREATE INDEX idx_loser ON Matches(loser_id)')
    cursor.execute('CREATE INDEX idx_date ON Matches(tourney_date)')
    
    conn.commit()
    conn.close()
    print(f"Database built successfully! {matches_inserted} matches inserted.")

if __name__ == "__main__":
    main()
