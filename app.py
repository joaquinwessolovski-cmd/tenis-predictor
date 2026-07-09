import customtkinter as ctk
import pickle
import numpy as np
import pandas as pd

class TennisPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Tennis Predictor Pro")
        self.geometry("700x550")
        
        # Load model and data
        self.load_data()
        
        # Setup UI
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.title_label = ctk.CTkLabel(self, text="Tennis Match Predictor", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10))
        
        # Player 1
        self.p1_label = ctk.CTkLabel(self, text="Player 1", font=ctk.CTkFont(size=16))
        self.p1_label.grid(row=1, column=0, padx=20, pady=(10, 0))
        
        self.p1_var = ctk.StringVar(value=self.player_list[0])
        self.p1_dropdown = ctk.CTkOptionMenu(self, variable=self.p1_var, values=self.player_list, width=250)
        self.p1_dropdown.grid(row=2, column=0, padx=20, pady=10)
        
        # Player 2
        self.p2_label = ctk.CTkLabel(self, text="Player 2", font=ctk.CTkFont(size=16))
        self.p2_label.grid(row=1, column=1, padx=20, pady=(10, 0))
        
        self.p2_var = ctk.StringVar(value=self.player_list[1])
        self.p2_dropdown = ctk.CTkOptionMenu(self, variable=self.p2_var, values=self.player_list, width=250)
        self.p2_dropdown.grid(row=2, column=1, padx=20, pady=10)
        
        # Surface
        self.surf_label = ctk.CTkLabel(self, text="Surface", font=ctk.CTkFont(size=16))
        self.surf_label.grid(row=3, column=0, columnspan=2, padx=20, pady=(20, 0))
        
        self.surf_var = ctk.StringVar(value="Hard")
        self.surf_dropdown = ctk.CTkSegmentedButton(self, variable=self.surf_var, values=["Hard", "Clay", "Grass"])
        self.surf_dropdown.grid(row=4, column=0, columnspan=2, padx=20, pady=10)
        
        # Predict Button
        self.predict_btn = ctk.CTkButton(self, text="Predict Outcome", command=self.predict, font=ctk.CTkFont(size=16, weight="bold"), height=40)
        self.predict_btn.grid(row=5, column=0, columnspan=2, padx=20, pady=30)
        
        # Results frame
        self.result_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.result_frame.grid(row=6, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_columnconfigure(1, weight=1)
        
        self.result_p1_label = ctk.CTkLabel(self.result_frame, text="", font=ctk.CTkFont(size=18, weight="bold"))
        self.result_p1_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.result_p2_label = ctk.CTkLabel(self.result_frame, text="", font=ctk.CTkFont(size=18, weight="bold"))
        self.result_p2_label.grid(row=0, column=1, padx=10, pady=5)
        
        self.stats_p1_label = ctk.CTkLabel(self.result_frame, text="", justify="left")
        self.stats_p1_label.grid(row=1, column=0, padx=10, pady=5)
        
        self.stats_p2_label = ctk.CTkLabel(self.result_frame, text="", justify="left")
        self.stats_p2_label.grid(row=1, column=1, padx=10, pady=5)
        
    def load_data(self):
        try:
            with open('tennis_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.elo_sys = data['elo_sys']
                self.player_stats = data['player_stats']
                
            with open('player_names.pkl', 'rb') as f:
                self.player_names = pickle.load(f)
                
            import os
            if os.path.exists('player_style_profiles.pkl'):
                with open('player_style_profiles.pkl', 'rb') as f:
                    mcp_data = pickle.load(f)
                    self.style_profiles = mcp_data['profiles']
                    self.style_medians = mcp_data['medians']
            else:
                self.style_profiles = {}
                self.style_medians = {}
                
            # Create a reverse lookup dictionary
            self.name_to_id = {v: k for k, v in self.player_names.items()}
            
            # Get top players to populate dropdown (based on number of matches played in dataset)
            active_players = [(p_id, stats['matches']) for p_id, stats in self.player_stats.items() if stats['matches'] > 10]
            active_players.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 300 players by match count
            top_ids = [p[0] for p in active_players[:300]]
            self.player_list = sorted([self.player_names.get(p_id, f"Unknown {p_id}") for p_id in top_ids if p_id in self.player_names])
            if not self.player_list:
                self.player_list = ["No players found"]
                
        except Exception as e:
            print(f"Error loading data: {e}")
            self.player_list = ["Error loading data"]
            self.name_to_id = {}
            
    def get_profile(self, p_id):
        if p_id not in self.player_stats or self.player_stats[p_id]['matches'] == 0:
            return [0, 0, 0, 0, 0]
            
        st = self.player_stats[p_id]
        svpt = max(st['svpt'], 1)
        
        ace_rate = st['ace'] / svpt
        df_rate = st['df'] / svpt
        first_win_rate = st['1stWon'] / max(st['1stIn'], 1)
        second_win_rate = st['2ndWon'] / max(svpt - st['1stIn'], 1)
        bp_saved_rate = st.get('bpSaved', 0) / max(st.get('bpFaced', 1), 1)
        
        return [ace_rate, df_rate, first_win_rate, second_win_rate, bp_saved_rate]
        
    def get_form(self, p_id, surf):
        st = self.player_stats.get(p_id, {})
        player_form = st.get('form', {'all': [], 'surf': {}})
        all_form = np.mean(player_form['all']) if player_form['all'] else 0.5
        surf_form = np.mean(player_form.get('surf', {}).get(surf, [])) if player_form.get('surf', {}).get(surf) else 0.5
        return [all_form, surf_form]
        
    def get_style(self, pid):
        sp = self.style_profiles.get(pid, {})
        def_agg = self.style_medians.get('aggressiveness', 0.15)
        def_ue = self.style_medians.get('ue_rate', 0.18)
        def_fh = self.style_medians.get('fh_preference', 0.66)
        def_net = self.style_medians.get('net_tendency', 0.17)
        return [
            sp.get('aggressiveness', def_agg),
            sp.get('ue_rate', def_ue),
            sp.get('fh_preference', def_fh),
            sp.get('net_tendency', def_net)
        ]
            
    def predict(self):
        name1 = self.p1_var.get()
        name2 = self.p2_var.get()
        surf = self.surf_var.get()
        
        if name1 == name2:
            self.result_p1_label.configure(text="Please select different players.", text_color="red")
            self.result_p2_label.configure(text="")
            return
            
        id1 = self.name_to_id.get(name1)
        id2 = self.name_to_id.get(name2)
        
        if not id1 or not id2:
            return
            
        elo1 = self.elo_sys.get_elo(id1)
        elo2 = self.elo_sys.get_elo(id2)
        surf_elo1 = self.elo_sys.get_elo(id1, surf)
        surf_elo2 = self.elo_sys.get_elo(id2, surf)
        
        prof1 = self.get_profile(id1)
        prof2 = self.get_profile(id2)
        
        form1 = self.get_form(id1, surf)
        form2 = self.get_form(id2, surf)
        
        style1 = self.get_style(id1)
        style2 = self.get_style(id2)
        
        h2h_records = self.player_stats.get('GLOBAL_H2H_RECORDS', {})
        pair_key = tuple(sorted([id1, id2]))
        id1_wins = h2h_records.get(pair_key, {}).get(id1, 0)
        id2_wins = h2h_records.get(pair_key, {}).get(id2, 0)
        total_h2h = id1_wins + id2_wins
        if total_h2h > 0:
            id1_h2h_rate = id1_wins / total_h2h
            id2_h2h_rate = id2_wins / total_h2h
        else:
            id1_h2h_rate = 0.5
            id2_h2h_rate = 0.5
            
        features = [elo1, elo2, surf_elo1, surf_elo2, id1_h2h_rate, id2_h2h_rate] + prof1 + prof2 + form1 + form2 + style1 + style2
        
        cols = ['A_elo', 'B_elo', 'A_surf_elo', 'B_surf_elo', 'A_h2h', 'B_h2h',
            'A_ace', 'A_df', 'A_1w', 'A_2w', 'A_bp',
            'B_ace', 'B_df', 'B_1w', 'B_2w', 'B_bp',
            'A_form_all', 'A_form_surf', 'B_form_all', 'B_form_surf',
            'A_agg', 'A_ue', 'A_fh', 'A_net',
            'B_agg', 'B_ue', 'B_fh', 'B_net']
            
        # Predict A beats B
        prob_A = self.model.predict_proba(pd.DataFrame([features], columns=cols))[0][1]
        
        prob1_pct = prob_A * 100
        prob2_pct = (1 - prob_A) * 100
        
        color1 = "green" if prob1_pct > 50 else "white"
        color2 = "green" if prob2_pct > 50 else "white"
        
        self.result_p1_label.configure(text=f"{name1}: {prob1_pct:.1f}% Win", text_color=color1)
        self.result_p2_label.configure(text=f"{name2}: {prob2_pct:.1f}% Win", text_color=color2)
        
        # Display Stats
        stats1 = f"Overall Elo: {int(elo1)}\n{surf} Elo: {int(surf_elo1)}\nAce Rate: {prof1[0]*100:.1f}%\nDF Rate: {prof1[1]*100:.1f}%\n1st Serve Win: {prof1[2]*100:.1f}%"
        stats2 = f"Overall Elo: {int(elo2)}\n{surf} Elo: {int(surf_elo2)}\nAce Rate: {prof2[0]*100:.1f}%\nDF Rate: {prof2[1]*100:.1f}%\n1st Serve Win: {prof2[2]*100:.1f}%"
        
        self.stats_p1_label.configure(text=stats1)
        self.stats_p2_label.configure(text=stats2)

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = TennisPredictorApp()
    app.mainloop()
