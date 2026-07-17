import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sqlite3
import api_data
import os
import pickle
from tournament_engine import TournamentEngine, scrape_tournaments, generate_mock_draw

st.set_page_config(page_title="Tennis Predictor Pro", layout="centered", page_icon="🎾")

# SQLite Connection
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_db_connection():
    db_path = os.path.join(BASE_DIR, 'tennis_database.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@st.cache_data(ttl=600)
def cached_fetch_tournaments():
    return api_data.fetch_current_tournaments()

class EloSystem:
    def __init__(self, k=32, surface_k=32, inactivity_decay=0.1):
        self.overall_elo = {}
        self.surface_elo = {'Hard': {}, 'Clay': {}, 'Grass': {}, 'Carpet': {}}
        self.last_played = {}
        self.k = k
        self.surface_k = surface_k
        self.default_elo = 1500
        self.inactivity_decay = inactivity_decay
        
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

@st.cache_resource
def load_ml_model():
    # Cache busted to load new 24-feature model
    try:
        model_path = os.path.join(BASE_DIR, 'tennis_model.pkl')
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            elo_sys = data['elo_sys']
            player_stats = data['player_stats']
        return model, elo_sys, player_stats
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None, None, None

model, elo_sys, player_stats = load_ml_model()

@st.cache_data
def get_player_list():
    conn = get_db_connection()
    # Solo cargar jugadores con al menos 1 partido oficial
    players = conn.execute("""
        SELECT p.id, p.full_name, COUNT(m.id) as match_count
        FROM Players p
        JOIN Matches m ON p.id = m.winner_id OR p.id = m.loser_id
        GROUP BY p.id
        HAVING match_count > 0
        ORDER BY match_count DESC
    """).fetchall()
    conn.close()
    
    player_list = [p['full_name'] for p in players]
    name_to_id = {p['full_name']: p['id'] for p in players}
    return player_list, name_to_id

player_list, name_to_id = get_player_list()

# Instantiate the engine globally so all tabs can use it
engine = TournamentEngine(model, elo_sys, player_stats, name_to_id, player_list)
def get_baseline_profile(p_id):
    if p_id not in player_stats or player_stats[p_id]['matches'] == 0:
        return [0, 0, 0, 0, 0]
    st_data = player_stats[p_id]
    svpt = max(st_data['svpt'], 1)
    ace_rate = st_data['ace'] / svpt
    df_rate = st_data['df'] / svpt
    first_win_rate = st_data['1stWon'] / max(st_data['1stIn'], 1)
    second_win_rate = st_data['2ndWon'] / max(svpt - st_data['1stIn'], 1)
    bp_saved_rate = st_data.get('bpSaved', 0) / max(st_data.get('bpFaced', 1), 1)
    return [ace_rate, df_rate, first_win_rate, second_win_rate, bp_saved_rate]

@st.cache_data(ttl=3600)
def get_h2h(p1_id, p2_id):
    conn = get_db_connection()
    wins1 = conn.execute("SELECT COUNT(*) FROM Matches WHERE winner_id = ? AND loser_id = ?", (p1_id, p2_id)).fetchone()[0]
    wins2 = conn.execute("SELECT COUNT(*) FROM Matches WHERE winner_id = ? AND loser_id = ?", (p2_id, p1_id)).fetchone()[0]
    return wins1, wins2

@st.cache_data(ttl=3600)
def get_surf_h2h(p1_id, p2_id, surface):
    conn = get_db_connection()
    wins1 = conn.execute("SELECT COUNT(*) FROM Matches WHERE winner_id = ? AND loser_id = ? AND surface = ?", (p1_id, p2_id, surface)).fetchone()[0]
    wins2 = conn.execute("SELECT COUNT(*) FROM Matches WHERE winner_id = ? AND loser_id = ? AND surface = ?", (p2_id, p1_id, surface)).fetchone()[0]
    return wins1, wins2

@st.cache_data(ttl=3600)
def get_player_info(p_id):
    conn = get_db_connection()
    row = conn.execute("""
        SELECT winner_age, winner_ht, winner_rank, loser_age, loser_ht, loser_rank, winner_id
        FROM Matches 
        WHERE winner_id = ? OR loser_id = ?
        ORDER BY tourney_date DESC LIMIT 1
    """, (p_id, p_id)).fetchone()
    
    if row:
        if row['winner_id'] == p_id:
            return float(row['winner_age'] or 25.0), float(row['winner_ht'] or 185.0), int(row['winner_rank'] or 100)
        else:
            return float(row['loser_age'] or 25.0), float(row['loser_ht'] or 185.0), int(row['loser_rank'] or 100)
    return 25.0, 185.0, 100

@st.cache_data(ttl=3600)
def get_surface_winrate(pid, surf):
    conn = get_db_connection()
    stats = conn.execute("""
        SELECT
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins,
            COUNT(*) as total
        FROM Matches
        WHERE (winner_id = ? OR loser_id = ?) AND surface = ?
    """, (pid, pid, pid, surf)).fetchone()
    conn.close()
    if stats['total'] == 0:
        return 0.0, 0
    return (stats['wins'] / stats['total']) * 100, stats['total']

def get_player_style(pid):
    conn = get_db_connection()
    style = conn.execute("SELECT aggressiveness, ue_rate, fh_preference, net_tendency FROM Players WHERE id = ?", (pid,)).fetchone()
    conn.close()
    if style:
        return [style['aggressiveness'], style['ue_rate'], style['fh_preference'], style['net_tendency']]
    return [0.15, 0.18, 0.66, 0.17] # Medians if not found

st.title("🎾 Tennis Predictor Pro")
st.write("Motor predictivo XGBoost + Elo impulsado por Base de Datos Relacional y perfiles Match Charting Project.")

def render_prediction(p1_name, p2_name, surf):
    if p1_name == p2_name:
        st.warning("Por favor, selecciona dos jugadores diferentes.")
        return
        
    id1 = name_to_id.get(p1_name)
    id2 = name_to_id.get(p2_name)
    
    if id1 and id2:
        # H2H and WinRates
        p1_wins, p2_wins = get_h2h(id1, id2)
        p1_surf_wins, p2_surf_wins = get_surf_h2h(id1, id2, surf)
        wr1, tot1 = get_surface_winrate(id1, surf)
        wr2, tot2 = get_surface_winrate(id2, surf)
        
        age1, ht1, rank1 = get_player_info(id1)
        age2, ht2, rank2 = get_player_info(id2)
        
        # Machine Learning Features
        elo1 = elo_sys.get_elo(id1)
        elo2 = elo_sys.get_elo(id2)
        surf_elo1 = elo_sys.get_elo(id1, surf)
        surf_elo2 = elo_sys.get_elo(id2, surf)
        
        delta_elo = elo1 - elo2
        delta_rank = rank1 - rank2
        indoor = 1 if surf == 'Carpet' else 0
        streak1, streak2 = 0, 0 # Approximated as 0 for fresh matches
        
        prof1 = get_baseline_profile(id1)
        prof2 = get_baseline_profile(id2)
        
        form1 = engine.get_form(id1, surf)
        form2 = engine.get_form(id2, surf)
        
        style1 = get_player_style(id1)
        style2 = get_player_style(id2)
        
        # Exact feature vector order expected by the model
        total_h2h = p1_wins + p2_wins
        id1_h2h_rate = p1_wins / total_h2h if total_h2h > 0 else 0.5
        id2_h2h_rate = p2_wins / total_h2h if total_h2h > 0 else 0.5
        
        total_surf_h2h = p1_surf_wins + p2_surf_wins
        id1_surf_h2h_rate = p1_surf_wins / total_surf_h2h if total_surf_h2h > 0 else 0.5
        id2_surf_h2h_rate = p2_surf_wins / total_surf_h2h if total_surf_h2h > 0 else 0.5
        
        features = [
            elo1, elo2, surf_elo1, surf_elo2, delta_elo,
            id1_h2h_rate, id2_h2h_rate, id1_surf_h2h_rate, id2_surf_h2h_rate,
            age1, age2, ht1, ht2, rank1, rank2, delta_rank,
            indoor, streak1, streak2
        ] + prof1 + prof2 + form1 + form2 + style1 + style2
        
        cols = ['A_elo', 'B_elo', 'A_surf_elo', 'B_surf_elo', 'delta_elo',
                'A_h2h', 'B_h2h', 'A_surf_h2h', 'B_surf_h2h',
                'A_age', 'B_age', 'A_ht', 'B_ht', 'A_rank', 'B_rank', 'delta_rank',
                'indoor', 'A_streak', 'B_streak',
                'A_ace', 'A_df', 'A_1w', 'A_2w', 'A_bp', 
                'B_ace', 'B_df', 'B_1w', 'B_2w', 'B_bp',
                'A_form_all', 'A_form_surf', 'B_form_all', 'B_form_surf',
                'A_agg', 'A_ue', 'A_fh', 'A_net',
                'B_agg', 'B_ue', 'B_fh', 'B_net']
                
        df_feat = pd.DataFrame([features], columns=cols)
        prob_A = model.predict_proba(df_feat)[0][1]
        
        prob1_pct = prob_A * 100
        prob2_pct = (1 - prob_A) * 100
        
        # UI H2H Banner
        st.markdown("---")
        st.markdown(f"<h3 style='text-align: center;'>Head-to-Head</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{p1_name} {p1_wins} - {p2_wins} {p2_name}</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        
        # Estimate sets based on prob for UI
        est_sets = "2-0" if max(prob_A, 1-prob_A) > 0.65 else "2-1"
        if surf == "Grass" and prob_A != 0.5: # Simple heuristic for Wimbledon best-of-5
            est_sets = "3-0" if max(prob_A, 1-prob_A) > 0.75 else ("3-1" if max(prob_A, 1-prob_A) > 0.60 else "3-2")
            
        odds1 = 1 / prob_A if prob_A > 0 else 0.0
        odds2 = 1 / (1 - prob_A) if (1 - prob_A) > 0 else 0.0
        
        with c1:
            st.subheader(f"{p1_name}")
            if prob1_pct > 50:
                st.success(f"Probabilidad: {prob1_pct:.1f}% (Est. Sets: {est_sets})")
            else:
                st.error(f"Probabilidad: {prob1_pct:.1f}%")
                
            st.info(f"💰 **Cuota de apuesta (Fair Odds):** {odds1:.2f}")
                
            st.write("📊 **Estadísticas Clave**")
            st.write(f"- **Overall ELO:** {int(elo1)}")
            st.write(f"- **{surf} ELO:** {int(surf_elo1)}")
            st.write(f"- **Win Rate ({surf}):** {wr1:.1f}% (en {tot1} partidos)")
            
            st.write(f"- **Forma (últimos 5):** {form1[0]*100:.1f}%")
            
            st.write("🎯 **Perfil de Juego (MCP)**")
            st.write(f"- **Agresividad (Winners):** {style1[0]*100:.1f}%")
            st.write(f"- **Errores No Forzados:** {style1[1]*100:.1f}%")
            st.write(f"- **Tendencia a la Red:** {style1[3]*100:.1f}%")
            st.write(f"- **Preferencia de Drive:** {style1[2]*100:.1f}%")
            
        with c2:
            st.subheader(f"{p2_name}")
            if prob2_pct > 50:
                st.success(f"Probabilidad: {prob2_pct:.1f}% (Est. Sets: {est_sets})")
            else:
                st.error(f"Probabilidad: {prob2_pct:.1f}%")
                
            st.info(f"💰 **Cuota de apuesta (Fair Odds):** {odds2:.2f}")
                
            st.write("📊 **Estadísticas Clave**")
            st.write(f"- **Overall ELO:** {int(elo2)}")
            st.write(f"- **{surf} ELO:** {int(surf_elo2)}")
            st.write(f"- **Win Rate ({surf}):** {wr2:.1f}% (en {tot2} partidos)")
            
            st.write(f"- **Forma (últimos 5):** {form2[0]*100:.1f}%")
            
            st.write("🎯 **Perfil de Juego (MCP)**")
            st.write(f"- **Agresividad (Winners):** {style2[0]*100:.1f}%")
            st.write(f"- **Errores No Forzados:** {style2[1]*100:.1f}%")
            st.write(f"- **Tendencia a la Red:** {style2[3]*100:.1f}%")
            st.write(f"- **Preferencia de Drive:** {style2[2]*100:.1f}%")
    else:
        st.error("No se encontraron estadísticas para uno de los jugadores en la base de datos.")

tab1, tab2, tab3, tab4 = st.tabs(["📅 Partidos de la Semana", "🔍 Predicción Personalizada", "🏆 Simulación de Torneos", "📈 Ranking ELO"])

with tab1:
    st.header("Torneos y Partidos ATP Actuales")
    
    with st.spinner("Cargando datos en vivo de ESPN..."):
        tournaments = cached_fetch_tournaments()
        
    if not tournaments:
        st.info("No hay torneos ATP principales en disputa en este momento.")
    else:
        tourney_names = [t['name'] for t in tournaments]
        selected_t_name = st.selectbox("Selecciona un Torneo:", tourney_names)
        
        selected_tournament = next((t for t in tournaments if t['name'] == selected_t_name), None)
        
        if selected_tournament:
            st.write(f"**Superficie Detectada:** {selected_tournament['surface']}")
            
            matches = selected_tournament['matches']
            if not matches:
                st.info("No hay partidos programados para este torneo.")
            else:
                rounds_available = list(dict.fromkeys(m.get('round', 'Fase Desconocida') for m in matches))
                rounds_available.insert(0, "Todas las fases")
                selected_round = st.selectbox("Filtrar por Fase:", rounds_available)
                
                filtered_matches = matches if selected_round == "Todas las fases" else [m for m in matches if m.get('round') == selected_round]
                
                if not filtered_matches:
                    st.info("No hay partidos en esta fase.")
                else:
                    match_options = [f"{m['player1']} vs {m['player2']} ({m.get('round', '')} - {m['status']})" for m in filtered_matches]
                    selected_match_str = st.selectbox("Selecciona un Partido:", match_options)
                    
                    selected_match_idx = match_options.index(selected_match_str)
                    selected_match = filtered_matches[selected_match_idx]
                    
                    st.markdown(f"### {selected_match['player1']} vs {selected_match['player2']}")
                
                internal_p1 = api_data.fuzzy_match_player(selected_match['player1'], player_list)
                internal_p2 = api_data.fuzzy_match_player(selected_match['player2'], player_list)
                
                if st.button("Predecir Partido Actual", type="primary", use_container_width=True):
                    if internal_p1 and internal_p2:
                        render_prediction(internal_p1, internal_p2, selected_tournament['surface'])
                    else:
                        st.warning("Ambos jugadores deben estar en la base de datos para realizar la predicción.")

with tab2:
    st.header("Predicción Personalizada")
    st.write("Escribe o busca el nombre del jugador. El autocompletado inteligente te ayudará a encontrarlos.")
    
    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("Player 1", options=player_list, index=player_list.index('Jannik Sinner') if 'Jannik Sinner' in player_list else 0)
    with col2:
        p2_name = st.selectbox("Player 2", options=player_list, index=player_list.index('Carlos Alcaraz') if 'Carlos Alcaraz' in player_list else 1)
    
    surf = st.radio("Superficie", ["Hard", "Clay", "Grass"], horizontal=True)
    
    if st.button("Analizar Matchup", type="primary", use_container_width=True, key="custom_predict"):
        render_prediction(p1_name, p2_name, surf)

with tab3:
    st.header("Simulación de Torneos")
    st.write("Simula torneos completos utilizando nuestro modelo predictivo.")
    
    sim_mode = st.radio("Modo de Simulación", ["Cuadro Aleatorio", "Torneo Real (Próximos)"], horizontal=True)
    
    if sim_mode == "Cuadro Aleatorio":
        size = st.selectbox("Número de Jugadores (Ronda 1)", [8, 16, 32])
        surf = st.radio("Superficie del Torneo", ["Hard", "Clay", "Grass"], horizontal=True, key="rand_surf")
        
        if st.button("Generar y Simular Torneo Aleatorio", type="primary"):
            with st.spinner("Simulando torneo..."):
                draw = generate_mock_draw(player_list, size)
                rounds, champion = engine.simulate_tournament(draw, surf)
                
                st.success(f"¡El campeón es **{champion}**!")
                
                for i, r_results in enumerate(rounds):
                    st.markdown(f"#### Ronda {i+1}")
                    for match in r_results:
                        p1, p2, winner, prob, est_sets = match
                        # Adjust sets if Grass (Wimbledon) just for visual effect
                        if surf == "Grass":
                            est_sets = "3-0" if prob > 0.75 else ("3-1" if prob > 0.60 else "3-2")
                        st.write(f"{p1} vs {p2} ➔ **Ganador: {winner}** ({est_sets}) ({prob*100:.1f}%)")
    else:
        st.info("Obteniendo cuadros reales en vivo de ESPN API...")
        real_tournaments = cached_fetch_tournaments()
        if real_tournaments:
            t_names = [t["name"] for t in real_tournaments]
            selected_t = st.selectbox("Selecciona el Torneo Actual", t_names)
            t_data = next((t for t in real_tournaments if t["name"] == selected_t), None)
            
            if t_data:
                active_players = set()
                eliminated = set()
                for m in t_data['matches']:
                    p1 = api_data.fuzzy_match_player(m['player1'], player_list) or m['player1']
                    p2 = api_data.fuzzy_match_player(m['player2'], player_list) or m['player2']
                    active_players.add(p1)
                    active_players.add(p2)
                    
                    if m.get('p1_winner'):
                        eliminated.add(p2)
                    elif m.get('p2_winner'):
                        eliminated.add(p1)
                        
                final_draw = [p for p in list(active_players - eliminated) if p in player_list]
                
                st.write(f"Jugadores activos detectados ({len(final_draw)}): {', '.join(final_draw[:5])}...")
                
                if st.button(f"Ejecutar Simulación Monte Carlo (10,000 runs) para {t_data['name']}", type="primary"):
                    if len(final_draw) < 2:
                        st.error("Debes seleccionar al menos 2 jugadores para simular el torneo.")
                    else:
                        with st.spinner(f"Simulando {t_data['name']} 10,000 veces con {len(final_draw)} jugadores..."):
                            mc_results = engine.simulate_monte_carlo(final_draw, t_data['surface'], runs=10000)
                            
                            st.markdown(f"### Probabilidad de Campeonato - {t_data['name']}")
                            # Format as a nice dataframe for UI
                            df_res = pd.DataFrame(mc_results, columns=['Jugador', 'Probabilidad de Campeonato'])
                            df_res['Probabilidad de Campeonato'] = (df_res['Probabilidad de Campeonato'] * 100).map("{:.1f}%".format)
                            
                            st.dataframe(df_res, use_container_width=True)

with tab4:
    st.header("Ranking ELO Actual")
    st.write("Calculado en base a todos los jugadores que han disputado al menos un partido en los últimos 365 días.")
    
    
    if st.button("Forzar Actualización desde Tennis Abstract"):
        with st.spinner("Descargando ELO más reciente..."):
            ta_df = api_data.fetch_tennis_abstract_elo()
            if not ta_df.empty:
                for idx, row in ta_df.iterrows():
                    ta_name = row['Player']
                    if pd.isna(ta_name) or not isinstance(ta_name, str):
                        continue
                    ta_name = ta_name.replace('\xa0', ' ')
                    p_id = name_to_id.get(ta_name)
                    if p_id:
                        elo_sys.overall_elo[p_id] = float(row['Elo'])
                        if not pd.isna(row.get('hElo')): elo_sys.surface_elo['Hard'][p_id] = float(row['hElo'])
                        if not pd.isna(row.get('cElo')): elo_sys.surface_elo['Clay'][p_id] = float(row['cElo'])
                        if not pd.isna(row.get('gElo')): elo_sys.surface_elo['Grass'][p_id] = float(row['gElo'])
                st.success("ELO actualizado correctamente con los datos en vivo.")
            else:
                st.error("No se pudo obtener la actualización.")
                
    if not elo_sys.last_played:
        st.info("No hay datos recientes disponibles.")
    else:
        import pandas as pd
        max_date = max(pd.to_datetime(d) for d in elo_sys.last_played.values())
        cutoff = max_date - pd.Timedelta(days=365)
        
        active_ids = []
        for pid, d in elo_sys.last_played.items():
            form_all = player_stats.get(pid, {}).get('form', {}).get('all', [])
            if len(form_all) == 10 and pd.to_datetime(d) >= cutoff:
                active_ids.append(pid)
        
        if active_ids:
            records = []
            for pid in active_ids:
                name = next((n for n, i in name_to_id.items() if i == pid), "Desconocido")
                records.append({
                    "Jugador": name,
                    "General": int(elo_sys.get_elo(pid)),
                    "Hard": int(elo_sys.get_elo(pid, "Hard")),
                    "Clay": int(elo_sys.get_elo(pid, "Clay")),
                    "Grass": int(elo_sys.get_elo(pid, "Grass"))
                })
            
            df_elo = pd.DataFrame(records).sort_values("General", ascending=False).reset_index(drop=True)
            df_elo.index += 1
            st.dataframe(df_elo, use_container_width=True)
        else:
            st.info("No hay datos recientes disponibles.")
