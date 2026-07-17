# Tennis Predictor Pro 🎾

Tennis Predictor Pro es un sistema predictivo impulsado por Machine Learning (XGBoost) diseñado para predecir resultados de partidos de tenis profesional de la ATP. 

El modelo alcanza cerca de un **72% de precisión (Accuracy) en torneos de Grand Slam** (basado en backtesting del año 2026 estricto sin *Data Leakage*) y en torno a un 68% de precisión general para el resto del circuito. Utiliza un riguroso análisis histórico de más de 230,000 partidos, procesando perfiles de los jugadores (Match Charting Project) e inyectando un sistema "ELO" personalizado que aprende día a día.

## 🚀 Características Principales

- **Sistema ELO Adaptativo**: Calcula y mantiene dos puntuaciones de nivel para cada jugador: su nivel general y su nivel específico por superficie (Tierra Batida, Césped, Pista Dura, etc).
- **H2H de Superficie Inteligente**: El algoritmo diferencia el Historial Directo (Head-to-Head) general del historial directo en la superficie donde se va a disputar el partido.
- **Rachas de Momentum (Win Streaks)**: Captura el impacto psicológico midiendo las rachas de victorias consecutivas del jugador, lo que inyecta una precisión excepcional en torneos mayores.
- **Perfilamiento Táctico (MCP)**: Integra estadísticas profundas de los jugadores como agresividad (proporción de *winners*), errores no forzados y tendencia a subir a la red para determinar qué estilo de juego se sobrepondrá al otro.
- **Fair Odds (Cuotas de Apuesta Justas)**: El algoritmo transforma las probabilidades del XGBoost en cuotas teóricas (Odds), permitiendo detectar oportunidades de apuestas de valor (*Value Bets*) frente a las casas de apuestas reales.
- **Interfaz Web Interactiva**: App construida en Streamlit para seleccionar fácilmente torneos vigentes, buscar jugadores y visualizar el desglose del partido en vivo.

## 🛠️ Tecnologías

- **Machine Learning**: `xgboost`, `scikit-learn`
- **Base de Datos**: `sqlite3`, `pandas`
- **Interfaz de Usuario**: `streamlit`
- **Procesamiento de Datos**: `numpy`, `thefuzz`, `requests`, `beautifulsoup4`

## ⚙️ Instalación y Uso

1. **Clonar el repositorio y entrar en la carpeta**
   ```bash
   git clone https://github.com/joaquinwessolovski-cmd/tenis-predictor.git
   cd tenis-predictor
   ```

2. **Activar el entorno virtual**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   pip install thefuzz
   ```

4. **Ejecutar la Interfaz Web (Streamlit)**
   Ejecuta el siguiente comando apuntando directamente al ejecutable interno del entorno virtual para evitar conflictos de PATH:
   ```bash
   python -m streamlit run app.py
   # o si te encuentras utilizando streamlit_app.py:
   # python -m streamlit run streamlit_app.py
   ```

## 🧠 Estructura de Archivos

- `train_model.py`: Script central para re-entrenar el modelo, recalcular ELOs de cero, extraer features avanzados, hacer backtesting y guardar el `.pkl` definitivo.
- `db_builder.py`: Constructor de la base de datos `tennis_database.db` y formateador de los archivos CSV históricos y los datos tácticos de MCP.
- `streamlit_app.py`: El Front-End visual. Interfaz web para visualizar las estadísticas clave y cruzar a dos tenistas para una predicción instantánea.
- `tournament_engine.py`: Scraper y procesador que alimenta a la web con los torneos vigentes de la ATP y el ranking de forma automatizada.
- `tennis_model.pkl`: El "cerebro" empaquetado del algoritmo. Contiene el modelo XGBoost optimizado y el diccionario maestro del ELO para uso rápido en producción.

## 📈 Roadmap (Futuras Mejoras)

- Extender la integración de **Challengers y Qualys** para enriquecer las estadísticas de jugadores emergentes, calibrando con pesos según nivel del torneo.
- Ponderar el resultado del Accuracy frente a las **cuotas justas de mercado (ROI metrics)** para evaluar rentabilidad neta real.
- Incorporar factores externos determinantes, tales como el factor de desgaste por sets disputados o influencia de condiciones meteorológicas.

---
*Desarrollado para la predicción inteligente del circuito ATP.*
