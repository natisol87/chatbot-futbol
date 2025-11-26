import streamlit as st
import google.generativeai as genai
import requests
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from datetime import datetime, timedelta, timezone

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Chatbot Futbolero 🇦🇷",
    page_icon="⚽",
    layout="centered"
)

# --- CARGAR API KEYS ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    FOOTBALL_API_KEY = st.secrets["FOOTBALL_API_KEY"]
except:
    GOOGLE_API_KEY = "AIzaSyB1M0QvQGJ0A9G9TBcacnmjSXmCOT5IlH8"
    FOOTBALL_API_KEY = "0cd801485b8c48e0aed87e956d7f9a54"

genai.configure(api_key=GOOGLE_API_KEY)
BASE_URL = "https://api.football-data.org/v4/"

# --- MAPA DE LIGAS Y CLASES ---
MAPA_LIGAS = {
    'premier-league': 'PL',
    'la-liga': 'PD',
    'bundesliga': 'BL1',
    'ligue-1': 'FL1',
    'french-ligue-1': 'FL1',
    'serie-a': 'SA',
}
CLASS_NAMES = ['bundesliga', 'french-ligue-1', 'la-liga', 'premier-league', 'serie-a']

# --- CARGAR EL MODELO TFLITE (NUEVO) ---
@st.cache_resource
def load_tflite_model():
    # Cargamos el intérprete (es más liviano)
    interpreter = tf.lite.Interpreter(model_path="modelo_ligas.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    # Obtenemos detalles de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.sidebar.success("✅ Cerebro (TFLite) cargado.")
except Exception as e:
    st.sidebar.error(f"⚠️ Error cargando modelo: {e}")

# --- FUNCIONES DE FUTBOL ---
def get_team_data(team_name: str, league_code: str):
    url = BASE_URL + f"competitions/{league_code}/standings"
    headers = {"X-Auth-Token": FOOTBALL_API_KEY}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        standings = data["standings"][0]["table"]
        for team in standings:
            if team_name.lower() in team["team"]["name"].lower():
                return f"{team['team']['name']} está puesto #{team['position']} con {team['points']} puntos."
        return "No encontré ese equipo en esta liga, che."
    except: return "Hubo un error consultando la API de fútbol."

def get_matches(period='TODAY'):
    url = BASE_URL + "matches"
    headers = {"X-Auth-Token": FOOTBALL_API_KEY}
    today = datetime.now(timezone.utc).date()
    params = {"dateFrom": today.strftime("%Y-%m-%d"), "dateTo": today.strftime("%Y-%m-%d")}
    if period == 'TOMORROW':
        t = today + timedelta(days=1)
        params = {"dateFrom": t.strftime("%Y-%m-%d"), "dateTo": t.strftime("%Y-%m-%d")}
    try:
        res = requests.get(url, headers=headers, params=params).json()
        matches = [f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}" for m in res.get('matches', [])]
        return str(matches) if matches else "No hay partidos programados para esta fecha."
    except: return "Error buscando partidos."

def consultar_partidos_interactivo(codigo_liga, tipo):
    url = BASE_URL + f"competitions/{codigo_liga}/matches"
    headers = {"X-Auth-Token": FOOTBALL_API_KEY}
    today = datetime.now(timezone.utc).date()
    
    if tipo == 'future':
        date_from = today.strftime("%Y-%m-%d")
        date_to = (today + timedelta(days=10)).strftime("%Y-%m-%d")
        status = 'SCHEDULED'
    else: 
        date_from = (today - timedelta(days=10)).strftime("%Y-%m-%d")
        date_to = today.strftime("%Y-%m-%d")
        status = 'FINISHED'

    params = {"dateFrom": date_from, "dateTo": date_to, "status": status}

    try:
        res = requests.get(url, headers=headers, params=params).json()
        matches = res.get('matches', [])
        if not matches: return "⚠️ No encontré partidos cerca de esa fecha."
        
        resultados = []
        for m in matches[:5]: 
            fecha = m['utcDate'][:10]
            if status == 'FINISHED':
                score = m['score']['fullTime']
                resultados.append(f"⚽ {fecha}: {m['homeTeam']['name']} ({score['home']}) - ({score['away']}) {m['awayTeam']['name']}")
            else:
                resultados.append(f"🗓️ {fecha}: {m['homeTeam']['name']} vs {m['awayTeam']['name']}")
        return "\n\n".join(resultados)
    except Exception as e: return f"🚨 Se rompió algo: {e}"

# --- CONFIGURACIÓN GEMINI ---
system_prompt = """
Sos un asistente experto en fútbol, hablas con modismos argentinos (voseo).
Solo sabes de las 5 grandes ligas: Premier, La Liga, Bundesliga, Serie A y Ligue 1.
"""
tools = [get_team_data, get_matches]
model_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash-thinking-exp-01-21", tools=tools, system_instruction=system_prompt)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.gemini_chat = model_gemini.start_chat(enable_automatic_function_calling=True)

# --- INTERFAZ GRÁFICA ---
st.title("🤖 Chatbot Futbolero 🇦🇷")
st.markdown("¡Hola, maestro! Soy tu asistente de fútbol con **Visión Artificial**.")

tab1, tab2 = st.tabs(["💬 Chat General", "📸 Analizar Escudo"])

# --- TAB 1: CHAT ---
with tab1:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Preguntame algo de fútbol..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = st.session_state.gemini_chat.send_message(prompt)
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Se pinchó la conexión: {e}")

# --- TAB 2: VISIÓN ARTIFICIAL (LÓGICA TFLITE) ---
with tab2:
    st.header("Ojo de Halcón (CNN)")
    st.write("Subí la foto de un escudo y te digo de qué liga es.")
    
    uploaded_file = st.file_uploader("Elegí una imagen...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Escudo subido', width=150)
        
        if st.button("🔍 Analizar Escudo"):
            with st.spinner('La IA está mirando...'):
                # 1. Preprocesamiento
                size = (128, 128)
                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                img_array = np.array(image)
                img_array = img_array.astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # 2. Inferencia con TFLite (Cambió esta parte)
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])

                # 3. Resultados
                score = tf.nn.softmax(predictions[0])
                class_idx = np.argmax(score)
                liga_predicha = CLASS_NAMES[class_idx]
                confianza = 100 * np.max(score)

            st.success(f"👁️ Para mí, es de la **{liga_predicha.replace('-', ' ').title()}** ({confianza:.1f}% seguro).")

            if liga_predicha in MAPA_LIGAS:
                codigo = MAPA_LIGAS[liga_predicha]
                st.info(f"Tengo datos en vivo de la {liga_predicha}. ¿Qué querés ver?")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📅 Próximos Partidos"):
                        st.text(consultar_partidos_interactivo(codigo, 'future'))
                with col2:
                    if st.button("⚽ Resultados Pasados"):
                        st.text(consultar_partidos_interactivo(codigo, 'past'))
            else:
                st.warning("Identifiqué la liga, pero mi API no tiene datos en vivo de esta.")