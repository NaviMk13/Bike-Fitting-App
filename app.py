import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import tensorflow_hub as tfhub
import cv2
import tempfile
import os

# --- 1. DESIGN & EXTREM HOCHLESBARE UI ---
st.set_page_config(page_title="VELO-MATCH KI Pro", layout="wide", page_icon="🚴")

st.markdown("""
    <style>
    /* Dunkler, atmosphärischer Hintergrund */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.95)), 
                    url('https://images.unsplash.com/photo-1485965120184-e220f721d03e?q=80&w=1920') no-repeat center center fixed;
        background-size: cover;
        color: #ffffff !important; /* Erzwingt globales Weiß */
    }
    
    /* Titel-Styling mit extremem Kontrast */
    h1 {
        font-family: 'Impact', 'Arial Black', sans-serif;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #facc15 !important; /* Neon-Gelb */
        text-shadow: 3px 3px 6px #000000 !important;
    }
    h2, h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px #000000 !important;
    }
    
    /* Beschreibungstexte lesbar machen */
    .stMarkdown p {
        color: #f1f5f9 !important;
        font-size: 16px;
    }
    
    /* Eigene Ergebniskarten (Vollständig unabhängig von Streamlit-Farben) */
    .custom-card {
        background-color: #1e293b !important;
        border: 3px solid #facc15 !important;
        border-radius: 14px;
        padding: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.7);
        margin-bottom: 20px;
        text-align: center;
    }
    
    .custom-label {
        font-size: 14px !important;
        font-weight: 800 !important;
        color: #94a3b8 !important; /* Hellgrau */
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .custom-value {
        font-size: 36px !important;
        font-weight: 900 !important;
        color: #facc15 !important; /* Neon-Gelb */
        margin-bottom: 8px;
        text-shadow: 1px 1px 2px #000000;
    }
    
    .custom-target {
        font-size: 14px !important;
        font-weight: bold !important;
        color: #38bdf8 !important; /* Hellblau */
    }
    
    /* Info-Boxen für Empfehlungen */
    .rec-box {
        background-color: #0f172a;
        border-left: 6px solid #facc15;
        padding: 15px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 25px;
    }
    
    /* Fahrrad-Animation beim Laden */
    @keyframes ride {
        0% { transform: translateX(-30px); }
        50% { transform: translateX(30px); }
        100% { transform: translateX(-30px); }
    }
    .bike-loader {
        font-size: 60px;
        animation: ride 1.5s infinite ease-in-out;
        text-align: center;
        margin: 25px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚴 VELO-MATCH: LIVE AI BIKE FITTING")
st.write("Präzises Echtzeit-Tracking mit Google MoveNet KI für optimierte Biomechanik.")

# --- 2. KI-MODELL INITIALISIERUNG ---
@st.cache_resource
def load_movenet_model():
    model = tfhub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    movenet = model.signatures['serving_default']
    return movenet

try:
    movenet_model = load_movenet_model()
    model_loaded = True
except Exception as e:
    st.error(f"Fehler beim Laden der MoveNet-KI: {e}")
    model_loaded = False

# --- 3. WINKELBERECHNUNG ---
def calculate_angle(a, b, c, interior=True):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    if not interior:
        return angle
    else:
        return 180.0 - angle if angle > 90 else angle

# --- 4. LIVE-STREAMING VIDEOVERARBEITUNG ---
if model_loaded:
    uploaded_file = st.file_uploader("📂 Lade dein Bike-Fitting Video hoch (.mp4, .mov)", type=["mp4", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        video_placeholder = st.empty()
        status_text = st.empty()
        loader_anim = st.empty()
        
        status_text.info("⚙️ Starte Video-Stream und KI-Inferenz...")
        loader_anim.markdown("<div class='bike-loader'>🚴💨</div>", unsafe_allow_html=True)
        
        cap = cv2.VideoCapture(tfile.name)
        
        max_knee_angle = 0.0
        best_metrics = {'knee': 142.0, 'hip': 45.0, 'arm': 20.0, 'shoulder': 85.0, 'side': 'Unbekannt'}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_img, w_img, _ = frame_rgb.shape
            
            input_image = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 256, 256)
            input_image = tf.cast(input_image, dtype=tf.int32)
            
            outputs = movenet_model(input_image)
            keypoints = outputs['output_0'].numpy()[0, 0, :, :]
            
            # Validierung (Rechts vs Links)
            if keypoints[14][2] > 0.3 and keypoints[12][2] > 0.3:
                side = "Rechte Seite"
                hip = [keypoints[12][1] * h_img, keypoints[12][0] * w_img]
                knee = [keypoints[14][1] * h_img, keypoints[14][0] * w_img]
                ankle = [keypoints[16][1] * h_img, keypoints[16][0] * w_img]
                shoulder = [keypoints[6][1] * h_img, keypoints[6][0] * w_img]
                elbow = [keypoints[8][1] * h_img, keypoints[8][0] * w_img]
                wrist = [keypoints[10][1] * h_img, keypoints[10][0] * w_img]
            else:
                side = "Linke Seite"
                hip = [keypoints[11][1] * h_img, keypoints[11][0] * w_img]
                knee = [keypoints[13][1] * h_img, keypoints[13][0] * w_img]
                ankle = [keypoints[15][1] * h_img, keypoints[15][0] * w_img]
                shoulder = [keypoints[5][1] * h_img, keypoints[5][0] * w_img]
                elbow = [keypoints[7][1] * h_img, keypoints[7][0] * w_img]
                wrist = [keypoints[9][1] * h_img, keypoints[9][0] * w_img]
            
            current_knee = calculate_angle(hip, knee, ankle, interior=False)
            current_hip = calculate_angle(shoulder, hip, knee, interior=False)
            current_arm = calculate_angle(shoulder, elbow, wrist, interior=True)
            current_shoulder = calculate_angle(hip, shoulder, elbow, interior=False)
            
            if current_knee > max_knee_angle and current_knee < 165.0:
                max_knee_angle = current_knee
                best_metrics = {
                    'knee': current_knee,
                    'hip': current_hip,
                    'arm': current_arm,
                    'shoulder': current_shoulder,
                    'side': side
                }
            
            # Overlay via PIL zeichnen
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)
            
            # Dickere, leuchtende Skelett-Linien
            draw.line([tuple(hip), tuple(knee)], fill="#22c55e", width=8) 
            draw.line([tuple(knee), tuple(ankle)], fill="#22c55e", width=8) 
            draw.line([tuple(shoulder), tuple(hip)], fill="#06b6d4", width=6) 
            draw.line([tuple(shoulder), tuple(elbow)], fill="#eab308", width=6) 
            draw.line([tuple(elbow), tuple(wrist)], fill="#eab308", width=6) 
            
            # Gelenke markieren
            for pt in [hip, knee, ankle, shoulder, elbow, wrist]:
                draw.ellipse([pt[0]-10, pt[1]-10, pt[0]+10, pt[1]+10], fill="#ef4444")
                
            video_placeholder.image(pil_img, caption="Echtzeit KI-Gelenktracking (MoveNet Pro)", use_container_width=True)
            
        cap.release()
        os.unlink(tfile.name)
        
        status_text.empty()
        loader_anim.empty()
        st.success("🏁 Video-Analyse abgeschlossen!")
        
        # --- ERGONOMIE METRICS (HTML-basiert gegen Schwarz-auf-Schwarz Fehler) ---
        st.header(f"📊 Auswertung am tiefsten Pedalpunkt ({best_metrics['side']})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class='custom-card'>
                    <div class='custom-label'> Kniewinkel</div>
                    <div class='custom-value'>{best_metrics['knee']:.1f}°</div>
                    <div class='custom-target'>Optimal: 140° - 145°</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class='custom-card'>
                    <div class='custom-label'> Hüftwinkel</div>
                    <div class='custom-value'>{best_metrics['hip']:.1f}°</div>
                    <div class='custom-target'>Optimal: 40° - 50°</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div class='custom-card'>
                    <div class='custom-label'> Ellbogenbeugung</div>
                    <div class='custom-value'>{best_metrics['arm']:.1f}°</div>
                    <div class='custom-target'>Optimal: 15° - 25°</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
                <div class='custom-card'>
                    <div class='custom-label'> Schulterwinkel</div>
                    <div class='custom-value'>{best_metrics['shoulder']:.1f}°</div>
                    <div class='custom-target'>Optimal: 80° - 90°</div>
                </div>
            """, unsafe_allow_html=True)
        
        # --- KLAR LESBARE SETUP-EMPFEHLUNGEN ---
        st.header("🛠️ Professionelle Handlungsempfehlungen")
        
        # Sattel-Check
        if best_metrics['knee'] > 146.0:
            st.markdown(f"""
                <div class='rec-box' style='border-left-color: #ef4444;'>
                    <h3 style='margin:0; color:#ef4444 !important;'>❌ Sattelhöhe: Zu Hoch</h3>
                    <p style='margin:10px 0 0 0;'><strong>Empfehlung:</strong> Senke deinen Sattel um 3-5 mm. Ein zu hoher Sattel führt zu unruhigem Beckenkippen und überlastet die Sehnen deiner Kniekehle.</p>
                </div>
            """, unsafe_allow_html=True)
        elif best_metrics['knee'] < 139.0:
            st.markdown(f"""
                <div class='rec-box' style='border-left-color: #f59e0b;'>
                    <h3 style='margin:0; color:#f59e0b !important;'>⚠️ Sattelhöhe: Zu Niedrig</h3>
                    <p style='margin:10px 0 0 0;'><strong>Empfehlung:</strong> Schiebe den Sattel um 5-8 mm nach oben, um den Druck von der Kniescheibe zu nehmen und die Kraftübertragung zu verbessern.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='rec-box' style='border-left-color: #22c55e;'>
                    <h3 style='margin:0; color:#22c55e !important;'>✅ Sattelhöhe: Perfekt</h3>
                    <p style='margin:10px 0 0 0;'>Dein Kniewinkel liegt im ergonomischen Optimum! Die Kraftübertragung ist maximal effizient.</p>
                </div>
            """, unsafe_allow_html=True)
            
        # Cockpit-Check
        if best_metrics['arm'] < 12.0:
            st.markdown(f"""
                <div class='rec-box' style='border-left-color: #ef4444;'>
                    <h3 style='margin:0; color:#ef4444 !important;'>❌ Cockpit-Reach: Zu Gestreckt</h3>
                    <p style='margin:10px 0 0 0;'><strong>Empfehlung:</strong> Deine Arme sind zu stark durchgestreckt. Ein kürzerer Vorbau oder ein Lenker mit weniger Reach entlastet Hände und Nacken massiv.</p>
                </div>
            """, unsafe_allow_html=True)
        elif best_metrics['arm'] > 28.0:
            st.markdown(f"""
                <div class='rec-box' style='border-left-color: #f59e0b;'>
                    <h3 style='margin:0; color:#f59e0b !important;'>⚠️ Cockpit-Reach: Zu Kompakt</h3>
                    <p style='margin:10px 0 0 0;'><strong>Empfehlung:</strong> Du sitzt sehr gedrungen. Überprüfe, ob dir ein etwas längerer Vorbau eine sportlichere Position und freiere Atmung ermöglicht.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='rec-box' style='border-left-color: #22c55e;'>
                    <h3 style='margin:0; color:#22c55e !important;'>✅ Armhaltung: Optimal</h3>
                    <p style='margin:10px 0 0 0;'>Deine Ellbogen sind leicht angewinkelt, fangen Stöße perfekt ab und entspannen die Schultermuskulatur.</p>
                </div>
            """, unsafe_allow_html=True)
