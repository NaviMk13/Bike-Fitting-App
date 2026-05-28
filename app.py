import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import tensorflow_hub as tfhub
import cv2
import tempfile
import os

# --- 1. DESIGN & HOCHLESBARE RADSPORT-UI ---
st.set_page_config(page_title="VELO-MATCH KI Pro", layout="wide", page_icon="🚴")

st.markdown("""
    <style>
    /* Dunkler, atmosphärischer Hintergrund */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.95)), 
                    url('https://images.unsplash.com/photo-1485965120184-e220f721d03e?q=80&w=1920') no-repeat center center fixed;
        background-size: cover;
        color: #f8fafc;
    }
    
    /* Extrem gut lesbare Überschriften */
    h1 {
        font-family: 'Impact', 'Arial Black', sans-serif;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #facc15 !important; /* Neon-Gelb */
        text-shadow: 3px 3px 6px rgba(0,0,0,0.9);
    }
    h2, h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }
    
    /* Kontraststarke Infoboxen für perfekte Lesbarkeit */
    .metric-card {
        background-color: #1e293b;
        border: 3px solid #facc15;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }
    
    /* Große, fette Schrift für Werte */
    .metric-value {
        font-size: 32px;
        font-weight: 900;
        color: #facc15;
    }
    .metric-label {
        font-size: 16px;
        font-weight: bold;
        color: #94a3b8;
        text-transform: uppercase;
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

# --- 2. HOCHPRÄZISES MOVENET KI-MODELL (THUNDER) ---
@st.cache_resource
def load_movenet_model():
    # Lädt das präzise MoveNet Thunder Modell für hochauflösende Videoanalysen
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
        
        # UI Platzhalter für den Live-Videoplayer
        video_placeholder = st.empty()
        
        # Status-Meldungen
        status_text = st.empty()
        loader_anim = st.empty()
        status_text.info("⚙️ Starte Video-Stream und KI-Inferenz...")
        loader_anim.markdown("<div class='bike-loader'>🚴💨</div>", unsafe_allow_html=True)
        
        # Video über OpenCV einlesen
        cap = cv2.VideoCapture(tfile.name)
        
        max_knee_angle = 0.0
        best_metrics = {'knee': 142.0, 'hip': 45.0, 'arm': 20.0, 'shoulder': 85.0, 'side': 'Unbekannt'}
        
        # Frame-Schleife für Live-Playback
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # OpenCV nutzt BGR, wir brauchen RGB für PIL/Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_img, w_img, _ = frame_rgb.shape
            
            # Bild für MoveNet vorbereiten (Eingangsgröße zwingend 256x256 für Thunder)
            input_image = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 256, 256)
            input_image = tf.cast(input_image, dtype=tf.int32)
            
            # Inferenz ausführen
            outputs = movenet_model(input_image)
            keypoints = outputs['output_0'].numpy()[0, 0, :, :] # 17 Keypoints
            
            # MoveNet Keypoint-Indizes:
            # 5=L-Schulter, 6=R-Schulter, 7=L-Ellbogen, 8=R-Ellbogen, 9=L-Hand, 10=R-Hand
            # 11=L-Hüfte, 12=R-Hüfte, 13=L-Knie, 14=R-Knie, 15=L-Knöchel, 16=R-Knöchel
            
            # Validierung über Confidence Scores (Index 2 in den Keypoint-Daten)
            if keypoints[14][2] > 0.3 and keypoints[12][2] > 0.3: # Rechte Seite prüfen
                side = "Rechte Seite"
                hip = [keypoints[12][1] * h_img, keypoints[12][0] * w_img]
                knee = [keypoints[14][1] * h_img, keypoints[14][0] * w_img]
                ankle = [keypoints[16][1] * h_img, keypoints[16][0] * w_img]
                shoulder = [keypoints[6][1] * h_img, keypoints[6][0] * w_img]
                elbow = [keypoints[8][1] * h_img, keypoints[8][0] * w_img]
                wrist = [keypoints[10][1] * h_img, keypoints[10][0] * w_img]
            else: # Ausweichoption Linke Seite
                side = "Linke Seite"
                hip = [keypoints[11][1] * h_img, keypoints[11][0] * w_img]
                knee = [keypoints[13][1] * h_img, keypoints[13][0] * w_img]
                ankle = [keypoints[15][1] * h_img, keypoints[15][0] * w_img]
                shoulder = [keypoints[5][1] * h_img, keypoints[5][0] * w_img]
                elbow = [keypoints[7][1] * h_img, keypoints[7][0] * w_img]
                wrist = [keypoints[9][1] * h_img, keypoints[9][0] * w_img]
            
            # Berechne die aktuellen Geometrien
            current_knee = calculate_angle(hip, knee, ankle, interior=False)
            current_hip = calculate_angle(shoulder, hip, knee, interior=False)
            current_arm = calculate_angle(shoulder, elbow, wrist, interior=True)
            current_shoulder = calculate_angle(hip, shoulder, elbow, interior=False)
            
            # Wir suchen dynamisch nach dem Punkt der maximalen Beinstreckung (Pedal bei 6 Uhr)
            if current_knee > max_knee_angle and current_knee < 165.0:
                max_knee_angle = current_knee
                best_metrics = {
                    'knee': current_knee,
                    'hip': current_hip,
                    'arm': current_arm,
                    'shoulder': current_shoulder,
                    'side': side
                }
            
            # Overlay auf Frame zeichnen via PIL
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)
            
            # Skelett-Linien zeichnen (Neon-Farben)
            draw.line([tuple(hip), tuple(knee)], fill="#22c55e", width=6) # Bein Oben
            draw.line([tuple(knee), tuple(ankle)], fill="#22c55e", width=6) # Bein Unten
            draw.line([tuple(shoulder), tuple(hip)], fill="#06b6d4", width=5) # Torso
            draw.line([tuple(shoulder), tuple(elbow)], fill="#eab308", width=5) # Oberarm
            draw.line([tuple(elbow), tuple(wrist)], fill="#eab308", width=5) # Unterarm
            
            # Gelenkpunkte markieren
            for pt in [hip, knee, ankle, shoulder, elbow, wrist]:
                draw.ellipse([pt[0]-8, pt[1]-8, pt[0]+8, pt[1]+8], fill="#ef4444")
            
            # Jetzt wird das verarbeitete Bild SOFORT live im Player angezeigt
            video_placeholder.image(pil_img, caption="Echtzeit KI-Gelenktracking (MoveNet Pro)", use_container_width=True)
            
        cap.release()
        os.unlink(tfile.name)
        
        # Loader entfernen, wenn fertig
        status_text.empty()
        loader_anim.empty()
        st.success("🏁 Video-Analyse abgeschlossen! Hier sind deine biomechanischen Ergebnisse:")
        
        # --- ERGONOMIE METRICS (Hocheffektiver Kontrast) ---
        st.header(f"📊 Auswertung der optimalen Streckphase ({best_metrics['side']})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>🦵 Kniewinkel</div>
                    <div class='metric-value'>{best_metrics['knee']:.1f}°</div>
                    <div style='color: #a1a1aa; font-weight: bold; margin-top: 5px;'>Optimal: 140° - 145°</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>🧘 Hüftwinkel</div>
                    <div class='metric-value'>{best_metrics['hip']:.1f}°</div>
                    <div style='color: #a1a1aa; font-weight: bold; margin-top: 5px;'>Optimal: 40° - 50°</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>💪 Ellbogenbeugung</div>
                    <div class='metric-value'>{best_metrics['arm']:.1f}°</div>
                    <div style='color: #a1a1aa; font-weight: bold; margin-top: 5px;'>Optimal: 15° - 25°</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>📐 Schulterwinkel</div>
                    <div class='metric-value'>{best_metrics['shoulder']:.1f}°</div>
                    <div style='color: #a1a1aa; font-weight: bold; margin-top: 5px;'>Optimal: 80° - 90°</div>
                </div>
            """, unsafe_allow_html=True)
        
        # --- KLAR LESBARE SETUP-EMPFEHLUNGEN ---
        st.header("🛠️ Professionelle Handlungsempfehlungen")
        
        # Sattel
        if best_metrics['knee'] > 146.0:
            st.error("❌ **Sattelhöhe:** Dein Sattel ist **zu hoch** eingestellt.")
            st.markdown("<p style='font-size: 16px; color: #f8fafc;'><strong>Empfehlung:</strong> Senke deinen Sattel um 3-5 mm. Ein zu hoher Sattel führt zu unruhigem Beckenkippen und überlastet die Sehnen deiner Kniekehle.</p>", unsafe_allow_html=True)
        elif best_metrics['knee'] < 139.0:
            st.warning("⚠️ **Sattelhöhe:** Dein Sattel ist **zu niedrig** eingestellt.")
            st.markdown("<p style='font-size: 16px; color: #f8fafc;'><strong>Empfehlung:</strong> Schiebe den Sattel um 5-8 mm nach oben, um den Druck von der Kniescheibe zu nehmen und die Muskelrekrutierung zu optimieren.</p>", unsafe_allow_html=True)
        else:
            st.success("✅ **Sattelhöhe:** Dein Kniewinkel liegt im absoluten **Ergonomie-Optimum**! Perfekte Kraftübertragung.")
            
        # Cockpit & Arme
        if best_metrics['arm'] < 12.0:
            st.error("❌ **Cockpit-Reach:** Deine Arme sind **zu stark durchgestreckt**.")
            st.markdown("<p style='font-size: 16px; color: #f8fafc;'><strong>Empfehlung:</strong> Du sitzt zu gestreckt. Montiere einen kürzeren Vorbau oder wähle einen Lenker mit weniger Reach. Das entlastet Nacken und Hände massiv.</p>", unsafe_allow_html=True)
        elif best_metrics['arm'] > 28.0:
            st.warning("⚠️ **Cockpit-Reach:** Deine Ellbogenbeugung ist **sehr stark**.")
            st.markdown("<p style='font-size: 16px; color: #f8fafc;'><strong>Empfehlung:</strong> Du sitzt sehr gedrungen. Überprüfe, ob ein etwas längerer Vorbau dir eine aerodynamischere und freiere Atmung ermöglicht.</p>", unsafe_allow_html=True)
        else:
            st.success("✅ **Armhaltung:** Hervorragend! Deine Arme sind leicht angewinkelt, fangen Stöße perfekt ab und entspannen die Schultern.")
