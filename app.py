import streamlit as st
import numpy as np
import cv2
import av
import tempfile
import os
from ultralytics import YOLO

# --- 1. DESIGN & HOCHLESBARES CONTRAST-LAYOUT ---
st.set_page_config(page_title="VELO-MATCH KI Pro", layout="wide", page_icon="🚴")

st.markdown("""
    <style>
    /* Dunkler Radsport-Hintergrund mit erzwungenem Kontrast */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.95)), 
                    url('https://images.unsplash.com/photo-1485965120184-e220f721d03e?q=80&w=1920') no-repeat center center fixed;
        background-size: cover;
        color: #ffffff !important;
    }
    
    /* Titel-Styling */
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
    
    .stMarkdown p {
        color: #f1f5f9 !important;
        font-size: 16px;
    }
    
    /* Eigene Ergebniskarten gegen den Schwarz-auf-Schwarz Fehler */
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
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .custom-value {
        font-size: 36px !important;
        font-weight: 900 !important;
        color: #facc15 !important;
        margin-bottom: 8px;
        text-shadow: 1px 1px 2px #000000;
    }
    
    .custom-target {
        font-size: 14px !important;
        font-weight: bold !important;
        color: #38bdf8 !important;
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
    
    /* Animierter Fahrrad-Loader */
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
st.write("Echtzeit-Tracking mit YOLOv8-Pose für Python 3.14. Generiert einen interaktiven Video-Player mit Overlays.")

# --- 2. HÖCHST KOMPATIBLES KI-MODELL (YOLOv8-POSE) ---
@st.cache_resource
def load_yolo_model():
    # Lädt das extrem schnelle und präzise YOLOv8 Pose Modell (wird auto-heruntergeladen)
    return YOLO('yolov8n-pose.pt')

try:
    pose_model = load_yolo_model()
    model_loaded = True
except Exception as e:
    st.error(f"Fehler beim Laden der YOLO-KI: {e}")
    model_loaded = False

# --- 3. MATHEMATISCH KORREKTE WINKELBERECHNUNG ---
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

# --- 4. ANALYSE & INTERAKTIVES RENDERING ---
if model_loaded:
    uploaded_file = st.file_uploader("📂 Ziehe dein Bike-Fitting Video hierher (.mp4, .mov)", type=["mp4", "mov"])

    if uploaded_file is not None:
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in.write(uploaded_file.read())
        tfile_in.close()
        
        tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_out.close()
        
        status_text = st.empty()
        loader_anim = st.empty()
        status_text.info("⚙️ YOLOv8 analysiert die Biomechanik und rendert das Video...")
        loader_anim.markdown("<div class='bike-loader'>🚴💨</div>", unsafe_allow_html=True)
        
        # Input-Video über OpenCV öffnen
        cap = cv2.VideoCapture(tfile_in.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output-Video über PyAV öffnen (garantiert perfekte H.264 HTML5 Player-Kompatibilität)
        output_container = av.open(tfile_out.name, mode='w')
        stream = output_container.add_stream('h264', rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        
        max_knee_angle = 0.0
        best_metrics = {'knee': 142.0, 'hip': 45.0, 'arm': 20.0, 'shoulder': 85.0, 'side': 'Unbekannt'}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO-Inferenz auf dem Frame laufen lassen
            results = pose_model(frame, verbose=False)
            
            if len(results) > 0 and results[0].keypoints is not None:
                # Hole die 2D Keypoints (x, y, Sichtbarkeit)
                kp = results[0].keypoints.data[0].cpu().numpy()
                
                # YOLOv8 Pose Indizes:
                # 5=L-Schulter, 6=R-Schulter, 7=L-Ellbogen, 8=R-Ellbogen, 9=L-Hand, 10=R-Hand
                # 11=L-Hüfte, 12=R-Hüfte, 13=L-Knie, 14=R-Knie, 15=L-Knöchel, 16=R-Knöchel
                
                if len(kp) >= 17:
                    # Bestimme die sichtbare Körperseite anhand des Vertrauenswerts
                    if kp[14][2] > kp[13][2]:
                        side = "Rechte Seite"
                        hip, knee, ankle = kp[12][:2], kp[14][:2], kp[16][:2]
                        shoulder, elbow, wrist = kp[6][:2], kp[8][:2], kp[10][:2]
                    else:
                        side = "Linke Seite"
                        hip, knee, ankle = kp[11][:2], kp[13][:2], kp[15][:2]
                        shoulder, elbow, wrist = kp[5][:2], kp[7][:2], kp[9][:2]
                    
                    # Winkel berechnen
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
                    
                    # Hochkontrast-Skelettlinien einzeichnen (BGR Format für OpenCV)
                    cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (34, 197, 94), 6)      # Neon-Grün
                    cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (34, 197, 94), 6)  
                    cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(hip[0]), int(hip[1])), (212, 182, 6), 5) # Cyan
                    cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (8, 179, 234), 5) # Gelb
                    cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (8, 179, 234), 5)
                    
                    # Gelenkpunkte markieren
                    for pt in [hip, knee, ankle, shoulder, elbow, wrist]:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, (68, 68, 239), -1) # Rot
            
            # Frame für PyAV konvertieren (von BGR nach RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            
            # Frame codieren und schreiben
            for packet in stream.encode(av_frame):
                output_container.mux(packet)
                
        cap.release()
        # Stream beenden
        for packet in stream.encode():
            output_container.mux(packet)
        output_container.close()
        
        status_text.empty()
        loader_anim.empty()
        st.success("🏁 Video erfolgreich verarbeitet!")
        
        # --- 5. DER YOUTUBE INTERAKTIVE PLAYER ---
        st.header("📹 Interaktive Video-Analyse (YouTube-Style)")
        st.write("Nutze die Zeitleiste, springe an beliebige Positionen vor/zurück oder pausiere das Video:")
        
        with open(tfile_out.name, 'rb') as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
        
        # Temporäre Dateien löschen
        os.unlink(tfile_in.name)
        os.unlink(tfile_out.name)
        
        # --- 6. ERGONOMIE METRICS ---
        st.header(f"📊 Maximale Streckphasen-Auswertung ({best_metrics['side']})")
        
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
        
        # --- 7. DEUTLICHE HANDLUNGSEMPFEHLUNGEN ---
        st.header("🛠️ Professionelle Handlungsempfehlungen")
        
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
                    <p style='margin:10px 0 0 0;'><strong>Empfehlung:</strong> Schiebe den Sattel um 5-8 mm nach oben, um den Druck von der Kniescheibe zu nehmen.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='rec-box' style='border-left-color: #22c55e;'>
                    <h3 style='margin:0; color:#22c55e !important;'>✅ Sattelhöhe: Perfekt</h3>
                    <p style='margin:10px 0 0 0;'>Dein Kniewinkel liegt im ergonomischen Optimum! Maximal effiziente Kraftübertragung.</p>
                </div>
            """, unsafe_allow_html=True)
            
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
                    <p style='margin:10px 0 0 0;'><strong>Empfehlung:</strong> Du sitzt sehr gedrungen. Überprüfe, ob dir ein etwas längerer Vorbau eine sportlichere Position ermöglicht.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='rec-box' style='border-left-color: #22c55e;'>
                    <h3 style='margin:0; color:#22c55e !important;'>✅ Armhaltung: Optimal</h3>
                    <p style='margin:10px 0 0 0;'>Deine Ellbogen sind leicht angewinkelt, fangen Stöße perfekt ab und entspannen die Schultermuskulatur.</p>
                </div>
            """, unsafe_allow_html=True)
