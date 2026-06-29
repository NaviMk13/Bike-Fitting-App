import streamlit as st
import numpy as np
import cv2
import av
import tempfile
import os
from ultralytics import YOLO

# --- 1. PREMIUM DARK UI DESIGN & ANIMATIONS ---
st.set_page_config(page_title="VELO-MATCH Pro Biomechanics", layout="wide", page_icon="🚴")

st.markdown("""
    <style>
    /* Dark-Mode Radsport Atmosphäre */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.93), rgba(15, 23, 42, 0.97)), 
                    url('https://images.unsplash.com/photo-1485965120184-e220f721d03e?q=80&w=1920') no-repeat center center fixed;
        background-size: cover;
        color: #ffffff !important;
    }
    
    h1 {
        font-family: 'Impact', 'Arial Black', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #facc15 !important;
        text-shadow: 3px 3px 6px #000000;
    }
    
    h2, h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    
    /* Gamifizierter Score-Kasten */
    .score-container {
        text-align: center;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 4px solid #facc15;
        border-radius: 20px;
        padding: 35px;
        margin-bottom: 30px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    .score-headline {
        font-size: 18px !important;
        font-weight: bold !important;
        color: #94a3b8 !important;
        letter-spacing: 2px;
    }
    
    .score-value {
        font-size: 80px !important;
        font-weight: 900 !important;
        color: #facc15 !important;
        text-shadow: 0 0 20px rgba(250, 204, 21, 0.4);
    }
    
    /* Pro-Rider Badge Kasten */
    .rider-badge {
        background-color: #1e1b4b;
        border: 2px solid #6366f1;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-top: 15px;
    }
    
    /* Ergonomie Metrik Karten */
    .custom-card {
        background-color: #1e293b !important;
        border: 2px solid #334155 !important;
        border-top: 5px solid #38bdf8 !important;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .custom-card.optimal { border-top-color: #22c55e !important; }
    .custom-card.warning { border-top-color: #f59e0b !important; }
    .custom-card.danger { border-top-color: #ef4444 !important; }

    .custom-label {
        font-size: 13px !important;
        font-weight: bold !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
    }
    
    .custom-value {
        font-size: 36px !important;
        font-weight: 900 !important;
        color: #ffffff !important;
    }
    
    .custom-status {
        font-size: 14px !important;
        font-weight: bold !important;
        margin-top: 5px;
    }
    
    /* Handlungsanweisungen */
    .rec-box {
        background-color: #0f172a;
        border-left: 6px solid #facc15;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    /* Bike Loader Animation */
    @keyframes ride {
        0% { transform: translateX(-20px); }
        50% { transform: translateX(20px); }
        100% { transform: translateX(-20px); }
    }
    .bike-loader {
        font-size: 55px;
        animation: ride 1.2s infinite ease-in-out;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚴 VELO-MATCH: DYNAMIC PRO BIKE FITTING")
st.write("Analysiert Trittzyklen im Pedal-Tiefpunkt, filtert Messfehler und berechnet deinen persönlichen Ergonomie-Score.")

# --- 2. KI-MODELL INITIALISIERUNG (YOLOv8-MEDIUM POSE) ---
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8m-pose.pt')

try:
    pose_model = load_yolo_model()
    model_loaded = True
except Exception as e:
    st.error(f"Fehler beim Laden der KI: {e}")
    model_loaded = False

# --- 3. MATHEMATISCH KORREKTE WINKELBERECHNUNG ---
def calculate_angle(a, b, c, interior=True):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    if not interior:
        return angle
    return 180.0 - angle if angle > 90 else angle

# --- 4. VIDEO VERARBEITUNG & SEITEN-/TIEFSTPUNKT-PIPELINE ---
if model_loaded:
    uploaded_file = st.file_uploader("📂 Lade dein seitliches Bike-Fitting Video hoch (.mp4, .mov)", type=["mp4", "mov"])

    if uploaded_file is not None:
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in.write(uploaded_file.read())
        tfile_in.close()
        
        tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_out.close()
        
        status_text = st.empty()
        loader_anim = st.empty()
        status_text.info("🔄 Biomechanische Pipeline läuft... Trittzyklen werden isoliert.")
        loader_anim.markdown("<div class='bike-loader'>🚴‍♂️💨</div>", unsafe_allow_html=True)
        
        cap = cv2.VideoCapture(tfile_in.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_container = av.open(tfile_out.name, mode='w')
        stream = output_container.add_stream('h264', rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        
        # Datenspeicher für Trittphasen
        raw_knee_angles = []
        raw_hip_angles = []
        raw_arm_angles = []
        raw_shoulder_angles = []
        detected_side = "Unbekannt"
        ankle_y_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = pose_model(frame, conf=0.4, verbose=False)
            
            if len(results) > 0 and results[0].keypoints is not None:
                kp = results[0].keypoints.data[0].cpu().numpy()
                
                if len(kp) >= 17:
                    # Dynamische Seitenbestimmung (Rechts vs Links) basierend auf KI-Sicherheit
                    if kp[14][2] > kp[13][2]:
                        detected_side = "Rechte Seite"
                        hip, knee, ankle = kp[12][:2], kp[14][:2], kp[16][:2]
                        shoulder, elbow, wrist = kp[6][:2], kp[8][:2], kp[10][:2]
                        core_conf = [kp[12][2], kp[14][2], kp[16][2]]
                    else:
                        detected_side = "Linke Seite"
                        hip, knee, ankle = kp[11][:2], kp[13][:2], kp[15][:2]
                        shoulder, elbow, wrist = kp[5][:2], kp[7][:2], kp[9][:2]
                        core_conf = [kp[11][2], kp[13][2], kp[15][2]]
                    
                    if all(c > 0.5 for c in core_conf):
                        ankle_y_history.append(ankle[1])
                        
                        k_ang = calculate_angle(hip, knee, ankle, interior=False)
                        h_ang = calculate_angle(shoulder, hip, knee, interior=False)
                        a_ang = calculate_angle(shoulder, elbow, wrist, interior=True)
                        s_ang = calculate_angle(hip, shoulder, elbow, interior=False)
                        
                        # MESSUNG NUR IM PEDAL-TIEFSTPUNKT (6-Uhr-Stellung)
                        if len(ankle_y_history) > 5:
                            # Lokales Maximum auf der Y-Achse bedeutet tiefster Punkt im Videobild
                            if ankle_y_history[-3] == max(ankle_y_history[-5:]):
                                if 110 < k_ang < 170: # Grober Plausibilitätsfilter gegen Tracking-Sprünge
                                    raw_knee_angles.append(k_ang)
                                    raw_hip_angles.append(h_ang)
                                    raw_arm_angles.append(a_ang)
                                    raw_shoulder_angles.append(s_ang)
                        
                        # Neon-Overlay zeichnen
                        cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (34, 197, 94), 5)
                        cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (34, 197, 94), 5)
                        cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(hip[0]), int(hip[1])), (212, 182, 6), 4)
                        cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (8, 179, 234), 4)
                        cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (8, 179, 234), 4)
                        
                        for pt in [hip, knee, ankle, shoulder, elbow, wrist]:
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 7, (239, 68, 68), -1)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            for packet in stream.encode(av_frame):
                output_container.mux(packet)
                
        cap.release()
        for packet in stream.encode():
            output_container.mux(packet)
        output_container.close()
        
        status_text.empty()
        loader_anim.empty()
        
        # --- 5. GAMIFICATION & SCORE EXTRAKTION ---
        if len(raw_knee_angles) > 0:
            final_knee = np.median(raw_knee_angles)
            final_hip = np.median(raw_hip_angles)
            final_arm = np.median(raw_arm_angles)
            final_shoulder = np.median(raw_shoulder_angles)
            valid_cycles = len(raw_knee_angles)
            
            # Score Algorithmus: Abweichung vom Optimum bestraft Punktzahl
            knee_dev = abs(final_knee - 142.5)      # Optimum Knie: 142.5
            hip_dev = abs(final_hip - 45.0)         # Optimum Hüfte: 45.0
            arm_dev = abs(final_arm - 20.0)         # Optimum Arme: 20.0
            
            fit_score = int(100 - (knee_dev * 4.5 + hip_dev * 1.5 + arm_dev * 1.0))
            fit_score = max(15, min(100, fit_score))
        else:
            final_knee, final_hip, final_arm, final_shoulder = 135.0, 45.0, 20.0, 85.0
            valid_cycles = 0
            fit_score = 50

        # --- 6. DISPLAY GAMIFIED RATINGS ---
        col_score, col_badge = st.columns([2, 1])
        
        with col_score:
            st.markdown(f"""
                <div class='score-container'>
                    <div class='score-headline'>🏆 DEIN BIKE-FIT SCORE</div>
                    <div class='score-value'>{fit_score} / 100</div>
                    <div style='font-size:14px; color:#a1a1aa;'>Basiert auf {valid_cycles} dynamischen Kurbelumdrehungen</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_badge:
            # Pro-Rider Klassifizierung basierend auf der Hüftstreckung/Aggressivität
            if final_hip < 43.0:
                rider_name = "Max Verstappen (Aero Pro)"
                rider_desc = "Du sitzt extrem tief, flach und aerodynamisch auf dem Rad. Perfekt für maximale Geschwindigkeit im Windkanal-Style!"
            elif final_hip > 48.0:
                rider_name = "Gravel Grand Tourer"
                rider_desc = "Deine Position ist sehr aufrecht und komfortabel. Bereit für 200km Bikepacking-Touren ohne Rückenschmerzen."
            else:
                rider_name = "Tadej Pogačar (Allrounder)"
                rider_desc = "Die perfekte goldene Mitte aus aggressiver Aerodynamik und Langstrecken-Effizienz!"
                
            st.markdown(f"""
                <div class='rider-badge'>
                    <h3 style='margin:0; font-size:14px; color:#818cf8;'>👤 FAHRSTIL-PROFIL:</h3>
                    <h2 style='margin:5px 0; font-size:20px; color:#ffffff;'>{rider_name}</h2>
                    <p style='margin:0; font-size:12px; color:#94a3b8;'>{rider_desc}</p>
                </div>
            """, unsafe_allow_html=True)

        if fit_score >= 90:
            st.balloons() # Live Konfetti-Regen bei Top Setup!
            st.success("🔥 Sensationell! Dein Setup liefert die ultimative Symbiose aus Biomechanik und Aerodynamik!")

        # --- 7. INTERAKTIVER YOUTUBE-STYLE VIDEO PLAYER ---
        st.header("📹 Interaktiver Video-Player (Zyklen-Schnitt)")
        with open(tfile_out.name, 'rb') as video_file:
            st.video(video_file.read())
        
        os.unlink(tfile_in.name)
        os.unlink(tfile_out.name)
        
        # --- 8. DASHBOARD MIT FARBCODIERTEN PRO-STATUS-KARTEN ---
        st.header(f"📊 Evaluierte Gelenk-Mittelwerte ({detected_side})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        def card_meta(val, low, high):
            if val < low: return "danger", "🔴 ZU NIEDRIG"
            if val > high: return "danger", "🔴 ZU HOCH"
            return "optimal", "🟢 OPTIMAL"

        k_class, k_status = card_meta(final_knee, 140.0, 145.0)
        h_class, h_status = card_meta(final_hip, 40.0, 50.0)
        a_class, a_status = card_meta(final_arm, 15.0, 25.0)
        s_class, s_status = card_meta(final_shoulder, 80.0, 90.0)

        with col1:
            st.markdown(f"<div class='custom-card {k_class}'><div class='custom-label'>Kniewinkel</div><div class='custom-value'>{final_knee:.1f}°</div><div class='custom-status'>{k_status}</div><div style='font-size:11px; color:#64748b; margin-top:5px;'>Ziel: 140.0° - 145.0°</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='custom-card {h_class}'><div class='custom-label'>Hüftwinkel</div><div class='custom-value'>{final_hip:.1f}°</div><div class='custom-status'>{h_status}</div><div style='font-size:11px; color:#64748b; margin-top:5px;'>Ziel: 40.0° - 50.0°</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='custom-card {a_class}'><div class='custom-label'>Ellbogenbeugung</div><div class='custom-value'>{final_arm:.1f}°</div><div class='custom-status'>{a_status}</div><div style='font-size:11px; color:#64748b; margin-top:5px;'>Ziel: 15.0° - 25.0°</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='custom-card {s_class}'><div class='custom-label'>Schulterwinkel</div><div class='custom-value'>{final_shoulder:.1f}°</div><div class='custom-status'>{s_status}</div><div style='font-size:11px; color:#64748b; margin-top:5px;'>Ziel: 80.0° - 90.0°</div></div>", unsafe_allow_html=True)
        
        # --- 9. PROFESSIONELLE HANDLUNGSEMPFEHLUNGEN ---
        st.header("🛠️ Werkstatt-Anweisungen")
        
        if final_knee > 145.0:
            st.markdown("<div class='rec-box' style='border-left-color: #ef4444;'><h3>❌ Sattel runter!</h3><p>Dein Knie wird in der Streckphase überstreckt. Senke den Sattel schrittweise um <b>3-5 mm</b> ab. Das stabilisiert dein Becken und schützt deine Kniekehle.</p></div>", unsafe_allow_html=True)
        elif final_knee < 140.0:
            st.markdown("<div class='rec-box' style='border-left-color: #ef4444;'><h3>⚠️ Sattel höher!</h3><p>Dein Sattel ist zu niedrig eingestellt. Schiebe die Sattelstütze um <b>5-8 mm</b> nach oben. Das nimmt massiven Druck von der Kniescheibe und erhöht die Tritt-Effizienz.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='rec-box' style='border-left-color: #22c55e;'><h3>✅ Sattelhöhe perfekt</h3><p>Dein Kniewinkel im Tiefpunkt ist absolut makellos. Keine Justierung am Sattel notwendig!</p></div>", unsafe_allow_html=True)
            
        if final_arm < 15.0:
            st.markdown("<div class='rec-box' style='border-left-color: #ef4444;'><h3>❌ Cockpit-Reach verkürzen</h3><p>Deine Arme sind komplett durchgestreckt, wodurch Stöße direkt in den Nacken schlagen. Ein <b>10-20 mm kürzerer Vorbau</b> bewirkt hier Wunder.</p></div>", unsafe_allow_html=True)
        elif final_arm > 25.0:
            st.markdown("<div class='rec-box' style='border-left-color: #f59e0b;'><h3>⚠️ Cockpit-Reach verlängern</h3><p>Deine Sitzposition ist sehr gedrungen. Überlege, ob ein leicht längerer Vorbau dir mehr Raum für eine freie Atmung und flachere Aerodynamik bietet.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='rec-box' style='border-left-color: #22c55e;'><h3>✅ Armhaltung im Optimum</h3><p>Deine Ellbogen weisen genau die richtige Beugung auf, um flexibel auf Unebenheiten zu reagieren.</p></div>", unsafe_allow_html=True)
