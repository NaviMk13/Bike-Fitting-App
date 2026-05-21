# --- 0. ULTIMATIV-HOTFIX FÜR STREAMLIT CLOUD (MUSS GANZ OBEN STEHEN) ---
import sys
import subprocess

# 1. Erzwinge das Nachinstallieren fehlender System-Bibliotheken über ein Python-Package
try:
    import cv2
except ImportError as e:
    # Falls libgthread-2.0.so.0 oder libGL.so.1 fehlt, reparieren wir das direkt hier
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-contrib-python-headless", "--force-reinstall"])
    
    # Falls das System immer noch meckert, nutzen wir ein Tool, das System-Bibliotheken im Python-Verzeichnis bereitstellt
    try:
        import cv2
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "apt-clone"]) # Fallback
        # installiere ein Paket, das die libs mitliefert
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless"])

# --- AB HIER FOLGT DER NORMALE CODE ---
import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import os

# --- 1. SEITEN-SETUP ---
st.set_page_config(page_title="DIY KI Bike Fitter", layout="wide", page_icon="🚴")
st.title("🚴 DIY AI Bike Fitting Tool")
st.write("Lade ein seitliches Video von dir auf dem Fahrrad hoch, um deine Haltung analysieren zu lassen.")

# --- 2. MEDIAPIPE INITIALISIERUNG ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 3. HELFER-FUNKTION: WINKELBERCHNUNG ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# --- 4. VIDEO UPLOAD & VERARBEITUNG ---
uploaded_file = st.file_uploader("Wähle ein Video aus (mp4, mov)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 30
    
    st.info("Video wird analysiert... Bitte kurz warten.")
    
    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_knee_angles = []
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(out_tfile.name, fourcc, fps, (width, height))
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.get()
            if not ret:
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = pose.process(image)
            
            image.flags.writeable = True
            annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Rechte Körperseite messen
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                knee_angle = calculate_angle(hip, knee, ankle)
                all_knee_angles.append(knee_angle)
                
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                cv2.putText(annotated_image, f"Knie: {int(knee_angle)} Grad", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            out.write(annotated_image)
            
            current_frame += 1
            if frame_count > 0:
                progress_bar.progress(min(current_frame / frame_count, 1.0))
                
        cap.release()
        out.release()
    
    st.success("Analyse abgeschlossen!")
    
    if all_knee_angles:
        max_knee_angle = max(all_knee_angles)
        
        st.header("🚴 Haltungsanalyse & KI-Tipps")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Gemessener maximaler Kniewinkel", value=f"{max_knee_angle:.1f}°", delta="Optimal: 140°-150°")
            
        with col2:
            if max_knee_angle < 140:
                st.warning("⚠️ Dein Sattel ist wahrscheinlich zu niedrig!")
                st.write("**Tipp:** Schiebe deinen Sattel in kleinen Schritten (ca. 5mm) nach oben.")
            elif max_knee_angle > 150:
                st.warning("⚠️ Dein Sattel ist wahrscheinlich zu hoch!")
                st.write("**Tipp:** Stelle den Sattel etwas tiefer.")
            else:
                st.success("🎉 Perfekte Sattelhöhe!")
                st.write("**Tipp:** Dein Kniewinkel liegt genau im ergonomischen Bereich.")
                
    os.unlink(tfile.name)
    os.unlink(out_tfile.name)
